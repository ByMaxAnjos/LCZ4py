"""Plot a spatial grid/mesh (vector polygons or points) over a basemap,
optionally colored by an attribute.

New function for this repo (no direct R equivalent) — a lightweight
counterpart to lcz_plot_map.py for visualizing the raw grid/mesh geometry
itself (e.g. municipality boundaries, GHAP/ERA5 grid cells, LCZ zone
polygons) rather than a rendered raster.

Cartography
-----------
- Plots in Web Mercator (EPSG:3857) so contextily tiles align natively and
  north is always up.
- Scale bar length is computed from the **geodesic** width of the visible
  extent (pyproj ``Geod``), not from raw Mercator metres, so it stays correct
  away from the equator.
- Colorblind-safe colormaps by default (``viridis``); diverging variables can
  use ``RdBu``/``BrBG``.
- 300 dpi output with ``style="nature"``/``"science"``/``"generic_bw"`` or
  ``dpi=300``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import numpy as np
import pyproj

try:
    import contextily as cx

    HAS_CONTEXTILY = True
except ImportError:  # pragma: no cover - optional dependency
    cx = None  # type: ignore[assignment]
    HAS_CONTEXTILY = False

from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import (
    OUTPUT_DIR,
    _nice_scale_length,
    finalize_export,
    get_style,
)

logger = logging.getLogger(__name__)

#: Web Mercator — the contextily tile CRS and a north-up projected CRS.
_WEB_MERCATOR = "EPSG:3857"

#: Basemap attribution, keyed by provider name substring.
_ATTRIBUTIONS = (
    ("carto", "© OpenStreetMap contributors © CARTO"),
    ("stamen", "© Stamen Design © OpenStreetMap contributors"),
    ("esri", "Esri, Maxar, Earthstar Geographics"),
    ("osm", "© OpenStreetMap contributors"),
)

#: Extensions matplotlib can actually write (guards ``save_extension="html"``).
_MPL_EXTENSIONS = (
    "png", "pdf", "svg", "jpg", "jpeg", "tif", "tiff", "ps", "eps", "webp",
)

#: Above this many categories the categorical legend becomes unreadable.
_MAX_LEGEND_CATEGORIES = 30


def _attribution_for(source: object) -> str:
    """Best-effort basemap attribution for a contextily provider."""
    explicit = getattr(source, "attribution", None)
    if isinstance(explicit, str) and explicit.strip():
        # Provider attributions often carry HTML links; keep the text only.
        text = explicit
        for tag in ("<a", "</a>", "<span", "</span>", "<strong", "</strong>"):
            text = text.replace(tag, "")
        text = text.replace(">", "").replace("<", "").strip()
        if text:
            # xyzservices ships ASCII-safe "(C)"; use the real © glyph.
            return text.replace("(C)", "©")

    name = " ".join(
        str(getattr(source, attr, "") or "")
        for attr in ("name", "provider_name")
    ).lower()
    for key, value in _ATTRIBUTIONS:
        if key in name:
            return value
    return "© OpenStreetMap contributors"


def _is_point_layer(gdf) -> bool:
    try:
        return bool(gdf.geom_type.isin(("Point", "MultiPoint")).all())
    except Exception:  # pragma: no cover - exotic geometry columns
        return False


def _pick_zoom(ax, transformer: pyproj.Transformer) -> int:
    """Choose a Web-Mercator zoom level for the current axes extent.

    contextily's own auto-zoom returns -1 for small (city-scale) extents when
    the axes are large, so the level is derived from the requested ground
    resolution instead: one screen pixel ≈ one map pixel.
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width_m = float(xmax - xmin)
    if not np.isfinite(width_m) or width_m <= 0:
        return 12

    _, lat = transformer.transform((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
    lat = max(min(float(lat), 85.0), -85.0)
    figure_px = 1000.0
    try:
        figure_px = float(ax.figure.get_size_inches()[0] * ax.figure.dpi)
    except Exception:  # pragma: no cover - defensive
        pass

    # 156543.03392 m/px is the zoom-0 resolution at the equator.
    zoom = np.log2(156543.03392 * np.cos(np.deg2rad(lat)) * figure_px / width_m)
    return int(min(max(np.floor(zoom), 2), 18))


def _resolve_cmap(name: str, lang: str) -> str:
    """Validate a colormap name, falling back to a colorblind-safe one."""
    try:
        if name in matplotlib.colormaps:
            return name
    except Exception:  # pragma: no cover - very old matplotlib
        pass
    logger.warning(
        "Unknown colormap %r; falling back to 'viridis' (colorblind-safe).", name,
    )
    return "viridis"


def _geodesic_width_m(
    x0: float,
    x1: float,
    y: float,
    transformer: pyproj.Transformer,
    geod: pyproj.Geod,
) -> float:
    """Ground distance (m) between two Web-Mercator x positions at latitude ``y``."""
    lon0, lat0 = transformer.transform(x0, y)
    lon1, lat1 = transformer.transform(x1, y)
    distance = geod.inv(lon0, lat0, lon1, lat1)[2]
    return float(abs(distance))


def _draw_scalebar(ax, transformer, geod) -> None:
    """Two-tone scale bar whose length is geodesically accurate."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if not (np.isfinite([xmin, xmax, ymin, ymax]).all()) or xmax <= xmin:
        return

    # Evaluate the bar at the latitude where it is drawn (bottom-left).
    y_bar = ymin + 0.08 * (ymax - ymin)
    width_m = _geodesic_width_m(xmin, xmax, y_bar, transformer, geod)
    if not np.isfinite(width_m) or width_m <= 0:
        logger.debug("Skipping scale bar: could not compute a ground width.")
        return

    bar_m = _nice_scale_length(width_m)
    bar_frac = min(bar_m / width_m, 0.35)
    if bar_frac <= 0:
        return

    x0, y0, height = 0.03, 0.045, 0.009
    half = bar_frac / 2.0
    for offset, face in ((0.0, "black"), (half, "white")):
        ax.add_patch(plt.Rectangle(
            (x0 + offset, y0), half, height,
            transform=ax.transAxes, facecolor=face, edgecolor="black",
            linewidth=0.6, zorder=10, clip_on=False,
        ))

    label = f"{bar_m / 1000:g} km" if bar_m >= 1000 else f"{bar_m:g} m"
    text_kw = dict(
        transform=ax.transAxes, ha="center", va="bottom", fontsize=7,
        color="black", zorder=11,
        path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")],
    )
    ax.text(x0, y0 + height + 0.004, "0", **text_kw)
    ax.text(x0 + bar_frac, y0 + height + 0.004, label, **text_kw)


def _draw_north_arrow(ax) -> None:
    """North arrow (Web Mercator is north-up, so 'up' is true north)."""
    ax.annotate(
        "", xy=(0.95, 0.95), xytext=(0.95, 0.84),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(facecolor="black", edgecolor="black", width=2.5, headwidth=8, headlength=9),
        zorder=11, clip_on=False,
    )
    ax.text(
        0.95, 0.965, "N", transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9, fontweight="bold", color="black", zorder=12,
        path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")],
    )


def _legend_label(color_by: str, units: Optional[str], legend_label: Optional[str]) -> str:
    base = legend_label or color_by
    return f"{base} ({units})" if units else base


def plot_grid_only(
    grid: "gpd.GeoDataFrame",
    color_by: Optional[str] = None,
    cmap: str = "viridis",
    edgecolor: str = "black",
    linewidth: float = 0.3,
    alpha: float = 0.7,
    add_basemap: bool = True,
    basemap_source: Optional[object] = None,
    figsize: Optional[tuple[float, float]] = (10, 10),
    title: Optional[str] = None,
    isave: bool = False,
    save_extension: str = "png",
    style: str = "default",
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
    lang: str = "en",
    units: Optional[str] = None,
    legend_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_axes: bool = False,
    attribution: Optional[str] = None,
    dpi: Optional[int] = None,
    basemap_zoom: Optional[int] = None,
    save_filename: str = "lcz_plot_grid",
) -> "plt.Figure":
    """Plot a grid/mesh GeoDataFrame over a contextily basemap.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        Vector grid/mesh to plot (polygons or points) — municipality
        boundaries, GHAP/ERA5/CHIRPS grid cells, LCZ zone polygons, etc.
    color_by : str, optional
        Column name to color features by. None (default) plots a single
        uncolored fill. Numeric columns get a colorbar; text/categorical
        columns get a discrete legend.
    cmap : str
        Matplotlib colormap for ``color_by``. Default "viridis"
        (colorblind-safe). Use "RdBu"/"BrBG" for diverging variables
        (e.g. PDSI). Invalid names fall back to "viridis" with a warning.
    edgecolor : str
        Polygon outline color. Default "black".
    linewidth : float
        Polygon outline width. Default 0.3.
    alpha : float
        Fill opacity. Default 0.7.
    add_basemap : bool
        Overlay a CARTO/OpenStreetMap basemap via contextily. Default True.
        Skipped with a warning if contextily is not installed or the tiles
        cannot be fetched — the figure is still produced.
    basemap_source : contextily provider, optional
        Custom tile source (e.g. ``contextily.providers.CartoDB.Positron``).
        Defaults to CARTO Positron (light, keeps the data readable).
    figsize : tuple[float, float], optional
        Figure size in inches. Default (10, 10). Pass ``None`` to derive the
        width from the ``style`` preset.
    title : str, optional
        Plot title.
    isave : bool
        Save the figure to ``LCZ4r_output/``. Default False.
    save_extension : str
        File extension when ``isave``. Default "png".
    style : str
        Publication style preset: 'default', 'nature', 'science', or
        'generic_bw'. Controls font, figure size (mm), DPI, and palette
        used when isave and save_extension != 'html'.
    add_scalebar : bool
        Draw a cartographic scale bar (bottom-left), with a geodesically
        accurate length.
    add_north_arrow : bool
        Draw a north arrow (top-right).
    lang : str
        Message language. Default "en".
    units : str, optional
        Measurement unit appended to the legend label (e.g. "mm", "°C").
    legend_label : str, optional
        Legend/colorbar title. Defaults to ``color_by``.
    vmin, vmax : float, optional
        Fix the color scale (useful to compare several figures).
    show_axes : bool
        Show coordinate ticks/labels instead of a clean map frame.
    attribution : str, optional
        Override the basemap credit line. Defaults to the provider's own
        attribution (e.g. "© OpenStreetMap contributors © CARTO").
    dpi : int, optional
        Export resolution. Defaults to the ``style`` preset's DPI (150 for
        'default', 300 for the journal presets). Pass 300 for print.
    basemap_zoom : int, optional
        Force a contextily zoom level instead of letting it auto-select.
    save_filename : str
        Base filename (no extension) when ``isave``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from LCZ4py import plot_grid_only
    >>> fig = plot_grid_only(municipalities, color_by="pop_density",
    ...                      units="hab/km²", style="nature", isave=True)
    """
    import geopandas as gpd

    if not isinstance(grid, gpd.GeoDataFrame):
        raise TypeError(
            f"grid must be a geopandas.GeoDataFrame, got {type(grid).__name__}."
        )
    if grid.empty:
        raise ValueError("grid is empty; nothing to plot.")
    if grid.crs is None:
        raise ValueError(
            "grid has no CRS. Set one first, e.g. grid = grid.set_crs(4326)."
        )
    if color_by is not None and color_by not in grid.columns:
        raise ValueError(lcz_msg(
            "plot_grid_bad_color_by", lang,
            col=color_by, avail=", ".join(map(str, grid.columns)),
        ))
    if isave and save_extension.lower() not in _MPL_EXTENSIONS:
        raise ValueError(
            f"save_extension={save_extension!r} is not a matplotlib format. "
            f"Use one of: {', '.join(_MPL_EXTENSIONS)}."
        )

    spec = get_style(style, lang)
    export_dpi = int(dpi) if dpi else spec.dpi

    if figsize is None:
        figsize = (spec.width_mm["single"] / 25.4, spec.width_mm["single"] / 25.4)

    plot_grid = grid
    null_geom = plot_grid.geometry.isna()
    if bool(null_geom.any()):
        logger.warning("Dropping %d feature(s) with a null geometry.", int(null_geom.sum()))
        plot_grid = plot_grid.loc[~null_geom]
    if plot_grid.empty:
        raise ValueError("grid has no valid geometries to plot.")

    cmap_name = _resolve_cmap(cmap, lang)

    with plt.rc_context(spec.mpl_rcparams):
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_dpi(export_dpi)
        try:
            grid_3857 = plot_grid.to_crs(_WEB_MERCATOR)
        except Exception as exc:  # noqa: BLE001 - reported, then fallback
            logger.warning(
                "Could not reproject the grid to EPSG:3857 (%s); plotting in "
                "the source CRS without a scale bar.", exc,
            )
            grid_3857 = plot_grid
            projected = False
        else:
            projected = True

        is_points = _is_point_layer(grid_3857)
        draw_kwargs: dict = dict(
            ax=ax, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha,
        )

        legend_needed = color_by is not None
        if color_by is not None:
            import pandas as pd

            values = grid_3857[color_by]
            numeric = bool(pd.api.types.is_numeric_dtype(values.dtype))
            has_finite = False
            if numeric:
                try:
                    numeric_values = values.to_numpy(dtype="float64", na_value=np.nan)
                except (TypeError, ValueError):
                    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(
                        dtype="float64"
                    )
                has_finite = bool(np.isfinite(numeric_values).any())

            if numeric and not has_finite:
                logger.warning(
                    "Column %r has no finite values; plotting it uncolored.", color_by,
                )
                legend_needed = False

            if legend_needed and numeric:
                grid_3857.plot(
                    column=color_by, cmap=cmap_name, legend=True,
                    vmin=vmin, vmax=vmax,
                    missing_kwds=dict(color="lightgrey", label="No data"),
                    legend_kwds=dict(
                        shrink=0.55, pad=0.02, aspect=18,
                        label=_legend_label(color_by, units, legend_label),
                    ),
                    **draw_kwargs,
                )
                cbar = None
                for collection in reversed(ax.collections):
                    cbar = getattr(collection, "colorbar", None)
                    if cbar is not None:
                        break
                if cbar is not None:
                    cbar.ax.yaxis.set_major_formatter(
                        mticker.FuncFormatter(lambda v, _=None: f"{v:,.4g}")
                    )
                    cbar.ax.tick_params(labelsize=7)
            elif legend_needed:
                n_cats = int(grid_3857[color_by].nunique(dropna=True))
                if n_cats == 0:
                    logger.warning(
                        "Column %r has no values; plotting it uncolored.", color_by,
                    )
                    _plot_uncolored(ax, grid_3857, draw_kwargs, is_points)
                    n_cats = None
                show_legend = True
                if n_cats is not None and n_cats > _MAX_LEGEND_CATEGORIES:
                    logger.warning(
                        "Column %r has %d categories; legend suppressed (limit %d).",
                        color_by, n_cats, _MAX_LEGEND_CATEGORIES,
                    )
                    show_legend = False
                if n_cats is not None:
                    grid_3857.plot(
                        column=color_by, cmap=cmap_name, legend=show_legend,
                        categorical=True,
                        legend_kwds=dict(
                            loc="lower right", fontsize=7, frameon=True,
                            title=_legend_label(color_by, units, legend_label),
                            title_fontsize=8,
                        ),
                        **draw_kwargs,
                    )
            else:
                _plot_uncolored(ax, grid_3857, draw_kwargs, is_points)
        else:
            _plot_uncolored(ax, grid_3857, draw_kwargs, is_points)

        if projected:
            ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

        # ── Basemap ────────────────────────────────────────────────────────
        credit = None
        transformer = pyproj.Transformer.from_crs(_WEB_MERCATOR, "EPSG:4326", always_xy=True)
        if add_basemap:
            if not HAS_CONTEXTILY:
                logger.warning(lcz_msg("plot_grid_no_contextily", lang))
            elif not projected:
                logger.warning("Basemap skipped: the layer could not be reprojected to EPSG:3857.")
            else:
                source = basemap_source or cx.providers.CartoDB.Positron
                zoom = basemap_zoom if basemap_zoom is not None else _pick_zoom(ax, transformer)
                basemap_kwargs: dict = dict(source=source, crs=_WEB_MERCATOR, zoom=zoom)
                # contextily >= 1.7 adds its own credit line; suppress it so
                # the attribution is drawn once, with this figure's styling.
                try:
                    import inspect

                    if "attribution" in inspect.signature(cx.add_basemap).parameters:
                        basemap_kwargs["attribution"] = ""
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    pass
                try:
                    cx.add_basemap(ax, **basemap_kwargs)
                except Exception as exc:  # noqa: BLE001 - offline / tile error
                    logger.warning(
                        "Basemap unavailable (%s); continuing with a plain "
                        "background.", exc,
                    )
                else:
                    credit = attribution or _attribution_for(source)

        # ── Map furniture ──────────────────────────────────────────────────
        if credit:
            ax.text(
                0.995, 0.005, credit, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.5, color="#333333", zorder=12,
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5),
            )

        if projected:
            geod = pyproj.Geod(ellps="WGS84")
            if add_scalebar:
                _draw_scalebar(ax, transformer, geod)
            if add_north_arrow:
                _draw_north_arrow(ax)
        else:
            if add_scalebar or add_north_arrow:
                logger.debug(
                    "Skipping scale bar / north arrow: the layer could not be "
                    "projected to a metric CRS."
                )

        if title:
            ax.set_title(title, fontsize=max(8, spec.font_size + 2), pad=8)

        if show_axes:
            ax.set_axis_on()
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            ax.ticklabel_format(style="sci", scilimits=(-3, 4), axis="both")
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

        try:
            fig.tight_layout()
        except Exception:  # pragma: no cover - constrained layouts
            pass

        if isave and export_dpi != spec.dpi:
            # ``finalize_export`` saves at the preset DPI; honour an explicit
            # override while reusing the same preset styling and output dir.
            fig = finalize_export(
                fig, style=style, isave=False, save_extension=save_extension,
                filename=save_filename, lang=lang,
            )
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            save_path = os.path.join(OUTPUT_DIR, f"{save_filename}.{save_extension}")
            fig.savefig(save_path, dpi=export_dpi, bbox_inches="tight")
            logger.info("Saved: %s", os.path.abspath(save_path))
        else:
            fig = finalize_export(
                fig, style=style, isave=isave, save_extension=save_extension,
                filename=save_filename, lang=lang,
            )

    return fig


def _plot_uncolored(ax, gdf, draw_kwargs: dict, is_points: bool) -> None:
    """Draw a single-color layer (points get markers, polygons get fills)."""
    if is_points:
        gdf.plot(color="steelblue", markersize=8, ax=ax, edgecolor=draw_kwargs["edgecolor"],
                 linewidth=draw_kwargs["linewidth"], alpha=draw_kwargs["alpha"])
    else:
        gdf.plot(facecolor="none", ax=ax, edgecolor=draw_kwargs["edgecolor"],
                 linewidth=draw_kwargs["linewidth"], alpha=draw_kwargs["alpha"])
