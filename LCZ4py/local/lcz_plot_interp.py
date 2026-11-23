"""
lcz_plot_interp.py — publication-ready visualisation for interpolated rasters.

Renders multi-band GeoTIFFs (e.g. from lcz_interp_map, lcz_anomaly_map or
lcz_interp_map_plus) on real map coordinates, with a single shared colourbar
carrying the variable units.  On the Plotly path, rasters exceeding 4 M pixels
are pre-rendered with Datashader when it is installed.

The default renderer is the native static matplotlib/Agg backend
(``renderer="static"``): one 300 dpi PNG/PDF/SVG/TIFF written directly, with
no Plotly, Kaleido, Chromium or browser.  ``renderer="plotly"`` keeps the
original interactive figure with per-band dropdown menus.

Cartography follows :mod:`LCZ4py._internal.lcz_theme`: colourbar-safe palettes,
300 dpi journal presets (``style=``), and a north arrow + scale bar on
projected maps.  Residual/anomaly stacks get symmetric scales so zero stays at
the palette centre.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import logging
import os
from typing import Optional, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    import xarray as xr

    HAS_DATASHADER = True
except Exception:  # ImportError, or a broken datashader install
    ds = tf = xr = None  # type: ignore[assignment]
    HAS_DATASHADER = False

from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal import lcz_static
from LCZ4py._internal.lcz_static import resolve_renderer
from LCZ4py._internal.lcz_theme import (
    add_map_furniture, finalize_export, get_style,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 4)

# Diverging and sequential palettes that remain distinguishable under the most
# common colour-vision deficiencies (RdBu, viridis, cividis, Blues).
_PALETTES: dict[str, list[str]] = {
    # diverging — for anomalies / residuals
    "muted": ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
              "#fddbc7", "#f4a582", "#d6604d", "#b2182b"],
    # sequential — viridis
    "high": ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
    # sequential — cividis
    "viridi": ["#00224e", "#123570", "#3b496c", "#575d6d", "#707173",
               "#8a8678", "#a59c74", "#c3b369", "#e1cc55", "#fee838"],
    # sequential — Blues
    "deep": ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6",
             "#9ecae1", "#c6dbef", "#deebf7"],
}

_PALETTE_ALIASES = {
    "viridis": "high",
    "cividis": "viridi",
    "rdbu": "muted",
    "blues": "deep",
}


def _resolve_palette(name: str) -> list[str]:
    key = _PALETTE_ALIASES.get(str(name).lower(), str(name))
    if key not in _PALETTES:
        logger.warning(
            "Unknown palette '%s' — using 'muted'. Valid: %s",
            name, ", ".join(sorted(list(_PALETTES) + list(_PALETTE_ALIASES))),
        )
        key = "muted"
    return _PALETTES[key]


# Matplotlib's equivalents of the palettes above.  Same perceptual ordering and
# the same colour-vision-deficiency safety, so static and Plotly output agree.
_MPL_PALETTES: dict[str, str] = {
    "muted": "RdBu",
    "high": "viridis",
    "viridi": "cividis",
    "deep": "Blues",
}


def _mpl_palette_name(name: str) -> str:
    """Map a palette name to the equivalent registered matplotlib colormap."""
    key = _PALETTE_ALIASES.get(str(name).lower(), str(name))
    if key not in _MPL_PALETTES:
        logger.warning(
            "Unknown palette '%s' — using 'muted'. Valid: %s",
            name, ", ".join(sorted(list(_MPL_PALETTES) + list(_PALETTE_ALIASES))),
        )
        key = "muted"
    return _MPL_PALETTES[key]


def _colorscale(name: str, reverse: bool = False) -> list[tuple[float, str]]:
    pal = _resolve_palette(name)
    if reverse:
        pal = pal[::-1]
    n = len(pal)
    return [(i / (n - 1), c) for i, c in enumerate(pal)]


def _read_resampled(source, max_pixels: int = 4_000_000) -> tuple[np.ndarray, dict, tuple]:
    """Read a raster path or open dataset, decimating to ``max_pixels``."""
    ctx = nullcontext(source) if hasattr(source, "read") else rasterio.open(str(source))
    with ctx as src:
        h, w = src.height, src.width
        bounds = tuple(src.bounds)  # full-res extent; unaffected by resampling below
        if h * w > max_pixels:
            scale = (max_pixels / (h * w)) ** 0.5
            out_shape = (src.count, max(1, int(h * scale)), max(1, int(w * scale)))
            arr = src.read(out_shape=out_shape, resampling=Resampling.bilinear)
        else:
            arr = src.read()
        profile = src.profile.copy()
        profile.update(height=arr.shape[-2], width=arr.shape[-1])
        profile["descriptions"] = list(src.descriptions)
        profile["nodata"] = src.nodata
    return arr, profile, bounds


def _band_scale(band: np.ndarray, symmetric: Optional[bool],
                clip: Optional[tuple[float, float]]) -> tuple[float, float, bool]:
    """Colour limits for one band.

    Returns ``(vmin, vmax, is_symmetric)``.  ``symmetric`` is auto-detected:
    a band that straddles zero with roughly balanced extremes (a residual or
    anomaly field) is centred on zero so the palette midpoint means "no
    difference".
    """
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return 0.0, 1.0, False
    if clip is not None:
        lo, hi = (float(v) for v in clip)
        if not 0 <= lo < hi <= 100:
            raise ValueError(
                f"percentile_clip must satisfy 0 <= low < high <= 100, got {clip}."
            )
        vmin = float(np.percentile(finite, lo))
        vmax = float(np.percentile(finite, hi))
    else:
        vmin, vmax = float(finite.min()), float(finite.max())

    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmax == vmin:
        pad = abs(vmin) * 0.01 or 1.0
        vmin, vmax = vmin - pad, vmax + pad

    auto_sym = vmin < 0 < vmax and 0.4 <= abs(vmin) / max(abs(vmax), 1e-12) <= 2.5
    use_sym = bool(symmetric) if symmetric is not None else auto_sym
    if use_sym:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    return vmin, vmax, use_sym


def lcz_plot_interp(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader, np.ndarray],
    palette: str = "muted",
    direction: int = 1,
    isave: bool = False,
    save_extension: str = "html",
    *,
    style: str = "default",
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
    title: Optional[str] = None,
    caption: Optional[str] = None,
    lang: str = "en",
    units: Optional[str] = None,
    symmetric: Optional[bool] = None,
    facet: bool = False,
    percentile_clip: Optional[tuple[float, float]] = None,
    max_workers: Optional[int] = None,
    dpi: Optional[int] = None,
    renderer: Optional[str] = "static",
) -> object:
    """Render an interpolated raster stack as a publication-ready figure.

    Parameters
    ----------
    x : str, PathLike, rasterio dataset, or np.ndarray
        Source raster. Arrays must be shaped ``(bands, height, width)`` or
        ``(height, width)`` for a single band.
    palette : str
        Colour palette: ``"muted"`` (diverging), ``"high"``/``"viridis"``,
        ``"viridi"``/``"cividis"``, or ``"deep"``/``"blues"``.  All are
        colour-vision-deficiency safe.
    direction : {1, -1}
        ``1`` = normal palette; ``-1`` = reversed.
    isave : bool
        Save the figure to ``LCZ4r_output/``.
    save_extension : str
        Static output format: ``"png"`` (default), ``"pdf"``, ``"svg"``,
        ``"tiff"``/``"tif"``, ``"jpg"``/``"jpeg"`` or ``"webp"``. ``"html"``
        is only meaningful for ``renderer="plotly"`` and is coerced to
        ``"png"`` for the static renderer.
    style : str
        Publication style preset: 'default', 'nature', 'science', or
        'generic_bw'. Controls font, figure size (mm), DPI, and palette
        used when isave and save_extension != 'html'.
    add_scalebar : bool
        Add a cartographic scale bar (skipped for geographic/lon-lat CRS or
        when ``x`` is a bare ndarray with no known CRS).
    add_north_arrow : bool
        Add a north arrow (same skip conditions as ``add_scalebar``).
    title : str, optional
        Override the default figure title.
    caption : str, optional
        Small annotation added below the figure.
    lang : str
        Language for the default title — ``"en"``, ``"pt"``, ``"es"``, ``"zh"``.
    units : str, optional
        Units of the mapped variable (e.g. ``"°C"``, ``"mm"``).  Shown on the
        colourbar and hover text.
    symmetric : bool, optional
        Force a scale centred on zero (residual/anomaly panels).  ``None``
        (default) auto-detects bands that straddle zero with balanced extremes.
    facet : bool
        Render bands as a small-multiple grid (one panel per band) instead of
        a dropdown-switched single map.
    percentile_clip : (float, float), optional
        Percentiles used for the colour limits, e.g. ``(2, 98)`` to keep
        outliers from flattening the map.  ``None`` uses the full data range.
    max_workers : int, optional
        Thread workers for per-band preparation (default ``min(8, cpu_count)``).
    dpi : int, optional
        Export resolution; never below 300 dpi for static output.
    renderer : {"static", "plotly"}, optional
        ``"static"`` (default, also ``None``/``"matplotlib"``/``"agg"``)
        returns a matplotlib ``Figure`` and writes a native file with no
        Plotly/Kaleido dependency. ``"plotly"`` returns the original
        interactive figure with the per-band dropdown menus.

    Returns
    -------
    matplotlib.figure.Figure
        Static renderer (default).
    go.Figure
        ``renderer="plotly"``.
    """
    if x is None:
        raise ValueError("x is required.")
    if direction not in (1, -1):
        raise ValueError("direction must be 1 or -1.")
    if percentile_clip is not None and len(percentile_clip) != 2:
        raise ValueError("percentile_clip must be a (low, high) percentile pair.")

    renderer = resolve_renderer(renderer)

    bounds = None
    crs = None
    band_descs: list[str] = []
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            arr = x[np.newaxis, ...]
        elif x.ndim == 3:
            arr = x
        else:
            raise ValueError(
                f"Array input must be 2-D (height, width) or 3-D "
                f"(bands, height, width); got {x.ndim}-D."
            )
        if arr.shape[1] < 2 or arr.shape[2] < 2:
            raise ValueError(f"Raster is too small to map: {arr.shape[1]}x{arr.shape[2]}.")
        arr = np.asarray(arr, dtype=np.float32)
    elif hasattr(x, "read"):
        if x.count < 1 or x.height < 2 or x.width < 2:
            raise ValueError("Raster is empty or too small to map.")
        arr, profile, bounds = _read_resampled(x)
        band_descs = [d or "" for d in (profile.get("descriptions") or [])]
        crs = x.crs
        arr = np.asarray(arr, dtype=np.float32)
    else:
        arr, profile, bounds = _read_resampled(str(x))
        band_descs = [d or "" for d in (profile.get("descriptions") or [])]
        crs = profile.get("crs")
        arr = np.asarray(arr, dtype=np.float32)

    n, height, width = arr.shape
    if n < 1:
        raise ValueError("Raster has no bands.")
    if not np.isfinite(arr).any():
        raise ValueError("Raster contains no finite values — nothing to map.")

    labels = [
        band_descs[i] if i < len(band_descs) and band_descs[i] else f"Band {i + 1}"
        for i in range(n)
    ]
    unit_suffix = f" [{units}]" if units else ""

    # Real map coordinates so axes read in metres/degrees, not pixel indices.
    xs = ys = None
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        xs = xmin + (np.arange(width) + 0.5) * (xmax - xmin) / width
        ys = ymin + (np.arange(height) + 0.5) * (ymax - ymin) / height

    workers = DEFAULT_MAX_WORKERS if max_workers is None else max(1, int(max_workers))
    workers = min(workers, max(1, n))

    def _prepare(i: int):
        # flipud: raster row 0 is north, but Plotly y increases upwards — align
        # them (same convention as lcz_plot_map._create_continuous_map).
        band = np.flipud(arr[i])
        vmin, vmax, sym = _band_scale(arr[i], symmetric, percentile_clip)
        return i, band, vmin, vmax, sym

    if workers > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            prepared = list(ex.map(_prepare, range(n)))  # deterministic order
    else:
        prepared = [_prepare(i) for i in range(n)]

    scales = {i: (vmin, vmax, sym) for i, _b, vmin, vmax, sym in prepared}
    colorscale = _colorscale(palette, reverse=(direction == -1))
    n_sym = sum(1 for _i, _b, _v0, _v1, sym in prepared if sym)
    if n_sym:
        logger.info("%d band(s) plotted on a symmetric (zero-centred) scale.", n_sym)

    if renderer == "static":
        return _plot_interp_static(
            prepared,
            labels,
            palette=palette,
            direction=direction,
            units=units,
            title=title,
            caption=caption,
            bounds=bounds,
            crs=crs,
            style=style,
            dpi=dpi,
            isave=isave,
            save_extension=save_extension,
            lang=lang,
            facet=facet,
            add_scalebar=add_scalebar,
            add_north_arrow=add_north_arrow,
            height=height,
            width=width,
        )

    use_ds = HAS_DATASHADER and not facet and height * width > 4_000_000
    if use_ds:
        logger.info(
            "Raster has %.1f M pixels — pre-rendering with Datashader; the "
            "colourbar is omitted on this path.", height * width / 1e6,
        )

    if facet:
        ncols = 2 if n > 1 else 1
        nrows = int(np.ceil(n / ncols))
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=[f"{lb}{unit_suffix}" for lb in labels],
            horizontal_spacing=0.08, vertical_spacing=0.12,
        )
        for i, band, vmin, vmax, _sym in prepared:
            r, c = divmod(i, ncols)
            fig.add_trace(
                go.Heatmap(
                    z=band, x=xs, y=ys,
                    colorscale=colorscale, zmin=vmin, zmax=vmax,
                    name=labels[i],
                    hovertemplate=(
                        f"{labels[i]}: %{{z:.2f}}{unit_suffix}<extra></extra>"
                    ),
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=f"{labels[i]}{unit_suffix}", side="right"),
                        len=max(0.25, 0.9 / nrows),
                        thickness=12,
                        x=1.02 if c == 1 else 0.46,
                        y=1 - (r + 0.5) / nrows,
                    ),
                ),
                row=r + 1, col=c + 1,
            )
        # Equal aspect ratio per subplot.
        for i in range(n):
            suffix = "" if i == 0 else str(i + 1)
            fig.update_layout(**{
                f"yaxis{suffix}": dict(scaleanchor=f"x{suffix}", scaleratio=1),
            })
    else:
        fig = go.Figure()
        for i, band, vmin, vmax, _sym in prepared:
            if use_ds:
                x_arr = np.linspace(0, 1, band.shape[1])
                y_arr = np.linspace(0, 1, band.shape[0])
                da = xr.DataArray(band, coords=[("y", y_arr), ("x", x_arr)])
                cvs = ds.Canvas(plot_width=2048, plot_height=2048)
                agg = cvs.raster(da)
                img = tf.shade(agg, cmap=_resolve_palette(palette), how="linear")
                fig.add_trace(go.Image(
                    z=img.to_numpy(), name=labels[i], visible=(i == 0),
                ))
                logger.info(
                    "Datashader pre-rendered band '%s' — colourbar and units are "
                    "not available on this path.", labels[i],
                )
                continue
            fig.add_trace(go.Heatmap(
                z=band, x=xs, y=ys,
                colorscale=colorscale,
                zmin=vmin, zmax=vmax,
                name=labels[i],
                visible=(i == 0),
                coloraxis="coloraxis",
                hovertemplate=f"{labels[i]}: %{{z:.2f}}{unit_suffix}<extra></extra>",
            ))

        if not use_ds:
            v0, v1, _ = scales[0]
            fig.update_layout(coloraxis=dict(
                cmin=v0, cmax=v1,
                colorscale=colorscale,
                colorbar=dict(
                    title=dict(text=f"{labels[0]}{unit_suffix}", side="right"),
                    len=0.85, thickness=16, outlinewidth=0,
                ),
            ))

        if n > 1:
            steps = []
            for i in range(n):
                vmin, vmax, _sym = scales[i]
                layout_update = {"title.text": labels[i]}
                if not use_ds:
                    layout_update.update({
                        "coloraxis.cmin": vmin,
                        "coloraxis.cmax": vmax,
                        "coloraxis.colorbar.title.text": f"{labels[i]}{unit_suffix}",
                    })
                steps.append(dict(
                    method="update",
                    args=[{"visible": [j == i for j in range(n)]}, layout_update],
                    label=labels[i],
                ))
            fig.update_layout(updatemenus=[dict(
                active=0, buttons=steps, x=0.0, y=1.14, xanchor="left",
                direction="down", showactive=True,
            )])

    chart_title = title or lcz_msg("lcz_interp_map_title", lang)
    spec = get_style(style, lang)
    fig.update_layout(
        title=dict(text=chart_title, x=0.5, xanchor="center"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=spec.font_family, size=spec.font_size),
        margin=dict(l=70, r=30, t=110 if not facet else 80, b=60),
        hoverlabel=dict(bgcolor="white", font=dict(size=spec.font_size + 1)),
    )
    geographic = crs is not None and bool(getattr(crs, "is_geographic", False))
    if facet:
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    else:
        fig.update_xaxes(
            showgrid=False, zeroline=False,
            title_text=(("Longitude [°]" if geographic else "Easting [m]")
                        if crs is not None else ""),
            showticklabels=bounds is not None,
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False,
            title_text=(("Latitude [°]" if geographic else "Northing [m]")
                        if crs is not None else ""),
            showticklabels=bounds is not None,
            scaleanchor="x", scaleratio=1,
        )

    if caption:
        fig.add_annotation(
            text=caption, xref="paper", yref="paper",
            x=0.5, y=-0.08, showarrow=False,
            font=dict(size=max(8, spec.font_size - 2), color="dimgray"),
        )

    if bounds is not None and not facet:
        add_map_furniture(fig, bounds=bounds, crs=crs,
                          add_scalebar=add_scalebar,
                          add_north_arrow=add_north_arrow)

    fig = finalize_export(fig, style=style, isave=isave, save_extension=save_extension,
                          filename="lcz_interp_map", lang=lang)

    return fig


# ── Native static (matplotlib/Agg) renderer ──────────────────────────────────

def _pixel_axes(ax) -> None:
    """Label a non-georeferenced raster axes in pixel indices."""
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.set_aspect("equal", adjustable="box")


def _static_map_furniture(
    ax,
    extent: tuple,
    crs,
    *,
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
) -> bool:
    """Axis labels + scale bar + north arrow for a georeferenced panel.

    Calls the individual :mod:`lcz_static` helpers rather than
    ``add_map_furniture_mpl``, whose ``add_scalebar``/``add_north_arrow``
    keyword arguments shadow the module-level functions of the same name
    (same workaround as ``lcz_plot_map._create_static_map``).
    """
    st = lcz_static
    is_geo = st.apply_map_axes(ax, extent, crs)
    if add_scalebar:
        st.add_scalebar(ax, st.metric_width(extent, crs), is_geo)
    if add_north_arrow and not is_geo:
        st.add_north_arrow(ax)
    return is_geo


def _plot_interp_static(
    prepared: list,
    labels: list[str],
    *,
    palette: str,
    direction: int,
    units: Optional[str],
    title: Optional[str],
    caption: Optional[str],
    bounds,
    crs,
    style: str,
    dpi: Optional[int],
    isave: bool,
    save_extension: str,
    lang: str,
    facet: bool,
    add_scalebar: bool,
    add_north_arrow: bool,
    height: int,
    width: int,
):
    """Draw the interpolated stack with matplotlib (no Plotly/Kaleido needed).

    ``facet=False`` draws the first band on its own (the Plotly renderer uses a
    dropdown for the rest); ``facet=True`` draws every band as a small-multiple
    grid sharing one colourbar.
    """
    st = lcz_static

    n = len(prepared)
    unit_suffix = f" [{units}]" if units else ""
    mpl_palette = _mpl_palette_name(palette)
    colorscale = f"{mpl_palette}_r" if direction == -1 else mpl_palette
    chart_title = title or lcz_msg("lcz_interp_map_title", lang)

    if bounds is not None:
        xmin, ymin, xmax, ymax = (float(v) for v in bounds)
        extent = (xmin, xmax, ymin, ymax)
    else:
        extent = (0.0, float(width), 0.0, float(height))
    aspect = height / max(width, 1)
    is_geo = bool(getattr(crs, "is_geographic", False)) if crs is not None else False

    with st.style_context(style):
        if facet:
            ncols = 2 if n > 1 else 1
            nrows = int(np.ceil(n / ncols))

            # One colourbar for the whole grid when the per-band limits are
            # comparable; otherwise keep each panel's own scaling and give it
            # its own colourbar (a pooled scale would flatten the panels).
            ranges = [(v0, v1) for _i, _b, v0, v1, _s in prepared]
            vmin_shared = min(r[0] for r in ranges)
            vmax_shared = max(r[1] for r in ranges)
            all_sym = all(bool(s) for _i, _b, _v0, _v1, s in prepared)
            if all_sym:
                m = max(abs(vmin_shared), abs(vmax_shared))
                vmin_shared, vmax_shared = -m, m
            widest = max((v1 - v0 for v0, v1 in ranges), default=1.0) or 1.0
            share_scale = (vmax_shared - vmin_shared) <= 1.25 * widest
            if not share_scale:
                logger.info(
                    "Band ranges differ widely — drawing one colourbar per "
                    "panel instead of a pooled shared scale.",
                )

            _unused_spec, base_w, _unused_h = st.figure_dims(style)
            fig, spec, _ = st.new_figure(
                style,
                height_mm=(base_w / ncols) * aspect * nrows + 20.0,
                dpi=dpi,
            )
            if share_scale:
                fig.subplots_adjust(
                    left=0.09, right=0.85, bottom=0.17,
                    top=0.84 if nrows > 1 else 0.80,
                    wspace=0.16, hspace=0.30,
                )
            else:
                # Leave a gutter between panels for their own colourbars.
                fig.subplots_adjust(
                    left=0.08, right=0.80, bottom=0.17,
                    top=0.84 if nrows > 1 else 0.80,
                    wspace=0.50, hspace=0.34,
                )

            panel_axes = []
            panels = []
            for i, band, v0, v1, _sym in prepared:
                ax = fig.add_subplot(nrows, ncols, i + 1)
                image = st.plot_continuous_raster(
                    ax, band, extent, colorscale=colorscale,
                )
                image.set_clim(
                    vmin_shared if share_scale else v0,
                    vmax_shared if share_scale else v1,
                )
                ax.set_title(
                    f"{labels[i]}{unit_suffix}",
                    fontsize=max(6, spec.font_size - 1),
                    pad=3,
                )
                if bounds is not None:
                    st.apply_map_axes(ax, extent, crs)
                    if i == 0:
                        if add_scalebar:
                            st.add_scalebar(
                                ax, st.metric_width(extent, crs), is_geo,
                            )
                        if add_north_arrow and not is_geo:
                            st.add_north_arrow(ax)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.tick_params(length=0)
                panel_axes.append(ax)
                panels.append(image)

            if share_scale:
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize

                sm = ScalarMappable(
                    norm=Normalize(vmin_shared, vmax_shared),
                    cmap=st.mpl_cmap(colorscale),
                )
                sm.set_array(np.array([]))
                cb = fig.colorbar(
                    sm, ax=panel_axes, fraction=0.03, pad=0.02, shrink=0.92,
                )
                cb.set_label(units or "Value", fontsize=spec.font_size)
                cb.ax.tick_params(labelsize=max(6, spec.font_size - 1))
                cb.outline.set_linewidth(0.6)
            else:
                for image in panels:
                    cb = fig.colorbar(
                        image, fraction=0.035, pad=0.02, shrink=0.85,
                    )
                    cb.set_label(units or "Value", fontsize=max(5.5, spec.font_size - 1))
                    cb.ax.tick_params(labelsize=max(5.5, spec.font_size - 2))
                    cb.outline.set_linewidth(0.6)

            st.add_header(fig, chart_title, f"{n} panel(s)", spec)
            st.add_footer(fig, caption, None, spec)
            st.save_static(
                fig,
                "lcz_interp_map",
                isave=isave,
                save_extension=save_extension,
                style=style,
                dpi=dpi,
                lang=lang,
            )
            return fig

        # ── Single panel ────────────────────────────────────────────────────
        _i0, band0, vmin0, vmax0, _sym0 = prepared[0]
        if n > 1:
            logger.info(
                "Static renderer draws band 1 of %d ('%s'); use facet=True to "
                "show every band.", n, labels[0],
            )

        fig, spec, _ = st.new_figure(
            style, aspect=aspect, extra_height_mm=18.0, dpi=dpi,
        )
        ax = st.add_axes(fig, bottom=0.16, top=st.figure_top(True))

        image = st.plot_continuous_raster(
            ax, band0, extent, colorscale=colorscale,
        )
        image.set_clim(vmin0, vmax0)

        if bounds is not None:
            _static_map_furniture(
                ax, extent, crs,
                add_scalebar=add_scalebar,
                add_north_arrow=add_north_arrow,
            )
        else:
            _pixel_axes(ax)

        st.add_colorbar(
            fig, ax, image,
            units=f"{labels[0]}{unit_suffix}",
            fontsize=spec.font_size,
        )
        st.add_header(
            fig, chart_title,
            f"Band 1 of {n}" if n > 1 else None,
            spec,
        )
        st.add_footer(fig, caption, None, spec)
        st.save_static(
            fig,
            "lcz_interp_map",
            isave=isave,
            save_extension=save_extension,
            style=style,
            dpi=dpi,
            lang=lang,
        )

    return fig


__all__ = [
    "lcz_plot_interp", "HAS_DATASHADER", "DEFAULT_MAX_WORKERS",
]
