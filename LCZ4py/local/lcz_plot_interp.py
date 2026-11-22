"""
lcz_plot_interp.py — publication-ready visualisation for interpolated rasters.

Renders multi-band GeoTIFFs (e.g. from lcz_interp_map, lcz_anomaly_map or
lcz_interp_map_plus) as interactive Plotly heatmaps on real map coordinates,
with per-band dropdown menus and a single shared colourbar carrying the
variable units.  Rasters exceeding 4 M pixels are pre-rendered with Datashader
when it is installed.

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
) -> go.Figure:
    """Render an interpolated raster stack as an interactive Plotly figure.

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
        ``"html"`` (interactive) or ``"png"``/``"pdf"`` (static).
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

    Returns
    -------
    go.Figure
    """
    if x is None:
        raise ValueError("x is required.")
    if direction not in (1, -1):
        raise ValueError("direction must be 1 or -1.")
    if percentile_clip is not None and len(percentile_clip) != 2:
        raise ValueError("percentile_clip must be a (low, high) percentile pair.")

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


__all__ = [
    "lcz_plot_interp", "HAS_DATASHADER", "DEFAULT_MAX_WORKERS",
]
