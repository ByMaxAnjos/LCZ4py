"""
lcz_plot_interp.py — interactive visualisation for interpolated LCZ rasters.

Renders multi-band GeoTIFFs (e.g. from lcz_interp_map or lcz_anomaly_map) as
interactive Plotly heatmaps with per-band dropdown menus. Falls back to
Datashader pre-rendering for rasters exceeding 4 M pixels, if installed.
"""

from __future__ import annotations
import logging
import os
from typing import Optional, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling

import plotly.graph_objects as go

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    import xarray as xr
    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False

from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export, add_map_furniture

logger = logging.getLogger(__name__)

_PALETTES: dict[str, list[str]] = {
    "muted":  ["#3E5C76", "#5286B4", "#82B4C5", "#B0D2BC", "#E0E0A2", "#E0AC65", "#D07D3F", "#B04629", "#7D2A1D"],
    "high":   ["#001F3F", "#1E5288", "#5DAA9D", "#A3D29A", "#F5E08C", "#F2A65A", "#E16A4F", "#B92F26", "#5C0A0A"],
    "viridi": ["#440154", "#3E5C76", "#5286B4", "#82B4C5", "#B0D2BC", "#E0E0A2", "#FCFDBF"],
    "deep":   ["#00204C", "#072F5F", "#1B3F73", "#34568B", "#586891", "#7C7B98", "#A09CB0", "#C9C0CA", "#FCEAE3"],
}


def _colorscale(name: str, reverse: bool = False) -> list[tuple[float, str]]:
    pal = _PALETTES.get(name, _PALETTES["muted"])
    if reverse:
        pal = pal[::-1]
    n = len(pal)
    return [(i / (n - 1), c) for i, c in enumerate(pal)]


def _read_resampled(path: str, max_pixels: int = 4_000_000) -> tuple[np.ndarray, dict, tuple]:
    with rasterio.open(path) as src:
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
    return arr, profile, bounds


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
) -> go.Figure:
    """Render an interpolated raster stack as an interactive Plotly figure.

    Parameters
    ----------
    x : str, PathLike, rasterio dataset, or np.ndarray
        Source raster. Arrays must be shaped ``(bands, height, width)`` or
        ``(height, width)`` for a single band.
    palette : {"muted", "high", "viridi", "deep"}
        Diverging colour palette for the heatmap.
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

    Returns
    -------
    go.Figure
    """
    if x is None:
        raise ValueError("x is required.")

    bounds = None
    crs = None
    if isinstance(x, np.ndarray):
        arr = x if x.ndim == 3 else x[np.newaxis, ...]
        band_descs: list[str] = []
    elif hasattr(x, "read"):
        arr = x.read()
        band_descs = [x.descriptions[i] or f"Band {i + 1}" for i in range(x.count)]
        bounds = tuple(x.bounds)
        crs = x.crs
    else:
        arr, profile, bounds = _read_resampled(str(x))
        band_descs = list(profile.get("descriptions") or [])
        crs = profile.get("crs")

    finite_vals = arr[np.isfinite(arr)]
    vmin = float(finite_vals.min()) if len(finite_vals) else 0.0
    vmax = float(finite_vals.max()) if len(finite_vals) else 1.0

    n = arr.shape[0]
    labels = [band_descs[i] if i < len(band_descs) and band_descs[i] else f"Band {i + 1}"
               for i in range(n)]

    colorscale = _colorscale(palette, reverse=(direction == -1))
    use_ds = HAS_DATASHADER and arr.shape[1] * arr.shape[2] > 4_000_000

    fig = go.Figure()
    for i in range(n):
        # flipud: raster row 0 is north, but Plotly y0=ymin (south) — align
        # them (same convention as lcz_plot_map._create_continuous_map).
        band = np.flipud(arr[i])
        label = labels[i]
        if use_ds:
            x_arr = np.linspace(0, 1, band.shape[1])
            y_arr = np.linspace(0, 1, band.shape[0])
            da = xr.DataArray(band, coords=[("y", y_arr), ("x", x_arr)])
            cvs = ds.Canvas(plot_width=2048, plot_height=2048)
            agg = cvs.raster(da)
            img = tf.shade(agg, cmap=_PALETTES.get(palette, _PALETTES["muted"]), how="linear")
            fig.add_trace(go.Image(z=img.to_numpy(), name=label, visible=(i == 0)))
        else:
            fig.add_trace(go.Heatmap(
                z=band,
                colorscale=colorscale,
                zmin=vmin, zmax=vmax,
                name=label,
                visible=(i == 0),
                colorbar=dict(title=label, len=0.8) if n <= 4 else None,
                hovertemplate=f"{label}: %{{z:.2f}}<extra></extra>",
            ))

    if n > 1:
        steps = [
            dict(
                method="update",
                args=[
                    {"visible": [j == i for j in range(n)]},
                    {"title.text": labels[i]},
                ],
                label=labels[i],
            )
            for i in range(n)
        ]
        fig.update_layout(updatemenus=[dict(active=0, buttons=steps, x=0.1, y=1.12)])

    chart_title = title or lcz_msg("lcz_interp_map_title", lang)
    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=20), x=0.5, xanchor="center"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=0, r=40, t=80, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )

    if caption:
        fig.add_annotation(
            text=caption, xref="paper", yref="paper",
            x=0.5, y=-0.05, showarrow=False,
            font=dict(size=10, color="gray"),
        )

    if bounds is not None:
        add_map_furniture(fig, bounds=bounds, crs=crs,
                           add_scalebar=add_scalebar, add_north_arrow=add_north_arrow)

    fig = finalize_export(fig, style=style, isave=isave, save_extension=save_extension,
                           filename="lcz_interp_map", lang=lang)

    return fig


__all__ = ["lcz_plot_interp"]
