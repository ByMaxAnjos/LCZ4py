"""
lcz_plot_parameters.py

Ultra-fast interactive LCZ parameter visualization using:
- Plotly for interactive WebGL rendering
- Datashader for massive raster rendering (>100MP)
- GeoArrow for efficient spatial data transfer
- Polars for vectorized metadata operations
- WarperVRT for efficient remote raster handling
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Union, Literal

import numpy as np
import polars as pl
import pyarrow as pa
import rasterio
import rioxarray  # noqa: F401 - registers the .rio accessor used on xr.Dataset
import xarray as xr
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform

import plotly.graph_objects as go

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    import colorcet as cc
    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False

# GeoArrow imports
try:
    import geoarrow.pyarrow as ga
    import geoarrow.pandas as gap
    HAS_GEOARROW = True
except ImportError:
    HAS_GEOARROW = False

from LCZ4py._internal.lcz_parameters_data import (
    PARAM_NAMES, PARAM_UNITS,
    PARAM_PALETTE_DEFAULT, PARAM_PALETTE_INCLUSIVE,
    LCZ_NAMES, LCZ_COLORS, LCZ_COLORBLIND,
)
from LCZ4py._internal.lcz_theme import (
    finalize_export, add_map_furniture, METBREWER_PALETTES as _METBREWER_PALETTES,
)

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"


# ── MetBrewer palettes as Plotly colorscales ──────────────────────────────────
# _METBREWER_PALETTES now lives in LCZ4py._internal.lcz_theme (imported above
# as METBREWER_PALETTES) — kept the local alias so nothing below has to change.


@lru_cache(maxsize=32)
def _palette_to_plotly_colorscale(name: str, n: int = 256) -> list[tuple[float, str]]:
    """Convert MetBrewer palette to Plotly colorscale with interpolation."""
    palette = _METBREWER_PALETTES.get(name, _METBREWER_PALETTES["Archambault"])
    
    # Parse hex colors to RGB tuples
    rgb_colors = []
    for hex_color in palette:
        h = hex_color.lstrip("#")
        rgb_colors.append(tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4)))
    
    # Interpolate to n colors
    from scipy.interpolate import interp1d
    x = np.linspace(0, 1, len(rgb_colors))
    x_new = np.linspace(0, 1, n)
    
    interpolated = []
    for channel in range(3):
        f = interp1d(x, [c[channel] for c in rgb_colors], kind='cubic')
        interpolated.append(f(x_new))
    
    # Build Plotly colorscale: [(position, color), ...]
    colorscale = []
    for i in range(n):
        pos = i / (n - 1)
        r, g, b = interpolated[0][i], interpolated[1][i], interpolated[2][i]
        colorscale.append((pos, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"))
    
    return colorscale


def _palette_to_datashader_cmap(name: str) -> list[tuple[float, tuple]]:
    """Convert palette to Datashader-compatible colormap."""
    palette = _METBREWER_PALETTES.get(name, _METBREWER_PALETTES["Archambault"])
    rgb_colors = []
    for hex_color in palette:
        h = hex_color.lstrip("#")
        rgb_colors.append(tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4)))
    
    # Create evenly spaced positions
    return [(i / (len(rgb_colors) - 1), c) for i, c in enumerate(rgb_colors)]


# ── Ultra-fast raster reader with memory mapping ──────────────────────────────

class FastRasterReader:
    """High-performance raster reader with multiple optimization strategies."""
    
    def __init__(
        self,
        path: Union[str, Path, rasterio.io.DatasetReader],
        max_pixels: int = 10_000_000,
        use_mmap: bool = True,
        resampling: Resampling = Resampling.bilinear,
    ):
        self.max_pixels = max_pixels
        self.use_mmap = use_mmap
        self.resampling = resampling
        self._path = str(path) if not hasattr(path, 'read') else None
        self._dataset = path if hasattr(path, 'read') else None
        
    def __enter__(self):
        if self._dataset is None:
            self._dataset = rasterio.open(self._path)
        return self
    
    def __exit__(self, *args):
        if self._path is not None and self._dataset is not None:
            self._dataset.close()
            self._dataset = None
    
    def read_band(
        self,
        band: int = 1,
        window: Optional[rasterio.windows.Window] = None,
        out_shape: Optional[tuple[int, int]] = None,
    ) -> tuple[np.ndarray, dict]:
        """Read a single band with optimal strategy."""
        src = self._dataset
        h, w = src.height, src.width
        
        # Calculate target dimensions
        if out_shape is None and (h * w) > self.max_pixels:
            scale = (self.max_pixels / (h * w)) ** 0.5
            out_shape = (max(1, int(h * scale)), max(1, int(w * scale)))
        
        # Read with optional downsampling
        arr = src.read(
            band,
            window=window,
            out_shape=out_shape,
            resampling=self.resampling,
        )
        
        # Create profile dict
        profile = {
            "transform": src.transform,
            "crs": src.crs,
            "height": arr.shape[0],
            "width": arr.shape[1],
            "dtype": str(arr.dtype),
        }
        
        return arr, profile


def read_with_warped_vrt(
    path: str,
    target_crs: str = "EPSG:3857",
    max_pixels: int = 10_000_000,
) -> tuple[np.ndarray, dict]:
    """Read raster with on-the-fly reprojection via WarpedVRT (no temp files)."""
    with rasterio.open(path) as src:
        # Calculate optimal output dimensions
        h, w = src.height, src.width
        if h * w > max_pixels:
            scale = (max_pixels / (h * w)) ** 0.5
            out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))
        else:
            out_h, out_w = h, w
        
        # Use WarpedVRT for efficient reprojection
        with WarpedVRT(
            src,
            crs=target_crs,
            resampling=Resampling.bilinear,
            transform=calculate_default_transform(
                src.crs, target_crs, w, h, *src.bounds
            )[0],
            width=out_w,
            height=out_h,
        ) as vrt:
            arr = vrt.read(1)
            return arr, {
                "transform": vrt.transform,
                "crs": vrt.crs,
                "height": vrt.height,
                "width": vrt.width,
            }


# ── Datashader-based rendering for massive rasters ───────────────────────────

def _render_with_datashader(
    arr: np.ndarray,
    transform,
    palette_name: str,
    extent: Optional[tuple] = None,
) -> go.Image:
    """Render raster using Datashader for WebGL performance."""
    from datashader.mpl_ext import dsshow

    h, w = arr.shape
    
    # Create coordinate arrays from transform
    if extent is None:
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + w * transform.a
        ymin = ymax + h * transform.e
        extent = (xmin, xmax, ymin, ymax)
    
    # Create xarray DataArray
    x = np.linspace(extent[0], extent[1], w)
    y = np.linspace(extent[2], extent[3], h)
    da = xr.DataArray(arr, coords=[("y", y), ("x", x)], name="value")
    
    # Create canvas
    cvs = ds.Canvas(
        plot_width=min(w, 4096),  # Limit for WebGL
        plot_height=min(h, 4096),
        x_range=(extent[0], extent[1]),
        y_range=(extent[2], extent[3]),
    )
    
    # Aggregate and shade
    agg = cvs.raster(da, upsample_method="linear")
    cmap = _palette_to_datashader_cmap(palette_name)
    img = tf.shade(agg, cmap=cmap, how="linear")
    
    # Convert to Plotly
    rgba = img.to_numpy()
    return go.Image(
        z=rgba,
        x0=extent[0],
        y0=extent[2],
        dx=(extent[1] - extent[0]) / rgba.shape[1],
        dy=(extent[3] - extent[2]) / rgba.shape[0],
        hovertemplate="x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>",
    )


# ── Plotly interactive plot ──────────────────────────────────────────────────

def _plot_one_interactive(
    arr: np.ndarray,
    profile: dict,
    name: str,
    inclusive: bool,
    use_datashader: bool = False,
    palette_override: Optional[str] = None,
) -> go.Figure:
    """Create an interactive Plotly figure for one parameter."""
    
    palette_map = PARAM_PALETTE_INCLUSIVE if inclusive else PARAM_PALETTE_DEFAULT
    palette_name = palette_override or palette_map.get(name, "Archambault")
    colorscale = _palette_to_plotly_colorscale(palette_name)
    
    # Mask nodata
    masked = np.where((arr == 0) | np.isnan(arr) | np.isinf(arr), np.nan, arr)

    # Calculate extent
    h, w = arr.shape
    transform = profile.get("transform")
    if transform:
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + w * transform.a
        ymin = ymax + h * transform.e
        extent = (xmin, xmax, ymin, ymax)
        # flipud: raster row 0 is north, but Plotly y0=ymin (south) — align
        # them (same convention as lcz_plot_map._create_continuous_map).
        # The no-transform fallback below already compensates via an
        # inverted extent, so it must NOT be flipped again.
        arr = np.flipud(arr)
        masked = np.flipud(masked)
    else:
        extent = (0, w, h, 0)
    
    # Choose rendering strategy based on size
    h, w = arr.shape
    use_ds = HAS_DATASHADER and (use_datashader or (h * w > 4_000_000))

    fig = go.Figure()

    if use_ds and transform:
        # Use Datashader for massive rasters
        img_trace = _render_with_datashader(arr, transform, palette_name, extent)
        fig.add_trace(img_trace)
    else:
        # Direct Plotly heatmap
        fig.add_trace(go.Heatmap(
            z=masked,
            x0=extent[0],
            y0=extent[2],
            dx=(extent[1] - extent[0]) / w,
            dy=(extent[3] - extent[2]) / h,
            colorscale=colorscale,
            zmin=np.nanmin(masked) if not np.all(np.isnan(masked)) else 0,
            zmax=np.nanmax(masked) if not np.all(np.isnan(masked)) else 1,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=PARAM_UNITS.get(name, ""),
                    font=dict(size=14, color="black"),
                ),
                tickfont=dict(size=11),
                thickness=20,
                len=0.8,
            ),
            hovertemplate=(
                f"<b>{PARAM_NAMES.get(name, name)}</b><br>"
                "Value: %{z:.4f}<br>"
                "X: %{x:.5f}<br>"
                "Y: %{y:.5f}<extra></extra>"
            ),
            opacity=1.0,
        ))
    
    # Configure layout
    full_title = PARAM_NAMES.get(name, name)
    fig.update_layout(
        title=dict(
            text=full_title,
            font=dict(size=20, color="black"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=0, r=40, t=50, b=0),
        width=1000,
        height=800,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    return fig


# ── GeoArrow integration for spatial metadata ─────────────────────────────────

def create_geoarrow_metadata(
    profile: dict,
    arr_shape: tuple[int, int],
) -> Optional[pa.Table]:
    """Create a GeoArrow table with raster footprint metadata."""
    if not HAS_GEOARROW:
        return None
    
    transform = profile.get("transform")
    if transform is None:
        return None
    
    h, w = arr_shape
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + w * transform.a
    ymin = ymax + h * transform.e
    
    # Create bounding box as GeoArrow polygon
    bbox_coords = np.array([
        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]
    ])
    
    # Build Arrow arrays
    schema = ga.schema(["geometry:geometry[polygon]"])
    arrays = [
        pa.array([bbox_coords.ravel().tolist()], type=pa.list_(pa.float64(), 10))
    ]
    
    return pa.Table.from_arrays(arrays, schema=schema)


def _finalize_figure(
    fig: go.Figure,
    name: str,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    isave: bool,
    save_extension: str,
    style: str = "default",
    lang: str = "en",
) -> go.Figure:
    """Apply shared title/caption annotations, then delegate export to lcz_theme."""
    if title:
        fig.update_layout(title=dict(text=f"{title}<br><sup>{subtitle or ''}</sup>"))
    if caption:
        fig.add_annotation(
            text=caption,
            xref="paper", yref="paper",
            x=0.01, y=-0.02,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

    return finalize_export(
        fig, style=style, isave=isave, save_extension=save_extension,
        filename=name, lang=lang,
    )


def _plot_from_dataset(
    ds_: xr.Dataset,
    iselect: Union[str, Sequence[str], None],
    all_params: bool,
    inclusive: bool,
    isave: bool,
    save_extension: str,
    use_datashader: bool,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    renderer: Literal["plotly", "datashader", "auto"],
    style: str = "default",
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
    lang: str = "en",
) -> Union[go.Figure, list[go.Figure]]:
    """Plot parameters straight from an lcz_get_ucp ``combined_rasters`` Dataset.

    Unlike the file-based path, every data variable is a real parameter (no
    leading LCZ-class band to skip), and bands are addressed by name instead
    of position.
    """
    if all_params:
        band_names = list(ds_.data_vars)
    elif iselect is not None:
        band_names = [iselect] if isinstance(iselect, str) else list(iselect)
        missing = [n for n in band_names if n not in ds_.data_vars]
        if missing:
            raise ValueError(f"Parameter(s) not found in dataset: {missing}")
    else:
        raise ValueError("Provide either iselect='Param' or all_params=True")

    transform = ds_.rio.transform()
    crs = ds_.rio.crs
    figures = []

    for name in band_names:
        arr = np.squeeze(np.asarray(ds_[name].values, dtype=float))
        profile = {"transform": transform, "crs": crs, "height": arr.shape[0], "width": arr.shape[1]}

        use_ds = (renderer == "datashader") or (
            renderer == "auto" and use_datashader and arr.size > 4_000_000
        )
        fig = _plot_one_interactive(arr, profile, name, inclusive, use_datashader=use_ds)
        if transform is not None:
            h, w = arr.shape
            bounds = (transform.c, transform.f + h * transform.e, transform.c + w * transform.a, transform.f)
            add_map_furniture(fig, bounds=bounds, crs=crs, add_scalebar=add_scalebar, add_north_arrow=add_north_arrow)
        figures.append(_finalize_figure(fig, name, title, subtitle, caption, isave, save_extension, style=style, lang=lang))

    return figures if len(figures) > 1 else figures[0]


# ── Public entry point ────────────────────────────────────────────────────────

def _resolve_band_indices(
    descriptions: Sequence[Optional[str]],
    iselect: Union[str, Sequence[str], None],
    all_params: bool,
    has_class_band: bool,
) -> tuple[list[int], list[str]]:
    """Resolve 1-based band indices + display names from band descriptions.

    ``all_params=True`` returns every band except a leading class band (if
    ``has_class_band``). ``iselect`` looks each requested name up against
    ``descriptions`` directly — this fixes a prior bug where ``iselect``
    band indices were guessed positionally (1, 2, 3, ...) instead of
    actually matching the requested name to its real band position.
    """
    count = len(descriptions)
    start = 2 if (all_params and has_class_band) else 1

    if all_params:
        indices = list(range(start, count + 1))
        names = [descriptions[i - 1] or f"band_{i}" for i in indices]
        return indices, names

    requested = [iselect] if isinstance(iselect, str) else list(iselect)
    indices = []
    for name in requested:
        try:
            indices.append(descriptions.index(name) + 1)
        except ValueError:
            raise ValueError(
                f"No band with description {name!r}. Available: {list(descriptions)}"
            )
    return indices, requested


def lcz_plot_parameters(
    x: Union[str, Path, rasterio.io.DatasetReader, np.ndarray, dict, xr.Dataset],
    iselect: Union[str, Sequence[str], None] = None,
    all_params: bool = False,
    has_class_band: bool = True,
    chart_type: Literal["map", "scatter", "radar", "correlation"] = "map",
    lcz_map: Optional[Union[str, Path]] = None,
    index_x: Optional[str] = None,
    index_y: Optional[str] = None,
    size_by: Optional[str] = None,
    lcz_classes: Optional[list[int]] = None,
    normalize_radar: bool = True,
    inclusive: bool = False,
    isave: bool = False,
    save_extension: str = "html",
    style: str = "default",
    max_pixels: int = 10_000_000,
    use_datashader: bool = True,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    renderer: Literal["plotly", "datashader", "auto"] = "auto",
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
    lang: str = "en",
) -> Union[go.Figure, list[go.Figure], dict[str, go.Figure]]:
    """Plot LCZ parameter rasters with interactive Plotly visualizations.

    Parameters
    ----------
    x : str, Path, rasterio dataset, numpy array, dict, or xarray.Dataset
        Source data. A file path / rasterio dataset is read band-by-band by
        matching ``iselect`` names against the file's band descriptions (or
        every band via ``all_params``). A ``dict`` is the result of
        ``lcz_get_ucp(..., stations=None)`` — its ``'combined_rasters'``
        Dataset is plotted directly, with every variable treated as a real
        parameter (no band skipped). An ``xarray.Dataset`` (e.g.
        ``result['combined_rasters']``) is accepted the same way. For
        ``chart_type`` other than ``"map"``, ``x`` must be a path to an
        ``lcz_get_indices`` GeoTIFF stack.
    iselect : str or sequence of str or None
        Parameter names to plot — matched against the raster's band
        descriptions (e.g. ``"NDVI"`` for an ``lcz_get_indices`` stack, or
        a morphological parameter name for an ``lcz_get_parameters`` stack).
        For ``chart_type`` other than ``"map"``, this instead restricts
        which indices feed the chart (defaults to every band in ``x``).
    all_params : bool
        Plot every parameter band. Ignored for ``chart_type != "map"``.
    has_class_band : bool
        Whether band 1 is a leading LCZ-class band to skip when
        ``all_params=True`` — True for ``lcz_get_parameters`` output
        (default), False for stacks with no class band (e.g.
        ``lcz_get_indices`` or ``lcz_get_planetary_computer`` output).
        Ignored when ``iselect`` is given (bands are matched by name
        regardless of position) or for array/Dataset input.
    chart_type : {"map", "scatter", "radar", "correlation"}
        "map" (default): the spatial heatmap behavior described above.
        "scatter": per-pixel ``index_x`` vs ``index_y``, colored by LCZ
        class, optionally sized by ``size_by``. "radar": one polar trace
        per LCZ class (or ``lcz_classes``) across the requested indices.
        "correlation": Pearson correlation matrix between the requested
        indices. These three delegate to ``lcz_cal_indices`` (which is
        where their statistics/plotting logic actually lives — this is a
        convenience entry point, not a second implementation) and require
        ``lcz_map``.
    lcz_map : str or Path, optional
        Path to the LCZ class GeoTIFF (classes 1-17) that ``x`` was
        computed on the same grid of. Required for ``chart_type != "map"``.
    index_x, index_y : str, optional
        Required for ``chart_type="scatter"``.
    size_by : str, optional
        Optional third index controlling scatter marker size.
    lcz_classes : list of int, optional
        Restrict the radar chart to these LCZ classes.
    normalize_radar : bool
        Min-max normalize each index's radar axis across the plotted
        classes, so indices with very different natural ranges stay
        visually comparable. Default True.
    inclusive : bool
        Use the colorblind-friendly palette set.
    isave : bool
        Save each rendered figure.
    save_extension : str
        "html" for interactive, "png"/"pdf"/"svg"/"tiff" for static.
    style : str
        Publication style preset: "default", "nature", "science", or
        "generic_bw". Controls font, figure size (mm), DPI, and palette
        used when ``isave`` and ``save_extension`` != "html".
    max_pixels : int
        Down-sample threshold. Ignored for ``chart_type != "map"``.
    use_datashader : bool
        Use Datashader for large rasters. Ignored for ``chart_type != "map"``.
    title, subtitle, caption : str, optional
        Figure annotations.
    renderer : {"plotly", "datashader", "auto"}
        Rendering backend. Ignored for ``chart_type != "map"``.
    add_scalebar, add_north_arrow : bool
        Add a scale bar / north arrow to the rendered map. Only applies to
        ``chart_type="map"`` (skipped silently for geographic CRSs).
    lang : str
        Message language ("en"/"pt"/"es"/"zh"). Only used for
        ``chart_type != "map"``.

    Returns
    -------
    plotly.graph_objects.Figure, list thereof, or (unused here) dict thereof

    Examples
    --------
    >>> from LCZ4py.general.lcz_get_indices import lcz_get_indices
    >>> idx = lcz_get_indices(pc_result, indices=["NDVI", "NDBI"])
    >>> lcz_plot_parameters(idx.path, iselect="NDVI")
    >>> lcz_plot_parameters(idx.path, all_params=True, has_class_band=False)

    >>> lcz_plot_parameters(
    ...     idx.path, lcz_map="lcz_map.tif", chart_type="scatter",
    ...     index_x="NDVI", index_y="NDBI",
    ... )
    """
    if chart_type != "map":
        if lcz_map is None:
            raise ValueError(f"chart_type={chart_type!r} requires lcz_map (the LCZ class GeoTIFF)")
        if not isinstance(x, (str, Path)):
            raise ValueError(
                f"chart_type={chart_type!r} requires x to be a path to an "
                f"lcz_get_indices GeoTIFF stack, got {type(x)}"
            )

        # These three chart kinds are LCZ-class-conditioned statistics, not
        # spatial maps — that logic already lives in lcz_cal_indices, so
        # delegate rather than re-implement it here. Local import: avoids
        # paying the cost (and any future circularity risk) for the common
        # chart_type="map" path, which never touches this module.
        from LCZ4py.general.lcz_cal_indices import lcz_cal_indices

        requested = [iselect] if isinstance(iselect, str) else list(iselect) if iselect else None
        result = lcz_cal_indices(
            lcz_map, x, indices=requested, plot_type=chart_type,
            index_x=index_x, index_y=index_y, size_by=size_by,
            lcz_classes=lcz_classes, normalize_radar=normalize_radar,
            inclusive=inclusive, isave=isave, save_extension=save_extension,
            title=title, subtitle=subtitle, caption=caption, lang=lang,
        )
        return result.fig

    if x is None:
        raise ValueError("x must be a raster path / dataset / array")

    # Accept the dict returned by lcz_get_ucp() directly
    if isinstance(x, dict):
        x = x.get("combined_rasters")
        if x is None:
            raise ValueError(
                "lcz_get_ucp result has no 'combined_rasters' to plot — "
                "check 'failed_variables' in the result"
            )

    if isinstance(x, xr.Dataset):
        return _plot_from_dataset(
            x, iselect=iselect, all_params=all_params, inclusive=inclusive,
            isave=isave, save_extension=save_extension, use_datashader=use_datashader,
            title=title, subtitle=subtitle, caption=caption, renderer=renderer,
            style=style, add_scalebar=add_scalebar, add_north_arrow=add_north_arrow, lang=lang,
        )

    # Determine which bands to render
    if isinstance(x, np.ndarray):
        if all_params:
            raise ValueError("all_params=True requires a multi-band raster")
        band_indices, band_names = [1], [iselect if isinstance(iselect, str) else "band_1"]
    else:
        if not (all_params or iselect is not None):
            raise ValueError("Provide either iselect='Param' or all_params=True")
        with rasterio.open(str(x)) if isinstance(x, (str, Path)) else x as src:
            descriptions = list(src.descriptions)
        band_indices, band_names = _resolve_band_indices(
            descriptions, iselect, all_params, has_class_band,
        )

    figures = []

    for band_idx, name in zip(band_indices, band_names):
        # Read data with optimal strategy
        if isinstance(x, np.ndarray):
            arr = x
            profile = {"transform": None, "crs": None, "height": arr.shape[0], "width": arr.shape[1]}
        else:
            with FastRasterReader(x, max_pixels=max_pixels) as reader:
                arr, profile = reader.read_band(band=band_idx)

        # Determine rendering strategy
        use_ds = (renderer == "datashader") or (
            renderer == "auto" and use_datashader and
            arr.shape[0] * arr.shape[1] > 4_000_000
        )

        # Create interactive figure
        fig = _plot_one_interactive(
            arr, profile, name, inclusive, use_datashader=use_ds
        )

        transform = profile.get("transform")
        if transform is not None:
            h, w = arr.shape
            bounds = (transform.c, transform.f + h * transform.e, transform.c + w * transform.a, transform.f)
            add_map_furniture(fig, bounds=bounds, crs=profile.get("crs"),
                               add_scalebar=add_scalebar, add_north_arrow=add_north_arrow)

        figures.append(_finalize_figure(fig, name, title, subtitle, caption, isave, save_extension,
                                         style=style, lang=lang))

    return figures if len(figures) > 1 else figures[0]


__all__ = ["lcz_plot_parameters", "FastRasterReader", "read_with_warped_vrt"]