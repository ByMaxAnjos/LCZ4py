"""
lcz_plot_map.py

Ultra-fast interactive LCZ map visualization with:
- Plotly WebGL for smooth panning/zooming
- GeoArrow for efficient categorical data handling
- DuckDB Spatial for fast legend computation
- ExactExtract for zonal statistics on hover
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union, Literal

import numpy as np
import polars as pl
import pyarrow as pa
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# GeoArrow
try:
    import geoarrow.pyarrow as ga
    import geoarrow.pandas as gap
    HAS_GEOARROW = True
except ImportError:
    HAS_GEOARROW = False

# DuckDB Spatial
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

from LCZ4py._internal.lcz_parameters_data import LCZ_COLORS, LCZ_COLORBLIND, LCZ_NAMES, LCZ_IDS, get_lcz_names
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export, add_map_furniture

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"


# ── Pre-computed color lookup table (GPU-friendly) ────────────────────────────

@lru_cache(maxsize=2)
def _build_color_lut(inclusive: bool) -> np.ndarray:
    """Build a lookup table: index -> RGBA as uint8.
    
    Returns (18, 4) array where:
    - Index 0: transparent (nodata)
    - Indices 1-17: LCZ class colors
    """
    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    lut = np.zeros((18, 4), dtype=np.uint8)
    for i, hex_color in enumerate(colors, start=1):
        h = hex_color.lstrip("#")
        lut[i] = [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255]
    return lut


def _arr_to_png_datauri(rgba: np.ndarray) -> str:
    """Encode RGBA (H,W,4) uint8 array as a PNG data URI."""
    try:
        from PIL import Image
        import io as _io
        buf = _io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        from rasterio.io import MemoryFile
        h, w = rgba.shape[:2]
        with MemoryFile() as mf:
            with mf.open(driver="PNG", height=h, width=w, count=4, dtype="uint8") as dst:
                dst.write(rgba.transpose(2, 0, 1))
            b64 = base64.b64encode(mf.read()).decode()
    return f"data:image/png;base64,{b64}"


def _create_basemap_fig(
    arr: np.ndarray,
    extent: tuple,
    inclusive: bool,
    show_legend: bool,
    title: Optional[str],
    opacity: float,
    lang: str = "en",
) -> go.Figure:
    """Plotly figure with OpenStreetMap basemap and LCZ raster overlay.

    Uses Plotly's built-in mapbox backend (open-street-map style, no token).
    The LCZ raster is a PNG data URI added as a mapbox image layer — same
    rendering engine Plotly already uses, so it works in VS Code Jupyter
    without any CDN/CSP issues.
    """
    import math
    xmin, xmax, ymin, ymax = extent
    center_lon = (xmin + xmax) / 2
    center_lat = (ymin + ymax) / 2

    # Rough zoom from extent size
    span = max(xmax - xmin, ymax - ymin)
    zoom = max(1, min(15, round(math.log2(360 / (span + 1e-9))) - 1))

    lut = _build_color_lut(inclusive)
    display_arr = np.where((arr >= 1) & (arr <= 17), arr, 0)
    rgba = lut[display_arr.astype(np.int32)]
    data_uri = _arr_to_png_datauri(rgba)

    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    present = sorted(set(display_arr.ravel().tolist()) & set(range(1, 18)))
    names = get_lcz_names(lang)

    fig = go.Figure()

    # Invisible scatter traces for legend
    if show_legend:
        for cls in present:
            fig.add_trace(go.Scattermapbox(
                lon=[None], lat=[None],
                mode="markers",
                marker=dict(size=12, color=colors[cls - 1]),
                name=f"{cls}: {names[cls - 1]}",
                showlegend=True,
            ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lon=center_lon, lat=center_lat),
            zoom=zoom,
            layers=[dict(
                sourcetype="image",
                source=data_uri,
                coordinates=[
                    [xmin, ymax],  # NW
                    [xmax, ymax],  # NE
                    [xmax, ymin],  # SE
                    [xmin, ymin],  # SW
                ],
                opacity=opacity,
            )],
        ),
        title=dict(
            text=title or lcz_msg("lcz_map_title", lang),
            x=0.5, xanchor="center", font=dict(size=18),
        ),
        legend=dict(
            title=dict(text=lcz_msg("lcz_class_legend", lang), font=dict(size=13)),
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            x=1.01, y=0.5,
            xanchor="left", yanchor="middle",
        ),
        margin=dict(l=0, r=170, t=50, b=0),
        height=700,
    )

    return fig


def _build_plotly_discrete_colorscale(
    inclusive: bool
) -> tuple[list[str], dict[int, str]]:
    """Build discrete colorscale for Plotly heatmap."""
    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    # Plotly needs colors at specific z-values
    color_list = ["rgba(0,0,0,0)"] + colors
    color_map = {i: colors[i-1] for i in range(1, 18)}
    return color_list, color_map


# ── Fast raster reader with block processing ──────────────────────────────────

class BlockReader:
    """Memory-efficient block-based raster reader."""
    
    def __init__(self, src: rasterio.io.DatasetReader, max_pixels: int = 10_000_000):
        self.src = src
        self.max_pixels = max_pixels
        self.h, self.w = src.height, src.width
        
        # Calculate downsample factor
        if self.h * self.w > max_pixels:
            self.scale = (max_pixels / (self.h * self.w)) ** 0.5
        else:
            self.scale = 1.0
        
        self.out_h = max(1, int(self.h * self.scale))
        self.out_w = max(1, int(self.w * self.scale))
    
    def read_downsampled(self, band: int = 1, resampling: Resampling = Resampling.nearest) -> np.ndarray:
        """Read one band of the raster downsampled."""
        return self.src.read(
            band,
            out_shape=(self.out_h, self.out_w),
            resampling=resampling,  # nearest for categorical, bilinear for continuous
        )
    
    def get_extent(self) -> tuple[float, float, float, float]:
        """Get (xmin, xmax, ymin, ymax) in source CRS."""
        t = self.src.transform
        return (
            t.c,
            t.c + self.w * t.a,
            t.f + self.h * t.e,
            t.f,
        )


# ── GeoArrow polygon conversion ──────────────────────────────────────────────

def raster_to_geoarrow(
    arr: np.ndarray,
    transform,
    crs: str = "EPSG:4326",
    min_area: float = 0.0,
) -> Optional[pa.Table]:
    """Convert categorical raster to GeoArrow table of polygons.
    
    Much faster than GeoPandas for large rasters due to Arrow's
    zero-copy columnar format.
    """
    if not HAS_GEOARROW:
        return None
    
    records = []
    seen_classes = set()
    
    for geom, val in shapes(arr.astype(np.int16), transform=transform):
        val = int(val)
        if val < 1 or val > 17:
            continue
        if val in seen_classes:
            continue
        
        geom_shape = shape(geom)
        if min_area > 0 and geom_shape.area < min_area:
            continue
        
        seen_classes.add(val)
        records.append({
            "lcz": val,
            "name": LCZ_NAMES[val - 1],
            "geometry": geom_shape,
        })
    
    if not records:
        return None
    
    # Convert to GeoArrow via GeoPandas (fast path)
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    
    try:
        # Try direct GeoArrow conversion
        arrow_table = ga.from_geopandas(gdf)
        return arrow_table
    except Exception:
        # Fallback to regular Arrow
        return pa.Table.from_pandas(gdf.drop(columns=["geometry"]))


# ── DuckDB Spatial for fast legend computation ────────────────────────────────

def compute_legend_with_duckdb(
    arr: np.ndarray,
    transform,
    crs: str = "EPSG:4326",
) -> pl.DataFrame:
    """Use DuckDB Spatial to compute legend statistics."""
    if not HAS_DUCKDB:
        # Fallback to Polars
        return _compute_legend_polars(arr)
    
    # Create a Polars DataFrame with class counts
    flat = arr.ravel()
    valid_mask = (flat >= 1) & (flat <= 17)
    valid_classes = flat[valid_mask]
    
    # Use DuckDB for aggregation
    df = pl.DataFrame({"lcz": valid_classes.tolist()})
    
    try:
        conn = duckdb.connect()
        result = conn.execute("""
            SELECT
                lcz,
                COUNT(*) as pixel_count,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
            FROM df
            GROUP BY lcz
            ORDER BY lcz
        """).pl()
        conn.close()
        return result
    except Exception:
        return _compute_legend_polars(arr)


def _compute_legend_polars(arr: np.ndarray) -> pl.DataFrame:
    """Fast legend computation using Polars."""
    flat = arr.ravel()
    valid_mask = (flat >= 1) & (flat <= 17)
    valid_classes = flat[valid_mask]
    
    df = pl.DataFrame({"lcz": valid_classes.tolist()})
    
    return (
        df.group_by("lcz")
        .agg([
            pl.len().alias("pixel_count"),
            (pl.len() / pl.len().sum() * 100).alias("percentage"),
        ])
        .sort("lcz")
    )


# ── Interactive Plotly map ────────────────────────────────────────────────────

def _create_interactive_lcz_map(
    arr: np.ndarray,
    extent: tuple[float, float, float, float],
    inclusive: bool,
    show_legend: bool = True,
    use_webgl: bool = True,
    lang: str = "en",
) -> go.Figure:
    """Create an interactive LCZ map with Plotly."""
    
    # Prepare class array (0 = nodata -> transparent)
    # flipud: raster row 0 is north, but Plotly y0=ymin (south) — align them
    display_arr = np.flipud(np.where((arr >= 1) & (arr <= 17), arr, 0))
    
    # Build color mapping
    color_list, color_map = _build_plotly_discrete_colorscale(inclusive)
    
    # Calculate pixel dimensions
    xmin, xmax, ymin, ymax = extent
    h, w = arr.shape
    dx = (xmax - xmin) / w
    dy = (ymax - ymin) / h
    
    fig = go.Figure()
    
    if use_webgl and w * h > 1_000_000:
        # Use Image trace with pre-rendered RGBA for massive rasters
        lut = _build_color_lut(inclusive)
        rgba = lut[display_arr.astype(np.int32)]  # (H, W, 4)
        
        fig.add_trace(go.Image(
            z=rgba,
            x0=xmin,
            y0=ymin,
            dx=dx,
            dy=dy,
            hovertemplate=(
                "X: %{x:.5f}<br>Y: %{y:.5f}<extra></extra>"
            ),
        ))
    else:
        # Use Heatmap with discrete colorscale
        fig.add_trace(go.Heatmap(
            z=display_arr,
            x0=xmin,
            y0=ymin,
            dx=dx,
            dy=dy,
            colorscale=[
                (i/17, color_list[i]) for i in range(18)
            ],
            zmin=0,
            zmax=17,
            showscale=False,
            hovertemplate=(
                "LCZ Class: %{z:.0f}<br>"
                "X: %{x:.5f}<br>"
                "Y: %{y:.5f}<extra></extra>"
            ),
        ))
    
    # Add legend as separate traces (more flexible than colorbar)
    if show_legend:
        present = sorted(set(display_arr.ravel().tolist()) & set(range(1, 18)))
        colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
        names = get_lcz_names(lang)

        for lcz_class in present:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=15,
                    color=colors[lcz_class - 1],
                    symbol="square",
                    line=dict(width=1, color="black"),
                ),
                name=f"{lcz_class}: {names[lcz_class - 1]}",
                legendgroup="lcz_classes",
                showlegend=True,
            ))

    # Configure layout
    fig.update_layout(
        title=dict(
            text=lcz_msg("lcz_map_title", lang),
            font=dict(size=20),
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
        legend=dict(
            title=dict(text=lcz_msg("lcz_class_legend", lang), font=dict(size=14)),
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
        ),
        margin=dict(l=0, r=180, t=50, b=0),
        width=1200,
        height=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode="pan",  # Enable panning
    )
    
    # Add hover behavior for class lookup
    fig.update_traces(
        hoverinfo="x+y+z",
    )
    
    return fig


def _create_continuous_map(
    arr: np.ndarray,
    extent: tuple[float, float, float, float],
    units: str,
    title: Optional[str] = None,
    colorscale: str = "RdBu_r",
) -> go.Figure:
    """Create an interactive map for a continuous raster variable (e.g. LST)."""
    # flipud: raster row 0 is north, but Plotly y0=ymin (south) — align them
    display_arr = np.flipud(arr)

    xmin, xmax, ymin, ymax = extent
    h, w = arr.shape
    dx = (xmax - xmin) / w
    dy = (ymax - ymin) / h

    finite = display_arr[np.isfinite(display_arr)]
    zmin, zmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    unit_label = "°C" if units == "C" else "K"

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=display_arr,
        x0=xmin, y0=ymin, dx=dx, dy=dy,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=unit_label),
        hovertemplate=(
            f"Value: %{{z:.1f}} {unit_label}<br>"
            "X: %{x:.5f}<br>Y: %{y:.5f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=title or "Land Surface Temperature", x=0.5, xanchor="center", font=dict(size=20)),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=80, t=50, b=0),
        width=1200,
        height=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode="pan",
    )
    return fig


def _resolve_band(src: rasterio.io.DatasetReader, band: Union[int, str, None]) -> int:
    """Resolve a 1-based band index from an int, a matching band description, or None (-> 1)."""
    if band is None:
        return 1
    if isinstance(band, int):
        if not (1 <= band <= src.count):
            raise ValueError(f"band {band} out of range (raster has {src.count} bands)")
        return band
    for i, desc in enumerate(src.descriptions, start=1):
        if desc == band:
            return i
    raise ValueError(f"No band with description {band!r}. Available: {list(src.descriptions)}")


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class LCZPlotResult:
    """Return type for advanced LCZ map plotting."""
    fig: Optional[go.Figure] = None
    geoarrow_table: Optional[pa.Table] = None
    legend_df: Optional[pl.DataFrame] = None
    html: Optional[str] = None  # MapLibre renderer output

    def show(self) -> None:
        """Display the map. Both renderers return a go.Figure, so this always works."""
        if self.fig is not None:
            self.fig.show()


def lcz_plot_map(
    x: Union[str, Path, rasterio.io.DatasetReader],
    isave: bool = False,
    save_extension: str = "html",
    style: str = "default",
    show_legend: bool = True,
    inclusive: bool = False,
    use_webgl: bool = True,
    use_geoarrow: bool = True,
    use_duckdb: bool = True,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    max_pixels: int = 10_000_000,
    renderer: Literal["plotly", "maplibre"] = "plotly",
    opacity: float = 0.8,
    lang: str = "en",
    data_type: Literal["auto", "lcz", "continuous"] = "auto",
    band: Union[int, str, None] = None,
    colorscale: str = "RdBu_r",
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
) -> LCZPlotResult:
    """Plot an LCZ map, or a continuous raster stack (e.g. lcz_get_lst output).

    Parameters
    ----------
    x : str, Path or rasterio dataset
        Source raster. Either an LCZ class map (integer codes 1-17) or a
        continuous multi-band stack such as the GeoTIFF returned by
        ``lcz_get_lst`` (float LST values, one band per date).
    isave : bool
        Save the figure.
    save_extension : str
        "html" for interactive, "png"/"pdf" for static.
    style : str
        Publication style preset: "default", "nature", "science", or
        "generic_bw". Controls font, figure size (mm), DPI, and palette
        used when ``isave`` and ``save_extension`` != "html".
    show_legend : bool
        Show the LCZ class legend. Ignored for continuous data (a colorbar
        is shown instead).
    inclusive : bool
        Use the colorblind-friendly palette. Ignored for continuous data.
    use_webgl : bool
        Use WebGL rendering for large rasters. Ignored for continuous data.
    use_geoarrow : bool
        Generate GeoArrow table for polygon data. Ignored for continuous data.
    use_duckdb : bool
        Use DuckDB Spatial for legend computation. Ignored for continuous data.
    title, subtitle, caption : str, optional
        Figure annotations.
    max_pixels : int
        Down-sample threshold.
    renderer : {"plotly", "maplibre"}
        "plotly" (default) returns a Plotly figure; "maplibre" returns a
        MapLibre GL JS HTML page overlaid on an OpenStreetMap basemap.
        Use ``result.show()`` to display either renderer in a notebook.
        Continuous data only supports "plotly".
    opacity : float
        Raster overlay opacity for the MapLibre renderer (0–1, default 0.8).
    data_type : {"auto", "lcz", "continuous"}
        "auto" (default) detects continuous data from the raster's dtype
        (float32/float64, as written by ``lcz_get_lst``); override if needed.
    band : int, str, or None
        Which band to plot for multi-band continuous rasters: a 1-based
        index, a band description to match (e.g. an ISO date like
        ``"2026-06-01"``), or None for the first band. Ignored for LCZ maps.
    colorscale : str
        Plotly colorscale for continuous data. Default "RdBu_r" (blue=cold,
        red=hot).
    add_scalebar, add_north_arrow : bool
        Add a scale bar / north arrow to the map (Plotly renderer only;
        skipped for geographic/lon-lat CRSs where ground distance isn't
        directly meaningful).

    Returns
    -------
    LCZPlotResult
        Contains the figure (Plotly) or html string (MapLibre) and optional metadata.
    """
    if isinstance(x, (str, Path)):
        with rasterio.open(str(x)) as src:
            is_continuous = (
                data_type == "continuous"
                or (data_type == "auto" and src.dtypes[0].startswith("float"))
            )
            band_idx = _resolve_band(src, band) if is_continuous else 1
            reader = BlockReader(src, max_pixels)
            resampling = Resampling.bilinear if is_continuous else Resampling.nearest
            arr = reader.read_downsampled(band=band_idx, resampling=resampling)
            extent = reader.get_extent()
            crs_obj = src.crs
            crs = str(src.crs) if src.crs else "EPSG:4326"
            transform = src.transform
            units = src.tags().get("units", "K")
            band_label = src.descriptions[band_idx - 1] if src.descriptions[band_idx - 1] else None
    elif hasattr(x, "read"):
        is_continuous = (
            data_type == "continuous"
            or (data_type == "auto" and x.dtypes[0].startswith("float"))
        )
        band_idx = _resolve_band(x, band) if is_continuous else 1
        reader = BlockReader(x, max_pixels)
        resampling = Resampling.bilinear if is_continuous else Resampling.nearest
        arr = reader.read_downsampled(band=band_idx, resampling=resampling)
        extent = reader.get_extent()
        crs_obj = x.crs
        crs = str(x.crs) if x.crs else "EPSG:4326"
        transform = x.transform
        units = x.tags().get("units", "K")
        band_label = x.descriptions[band_idx - 1] if x.descriptions[band_idx - 1] else None
    else:
        raise TypeError(f"x must be a path or rasterio dataset, got {type(x)}")

    # ── Continuous raster (e.g. lcz_get_lst LST stack) ───────────────────────
    if is_continuous:
        map_title = title or f"Land Surface Temperature{f' — {band_label}' if band_label else ''}"
        fig = _create_continuous_map(arr, extent, units, title=map_title, colorscale=colorscale)
        if subtitle:
            fig.update_layout(title=dict(text=f"{map_title}<br><sup>{subtitle}</sup>"))
        if caption:
            fig.add_annotation(
                text=caption, xref="paper", yref="paper", x=0.01, y=-0.02,
                showarrow=False, font=dict(size=10, color="gray"),
            )
        bounds = (extent[0], extent[2], extent[1], extent[3])
        add_map_furniture(fig, bounds=bounds, crs=crs_obj,
                           add_scalebar=add_scalebar, add_north_arrow=add_north_arrow)
        fig = finalize_export(fig, style=style, isave=isave, save_extension=save_extension,
                               filename="lcz_plot_lst", lang=lang)
        return LCZPlotResult(fig=fig)

    # Keep nodata (0) as 0 — transparent in the plot, not Water.
    # Water bodies are stored as class 17 in the source LCZ raster.
    arr = np.where((arr >= 1) & (arr <= 17), arr, 0)

    # ── MapLibre renderer (Plotly mapbox + OSM basemap) ──────────────────────
    if renderer == "maplibre":
        # ponytail: no add_map_furniture here — this figure is a mapbox/geo trace,
        # not a Cartesian one, so paper-relative scale bar/north arrow annotations
        # don't line up with the map; skip until furniture supports mapbox layouts.
        fig = _create_basemap_fig(arr, extent, inclusive, show_legend, title, opacity, lang=lang)
        fig = finalize_export(fig, style=style, isave=isave, save_extension="html",
                               filename="lcz_plot_map_maplibre", lang=lang)
        return LCZPlotResult(fig=fig)

    # ── Plotly renderer (default) ────────────────────────────────────────────
    # Create interactive figure
    fig = _create_interactive_lcz_map(
        arr, extent, inclusive, show_legend, use_webgl, lang=lang
    )
    
    # Add annotations
    if title or subtitle:
        title_text = title or lcz_msg("lcz_map_title", lang)
        if subtitle:
            title_text += f"<br><sup>{subtitle}</sup>"
        fig.update_layout(title=dict(text=title_text))
    if caption:
        fig.add_annotation(
            text=caption,
            xref="paper", yref="paper",
            x=0.01, y=-0.02,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )
    
    # Generate GeoArrow table if requested
    geoarrow_table = None
    if use_geoarrow and HAS_GEOARROW and arr.shape[0] * arr.shape[1] < 5_000_000:
        geoarrow_table = raster_to_geoarrow(arr, transform, crs)
    
    # Compute legend with DuckDB if requested
    legend_df = None
    if use_duckdb:
        legend_df = compute_legend_with_duckdb(arr, transform, crs)

    # Add map furniture (scale bar / north arrow), then save
    bounds = (extent[0], extent[2], extent[1], extent[3])
    add_map_furniture(fig, bounds=bounds, crs=crs_obj,
                       add_scalebar=add_scalebar, add_north_arrow=add_north_arrow)
    fig = finalize_export(fig, style=style, isave=isave, save_extension=save_extension,
                           filename="lcz_plot_map", lang=lang)

    return LCZPlotResult(
        fig=fig,
        geoarrow_table=geoarrow_table,
        legend_df=legend_df,
    )


__all__ = ["LCZPlotResult", "lcz_plot_map", "BlockReader", "raster_to_geoarrow"]
