"""
lcz_plot_map.py

Publication-quality LCZ map visualization.

- Native static matplotlib/Agg renderer (default) — reproducible PNG/PDF/SVG/TIFF
  without Plotly, Kaleido, a browser, or a Chromium install
- Opt-in Plotly WebGL renderer for smooth panning/zooming on large rasters
- Opt-in MapLibre/OSM basemap renderer
- LCZ legend ordered by class, labelled from ``lcz_parameters_data``
- North arrow + scale bar for projected CRSs
- 300 dpi static export with the journal style presets

Optional dependencies (duckdb, geoarrow) are guarded with ``HAS_*`` flags and
fall back to the core Polars/NumPy implementation.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
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

# GeoArrow (optional)
try:  # pragma: no cover - depends on the environment
    import geoarrow.pyarrow as ga
    import geoarrow.pandas as gap  # noqa: F401 - registers the pandas extension

    HAS_GEOARROW = True
except ImportError:  # pragma: no cover
    ga = None  # type: ignore[assignment]
    HAS_GEOARROW = False

# DuckDB Spatial (optional)
try:  # pragma: no cover - depends on the environment
    import duckdb

    HAS_DUCKDB = True
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]
    HAS_DUCKDB = False

from LCZ4py._internal.lcz_parameters_data import (
    LCZ_COLORS,
    LCZ_COLORBLIND,
    LCZ_NAMES,
    get_lcz_names,
)
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import (
    add_map_furniture,
    finalize_export,
    get_style,
    save_plotly_figure,
)
from LCZ4py._internal import lcz_static
from LCZ4py._internal.lcz_static import resolve_renderer

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"

# Default cartographic credit line. Overridable through ``attribution=``.
DEFAULT_ATTRIBUTION = "LCZ4py | Stewart & Oke (2012) LCZ scheme"
BASEMAP_ATTRIBUTION = "OpenStreetMap contributors | OpenFreeMap | © OpenMapTiles"

# Effective minimum resolution for static exports. The "default" style preset
# targets 150 dpi for screen use; publication exports are rasterised at 300 dpi
# (Plotly's ``scale`` multiplies the style's native dpi).
MIN_STATIC_DPI = 300


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


def _present_classes(arr: np.ndarray) -> list[int]:
    """Sorted LCZ classes present in ``arr`` (1-17 only, ordered by class)."""
    valid = arr[(arr >= 1) & (arr <= 17)]
    if valid.size == 0:
        return []
    return [int(v) for v in np.unique(valid)]


def _arr_to_png_datauri(rgba: np.ndarray) -> str:
    """Encode RGBA (H,W,4) uint8 array as a PNG data URI."""
    try:
        from PIL import Image
        import io as _io

        buf = _io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
    except ImportError:  # pragma: no cover - Pillow ships with matplotlib
        from rasterio.io import MemoryFile

        h, w = rgba.shape[:2]
        with MemoryFile() as mf:
            with mf.open(driver="PNG", height=h, width=w, count=4, dtype="uint8") as dst:
                dst.write(rgba.transpose(2, 0, 1))
            b64 = base64.b64encode(mf.read()).decode()
    return f"data:image/png;base64,{b64}"


# ── Publication export helper ────────────────────────────────────────────────

def _finalize_publication(
    fig: go.Figure,
    *,
    style: str,
    isave: bool,
    save_extension: str,
    filename: str,
    dpi: Optional[int] = None,
    lang: str = "en",
) -> go.Figure:
    """Save ``fig`` through ``lcz_theme``, guaranteeing >=300 dpi static output.

    ``finalize_export`` rasterises at the style preset's own dpi (150 for
    "default"), which is too low for print. Here we compute the pixel size at
    the requested resolution and let Plotly's ``scale`` factor reach it, while
    reusing the preset's mm-based figure geometry.
    """
    if not (isave and save_extension != "html"):
        return finalize_export(
            fig, style=style, isave=isave, save_extension=save_extension,
            filename=filename, lang=lang,
        )

    spec = get_style(style, lang)
    target_dpi = int(dpi) if dpi else max(spec.dpi, MIN_STATIC_DPI)
    if target_dpi < 1:
        raise ValueError("dpi must be a positive integer")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{filename}.{save_extension}")
    width_mm = spec.width_mm.get("single", 120)
    # Preserve the figure's own aspect ratio (maps are 4:3, pies are almost
    # square, facets are wider than tall). Squashing every chart into a fixed
    # 3:4 box made multi-panel figures unreadable.
    try:
        fig_w = float(fig.layout.width or 0)
        fig_h = float(fig.layout.height or 0)
    except (TypeError, ValueError):  # pragma: no cover - layout is always numeric
        fig_w = fig_h = 0.0
    aspect = (fig_h / fig_w) if (fig_w > 0 and fig_h > 0) else 0.75
    height_mm = max(30.0, min(spec.max_height_mm, width_mm * aspect))
    width_px = max(1, round(width_mm / 25.4 * target_dpi))
    height_px = max(1, round(height_mm / 25.4 * target_dpi))
    scale = max(1, round(target_dpi / spec.dpi))

    fig.update_layout(template=spec.template)

    if save_extension == "tiff":
        png_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
        save_plotly_figure(
            fig, png_path, width_px=width_px, height_px=height_px,
            scale=scale, save_extension="png", lang=lang,
        )
        from PIL import Image

        with Image.open(png_path) as img:
            img.save(save_path, dpi=(target_dpi, target_dpi))
        os.remove(png_path)
    else:
        save_plotly_figure(
            fig, save_path, width_px=width_px, height_px=height_px,
            scale=scale, save_extension=save_extension, lang=lang,
        )
        if save_extension == "png":
            # Plotly/Kaleido writes no DPI tag; journals check it, so tag the
            # PNG with the effective resolution (lossless re-encode).
            from PIL import Image

            with Image.open(save_path) as img:
                img.save(save_path, pnginfo=None, dpi=(target_dpi, target_dpi))

    logger.info("Saved: %s", os.path.abspath(save_path))
    return fig


def _ensure_bottom_margin(fig: go.Figure, needed_px: float) -> None:
    """Grow the bottom margin so panels can sit *below* the map frame.

    Annotations placed at negative paper-y are clipped when the margin is too
    small — captions and credit lines used to disappear from static exports.
    """
    try:
        current = getattr(fig.layout.margin, "b", None) or 0
    except Exception:  # pragma: no cover - margin always exists
        current = 0
    if current < needed_px:
        fig.update_layout(margin=dict(b=needed_px))


def _bottom_text_y(fig: go.Figure, margin_px: float) -> float:
    """Paper-y for a text line that sits just above the figure's bottom edge."""
    try:
        height = float(fig.layout.height or 0) or 700.0
    except (TypeError, ValueError):  # pragma: no cover
        height = 700.0
    return -max(0.0, (margin_px - 28.0)) / height


def _add_attribution(fig: go.Figure, text: Optional[str]) -> None:
    """Credit line placed below the frame, right-aligned (print-safe)."""
    if not text:
        return
    _ensure_bottom_margin(fig, 58)
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=1.0, y=_bottom_text_y(fig, float(fig.layout.margin.b or 58)),
        xanchor="right", yanchor="top",
        showarrow=False,
        font=dict(size=8, color="rgba(60,60,60,0.95)"),
    )


def _add_caption(fig: go.Figure, caption: Optional[str]) -> None:
    """Figure caption below the panel, left-aligned, in paper coordinates.

    Cartesian charts (bar/pie) have tick labels and an axis title inside the
    bottom margin, so they need a deeper margin than a map panel; the caption
    is anchored to the figure's bottom edge either way.
    """
    if not caption:
        return
    try:
        is_chart = fig.layout.xaxis.showticklabels is not False
    except Exception:  # pragma: no cover - layout always exists
        is_chart = False
    needed = 96 if is_chart else 58
    _ensure_bottom_margin(fig, needed)
    fig.add_annotation(
        text=caption,
        xref="paper", yref="paper",
        x=0.0, y=_bottom_text_y(fig, float(fig.layout.margin.b or needed)),
        xanchor="left", yanchor="top",
        showarrow=False,
        align="left",
        font=dict(size=8, color="rgba(60,60,60,0.95)"),
    )


# ── Basemap renderer (MapLibre / plotly mapbox backend) ─────────────────────

def _create_basemap_fig(
    arr: np.ndarray,
    extent: tuple,
    inclusive: bool,
    show_legend: bool,
    title: Optional[str],
    opacity: float,
    lang: str = "en",
    attribution: Optional[str] = None,
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
    if span <= 0:
        raise ValueError("Extent has zero width/height; cannot compute a map zoom.")
    zoom = max(1, min(15, round(math.log2(360 / (span + 1e-9))) - 1))

    lut = _build_color_lut(inclusive)
    display_arr = np.where((arr >= 1) & (arr <= 17), arr, 0)
    rgba = lut[display_arr.astype(np.int32)]
    data_uri = _arr_to_png_datauri(rgba)

    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    present = _present_classes(arr)
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

    # Attribution: basemap credit + data credit (required for printed output).
    credit = " • ".join(
        t for t in (BASEMAP_ATTRIBUTION, attribution) if t
    )
    fig.add_annotation(
        text=credit,
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=10, color="rgba(40,40,40,0.9)"),
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="rgba(120,120,120,0.35)",
        borderpad=3,
    )

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
            itemsizing="constant",
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
    inclusive: bool,
) -> tuple[list[str], dict[int, str]]:
    """Build discrete colorscale for Plotly heatmap."""
    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    # Plotly needs colors at specific z-values
    color_list = ["rgba(0,0,0,0)"] + colors
    color_map = {i: colors[i - 1] for i in range(1, 18)}
    return color_list, color_map


# ── Fast raster reader with block processing ──────────────────────────────────

class BlockReader:
    """Memory-efficient block-based raster reader."""

    def __init__(self, src: rasterio.io.DatasetReader, max_pixels: int = 10_000_000):
        if int(max_pixels) < 1:
            raise ValueError("max_pixels must be a positive integer")
        self.src = src
        self.max_pixels = int(max_pixels)
        self.h, self.w = src.height, src.width
        if self.h < 1 or self.w < 1:
            raise ValueError("Raster has zero width/height.")

        # Calculate downsample factor
        if self.h * self.w > self.max_pixels:
            self.scale = (self.max_pixels / (self.h * self.w)) ** 0.5
        else:
            self.scale = 1.0

        self.out_h = max(1, int(self.h * self.scale))
        self.out_w = max(1, int(self.w * self.scale))

    def read_downsampled(
        self,
        band: int = 1,
        resampling: Resampling = Resampling.nearest,
    ) -> np.ndarray:
        """Read one band of the raster downsampled."""
        if band < 1 or band > self.src.count:
            raise ValueError(
                f"band {band} out of range (raster has {self.src.count} band(s))"
            )
        return self.src.read(
            band,
            out_shape=(self.out_h, self.out_w),
            resampling=resampling,  # nearest for categorical, bilinear for continuous
        )

    def get_extent(self) -> tuple[float, float, float, float]:
        """Get (xmin, xmax, ymin, ymax) in source CRS."""
        t = self.src.transform
        if t is None:
            raise ValueError("Raster has no geotransform; cannot place it on a map.")
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

    Returns ``None`` when GeoArrow is unavailable or the raster holds no LCZ
    classes (nothing to vectorise). One representative polygon per class is
    kept, so the table stays a legend/metadata artefact rather than a full
    vectorisation (mirrors LCZ4r's behaviour).
    """
    if not HAS_GEOARROW:
        return None
    if arr is None or np.asarray(arr).size == 0:
        return None
    if transform is None:
        return None

    records = []
    seen_classes: set[int] = set()

    try:
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
    except Exception as exc:  # noqa: BLE001 - vectorisation is best-effort
        logger.warning("Raster-to-GeoArrow conversion failed (%s); skipping.", exc)
        return None

    if not records:
        return None

    # Convert to GeoArrow via GeoPandas (fast path)
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    try:
        # Try direct GeoArrow conversion
        return ga.from_geopandas(gdf)
    except Exception as exc:  # noqa: BLE001 - fall back to plain Arrow
        logger.debug("GeoArrow conversion failed (%s); returning plain Arrow.", exc)
        return pa.Table.from_pandas(gdf.drop(columns=["geometry"]))


# ── Legend statistics ────────────────────────────────────────────────────────

def _legend_from_counts(
    values: np.ndarray,
    counts: np.ndarray,
) -> pl.DataFrame:
    """Build the legend frame from unique classes and their pixel counts."""
    total = int(counts.sum())
    if total == 0:
        return pl.DataFrame(
            schema={
                "lcz": pl.Int64,
                "pixel_count": pl.Int64,
                "percentage": pl.Float64,
            }
        )
    return pl.DataFrame({
        "lcz": values.astype(np.int64),
        "pixel_count": counts.astype(np.int64),
        "percentage": (counts / total * 100.0).round(4),
    }).sort("lcz")


def compute_legend_with_duckdb(
    arr: np.ndarray,
    transform=None,
    crs: str = "EPSG:4326",
) -> pl.DataFrame:
    """Compute legend statistics, using DuckDB when available.

    Pixel counts come from ``np.unique`` (one pass, no per-pixel Python list),
    so the DuckDB path stays cheap for 10 M+ pixel rasters.
    """
    if arr is None:
        return _legend_from_counts(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    flat = np.asarray(arr).ravel()
    valid = flat[(flat >= 1) & (flat <= 17)]
    if valid.size == 0:
        return _legend_from_counts(np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    values, counts = np.unique(valid.astype(np.int64), return_counts=True)

    if HAS_DUCKDB:
        try:
            df = pl.DataFrame({
                "lcz": values.tolist(),
                "pixel_count": counts.tolist(),
            })
            conn = duckdb.connect()
            try:
                return conn.execute("""
                    SELECT
                        lcz,
                        pixel_count,
                        ROUND(pixel_count * 100.0 / SUM(pixel_count) OVER (), 4) AS percentage
                    FROM df
                    ORDER BY lcz
                """).pl()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - fall back to numpy/polars
            logger.debug("DuckDB legend aggregation failed (%s); using Polars.", exc)

    return _legend_from_counts(values, counts)


def _compute_legend_polars(arr: np.ndarray) -> pl.DataFrame:
    """Fast legend computation using NumPy + Polars (no DuckDB)."""
    if arr is None:
        return _legend_from_counts(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    flat = np.asarray(arr).ravel()
    valid = flat[(flat >= 1) & (flat <= 17)]
    if valid.size == 0:
        return _legend_from_counts(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    values, counts = np.unique(valid.astype(np.int64), return_counts=True)
    return _legend_from_counts(values, counts)


# ── Interactive Plotly map ────────────────────────────────────────────────────

def _create_interactive_lcz_map(
    arr: np.ndarray,
    extent: tuple[float, float, float, float],
    inclusive: bool,
    show_legend: bool = True,
    use_webgl: bool = True,
    lang: str = "en",
    attribution: Optional[str] = None,
) -> go.Figure:
    """Create an interactive/publication LCZ map with Plotly."""

    # Prepare class array (0 = nodata -> transparent)
    # flipud: raster row 0 is north, but Plotly y0=ymin (south) — align them
    display_arr = np.flipud(np.where((arr >= 1) & (arr <= 17), arr, 0))

    # Build color mapping
    color_list, _ = _build_plotly_discrete_colorscale(inclusive)

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
            hovertemplate="X: %{x:.5f}<br>Y: %{y:.5f}<extra></extra>",
        ))
    else:
        # Use Heatmap with discrete colorscale
        fig.add_trace(go.Heatmap(
            z=display_arr,
            x0=xmin,
            y0=ymin,
            dx=dx,
            dy=dy,
            colorscale=[(i / 17, color_list[i]) for i in range(18)],
            zmin=0,
            zmax=17,
            showscale=False,
            hovertemplate=(
                "LCZ class: %{z:.0f}<br>"
                "X: %{x:.5f}<br>"
                "Y: %{y:.5f}<extra></extra>"
            ),
        ))

    # Add legend as separate traces (more flexible than colorbar). Classes are
    # emitted in ascending class order, matching the LCZ scheme.
    if show_legend:
        present = _present_classes(arr)
        colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
        names = get_lcz_names(lang)

        for lcz_class in present:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=13,
                    color=colors[lcz_class - 1],
                    symbol="square",
                    line=dict(width=1, color="black"),
                ),
                name=f"{lcz_class}: {names[lcz_class - 1]}",
                legendgroup="lcz_classes",
                showlegend=True,
                hoverinfo="name",
            ))

    # Configure layout
    fig.update_layout(
        title=dict(
            text=lcz_msg("lcz_map_title", lang),
            font=dict(size=18),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False,
            ticks="",
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False,
            ticks="",
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            title=dict(text=lcz_msg("lcz_class_legend", lang), font=dict(size=12)),
            font=dict(size=10),
            itemsizing="constant",
            traceorder="normal",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(90,90,90,0.6)",
            borderwidth=1,
            x=1.01,
            y=0.5,
            xanchor="left",
            yanchor="middle",
        ),
        margin=dict(l=10, r=190, t=55, b=25),
        width=1200,
        height=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode="pan",  # Enable panning
    )

    _add_attribution(fig, attribution)
    return fig


def _create_continuous_map(
    arr: np.ndarray,
    extent: tuple[float, float, float, float],
    units: str,
    title: Optional[str] = None,
    colorscale: str = "RdBu_r",
    attribution: Optional[str] = None,
) -> go.Figure:
    """Create a map for a continuous raster variable (e.g. LST)."""
    # flipud: raster row 0 is north, but Plotly y0=ymin (south) — align them
    display_arr = np.flipud(np.asarray(arr, dtype=float))
    display_arr[~np.isfinite(display_arr)] = np.nan

    xmin, xmax, ymin, ymax = extent
    h, w = display_arr.shape
    dx = (xmax - xmin) / w
    dy = (ymax - ymin) / h

    finite = display_arr[np.isfinite(display_arr)]
    zmin, zmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if zmin == zmax:
        zmin, zmax = zmin - 0.5, zmax + 0.5

    unit_label = (units or "").strip()
    # Common abbreviations -> proper symbols (kept explicit: a wrong unit on a
    # colourbar is worse than no unit at all).
    unit_label = {
        "C": "°C", "c": "°C", "celsius": "°C", "degc": "°C",
        "K": "K", "kelvin": "K",
        "m": "m", "km": "km", "m2": "m²",
    }.get(unit_label, unit_label)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=display_arr,
        x0=xmin, y0=ymin, dx=dx, dy=dy,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(
            title=dict(text=unit_label or None, side="right", font=dict(size=11)),
            thickness=16,
            len=0.75,
            tickfont=dict(size=10),
            outlinewidth=0.8,
            outlinecolor="black",
        ),
        hovertemplate=(
            f"Value: %{{z:.2f}}{' ' + unit_label if unit_label else ''}<br>"
            "X: %{x:.5f}<br>Y: %{y:.5f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text=title or "Land Surface Temperature",
            x=0.5, xanchor="center", font=dict(size=18),
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                   showline=False, ticks=""),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                   showline=False, ticks="", scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=90, t=55, b=25),
        width=1200,
        height=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode="pan",
    )
    _add_attribution(fig, attribution)
    return fig


def _resolve_band(src: rasterio.io.DatasetReader, band: Union[int, str, None]) -> int:
    """Resolve a 1-based band index from an int, a matching band description, or None (-> 1)."""
    if band is None:
        return 1
    if isinstance(band, bool):  # bool is an int subclass — reject explicitly
        raise TypeError("band must be an int, a band description string, or None")
    if isinstance(band, int):
        if not (1 <= band <= src.count):
            raise ValueError(f"band {band} out of range (raster has {src.count} bands)")
        return band
    for i, desc in enumerate(src.descriptions, start=1):
        if desc == band:
            return i
    raise ValueError(f"No band with description {band!r}. Available: {list(src.descriptions)}")


# ── Native static renderer (matplotlib/Agg — no Plotly/Kaleido/Chromium) ─────

def _create_static_map(
    arr: np.ndarray,
    extent: tuple,
    crs_obj,
    *,
    is_continuous: bool,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    attribution: Optional[str],
    style: str,
    dpi: Optional[int],
    lang: str,
    show_legend: bool,
    inclusive: bool,
    colorscale: str,
    units: str,
    add_scalebar: bool,
    add_north_arrow: bool,
    isave: bool,
    save_extension: str,
    filename: str,
    column: Optional[str] = None,
):
    """Build a publication-quality map with matplotlib (no browser needed)."""
    st = lcz_static

    aspect = arr.shape[0] / max(arr.shape[1], 1)
    has_legend = (
        not is_continuous
        and bool(show_legend)
        and bool(np.any((arr >= 1) & (arr <= 17)))
    )
    extra_mm = 26.0 if has_legend else 14.0

    with st.style_context(style):
        fig, spec, _ = st.new_figure(
            style,
            aspect=aspect,
            extra_height_mm=extra_mm,
            column=column,
            dpi=dpi,
        )
        has_header = bool(title or subtitle)
        ax = st.add_axes(
            fig,
            bottom=st.figure_bottom(has_legend),
            top=st.figure_top(has_header),
        )

        if is_continuous:
            image = st.plot_continuous_raster(
                ax, arr, extent, colorscale=colorscale
            )
        else:
            image = st.plot_lcz_raster(
                ax, arr, extent, inclusive=inclusive
            )

        is_geo = st.apply_map_axes(ax, extent, crs_obj)

        if is_continuous:
            st.add_colorbar(fig, ax, image, units=units, fontsize=spec.font_size)
        elif has_legend:
            st.class_legend(
                fig,
                arr,
                inclusive=inclusive,
                lang=lang,
                fontsize=max(5.5, spec.font_size - 1),
                bbox_to_anchor=(0.5, 0.085),
            )

        if add_scalebar or add_north_arrow:
            if add_scalebar:
                st.add_scalebar(ax, st.metric_width(extent, crs_obj), is_geo)
            if add_north_arrow and not is_geo:
                st.add_north_arrow(ax)

        st.add_header(fig, title, subtitle, spec)
        st.add_footer(fig, caption, attribution, spec)
        st.save_static(
            fig,
            filename,
            isave=isave,
            save_extension=save_extension,
            style=style,
            dpi=dpi,
            lang=lang,
        )

    return fig


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class LCZPlotResult:
    """Return type for advanced LCZ map plotting.

    ``fig`` is a matplotlib ``Figure`` for the default static renderer and a
    Plotly ``go.Figure`` for ``renderer="plotly"``/``"maplibre"``.
    """
    fig: Optional[object] = None
    geoarrow_table: Optional[pa.Table] = None
    legend_df: Optional[pl.DataFrame] = None
    html: Optional[str] = None  # MapLibre renderer output
    renderer: str = "static"

    def show(self) -> None:
        """Display the figure in the active frontend (matplotlib or Plotly)."""
        if self.fig is None:
            return
        if type(self.fig).__module__.startswith("matplotlib"):
            import matplotlib.pyplot as plt

            plt.show()
        else:
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
    renderer: Optional[Literal["static", "plotly", "maplibre"]] = "static",
    opacity: float = 0.8,
    lang: str = "en",
    data_type: Literal["auto", "lcz", "continuous"] = "auto",
    band: Union[int, str, None] = None,
    colorscale: str = "RdBu_r",
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
    dpi: Optional[int] = None,
    attribution: Optional[str] = None,
) -> LCZPlotResult:
    """Plot an LCZ map, or a continuous raster stack (e.g. lcz_get_lst output).

    Parameters
    ----------
    x : str, Path or rasterio dataset
        Source raster. Either an LCZ class map (integer codes 1-17) or a
        continuous multi-band stack such as the GeoTIFF returned by
        ``lcz_get_lst`` (float LST values, one band per date).
    isave : bool
        Save the figure. With the default static renderer this writes a
        native matplotlib file (PNG by default) — no Kaleido/Chromium needed.
    save_extension : str
        File extension for static output: "png" (default), "pdf", "svg",
        "tiff"/"tif", "jpg"/"jpeg" or "webp". "html" is accepted for the
        Plotly renderer and coerced to "png" for the static renderer.
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
    renderer : {"static", "plotly", "maplibre"}, optional
        "static" (default, also ``None``/``"matplotlib"``/``"agg"``) returns a
        matplotlib ``Figure`` and saves native PNG/PDF/SVG/TIFF — it never
        needs Plotly, Kaleido, Chromium or a browser. "plotly" returns an
        interactive WebGL Plotly figure; "maplibre" returns a Plotly mapbox
        figure overlaid on an OpenStreetMap basemap. Continuous data supports
        both "static" and "plotly".
    opacity : float
        Raster overlay opacity for the MapLibre renderer (0–1, default 0.8).
    lang : {"en", "pt", "es", "zh"}
        Language for legend labels and default titles.
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
    dpi : int, optional
        Resolution for static exports. Defaults to 300 dpi (or the style
        preset's own resolution if higher).
    attribution : str, optional
        Credit line printed in the bottom-right corner. Defaults to
        ``"LCZ4py | Stewart & Oke (2012) LCZ scheme"`` (plus the basemap
        credit for the MapLibre renderer).

    Returns
    -------
    LCZPlotResult
        Contains the figure (Plotly) or html string (MapLibre) and optional metadata.
    """
    # ── Argument validation ──────────────────────────────────────────────────
    if data_type not in ("auto", "lcz", "continuous"):
        raise ValueError(
            f"data_type must be 'auto', 'lcz' or 'continuous', got {data_type!r}"
        )
    # None / "matplotlib" / "agg" -> the native static renderer (default).
    renderer = resolve_renderer(renderer)
    if not 0.0 <= float(opacity) <= 1.0:
        raise ValueError(f"opacity must be between 0 and 1, got {opacity!r}")
    if (save_extension or "").lower().lstrip(".") not in (
        "html", "png", "pdf", "svg", "tiff", "jpg", "jpeg", "webp"
    ):
        raise ValueError(
            f"Unsupported save_extension {save_extension!r} "
            "(use html, png, pdf, svg, tiff, jpg or webp)."
        )
    save_extension = (save_extension or "html").lower().lstrip(".")

    # Reading a raster needs an open dataset; get one with correct lifetime.
    close_src = False
    if isinstance(x, (str, Path)):
        src = rasterio.open(str(x))
        close_src = True
    elif hasattr(x, "read") and hasattr(x, "dtypes"):
        src = x
    else:
        raise TypeError(f"x must be a path or rasterio dataset, got {type(x)}")

    try:
        if src.width < 1 or src.height < 1:
            raise ValueError("Raster has zero width/height; nothing to plot.")

        if band is not None and not isinstance(band, (int, str)):
            raise TypeError("band must be an int, a band description string, or None")
        if isinstance(band, int) and not (1 <= band <= src.count):
            # Validate even for class maps, where the band is otherwise unused:
            # silently ignoring an impossible band hides a caller mistake.
            raise ValueError(f"band {band} out of range (raster has {src.count} bands)")

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
        units = src.tags().get("units", "") if is_continuous else ""
        descriptions = list(src.descriptions)
        band_label = descriptions[band_idx - 1] if descriptions[band_idx - 1] else None
        nodata = src.nodata
    finally:
        if close_src:
            src.close()

    if extent[0] >= extent[1] or extent[2] >= extent[3]:
        raise ValueError(
            "Raster extent is degenerate (non-increasing bounds); check the GeoTIFF transform."
        )

    credit = attribution if attribution is not None else DEFAULT_ATTRIBUTION

    # ── Continuous raster (e.g. lcz_get_lst LST stack) ───────────────────────
    if is_continuous:
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        band_suffix = f" — {band_label}" if band_label else ""
        map_title = title or f"Land Surface Temperature{band_suffix}"
        if renderer == "static":
            fig = _create_static_map(
                arr, extent, crs_obj,
                is_continuous=True,
                title=map_title, subtitle=subtitle, caption=caption,
                attribution=credit, style=style, dpi=dpi, lang=lang,
                show_legend=False, inclusive=inclusive,
                colorscale=colorscale, units=units,
                add_scalebar=add_scalebar, add_north_arrow=add_north_arrow,
                isave=isave, save_extension=save_extension,
                filename="lcz_plot_lst",
            )
            return LCZPlotResult(fig=fig, renderer="static")
        fig = _create_continuous_map(
            arr, extent, units, title=map_title, colorscale=colorscale,
            attribution=credit,
        )
        if subtitle:
            fig.update_layout(title=dict(text=f"{map_title}<br><sup>{subtitle}</sup>"))
        _add_caption(fig, caption)
        bounds = (extent[0], extent[2], extent[1], extent[3])
        add_map_furniture(
            fig, bounds=bounds, crs=crs_obj,
            add_scalebar=add_scalebar, add_north_arrow=add_north_arrow,
        )
        fig = _finalize_publication(
            fig, style=style, isave=isave, save_extension=save_extension,
            filename="lcz_plot_lst", dpi=dpi, lang=lang,
        )
        return LCZPlotResult(fig=fig, renderer="plotly")

    # Keep nodata (0) as 0 — transparent in the plot, not Water.
    # Water bodies are stored as class 17 in the source LCZ raster.
    arr = np.asarray(arr)
    arr = np.where((arr >= 1) & (arr <= 17), arr, 0)

    # ── Native static renderer (default) ─────────────────────────────────────
    if renderer == "static":
        fig = _create_static_map(
            arr, extent, crs_obj,
            is_continuous=False,
            title=title or lcz_msg("lcz_map_title", lang),
            subtitle=subtitle, caption=caption, attribution=credit,
            style=style, dpi=dpi, lang=lang,
            show_legend=show_legend, inclusive=inclusive,
            colorscale=colorscale, units="",
            add_scalebar=add_scalebar, add_north_arrow=add_north_arrow,
            isave=isave, save_extension=save_extension,
            filename="lcz_plot_map",
        )
        return LCZPlotResult(fig=fig, renderer="static")

    # ── MapLibre renderer (Plotly mapbox + OSM basemap) ──────────────────────
    if renderer == "maplibre":
        # ponytail: no add_map_furniture here — this figure is a mapbox/geo trace,
        # not a Cartesian one, so paper-relative scale bar/north arrow annotations
        # don't line up with the map; skip until furniture supports mapbox layouts.
        fig = _create_basemap_fig(
            arr, extent, inclusive, show_legend, title, opacity,
            lang=lang, attribution=credit,
        )
        if subtitle:
            fig.update_layout(title=dict(
                text=f"{title or lcz_msg('lcz_map_title', lang)}<br><sup>{subtitle}</sup>",
            ))
        _add_caption(fig, caption)
        fig = finalize_export(
            fig, style=style, isave=isave, save_extension="html",
            filename="lcz_plot_map_maplibre", lang=lang,
        )
        return LCZPlotResult(fig=fig, renderer="maplibre")

    # ── Plotly renderer (default) ────────────────────────────────────────────
    fig = _create_interactive_lcz_map(
        arr, extent, inclusive, show_legend, use_webgl,
        lang=lang, attribution=credit,
    )

    # Add annotations
    if title or subtitle:
        title_text = title or lcz_msg("lcz_map_title", lang)
        if subtitle:
            title_text += f"<br><sup>{subtitle}</sup>"
        fig.update_layout(title=dict(text=title_text))
    _add_caption(fig, caption)

    # Generate GeoArrow table if requested
    geoarrow_table = None
    if use_geoarrow and HAS_GEOARROW and arr.shape[0] * arr.shape[1] < 5_000_000:
        geoarrow_table = raster_to_geoarrow(arr, transform, crs)

    # Compute legend with DuckDB if requested
    legend_df = None
    if use_duckdb:
        legend_df = compute_legend_with_duckdb(arr, transform, crs)
    if show_legend and not _present_classes(arr):
        logger.warning(
            "No LCZ classes (1-17) found in the raster; the map will be blank. "
            "Check the band/dtype (continuous data needs data_type='continuous')."
        )

    # Add map furniture (scale bar / north arrow), then save
    bounds = (extent[0], extent[2], extent[1], extent[3])
    add_map_furniture(
        fig, bounds=bounds, crs=crs_obj,
        add_scalebar=add_scalebar, add_north_arrow=add_north_arrow,
    )
    fig = _finalize_publication(
        fig, style=style, isave=isave, save_extension=save_extension,
        filename="lcz_plot_map", dpi=dpi, lang=lang,
    )

    return LCZPlotResult(
        fig=fig,
        geoarrow_table=geoarrow_table,
        legend_df=legend_df,
        renderer="plotly",
    )


__all__ = [
    "LCZPlotResult",
    "lcz_plot_map",
    "BlockReader",
    "raster_to_geoarrow",
    "compute_legend_with_duckdb",
]
