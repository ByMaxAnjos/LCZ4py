"""
lcz_cal_area.py

LCZ area statistics and publication-quality charts.

- NumPy ``bincount`` aggregation (O(n), no per-pixel Python objects)
- Optional DuckDB for the final percentage rollup (``HAS_DUCKDB`` guarded)
- **Default** native static matplotlib/Agg renderer (``renderer="static"``):
  bar / pie / donut / nested-sunburst / treemap drawn with the shared
  ``_internal/lcz_static`` helpers and the LCZ palette (or its colour-blind
  variant via ``inclusive=True``). Saves PNG/PDF/SVG/TIFF at >=300 dpi with
  no Plotly, Kaleido, Chromium or browser involved.
- Opt-in Plotly renderer (``renderer="plotly"``) with the previous bar / pie /
  donut / sunburst / treemap figures, readable labels, contrast-aware
  annotation colours and hover metadata.
- >=300 dpi static export in both backends
"""

from __future__ import annotations

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
from rasterio.crs import CRS as RioCRS
from rasterio.transform import Affine

import plotly.graph_objects as go

# DuckDB (optional)
try:  # pragma: no cover - depends on the environment
    import duckdb

    HAS_DUCKDB = True
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]
    HAS_DUCKDB = False

# GeoArrow (optional)
try:  # pragma: no cover - depends on the environment
    import geoarrow.pyarrow as ga

    HAS_GEOARROW = True
except ImportError:  # pragma: no cover
    ga = None  # type: ignore[assignment]
    HAS_GEOARROW = False

# ExactExtract (optional; not required by the area calculation itself)
try:  # pragma: no cover - depends on the environment
    from exactextract import exact_extract  # noqa: F401

    HAS_EXACTEXTRACT = True
except ImportError:  # pragma: no cover
    HAS_EXACTEXTRACT = False

from LCZ4py._internal.lcz_parameters_data import (
    LCZ_COLORS,
    LCZ_COLORBLIND,
    get_lcz_names,
)
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal import lcz_static as _st
from LCZ4py.general.lcz_plot_map import _add_caption, _finalize_publication

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"

VALID_LCZ_CLASSES = tuple(range(1, 18))


# ── Fast cell area computation ────────────────────────────────────────────────

def _crs_is_geographic(crs: Optional[str]) -> Optional[bool]:
    """Best-effort geographic/projected detection from a CRS string."""
    if not crs:
        return None
    try:
        return bool(RioCRS.from_user_input(crs).is_geographic)
    except Exception:
        return None


def _crs_linear_factor(crs: Optional[str]) -> float:
    """Metres per CRS linear unit (1.0 for metres, 0.3048 for feet, ...)."""
    if not crs:
        return 1.0
    try:
        _, factor = RioCRS.from_user_input(crs).linear_units_factor
        return float(factor) or 1.0
    except Exception:
        return 1.0


@lru_cache(maxsize=128)
def _cell_area_km2(
    transform: Affine,
    shape: tuple[int, int],
    crs: Optional[str] = None,
) -> np.ndarray:
    """Compute per-cell area in km² with vectorized operations.

    Handles both geographic (degrees) and projected (metres/feet) CRSs. For
    geographic rasters the cell size is computed at each row's *true* latitude
    using the WGS84 radii of curvature, so area totals stay accurate from the
    equator to high latitudes (a constant ``cos(lat)`` factor over a tall tile
    is off by several percent).
    """
    h, w = int(shape[0]), int(shape[1])
    if h < 1 or w < 1:
        raise ValueError(f"Raster shape must be positive, got {shape}")
    if transform is None:
        raise ValueError("Raster has no transform; cannot compute cell areas.")

    is_geo = _crs_is_geographic(crs)
    if is_geo is None:
        # No/unknown CRS: fall back to the degree-sized heuristic.
        is_geo = abs(transform.a) <= 1.0 and abs(transform.e) <= 1.0

    if is_geo:
        res_y_deg = abs(transform.e)
        res_x_deg = abs(transform.a)

        # Row-centre latitudes (transform.f = northern edge, e is negative)
        lats = transform.f + (np.arange(h) + 0.5) * transform.e

        # WGS84 radii of curvature: M (meridian) and N (prime vertical).
        # For a small lat/lon cell the exact area is
        #   dφ · dλ · M(φ) · N(φ) · cos(φ)
        # a = 6378.137 km, b = 6356.752 km, f = 1/298.257
        a, b = 6378.137, 6356.752
        e2 = 1 - (b / a) ** 2
        sin_lat = np.sin(np.radians(lats))
        w_den = 1 - e2 * sin_lat**2
        n_radius = a / np.sqrt(w_den)              # km
        m_radius = a * (1 - e2) / (w_den ** 1.5)   # km

        dy_km = res_y_deg * (np.pi / 180) * m_radius
        dx_km = res_x_deg * (np.pi / 180) * n_radius * np.cos(np.radians(lats))

        cell_km2 = np.repeat((dy_km * dx_km)[:, np.newaxis], w, axis=1)
    else:
        # Projected CRS: pixel area x (metres per unit)²
        factor = _crs_linear_factor(crs)
        cell_km2 = np.full(
            (h, w), abs(transform.a * transform.e) * factor * factor / 1e6
        )

    return np.asarray(cell_km2, dtype=np.float64)


# ── Fast aggregation ─────────────────────────────────────────────────────────

def _aggregate_classes(
    classes: np.ndarray,
    areas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (class ids, pixel counts, total area) for LCZ classes 1-17.

    ``np.bincount`` over the class codes is O(n) with no per-pixel Python
    objects — the previous implementation materialised a Polars column from
    ``.tolist()``, which dominated runtime (and RAM) on continent-sized rasters.
    """
    classes = np.asarray(classes)
    areas = np.asarray(areas, dtype=np.float64)
    if classes.size != areas.size:
        raise ValueError(
            f"classes and areas must have the same size ({classes.size} vs {areas.size})"
        )
    if classes.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty.copy(), np.array([], dtype=np.float64)

    codes = classes.astype(np.int64, copy=False)
    codes = np.clip(codes, 0, 17)
    counts = np.bincount(codes, minlength=18)[1:18]
    sums = np.bincount(codes, weights=areas, minlength=18)[1:18]

    present = counts > 0
    ids = np.arange(1, 18, dtype=np.int64)[present]
    return ids, counts[present], sums[present]


def _summary_from_arrays(
    ids: np.ndarray,
    counts: np.ndarray,
    sums: np.ndarray,
) -> pl.DataFrame:
    """Build the summary frame (class, count, area, percentage)."""
    if ids.size == 0:
        return pl.DataFrame(
            schema={
                "lcz": pl.Int64,
                "count": pl.Int64,
                "area_km2": pl.Float64,
                "area_perc": pl.Float64,
            }
        )
    total = float(sums.sum())
    df = pl.DataFrame({
        "lcz": ids.astype(np.int64),
        "count": counts.astype(np.int64),
        "area_km2": np.round(sums, 2),
        "area_perc": np.round(sums / total * 100.0, 2) if total > 0 else np.zeros_like(sums),
    })
    return df.sort("lcz")


def _build_summary_polars(
    classes: np.ndarray,
    areas: np.ndarray,
) -> pl.DataFrame:
    """Summary using NumPy + Polars (always available)."""
    ids, counts, sums = _aggregate_classes(classes, areas)
    return _summary_from_arrays(ids, counts, sums)


def _build_summary_duckdb(
    classes: np.ndarray,
    areas: np.ndarray,
) -> pl.DataFrame:
    """Summary using DuckDB (optional; falls back to Polars)."""
    if not HAS_DUCKDB:
        return _build_summary_polars(classes, areas)

    ids, counts, sums = _aggregate_classes(classes, areas)
    if ids.size == 0:
        return _summary_from_arrays(ids, counts, sums)

    try:
        df = pl.DataFrame({
            "lcz": ids.tolist(),
            "count": counts.tolist(),
            "area_km2": sums.tolist(),
        })
        conn = duckdb.connect()
        try:
            result = conn.execute("""
                SELECT
                    lcz,
                    count,
                    ROUND(area_km2, 2) AS area_km2,
                    ROUND(area_km2 * 100.0 / SUM(area_km2) OVER (), 2) AS area_perc
                FROM df
                ORDER BY lcz
            """).pl()
        finally:
            conn.close()
        return result
    except Exception as exc:  # noqa: BLE001 - DuckDB is an optimisation only
        logger.warning("DuckDB aggregation failed (%s); falling back to Polars.", exc)
        return _summary_from_arrays(ids, counts, sums)


# ── Add metadata columns ──────────────────────────────────────────────────────

def _contrast_color(hex_color: str) -> str:
    """Black or white annotation text, whichever is readable on ``hex_color``."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return "black"
    # W3C relative luminance
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "black" if luminance > 0.55 else "white"


def _attach_metadata(
    df: pl.DataFrame,
    inclusive: bool,
    lang: str = "en",
) -> pl.DataFrame:
    """Add name, colour and readable-label columns."""
    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS

    names_df = pl.DataFrame({
        "lcz": list(range(1, 18)),
        "lcz_name": get_lcz_names(lang),
        "lcz_col": LCZ_COLORS,
        "lcz_colorblind": LCZ_COLORBLIND,
    })

    result = df.join(names_df, on="lcz", how="left")
    color_col = "lcz_colorblind" if inclusive else "lcz_col"
    result = result.with_columns(
        pl.col(color_col).alias("color"),
    ).with_columns(
        (
            pl.col("lcz").cast(pl.String) + pl.lit(": ") + pl.col("lcz_name").fill_null("Unknown")
        ).alias("readable"),
        pl.col("color")
        .map_elements(_contrast_color, return_dtype=pl.String)
        .alias("text_color"),
    )
    return result


# ── Plotly charts ────────────────────────────────────────────────────────────

def _base_title(text: str) -> dict:
    return dict(text=text, x=0.5, xanchor="center", font=dict(size=16))


def _legend(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=11)),
        font=dict(size=10),
        itemsizing="constant",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(90,90,90,0.5)",
        borderwidth=1,
        yanchor="middle",
        y=0.5,
    )


def _plot_bar_interactive(
    df: pl.DataFrame,
    xlab: str = "LCZ code",
    ylab: str = "Area [km²]",
    lang: str = "en",
) -> go.Figure:
    """Bar chart of area per LCZ class, ordered by class code."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["lcz"].cast(pl.String),
        y=df["area_km2"],
        marker_color=df["color"].to_list(),
        marker_line_color="black",
        marker_line_width=0.6,
        text=[
            f"{p:.1f}%" if p >= 0.1 else ""
            for p in df["area_perc"].to_list()
        ],
        textposition="outside",
        textfont=dict(size=9),
        cliponaxis=False,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Area: %{y:,.2f} km²<br>"
            "Share: %{customdata[1]:.2f}%<br>"
            "Pixels: %{customdata[2]:,}<extra></extra>"
        ),
        customdata=np.column_stack([
            df["readable"].to_list(),
            df["area_perc"].to_numpy(),
            df["count"].to_numpy(),
        ]),
    ))

    fig.update_layout(
        title=_base_title(lcz_msg("lcz_area_title", lang)),
        xaxis=dict(
            title=dict(text=xlab, font=dict(size=11)),
            tickfont=dict(size=10),
            type="category",
            categoryorder="array",
            categoryarray=[str(c) for c in df["lcz"].to_list()],
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            tickangle=0,
        ),
        yaxis=dict(
            title=dict(text=ylab, font=dict(size=11)),
            tickfont=dict(size=10),
            gridcolor="#e8e8e8",
            gridwidth=0.6,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            rangemode="tozero",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=25, t=60, b=55),
        width=1000,
        height=600,
        showlegend=False,
        bargap=0.25,
        hoverlabel=dict(bgcolor="white", font=dict(size=11)),
    )

    return fig


def _pie_like(
    df: pl.DataFrame,
    *,
    hole: float,
    show_percent: bool,
    title: str,
    lang: str,
    center_text: Optional[str] = None,
) -> go.Figure:
    """Shared implementation for pie and donut charts.

    Only the class code goes inside a slice (short and never overlapping);
    the full class name, area and share live in the legend/hover.
    """
    fig = go.Figure(go.Pie(
        # Full "code: name" labels feed the legend; only the percentage goes
        # inside the slice, so long class names never overlap each other.
        labels=df["readable"].to_list(),
        values=df["area_km2"].to_list(),
        marker_colors=df["color"].to_list(),
        marker_line_color="white",
        marker_line_width=1.5,
        textinfo="percent" if show_percent else "none",
        textposition="inside",
        insidetextorientation="horizontal",
        textfont=dict(size=10, color=df["text_color"].to_list()),
        customdata=np.column_stack([
            df["readable"].to_list(),
            df["area_perc"].to_numpy(),
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Area: %{value:,.2f} km²<br>"
            "Share: %{customdata[1]:.2f}%<extra></extra>"
        ),
        hole=hole,
        sort=False,
        direction="clockwise",
        rotation=0,
    ))

    annotations = []
    if center_text:
        annotations.append(dict(
            text=center_text,
            x=0.5, y=0.5,
            font=dict(size=13),
            showarrow=False,
        ))

    fig.update_layout(
        title=_base_title(title),
        margin=dict(l=20, r=20, t=65, b=20),
        width=850,
        height=800,
        showlegend=True,
        legend=_legend(lcz_msg("lcz_class_legend", lang)),
        annotations=annotations,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _plot_pie_interactive(df: pl.DataFrame, lang: str = "en") -> go.Figure:
    """Interactive pie chart with Plotly."""
    total = float(df["area_km2"].sum())
    return _pie_like(
        df,
        hole=0.0,
        show_percent=True,
        title=f"{lcz_msg('lcz_area_title', lang)}<br><sup>Total: {total:,.0f} km²</sup>",
        lang=lang,
    )


def _plot_donut_interactive(df: pl.DataFrame, lang: str = "en") -> go.Figure:
    """Interactive donut chart with Plotly."""
    total = float(df["area_km2"].sum())
    return _pie_like(
        df,
        hole=0.45,
        show_percent=True,
        title=lcz_msg("lcz_area_title", lang),
        lang=lang,
        center_text=f"Total<br>{total:,.0f} km²",
    )


def _plot_sunburst_interactive(df: pl.DataFrame, lang: str = "en") -> go.Figure:
    """Interactive sunburst chart (urban / natural / other hierarchy)."""
    total = float(df["area_km2"].sum())

    _cat_labels = {
        "en": ("Urban", "Natural", "Other"),
        "pt": ("Urbano", "Natural", "Outro"),
        "es": ("Urbano", "Natural", "Otro"),
        "zh": ("城市", "自然", "其他"),
    }
    cat_urban, cat_natural, cat_other = _cat_labels.get(lang, _cat_labels["en"])

    def get_category(lcz: int) -> str:
        if 1 <= lcz <= 10:
            return cat_urban
        if 11 <= lcz <= 14:
            return cat_natural
        return cat_other

    category_col = [get_category(int(v)) for v in df["lcz"].to_list()]
    df_with_cat = df.with_columns(pl.Series("category", category_col, dtype=pl.String))

    categories = (
        df_with_cat.group_by("category")
        .agg(pl.col("area_km2").sum().alias("area"))
        .sort("area", descending=True)
    )

    # Distinct, low-saturation parents; children keep their LCZ colours.
    parent_colors = {
        cat_urban: "#B8B8B8",
        cat_natural: "#8FBF9F",
        cat_other: "#9FB8D0",
    }

    labels = categories["category"].to_list() + df_with_cat["readable"].to_list()
    parents = [""] * len(categories) + df_with_cat["category"].to_list()
    values = categories["area"].to_list() + df_with_cat["area_km2"].to_list()
    colors = [
        parent_colors.get(c, "#B8B8B8") for c in categories["category"].to_list()
    ] + df_with_cat["color"].to_list()

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker_colors=colors,
        marker_line_color="white",
        marker_line_width=0.8,
        branchvalues="total",
        insidetextorientation="radial",
        textfont=dict(size=10),
        hovertemplate="<b>%{label}</b><br>Area: %{value:,.2f} km²<extra></extra>",
    ))

    fig.update_layout(
        title=_base_title(
            f"{lcz_msg('lcz_area_title', lang)}<br><sup>Total: {total:,.0f} km²</sup>"
        ),
        margin=dict(t=70, l=10, r=10, b=10),
        width=850,
        height=800,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def _plot_treemap_interactive(df: pl.DataFrame, lang: str = "en") -> go.Figure:
    """Interactive treemap chart."""
    total = float(df["area_km2"].sum())

    fig = go.Figure(go.Treemap(
        labels=df["readable"].to_list(),
        parents=[""] * df.height,
        values=df["area_km2"].to_list(),
        marker_colors=df["color"].to_list(),
        marker_line_color="white",
        marker_line_width=1.5,
        textfont=dict(size=10, color=df["text_color"].to_list()),
        textinfo="label+percent entry",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Area: %{value:,.2f} km²<br>"
            "Share: %{percentEntry}<extra></extra>"
        ),
        branchvalues="total",
        tiling=dict(pad=1),
    ))

    fig.update_layout(
        title=_base_title(
            f"{lcz_msg('lcz_area_title', lang)}<br><sup>Total: {total:,.0f} km²</sup>"
        ),
        margin=dict(t=70, l=10, r=10, b=10),
        width=1000,
        height=800,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# ── Native static charts (matplotlib/Agg — no Plotly/Kaleido/browser) ────────

#: Urban / natural / other groups used by the sunburst hierarchy.
_CATEGORY_LABELS: dict[str, tuple[str, str, str]] = {
    "en": ("Urban", "Natural", "Other"),
    "pt": ("Urbano", "Natural", "Outro"),
    "es": ("Urbano", "Natural", "Otro"),
    "zh": ("城市", "自然", "其他"),
}


def _category_of(lcz: int, lang: str) -> str:
    """Vendor-neutral LCZ -> {urban, natural, other} grouping (1-10/11-14/15-17)."""
    urban, natural, other = _CATEGORY_LABELS.get(lang, _CATEGORY_LABELS["en"])
    if 1 <= lcz <= 10:
        return urban
    if 11 <= lcz <= 14:
        return natural
    return other


def _legend_handles(df: pl.DataFrame) -> list:
    """Legend patches in class order, using the palette already in ``df``."""
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=color, edgecolor="black", linewidth=0.3, label=label)
        for color, label in zip(df["color"].to_list(), df["readable"].to_list())
    ]


def _class_legend(
    fig,
    df: pl.DataFrame,
    spec,
    *,
    lang: str = "en",
    ncol: int = 3,
    y: float = 0.03,
    title: Optional[str] = None,
):
    """Ordered LCZ class legend below the panel (no plot area consumed)."""
    handles = _legend_handles(df)
    if not handles:
        return None
    fontsize = max(5.5, spec.font_size - 2)
    return fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=min(int(ncol), max(2, len(handles))),
        frameon=False,
        fontsize=fontsize,
        handlelength=1.2,
        columnspacing=1.3,
        handletextpad=0.45,
        title=title or lcz_msg("lcz_class_legend", lang),
        title_fontsize=fontsize,
    )


def _worst_ratio(row: list[float], side: float) -> float:
    """Bruls/Squarify aspect ratio of a row laid along ``side``."""
    total = float(sum(row))
    if total <= 0 or side <= 0:
        return float("inf")
    worst = 0.0
    for area in row:
        if area <= 0:
            return float("inf")
        ratio = max((side * side * area) / (total * total),
                    (total * total) / (side * side * area))
        worst = max(worst, ratio)
    return worst


def _squarify(values: list[float], x: float, y: float, w: float, h: float) -> list[tuple]:
    """Squarified treemap layout; returns ``[(x, y, w, h), ...]`` in data units."""
    total = float(sum(values))
    if total <= 0 or w <= 0 or h <= 0:
        return []
    scale = (w * h) / total
    remaining = [max(float(v), 0.0) * scale for v in values]
    rects: list[tuple] = []

    while remaining:
        side = h if w >= h else w
        row = [remaining[0]]
        i = 1
        while i < len(remaining) and _worst_ratio(row + [remaining[i]], side) <= _worst_ratio(row, side):
            row.append(remaining[i])
            i += 1

        row_sum = float(sum(row))
        if w >= h:
            strip_w = row_sum / h if h > 0 else 0.0
            cursor = y
            for area in row:
                rh = area / strip_w if strip_w > 0 else 0.0
                rects.append((x, cursor, strip_w, rh))
                cursor += rh
            x += strip_w
            w -= strip_w
        else:
            strip_h = row_sum / w if w > 0 else 0.0
            cursor = x
            for area in row:
                rw = area / strip_h if strip_h > 0 else 0.0
                rects.append((cursor, y, rw, strip_h))
                cursor += rw
            y += strip_h
            h -= strip_h
        remaining = remaining[len(row):]

    return rects


def _draw_area_bar(
    df: pl.DataFrame,
    *,
    xlab: str,
    ylab: str,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    style: str,
    dpi: Optional[int],
):
    """Vertical bar chart of area per LCZ class, ordered by class code."""
    n_bars = max(int(df.height), 1)
    # 17 percentages on one row need room: widen the panel beyond the preset's
    # single column so the labels stay legible (and >=5 pt) instead of
    # overprinting each other.
    _, single_mm, _ = _st.figure_dims(style, column="single")
    width_mm = max(float(single_mm), 5.8 * n_bars + 30.0)

    fig, spec, _ = _st.new_figure(
        style, width_mm=width_mm, aspect=0.66, dpi=dpi,
    )
    has_header = bool(title or subtitle)
    left, right = 0.13, 0.97
    ax = _st.add_axes(
        fig, bottom=0.19, top=_st.figure_top(has_header), left=left, right=right,
    )

    x = np.arange(n_bars, dtype=float)
    areas = df["area_km2"].to_numpy().astype(float)
    percentages = df["area_perc"].to_numpy().astype(float)
    ax.bar(
        x, areas, width=0.78, color=df["color"].to_list(),
        edgecolor="black", linewidth=0.45, zorder=2,
    )

    # Label size derived from the space each bar actually has (points).
    axes_width_pt = float(fig.get_size_inches()[0]) * 72.0 * (right - left)
    bar_width_pt = axes_width_pt / n_bars
    label_font = max(5.0, min(8.0, bar_width_pt * 0.33))
    for xi, yi, pct in zip(x, areas, percentages):
        if pct >= 0.1:
            ax.annotate(
                f"{pct:.1f}%", (xi, yi), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=label_font, zorder=3,
            )

    top = float(np.nanmax(areas)) if areas.size and np.isfinite(areas).any() else 1.0
    ax.set_ylim(0.0, top * 1.14 if top > 0 else 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in df["lcz"].to_list()])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.tick_params(axis="both", labelsize=max(6.0, spec.font_size - 2), length=3, width=0.6)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _draw_area_pie(
    df: pl.DataFrame,
    *,
    hole: float,
    show_legend: bool,
    center_text: Optional[str],
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Pie (``hole=0``) or donut chart with percentages inside the slices."""
    fig, spec, _ = _st.new_figure(
        style, aspect=0.86, extra_height_mm=28.0, column="single", dpi=dpi,
    )
    has_header = bool(title or subtitle)
    ax = _st.add_axes(
        fig, bottom=0.30, top=_st.figure_top(has_header), left=0.06, right=0.94,
    )
    ax.set_aspect("equal")

    values = df["area_km2"].to_numpy().astype(float)
    colors = df["color"].to_list()
    fontsize = max(5.5, spec.font_size - 2)
    total_area = float(np.nansum(values)) or 1.0

    wedges, _texts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=max(1.0 - hole, 0.05), edgecolor="white", linewidth=0.9),
    )

    # Percentages: inside the wedge when it is wide enough, otherwise just
    # outside with a thin leader line — a fixed pctdistance made neighbouring
    # small slices overprint each other.
    inside_r = 0.72 if hole else 0.68
    for wedge, area, color in zip(wedges, values, colors):
        share = area / total_area
        if share <= 0 or not np.isfinite(share):
            continue
        mid = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        cos, sin = float(np.cos(mid)), float(np.sin(mid))
        label = f"{share * 100:.1f}%"
        if share * 100.0 >= 4.0:
            ax.text(
                inside_r * cos, inside_r * sin, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=_contrast_color(color),
            )
        else:
            ax.annotate(
                label,
                xy=(0.98 * cos, 0.98 * sin),
                xytext=(1.10 * cos, 1.10 * sin),
                ha="left" if cos >= 0 else "right", va="center",
                fontsize=max(4.5, spec.font_size - 3), color="#222222",
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4,
                                shrinkA=0, shrinkB=0),
            )
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")

    if center_text:
        ax.text(
            0.0, 0.0, center_text, ha="center", va="center",
            fontsize=max(6.0, spec.font_size - 1), linespacing=1.35,
        )

    if show_legend:
        _class_legend(fig, df, spec, lang=lang, ncol=3, y=0.035)

    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _draw_area_sunburst(
    df: pl.DataFrame,
    *,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Nested donut (categories inside, LCZ classes outside) — a static sunburst.

    matplotlib has no sunburst primitive; two concentric rings carry the same
    two-level hierarchy without losing the class labels or the total.
    """
    fig, spec, _ = _st.new_figure(
        style, aspect=0.88, extra_height_mm=26.0, column="single", dpi=dpi,
    )
    has_header = bool(title or subtitle)
    ax = _st.add_axes(
        fig, bottom=0.30, top=_st.figure_top(has_header), left=0.06, right=0.94,
    )
    ax.set_aspect("equal")

    urban, natural, other = _CATEGORY_LABELS.get(lang, _CATEGORY_LABELS["en"])
    parent_colors = {urban: "#B8B8B8", natural: "#8FBF9F", other: "#9FB8D0"}
    codes = [int(v) for v in df["lcz"].to_list()]
    areas_df = df["area_km2"].to_numpy().astype(float)
    categories = [_category_of(c, lang) for c in codes]

    totals: dict[str, float] = {}
    for cat, area in zip(categories, areas_df):
        totals[cat] = totals.get(cat, 0.0) + float(area)
    cat_order = sorted(totals, key=lambda c: totals[c], reverse=True)
    cat_values = [totals[c] for c in cat_order]
    cat_colors = [parent_colors.get(c, "#B8B8B8") for c in cat_order]

    fontsize = max(5.5, spec.font_size - 2)
    total_area = float(np.nansum(areas_df)) or 1.0

    # Outer ring: LCZ classes, in class order.
    ax.pie(
        areas_df, radius=1.0, colors=df["color"].to_list(),
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.30, edgecolor="white", linewidth=0.8),
    )
    # Inner ring: urban / natural / other aggregates.
    inner_wedges, _t = ax.pie(
        cat_values, radius=0.70, colors=cat_colors,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.28, edgecolor="white", linewidth=0.8),
    )

    # Category name + share on the inner ring: the name runs *tangentially*
    # and the share horizontally, so neither sprawls across the ring band.
    for wedge, cat, value in zip(inner_wedges, cat_order, cat_values):
        share = value / total_area
        if share < 0.04:
            continue
        mid = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        cos, sin = float(np.cos(mid)), float(np.sin(mid))
        if share >= 0.09:
            rot = (wedge.theta1 + wedge.theta2) / 2.0 + 90.0
            if cos < 0:  # keep the label upright on the left half
                rot += 180.0
            ax.text(
                0.55 * cos, 0.55 * sin, cat,
                ha="center", va="center", rotation=rot,
                rotation_mode="anchor", fontsize=max(4.5, fontsize - 1),
                color="#222222",
            )
            ax.text(
                0.655 * cos, 0.655 * sin, f"{share * 100:.0f}%",
                ha="center", va="center", fontsize=max(4.5, fontsize - 2),
                color="#222222", fontweight="bold",
            )
        else:
            ax.text(
                0.56 * cos, 0.56 * sin, f"{share * 100:.0f}%",
                ha="center", va="center", fontsize=max(4.0, fontsize - 2),
                color="#222222",
            )

    # Class codes on the outer ring, only on slices wide enough to read.
    outer_wedges = ax.patches[: df.height]
    text_colors = df["text_color"].to_list()
    for pos, (wedge, code, area) in enumerate(zip(outer_wedges, codes, areas_df)):
        if area / total_area < 0.05:
            continue
        mid = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        ax.text(
            0.86 * np.cos(mid), 0.86 * np.sin(mid), str(code),
            ha="center", va="center", rotation=mid * 180 / np.pi,
            rotation_mode="anchor", fontsize=fontsize,
            color=text_colors[pos], fontweight="bold",
        )

    _class_legend(fig, df, spec, lang=lang, ncol=3, y=0.035)
    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _draw_area_treemap(
    df: pl.DataFrame,
    *,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Squarified treemap of class areas (area-proportional rectangles)."""
    from matplotlib.patches import Rectangle

    fig, spec, _ = _st.new_figure(
        style, aspect=0.74, extra_height_mm=24.0, column="single", dpi=dpi,
    )
    has_header = bool(title or subtitle)
    ax = _st.add_axes(
        fig, bottom=0.27, top=_st.figure_top(has_header), left=0.03, right=0.97,
    )

    values = df["area_km2"].to_numpy().astype(float)
    colors = df["color"].to_list()
    codes = [int(v) for v in df["lcz"].to_list()]
    percentages = df["area_perc"].to_numpy().astype(float)
    total_area = float(np.nansum(values)) or 1.0

    order = list(np.argsort(-values))
    rects = _squarify([float(values[i]) for i in order], 0.0, 0.0, 1.0, 1.0)

    body_font = max(5.0, min(9.0, spec.font_size - 2))
    for (rx, ry, rw, rh), idx in zip(rects, order):
        ax.add_patch(Rectangle(
            (rx, ry), rw, rh, facecolor=colors[idx],
            edgecolor="white", linewidth=0.9,
        ))
        if min(rw, rh) < 0.055:
            continue
        label = str(codes[idx])
        if rw * rh >= 0.045 and min(rw, rh) >= 0.10:
            label += f"\n{percentages[idx]:.1f}%"
        ax.text(
            rx + rw / 2.0, ry + rh / 2.0, label,
            ha="center", va="center", fontsize=body_font,
            color=_contrast_color(colors[idx]), linespacing=1.15,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    _class_legend(fig, df, spec, lang=lang, ncol=4, y=0.025)
    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _plot_area_static(
    df: pl.DataFrame,
    plot_type: str,
    *,
    xlab: str,
    ylab: str,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    show_legend: bool,
    lang: str,
    style: str,
    dpi: Optional[int],
    isave: bool,
    save_extension: str,
):
    """Build (and optionally save) the chart with matplotlib/Agg."""
    total = float(df["area_km2"].sum())
    base_title = title or lcz_msg("lcz_area_title", lang)
    base_subtitle = subtitle
    if base_subtitle is None and plot_type in ("pie", "sunburst", "treemap"):
        base_subtitle = f"Total: {total:,.0f} km²"

    with _st.style_context(style):
        if plot_type == "bar":
            fig = _draw_area_bar(
                df, xlab=xlab, ylab=ylab, title=base_title,
                subtitle=base_subtitle, caption=caption, style=style, dpi=dpi,
            )
        elif plot_type == "pie":
            fig = _draw_area_pie(
                df, hole=0.0, show_legend=show_legend, center_text=None,
                title=base_title, subtitle=base_subtitle, caption=caption,
                lang=lang, style=style, dpi=dpi,
            )
        elif plot_type == "donut":
            fig = _draw_area_pie(
                df, hole=0.45, show_legend=show_legend,
                center_text=f"Total\n{total:,.0f} km²",
                title=base_title, subtitle=subtitle, caption=caption,
                lang=lang, style=style, dpi=dpi,
            )
        elif plot_type == "sunburst":
            fig = _draw_area_sunburst(
                df, title=base_title, subtitle=base_subtitle, caption=caption,
                lang=lang, style=style, dpi=dpi,
            )
        else:  # treemap
            fig = _draw_area_treemap(
                df, title=base_title, subtitle=base_subtitle, caption=caption,
                lang=lang, style=style, dpi=dpi,
            )

        _st.save_static(
            fig, f"lcz4r_area_{plot_type}", isave=isave,
            save_extension=save_extension, style=style, dpi=dpi, lang=lang,
        )

    return fig


# ── GeoArrow export ───────────────────────────────────────────────────────────

def _to_geoarrow(df: pl.DataFrame, crs: str = "EPSG:4326") -> Optional[pa.Table]:
    """Export summary to GeoArrow format (representative points)."""
    if not HAS_GEOARROW:
        return None
    if df.height == 0:
        return None

    import geopandas as gpd
    from shapely.geometry import Point

    try:
        gdf = gpd.GeoDataFrame(
            df.to_pandas(),
            geometry=[Point(0, 0) for _ in range(df.height)],
            crs=crs,
        )
        return ga.from_geopandas(gdf)
    except Exception as exc:  # noqa: BLE001 - export is best-effort
        logger.debug("GeoArrow export failed (%s); omitting it.", exc)
        return None


# ── Public entry point ────────────────────────────────────────────────────────

@dataclass
class LCZAreaResult:
    """Return type of lcz_cal_area.

    ``fig`` is a matplotlib ``Figure`` for the default static renderer and a
    Plotly ``go.Figure`` for ``renderer="plotly"``.
    """
    df: pl.DataFrame
    fig: Optional[object] = None
    geoarrow_table: Optional[pa.Table] = None
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


def _read_class_band(
    x: Union[str, Path, "rasterio.io.DatasetReader"],
    band: int = 1,
) -> tuple[np.ndarray, Affine, Optional[str], tuple[int, int]]:
    """Read and validate the LCZ class band from a path or dataset."""
    if isinstance(x, (str, Path)):
        with rasterio.open(str(x)) as src:
            count, transform, crs_obj = src.count, src.transform, src.crs
            height, width = src.height, src.width
            if not 1 <= int(band) <= count:
                raise ValueError(f"band {band} out of range (raster has {count} band(s))")
            classes = src.read(int(band))
    elif hasattr(x, "read") and hasattr(x, "dtypes"):
        count = x.count
        if not 1 <= int(band) <= count:
            raise ValueError(f"band {band} out of range (raster has {count} band(s))")
        classes = x.read(int(band))
        transform = x.transform
        crs_obj = x.crs
        height, width = x.height, x.width
    else:
        raise TypeError(f"x must be a path or rasterio dataset, got {type(x)}")

    if transform is None:
        raise ValueError("Raster has no geotransform; LCZ areas cannot be computed.")

    # A continuous surface (LST, NDVI, ...) has no LCZ classes to tally.
    if classes.dtype.kind == "f":
        finite = classes[np.isfinite(classes)]
        if finite.size == 0:
            raise ValueError("Raster band contains no finite values.")
        if not np.allclose(finite, np.round(finite), atol=1e-6):
            raise ValueError(
                "lcz_cal_area expects an LCZ *class* raster (integer codes 1-17), "
                f"but band {band} holds continuous values "
                f"({float(finite.min()):.3g}..{float(finite.max()):.3g}). "
                "Use lcz_plot_map(..., data_type='continuous') for continuous data."
            )

    crs = str(crs_obj) if crs_obj else None
    return classes, transform, crs, (height, width)


def lcz_cal_area(
    x: Union[str, Path, "rasterio.io.DatasetReader"],
    plot_type: Literal["bar", "pie", "donut", "sunburst", "treemap"] = "bar",
    iplot: bool = True,
    isave: bool = False,
    save_extension: str = "html",
    style: str = "default",
    show_legend: bool = True,
    inclusive: bool = False,
    use_duckdb: bool = True,
    use_geoarrow: bool = True,
    xlab: str = "LCZ code",
    ylab: str = "Area [km²]",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    lang: str = "en",
    band: int = 1,
    dpi: Optional[int] = None,
    renderer: Optional[str] = "static",
) -> Union[LCZAreaResult, pl.DataFrame, go.Figure]:
    """Calculate LCZ areas and render publication-quality charts.

    Parameters
    ----------
    x : str, Path, or rasterio dataset
        Path to a GeoTIFF (or an open dataset) holding LCZ class codes 1-17.
    plot_type : {"bar", "pie", "donut", "sunburst", "treemap"}
        Chart kind.
    iplot : bool
        If False, return only the dataframe.
    isave : bool
        Save chart + CSV.
    save_extension : str
        "png" (default for the static renderer), "pdf", "svg", "tiff"/"tif",
        "jpg"/"jpeg", "webp", or "html" for the interactive Plotly renderer
        ("html" is coerced to "png" for the static renderer). Static output is
        rasterised at >=300 dpi.
    style : str
        Publication style preset: "default", "nature", "science", or
        "generic_bw". Controls font, figure size (mm), DPI, and palette
        used when ``isave`` and ``save_extension`` != "html".
    show_legend : bool
        Show the legend on pie/donut charts (bar/treemap/sunburst label
        their slices directly).
    inclusive : bool
        Use the colorblind-friendly palette.
    use_duckdb : bool
        Use DuckDB for the percentage rollup when installed (a fast NumPy
        aggregation always runs first; DuckDB is optional).
    use_geoarrow : bool
        Generate GeoArrow output when the optional dependency is installed.
    xlab, ylab : str
        Axis labels (bar chart only).
    title, subtitle, caption : str, optional
        Figure annotations.
    lang : {"en", "pt", "es", "zh"}
        Language for class names and default labels.
    band : int
        1-based band holding the LCZ classes (default 1).
    dpi : int, optional
        Resolution for static exports (default >= 300 dpi).
    renderer : {"static", "plotly"}, optional
        "static" (default, also ``None``/``"matplotlib"``/``"agg"``) renders a
        matplotlib ``Figure`` and saves native PNG/PDF/SVG/TIFF without Plotly,
        Kaleido, Chromium or a browser. "plotly" returns the previous
        interactive figure.

    Returns
    -------
    LCZAreaResult, pl.DataFrame, or go.Figure
        ``LCZAreaResult.fig`` is a matplotlib ``Figure`` for the default
        static renderer (Plotly ``go.Figure`` when ``renderer="plotly"``).

    Raises
    ------
    ValueError
        If ``plot_type`` is unknown, the raster contains no LCZ classes 1-17,
        or the selected band holds continuous (non-class) data.
    """
    valid_types = {"bar", "pie", "donut", "sunburst", "treemap"}
    if plot_type not in valid_types:
        raise ValueError(f"plot_type must be one of {valid_types}, got {plot_type!r}")
    # None / "matplotlib" / "agg" -> the native static renderer (default).
    renderer = _st.resolve_renderer(renderer)
    if (save_extension or "").lower().lstrip(".") not in (
        "html", "png", "pdf", "svg", "tiff", "jpg", "jpeg", "webp"
    ):
        raise ValueError(
            f"Unsupported save_extension {save_extension!r} "
            "(use html, png, pdf, svg, tiff, jpg or webp)."
        )
    save_extension = (save_extension or "html").lower().lstrip(".")

    # Read raster and compute cell areas
    classes, transform, crs, shape = _read_class_band(x, band=band)

    # Compute cell areas
    cell_km2 = _cell_area_km2(transform, shape, crs)

    # Filter valid classes
    valid = (classes >= 1) & (classes <= 17)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise ValueError(
            "No LCZ classes (1-17) found in this raster. Check the band/dtype "
            "or confirm the file really is an LCZ class map."
        )
    classes_valid = classes[valid]
    areas_valid = cell_km2[valid]

    # Build summary with chosen backend
    if use_duckdb and HAS_DUCKDB:
        df = _build_summary_duckdb(classes_valid, areas_valid)
    else:
        df = _build_summary_polars(classes_valid, areas_valid)

    # Translate default axis labels when user left English defaults unchanged
    if xlab == "LCZ code":
        xlab = lcz_msg("lcz_code_label", lang)
    if ylab == "Area [km²]":
        ylab = lcz_msg("area_km2_label", lang)

    # Attach metadata (class names in chosen language)
    df = _attach_metadata(df, inclusive, lang=lang)

    if not iplot:
        return df.drop("color", "text_color", "readable")

    # ── Native static renderer (default): matplotlib/Agg, no browser ────────
    if renderer == "static":
        fig = _plot_area_static(
            df, plot_type,
            xlab=xlab, ylab=ylab,
            title=title, subtitle=subtitle, caption=caption,
            show_legend=show_legend, lang=lang, style=style, dpi=dpi,
            isave=isave, save_extension=save_extension,
        )
    else:
        # ── Plotly renderer (opt-in): previous interactive figure ───────────
        plot_fns = {
            "bar": _plot_bar_interactive,
            "pie": _plot_pie_interactive,
            "donut": _plot_donut_interactive,
            "sunburst": _plot_sunburst_interactive,
            "treemap": _plot_treemap_interactive,
        }

        if plot_type == "bar":
            fig = plot_fns[plot_type](df, xlab, ylab, lang=lang)
        else:
            fig = plot_fns[plot_type](df, lang=lang)

        if plot_type in ("pie", "donut") and not show_legend:
            fig.update_layout(showlegend=False)

        # Add annotations
        if title or subtitle:
            title_text = title or lcz_msg("lcz_area_title", lang)
            if subtitle:
                title_text += f"<br><sup>{subtitle}</sup>"
            fig.update_layout(title=dict(text=title_text))
        _add_caption(fig, caption)

        fig = _finalize_publication(
            fig, style=style, isave=isave, save_extension=save_extension,
            filename=f"lcz4r_area_{plot_type}", dpi=dpi, lang=lang,
        )

    # Generate GeoArrow if requested
    geoarrow_table = None
    if use_geoarrow and HAS_GEOARROW:
        geoarrow_table = _to_geoarrow(df, crs or "EPSG:4326")

    df_out = df.drop("color", "text_color", "readable")
    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Save CSV (Polars is faster than Pandas for this)
        csv_path = os.path.join(OUTPUT_DIR, "lcz4r_area_df.csv")
        df_out.write_csv(csv_path)
        logger.info("Saved: %s", os.path.abspath(OUTPUT_DIR))

    return LCZAreaResult(
        df=df_out,
        fig=fig,
        geoarrow_table=geoarrow_table,
        renderer=renderer,
    )


__all__ = ["LCZAreaResult", "lcz_cal_area"]
