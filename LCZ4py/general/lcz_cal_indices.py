"""
lcz_cal_indices.py

Per-LCZ-class statistics and charts for spectral indices
(``lcz_get_indices`` output) — the statistics/visualization counterpart to
``lcz_cal_area.py``, generalized from area/count aggregation to arbitrary
continuous index values.

Two chart kinds, both built from the *same* aggregated statistics table
(no second pass over raw pixels for plotting):
- ``"box"``  — per-class distribution (quartiles/whiskers/mean).
- ``"bar"``  — per-class mean with a 95% CI error bar (mirrors the R
  package's ``plot_lcz_parameter_comparison``).

When multiple indices are requested, both kinds render as small-multiples
(one subplot per index) instead of one chart per call.

Rendering
---------
The **default** renderer is native static matplotlib/Agg (``renderer="static"``,
also ``None``/``"matplotlib"``/``"agg"``): box, bar, scatter, radar,
correlation-heatmap and effect-size charts are drawn with the shared
``_internal/lcz_static`` helpers and saved at >=300 dpi without Plotly,
Kaleido, Chromium or a browser. ``renderer="plotly"`` keeps the previous
interactive figures unchanged.
"""

from __future__ import annotations

import logging
import re
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import polars as pl
import pyarrow as pa
import rasterio
from scipy import stats as spstats

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from LCZ4py.general.lcz_cal_area import _attach_metadata
from LCZ4py.general.lcz_get_indices import LCZIndicesResult
from LCZ4py._internal.lcz_parameters_data import (
    LCZ_COLORS, LCZ_COLORBLIND, PARAM_UNITS,
)
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import get_style, save_plotly_figure
from LCZ4py._internal import lcz_static as _st

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"


def _resolve_workers(n_items: int, max_workers: Optional[int] = None) -> int:
    """Bounded worker count: ``min(8, cpu_count, n_items)`` unless overridden."""
    if max_workers is not None:
        workers = int(max_workers)
    else:
        workers = min(8, os.cpu_count() or 4)
    return max(1, min(workers, max(1, n_items)))


def _axis_label(name: str) -> str:
    """``'NDVI [-1 - 1]'`` — index name plus its documented unit bracket."""
    units = PARAM_UNITS.get(name)
    return f"{name} {units}" if units else name


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_classes(
    lcz_map: Union[str, Path],
) -> tuple[np.ndarray, tuple[int, int], object, object]:
    """Read the LCZ class raster; return (classes, shape, transform, crs)."""
    path = Path(lcz_map)
    if not path.exists():
        raise FileNotFoundError(f"LCZ map not found: {path}")
    with rasterio.open(str(path)) as src:
        classes = src.read(1)
        shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs
    return classes, shape, transform, crs


def _load_indices_stack(
    indices_x: Union[str, Path, LCZIndicesResult],
) -> tuple[np.ndarray, list[str], tuple[int, int], object, object]:
    """Return (array, band names, shape, transform, crs).

    ``transform``/``crs`` are ``None`` for an in-memory ``LCZIndicesResult``
    (alignment is then checked on shape only).
    """
    if isinstance(indices_x, LCZIndicesResult):
        if indices_x.array is not None and indices_x.indices is not None:
            arr = np.asarray(indices_x.array, dtype=np.float32)
            return arr, list(indices_x.indices), arr.shape[1:], None, None
        if not indices_x.path or not os.path.exists(indices_x.path):
            raise ValueError(
                "LCZIndicesResult has no array and no on-disk raster — call "
                "lcz_get_indices with isave=True, or pass a raster path directly."
            )
        indices_x = indices_x.path

    path = Path(indices_x)
    if not path.exists():
        raise FileNotFoundError(f"Indices stack not found: {path}")
    with rasterio.open(str(path)) as src:
        arr = src.read(masked=True).astype(np.float32).filled(np.nan)
        names = [src.descriptions[i] or f"band_{i + 1}" for i in range(src.count)]
        transform = src.transform
        crs = src.crs
    return arr, names, arr.shape[1:], transform, crs


# ── Polars aggregation (generalizes lcz_cal_area's count/area pattern) ───────

def _summarize_one(name: str, arr: np.ndarray, lcz_flat: np.ndarray, valid_lcz: np.ndarray) -> Optional[pl.DataFrame]:
    """Per-LCZ descriptive statistics for one index (or None if all NaN)."""
    val_flat = np.asarray(arr, dtype=np.float64).ravel()
    keep = valid_lcz & np.isfinite(val_flat)
    if not keep.any():
        return None
    return (
        pl.DataFrame({"lcz": lcz_flat[keep], "value": val_flat[keep]})
        .lazy()
        .group_by("lcz")
        .agg([
            pl.len().alias("count"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").median().alias("median"),
            pl.col("value").std().alias("std"),
            pl.col("value").min().alias("min"),
            pl.col("value").max().alias("max"),
            pl.col("value").quantile(0.25).alias("q25"),
            pl.col("value").quantile(0.75).alias("q75"),
        ])
        .with_columns(pl.lit(name).alias("parameter"))
        .sort("lcz")
        .collect()
    )


def _build_indices_summary(
    classes: np.ndarray,
    values: dict[str, np.ndarray],
    max_workers: Optional[int] = None,
) -> pl.DataFrame:
    """Per-(parameter, lcz) descriptive statistics, long format.

    Indices are aggregated in parallel (Polars releases the GIL inside each
    group-by), and the output row order follows ``values`` regardless of the
    worker count.
    """
    empty_schema = {
        "lcz": pl.Int16, "count": pl.UInt32, "mean": pl.Float64,
        "median": pl.Float64, "std": pl.Float64, "min": pl.Float64,
        "max": pl.Float64, "q25": pl.Float64, "q75": pl.Float64,
        "parameter": pl.String,
    }

    lcz_flat = np.asarray(classes).ravel()
    valid_lcz = np.isfinite(lcz_flat) & (lcz_flat >= 1) & (lcz_flat <= 17)
    if not valid_lcz.any():
        return pl.DataFrame(schema=empty_schema)

    # Normalize once: a single int16 copy shared by every index.
    safe_lcz = np.zeros(lcz_flat.shape, dtype=np.int16)
    safe_lcz[valid_lcz] = lcz_flat[valid_lcz].astype(np.int16)

    items = list(values.items())
    workers = _resolve_workers(len(items), max_workers)
    if workers == 1 or len(items) == 1:
        results = [_summarize_one(n, a, safe_lcz, valid_lcz) for n, a in items]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
            results = list(pool.map(
                lambda item: _summarize_one(item[0], item[1], safe_lcz, valid_lcz),
                items,
            ))

    rows = [r for r in results if r is not None]
    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.concat(rows)


def _attach_type(df: pl.DataFrame, lang: str = "en") -> pl.DataFrame:
    """Add the standard Stewart & Oke LCZ 1-10 (built-up) / 11-17 (natural)
    binary grouping column — useful alongside the 17-class breakdown for a
    coarse urban-vs-natural comparison."""
    urban_label = lcz_msg("indices_type_urban", lang)
    natural_label = lcz_msg("indices_type_natural", lang)
    return df.with_columns(
        pl.when(pl.col("lcz") <= 10)
        .then(pl.lit(urban_label))
        .otherwise(pl.lit(natural_label))
        .alias("type")
    )


def _magnitude_bucket(d: float, lang: str = "en") -> str:
    """Standard Cohen's d magnitude convention: <0.2 negligible, 0.2/0.5/0.8
    small/medium/large thresholds."""
    if not np.isfinite(d):
        return "N/A"
    ad = abs(d)
    if ad >= 0.8:
        return lcz_msg("indices_magnitude_large", lang)
    if ad >= 0.5:
        return lcz_msg("indices_magnitude_medium", lang)
    if ad >= 0.2:
        return lcz_msg("indices_magnitude_small", lang)
    return lcz_msg("indices_magnitude_negligible", lang)


def _calculate_cohens_d(
    classes: np.ndarray, values: dict[str, np.ndarray], lang: str = "en",
) -> pl.DataFrame:
    """Cohen's d effect size between the built-up (LCZ 1-10) and natural
    (LCZ 11-17) pixel groups, per index — a real computation (unlike the
    ``lcz_multi_app`` source app, whose Cohen's d is entirely
    ``np.random.uniform(-2, 2)``, never derived from actual pixel values).
    """
    lcz_flat = np.asarray(classes).ravel()
    finite = np.isfinite(lcz_flat)
    urban_mask = finite & (lcz_flat >= 1) & (lcz_flat <= 10)
    natural_mask = finite & (lcz_flat >= 11) & (lcz_flat <= 17)

    rows = []
    for name, arr in values.items():
        val_flat = arr.astype(np.float64).ravel()
        valid = np.isfinite(val_flat)
        urban = val_flat[urban_mask & valid]
        natural = val_flat[natural_mask & valid]
        if len(urban) < 2 or len(natural) < 2:
            continue
        n1, n2 = len(urban), len(natural)
        m1, m2 = float(urban.mean()), float(natural.mean())
        s1, s2 = float(urban.std(ddof=1)), float(natural.std(ddof=1))
        pooled_std = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        d = (m1 - m2) / pooled_std if pooled_std > 0 else np.nan
        rows.append({
            "parameter": name, "cohens_d": d, "magnitude": _magnitude_bucket(d, lang),
            "mean_urban": m1, "mean_natural": m2, "n_urban": n1, "n_natural": n2,
        })

    if not rows:
        return pl.DataFrame(schema={
            "parameter": pl.String, "cohens_d": pl.Float64, "magnitude": pl.String,
            "mean_urban": pl.Float64, "mean_natural": pl.Float64,
            "n_urban": pl.Int64, "n_natural": pl.Int64,
        })
    return pl.DataFrame(rows)


def _add_confidence_interval(df: pl.DataFrame, confidence: float = 0.95) -> pl.DataFrame:
    """Two-tailed t-distribution CI for the mean of each (parameter, lcz) group.

    ponytail: assumes independent pixels (no spatial-autocorrelation
    correction) — a deliberate simplification vs. the R package's
    effective-sample-size adjustment; fine for a quick descriptive chart,
    revisit if the CI needs to be publication-grade.
    """
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if df.height == 0:
        return df.with_columns([
            pl.Series("ci_lower", [], dtype=pl.Float64),
            pl.Series("ci_upper", [], dtype=pl.Float64),
        ])

    n = df["count"].to_numpy().astype(np.float64)
    std = df["std"].fill_null(0.0).to_numpy().astype(np.float64)
    mean = df["mean"].to_numpy().astype(np.float64)
    se = std / np.sqrt(np.maximum(n, 1))
    tcrit = spstats.t.ppf(1 - (1 - confidence) / 2, np.maximum(n - 1, 1))
    half = tcrit * se
    return df.with_columns([
        pl.Series("ci_lower", mean - half),
        pl.Series("ci_upper", mean + half),
    ])


# ── Chart builders (small-multiples across parameters) ───────────────────────

def _subplot_grid(n: int) -> tuple[int, int]:
    cols = min(3, n)
    rows = math.ceil(n / cols)
    return rows, cols


def _plot_box_interactive(df: pl.DataFrame, params: list[str], lang: str = "en") -> go.Figure:
    """Precomputed-statistics box plot per LCZ class, one subplot per parameter."""
    rows, cols = _subplot_grid(len(params))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=params)

    for i, param in enumerate(params):
        sub = df.filter(pl.col("parameter") == param)
        r, c = i // cols + 1, i % cols + 1
        # One trace per LCZ class: go.Box's precomputed-statistics mode only
        # accepts a single marker color per trace, so per-class coloring
        # (the official LCZ palette) requires per-class traces, not a single
        # trace with a color list (unlike go.Bar, which does allow that).
        for row in sub.iter_rows(named=True):
            fig.add_trace(
                go.Box(
                    x=[str(row["lcz"])],
                    q1=[row["q25"]], median=[row["median"]], q3=[row["q75"]],
                    lowerfence=[row["min"]], upperfence=[row["max"]],
                    mean=[row["mean"]], sd=[row["std"]],
                    marker_color=row["color"],
                    line=dict(color="black", width=1),
                    boxmean=True,
                    name=row["lcz_name"],
                    customdata=[[row["lcz_name"], row["count"]]],
                    hovertemplate=(
                        "<b>LCZ %{x}</b> — %{customdata[0]}<br>"
                        "median: %{median:.3f}<br>"
                        "q1-q3: %{q1:.3f} - %{q3:.3f}<br>"
                        "n=%{customdata[1]}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=r, col=c,
            )
        fig.update_yaxes(title=dict(text=_axis_label(param), font=dict(size=11)),
                         row=r, col=c)

    fig.update_layout(
        title=dict(
            text=lcz_msg("indices_stats_title", lang),
            font=dict(size=18), x=0.5, xanchor="center",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=90, b=60),
        width=420 * cols,
        height=380 * rows,
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    fig.update_xaxes(title=dict(text=lcz_msg("lcz_code_label", lang), font=dict(size=12)))
    return fig


def _plot_bar_interactive(df: pl.DataFrame, params: list[str], lang: str = "en") -> go.Figure:
    """Mean ± 95% CI bar chart per LCZ class, one subplot per parameter."""
    rows, cols = _subplot_grid(len(params))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=params)

    for i, param in enumerate(params):
        sub = df.filter(pl.col("parameter") == param)
        r, c = i // cols + 1, i % cols + 1
        err_plus = (sub["ci_upper"] - sub["mean"]).to_list()
        err_minus = (sub["mean"] - sub["ci_lower"]).to_list()
        fig.add_trace(
            go.Bar(
                x=sub["lcz"].cast(str),
                y=sub["mean"],
                marker_color=sub["color"].to_list(),
                marker_line_color="black",
                marker_line_width=0.5,
                error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus),
                customdata=sub.select(["lcz_name", "count"]),
                hovertemplate=(
                    "<b>LCZ %{x}</b> — %{customdata[0]}<br>"
                    "mean: %{y:.3f}<br>"
                    "n=%{customdata[1]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=r, col=c,
        )
        fig.update_yaxes(title=dict(text=_axis_label(param), font=dict(size=11)),
                         row=r, col=c)

    fig.update_layout(
        title=dict(
            text=lcz_msg("indices_stats_title", lang),
            font=dict(size=18), x=0.5, xanchor="center",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=90, b=60),
        width=420 * cols,
        height=380 * rows,
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    fig.update_xaxes(title=dict(text=lcz_msg("lcz_code_label", lang), font=dict(size=12)))
    return fig


_MAGNITUDE_COLORS: dict[str, str] = {
    "N/A": "#999999",
}
# Cividis-sampled, negligible -> large: monotonic in lightness and
# colorblind-safe (unlike a red-green ramp).
_MAGNITUDE_COLOR_ORDER = ["#d9d9d9", "#a59c74", "#575d6d", "#00204d"]


def _plot_effect_size_interactive(cohens_d_df: pl.DataFrame, lang: str = "en") -> go.Figure:
    """Horizontal bar of Cohen's d per index, sorted by |d| descending —
    doubles as a Top-N "which indices best separate built-up from natural"
    ranking, with reference lines at the conventional 0.2/0.5/0.8 thresholds.
    """
    # Magnitude labels are localized strings from _magnitude_bucket; map them
    # back to a fixed negligible->large color order regardless of language.
    magnitude_order = [
        lcz_msg("indices_magnitude_negligible", lang), lcz_msg("indices_magnitude_small", lang),
        lcz_msg("indices_magnitude_medium", lang), lcz_msg("indices_magnitude_large", lang),
    ]
    color_map = dict(zip(magnitude_order, _MAGNITUDE_COLOR_ORDER))

    df_sorted = cohens_d_df.filter(pl.col("cohens_d").is_not_nan()).sort(
        pl.col("cohens_d").abs(), descending=True,
    )
    colors = [color_map.get(m, "#999999") for m in df_sorted["magnitude"]]

    fig = go.Figure(go.Bar(
        x=df_sorted["cohens_d"], y=df_sorted["parameter"], orientation="h",
        marker_color=colors, marker_line_color="black", marker_line_width=0.5,
        customdata=df_sorted.select(["magnitude", "n_urban", "n_natural"]),
        hovertemplate=(
            "<b>%{y}</b><br>Cohen's d: %{x:.3f}<br>"
            "%{customdata[0]}<br>n_urban=%{customdata[1]} n_natural=%{customdata[2]}"
            "<extra></extra>"
        ),
    ))
    for thr in (0.2, 0.5, 0.8):
        fig.add_vline(x=thr, line_dash="dot", line_color="gray")
        fig.add_vline(x=-thr, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="black")

    fig.update_layout(
        title=dict(
            text=lcz_msg("indices_effect_size_title", lang),
            font=dict(size=20), x=0.5, xanchor="center",
        ),
        xaxis=dict(title=dict(text="Cohen's d", font=dict(size=14))),
        yaxis=dict(autorange="reversed"),  # largest |d| at top
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        margin=dict(l=140, r=40, t=60, b=60),
        width=900, height=max(400, 32 * len(df_sorted) + 150),
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    return fig


def _plot_scatter_interactive(
    classes: np.ndarray, values: dict[str, np.ndarray],
    index_x: str, index_y: str, size_by: Optional[str] = None,
    inclusive: bool = False, lang: str = "en",
    max_points: int = 20_000, seed: int = 0,
) -> go.Figure:
    """Per-pixel scatter of two indices, colored by LCZ class (official
    palette), optionally sized by a third index. Subsampled for large
    rasters — a scatter with millions of points bloats the HTML output and
    doesn't convey more information than a representative sample.
    """
    lcz_flat = np.asarray(classes).ravel()
    x = values[index_x].astype(np.float64).ravel()
    y = values[index_y].astype(np.float64).ravel()
    valid = (
        np.isfinite(lcz_flat) & (lcz_flat >= 1) & (lcz_flat <= 17)
        & np.isfinite(x) & np.isfinite(y)
    )
    if size_by:
        s_all = values[size_by].astype(np.float64).ravel()
        valid &= np.isfinite(s_all)

    idx = np.where(valid)[0]
    if len(idx) == 0:
        raise ValueError(
            f"No valid pixels for the scatter: {index_x}/{index_y} are NaN "
            "wherever LCZ classes 1-17 are present."
        )
    if len(idx) > max_points:
        idx = np.random.default_rng(seed).choice(idx, size=max_points, replace=False)

    data = {"lcz": lcz_flat[idx].astype(np.int16), "x": x[idx], "y": y[idx]}
    if size_by:
        data["size"] = s_all[idx]
    df = pl.DataFrame(data)
    df = _attach_metadata(df, inclusive, lang=lang)

    fig = go.Figure()
    for lcz_val in sorted(df["lcz"].unique().to_list()):
        sub = df.filter(pl.col("lcz") == lcz_val)
        lcz_name = sub["lcz_name"].to_list()[0]
        marker = dict(color=sub["color"].to_list()[0], size=7,
                      line=dict(color="black", width=0.3))
        hovertemplate = (
            f"<b>LCZ {lcz_val}</b> — {lcz_name}<br>"
            f"{index_x}: %{{x:.3f}}<br>{index_y}: %{{y:.3f}}"
        )
        customdata = None
        if size_by:
            sizes = sub["size"].to_numpy()
            lo, hi = np.nanmin(sizes), np.nanmax(sizes)
            marker["size"] = 6 + 20 * (sizes - lo) / (hi - lo + 1e-9)
            hovertemplate += f"<br>{size_by}: %{{customdata:.3f}}"
            customdata = sizes
        fig.add_trace(go.Scattergl(
            x=sub["x"], y=sub["y"], mode="markers", marker=marker,
            name=lcz_name, customdata=customdata,
            hovertemplate=hovertemplate + "<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=lcz_msg("indices_scatter_title", lang, x=index_x, y=index_y),
            font=dict(size=20), x=0.5, xanchor="center",
        ),
        xaxis=dict(title=dict(text=index_x, font=dict(size=14))),
        yaxis=dict(title=dict(text=index_y, font=dict(size=14))),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        legend=dict(title=dict(text=lcz_msg("lcz_class_legend", lang))),
        margin=dict(l=70, r=40, t=60, b=60),
        width=900, height=700,
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    return fig


def _plot_radar_interactive(
    df: pl.DataFrame, params: list[str], lcz_classes: Optional[list[int]] = None,
    normalize: bool = True, lang: str = "en",
) -> go.Figure:
    """One radar/polar trace per LCZ class, axes = requested indices, values
    = per-class mean (reuses the same aggregation as box/bar — no re-read of
    the rasters). ``normalize`` min-max scales each axis across the plotted
    classes so indices with very different natural ranges (e.g. NDVI's
    -1..1 vs RI's 10s-1000s) stay visually comparable on the same chart.
    """
    present = sorted(df["lcz"].unique().to_list())
    classes_to_plot = [c for c in (lcz_classes or present) if c in present]
    if not classes_to_plot:
        raise ValueError(f"None of the requested lcz_classes are present. Available: {present}")

    sub_all = df.filter(pl.col("parameter").is_in(params) & pl.col("lcz").is_in(classes_to_plot))
    bounds: dict[str, tuple[float, float]] = {}
    if normalize:
        bounds_df = sub_all.group_by("parameter").agg([
            pl.col("mean").min().alias("pmin"), pl.col("mean").max().alias("pmax"),
        ])
        bounds = {r["parameter"]: (r["pmin"], r["pmax"]) for r in bounds_df.iter_rows(named=True)}

    fig = go.Figure()
    for lcz_val in classes_to_plot:
        sub = sub_all.filter(pl.col("lcz") == lcz_val)
        by_param = {row["parameter"]: row for row in sub.iter_rows(named=True)}
        theta = [p for p in params if p in by_param]
        if not theta:
            continue
        r_values = []
        for p in theta:
            v = by_param[p]["mean"]
            if normalize:
                lo, hi = bounds[p]
                v = 0.5 if hi == lo else (v - lo) / (hi - lo)
            r_values.append(v)
        color = by_param[theta[0]]["color"]
        name = by_param[theta[0]]["lcz_name"]
        fig.add_trace(go.Scatterpolar(
            r=r_values + [r_values[0]], theta=theta + [theta[0]],
            fill="toself", name=f"LCZ {lcz_val} — {name}",
            line=dict(color=color), opacity=0.6,
        ))

    fig.update_layout(
        title=dict(
            text=lcz_msg("indices_radar_title", lang),
            font=dict(size=20), x=0.5, xanchor="center",
        ),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1] if normalize else None)),
        paper_bgcolor="white",
        width=800, height=800,
        showlegend=True,
    )
    return fig


def _plot_correlation_heatmap(
    classes: np.ndarray, values: dict[str, np.ndarray], params: list[str], lang: str = "en",
) -> go.Figure:
    """Pearson correlation matrix between the requested indices, over every
    valid (LCZ 1-17) pixel — a quick way to spot redundant indices among the
    56 available (e.g. LSWI/SOIL_MOISTURE/NDMI are literally the same
    formula and will show r=1.0).
    """
    lcz_flat = np.asarray(classes).ravel()
    valid = np.isfinite(lcz_flat) & (lcz_flat >= 1) & (lcz_flat <= 17)
    cols = {}
    for p in params:
        v = np.asarray(values[p], dtype=np.float64).ravel()
        valid = valid & np.isfinite(v)
        cols[p] = v
    if len(params) < 2:
        raise ValueError(
            "plot_type='correlation' needs at least two indices to correlate."
        )
    if not valid.any():
        raise ValueError(
            "No valid (LCZ 1-17, finite) pixels to correlate — nothing to plot."
        )
    df = pl.DataFrame(cols).filter(pl.Series(valid))

    corr = df.corr()
    z = corr.to_numpy()

    fig = go.Figure(go.Heatmap(
        z=z, x=params, y=params, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title=dict(text="Pearson r (dimensionless)")),
        hovertemplate="%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=lcz_msg("indices_correlation_title", lang),
            font=dict(size=18), x=0.5, xanchor="center",
        ),
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="white",
        width=max(500, 60 * len(params)), height=max(500, 60 * len(params)),
        margin=dict(l=100, r=40, t=60, b=100),
    )
    return fig


def _to_geoarrow(df: pl.DataFrame) -> Optional[pa.Table]:
    """Statistics table as an Arrow table (no geometry -> plain Arrow).

    Uses Polars' Arrow bridge, so the optional ``geoarrow`` package is not
    needed for this (non-spatial) table.
    """
    try:
        return df.to_arrow()
    except Exception as exc:  # noqa: BLE001 - pyarrow/polars interop edge case
        logger.warning("Could not convert statistics table to Arrow: %s", exc)
        return None


# ── Native static charts (matplotlib/Agg — no Plotly/Kaleido/browser) ────────

#: Marker cycle giving the scatter a non-colour encoding for every class.
_MARKER_CYCLE = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p")

#: ``lcz_msg`` leaves rich markup such as ``{val NDVI}`` for rich printers;
#: matplotlib titles need the plain text.
_RICH_MARKUP_RE = re.compile(r"\{(?:val|arg|fn|path|url)\s+([^{}]*)\}")


def _plain(text: str) -> str:
    """Unwrap ``lcz_msg``'s ``{val ...}``-style markup for plain-text titles."""
    return _RICH_MARKUP_RE.sub(r"\1", text or "")


def _finite_or(value: object, fallback: float) -> float:
    """``float(value)`` when finite, else ``fallback``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return out if np.isfinite(out) else fallback


def _style_axes(ax, spec, *, grid_axis: str = "y", rotate: float = 0.0) -> None:
    """Consistent panel styling: small ticks, light grid, open frame."""
    ax.tick_params(axis="both", labelsize=max(5.5, spec.font_size - 2), length=2.5, width=0.6)
    if rotate:
        for label in ax.get_xticklabels():
            label.set_rotation(rotate)
            label.set_ha("right")
            label.set_rotation_mode("anchor")
    ax.grid(axis=grid_axis, color="#e8e8e8", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _legend_entries(df: pl.DataFrame) -> list:
    """``(colour, "code: name")`` pairs in class order."""
    return list(zip(df["color"].to_list(), df["readable"].to_list()))


def _class_legend(
    fig,
    entries: list,
    spec,
    *,
    lang: str = "en",
    ncol: int = 3,
    y: float = 0.02,
    markers: bool = False,
) -> None:
    """Class legend below the panel(s); patches for bars, markers for scatter."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not entries:
        return None
    fontsize = max(5.0, spec.font_size - 2)
    if markers:
        handles = [
            Line2D([], [], linestyle="", marker=_MARKER_CYCLE[i % len(_MARKER_CYCLE)],
                   markersize=4.5, markerfacecolor=color, markeredgecolor="black",
                   markeredgewidth=0.3, label=label)
            for i, (color, label) in enumerate(entries)
        ]
    else:
        handles = [
            Patch(facecolor=color, edgecolor="black", linewidth=0.3, label=label)
            for color, label in entries
        ]
    return fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=min(int(ncol), max(2, len(handles))),
        frameon=False,
        fontsize=fontsize,
        handlelength=1.1 if not markers else 0.9,
        columnspacing=1.2,
        handletextpad=0.4,
        title=lcz_msg("lcz_class_legend", lang),
        title_fontsize=fontsize,
    )


def _panel_layout(n_panels: int, style: str, dpi: Optional[int]):
    """Figure + gridspec sized for ``n_panels`` small multiples."""
    rows, cols = _subplot_grid(n_panels)
    column = "single" if cols == 1 else "double"
    height_mm = 64.0 * rows + 30.0
    fig, spec, _ = _st.new_figure(
        style, column=column, height_mm=height_mm, dpi=dpi,
    )
    gs = fig.add_gridspec(
        rows, cols,
        left=0.085, right=0.985,
        top=_st.figure_top(True), bottom=0.16,
        hspace=0.55 if rows > 1 else 0.0,
        wspace=0.36 if cols > 1 else 0.0,
    )
    return fig, spec, gs, (rows, cols)


def _draw_box_static(
    df: pl.DataFrame,
    params: list[str],
    *,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Precomputed-statistics box plot per LCZ class (one panel per index)."""
    fig, spec, gs, (rows, cols) = _panel_layout(len(params), style, dpi)
    multi = cols > 1

    for i, param in enumerate(params):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        sub = df.filter(pl.col("parameter") == param)
        stats_list, colors = [], []
        for row in sub.iter_rows(named=True):
            q1 = _finite_or(row["q25"], np.nan)
            q3 = _finite_or(row["q75"], np.nan)
            med = _finite_or(row["median"], np.nan)
            if not (np.isfinite(q1) and np.isfinite(q3) and np.isfinite(med)):
                continue
            lo = min(_finite_or(row["min"], q1), q1)
            hi = max(_finite_or(row["max"], q3), q3)
            stats_list.append(dict(
                label=str(int(row["lcz"])), med=med, q1=q1, q3=q3,
                whislo=lo, whishi=hi, fliers=[],
                mean=_finite_or(row["mean"], med),
            ))
            colors.append(row["color"])

        if not stats_list:  # pragma: no cover - no finite stats for this index
            ax.set_visible(False)
            continue

        bxp = ax.bxp(
            stats_list, showmeans=True, patch_artist=True, widths=0.62,
            boxprops=dict(linewidth=0.5, edgecolor="black"),
            medianprops=dict(color="black", linewidth=0.8),
            whiskerprops=dict(linewidth=0.5),
            capprops=dict(linewidth=0.5),
            meanprops=dict(marker="o", markersize=2.6, markerfacecolor="white",
                           markeredgecolor="black", markeredgewidth=0.5),
        )
        for patch, color in zip(bxp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title(param, fontsize=max(6.0, spec.font_size), pad=3)
        ax.set_xlabel(lcz_msg("lcz_code_label", lang))
        ax.set_ylabel(_axis_label(param))
        _style_axes(ax, spec, rotate=90 if multi else 0)

    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _draw_bar_static(
    df: pl.DataFrame,
    params: list[str],
    *,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Per-class mean with a 95% CI error bar (one panel per index)."""
    fig, spec, gs, (rows, cols) = _panel_layout(len(params), style, dpi)
    multi = cols > 1

    for i, param in enumerate(params):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        sub = df.filter(pl.col("parameter") == param)
        codes = [str(int(v)) for v in sub["lcz"].to_list()]
        mean = np.nan_to_num(sub["mean"].to_numpy().astype(float), nan=0.0)
        lower = np.nan_to_num(sub["ci_lower"].to_numpy().astype(float), nan=0.0)
        upper = np.nan_to_num(sub["ci_upper"].to_numpy().astype(float), nan=0.0)
        yerr = np.vstack([np.maximum(mean - lower, 0.0), np.maximum(upper - mean, 0.0)])

        x = np.arange(len(codes), dtype=float)
        ax.bar(
            x, mean, yerr=yerr, width=0.74,
            color=sub["color"].to_list(), edgecolor="black", linewidth=0.4,
            error_kw=dict(elinewidth=0.6, capsize=1.6, ecolor="#333333"),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(codes)
        ax.set_title(param, fontsize=max(6.0, spec.font_size), pad=3)
        ax.set_xlabel(lcz_msg("lcz_code_label", lang))
        ax.set_ylabel(_axis_label(param))
        _style_axes(ax, spec, rotate=90 if multi else 0)

    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _scatter_points(
    classes: np.ndarray,
    values: dict[str, np.ndarray],
    index_x: str,
    index_y: str,
    size_by: Optional[str] = None,
    *,
    max_points: int = 20_000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Valid (LCZ 1-17) per-pixel samples for the scatter, subsampled."""
    lcz_flat = np.asarray(classes).ravel()
    x = np.asarray(values[index_x], dtype=np.float64).ravel()
    y = np.asarray(values[index_y], dtype=np.float64).ravel()
    valid = (
        np.isfinite(lcz_flat) & (lcz_flat >= 1) & (lcz_flat <= 17)
        & np.isfinite(x) & np.isfinite(y)
    )
    sizes = None
    if size_by:
        sizes = np.asarray(values[size_by], dtype=np.float64).ravel()
        valid = valid & np.isfinite(sizes)

    idx = np.where(valid)[0]
    if idx.size == 0:
        raise ValueError(
            f"No valid pixels for the scatter: {index_x}/{index_y} are NaN "
            "wherever LCZ classes 1-17 are present."
        )
    if idx.size > max_points:
        idx = np.random.default_rng(seed).choice(idx, size=max_points, replace=False)

    out = {"lcz": lcz_flat[idx].astype(int), "x": x[idx], "y": y[idx]}
    if sizes is not None:
        out["size"] = sizes[idx]
    return out


def _draw_scatter_static(
    classes: np.ndarray,
    values: dict[str, np.ndarray],
    index_x: str,
    index_y: str,
    *,
    size_by: Optional[str] = None,
    inclusive: bool = False,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Per-pixel scatter of two indices coloured by LCZ class."""
    from LCZ4py._internal.lcz_parameters_data import get_lcz_names

    samples = _scatter_points(classes, values, index_x, index_y, size_by)
    names = get_lcz_names(lang)
    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    present = sorted(int(v) for v in np.unique(samples["lcz"]))

    fig, spec, _ = _st.new_figure(
        style, aspect=0.72, extra_height_mm=36.0, column="single", dpi=dpi,
    )
    ax = _st.add_axes(
        fig, bottom=0.39, top=_st.figure_top(True), left=0.14, right=0.97,
    )

    sizes = samples.get("size")
    marker_area = None
    if sizes is not None and sizes.size:
        lo, hi = float(np.nanmin(sizes)), float(np.nanmax(sizes))
        marker_area = 8.0 + 26.0 * (sizes - lo) / (hi - lo + 1e-9)

    for i, lcz_val in enumerate(present):
        mask = samples["lcz"] == lcz_val
        ax.scatter(
            samples["x"][mask], samples["y"][mask],
            s=marker_area[mask] if marker_area is not None else 10.0,
            c=colors[lcz_val - 1], marker=_MARKER_CYCLE[i % len(_MARKER_CYCLE)],
            edgecolors="black", linewidths=0.2, alpha=0.85,
        )

    ax.set_xlabel(index_x)
    ax.set_ylabel(index_y)
    if size_by and marker_area is not None:
        ax.text(
            0.99, 0.01, f"marker size ∝ {size_by}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=max(5.0, spec.font_size - 3),
            color="#444444",
        )
    _style_axes(ax, spec, grid_axis="both")

    _class_legend(
        fig,
        [(colors[c - 1], f"{c}: {names[c - 1]}") for c in present],
        spec, lang=lang, ncol=3, y=0.02, markers=True,
    )
    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _draw_radar_static(
    df: pl.DataFrame,
    params: list[str],
    *,
    lcz_classes: Optional[list[int]] = None,
    normalize: bool = True,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """One polar trace per LCZ class, axes = requested indices."""
    from matplotlib.lines import Line2D

    present = sorted(int(v) for v in df["lcz"].unique().to_list())
    classes_to_plot = [c for c in (lcz_classes or present) if c in present]
    if not classes_to_plot:
        raise ValueError(
            f"None of the requested lcz_classes are present. Available: {present}"
        )

    sub_all = df.filter(
        pl.col("parameter").is_in(params) & pl.col("lcz").is_in(classes_to_plot)
    )
    bounds: dict[str, tuple[float, float]] = {}
    if normalize:
        bounds_df = sub_all.group_by("parameter").agg([
            pl.col("mean").min().alias("pmin"),
            pl.col("mean").max().alias("pmax"),
        ])
        bounds = {
            r["parameter"]: (r["pmin"], r["pmax"])
            for r in bounds_df.iter_rows(named=True)
        }

    fig, spec, _ = _st.new_figure(
        style, aspect=0.95, extra_height_mm=28.0, column="single", dpi=dpi,
    )
    ax = fig.add_axes([0.16, 0.26, 0.68, 0.60], projection="polar")

    theta = np.linspace(0.0, 2.0 * np.pi, len(params), endpoint=False)
    angles = np.append(theta, theta[0])
    handles = []
    vmin, vmax = np.inf, -np.inf

    for lcz_val in classes_to_plot:
        sub = sub_all.filter(pl.col("lcz") == lcz_val)
        by_param = {r["parameter"]: r for r in sub.iter_rows(named=True)}
        values, color, name = [], None, None
        for param in params:
            row = by_param.get(param)
            if row is None:
                values.append(np.nan)
                continue
            if color is None:
                color, name = row["color"], row["lcz_name"]
            value = _finite_or(row["mean"], np.nan)
            if normalize and param in bounds:
                lo, hi = bounds[param]
                value = 0.5 if hi == lo else (value - lo) / (hi - lo)
            values.append(value)

        if color is None:  # pragma: no cover - class has no plotted parameter
            continue
        radii = np.asarray(values, dtype=float)
        finite = radii[np.isfinite(radii)]
        if finite.size:
            vmin = min(vmin, float(finite.min()))
            vmax = max(vmax, float(finite.max()))
        closed = np.append(radii, radii[0])
        ax.plot(angles, closed, color=color, linewidth=1.4)
        ax.fill(angles, closed, color=color, alpha=0.10)
        handles.append(Line2D([], [], color=color, linewidth=1.6,
                              label=f"{lcz_val}: {name}"))

    ax.set_xticks(theta)
    ax.set_xticklabels(params, fontsize=max(5.0, spec.font_size - 3))
    ax.set_rlabel_position(100.0)
    ax.tick_params(axis="y", labelsize=max(4.5, spec.font_size - 4))
    if normalize:
        ax.set_ylim(0.0, 1.0)
    elif np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
        ax.set_ylim(min(0.0, vmin), vmax * 1.05 if vmax > 0 else vmax * 0.95 + 0.05)

    fontsize = max(5.0, spec.font_size - 2)
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=min(3, max(2, len(handles))), frameon=False, fontsize=fontsize,
        handlelength=1.1, columnspacing=1.2, handletextpad=0.4,
        title=lcz_msg("lcz_class_legend", lang), title_fontsize=fontsize,
    )
    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _draw_correlation_static(
    classes: np.ndarray,
    values: dict[str, np.ndarray],
    params: list[str],
    *,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Annotated Pearson correlation matrix (diverging, colourblind-safe)."""
    lcz_flat = np.asarray(classes).ravel()
    valid = np.isfinite(lcz_flat) & (lcz_flat >= 1) & (lcz_flat <= 17)
    cols = {}
    for param in params:
        v = np.asarray(values[param], dtype=np.float64).ravel()
        valid = valid & np.isfinite(v)
        cols[param] = v
    if len(params) < 2:
        raise ValueError(
            "plot_type='correlation' needs at least two indices to correlate."
        )
    if not valid.any():
        raise ValueError(
            "No valid (LCZ 1-17, finite) pixels to correlate — nothing to plot."
        )
    corr = pl.DataFrame(cols).filter(pl.Series(valid)).corr()
    matrix = corr.to_numpy()

    n = len(params)
    width_mm = max(95.0, 20.0 * n + max(50.0, 6.0 * max(len(p) for p in params)))
    height_mm = max(95.0, 20.0 * n + 45.0)
    fig, spec, _ = _st.new_figure(
        style, width_mm=width_mm, height_mm=height_mm, dpi=dpi,
    )
    ax = _st.add_axes(
        fig, bottom=0.30, top=_st.figure_top(True),
        left=min(0.42, 0.16 + 0.012 * max(len(p) for p in params)), right=0.95,
    )

    image = ax.imshow(
        matrix, cmap=_st.mpl_cmap("RdBu_r"), vmin=-1.0, vmax=1.0, aspect="auto",
    )
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(params, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(params)
    ax.tick_params(labelsize=max(5.0, spec.font_size - 2), length=2.0, width=0.5)

    cell_font = max(4.5, min(8.0, spec.font_size - 2))
    for i in range(n):
        for j in range(n):
            value = float(matrix[i, j])
            if not np.isfinite(value):
                continue
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center", fontsize=cell_font,
                color="white" if abs(value) > 0.6 else "black",
            )

    _st.add_colorbar(
        fig, ax, image, units="Pearson r (dimensionless)",
        fontsize=max(5.5, spec.font_size - 2),
    )
    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _draw_effect_size_static(
    cohens_d_df: pl.DataFrame,
    *,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Ranked horizontal bar of Cohen's d per index, with magnitude brackets."""
    from matplotlib.patches import Patch

    magnitude_order = [
        lcz_msg("indices_magnitude_negligible", lang),
        lcz_msg("indices_magnitude_small", lang),
        lcz_msg("indices_magnitude_medium", lang),
        lcz_msg("indices_magnitude_large", lang),
    ]
    color_map = dict(zip(magnitude_order, _MAGNITUDE_COLOR_ORDER))

    df_sorted = cohens_d_df.filter(pl.col("cohens_d").is_not_nan()).sort(
        pl.col("cohens_d").abs(), descending=True,
    )
    if df_sorted.height == 0:  # pragma: no cover - guarded by the caller
        raise ValueError("No finite Cohen's d values to plot.")

    n = df_sorted.height
    height_mm = min(36.0 + 8.0 * n, 200.0)
    fig, spec, _ = _st.new_figure(
        style, aspect=height_mm / 120.0, column="single", dpi=dpi,
    )
    ax = _st.add_axes(
        fig, bottom=0.17, top=_st.figure_top(True), left=0.34, right=0.96,
    )

    y = np.arange(n, dtype=float)
    values = df_sorted["cohens_d"].to_numpy().astype(float)
    colors = [color_map.get(m, "#999999") for m in df_sorted["magnitude"].to_list()]
    ax.barh(y, values, height=0.68, color=colors, edgecolor="black", linewidth=0.4)

    for thr in (0.2, 0.5, 0.8):
        ax.axvline(thr, linestyle=":", color="gray", linewidth=0.6, zorder=0)
        ax.axvline(-thr, linestyle=":", color="gray", linewidth=0.6, zorder=0)
    ax.axvline(0.0, color="black", linewidth=0.8, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["parameter"].to_list())
    ax.set_xlabel("Cohen's d")
    ax.invert_yaxis()  # largest |d| at the top
    _style_axes(ax, spec, grid_axis="x")

    present_magnitudes = [m for m in magnitude_order if m in set(df_sorted["magnitude"].to_list())]
    if present_magnitudes:
        ax.legend(
            handles=[Patch(facecolor=color_map[m], edgecolor="black",
                           linewidth=0.3, label=m) for m in present_magnitudes],
            loc="best", frameon=False,
            fontsize=max(5.0, spec.font_size - 3), title="|d|",
            title_fontsize=max(5.0, spec.font_size - 3),
        )

    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _static_figures(
    df: pl.DataFrame,
    values: dict[str, np.ndarray],
    classes: np.ndarray,
    plot_type: str,
    selected: list[str],
    cohens_d_df: Optional[pl.DataFrame],
    *,
    index_x: Optional[str],
    index_y: Optional[str],
    size_by: Optional[str],
    lcz_classes: Optional[list[int]],
    normalize_radar: bool,
    inclusive: bool,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Build the requested chart(s) with matplotlib/Agg.

    Returns the same shape as the Plotly path: a single figure, or a dict of
    figures for ``plot_type="both"``/``"all"``.
    """
    common = dict(subtitle=subtitle, caption=caption, lang=lang, style=style, dpi=dpi)

    def _title(default_key: str, **fmt) -> str:
        return _plain(title or lcz_msg(default_key, lang, **fmt))

    with _st.style_context(style):
        if plot_type == "box":
            return _draw_box_static(
                df, selected, title=_title("indices_stats_title"), **common,
            )
        if plot_type == "bar":
            return _draw_bar_static(
                df, selected, title=_title("indices_stats_title"), **common,
            )
        if plot_type == "both":
            return {
                "box": _draw_box_static(
                    df, selected, title=_title("indices_stats_title"), **common,
                ),
                "bar": _draw_bar_static(
                    df, selected, title=_title("indices_stats_title"), **common,
                ),
            }
        if plot_type == "scatter":
            return _draw_scatter_static(
                classes, values, index_x, index_y, size_by=size_by,
                inclusive=inclusive,
                title=_title("indices_scatter_title", x=index_x, y=index_y),
                **common,
            )
        if plot_type == "radar":
            return _draw_radar_static(
                df, selected, lcz_classes=lcz_classes, normalize=normalize_radar,
                title=_title("indices_radar_title"), **common,
            )
        if plot_type == "correlation":
            return _draw_correlation_static(
                classes, values, selected,
                title=_title("indices_correlation_title"), **common,
            )
        if plot_type == "effect_size":
            return _draw_effect_size_static(
                cohens_d_df, title=_title("indices_effect_size_title"), **common,
            )

        # "all": every applicable figure, mirroring the Plotly branch.
        figs = {
            "box": _draw_box_static(
                df, selected, title=_title("indices_stats_title"), **common,
            ),
            "bar": _draw_bar_static(
                df, selected, title=_title("indices_stats_title"), **common,
            ),
            "radar": _draw_radar_static(
                df, selected, lcz_classes=lcz_classes, normalize=normalize_radar,
                title=_title("indices_radar_title"), **common,
            ),
            "correlation": _draw_correlation_static(
                classes, values, selected,
                title=_title("indices_correlation_title"), **common,
            ),
            "effect_size": _draw_effect_size_static(
                cohens_d_df, title=_title("indices_effect_size_title"), **common,
            ),
        }
        if index_x and index_y:
            figs["scatter"] = _draw_scatter_static(
                classes, values, index_x, index_y, size_by=size_by,
                inclusive=inclusive,
                title=_title("indices_scatter_title", x=index_x, y=index_y),
                **common,
            )
        else:
            logger.info("plot_type='all': skipping scatter (index_x/index_y not given).")
        return figs


# ── Public entry point ────────────────────────────────────────────────────────

@dataclass
class LCZIndicesStatsResult:
    """Return type of lcz_cal_indices.

    ``fig`` is a matplotlib ``Figure`` (or a ``dict`` of them) for the default
    static renderer, and a Plotly ``go.Figure`` (or a dict of them) when
    ``renderer="plotly"``.
    """
    df: pl.DataFrame
    fig: Optional[object] = None
    geoarrow_table: Optional[pa.Table] = None
    cohens_d: Optional[pl.DataFrame] = None
    renderer: str = "static"

    def show(self) -> None:
        """Display the figure(s) in the active frontend (matplotlib or Plotly)."""
        figs = self.fig
        if figs is None:
            return
        if isinstance(figs, dict):
            for figure in figs.values():
                _show_figure(figure)
        else:
            _show_figure(figs)


def _show_figure(figure) -> None:
    """Show one figure, dispatching matplotlib vs Plotly."""
    if figure is None:
        return
    if type(figure).__module__.startswith("matplotlib"):
        import matplotlib.pyplot as plt

        plt.show()
    else:
        figure.show()


_PLOT_TYPES = {"box", "bar", "both", "scatter", "radar", "correlation", "effect_size", "all"}


def lcz_cal_indices(
    lcz_map: Union[str, Path],
    indices_x: Union[str, Path, LCZIndicesResult],
    indices: Optional[list[str]] = None,
    plot_type: Literal["box", "bar", "both", "scatter", "radar", "correlation", "effect_size", "all"] = "box",
    index_x: Optional[str] = None,
    index_y: Optional[str] = None,
    size_by: Optional[str] = None,
    lcz_classes: Optional[list[int]] = None,
    normalize_radar: bool = True,
    iplot: bool = True,
    isave: bool = False,
    save_extension: str = "html",
    inclusive: bool = False,
    confidence: float = 0.95,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    lang: str = "en",
    style: str = "default",
    max_workers: Optional[int] = None,
    dpi: Optional[int] = None,
    renderer: Optional[str] = "static",
) -> Union[LCZIndicesStatsResult, pl.DataFrame]:
    """Per-LCZ-class descriptive statistics and charts for spectral indices.

    Parameters
    ----------
    lcz_map : str or Path
        Path to the LCZ class GeoTIFF (classes 1-17) that ``indices_x`` was
        computed on the same grid of.
    indices_x : str, Path, or LCZIndicesResult
        Output of ``lcz_get_indices`` — a path to its GeoTIFF stack, or the
        result object directly (its in-memory ``array`` is used if present,
        else its ``path`` is read).
    indices : list of str, optional
        Subset of index bands to summarize/plot. Defaults to all bands
        present in ``indices_x``. For ``plot_type="scatter"``, must include
        ``index_x``/``index_y``/``size_by`` if you also restrict this list.
    plot_type : {"box", "bar", "both", "scatter", "radar", "correlation", "effect_size", "all"}
        "box": per-class distribution (quartiles/whiskers/mean). "bar":
        per-class mean with a 95% CI error bar. "both": dict with both.
        "scatter": per-pixel scatter of ``index_x`` vs ``index_y``, colored
        by LCZ class, optionally sized by ``size_by`` (all three required
        in ``indices`` or left as the default "all bands"). "radar": one
        polar trace per LCZ class (or ``lcz_classes``) across the requested
        indices. "correlation": Pearson correlation matrix between the
        requested indices. "effect_size": Cohen's d (built-up LCZ 1-10 vs
        natural LCZ 11-17) per index, as a ranked horizontal bar — the
        underlying table is returned in ``.cohens_d`` (computed only for
        this ``plot_type`` and ``"all"``, otherwise ``None``, to skip the
        extra pass when not needed). "all": dict with every applicable
        figure (skips "scatter" with a log message if ``index_x``/
        ``index_y`` weren't given).
        Multiple indices render as small-multiples (one subplot per index)
        for box/bar; effect_size/scatter/radar/correlation are always a
        single figure covering every requested index at once.
    index_x, index_y : str, optional
        Required for ``plot_type="scatter"`` — the two indices to plot
        against each other.
    size_by : str, optional
        Optional third index controlling marker size in the scatter plot.
    lcz_classes : list of int, optional
        Restrict the radar chart to these LCZ classes. Defaults to every
        class present in the data.
    normalize_radar : bool
        Min-max normalize each index's axis on the radar chart (across the
        plotted classes) so indices with very different natural ranges
        (e.g. NDVI's -1..1 vs RI's 10s-1000s) stay visually comparable.
        Default True.
    iplot : bool
        If False, skip chart building and return only the statistics
        DataFrame.
    isave : bool
        Save chart(s) + CSV to ``LCZ4r_output/``.
    save_extension : str
        "png" (default for the static renderer), "pdf", "svg", "tiff", or
        "html" for the interactive Plotly renderer ("html" is coerced to
        "png" for the static renderer).
    inclusive : bool
        Use the colorblind-friendly LCZ palette.
    confidence : float
        Confidence level for the bar chart's error bars. Default 0.95.
    title, subtitle, caption : str, optional
        Figure annotations.
    lang : str
        Message language ("en"/"pt"/"es"/"zh").
    style : str
        Publication style preset applied to every returned figure
        ("default", "nature", "science", "generic_bw").
    max_workers : int, optional
        Bound on the thread pool used to aggregate indices in parallel.
        Default ``min(8, cpu_count)``. Output row order is independent of it.
    dpi : int, optional
        Resolution for static exports. Defaults to 300 dpi (or the style
        preset's own resolution if higher).
    renderer : {"static", "plotly"}, optional
        "static" (default, also ``None``/``"matplotlib"``/``"agg"``) returns
        matplotlib ``Figure`` object(s) and saves native PNG/PDF/SVG/TIFF with
        no Plotly, Kaleido, Chromium or browser. "plotly" returns the previous
        interactive figure(s).

    Returns
    -------
    LCZIndicesStatsResult or pl.DataFrame
        Long-format ``df`` has one row per (parameter, lcz) with columns
        ``parameter, lcz, lcz_name, type (Built-up/Natural), count, mean,
        median, std, min, max, q25, q75, ci_lower, ci_upper``. ``fig`` holds
        matplotlib ``Figure`` object(s) for the default static renderer
        (Plotly ``go.Figure`` when ``renderer="plotly"``). ``cohens_d``
        is a separate per-parameter table (set when ``plot_type`` is
        ``"effect_size"``/``"all"``, else ``None``).

    Examples
    --------
    >>> stats = lcz_cal_indices("lcz_map.tif", idx_result, plot_type="both")
    >>> stats.df.filter(pl.col("parameter") == "NDVI")
    >>> stats.fig["box"].show()

    >>> ranked = lcz_cal_indices("lcz_map.tif", idx_result, plot_type="effect_size")
    >>> ranked.cohens_d.sort("cohens_d", descending=True)

    >>> scatter = lcz_cal_indices(
    ...     "lcz_map.tif", idx_result, plot_type="scatter",
    ...     index_x="NDVI", index_y="NDBI", size_by="LST_C",
    ... )
    """
    if plot_type not in _PLOT_TYPES:
        raise ValueError(f"plot_type must be one of {_PLOT_TYPES}, got {plot_type!r}")
    # None / "matplotlib" / "agg" -> the native static renderer (default).
    renderer = _st.resolve_renderer(renderer)
    if plot_type == "scatter" and not (index_x and index_y):
        raise ValueError("plot_type='scatter' requires both index_x and index_y")

    spec = get_style(style, lang)

    classes, lcz_shape, lcz_transform, lcz_crs = _load_classes(lcz_map)
    stack, band_names, idx_shape, idx_transform, idx_crs = _load_indices_stack(indices_x)

    if lcz_shape != idx_shape:
        raise ValueError(lcz_msg(
            "indices_shape_mismatch", lang, lcz_shape=lcz_shape, idx_shape=idx_shape,
        ))
    # Same shape is not enough: the two rasters must share the same grid.
    if idx_transform is not None and lcz_transform is not None:
        if not lcz_transform.almost_equals(idx_transform, precision=1e-9):
            raise ValueError(
                "The LCZ map and the indices stack share a shape but not a "
                f"transform:\n  lcz: {tuple(lcz_transform)[:6]}\n  idx: "
                f"{tuple(idx_transform)[:6]}\nRecompute the indices on this "
                "exact LCZ map grid."
            )
    if idx_crs is not None and lcz_crs is not None and idx_crs != lcz_crs:
        raise ValueError(
            f"The LCZ map ({lcz_crs}) and the indices stack ({idx_crs}) have "
            "different CRSs. Recompute the indices on this exact LCZ map grid."
        )

    if not band_names:
        raise ValueError("The indices stack has no bands.")

    if indices is not None:
        if isinstance(indices, str):
            indices = [indices]
        missing = [n for n in indices if n not in band_names]
        if missing:
            raise ValueError(f"Index/indices not found in indices_x: {missing}. Available: {band_names}")
        selected = list(dict.fromkeys(indices))  # de-duplicate, keep order
    else:
        selected = list(band_names)
    if not selected:
        raise ValueError("No indices selected to summarize.")

    for name, arg in (("index_x", index_x), ("index_y", index_y), ("size_by", size_by)):
        if arg and arg not in band_names:
            raise ValueError(f"{name}={arg!r} not found in indices_x. Available: {band_names}")

    selected_values = {name: stack[band_names.index(name)] for name in selected}
    values = dict(selected_values)  # may gain index_x/index_y/size_by below, for chart-only use
    for extra in (index_x, index_y, size_by):
        if extra and extra not in values:
            values[extra] = stack[band_names.index(extra)]

    df = _build_indices_summary(classes, selected_values, max_workers=max_workers)
    df = _add_confidence_interval(df, confidence=confidence)
    df = _attach_metadata(df, inclusive, lang=lang)
    df = _attach_type(df, lang=lang)

    if df.height == 0:
        raise ValueError(
            "No valid pixels: every selected index is NaN (or the LCZ map has "
            "no classes 1-17) over the shared grid. Nothing to summarize."
        )

    summarized = set(df["parameter"].unique().to_list())
    missing_params = [p for p in selected if p not in summarized]
    if missing_params:
        logger.warning(
            "These indices are all-NaN over valid LCZ pixels and are omitted "
            "from the statistics/charts: %s", missing_params,
        )

    if not iplot:
        return df.drop("color")

    cohens_d_df = None
    if plot_type in ("effect_size", "all"):
        cohens_d_df = _calculate_cohens_d(classes, selected_values, lang=lang)
        if plot_type == "effect_size" and cohens_d_df.height == 0:
            raise ValueError(
                "Cannot compute Cohen's d: need at least 2 finite pixels in "
                "both built-up (LCZ 1-10) and natural (LCZ 11-17) groups."
            )

    if renderer == "static":
        fig = _static_figures(
            df, values, classes, plot_type, selected, cohens_d_df,
            index_x=index_x, index_y=index_y, size_by=size_by,
            lcz_classes=lcz_classes, normalize_radar=normalize_radar,
            inclusive=inclusive, title=title, subtitle=subtitle, caption=caption,
            lang=lang, style=style, dpi=dpi,
        )
    elif plot_type == "box":
        fig = _plot_box_interactive(df, selected, lang=lang)
    elif plot_type == "bar":
        fig = _plot_bar_interactive(df, selected, lang=lang)
    elif plot_type == "both":
        fig = {
            "box": _plot_box_interactive(df, selected, lang=lang),
            "bar": _plot_bar_interactive(df, selected, lang=lang),
        }
    elif plot_type == "scatter":
        fig = _plot_scatter_interactive(
            classes, values, index_x, index_y, size_by=size_by, inclusive=inclusive, lang=lang,
        )
    elif plot_type == "radar":
        fig = _plot_radar_interactive(
            df, selected, lcz_classes=lcz_classes, normalize=normalize_radar, lang=lang,
        )
    elif plot_type == "correlation":
        fig = _plot_correlation_heatmap(classes, values, selected, lang=lang)
    elif plot_type == "effect_size":
        fig = _plot_effect_size_interactive(cohens_d_df, lang=lang)
    else:  # "all"
        fig = {
            "box": _plot_box_interactive(df, selected, lang=lang),
            "bar": _plot_bar_interactive(df, selected, lang=lang),
            "radar": _plot_radar_interactive(
                df, selected, lcz_classes=lcz_classes, normalize=normalize_radar, lang=lang,
            ),
            "correlation": _plot_correlation_heatmap(classes, values, selected, lang=lang),
            "effect_size": _plot_effect_size_interactive(cohens_d_df, lang=lang),
        }
        if index_x and index_y:
            fig["scatter"] = _plot_scatter_interactive(
                classes, values, index_x, index_y, size_by=size_by, inclusive=inclusive, lang=lang,
            )
        else:
            logger.info("plot_type='all': skipping scatter (index_x/index_y not given).")

    figs = fig if isinstance(fig, dict) else {"_": fig}
    if renderer != "static":
        # Plotly-only annotations/template (the static builders drew the
        # header/footer themselves).
        for f in figs.values():
            if title or subtitle:
                title_text = title or lcz_msg("indices_stats_title", lang)
                if subtitle:
                    title_text += f"<br><sup>{subtitle}</sup>"
                f.update_layout(title=dict(text=title_text))
            if caption:
                f.add_annotation(
                    text=caption, xref="paper", yref="paper", x=0.5, y=-0.08,
                    showarrow=False, font=dict(size=10, color="gray"),
                )
            # Publication style: consistent fonts/margins/ticks across figures.
            f.update_layout(template=spec.template)

    geoarrow_table = _to_geoarrow(df.drop("color"))

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for key, f in figs.items():
            suffix = "" if key == "_" else f"_{key}"
            if renderer == "static":
                _st.save_static(
                    f, f"lcz4r_indices{suffix}", isave=True,
                    save_extension=save_extension, style=style, dpi=dpi, lang=lang,
                )
                continue
            plot_path = os.path.join(OUTPUT_DIR, f"lcz4r_indices{suffix}.{save_extension}")
            if save_extension == "html":
                f.write_html(plot_path, include_plotlyjs="cdn")
            else:
                width_px = int(f.layout.width or 1200)
                height_px = int(f.layout.height or 800)
                save_plotly_figure(
                    f, plot_path,
                    width_px=width_px, height_px=height_px, scale=2,
                    save_extension=save_extension, lang=lang,
                )
        csv_path = os.path.join(OUTPUT_DIR, "lcz4r_indices_df.csv")
        df.drop("color").write_csv(csv_path)
        if cohens_d_df is not None:
            cohens_d_df.write_csv(os.path.join(OUTPUT_DIR, "lcz4r_indices_cohens_d.csv"))
        logger.info("Saved to: %s", os.path.abspath(OUTPUT_DIR))

    return LCZIndicesStatsResult(
        df=df.drop("color"), fig=fig, geoarrow_table=geoarrow_table,
        cohens_d=cohens_d_df, renderer=renderer,
    )


__all__ = ["lcz_cal_indices", "LCZIndicesStatsResult"]
