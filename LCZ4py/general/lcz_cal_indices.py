"""
lcz_cal_indices.py

Per-LCZ-class statistics and interactive charts for spectral indices
(``lcz_get_indices`` output) — the statistics/visualization counterpart to
``lcz_cal_area.py``, generalized from area/count aggregation to arbitrary
continuous index values.

Two chart kinds, both built from the *same* aggregated statistics table
(no second pass over raw pixels for plotting):
- ``"box"``  — per-class distribution (quartiles/whiskers/mean), Plotly's
  precomputed-statistics Box trace.
- ``"bar"``  — per-class mean with a 95% CI error bar (mirrors the R
  package's ``plot_lcz_parameter_comparison``).

When multiple indices are requested, both kinds render as small-multiples
(one subplot per index) instead of one chart per call.
"""

from __future__ import annotations

import logging
import math
import os
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

try:
    import geoarrow.pyarrow as ga
    HAS_GEOARROW = True
except ImportError:
    HAS_GEOARROW = False

from LCZ4py.general.lcz_cal_area import _attach_metadata
from LCZ4py.general.lcz_get_indices import LCZIndicesResult
from LCZ4py._internal.lcz_parameters_data import LCZ_COLORS, LCZ_COLORBLIND
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_classes(lcz_map: Union[str, Path]) -> tuple[np.ndarray, tuple[int, int]]:
    with rasterio.open(str(lcz_map)) as src:
        classes = src.read(1)
        shape = (src.height, src.width)
    return classes, shape


def _load_indices_stack(
    indices_x: Union[str, Path, LCZIndicesResult],
) -> tuple[np.ndarray, list[str], tuple[int, int]]:
    if isinstance(indices_x, LCZIndicesResult):
        if indices_x.array is not None and indices_x.indices is not None:
            arr = indices_x.array
            return arr, list(indices_x.indices), arr.shape[1:]
        if not indices_x.path or not os.path.exists(indices_x.path):
            raise ValueError(
                "LCZIndicesResult has no array and no on-disk raster — call "
                "lcz_get_indices with isave=True, or pass a raster path directly."
            )
        indices_x = indices_x.path

    with rasterio.open(str(indices_x)) as src:
        arr = src.read()
        names = [src.descriptions[i] or f"band_{i + 1}" for i in range(src.count)]
    return arr, names, arr.shape[1:]


# ── Polars aggregation (generalizes lcz_cal_area's count/area pattern) ───────

def _build_indices_summary(
    classes: np.ndarray, values: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Per-(parameter, lcz) descriptive statistics, long format."""
    lcz_flat = classes.astype(np.int16).ravel()
    valid_lcz = (lcz_flat >= 1) & (lcz_flat <= 17)

    rows = []
    for name, arr in values.items():
        val_flat = arr.astype(np.float64).ravel()
        keep = valid_lcz & np.isfinite(val_flat)
        if not keep.any():
            continue
        agg = (
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
        rows.append(agg)

    if not rows:
        return pl.DataFrame(schema={
            "lcz": pl.Int16, "count": pl.UInt32, "mean": pl.Float64,
            "median": pl.Float64, "std": pl.Float64, "min": pl.Float64,
            "max": pl.Float64, "q25": pl.Float64, "q75": pl.Float64,
            "parameter": pl.String,
        })
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
    lcz_flat = classes.astype(np.int16).ravel()
    urban_mask = (lcz_flat >= 1) & (lcz_flat <= 10)
    natural_mask = (lcz_flat >= 11) & (lcz_flat <= 17)

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
    n = df["count"].to_numpy()
    std = df["std"].fill_null(0.0).to_numpy()
    mean = df["mean"].to_numpy()
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

    fig.update_layout(
        title=dict(
            text=lcz_msg("indices_stats_title", lang),
            font=dict(size=20), x=0.5, xanchor="center",
        ),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=90, b=60),
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

    fig.update_layout(
        title=dict(
            text=lcz_msg("indices_stats_title", lang),
            font=dict(size=20), x=0.5, xanchor="center",
        ),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=90, b=60),
        width=420 * cols,
        height=380 * rows,
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    fig.update_xaxes(title=dict(text=lcz_msg("lcz_code_label", lang), font=dict(size=12)))
    return fig


_MAGNITUDE_COLORS: dict[str, str] = {
    "N/A": "#999999",
}
_MAGNITUDE_COLOR_ORDER = ["#cccccc", "#91bfdb", "#fc8d59", "#d73027"]  # negligible -> large


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
    lcz_flat = classes.astype(np.int16).ravel()
    x = values[index_x].astype(np.float64).ravel()
    y = values[index_y].astype(np.float64).ravel()
    valid = (lcz_flat >= 1) & (lcz_flat <= 17) & np.isfinite(x) & np.isfinite(y)
    if size_by:
        s_all = values[size_by].astype(np.float64).ravel()
        valid &= np.isfinite(s_all)

    idx = np.where(valid)[0]
    if len(idx) > max_points:
        idx = np.random.default_rng(seed).choice(idx, size=max_points, replace=False)

    data = {"lcz": lcz_flat[idx], "x": x[idx], "y": y[idx]}
    if size_by:
        data["size"] = s_all[idx]
    df = pl.DataFrame(data)
    df = _attach_metadata(df, inclusive, lang=lang)

    fig = go.Figure()
    for lcz_val in sorted(df["lcz"].unique().to_list()):
        sub = df.filter(pl.col("lcz") == lcz_val)
        marker = dict(color=sub["color"][0], size=7, line=dict(color="black", width=0.3))
        hovertemplate = (
            f"<b>LCZ {lcz_val}</b> — {sub['lcz_name'][0]}<br>"
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
            name=str(sub["lcz_name"][0]), customdata=customdata,
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
    lcz_flat = classes.astype(np.int16).ravel()
    valid = (lcz_flat >= 1) & (lcz_flat <= 17)
    cols = {}
    for p in params:
        v = values[p].astype(np.float64).ravel()
        valid = valid & np.isfinite(v)
        cols[p] = v
    df = pl.DataFrame(cols).filter(pl.Series(valid))

    corr = df.corr()
    z = corr.to_numpy()

    fig = go.Figure(go.Heatmap(
        z=z, x=params, y=params, colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title=dict(text="Pearson r")),
        hovertemplate="%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=lcz_msg("indices_correlation_title", lang),
            font=dict(size=20), x=0.5, xanchor="center",
        ),
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="white",
        width=max(500, 60 * len(params)), height=max(500, 60 * len(params)),
        margin=dict(l=100, r=40, t=60, b=100),
    )
    return fig


def _to_geoarrow(df: pl.DataFrame) -> Optional[pa.Table]:
    if not HAS_GEOARROW:
        return None
    try:
        return pa.Table.from_pandas(df.to_pandas())
    except Exception:
        return None


# ── Public entry point ────────────────────────────────────────────────────────

@dataclass
class LCZIndicesStatsResult:
    """Return type of lcz_cal_indices."""
    df: pl.DataFrame
    fig: Optional[Union[go.Figure, dict[str, go.Figure]]] = None
    geoarrow_table: Optional[pa.Table] = None
    cohens_d: Optional[pl.DataFrame] = None


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
        "html" for interactive, "png"/"pdf" for static.
    inclusive : bool
        Use the colorblind-friendly LCZ palette.
    confidence : float
        Confidence level for the bar chart's error bars. Default 0.95.
    title, subtitle, caption : str, optional
        Figure annotations.
    lang : str
        Message language ("en"/"pt"/"es"/"zh").

    Returns
    -------
    LCZIndicesStatsResult or pl.DataFrame
        Long-format ``df`` has one row per (parameter, lcz) with columns
        ``parameter, lcz, lcz_name, type (Built-up/Natural), count, mean,
        median, std, min, max, q25, q75, ci_lower, ci_upper``. ``cohens_d``
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
    if plot_type == "scatter" and not (index_x and index_y):
        raise ValueError("plot_type='scatter' requires both index_x and index_y")

    classes, lcz_shape = _load_classes(lcz_map)
    stack, band_names, idx_shape = _load_indices_stack(indices_x)

    if lcz_shape != idx_shape:
        raise ValueError(lcz_msg(
            "indices_shape_mismatch", lang, lcz_shape=lcz_shape, idx_shape=idx_shape,
        ))

    if indices is not None:
        missing = [n for n in indices if n not in band_names]
        if missing:
            raise ValueError(f"Index/indices not found in indices_x: {missing}. Available: {band_names}")
        selected = list(indices)
    else:
        selected = list(band_names)

    for name, arg in (("index_x", index_x), ("index_y", index_y), ("size_by", size_by)):
        if arg and arg not in band_names:
            raise ValueError(f"{name}={arg!r} not found in indices_x. Available: {band_names}")

    selected_values = {name: stack[band_names.index(name)] for name in selected}
    values = dict(selected_values)  # may gain index_x/index_y/size_by below, for chart-only use
    for extra in (index_x, index_y, size_by):
        if extra and extra not in values:
            values[extra] = stack[band_names.index(extra)]

    df = _build_indices_summary(classes, selected_values)
    df = _add_confidence_interval(df, confidence=confidence)
    df = _attach_metadata(df, inclusive, lang=lang)
    df = _attach_type(df, lang=lang)

    if not iplot:
        return df.drop("color")

    cohens_d_df = None
    if plot_type in ("effect_size", "all"):
        cohens_d_df = _calculate_cohens_d(classes, selected_values, lang=lang)

    if plot_type == "box":
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

    geoarrow_table = _to_geoarrow(df.drop("color"))

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for key, f in figs.items():
            suffix = "" if key == "_" else f"_{key}"
            plot_path = os.path.join(OUTPUT_DIR, f"lcz4r_indices{suffix}.{save_extension}")
            if save_extension == "html":
                f.write_html(plot_path, include_plotlyjs="cdn")
            else:
                f.write_image(plot_path, width=1200, height=800, scale=2)
        csv_path = os.path.join(OUTPUT_DIR, "lcz4r_indices_df.csv")
        df.drop("color").write_csv(csv_path)
        if cohens_d_df is not None:
            cohens_d_df.write_csv(os.path.join(OUTPUT_DIR, "lcz4r_indices_cohens_d.csv"))
        logger.info("Saved to: %s", os.path.abspath(OUTPUT_DIR))

    return LCZIndicesStatsResult(
        df=df.drop("color"), fig=fig, geoarrow_table=geoarrow_table, cohens_d=cohens_d_df,
    )


__all__ = ["lcz_cal_indices", "LCZIndicesStatsResult"]
