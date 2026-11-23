"""
lcz_anomaly.py — per-station temperature anomaly relative to the overall mean.

Rendering backends
------------------
- ``renderer="static"`` *(default)* — native matplotlib/Agg figure saved as a
  300 dpi PNG/PDF/SVG/TIFF. No Plotly, Kaleido, Chromium or browser needed.
- ``renderer="plotly"`` — the interactive figure.

Uses Polars groupby for O(N) anomaly math. All connectors (dot plot,
lollipop) are drawn as one vectorised call instead of one figure shape per
station.
"""

from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, normalise_input_df, normalise_missing,
    select_by_date, assign_station_lcz, lcz_palette,
    require_nonempty, add_by_column, by_sorted_groups,
)
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export
from LCZ4py._internal import lcz_static
from LCZ4py._internal.lcz_static import resolve_renderer

logger = logging.getLogger(__name__)

_VALID_PLOT_TYPES = ("diverging_bar", "bar", "dot", "lollipop", "scatter")

# Colorblind-safe diverging pair (warm = positive anomaly, cool = negative)
_WARM_COLOR = "#B2182B"
_COOL_COLOR = "#2166AC"

# Labels that are not in the central i18n table yet. Kept module-local so this
# module never renders "<missing i18n key: ...>" into a figure.
_EXTRA_MESSAGES: dict[str, dict[str, str]] = {
    "station_label": {
        "en": "Station", "pt": "Estação", "es": "Estación", "zh": "站点",
    },
    "warmer_label": {
        "en": "Warmer than reference",
        "pt": "Mais quente que a referência",
        "es": "Más cálido que la referencia",
        "zh": "高于参考值",
    },
    "cooler_label": {
        "en": "Cooler than reference",
        "pt": "Mais frio que a referência",
        "es": "Más frío que la referencia",
        "zh": "低于参考值",
    },
}


def _xmsg(key: str, lang: str, **kwargs) -> str:
    """Look up a module-local label first, then the central ``LCZ_MESSAGES``."""
    if key in _EXTRA_MESSAGES:
        table = _EXTRA_MESSAGES[key]
        template = table.get(lang, table["en"])
        for k, v in kwargs.items():
            template = template.replace("{" + k + "}", str(v))
        return template
    return lcz_msg(key, lang, **kwargs)



def _compute_anomaly(df: pl.DataFrame) -> pl.DataFrame:
    """Vectorised anomaly = station mean − overall mean, sorted ascending.

    Returns columns ``station, lcz, mean_value, reference_value, anomaly, n``.
    """
    if df.is_empty():
        raise ValueError("Cannot compute anomalies from an empty DataFrame.")

    overall_mean = df["var_interp"].mean()
    if overall_mean is None:
        raise ValueError(
            "The reference (overall) mean is null — every value in 'var_interp' "
            "is missing."
        )

    return (
        df.group_by("station")
          .agg([
              pl.col("var_interp").mean().alias("mean_value"),
              pl.col("var_interp").count().alias("n"),
              pl.col("lcz").first(),
              pl.col("latitude").first(),
              pl.col("longitude").first(),
          ])
          .with_columns(reference_value=pl.lit(float(overall_mean)))
          .with_columns(
              (pl.col("mean_value") - pl.col("reference_value")).round(2).alias("anomaly")
          )
          .drop_nulls("anomaly")
          .sort(["anomaly", "station"])
    )


def _base_layout(title: str, **kw) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
        **kw,
    )


def _lcz_colors(df: pl.DataFrame, colors: dict[str, str]) -> list[str]:
    """Map each row's LCZ class to its palette colour (vectorised lookup)."""
    return [
        colors.get(str(int(lv)), "#888888") if lv is not None else "#888888"
        for lv in df["lcz"].to_list()
    ]


def _plot_diverging_bar(df, colors, title, xlab, lang):
    bar_colors = _lcz_colors(df, colors)
    fig = go.Figure(go.Bar(
        y=df["station"],
        x=df["anomaly"],
        orientation="h",
        marker_color=bar_colors,
        marker_line_color="black",
        marker_line_width=0.5,
        text=[f"{a:+.2f}" for a in df["anomaly"]],
        textposition="outside",
        textfont=dict(size=11),
        customdata=df.select(["lcz", "mean_value", "n"]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            + f"{xlab}: %{{x:+.2f}}<br>"
            + f"{lcz_msg('lcz_code_label', lang)}: %{{customdata[0]}}<br>"
            + f"{lcz_msg('temperature_label', lang)}: %{{customdata[1]:.2f}}<br>"
            + "n: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_color="#666666", line_width=1)
    fig.update_layout(**_base_layout(
        title,
        xaxis=dict(
            title=dict(text=xlab, font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="#f0f0f0",
            zeroline=True, zerolinecolor="#aaaaaa", zerolinewidth=1,
        ),
        yaxis=dict(categoryorder="total ascending", tickfont=dict(size=11),
                   title=dict(text=_xmsg("station_label", lang))),
        margin=dict(l=160, r=80, t=70, b=60),
        height=max(400, 30 * len(df)),
    ))
    return fig


def _plot_bar(df, colors, title, xlab, lang):
    """Vertical bars, split into warm/cool traces with a diverging palette."""
    warm = df.filter(pl.col("anomaly") >= 0)
    cool = df.filter(pl.col("anomaly") < 0)
    fig = go.Figure()
    for subset, color, name in (
        (warm, _WARM_COLOR, _xmsg("warmer_label", lang)),
        (cool, _COOL_COLOR, _xmsg("cooler_label", lang)),
    ):
        if subset.is_empty():
            continue
        fig.add_trace(go.Bar(
            x=subset["station"],
            y=subset["anomaly"],
            name=name,
            marker_color=color,
            marker_line_color="black",
            marker_line_width=0.5,
            text=[f"{a:+.2f}" for a in subset["anomaly"]],
            textposition="outside",
            textfont=dict(size=11),
            customdata=subset.select(["lcz", "n"]),
            hovertemplate=(
                "<b>%{x}</b><br>"
                + f"{xlab}: %{{y:+.2f}}<br>"
                + f"{lcz_msg('lcz_code_label', lang)}: %{{customdata[0]}}<br>"
                + "n: %{customdata[1]}<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line_color="#666666", line_width=1)
    fig.update_layout(**_base_layout(
        title,
        xaxis=dict(tickfont=dict(size=10), tickangle=-40, gridcolor="#f0f0f0",
                   title=dict(text=_xmsg("station_label", lang))),
        yaxis=dict(
            title=dict(text=xlab, font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="#f0f0f0",
            zeroline=True, zerolinecolor="#aaaaaa", zerolinewidth=1,
        ),
        margin=dict(l=80, r=40, t=70, b=140),
        height=max(450, 20 * len(df)),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02),
    ))
    return fig


def _connector_trace(pairs: list[tuple[float, float, str]]) -> go.Scatter:
    """One line trace drawing every (ref → mean) connector, no per-row shapes."""
    xs: list[Optional[float]] = []
    ys: list[Optional[str]] = []
    for ref, mean, station in pairs:
        xs.extend([ref, mean, None])
        ys.extend([station, station, None])
    return go.Scatter(
        x=xs, y=ys, mode="lines", name="",
        line=dict(color="#cccccc", width=1.5),
        hoverinfo="skip", showlegend=False,
    )


def _plot_dot(df, colors, title, xlab, lang):
    """Cleveland dot plot: mean_value (LCZ color) vs reference (gray), connected."""
    ref = float(df["reference_value"][0])
    temp_label = lcz_msg("temperature_label", lang)

    pairs = list(zip(
        df["reference_value"].to_list(),
        df["mean_value"].to_list(),
        df["station"].to_list(),
    ))
    fig = go.Figure()
    fig.add_trace(_connector_trace(pairs))
    fig.add_trace(go.Scatter(
        x=[ref] * len(df),
        y=df["station"],
        mode="markers",
        marker=dict(color="#999999", size=10, symbol="circle"),
        name=f"{temp_label} (ref)",
        hovertemplate=f"Reference: {ref:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["mean_value"],
        y=df["station"],
        mode="markers",
        marker=dict(
            color=_lcz_colors(df, colors),
            size=12, symbol="circle",
            line=dict(color="black", width=0.5),
        ),
        name=temp_label,
        customdata=df.select(["anomaly", "lcz"]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            + f"{temp_label}: %{{x:.2f}}<br>"
            + f"{xlab}: %{{customdata[0]:+.2f}}<br>"
            + f"{lcz_msg('lcz_code_label', lang)}: %{{customdata[1]}}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(**_base_layout(
        title,
        xaxis=dict(
            title=dict(text=temp_label, font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(categoryorder="total ascending", tickfont=dict(size=11),
                   title=dict(text=_xmsg("station_label", lang))),
        margin=dict(l=160, r=80, t=70, b=60),
        height=max(400, 32 * len(df)),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.05),
    ))
    return fig


def _plot_lollipop(df, colors, title, xlab, lang):
    """Lollipop: stem from 0 to anomaly, dot at the tip."""
    stations = df["station"].to_list()
    anomalies = df["anomaly"].to_list()
    dot_colors = _lcz_colors(df, colors)

    xs: list[Optional[str]] = []
    ys: list[Optional[float]] = []
    for st, a in zip(stations, anomalies):
        xs.extend([st, st, None])
        ys.extend([0.0, a, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="",
        line=dict(color="#999999", width=1.2),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=stations,
        y=anomalies,
        mode="markers",
        marker=dict(color=dot_colors, size=14, line=dict(color="black", width=0.8)),
        customdata=df.select(["lcz", "n"]),
        hovertemplate=(
            "<b>%{x}</b><br>"
            + f"{xlab}: %{{y:+.2f}}<br>"
            + f"{lcz_msg('lcz_code_label', lang)}: %{{customdata[0]}}<br>"
            + "n: %{customdata[1]}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_hline(y=0, line_color="#666666", line_width=1)
    fig.update_layout(**_base_layout(
        title,
        xaxis=dict(tickfont=dict(size=10), tickangle=-40, gridcolor="#f0f0f0",
                   title=dict(text=_xmsg("station_label", lang))),
        yaxis=dict(
            title=dict(text=xlab, font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="#f0f0f0",
            zeroline=True, zerolinecolor="#aaaaaa", zerolinewidth=1,
        ),
        margin=dict(l=80, r=40, t=70, b=140),
        height=max(450, 20 * len(df)),
    ))
    return fig


def _plot_scatter(df, colors, title, xlab, lang):
    """Scatter: LCZ class (x) vs mean temperature (y), dot size = |anomaly|."""
    temp_label = lcz_msg("temperature_label", lang)
    fig = go.Figure(go.Scatter(
        x=df["lcz"].cast(pl.String),
        y=df["mean_value"],
        mode="markers+text",
        text=df["station"],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            color=_lcz_colors(df, colors),
            size=[max(8.0, abs(float(a)) * 10.0) for a in df["anomaly"]],
            line=dict(color="black", width=0.5),
            opacity=0.85,
        ),
        customdata=df.select(["anomaly", "n"]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            + f"{lcz_msg('lcz_code_label', lang)}: %{{x}}<br>"
            + f"{temp_label}: %{{y:.2f}}<br>"
            + f"{xlab}: %{{customdata[0]:+.2f}}<br>"
            + "n: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.add_hline(y=float(df["reference_value"][0]), line_dash="dash",
                  line_color="gray")
    fig.update_layout(**_base_layout(
        title,
        xaxis=dict(
            title=dict(text=lcz_msg("lcz_code_label", lang), font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(
            title=dict(text=temp_label, font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="#f0f0f0",
        ),
        margin=dict(l=80, r=40, t=70, b=60),
        height=500,
    ))
    return fig


_PLOT_FNS = {
    "diverging_bar": _plot_diverging_bar,
    "bar": _plot_bar,
    "dot": _plot_dot,
    "lollipop": _plot_lollipop,
    "scatter": _plot_scatter,
}


# ── Native static renderer (matplotlib/Agg — no Plotly/Kaleido/Chromium) ─────

def _panel_axes(fig, n: int, *, has_header: bool, has_legend: bool,
                gap: float = 0.055, left: float = 0.22,
                bottom: Optional[float] = None, top_trim: float = 0.0):
    """Stacked full-width axes inside ``fig`` (top panel first)."""
    top = lcz_static.figure_top(has_header) - float(top_trim)
    bottom = lcz_static.figure_bottom(has_legend) if bottom is None else float(bottom)
    n = max(int(n), 1)
    span = max(top - bottom - gap * (n - 1), 0.10)
    height = span / n
    axes = []
    for i in range(n):
        bottom_i = top - (i + 1) * height - i * gap
        axes.append(fig.add_axes([left, bottom_i, 0.965 - left, height]))
    return axes


def _legend_bottom(fig, fontsize: float, rows: int, *, with_title: bool = True,
                   xlabel: bool = True) -> float:
    """Fraction of the figure height to reserve below the axes for a legend."""
    fig_h = max(float(fig.get_figheight()), 0.5)
    line_in = (float(fontsize) * 1.75) / 72.0
    legend_in = max(int(rows), 0) * line_in + (0.22 if with_title else 0.12)
    extra_in = 0.58 if xlabel else 0.18
    return float(min(0.45, max(0.14, (legend_in + extra_in) / fig_h)))


def _style_axis(ax, *, spec, grid_axis: str = "x") -> None:
    """Shared tick/spine styling for the static anomaly charts."""
    ax.grid(True, axis=grid_axis, lw=0.35, color="0.88", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", labelsize=max(5.5, spec.font_size - 2.5),
                   length=3, width=0.6)


def _static_class_legend(fig, classes, *, inclusive: bool, lang: str,
                         spec, title: Optional[str] = None,
                         extra_handles: Optional[list] = None):
    """LCZ class swatch legend shared by the static anomaly charts."""
    from matplotlib.patches import Patch

    from LCZ4py._internal.lcz_parameters_data import (
        LCZ_COLORBLIND, LCZ_COLORS, get_lcz_names,
    )

    present = sorted({int(c) for c in classes if c is not None})
    if not present and not extra_handles:
        return None
    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    names = get_lcz_names(lang)
    handles = [
        Patch(facecolor=colors[c - 1], edgecolor="black", linewidth=0.3,
              label=f"{c}: {names[c - 1]}")
        for c in present
    ]
    handles.extend(extra_handles or [])
    fontsize = max(5.5, spec.font_size - 2.5)
    return fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.006),
        ncol=min(len(handles), 4), frameon=False, fontsize=fontsize,
        handlelength=1.1, columnspacing=1.0,
        title=title if title is not None else lcz_msg("lcz_class_legend", lang),
        title_fontsize=fontsize,
    )


def _static_anomaly_axes(ax, adf: pl.DataFrame, kind: str, xlab: str,
                         colors: dict, lang: str, spec) -> None:
    """Draw one anomaly panel on ``ax`` for the requested ``kind``."""
    stations = [str(s) for s in adf["station"].to_list()]
    anomalies = np.asarray(adf["anomaly"].to_numpy(), dtype=float)
    means = np.asarray(adf["mean_value"].to_numpy(), dtype=float)
    lcz_vals = adf["lcz"].to_list()
    bar_colors = [colors.get(str(int(v)), "#888888") for v in lcz_vals]
    label_size = max(6.0, spec.font_size - 1.5)
    annotate = len(stations) <= 40

    if kind == "diverging_bar":
        ypos = np.arange(len(stations))
        ax.barh(ypos, anomalies, color=bar_colors, edgecolor="black",
                linewidth=0.4, height=0.72, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(stations, fontsize=max(5.0, spec.font_size - 3.0))
        ax.axvline(0, color="#666666", lw=0.8, zorder=2)
        ax.set_xlabel(xlab, fontsize=label_size)
        _style_axis(ax, spec=spec, grid_axis="x")
        if annotate:
            span = float(np.nanmax(np.abs(anomalies))) if anomalies.size else 1.0
            pad = 0.03 * (span if span > 0 else 1.0)
            for yi, value in zip(ypos, anomalies):
                ax.text(
                    value + (pad if value >= 0 else -pad), yi, f"{value:+.2f}",
                    va="center", ha="left" if value >= 0 else "right",
                    fontsize=max(4.5, spec.font_size - 3.5), zorder=4,
                )
            ax.margins(x=0.16)
        return

    if kind == "bar":
        xpos = np.arange(len(stations))
        warm = anomalies >= 0
        if warm.any():
            ax.bar(xpos[warm], anomalies[warm], color=_WARM_COLOR,
                   edgecolor="black", linewidth=0.4, width=0.78, zorder=3,
                   label=_xmsg("warmer_label", lang))
        cool = ~warm
        if cool.any():
            ax.bar(xpos[cool], anomalies[cool], color=_COOL_COLOR,
                   edgecolor="black", linewidth=0.4, width=0.78, zorder=3,
                   label=_xmsg("cooler_label", lang))
        ax.axhline(0, color="#666666", lw=0.8, zorder=2)
        ax.set_xticks(xpos)
        rotation = 90 if len(stations) > 8 else 0
        ax.set_xticklabels(
            stations, fontsize=max(5.0, spec.font_size - 3.0),
            rotation=rotation, ha="center",
        )
        ax.set_ylabel(xlab, fontsize=label_size)
        _style_axis(ax, spec=spec, grid_axis="y")
        return

    if kind == "dot":
        reference = float(adf["reference_value"][0])
        ypos = np.arange(len(stations))
        for yi, (ref, mean) in enumerate(zip(adf["reference_value"].to_list(), means)):
            ax.plot([float(ref), float(mean)], [yi, yi], color="#cccccc",
                    lw=1.1, zorder=2, solid_capstyle="round")
        ax.scatter([reference] * len(stations), ypos, s=22, color="#999999",
                   zorder=3)
        ax.scatter(means, ypos, s=34, c=bar_colors, edgecolors="black",
                   linewidths=0.5, zorder=4)
        ax.set_yticks(ypos)
        ax.set_yticklabels(stations, fontsize=max(5.0, spec.font_size - 3.0))
        ax.set_xlabel(lcz_msg("temperature_label", lang), fontsize=label_size)
        _style_axis(ax, spec=spec, grid_axis="x")
        return

    if kind == "lollipop":
        xpos = np.arange(len(stations))
        ax.vlines(xpos, 0.0, anomalies, color="#999999", lw=1.0, zorder=2)
        ax.scatter(xpos, anomalies, s=46, c=bar_colors, edgecolors="black",
                   linewidths=0.7, zorder=3)
        ax.axhline(0, color="#666666", lw=0.8, zorder=1)
        ax.set_xticks(xpos)
        rotation = 90 if len(stations) > 8 else 0
        ax.set_xticklabels(
            stations, fontsize=max(5.0, spec.font_size - 3.0),
            rotation=rotation, ha="center" if rotation else "left",
        )
        ax.set_ylabel(xlab, fontsize=label_size)
        _style_axis(ax, spec=spec, grid_axis="y")
        return

    # kind == "scatter": LCZ class (x) vs mean temperature (y)
    classes = sorted({int(v) for v in lcz_vals if v is not None})
    rank: dict[int, int] = {}
    total_per_class: dict[int, int] = {}
    for v in lcz_vals:
        if v is None:
            continue
        total_per_class[int(v)] = total_per_class.get(int(v), 0) + 1
    xs = []
    for v in lcz_vals:
        if v is None:
            xs.append(np.nan)
            continue
        idx = rank.get(int(v), 0)
        rank[int(v)] = idx + 1
        n_here = total_per_class.get(int(v), 1)
        offset = 0.0 if n_here <= 1 else (idx / (n_here - 1) - 0.5) * 0.55
        xs.append(classes.index(int(v)) + offset)
    xs = np.asarray(xs, dtype=float)
    sizes = np.maximum(20.0, np.abs(anomalies) * 60.0)
    ax.scatter(xs, means, s=sizes, c=bar_colors, edgecolors="black",
               linewidths=0.5, alpha=0.9, zorder=3)
    if len(stations) <= 12:
        for xi, yi, name in zip(xs, means, stations):
            ax.annotate(name, (xi, yi), textcoords="offset points",
                        xytext=(0, 5), ha="center",
                        fontsize=max(4.5, spec.font_size - 3.5))
    reference = float(adf["reference_value"][0])
    ax.axhline(reference, color="0.45", lw=0.8, ls="--", zorder=2)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels([f"{c}" for c in classes],
                       fontsize=max(5.0, spec.font_size - 2.5))
    ax.set_xlabel(lcz_msg("lcz_code_label", lang), fontsize=label_size)
    ax.set_ylabel(lcz_msg("temperature_label", lang), fontsize=label_size)
    ax.margins(x=0.07, y=0.16)
    _style_axis(ax, spec=spec, grid_axis="y")


def _static_anomaly_figure(
    kind: str,
    panels: list,
    *,
    title: Optional[str],
    xlab: str,
    lang: str = "en",
    inclusive: bool = False,
    style: str = "default",
    dpi: Optional[int] = None,
    caption: Optional[str] = None,
    isave: bool = False,
    save_extension: str = "html",
    filename: str = "lcz4r_anomaly_plot",
):
    """Publication-quality anomaly figure (one axes per ``by=`` facet)."""
    st = lcz_static
    colors = lcz_palette(inclusive=inclusive)
    panels = list(panels) or [("", pl.DataFrame({"station": [], "anomaly": []}))]
    n = len(panels)
    total_rows = sum(max(sub.height, 1) for _, sub in panels)

    if kind in ("diverging_bar", "dot"):
        height_mm = 3.0 * total_rows + 26.0 * n + 22.0
    else:
        height_mm = 2.2 * total_rows + 30.0 * n + 24.0

    classes = sorted({
        int(v)
        for _, adf in panels if not adf.is_empty()
        for v in adf["lcz"].to_list()
        if v is not None
    })
    has_class_legend = kind in ("diverging_bar", "dot", "lollipop", "scatter") and bool(classes)
    has_warm_cool = kind == "bar"
    has_legend = has_class_legend or has_warm_cool
    legend_rows = max(1, -(-len(classes) // 4)) if has_class_legend else 1
    height_mm += 7.0 * max(legend_rows - 1, 0)
    height_mm = float(min(max(height_mm, 70.0), 195.0))

    with st.style_context(style):
        fig, spec, _ = st.new_figure(style, height_mm=height_mm, dpi=dpi)
        axes = _panel_axes(
            fig, n,
            has_header=bool(title),
            has_legend=has_legend,
            bottom=(
                _legend_bottom(
                    fig, max(5.5, spec.font_size - 2.5), legend_rows,
                    with_title=has_class_legend,
                ) if has_legend
                else lcz_static.figure_bottom(False)
            ),
            top_trim=0.03 if any(label for label, _ in panels) else 0.0,
        )
        for ax, (label, adf) in zip(axes, panels):
            if label:
                ax.set_title(str(label), fontsize=max(6.5, spec.font_size - 1.5),
                             loc="left", pad=2.5)
            if adf.is_empty():
                continue
            _static_anomaly_axes(ax, adf, kind, xlab, colors, lang, spec)

        if has_class_legend:
            extra = None
            if kind == "dot":
                from matplotlib.lines import Line2D

                temp_label = lcz_msg("temperature_label", lang)
                extra = [
                    Line2D([], [], marker="o", ls="", ms=4.2, color="#999999",
                           label=f"{temp_label} (ref)"),
                    Line2D([], [], marker="o", ls="", ms=4.6, color="#444444",
                           markeredgecolor="black", markeredgewidth=0.4,
                           label=temp_label),
                ]
            _static_class_legend(fig, classes, inclusive=inclusive, lang=lang,
                                 spec=spec, extra_handles=extra)
        elif has_warm_cool:
            from matplotlib.patches import Patch

            fontsize = max(5.5, spec.font_size - 2.5)
            fig.legend(
                handles=[
                    Patch(facecolor=_WARM_COLOR, edgecolor="black", linewidth=0.4,
                          label=_xmsg("warmer_label", lang)),
                    Patch(facecolor=_COOL_COLOR, edgecolor="black", linewidth=0.4,
                          label=_xmsg("cooler_label", lang)),
                ],
                loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=2,
                frameon=False, fontsize=fontsize, handlelength=1.3,
                columnspacing=1.6,
            )

        st.add_header(fig, title, None, spec)
        st.add_footer(fig, caption, None, spec)
        st.save_static(
            fig, filename, isave=isave, save_extension=save_extension,
            style=style, dpi=dpi, lang=lang,
        )
    return fig


def lcz_anomaly(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
    data_frame,
    var: str = "",
    station_id: str = "",
    *,
    plot_type: str = "diverging_bar",
    by: Optional[str] = None,
    iplot: bool = True,
    isave: bool = False,
    save_extension: str = "html",
    style: str = "default",
    inclusive: bool = False,
    title: Optional[str] = None,
    caption: Optional[str] = None,
    lang: str = "en",
    year=None, month=None, day=None, hour=None, start=None, end=None,
    renderer: Optional[str] = "static",
    dpi: Optional[int] = None,
) -> Union[go.Figure, pl.DataFrame, "Figure"]:
    """Compute per-station temperature anomaly and render an anomaly chart.

    Parameters
    ----------
    x : str, PathLike, or rasterio dataset
        LCZ raster used to assign each station to an LCZ class.
    data_frame : pd.DataFrame or pl.DataFrame
        Observation table with lat/lon/date/var/station columns.
    var : str
        Column name for the meteorological variable.
    station_id : str
        Column name for the station identifier.
    plot_type : {"diverging_bar", "bar", "dot", "lollipop", "scatter"}
        Chart style:

        - ``"diverging_bar"`` *(default)* — horizontal bar diverging from zero.
        - ``"bar"`` — vertical bar; red = warm anomaly, blue = cool anomaly.
        - ``"dot"`` — Cleveland dot plot showing mean temperature vs reference,
          connected by lines; anomaly magnitude shown on hover.
        - ``"lollipop"`` — vertical stems from zero to anomaly value, with a
          circle at the tip.
        - ``"scatter"`` — LCZ class (x) vs mean temperature (y); dot area
          encodes |anomaly|, useful for cross-LCZ comparison.
    by : str, optional
        Split the anomaly computation into faceted subplots by temporal group.
        Options: ``"year"``, ``"season"``, ``"seasonyear"``, ``"month"``,
        ``"monthyear"``, ``"weekday"``, ``"weekend"``, ``"site"``,
        ``"daylight"`` (UTC, NOAA formula), ``"dst"`` (approximate).
        ``None`` = single chart (default).
    iplot : bool
        If False, return the anomaly DataFrame instead of a figure.
    isave : bool
        Save the figure to ``LCZ4r_output/``.
    save_extension : str
        ``"html"`` (interactive) or ``"png"``/``"pdf"`` (static).
    style : str
        Publication style preset: 'default', 'nature', 'science', or
        'generic_bw'. Controls font, figure size (mm), DPI, and palette
        used when isave and save_extension != 'html'.
    inclusive : bool
        Use the colorblind-friendly LCZ palette.
    title : str, optional
        Override the default chart title.
    caption : str, optional
        Small annotation added below the chart.
    lang : str
        Language for labels — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.
    year, month, day, hour : int or sequence, optional
        Date/time component filters (validated; out-of-range values raise).
    start, end : str or datetime, optional
        Inclusive date range bounds.
    renderer : {"static", "plotly"}, optional
        ``"static"`` (default, also ``None``/``"matplotlib"``/``"agg"``)
        returns a matplotlib ``Figure`` drawn with the Agg canvas — no Plotly,
        Kaleido, Chromium or browser needed — and saves a >=300 dpi PNG (or
        PDF/SVG/TIFF) via ``lcz_static.save_static``. ``"plotly"`` returns the
        interactive figure.
    dpi : int, optional
        Export resolution for the static renderer (never below 300 dpi).

    Returns
    -------
    matplotlib.figure.Figure, go.Figure or pl.DataFrame
        A matplotlib ``Figure`` for the default static renderer, a Plotly
        ``go.Figure`` for ``renderer="plotly"``, or the anomaly DataFrame when
        ``iplot=False``. DataFrame columns: ``station, lcz, mean_value,
        reference_value, anomaly, n, latitude, longitude``.

    Raises
    ------
    ValueError
        On an unknown ``plot_type``/``by``, empty input, missing columns, or
        when no station maps to LCZ class 1-17.
    """
    if plot_type not in _PLOT_FNS:
        raise ValueError(f"plot_type must be one of {_VALID_PLOT_TYPES!r}, got {plot_type!r}")

    renderer = resolve_renderer(renderer)
    static = renderer == "static"

    df = normalise_input_df(data_frame, var=var, station_id=station_id)
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null())
    require_nonempty(df, "observations")

    ds = load_lcz_raster(x)
    df = assign_station_lcz(df, ds)
    require_nonempty(df, "observations at stations inside the LCZ raster")
    df = df.with_columns(
        (pl.col("station") + " (" + pl.col("lcz").cast(pl.String) + ")").alias("station")
    )

    if not iplot:
        if by is None:
            return _compute_anomaly(df)
        lat_mean = float(df["latitude"].mean())
        lon_mean = float(df["longitude"].mean())
        return (
            add_by_column(df, by, lat=lat_mean, lon=lon_mean)
            .group_by("_by")
            .map_groups(lambda g: _compute_anomaly(g).with_columns(
                pl.lit(g["_by"][0]).alias("_by")
            ))
            .sort(["anomaly", "station", "_by"])
        )

    chart_title = title or lcz_msg("lcz_anomaly_title", lang)
    xlab = lcz_msg("anomaly_label", lang)
    colors = lcz_palette(inclusive=inclusive)
    plot_fn = _PLOT_FNS[plot_type]

    if by is not None:
        lat_mean = float(df["latitude"].mean())
        lon_mean = float(df["longitude"].mean())
        df = add_by_column(df, by, lat=lat_mean, lon=lon_mean)
        groups = by_sorted_groups(df["_by"], by)

        # Only facet groups that can actually produce an anomaly (>=2 stations)
        usable: list[tuple[str, pl.DataFrame]] = []
        for g in groups:
            sub = df.filter(pl.col("_by") == g)
            n_stations = sub["station"].n_unique()
            if n_stations < 2:
                logger.warning(
                    "Skipping by-group %r: %d station(s), need at least 2 for "
                    "an anomaly reference.", g, n_stations,
                )
                continue
            usable.append((g, sub))
        if not usable:
            raise ValueError(
                f"No by={by!r} group contains at least 2 stations, so no "
                "anomaly can be computed."
            )

        if static:
            fig = _static_anomaly_figure(
                plot_type,
                [(g, _compute_anomaly(sub)) for g, sub in usable],
                title=chart_title, xlab=xlab, lang=lang, inclusive=inclusive,
                style=style, dpi=dpi, caption=caption, isave=isave,
                save_extension=save_extension,
            )
        else:
            fig = make_subplots(
                rows=len(usable), cols=1,
                subplot_titles=[g for g, _ in usable],
                shared_xaxes=False,
                vertical_spacing=0.05,
            )
            for i, (g, sub) in enumerate(usable, 1):
                adf = _compute_anomaly(sub)
                sub_fig = plot_fn(adf, colors, g, xlab, lang)
                for trace in sub_fig.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=i, col=1)
            fig.update_layout(
                title=dict(text=chart_title, font=dict(size=20), x=0.5, xanchor="center"),
                height=max(500, 350 * len(usable)),
                plot_bgcolor="#fafafa",
                paper_bgcolor="white",
                margin=dict(l=160, r=80, t=70, b=60),
                hoverlabel=dict(bgcolor="white", font=dict(size=13)),
            )
    else:
        anomaly_df = _compute_anomaly(df)
        if anomaly_df.is_empty():
            raise ValueError(
                "No anomaly could be computed — every station mean is missing."
            )
        if static:
            fig = _static_anomaly_figure(
                plot_type, [("", anomaly_df)],
                title=chart_title, xlab=xlab, lang=lang, inclusive=inclusive,
                style=style, dpi=dpi, caption=caption, isave=isave,
                save_extension=save_extension,
            )
        else:
            fig = plot_fn(anomaly_df, colors, chart_title, xlab, lang)

    if static:
        # Captions and saving are handled inside the static builder.
        return fig

    if caption:
        fig.add_annotation(
            text=caption, xref="paper", yref="paper",
            x=0.5, y=-0.08, showarrow=False,
            font=dict(size=10, color="gray"),
        )

    fig = finalize_export(fig, style=style, isave=isave, save_extension=save_extension,
                           filename="lcz4r_anomaly_plot", lang=lang)

    return fig


__all__ = ["lcz_anomaly"]
