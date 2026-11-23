"""
lcz_temporal.py — Diurnal Temperature Range and Degree-Hours analysis
stratified by Local Climate Zone (LCZ) class.

Provides:
- ``lcz_dtr``  — Diurnal Temperature Range per station aggregated per LCZ class.
- ``lcz_degree_hours`` — Cooling / Heating Degree Hours per station aggregated
  per LCZ class.

Both functions optionally facet by a temporal grouping (``by=``), matching the
``by=`` semantics of :mod:`lcz_ts` and :mod:`lcz_anomaly`.

Rendering backends
------------------
- ``renderer="static"`` *(default)* — native matplotlib/Agg figure (box plot of
  DTR, or bar chart of degree hours) with LCZ colours and the urban/natural
  divider. No Plotly, Kaleido, Chromium or browser needed.
- ``renderer="plotly"`` — the interactive figure.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster,
    normalise_input_df,
    normalise_missing,
    select_by_date,
    assign_station_lcz,
    add_by_column,
    by_sorted_groups,
    require_nonempty,
)
from LCZ4py._internal.lcz_parameters_data import LCZ_COLORS, get_lcz_names
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export
from LCZ4py._internal import lcz_static
from LCZ4py._internal.lcz_static import resolve_renderer

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class DTRResult:
    """Return value of :func:`lcz_dtr`.

    Attributes
    ----------
    df : pl.DataFrame
        Per-station daily DTR with columns
        ``station, lcz, date, dtr, t_max, t_min, n_obs, lcz_name``.
    plot : matplotlib.figure.Figure or go.Figure
        Box plot of DTR by LCZ class (faceted when ``by=`` is used). A
        matplotlib ``Figure`` for the default static renderer, a Plotly
        ``go.Figure`` for ``renderer="plotly"``.
    renderer : str
        Which backend produced ``plot`` — ``"static"`` or ``"plotly"``.
    """

    df: pl.DataFrame
    plot: object
    renderer: str = "static"

    def show(self) -> None:
        """Display the figure in the active frontend (matplotlib or Plotly)."""
        if self.plot is None:
            return
        if type(self.plot).__module__.startswith("matplotlib"):
            import matplotlib.pyplot as plt

            plt.show()
        else:
            self.plot.show()


@dataclass
class DegreeHoursResult:
    """Return value of :func:`lcz_degree_hours`.

    Attributes
    ----------
    df : pl.DataFrame
        Per-station daily accumulated degree hours with columns
        ``station, lcz, date, degree_hours, n_obs, lcz_name``.
    total : pl.DataFrame
        Aggregated per LCZ class with columns
        ``lcz, mean, std, total, n, lcz_name``.
    plot : matplotlib.figure.Figure or go.Figure
        Bar chart of total degree hours per LCZ class (faceted when ``by=``).
        A matplotlib ``Figure`` for the default static renderer, a Plotly
        ``go.Figure`` for ``renderer="plotly"``.
    renderer : str
        Which backend produced ``plot`` — ``"static"`` or ``"plotly"``.
    """

    df: pl.DataFrame
    total: pl.DataFrame
    plot: object
    renderer: str = "static"

    def show(self) -> None:
        """Display the figure in the active frontend (matplotlib or Plotly)."""
        if self.plot is None:
            return
        if type(self.plot).__module__.startswith("matplotlib"):
            import matplotlib.pyplot as plt

            plt.show()
        else:
            self.plot.show()


# ── Helpers ───────────────────────────────────────────────────────────────────

_URBAN_IDS = set(range(1, 11))
_NATURAL_IDS = set(range(11, 18))


def _lcz_name_map(lang: str) -> dict[int, str]:
    """LCZ class int → localised name."""
    names = get_lcz_names(lang)
    return {i + 1: names[i] for i in range(len(names))}


def _lcz_name_frame(lang: str) -> pl.DataFrame:
    """Lookup frame used to attach ``lcz_name`` with a join (no map_elements)."""
    return pl.DataFrame({
        "lcz": pl.Series(list(range(1, 18)), dtype=pl.Int32),
        "lcz_name": pl.Series(get_lcz_names(lang), dtype=pl.String),
    })


def _base_layout(title: str, **kw) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
        margin=dict(l=80, r=40, t=90, b=60),
        **kw,
    )


def _lcz_color_map(inclusive: bool = False) -> dict[str, str]:
    """LCZ class string → hex colour (from the central palette, never hardcoded)."""
    from LCZ4py._internal.lcz_parameters_data import LCZ_COLORBLIND

    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    return {str(i + 1): c for i, c in enumerate(colors)}


def _group_label(lcz_int: int) -> str:
    return "Urban (1–10)" if lcz_int in _URBAN_IDS else "Natural (11–17)"


def _add_urban_natural_separator(fig, lcz_classes: list[int], lang: str,
                                row: Optional[int] = None,
                                col: Optional[int] = None) -> None:
    """Dashed divider + group labels between the urban and natural classes."""
    urban_max = max((c for c in lcz_classes if c in _URBAN_IDS), default=None)
    natural_min = min((c for c in lcz_classes if c in _NATURAL_IDS), default=None)
    if urban_max is None or natural_min is None:
        return

    fig.add_vline(
        x=f"LCZ {urban_max}", xref="x" if row is None else f"x{row}",
        line=dict(color="#cccccc", width=1, dash="dash"),
        row=row, col=col,
    )
    n_urban = sum(1 for c in lcz_classes if c in _URBAN_IDS)
    n_total = max(len(lcz_classes), 1)
    x_urban = n_urban / 2 / n_total
    x_natural = (n_urban + (n_total - n_urban) / 2) / n_total
    y = 1.10 if row is None else 1.05
    if row is None:
        fig.add_annotation(
            text=f"<b>{_xmsg('urban_label', lang)}</b>",
            xref="paper", yref="paper", x=x_urban, y=y,
            showarrow=False, font=dict(size=11, color="#555555"),
        )
        fig.add_annotation(
            text=f"<b>{_xmsg('natural_label', lang)}</b>",
            xref="paper", yref="paper", x=x_natural, y=y,
            showarrow=False, font=dict(size=11, color="#555555"),
        )


# ── i18n keys for this module ─────────────────────────────────────────────────

_EXTRA_MESSAGES: dict[str, dict[str, str]] = {
    "dtr_title": {
        "en": "Diurnal Temperature Range by LCZ Class",
        "pt": "Amplitude Térmica Diurna por Classe LCZ",
        "es": "Amplitud Térmica Diurna por Clase SCL",
        "zh": "按 LCZ 类型的日温差分布",
    },
    "dtr_ylabel": {
        "en": "DTR [°C]",
        "pt": "ATD [°C]",
        "es": "ATD [°C]",
        "zh": "日温差 [°C]",
    },
    "dtr_xlabel": {
        "en": "LCZ Class",
        "pt": "Classe LCZ",
        "es": "Clase SCL",
        "zh": "LCZ 类型",
    },
    "degree_hours_title_cooling": {
        "en": "Cooling Degree Hours by LCZ Class",
        "pt": "Graus-Hora de Refrigeração por Classe LCZ",
        "es": "Grados-Hora de Enfriamiento por Clase SCL",
        "zh": "按 LCZ 类型的制冷度时",
    },
    "degree_hours_title_heating": {
        "en": "Heating Degree Hours by LCZ Class",
        "pt": "Graus-Hora de Aquecimento por Classe LCZ",
        "es": "Grados-Hora de Calefacción por Clase SCL",
        "zh": "按 LCZ 类型的采暖度时",
    },
    "degree_hours_ylabel": {
        "en": "Degree Hours [°C·h]",
        "pt": "Graus-Hora [°C·h]",
        "es": "Grados-Hora [°C·h]",
        "zh": "度时 [°C·h]",
    },
    "degree_hours_xlabel": {
        "en": "LCZ Class",
        "pt": "Classe LCZ",
        "es": "Clase SCL",
        "zh": "LCZ 类型",
    },
    "urban_label": {
        "en": "Urban",
        "pt": "Urbano",
        "es": "Urbano",
        "zh": "城市",
    },
    "natural_label": {
        "en": "Natural",
        "pt": "Natural",
        "es": "Natural",
        "zh": "自然",
    },
    "observation_label": {
        "en": "Observations",
        "pt": "Observações",
        "es": "Observaciones",
        "zh": "观测数",
    },
    "station_count_label": {
        "en": "Station-days",
        "pt": "Estação-dias",
        "es": "Estación-días",
        "zh": "站点-日",
    },
}


def _xmsg(key: str, lang: str, **kwargs) -> str:
    """Look up a message first in the module-local table, then fall back to
    the central ``LCZ_MESSAGES``."""
    if key in _EXTRA_MESSAGES:
        tbl = _EXTRA_MESSAGES[key]
        template = tbl.get(lang, tbl["en"])
        if kwargs:
            for k, v in kwargs.items():
                template = template.replace("{" + k + "}", str(v))
        return template
    return lcz_msg(key, lang, **kwargs)


def _box_traces(
    fig: go.Figure,
    df: pl.DataFrame,
    value_col: str,
    colors: dict[str, str],
    ylab: str,
    lang: str,
    row: Optional[int] = None,
    col: Optional[int] = None,
    showlegend: bool = True,
) -> list[int]:
    """Add one Box trace per LCZ class (not per row) and return class order."""
    classes = [int(v) for v in df["lcz"].unique().sort().to_list()]
    if not classes:
        return classes
    partitions = df.partition_by("lcz", as_dict=True, maintain_order=True)
    for lcz_val in classes:
        sub = partitions.get((lcz_val,)) if (lcz_val,) in partitions else partitions.get(lcz_val)
        if sub is None or sub.is_empty():
            continue
        lcz_str = str(lcz_val)
        fig.add_trace(go.Box(
            y=sub[value_col],
            name=f"LCZ {lcz_str}",
            marker_color=colors.get(lcz_str, "#888888"),
            boxmean="sd",
            customdata=sub.select(["n_obs"]),
            hovertemplate=(
                f"<b>LCZ {lcz_str}</b><br>"
                f"{ylab}: %{{y:.2f}}<br>"
                f"{_xmsg('observation_label', lang)}: %{{customdata[0]}}"
                "<extra></extra>"
            ),
            showlegend=showlegend,
        ), row=row, col=col)
    return classes


def _class_totals(daily: pl.DataFrame) -> pl.DataFrame:
    """Per-LCZ summary of daily degree hours."""
    return (
        daily.group_by("lcz")
             .agg([
                 pl.col("degree_hours").mean().round(2).alias("mean"),
                 pl.col("degree_hours").std().round(2).alias("std"),
                 pl.col("degree_hours").sum().round(2).alias("total"),
                 pl.len().alias("n"),
             ])
             .with_columns(pl.col("std").fill_null(0.0))
             .sort("lcz")
    )


def _bar_traces(
    fig: go.Figure,
    totals: pl.DataFrame,
    colors: dict[str, str],
    ylab: str,
    lang: str,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> None:
    """One bar trace for all LCZ classes of *totals* (single trace, no loop)."""
    if totals.is_empty():
        return
    lcz_vals = [int(v) for v in totals["lcz"].to_list()]
    fig.add_trace(go.Bar(
        x=[f"LCZ {v}" for v in lcz_vals],
        y=totals["total"],
        marker_color=[colors.get(str(v), "#888888") for v in lcz_vals],
        marker_line_color="black",
        marker_line_width=0.5,
        error_y=dict(
            type="data",
            array=totals["std"].to_numpy(),
            visible=True,
        ),
        customdata=totals.select(["mean", "n"]),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{ylab}: %{{y:.2f}}<br>"
            f"mean: %{{customdata[0]:.2f}}<br>"
            f"{_xmsg('station_count_label', lang)}: %{{customdata[1]}}"
            "<extra></extra>"
        ),
    ), row=row, col=col)


# ── Native static renderer (matplotlib/Agg — no Plotly/Kaleido/Chromium) ─────

def _panel_axes(fig, n: int, *, has_header: bool, gap: float = 0.07,
                left: float = 0.115, right: float = 0.975):
    """Stacked full-width axes inside ``fig`` (top panel first).

    The top margin is pulled down a little further than ``figure_top`` so the
    Urban/Natural group labels drawn above the x axis stay clear of the
    figure title.
    """
    top = lcz_static.figure_top(has_header) - 0.05
    bottom = lcz_static.figure_bottom(False) - 0.02
    n = max(int(n), 1)
    span = max(top - bottom - gap * (n - 1), 0.10)
    height = span / n
    axes = []
    for i in range(n):
        bottom_i = top - (i + 1) * height - i * gap
        axes.append(fig.add_axes([left, bottom_i, right - left, height]))
    return axes


def _style_axis(ax, spec, grid_axis: str = "y") -> None:
    """Shared tick/spine styling for the static temporal charts."""
    ax.grid(True, axis=grid_axis, lw=0.35, color="0.88", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", labelsize=max(5.5, spec.font_size - 2.5),
                   length=3, width=0.6)


def _static_urban_natural_divider(ax, classes: list, lang: str, spec) -> None:
    """Dashed divider plus Urban/Natural group labels above the x axis."""
    urban = [c for c in classes if c in _URBAN_IDS]
    natural = [c for c in classes if c in _NATURAL_IDS]
    if not urban or not natural:
        return
    k = sum(1 for c in classes if c in _URBAN_IDS)
    total = len(classes)
    fontsize = max(5.5, spec.font_size - 2.0)
    ax.axvline(k + 0.5, color="#cccccc", lw=0.8, ls="--", zorder=1)
    trans = ax.get_xaxis_transform()
    ax.text((1 + k) / 2.0, 1.015, _xmsg("urban_label", lang), transform=trans,
            ha="center", va="bottom", fontsize=fontsize, color="#555555",
            fontweight="bold")
    ax.text((k + 1 + total) / 2.0, 1.015, _xmsg("natural_label", lang),
            transform=trans, ha="center", va="bottom", fontsize=fontsize,
            color="#555555", fontweight="bold")


def _static_dtr_figure(
    panels: list,
    *,
    title: Optional[str],
    ylab: str,
    xlab: str,
    lang: str = "en",
    inclusive: bool = False,
    style: str = "default",
    dpi: Optional[int] = None,
):
    """Box plot of DTR by LCZ class (mean ± sd), one axes per ``by=`` facet."""
    from matplotlib.colors import to_rgba

    from LCZ4py._internal.lcz_parameters_data import LCZ_COLORBLIND

    st = lcz_static
    panels = list(panels) or [("", pl.DataFrame({"lcz": [], "dtr": []}))]
    n = len(panels)
    height_mm = float(min(max(70.0 * n + 26.0, 80.0), 195.0))
    palette = LCZ_COLORBLIND if inclusive else LCZ_COLORS

    with st.style_context(style):
        fig, spec, _ = st.new_figure(style, height_mm=height_mm, dpi=dpi)
        label_size = max(6.0, spec.font_size - 1.5)
        axes = _panel_axes(fig, n, has_header=bool(title))

        for idx, (ax, panel) in enumerate(zip(axes, panels)):
            label, sub = panel
            if label:
                ax.set_title(str(label), fontsize=label_size, loc="left", pad=2.5)
            if sub.is_empty():
                continue

            classes = [int(v) for v in sub["lcz"].unique().sort().to_list()]
            partitions = sub.partition_by("lcz", as_dict=True, maintain_order=True)
            data, means, stds, positions, facecolors = [], [], [], [], []
            for pos, lcz_val in enumerate(classes, start=1):
                if (lcz_val,) in partitions:
                    part = partitions[(lcz_val,)]
                else:
                    part = partitions.get(lcz_val)
                if part is None or part.is_empty():
                    continue
                values = np.asarray(part["dtr"].to_numpy(), dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                data.append(values)
                means.append(float(values.mean()))
                stds.append(float(values.std(ddof=1)) if values.size > 1 else 0.0)
                positions.append(float(pos))
                facecolors.append(palette[(lcz_val - 1) % len(palette)])

            if not data:
                continue

            box = ax.boxplot(
                data, positions=positions, widths=0.62, patch_artist=True,
                showfliers=True, whis=1.5,
                medianprops=dict(color="black", lw=0.9),
                whiskerprops=dict(color="0.35", lw=0.7),
                capprops=dict(color="0.35", lw=0.7),
                flierprops=dict(marker=".", markersize=1.8, markerfacecolor="0.4",
                                markeredgecolor="0.4", alpha=0.7),
                boxprops=dict(edgecolor="black", linewidth=0.5),
            )
            for patch, color in zip(box["boxes"], facecolors):
                patch.set_facecolor(to_rgba(color, 0.85))

            ax.errorbar(positions, means, yerr=stds, fmt="D", ms=3.2,
                        color="white", ecolor="0.25", elinewidth=0.7,
                        capsize=1.8, markeredgecolor="black",
                        markeredgewidth=0.5, zorder=6)

            ax.set_xticks(positions)
            ax.set_xticklabels([f"LCZ {c}" for c in classes],
                               fontsize=max(5.5, spec.font_size - 2.5))
            ax.set_xlim(0.4, len(classes) + 0.6)
            ax.set_ylabel(ylab, fontsize=label_size)
            if idx == n - 1:
                ax.set_xlabel(xlab, fontsize=label_size)
            _style_axis(ax, spec)
            _static_urban_natural_divider(ax, classes, lang, spec)

        st.add_header(fig, title, None, spec)
    return fig


def _static_degree_hours_figure(
    panels: list,
    *,
    title: Optional[str],
    ylab: str,
    xlab: str,
    lang: str = "en",
    inclusive: bool = False,
    style: str = "default",
    dpi: Optional[int] = None,
):
    """Bar chart of total degree hours per LCZ class (± sd), one axes per facet."""
    st = lcz_static
    panels = list(panels) or [("", pl.DataFrame({"lcz": [], "total": []}))]
    n = len(panels)
    height_mm = float(min(max(68.0 * n + 26.0, 80.0), 195.0))
    colors = _lcz_color_map(inclusive=inclusive)

    with st.style_context(style):
        fig, spec, _ = st.new_figure(style, height_mm=height_mm, dpi=dpi)
        label_size = max(6.0, spec.font_size - 1.5)
        axes = _panel_axes(fig, n, has_header=bool(title))

        for idx, (ax, panel) in enumerate(zip(axes, panels)):
            label, totals = panel
            if label:
                ax.set_title(str(label), fontsize=label_size, loc="left", pad=2.5)
            if totals.is_empty():
                continue

            classes = [int(v) for v in totals["lcz"].to_list()]
            positions = np.arange(1, len(classes) + 1, dtype=float)
            values = np.asarray(totals["total"].to_numpy(), dtype=float)
            errors = np.asarray(totals["std"].fill_null(0.0).to_numpy(), dtype=float)
            ax.bar(
                positions, values, width=0.72,
                color=[colors.get(str(c), "#888888") for c in classes],
                edgecolor="black", linewidth=0.45, zorder=3,
            )
            ax.errorbar(positions, values, yerr=errors, fmt="none",
                        ecolor="0.25", elinewidth=0.7, capsize=1.8, zorder=4)
            ax.axhline(0, color="0.45", lw=0.7, zorder=2)
            ax.set_xticks(positions)
            ax.set_xticklabels([f"LCZ {c}" for c in classes],
                               fontsize=max(5.5, spec.font_size - 2.5))
            ax.set_xlim(0.4, len(classes) + 0.6)
            ax.set_ylabel(ylab, fontsize=label_size)
            if idx == n - 1:
                ax.set_xlabel(xlab, fontsize=label_size)
            _style_axis(ax, spec)
            _static_urban_natural_divider(ax, classes, lang, spec)

        st.add_header(fig, title, None, spec)
    return fig


def lcz_dtr(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
    data_frame=None,
    var: str = "",
    station_id: str = "",
    *,
    sp_res: float = 100.0,
    tp_res: str = "day",
    by: Optional[str] = None,
    style: str = "default",
    lang: str = "en",
    year=None,
    month=None,
    day=None,
    hour=None,
    start=None,
    end=None,
    inclusive: bool = False,
    renderer: Optional[str] = "static",
    dpi: Optional[int] = None,
) -> DTRResult:
    """Compute Diurnal Temperature Range (DTR) per station and aggregate by
    LCZ class.

    DTR is defined as ``max(T) − min(T)`` for each station on each calendar
    day.  The result DataFrame contains one row per station per day, and the
    figure shows a box plot of DTR values grouped by LCZ class with urban
    (1–10) and natural (11–17) distinguished.

    Parameters
    ----------
    x : str, PathLike, or rasterio dataset
        LCZ raster used to assign each station to an LCZ class.
    data_frame : pd.DataFrame or pl.DataFrame
        Observation table with lat/lon/date/var/station columns.
    var : str
        Column name for the meteorological variable (typically temperature).
    station_id : str
        Column name for the station identifier.
    sp_res : float
        Spatial resolution in metres (informational, unused internally).
    tp_res : str
        Temporal resolution label (informational; daily DTR is always computed
        from sub-daily observations).
    by : str, optional
        Split the analysis into faceted subplots by temporal group, e.g.
        ``"season"``, ``"month"``, ``"year"`` (see ``lcz_ts`` for the options).
    style : str
        Publication style preset: 'default', 'nature', 'science', or
        'generic_bw'. Controls font, figure size (mm), DPI, and palette
        used when isave and save_extension != 'html'.
    lang : str
        Language for labels — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.
    year, month, day, hour : int or sequence, optional
        Date/time component filters.
    start, end : str or datetime, optional
        Inclusive date range bounds.
    inclusive : bool
        Use the colorblind-friendly LCZ palette. Added at the end of the
        signature for backward compatibility.
    renderer : {"static", "plotly"}, optional
        ``"static"`` (default, also ``None``/``"matplotlib"``/``"agg"``)
        returns a matplotlib ``Figure`` (Agg canvas — no Plotly, Kaleido,
        Chromium or browser needed). ``"plotly"`` returns the interactive
        figure.
    dpi : int, optional
        Figure resolution for the static renderer (defaults to the style
        preset's dpi, 150 for ``"default"``).

    Returns
    -------
    DTRResult
        Dataclass with ``df`` (Polars DataFrame), ``plot`` (matplotlib or
        Plotly figure, see ``renderer``) and ``renderer``.

    Raises
    ------
    ValueError
        If ``data_frame`` is missing/empty, required columns are absent, or no
        station maps to LCZ class 1-17.
    """
    if data_frame is None:
        raise ValueError("data_frame is required.")

    renderer = resolve_renderer(renderer)
    static = renderer == "static"

    df = normalise_input_df(data_frame, var=var, station_id=station_id)
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null())
    require_nonempty(df, "observations")

    df = assign_station_lcz(df, load_lcz_raster(x))
    require_nonempty(df, "observations at stations inside the LCZ raster")

    name_frame = _lcz_name_frame(lang)

    # ── Compute daily DTR per station ──────────────────────────────────────
    daily = (
        df.with_columns(pl.col("date").dt.date().alias("date_day"))
          .group_by(["station", "lcz", "date_day"])
          .agg([
              pl.col("var_interp").max().alias("t_max"),
              pl.col("var_interp").min().alias("t_min"),
              pl.col("var_interp").count().alias("n_obs"),
          ])
          .with_columns(
              (pl.col("t_max") - pl.col("t_min")).round(2).alias("dtr")
          )
          .rename({"date_day": "date"})
          .join(name_frame, on="lcz", how="left")
          .sort(["station", "date"])
    )

    n_single = daily.filter(pl.col("n_obs") < 2).height
    if n_single:
        logger.warning(
            "%d station-day(s) have a single observation, so their DTR is 0 "
            "by construction (max == min). Feed sub-daily data for a "
            "meaningful DTR.", n_single,
        )

    # ── Figure (static matplotlib by default, Plotly on request) ───────────
    chart_title = _xmsg("dtr_title", lang)
    ylab = _xmsg("dtr_ylabel", lang)
    xlab = _xmsg("dtr_xlabel", lang)
    colors = _lcz_color_map(inclusive=inclusive)

    if by is not None:
        lat_mean = float(df["latitude"].mean())
        lon_mean = float(df["longitude"].mean())
        daily = add_by_column(daily, by, lat=lat_mean, lon=lon_mean)
        groups = by_sorted_groups(daily["_by"], by)
        if not groups:
            raise ValueError(f"by={by!r} produced no groups for this dataset.")
    else:
        groups = None

    if static:
        panels = (
            [("", daily)] if groups is None
            else [(g, daily.filter(pl.col("_by") == g)) for g in groups]
        )
        fig = _static_dtr_figure(
            panels, title=chart_title, ylab=ylab, xlab=xlab, lang=lang,
            inclusive=inclusive, style=style, dpi=dpi,
        )
    elif groups is None:
        fig = go.Figure()
        classes = _box_traces(fig, daily, "dtr", colors, ylab, lang)
        fig.update_layout(**_base_layout(
            chart_title,
            xaxis=dict(
                title=dict(text=xlab, font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor="#f0f0f0",
            ),
            yaxis=dict(
                title=dict(text=ylab, font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor="#f0f0f0",
            ),
            boxmode="group",
            showlegend=False,
        ))
        _add_urban_natural_separator(fig, classes, lang)
    else:
        fig = make_subplots(
            rows=len(groups), cols=1,
            subplot_titles=groups,
            shared_xaxes=False,
            vertical_spacing=0.06,
        )
        for i, g in enumerate(groups, 1):
            sub = daily.filter(pl.col("_by") == g)
            if sub.is_empty():
                continue
            classes = _box_traces(fig, sub, "dtr", colors, ylab, lang,
                                  row=i, col=1, showlegend=False)
            _add_urban_natural_separator(fig, classes, lang, row=i, col=1)
        fig.update_layout(**_base_layout(
            chart_title,
            height=max(380, 300 * len(groups)),
            xaxis=dict(title=dict(text=xlab, font=dict(size=14))),
            yaxis=dict(title=dict(text=ylab, font=dict(size=14))),
            boxmode="group",
            showlegend=False,
        ))

    if not static:
        fig = finalize_export(fig, style=style, lang=lang)

    return DTRResult(df=daily, plot=fig, renderer=renderer)


# ── lcz_degree_hours ──────────────────────────────────────────────────────────


def lcz_degree_hours(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
    data_frame=None,
    var: str = "",
    station_id: str = "",
    *,
    base_temp: float = 18.0,
    degree_type: str = "cooling",
    sp_res: float = 100.0,
    by: Optional[str] = None,
    style: str = "default",
    lang: str = "en",
    year=None,
    month=None,
    day=None,
    hour=None,
    start=None,
    end=None,
    inclusive: bool = False,
    renderer: Optional[str] = "static",
    dpi: Optional[int] = None,
) -> DegreeHoursResult:
    """Compute Cooling Degree Hours (CDH) or Heating Degree Hours (HDH)
    per station and aggregate by LCZ class.

    - **Cooling**: ``CDH = Σ max(0, T − base_temp)`` per observation,
      accumulated per day.
    - **Heating**: ``HDH = Σ max(0, base_temp − T)`` per observation,
      accumulated per day.

    The accumulation assumes hourly observations: with a different sampling
    interval the sum is per *observation*, not per clock hour. Check the
    ``n_obs`` column to confirm the sampling density.

    Parameters
    ----------
    x : str, PathLike, or rasterio dataset
        LCZ raster used to assign each station to an LCZ class.
    data_frame : pd.DataFrame or pl.DataFrame
        Observation table with lat/lon/date/var/station columns.
    var : str
        Column name for the meteorological variable (typically temperature).
    station_id : str
        Column name for the station identifier.
    base_temp : float
        Comfort threshold in °C (default 18.0).
    degree_type : {"cooling", "heating"}
        ``"cooling"`` → CDH, ``"heating"`` → HDH.
    sp_res : float
        Spatial resolution in metres (informational, unused internally).
    by : str, optional
        Split the analysis into faceted subplots by temporal group, e.g.
        ``"season"``, ``"month"``, ``"year"``.
    style : str
        Publication style preset: 'default', 'nature', 'science', or
        'generic_bw'. Controls font, figure size (mm), DPI, and palette
        used when isave and save_extension != 'html'.
    lang : str
        Language for labels — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.
    year, month, day, hour : int or sequence, optional
        Date/time component filters.
    start, end : str or datetime, optional
        Inclusive date range bounds.
    inclusive : bool
        Use the colorblind-friendly LCZ palette. Added at the end of the
        signature for backward compatibility.
    renderer : {"static", "plotly"}, optional
        ``"static"`` (default, also ``None``/``"matplotlib"``/``"agg"``)
        returns a matplotlib ``Figure`` (Agg canvas — no Plotly, Kaleido,
        Chromium or browser needed). ``"plotly"`` returns the interactive
        figure.
    dpi : int, optional
        Figure resolution for the static renderer (defaults to the style
        preset's dpi, 150 for ``"default"``).

    Returns
    -------
    DegreeHoursResult
        Dataclass with ``df``, ``total``, ``plot`` (matplotlib or Plotly
        figure, see ``renderer``) and ``renderer``.

    Raises
    ------
    ValueError
        On an unknown ``degree_type``, a non-finite ``base_temp``, empty input,
        missing columns, or when no station maps to LCZ class 1-17.
    """
    if data_frame is None:
        raise ValueError("data_frame is required.")

    renderer = resolve_renderer(renderer)
    static = renderer == "static"

    if degree_type not in ("cooling", "heating"):
        raise ValueError(f"degree_type must be 'cooling' or 'heating', got {degree_type!r}")
    if not np.isfinite(base_temp):
        raise ValueError(f"base_temp must be a finite number, got {base_temp!r}")

    df = normalise_input_df(data_frame, var=var, station_id=station_id)
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null())
    require_nonempty(df, "observations")

    df = assign_station_lcz(df, load_lcz_raster(x))
    require_nonempty(df, "observations at stations inside the LCZ raster")

    name_frame = _lcz_name_frame(lang)

    # ── Per-observation degree hours ───────────────────────────────────────
    if degree_type == "cooling":
        df = df.with_columns(
            pl.when(pl.col("var_interp") > base_temp)
              .then(pl.col("var_interp") - base_temp)
              .otherwise(0.0)
              .round(4)
              .alias("degree_hours")
        )
    else:
        df = df.with_columns(
            pl.when(pl.col("var_interp") < base_temp)
              .then(base_temp - pl.col("var_interp"))
              .otherwise(0.0)
              .round(4)
              .alias("degree_hours")
        )

    # ── Accumulate per day per station ─────────────────────────────────────
    daily = (
        df.with_columns(pl.col("date").dt.date().alias("date_day"))
          .group_by(["station", "lcz", "date_day"])
          .agg([
              pl.col("degree_hours").sum().round(2).alias("degree_hours"),
              pl.col("var_interp").count().alias("n_obs"),
          ])
          .rename({"date_day": "date"})
          .join(name_frame, on="lcz", how="left")
          .sort(["station", "date"])
    )

    # ── Figure (static matplotlib by default, Plotly on request) ───────────
    title_key = "degree_hours_title_cooling" if degree_type == "cooling" else "degree_hours_title_heating"
    chart_title = _xmsg(title_key, lang)
    ylab = _xmsg("degree_hours_ylabel", lang)
    xlab = _xmsg("degree_hours_xlabel", lang)
    colors = _lcz_color_map(inclusive=inclusive)

    if by is not None:
        lat_mean = float(df["latitude"].mean())
        lon_mean = float(df["longitude"].mean())
        daily = add_by_column(daily, by, lat=lat_mean, lon=lon_mean)
        groups = by_sorted_groups(daily["_by"], by)
        if not groups:
            raise ValueError(f"by={by!r} produced no groups for this dataset.")
        totals = [
            _class_totals(daily.filter(pl.col("_by") == g)).join(name_frame, on="lcz", how="left")
                .with_columns(pl.lit(g).alias("_by"))
            for g in groups
        ]
        total = (
            pl.concat([t for t in totals if not t.is_empty()], how="vertical")
            if any(not t.is_empty() for t in totals)
            else _class_totals(daily).join(name_frame, on="lcz", how="left")
        )
    else:
        groups = None
        total = _class_totals(daily).join(name_frame, on="lcz", how="left")

    if static:
        panels = (
            [("", total)] if groups is None
            else [(g, t) for g, t in zip(groups, totals)]
        )
        fig = _static_degree_hours_figure(
            panels, title=chart_title, ylab=ylab, xlab=xlab, lang=lang,
            inclusive=inclusive, style=style, dpi=dpi,
        )
    elif groups is None:
        fig = go.Figure()
        _bar_traces(fig, total, colors, ylab, lang)
        fig.update_layout(**_base_layout(
            chart_title,
            xaxis=dict(
                title=dict(text=xlab, font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor="#f0f0f0",
            ),
            yaxis=dict(
                title=dict(text=ylab, font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor="#f0f0f0",
            ),
            showlegend=False,
        ))
        _add_urban_natural_separator(fig, [int(v) for v in total["lcz"].to_list()], lang)
    else:
        fig = make_subplots(
            rows=len(groups), cols=1,
            subplot_titles=groups,
            shared_xaxes=False,
            vertical_spacing=0.06,
        )
        for i, (g, t) in enumerate(zip(groups, totals), 1):
            if t.is_empty():
                continue
            _bar_traces(fig, t, colors, ylab, lang, row=i, col=1)
            _add_urban_natural_separator(fig, [int(v) for v in t["lcz"].to_list()], lang,
                                          row=i, col=1)
        fig.update_layout(**_base_layout(
            chart_title,
            height=max(380, 300 * len(groups)),
            xaxis=dict(title=dict(text=xlab, font=dict(size=14))),
            yaxis=dict(title=dict(text=ylab, font=dict(size=14))),
            showlegend=False,
        ))

    if not static:
        fig = finalize_export(fig, style=style, lang=lang)

    return DegreeHoursResult(df=daily, total=total, plot=fig, renderer=renderer)


__all__ = ["lcz_dtr", "lcz_degree_hours", "DTRResult", "DegreeHoursResult"]
