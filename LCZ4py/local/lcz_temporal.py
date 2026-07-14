"""
lcz_temporal.py — Diurnal Temperature Range and Degree-Hours analysis
stratified by Local Climate Zone (LCZ) class.

Provides:
- ``lcz_dtr``  — Diurnal Temperature Range per station aggregated per LCZ class.
- ``lcz_degree_hours`` — Cooling / Heating Degree Hours per station aggregated
  per LCZ class.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import polars as pl
import plotly.graph_objects as go
import rasterio

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster,
    normalise_input_df,
    normalise_missing,
    select_by_date,
    extract_lcz_at_points,
    add_by_column,
    by_sorted_groups,
    OUTPUT_DIR,
)
from LCZ4py._internal.lcz_parameters_data import LCZ_COLORS, get_lcz_names
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class DTRResult:
    """Return value of :func:`lcz_dtr`.

    Attributes
    ----------
    df : pl.DataFrame
        Per-station daily DTR with columns
        ``station, lcz, date, dtr, t_max, t_min, lcz_name``.
    plot : go.Figure
        Box plot of DTR by LCZ class.
    """

    df: pl.DataFrame
    plot: go.Figure


@dataclass
class DegreeHoursResult:
    """Return value of :func:`lcz_degree_hours`.

    Attributes
    ----------
    df : pl.DataFrame
        Per-station daily accumulated degree hours with columns
        ``station, lcz, date, degree_hours, lcz_name``.
    total : pl.DataFrame
        Aggregated per LCZ class with columns
        ``lcz, lcz_name, mean, std, total``.
    plot : go.Figure
        Bar chart of total degree hours per LCZ class.
    """

    df: pl.DataFrame
    total: pl.DataFrame
    plot: go.Figure


# ── Helpers ───────────────────────────────────────────────────────────────────

_URBAN_IDS = set(range(1, 11))
_NATURAL_IDS = set(range(11, 18))


def _assign_lcz(df: pl.DataFrame, x: Union[str, os.PathLike, rasterio.io.DatasetReader]) -> pl.DataFrame:
    """Extract LCZ class for each unique station location and join back."""
    ds = load_lcz_raster(x)
    stations = df.unique(subset=["latitude", "longitude"])
    lcz_vals, mask = extract_lcz_at_points(
        ds, stations["longitude"].to_numpy(), stations["latitude"].to_numpy()
    )
    stations = stations.with_columns(
        pl.Series(name="lcz", values=np.where(mask, lcz_vals, 0)).cast(pl.Int32)
    )
    df = df.join(
        stations.select(["latitude", "longitude", "lcz"]),
        on=["latitude", "longitude"],
        how="left",
    )
    return df.filter(pl.col("lcz").is_between(1, 17))


def _lcz_name_map(lang: str) -> dict[int, str]:
    """LCZ class int → localised name."""
    names = get_lcz_names(lang)
    return {i + 1: names[i] for i in range(len(names))}


def _base_layout(title: str, **kw) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
        margin=dict(l=80, r=40, t=70, b=60),
        **kw,
    )


def _lcz_color_map() -> dict[str, str]:
    """LCZ class string → hex colour."""
    return {str(i + 1): c for i, c in enumerate(LCZ_COLORS)}


def _group_label(lcz_int: int) -> str:
    return "Urban (1–10)" if lcz_int in _URBAN_IDS else "Natural (11–17)"


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


# ── lcz_dtr ───────────────────────────────────────────────────────────────────


def lcz_dtr(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
    data_frame=None,
    var: str = "",
    station_id: str = "",
    *,
    sp_res: float = 100.0,
    tp_res: str = "day",
    by: Optional[str] = None,
    lang: str = "en",
    year=None,
    month=None,
    day=None,
    hour=None,
    start=None,
    end=None,
) -> DTRResult:
    """Compute Diurnal Temperature Range (DTR) per station and aggregate by
    LCZ class.

    DTR is defined as ``max(T) − min(T)`` for each station on each calendar
    day.  The result DataFrame contains one row per station per day, and the
    Plotly figure shows a box plot of DTR values grouped by LCZ class with
    urban (1–10) and natural (11–17) distinguished.

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
        Temporal resolution — ``"day"`` (default).
    by : str, optional
        Split the analysis into faceted subplots by temporal group.
    lang : str
        Language for labels — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.
    year, month, day, hour : int or sequence, optional
        Date/time component filters.
    start, end : str or datetime, optional
        Inclusive date range bounds.

    Returns
    -------
    DTRResult
        Named tuple with ``df`` (Polars DataFrame) and ``plot`` (Plotly Figure).
    """
    if data_frame is None:
        raise ValueError("data_frame is required.")

    df = normalise_input_df(
        data_frame.to_pandas() if isinstance(data_frame, pl.DataFrame) else data_frame,
        var=var,
        station_id=station_id,
    )
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null())

    df = _assign_lcz(df, x)

    name_map = _lcz_name_map(lang)

    # ── Compute daily DTR per station ──────────────────────────────────────
    daily = (
        df.with_columns(pl.col("date").dt.date().alias("date_day"))
          .group_by(["station", "lcz", "date_day"])
          .agg([
              pl.col("var_interp").max().alias("t_max"),
              pl.col("var_interp").min().alias("t_min"),
          ])
          .with_columns(
              (pl.col("t_max") - pl.col("t_min")).round(2).alias("dtr")
          )
          .with_columns(
              pl.col("lcz").map_elements(lambda v: name_map.get(int(v), ""), return_dtype=pl.Utf8)
              .alias("lcz_name")
          )
          .rename({"date_day": "date"})
          .sort(["station", "date"])
    )

    # ── Plotly box plot ────────────────────────────────────────────────────
    chart_title = _xmsg("dtr_title", lang)
    ylab = _xmsg("dtr_ylabel", lang)
    xlab = _xmsg("dtr_xlabel", lang)
    colors = _lcz_color_map()

    fig = go.Figure()
    lcz_classes = daily["lcz"].unique().sort().to_list()
    for lcz_val in lcz_classes:
        sub = daily.filter(pl.col("lcz") == lcz_val)
        if sub.is_empty():
            continue
        lcz_str = str(int(lcz_val))
        fig.add_trace(go.Box(
            y=sub["dtr"],
            name=f"LCZ {lcz_str}",
            marker_color=colors.get(lcz_str, "#888888"),
            boxmean="sd",
            hovertemplate=(
                f"<b>LCZ {lcz_str}</b><br>"
                f"{ylab}: " + "%{y:.2f}<extra></extra>"
            ),
        ))

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
    ))

    # Add a secondary x-axis or annotation to visually separate urban vs natural
    urban_max = max((c for c in lcz_classes if c in _URBAN_IDS), default=None)
    natural_min = min((c for c in lcz_classes if c in _NATURAL_IDS), default=None)
    if urban_max is not None and natural_min is not None:
        fig.add_vline(
            x=f"LCZ {urban_max}", xref="x",
            line=dict(color="#cccccc", width=1, dash="dash"),
        )
        fig.add_annotation(
            text=f"<b>{_xmsg('urban_label', lang)}</b>",
            xref="paper", yref="paper",
            x=0.15, y=1.08, showarrow=False,
            font=dict(size=11, color="#555555"),
        )
        fig.add_annotation(
            text=f"<b>{_xmsg('natural_label', lang)}</b>",
            xref="paper", yref="paper",
            x=0.85, y=1.08, showarrow=False,
            font=dict(size=11, color="#555555"),
        )

    return DTRResult(df=daily, plot=fig)


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
    lang: str = "en",
    year=None,
    month=None,
    day=None,
    hour=None,
    start=None,
    end=None,
) -> DegreeHoursResult:
    """Compute Cooling Degree Hours (CDH) or Heating Degree Hours (HDH)
    per station and aggregate by LCZ class.

    - **Cooling**: ``CDH = Σ max(0, T − base_temp)`` per hour, accumulated
      per day.
    - **Heating**: ``HDH = Σ max(0, base_temp − T)`` per hour, accumulated
      per day.

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
        Split the analysis into faceted subplots by temporal group.
    lang : str
        Language for labels — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.
    year, month, day, hour : int or sequence, optional
        Date/time component filters.
    start, end : str or datetime, optional
        Inclusive date range bounds.

    Returns
    -------
    DegreeHoursResult
        Named tuple with ``df``, ``total``, and ``plot``.
    """
    if data_frame is None:
        raise ValueError("data_frame is required.")

    if degree_type not in ("cooling", "heating"):
        raise ValueError(f"degree_type must be 'cooling' or 'heating', got {degree_type!r}")

    df = normalise_input_df(
        data_frame.to_pandas() if isinstance(data_frame, pl.DataFrame) else data_frame,
        var=var,
        station_id=station_id,
    )
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null())

    df = _assign_lcz(df, x)

    name_map = _lcz_name_map(lang)

    # ── Hourly degree hours ────────────────────────────────────────────────
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
          .agg(pl.col("degree_hours").sum().round(2).alias("degree_hours"))
          .with_columns(
              pl.col("lcz").map_elements(lambda v: name_map.get(int(v), ""), return_dtype=pl.Utf8)
              .alias("lcz_name")
          )
          .rename({"date_day": "date"})
          .sort(["station", "date"])
    )

    # ── Aggregate per LCZ class ────────────────────────────────────────────
    total = (
        daily.group_by("lcz")
             .agg([
                 pl.col("degree_hours").mean().round(2).alias("mean"),
                 pl.col("degree_hours").std().round(2).alias("std"),
                 pl.col("degree_hours").sum().round(2).alias("total"),
             ])
             .with_columns(
                 pl.col("lcz").map_elements(lambda v: name_map.get(int(v), ""), return_dtype=pl.Utf8)
                 .alias("lcz_name")
             )
             .sort("lcz")
    )
    total = total.with_columns(
        pl.when(pl.col("std").is_null()).then(0.0).otherwise(pl.col("std")).alias("std")
    )

    # ── Plotly bar chart ───────────────────────────────────────────────────
    title_key = "degree_hours_title_cooling" if degree_type == "cooling" else "degree_hours_title_heating"
    chart_title = _xmsg(title_key, lang)
    ylab = _xmsg("degree_hours_ylabel", lang)
    xlab = _xmsg("degree_hours_xlabel", lang)
    colors = _lcz_color_map()

    lcz_vals = total["lcz"].to_list()
    bar_colors = [
        colors.get(str(int(v)), "#888888") for v in lcz_vals
    ]

    fig = go.Figure(go.Bar(
        x=[f"LCZ {int(v)}" for v in lcz_vals],
        y=total["total"],
        marker_color=bar_colors,
        marker_line_color="black",
        marker_line_width=0.5,
        error_y=dict(
            type="data",
            array=total["std"],
            visible=True,
        ),
        hovertemplate=(
            "<b>LCZ %{x}</b><br>"
            + f"{ylab}: " + "%{y:.2f}<extra></extra>"
        ),
    ))

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

    # Urban / natural grouping annotation
    urban_lcz = [v for v in lcz_vals if int(v) in _URBAN_IDS]
    natural_lcz = [v for v in lcz_vals if int(v) in _NATURAL_IDS]
    if urban_lcz and natural_lcz:
        fig.add_vline(
            x=f"LCZ {int(max(urban_lcz))}", xref="x",
            line=dict(color="#cccccc", width=1, dash="dash"),
        )
        mid_urban = len(urban_lcz) / 2 / max(len(lcz_vals), 1)
        mid_natural = (len(urban_lcz) + len(natural_lcz) / 2) / max(len(lcz_vals), 1)
        fig.add_annotation(
            text=f"<b>{_xmsg('urban_label', lang)}</b>",
            xref="paper", yref="paper",
            x=mid_urban, y=1.08, showarrow=False,
            font=dict(size=11, color="#555555"),
        )
        fig.add_annotation(
            text=f"<b>{_xmsg('natural_label', lang)}</b>",
            xref="paper", yref="paper",
            x=mid_natural, y=1.08, showarrow=False,
            font=dict(size=11, color="#555555"),
        )

    return DegreeHoursResult(df=daily, total=total, plot=fig)


__all__ = ["lcz_dtr", "lcz_degree_hours", "DTRResult", "DegreeHoursResult"]
