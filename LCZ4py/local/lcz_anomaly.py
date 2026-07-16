"""
lcz_anomaly.py — per-station temperature anomaly relative to the overall mean.

Uses Polars groupby for O(N) anomaly math and Plotly for interactive charts
coloured by LCZ class.
"""

from __future__ import annotations
import logging
import os
from typing import Optional, Union

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, normalise_input_df, normalise_missing,
    select_by_date, extract_lcz_at_points, lcz_palette, OUTPUT_DIR,
    add_by_column, by_sorted_groups,
)
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export

logger = logging.getLogger(__name__)

_VALID_PLOT_TYPES = ("diverging_bar", "bar", "dot", "lollipop", "scatter")


def _compute_anomaly(df: pl.DataFrame) -> pl.DataFrame:
    """Vectorised anomaly = station mean − overall mean, sorted ascending."""
    overall_mean = df["var_interp"].mean()
    return (
        df.group_by("station")
          .agg([pl.col("var_interp").mean().alias("mean_value"), pl.col("lcz").first()])
          .with_columns(reference_value=pl.lit(overall_mean))
          .with_columns(anomaly=(pl.col("mean_value") - pl.col("reference_value")).round(2))
          .sort("anomaly")
    )


def _base_layout(title: str, **kw) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
        **kw,
    )


def _plot_diverging_bar(df, colors, title, xlab, lang):
    fig = go.Figure(go.Bar(
        y=df["station"],
        x=df["anomaly"],
        orientation="h",
        marker_color=[colors.get(str(int(lv)), "#888888") for lv in df["lcz"]],
        marker_line_color="black",
        marker_line_width=0.5,
        text=[f"{a:.2f}" for a in df["anomaly"]],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>" + f"{xlab}: %{{x:.2f}}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title,
        xaxis=dict(
            title=dict(text=xlab, font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="#f0f0f0",
            zeroline=True, zerolinecolor="#aaaaaa", zerolinewidth=1,
        ),
        yaxis=dict(categoryorder="total ascending", tickfont=dict(size=11)),
        margin=dict(l=160, r=80, t=70, b=60),
        height=max(400, 30 * len(df)),
    ))
    return fig


def _plot_bar(df, colors, title, xlab, lang):
    bar_colors = ["#c0392b" if a >= 0 else "#2980b9" for a in df["anomaly"]]
    fig = go.Figure(go.Bar(
        x=df["station"],
        y=df["anomaly"],
        marker_color=bar_colors,
        marker_line_color="black",
        marker_line_width=0.5,
        text=[f"{a:.2f}" for a in df["anomaly"]],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{x}</b><br>" + f"{xlab}: %{{y:.2f}}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title,
        xaxis=dict(tickfont=dict(size=10), tickangle=-40, gridcolor="#f0f0f0"),
        yaxis=dict(
            title=dict(text=xlab, font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="#f0f0f0",
            zeroline=True, zerolinecolor="#aaaaaa", zerolinewidth=1,
        ),
        margin=dict(l=80, r=40, t=70, b=120),
        height=max(450, 20 * len(df)),
        showlegend=False,
    ))
    return fig


def _plot_dot(df, colors, title, xlab, lang):
    """Cleveland dot plot: mean_value (LCZ color) vs reference (gray), connected."""
    ref = float(df["reference_value"][0])
    temp_label = lcz_msg("temperature_label", lang)

    fig = go.Figure()
    # Connecting lines (mean → reference)
    for row in df.iter_rows(named=True):
        fig.add_shape(
            type="line",
            x0=row["mean_value"], x1=ref,
            y0=row["station"], y1=row["station"],
            line=dict(color="#cccccc", width=1.5),
        )
    # Reference dots (gray)
    fig.add_trace(go.Scatter(
        x=[ref] * len(df),
        y=df["station"],
        mode="markers",
        marker=dict(color="#999999", size=10, symbol="circle"),
        name=lcz_msg("temperature_label", lang) + " (ref)",
        hovertemplate=f"Reference: {ref:.2f}<extra></extra>",
    ))
    # Station mean dots (LCZ color)
    fig.add_trace(go.Scatter(
        x=df["mean_value"],
        y=df["station"],
        mode="markers",
        marker=dict(
            color=[colors.get(str(int(lv)), "#888888") for lv in df["lcz"]],
            size=12, symbol="circle",
            line=dict(color="black", width=0.5),
        ),
        name=temp_label,
        hovertemplate=(
            "<b>%{y}</b><br>"
            + f"{temp_label}: %{{x:.2f}}<br>"
            + f"{xlab}: %{{customdata:.2f}}<extra></extra>"
        ),
        customdata=df["anomaly"],
    ))
    fig.update_layout(**_base_layout(
        title,
        xaxis=dict(
            title=dict(text=temp_label, font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(categoryorder="total ascending", tickfont=dict(size=11)),
        margin=dict(l=160, r=80, t=70, b=60),
        height=max(400, 32 * len(df)),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.05),
    ))
    return fig


def _plot_lollipop(df, colors, title, xlab, lang):
    """Lollipop: vertical stem from 0 to anomaly, dot at tip."""
    stations = df["station"].to_list()
    anomalies = df["anomaly"].to_list()
    lcz_vals = df["lcz"].to_list()

    fig = go.Figure()
    # Stems
    for i, (st, a) in enumerate(zip(stations, anomalies)):
        fig.add_shape(
            type="line",
            x0=st, x1=st, y0=0, y1=a,
            line=dict(
                color=colors.get(str(int(lcz_vals[i])), "#888888"),
                width=2,
            ),
        )
    # Dots at tip
    fig.add_trace(go.Scatter(
        x=stations,
        y=anomalies,
        mode="markers",
        marker=dict(
            color=[colors.get(str(int(lv)), "#888888") for lv in lcz_vals],
            size=14,
            line=dict(color="black", width=0.8),
        ),
        hovertemplate="<b>%{x}</b><br>" + f"{xlab}: %{{y:.2f}}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**_base_layout(
        title,
        xaxis=dict(tickfont=dict(size=10), tickangle=-40, gridcolor="#f0f0f0"),
        yaxis=dict(
            title=dict(text=xlab, font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="#f0f0f0",
            zeroline=True, zerolinecolor="#aaaaaa", zerolinewidth=1,
        ),
        margin=dict(l=80, r=40, t=70, b=120),
        height=max(450, 20 * len(df)),
    ))
    return fig


def _plot_scatter(df, colors, title, xlab, lang):
    """Scatter: LCZ class (x) vs mean temperature (y), dot size = |anomaly|."""
    temp_label = lcz_msg("temperature_label", lang)
    fig = go.Figure(go.Scatter(
        x=df["lcz"].cast(str),
        y=df["mean_value"],
        mode="markers+text",
        text=df["station"],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            color=[colors.get(str(int(lv)), "#888888") for lv in df["lcz"]],
            size=[max(8, abs(a) * 10) for a in df["anomaly"]],
            line=dict(color="black", width=0.5),
            opacity=0.85,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "LCZ: %{x}<br>"
            + f"{temp_label}: %{{y:.2f}}<br>"
            + f"{xlab}: %{{customdata:.2f}}<extra></extra>"
        ),
        customdata=df["anomaly"],
    ))
    fig.update_layout(**_base_layout(
        title,
        xaxis=dict(
            title=dict(text="LCZ class", font=dict(size=14)),
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
) -> Union[go.Figure, pl.DataFrame]:
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
          circle at the tip sized by |anomaly|.
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

    Returns
    -------
    go.Figure or pl.DataFrame
    """
    if plot_type not in _PLOT_FNS:
        raise ValueError(f"plot_type must be one of {_VALID_PLOT_TYPES!r}, got {plot_type!r}")

    df = normalise_input_df(
        data_frame.to_pandas() if isinstance(data_frame, pl.DataFrame) else data_frame,
        var=var, station_id=station_id,
    )
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null())

    ds = load_lcz_raster(x)
    stations = df.unique(subset=["latitude", "longitude"])
    lcz_vals, mask = extract_lcz_at_points(
        ds, stations["longitude"].to_numpy(), stations["latitude"].to_numpy()
    )
    stations = stations.with_columns(
        pl.Series(name="lcz", values=np.where(mask, lcz_vals, 0)).cast(pl.Int32)
    )
    df = df.join(stations.select(["latitude", "longitude", "lcz"]), on=["latitude", "longitude"], how="left")
    df = df.with_columns(
        pl.col("station") + " (" + pl.col("lcz").cast(str) + ")"
    )
    df = df.filter(pl.col("lcz").is_between(1, 17))

    if not iplot:
        return _compute_anomaly(df)

    chart_title = title or lcz_msg("lcz_anomaly_title", lang)
    xlab = lcz_msg("anomaly_label", lang)
    colors = lcz_palette(inclusive=inclusive)
    plot_fn = _PLOT_FNS[plot_type]

    if by is not None:
        lat_mean = float(df["latitude"].mean())
        lon_mean = float(df["longitude"].mean())
        df = add_by_column(df, by, lat=lat_mean, lon=lon_mean)
        groups = by_sorted_groups(df["_by"], by)
        fig = make_subplots(
            rows=len(groups), cols=1,
            subplot_titles=groups,
            shared_xaxes=False,
            vertical_spacing=0.05,
        )
        for i, g in enumerate(groups, 1):
            sub = df.filter(pl.col("_by") == g)
            if len(sub) < 2:
                continue
            adf = _compute_anomaly(sub)
            sub_fig = plot_fn(adf, colors, g, xlab, lang)
            for trace in sub_fig.data:
                trace.showlegend = False
                fig.add_trace(trace, row=i, col=1)
        fig.update_layout(
            title=dict(text=chart_title, font=dict(size=20), x=0.5, xanchor="center"),
            height=max(500, 350 * len(groups)),
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
            margin=dict(l=160, r=80, t=70, b=60),
            hoverlabel=dict(bgcolor="white", font=dict(size=13)),
        )
    else:
        anomaly_df = _compute_anomaly(df)
        fig = plot_fn(anomaly_df, colors, chart_title, xlab, lang)

    if caption:
        fig.add_annotation(
            text=caption, xref="paper", yref="paper",
            x=0.5, y=-0.05, showarrow=False,
            font=dict(size=10, color="gray"),
        )

    fig = finalize_export(fig, style=style, isave=isave, save_extension=save_extension,
                           filename="lcz4r_anomaly_plot", lang=lang)

    return fig


__all__ = ["lcz_anomaly"]
