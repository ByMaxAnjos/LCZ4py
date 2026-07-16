"""
lcz_uhi_intensity.py — Urban Heat Island intensity from LCZ-stratified stations.

Urban (LCZ 1–10) and rural (LCZ 11–16) stations are either chosen automatically
from the LCZ raster or supplied manually. UHI = T_urban − T_rural.

Plotting uses Plotly with optional dual y-axes (urban/rural temps + UHI).
"""

from __future__ import annotations
import os
from typing import Optional, Sequence, Union

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, normalise_input_df, normalise_missing,
    select_by_date, extract_lcz_at_points, lcz_palette, OUTPUT_DIR,
    add_by_column, by_sorted_groups, _break_gaps_numpy,
)
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export

URBAN_CLASSES = [str(i) for i in range(1, 11)]
RURAL_CLASSES = [str(i) for i in range(11, 17)]


# ── Time averaging ────────────────────────────────────────────────────────────

def _time_avg(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    freq_map = {"hour": "1h", "day": "1d", "month": "1mo", "year": "1y"}
    return (
        df.sort("date")
          .group_by_dynamic("date", every=freq_map.get(freq, "1h"), closed="left")
          .agg(pl.col("var_interp").mean())
    )


# ── Plotly figures ────────────────────────────────────────────────────────────

def _plot_single(df: pl.DataFrame, title: str, xlab: str, ylab: str) -> go.Figure:
    fig = go.Figure()
    dates_np = df["date"].to_numpy()
    vals_np = df["uhi"].to_numpy()
    if len(dates_np) > 2:
        diffs_h = np.diff(dates_np.astype("datetime64[s]")).astype(np.float64) / 3600
        pos = diffs_h[diffs_h > 0]
        threshold_h = max(2.0 * float(np.median(pos)), 1.5) if len(pos) else 2.0
        dates_p, vals_p = _break_gaps_numpy(dates_np, vals_np, threshold_h)
    else:
        dates_p, vals_p = list(dates_np), list(vals_np)
    fig.add_trace(go.Scatter(
        x=dates_p, y=vals_p,
        mode="lines", name="UHI",
        line=dict(color="#333333", width=1.5),
        hovertemplate=f"{ylab}: %{{y:.2f}}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=xlab, font=dict(size=14)), tickfont=dict(size=12), gridcolor="#f0f0f0"),
        yaxis=dict(title=dict(text=ylab, font=dict(size=14)), tickfont=dict(size=12), gridcolor="#f0f0f0"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=70, b=60),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    return fig


def _plot_dual(
    df: pl.DataFrame,
    u_color: str,
    r_color: str,
    title: str,
    xlab: str,
    ylab: str,
    ylab2: str,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dates_np = df["date"].to_numpy()
    if len(dates_np) > 2:
        diffs_h = np.diff(dates_np.astype("datetime64[s]")).astype(np.float64) / 3600
        pos = diffs_h[diffs_h > 0]
        threshold_h = max(2.0 * float(np.median(pos)), 1.5) if len(pos) else 2.0
    else:
        threshold_h = 2.0

    def _gapped_series(col: str):
        return _break_gaps_numpy(dates_np, df[col].to_numpy(), threshold_h)

    u_d, u_v = _gapped_series("urban")
    r_d, r_v = _gapped_series("rural")
    h_d, h_v = _gapped_series("uhi")

    fig.add_trace(
        go.Scatter(x=u_d, y=u_v, name="Urban",
                   line=dict(color=u_color, width=1.5),
                   hovertemplate=f"Urban {ylab}: %{{y:.2f}}<extra></extra>"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=r_d, y=r_v, name="Rural",
                   line=dict(color=r_color, width=1.5),
                   hovertemplate=f"Rural {ylab}: %{{y:.2f}}<extra></extra>"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=h_d, y=h_v, name="UHI",
                   line=dict(color="#333333", width=1.5, dash="dot"),
                   hovertemplate=f"{ylab2}: %{{y:.2f}}<extra></extra>"),
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text=ylab, secondary_y=False,
        gridcolor="#f0f0f0", tickfont=dict(size=12),
    )
    fig.update_yaxes(
        title_text=ylab2, secondary_y=True,
        tickfont=dict(size=12),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=xlab, font=dict(size=14)), tickfont=dict(size=12), gridcolor="#f0f0f0"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=70, b=60),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    return fig


# ── Public API ────────────────────────────────────────────────────────────────

def lcz_uhi_intensity(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
    data_frame=None,
    var: str = "",
    station_id: str = "",
    *,
    time_freq: str = "hour",
    by: Optional[str] = None,
    method: str = "LCZ",
    T_urban: Optional[Sequence[str]] = None,
    T_rural: Optional[Sequence[str]] = None,
    group: bool = False,
    iplot: bool = True,
    isave: bool = False,
    save_extension: str = "html",
    style: str = "default",
    ylab: Optional[str] = None,
    xlab: Optional[str] = None,
    ylab2: Optional[str] = None,
    title: Optional[str] = None,
    caption: Optional[str] = None,
    lang: str = "en",
    year=None, month=None, day=None, hour=None, start=None, end=None,
) -> Union[go.Figure, pl.DataFrame]:
    """Compute and plot Urban Heat Island intensity.

    UHI intensity is defined as the mean temperature of urban stations minus
    the mean temperature of rural stations, averaged over ``time_freq``.

    Parameters
    ----------
    x : str, PathLike, or rasterio dataset
        LCZ raster used to classify stations as urban or rural.
    data_frame : pd.DataFrame or pl.DataFrame
        Observation table with lat/lon/date/var/station columns.
    var : str
        Column name for the meteorological variable.
    station_id : str
        Column name for the station identifier.
    time_freq : {"hour", "day", "month"}
        Time-averaging frequency.
    by : str, optional
        Split the UHI computation into faceted subplots by temporal group.
        Options: ``"year"``, ``"season"``, ``"seasonyear"``, ``"month"``,
        ``"monthyear"``, ``"weekday"``, ``"weekend"``, ``"site"``,
        ``"daylight"`` (UTC, NOAA formula), ``"dst"`` (approximate).
        ``None`` = single chart (default).
    method : {"LCZ", "manual"}
        ``"LCZ"`` — auto-assign urban/rural from LCZ classes;
        ``"manual"`` — use ``T_urban``/``T_rural`` station lists.
    T_urban : list of str, optional
        Station IDs to treat as urban (``method="manual"`` only).
    T_rural : list of str, optional
        Station IDs to treat as rural (``method="manual"`` only).
    group : bool
        If True, plot urban + rural temperatures and UHI on dual axes.
        If False, plot only the UHI series.
    iplot : bool
        If False, return the wide-format DataFrame instead of a figure.
    isave : bool
        Save the figure to ``LCZ4r_output/``.
    save_extension : str
        ``"html"`` (interactive) or ``"png"``/``"pdf"`` (static).
    style : str
        Publication style preset: "default", "nature", "science", or
        "generic_bw". Controls font, figure size (mm), DPI, and palette
        used when isave and save_extension != "html".
    ylab, xlab, ylab2 : str, optional
        Axis labels; default to translated strings via ``lang``.
    title : str, optional
        Override the default chart title.
    caption : str, optional
        Small annotation added below the chart.
    lang : str
        Language for labels — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.

    Returns
    -------
    go.Figure or pl.DataFrame
        Wide DataFrame has columns: date, urban, rural, uhi.
    """
    if data_frame is None:
        raise ValueError("data_frame is required.")

    df = normalise_input_df(
        data_frame.to_pandas() if isinstance(data_frame, pl.DataFrame) else data_frame,
        var=var, station_id=station_id,
    )
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null() & pl.col("latitude").is_not_null())

    ds = load_lcz_raster(x)
    stations = df.unique(subset=["latitude", "longitude"])
    lcz_vals, mask = extract_lcz_at_points(
        ds, stations["longitude"].to_numpy(), stations["latitude"].to_numpy()
    )
    stations = stations.with_columns(
        pl.Series(name="lcz", values=np.where(mask, lcz_vals, 0))
    )
    df = df.join(stations.select(["latitude", "longitude", "lcz"]), on=["latitude", "longitude"], how="left")
    df = df.filter(pl.col("lcz").is_between(1, 17)).with_columns(pl.col("lcz").cast(str))

    if method == "LCZ":
        u_ids = df.filter(pl.col("lcz").is_in(URBAN_CLASSES))["station"].unique().to_list()
        r_ids = df.filter(pl.col("lcz").is_in(RURAL_CLASSES))["station"].unique().to_list()
        if not u_ids or not r_ids:
            raise ValueError(lcz_msg("no_urban_rural", lang))
    else:
        u_ids = [str(s) for s in (T_urban or [])]
        r_ids = [str(s) for s in (T_rural or [])]

    palette = lcz_palette()
    u_lcz = df.filter(pl.col("station").is_in(u_ids))["lcz"].mode()[0]
    r_lcz = df.filter(pl.col("station").is_in(r_ids))["lcz"].mode()[0]
    u_color = palette.get(u_lcz, "#CC3333")
    r_color = palette.get(r_lcz, "#3366CC")

    model_df = df.filter(pl.col("station").is_in(u_ids + r_ids)).with_columns(
        pl.when(pl.col("station").is_in(u_ids))
        .then(pl.lit("urban"))
        .otherwise(pl.lit("rural"))
        .alias("reference")
    )

    freq_map = {"hour": "1h", "day": "1d", "month": "1mo", "year": "1y"}
    p_freq = freq_map.get(time_freq, "1h")

    if by is not None:
        lat_mean = float(df["latitude"].mean())
        lon_mean = float(df["longitude"].mean())
        model_df = add_by_column(model_df, by, lat=lat_mean, lon=lon_mean)
        group_by_cols = ["reference", "_by"]
        pivot_index = ["date", "_by"]
    else:
        group_by_cols = "reference"
        pivot_index = "date"

    avg = (
        model_df.sort("date")
        .group_by_dynamic("date", every=p_freq, closed="left", group_by=group_by_cols)
        .agg(pl.col("var_interp").mean())
    )
    wide = avg.pivot(index=pivot_index, on="reference", values="var_interp").drop_nulls(
        subset=["urban", "rural"]
    )
    wide = wide.with_columns(
        (pl.col("urban") - pl.col("rural")).round(2).alias("uhi")
    )

    if not iplot:
        return wide

    chart_title = title or lcz_msg("lcz_uhi_title", lang)
    x_label = xlab or lcz_msg("time_label", lang)
    y_label = ylab or lcz_msg("temperature_label", lang)
    y2_label = ylab2 or lcz_msg("uhi_label", lang)

    if by is not None:
        groups = by_sorted_groups(wide["_by"], by)
        n = len(groups)
        specs = [[{"secondary_y": True}]] * n if group else [[{}]] * n
        fig = make_subplots(
            rows=n, cols=1,
            subplot_titles=groups,
            shared_xaxes=False,
            vertical_spacing=0.05,
            specs=specs,
        )
        for i, g in enumerate(groups, 1):
            sub = wide.filter(pl.col("_by") == g)
            if sub.is_empty():
                continue
            if group:
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub["urban"], name=f"Urban ({g})",
                    line=dict(color=u_color, width=1.5), showlegend=(i == 1),
                    hovertemplate=f"Urban {y_label}: %{{y:.2f}}<extra></extra>",
                ), row=i, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub["rural"], name=f"Rural ({g})",
                    line=dict(color=r_color, width=1.5), showlegend=(i == 1),
                    hovertemplate=f"Rural {y_label}: %{{y:.2f}}<extra></extra>",
                ), row=i, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub["uhi"], name=f"UHI ({g})",
                    line=dict(color="#333333", width=1.5, dash="dot"), showlegend=(i == 1),
                    hovertemplate=f"{y2_label}: %{{y:.2f}}<extra></extra>",
                ), row=i, col=1, secondary_y=True)
            else:
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub["uhi"], name=f"UHI ({g})",
                    line=dict(color="#333333", width=1.5), showlegend=(i == 1),
                    hovertemplate=f"{y2_label}: %{{y:.2f}}<extra></extra>",
                ), row=i, col=1)
        fig.update_layout(
            title=dict(text=chart_title, font=dict(size=20), x=0.5, xanchor="center"),
            height=max(400, 280 * n),
            plot_bgcolor="#fafafa", paper_bgcolor="white",
            margin=dict(l=80, r=80 if group else 40, t=70, b=60),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="white", font=dict(size=13)),
        )
    else:
        fig = (
            _plot_dual(wide, u_color, r_color, chart_title, x_label, y_label, y2_label)
            if group else
            _plot_single(wide, chart_title, x_label, y2_label)
        )

    if caption:
        fig.add_annotation(
            text=caption, xref="paper", yref="paper",
            x=0.5, y=-0.05, showarrow=False,
            font=dict(size=10, color="gray"),
        )

    fig = finalize_export(fig, style=style, isave=isave, save_extension=save_extension,
                           filename="lcz4r_uhi_plot", lang=lang)

    return fig


__all__ = ["lcz_uhi_intensity"]
