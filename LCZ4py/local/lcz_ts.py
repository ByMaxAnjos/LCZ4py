"""
lcz_ts.py — LCZ-stratified time-series visualisation.

Handles large datasets efficiently:
- Polars lazy groupby for time averaging
- Plotly WebGL (Scattergl) for smooth panning/zooming
- Datashader pre-rendering for massive arrays (>4M points), if installed
- Automatic 10× decimation when a single station exceeds 100k points
"""

from __future__ import annotations
import datetime as _dt_mod
import os
from typing import Optional, Union

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    import xarray as xr
    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, normalise_input_df, normalise_missing,
    select_by_date, extract_lcz_at_points, OUTPUT_DIR,
    add_by_column, by_sorted_groups,
    _break_gaps_numpy, _daylight_intervals_batch,
)
from LCZ4py.general.lcz_plot_parameters import _palette_to_plotly_colorscale
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export


def _station_colors(stations, palette_name: str) -> list[str]:
    colorscale = _palette_to_plotly_colorscale(palette_name, n=max(len(stations), 2))
    step = max(1, len(colorscale) // max(len(stations), 1))
    return [colorscale[min(i * step, len(colorscale) - 1)][1] for i in range(len(stations))]


def _plot_basic_line(
    df: pl.DataFrame,
    palette_name: str,
    title: str,
    xlab: str,
    ylab: str,
) -> go.Figure:
    fig = go.Figure()
    stations = df["station"].unique().sort()
    colors = _station_colors(stations, palette_name)
    for i, st in enumerate(stations):
        sub = df.filter(pl.col("station") == st).sort("date")
        if len(sub) > 100_000:
            sub = sub.with_columns(pl.col("date").rank(method="ordinal").alias("_rank"))
            sub = sub.filter((pl.col("_rank") % 10) == 0).drop("_rank")

        dates_np = sub["date"].to_numpy()
        vals_np = sub["var_interp"].to_numpy()

        # Break the line at large time gaps to prevent false diagonals
        if len(dates_np) > 2:
            diffs_h = np.diff(dates_np.astype("datetime64[s]")).astype(np.float64) / 3600
            pos = diffs_h[diffs_h > 0]
            median_step_h = float(np.median(pos)) if len(pos) else 1.0
            threshold_h = max(2.0 * median_step_h, 1.5)
        else:
            threshold_h = 2.0
        dates_plot, vals_plot = _break_gaps_numpy(dates_np, vals_np, threshold_h)

        fig.add_trace(go.Scattergl(
            x=dates_plot, y=vals_plot,
            name=st,
            line=dict(color=colors[i], width=1),
            hovertemplate=f"<b>{st}</b><br>{xlab}: %{{x}}<br>{ylab}: %{{y:.2f}}<extra></extra>",
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


def _plot_heatmap(
    df: pl.DataFrame,
    facet_var: str,
    palette_name: str,
    title: str,
    xlab: str,
    ylab: str,
) -> go.Figure:
    facets = df[facet_var].unique().sort()
    fig = make_subplots(
        rows=len(facets), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[str(f) for f in facets],
    )
    for i, f in enumerate(facets):
        sub = df.filter(pl.col(facet_var) == f).sort("date")
        fig.add_trace(go.Heatmap(
            z=sub["var_interp"].to_numpy().reshape(1, -1),
            x=sub["date"],
            y=[str(f)],
            colorscale="Viridis",
            showscale=(i == 0),
            name=str(f),
            hovertemplate=f"{ylab}: %{{y}}<br>{xlab}: %{{x}}<br>Value: %{{z:.2f}}<extra></extra>",
        ), row=i + 1, col=1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        height=max(300, 200 * len(facets)),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=70, b=60),
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    return fig


def _faceted_line_fig(
    avg: pl.DataFrame,
    groups: list[str],
    palette_name: str,
    chart_title: str,
    x_label: str,
    y_label: str,
) -> go.Figure:
    """Build a multi-row subplot figure, one row per by-group."""
    fig = make_subplots(
        rows=len(groups), cols=1,
        subplot_titles=groups,
        shared_xaxes=False,
        vertical_spacing=0.04,
    )
    seen_stations: set[str] = set()
    for i, g in enumerate(groups, 1):
        sub = avg.filter(pl.col("_by") == g)
        sub_fig = _plot_basic_line(sub, palette_name, g, x_label, y_label)
        for trace in sub_fig.data:
            trace.showlegend = trace.name not in seen_stations
            seen_stations.add(trace.name)
            fig.add_trace(trace, row=i, col=1)
    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=20), x=0.5, xanchor="center"),
        height=max(400, 280 * len(groups)),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=70, b=60),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    return fig


def _plot_daylight_unified(
    df: pl.DataFrame,
    palette_name: str,
    title: str,
    xlab: str,
    ylab: str,
    lat: float,
    lon: float,
) -> go.Figure:
    """Full time series with amber vrects marking each daytime window.

    No subplot split — daytime/nighttime are shown on a single axis with
    shaded background regions computed from the NOAA solar elevation formula.
    Valid for all latitudes (polar day/night handled gracefully).
    """
    fig = _plot_basic_line(df, palette_name, title, xlab, ylab)

    # Unique calendar dates present in the dataset
    unique_dates = sorted({
        _dt_mod.date(d.year, d.month, d.day)
        for d in df["date"].to_list()
    })
    intervals_per_day = _daylight_intervals_batch(lat, lon, unique_dates)

    for d, intervals in zip(unique_dates, intervals_per_day):
        base = _dt_mod.datetime(d.year, d.month, d.day)
        for (rise_h, set_h) in intervals:
            fig.add_vrect(
                x0=base + _dt_mod.timedelta(hours=rise_h),
                x1=base + _dt_mod.timedelta(hours=set_h),
                fillcolor="rgba(255, 200, 50, 0.13)",
                line_width=0,
                layer="below",
            )

    # Proxy legend entry explaining the amber shading
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="square", size=12, color="rgba(255, 200, 50, 0.55)"),
        name="Daytime",
    ))
    return fig


def lcz_ts(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
    data_frame=None,
    var: str = "",
    station_id: str = "",
    *,
    time_freq: str = "hour",
    by: Optional[str] = None,
    plot_type: str = "basic_line",
    facet_plot: str = "lcz",
    iplot: bool = True,
    isave: bool = False,
    save_extension: str = "html",
    style: str = "default",
    palette: str = "VanGogh2",
    ylab: Optional[str] = None,
    xlab: Optional[str] = None,
    title: Optional[str] = None,
    caption: Optional[str] = None,
    lang: str = "en",
    year=None, month=None, day=None, hour=None, start=None, end=None,
) -> Union[go.Figure, pl.DataFrame]:
    """Render LCZ-stratified time-series charts.

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
    time_freq : {"hour", "day", "month"}
        Time-averaging frequency.
    by : str, optional
        Split the time series into faceted subplots by temporal group.
        Options: ``"year"``, ``"season"``, ``"seasonyear"``, ``"month"``,
        ``"monthyear"``, ``"weekday"``, ``"weekend"``, ``"site"``,
        ``"dst"`` (approximate, hemisphere-aware).
        Special: ``"daylight"`` produces a single chart with amber shading
        for daytime windows (UTC, NOAA solar elevation formula, valid globally).
        ``None`` = single chart (default).
    plot_type : {"basic_line", "heatmap"}
        Chart kind.  ``"heatmap"`` is not supported with ``by=``.
    facet_plot : str
        Column for the heatmap facet variable (heatmap only).
    iplot : bool
        If False, return the averaged DataFrame instead of a figure.
    isave : bool
        Save the figure to ``LCZ4r_output/``.
    save_extension : str
        ``"html"`` (interactive) or ``"png"``/``"pdf"`` (static).
    style : str
        Publication style preset: "default", "nature", "science", or
        "generic_bw". Controls font, figure size (mm), DPI, and palette
        used when isave and save_extension != "html".
    palette : str
        MetBrewer palette name for line colours.
    ylab, xlab : str, optional
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
    df = df.filter(pl.col("var_interp").is_not_null())

    ds_raster = load_lcz_raster(x)
    stations = df.unique(subset=["latitude", "longitude"])
    lcz_vals, mask = extract_lcz_at_points(
        ds_raster, stations["longitude"].to_numpy(), stations["latitude"].to_numpy()
    )
    stations = stations.with_columns(
        pl.Series(name="lcz", values=np.where(mask, lcz_vals, 0)).cast(pl.Int32)
    )
    df = df.join(stations.select(["latitude", "longitude", "lcz"]), on=["latitude", "longitude"], how="left")
    df = df.filter(pl.col("lcz").is_between(1, 17))
    df = df.with_columns(
        (pl.col("station") + " (" + pl.col("lcz").cast(str) + ")").alias("station"),
        pl.col("lcz").cast(str),
    )

    lat_mean = float(df["latitude"].mean())
    lon_mean = float(df["longitude"].mean())

    freq_map = {"hour": "1h", "day": "1d", "month": "1mo"}
    freq = freq_map.get(time_freq, "1h")

    chart_title = title or lcz_msg("lcz_ts_title", lang)
    x_label = xlab or lcz_msg("time_label", lang)
    y_label = ylab or lcz_msg("temperature_label", lang)

    # For daylight: compute the base per-station avg WITHOUT _by, then shade
    if by == "daylight":
        avg = (
            df.sort("date")
              .group_by_dynamic("date", every=freq, group_by="station")
              .agg(pl.col("var_interp").mean(), pl.col("lcz").first())
        )
        if not iplot:
            return avg
        fig = _plot_daylight_unified(
            avg, palette, chart_title, x_label, y_label,
            lat=lat_mean, lon=lon_mean,
        )

    elif by is not None:
        df = add_by_column(df, by, lat=lat_mean, lon=lon_mean)
        avg = (
            df.sort("date")
              .group_by_dynamic("date", every=freq, group_by=["station", "_by"])
              .agg(pl.col("var_interp").mean(), pl.col("lcz").first())
        )
        if not iplot:
            return avg
        if plot_type != "basic_line":
            raise NotImplementedError(
                f"plot_type={plot_type!r} with by= is not supported; use 'basic_line'."
            )
        groups = by_sorted_groups(avg["_by"], by)
        fig = _faceted_line_fig(avg, groups, palette, chart_title, x_label, y_label)

    else:
        avg = (
            df.sort("date")
              .group_by_dynamic("date", every=freq, group_by="station")
              .agg(pl.col("var_interp").mean(), pl.col("lcz").first())
        )
        if not iplot:
            return avg
        if plot_type == "basic_line":
            fig = _plot_basic_line(avg, palette, chart_title, x_label, y_label)
        elif plot_type == "heatmap":
            facet_col = facet_plot.lower() if facet_plot.lower() in avg.columns else facet_plot
            fig = _plot_heatmap(avg, facet_col, palette, chart_title, x_label, y_label)
        else:
            raise NotImplementedError(
                f"plot_type {plot_type!r} not supported. Use 'basic_line' or 'heatmap'."
            )

    if caption:
        fig.add_annotation(
            text=caption, xref="paper", yref="paper",
            x=0.5, y=-0.05, showarrow=False,
            font=dict(size=10, color="gray"),
        )

    fig = finalize_export(fig, style=style, isave=isave, save_extension=save_extension,
                           filename="lcz4r_ts_plot", lang=lang)

    return fig


__all__ = ["lcz_ts"]
