"""
lcz_ts.py — LCZ-stratified time-series visualisation.

Handles large datasets efficiently:
- Polars groupby for time averaging and per-station partitioning (no per-station
  boolean-filter loops)
- Plotly WebGL (Scattergl) for smooth panning/zooming
- Automatic 10× decimation when a single station exceeds 100k points
- Line breaks at data gaps (never draws a straight line across missing days)
"""

from __future__ import annotations
import datetime as _dt_mod
import logging
import os
from typing import Optional, Union

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio

try:
    import datashader as ds  # noqa: F401 - optional, kept for API parity
    import datashader.transfer_functions as tf  # noqa: F401
    import xarray as xr  # noqa: F401
    HAS_DATASHADER = True
except ImportError:  # pragma: no cover - depends on the local environment
    HAS_DATASHADER = False

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, normalise_input_df, normalise_missing,
    select_by_date, assign_station_lcz, require_nonempty,
    add_by_column, by_sorted_groups, gap_threshold_hours,
    _break_gaps_numpy, _daylight_intervals_batch,
)
from LCZ4py.general.lcz_plot_parameters import _palette_to_plotly_colorscale
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export

logger = logging.getLogger(__name__)

_MAX_POINTS_PER_STATION = 100_000
_DECIMATION = 10
_MAX_DAYLIGHT_RECTS = 2000


def _station_colors(stations, palette_name: str) -> list[str]:
    n = max(len(stations), 1)
    colorscale = _palette_to_plotly_colorscale(palette_name, n=max(n, 2))
    step = max(1, len(colorscale) // n)
    return [colorscale[min(i * step, len(colorscale) - 1)][1] for i in range(n)]


def _partition_by_station(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split once by station, keeping a deterministic (sorted) key order."""
    if df.is_empty():
        return {}
    partitions = df.partition_by("station", as_dict=True, maintain_order=True)
    return {str(k[0]) if isinstance(k, tuple) else str(k): v
            for k, v in sorted(partitions.items(), key=lambda kv: str(kv[0]))}


def _plot_basic_line(
    df: pl.DataFrame,
    palette_name: str,
    title: str,
    xlab: str,
    ylab: str,
    lang: str = "en",
) -> go.Figure:
    fig = go.Figure()
    if df.is_empty():
        logger.warning("_plot_basic_line: no data to plot.")
        fig.update_layout(title=dict(text=title))
        return fig

    partitions = _partition_by_station(df)
    stations = list(partitions)
    colors = _station_colors(stations, palette_name)

    for i, st in enumerate(stations):
        sub = partitions[st].sort("date")
        if len(sub) > _MAX_POINTS_PER_STATION:
            sub = sub.gather_every(_DECIMATION)
            logger.debug("%s: decimated to %d points for display.", st, len(sub))

        dates_np = sub["date"].to_numpy()
        vals_np = sub["var_interp"].to_numpy()
        threshold_h = gap_threshold_hours(dates_np)
        dates_plot, vals_plot = _break_gaps_numpy(dates_np, vals_np, threshold_h)

        fig.add_trace(go.Scattergl(
            x=dates_plot, y=vals_plot,
            name=st,
            mode="lines",
            line=dict(color=colors[i], width=1),
            hovertemplate=(
                f"<b>{st}</b><br>{xlab}: %{{x}}<br>{ylab}: %{{y:.2f}}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=xlab, font=dict(size=14)),
                   tickfont=dict(size=12), gridcolor="#f0f0f0"),
        yaxis=dict(title=dict(text=ylab, font=dict(size=14)),
                   tickfont=dict(size=12), gridcolor="#f0f0f0"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=70, b=60),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
        legend=dict(title=dict(text=lcz_msg("lcz_class_legend", lang))),
    )
    return fig


def _plot_heatmap(
    df: pl.DataFrame,
    facet_var: str,
    palette_name: str,
    title: str,
    xlab: str,
    ylab: str,
    lang: str = "en",
) -> go.Figure:
    if facet_var not in df.columns:
        raise ValueError(
            f"facet_plot={facet_var!r} is not an aggregated column; "
            f"choose one of {sorted(df.columns)}."
        )
    if df.is_empty():
        raise ValueError("No data to plot as a heatmap.")

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
            colorbar=dict(title=dict(text=ylab)) if i == 0 else None,
            name=str(f),
            hovertemplate=(
                f"{facet_var}: %{{y}}<br>{xlab}: %{{x}}<br>"
                f"{ylab}: %{{z:.2f}}<extra></extra>"
            ),
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
    lang: str = "en",
) -> go.Figure:
    """Build a multi-row subplot figure, one row per by-group."""
    fig = make_subplots(
        rows=max(len(groups), 1), cols=1,
        subplot_titles=groups,
        shared_xaxes=False,
        vertical_spacing=0.04,
    )
    seen_stations: set[str] = set()
    for i, g in enumerate(groups, 1):
        sub = avg.filter(pl.col("_by") == g)
        sub_fig = _plot_basic_line(sub, palette_name, g, x_label, y_label, lang)
        for trace in sub_fig.data:
            trace.showlegend = trace.name not in seen_stations
            seen_stations.add(trace.name)
            fig.add_trace(trace, row=i, col=1)
    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=20), x=0.5, xanchor="center"),
        height=max(400, 280 * max(len(groups), 1)),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=70, b=60),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
        legend=dict(title=dict(text=lcz_msg("lcz_class_legend", lang))),
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
    lang: str = "en",
) -> go.Figure:
    """Full time series with amber vrects marking each daytime window.

    No subplot split — daytime/nighttime are shown on a single axis with
    shaded background regions computed from the NOAA solar elevation formula.
    Valid for all latitudes (polar day/night handled gracefully). When the
    series spans too many days for readable shading, the shading is skipped
    with a warning instead of silently producing thousands of shapes.
    """
    fig = _plot_basic_line(df, palette_name, title, xlab, ylab, lang)

    unique_dates = sorted({
        _dt_mod.date(d.year, d.month, d.day)
        for d in df["date"].to_list()
        if d is not None
    })
    if len(unique_dates) > _MAX_DAYLIGHT_RECTS // 2:
        logger.warning(
            "Daylight shading skipped: %d days would need too many shapes "
            "(limit %d).",
            len(unique_dates), _MAX_DAYLIGHT_RECTS,
        )
        fig.add_annotation(
            text=f"Daytime shading omitted ({len(unique_dates)} days)",
            xref="paper", yref="paper", x=0.99, y=1.02,
            showarrow=False, xanchor="right",
            font=dict(size=9, color="gray"),
        )
        return fig

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


def _parse_time_freq(time_freq: str) -> str:
    """Translate the public ``time_freq`` value into a Polars duration."""
    freq_map = {
        "hour": "1h", "hourly": "1h",
        "day": "1d", "daily": "1d",
        "week": "1w", "weekly": "1w",
        "month": "1mo", "monthly": "1mo",
        "year": "1y", "yearly": "1y",
    }
    freq = freq_map.get(str(time_freq).lower())
    if freq is None:
        raise ValueError(
            f"time_freq must be one of {sorted(set(freq_map))}, got {time_freq!r}"
        )
    return freq


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
    time_freq : {"hour", "day", "week", "month", "year"}
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
        Column for the heatmap facet variable (heatmap only); default ``"lcz"``.
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
    year, month, day, hour : int or sequence, optional
        Date/time component filters (validated; out-of-range values raise).
    start, end : str or datetime, optional
        Inclusive date range bounds.

    Returns
    -------
    go.Figure or pl.DataFrame

    Raises
    ------
    ValueError
        If ``data_frame`` is missing/empty, required columns are absent, no
        station maps to LCZ class 1-17, or a filter/frequency value is invalid.
    """
    if data_frame is None:
        raise ValueError("data_frame is required.")

    if plot_type not in ("basic_line", "heatmap"):
        raise NotImplementedError(
            f"plot_type {plot_type!r} not supported. Use 'basic_line' or 'heatmap'."
        )
    if by is not None and plot_type != "basic_line":
        raise NotImplementedError(
            f"plot_type={plot_type!r} with by= is not supported; use 'basic_line'."
        )

    freq = _parse_time_freq(time_freq)

    df = normalise_input_df(data_frame, var=var, station_id=station_id)
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null())
    require_nonempty(df, "observations")

    ds_raster = load_lcz_raster(x)
    df = assign_station_lcz(df, ds_raster)
    require_nonempty(df, "observations at stations inside the LCZ raster")

    df = df.with_columns(
        (pl.col("station") + " (" + pl.col("lcz").cast(pl.String) + ")").alias("station"),
        pl.col("lcz").cast(pl.String),
    )

    lat_mean = float(df["latitude"].mean())
    lon_mean = float(df["longitude"].mean())

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
            return avg.sort(["station", "date"])
        fig = _plot_daylight_unified(
            avg, palette, chart_title, x_label, y_label,
            lat=lat_mean, lon=lon_mean, lang=lang,
        )

    elif by is not None:
        df = add_by_column(df, by, lat=lat_mean, lon=lon_mean)
        avg = (
            df.sort("date")
              .group_by_dynamic("date", every=freq, group_by=["station", "_by"])
              .agg(pl.col("var_interp").mean(), pl.col("lcz").first())
        )
        if not iplot:
            return avg.sort(["station", "_by", "date"])
        groups = by_sorted_groups(avg["_by"], by)
        if not groups:
            raise ValueError(f"by={by!r} produced no groups for this dataset.")
        fig = _faceted_line_fig(
            avg, groups, palette, chart_title, x_label, y_label, lang
        )

    else:
        avg = (
            df.sort("date")
              .group_by_dynamic("date", every=freq, group_by="station")
              .agg(pl.col("var_interp").mean(), pl.col("lcz").first())
        )
        if not iplot:
            return avg.sort(["station", "date"])
        if plot_type == "basic_line":
            fig = _plot_basic_line(avg, palette, chart_title, x_label, y_label, lang)
        else:
            facet_col = facet_plot.lower() if facet_plot.lower() in avg.columns else facet_plot
            fig = _plot_heatmap(
                avg, facet_col, palette, chart_title, x_label, y_label, lang
            )

    if caption:
        fig.add_annotation(
            text=caption, xref="paper", yref="paper",
            x=0.5, y=-0.08, showarrow=False,
            font=dict(size=10, color="gray"),
        )

    fig = finalize_export(fig, style=style, isave=isave, save_extension=save_extension,
                           filename="lcz4r_ts_plot", lang=lang)

    return fig


__all__ = ["lcz_ts"]
