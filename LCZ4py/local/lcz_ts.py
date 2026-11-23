"""
lcz_ts.py — LCZ-stratified time-series visualisation.

Rendering backends
------------------
- ``renderer="static"`` *(default)* — native matplotlib/Agg figure saved as a
  300 dpi PNG/PDF/SVG/TIFF. No Plotly, Kaleido, Chromium or browser needed.
- ``renderer="plotly"`` — the interactive WebGL (Scattergl) figure.

Handles large datasets efficiently:
- Polars groupby for time averaging and per-station partitioning (no per-station
  boolean-filter loops)
- Plotly WebGL (Scattergl) for smooth panning/zooming (``renderer="plotly"``)
- Automatic 10× decimation when a single station exceeds 100k points
- Line breaks at data gaps (never draws a straight line across missing days)
"""

from __future__ import annotations
import datetime as _dt_mod
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
from LCZ4py._internal import lcz_static
from LCZ4py._internal.lcz_static import resolve_renderer

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


# ── Native static renderer (matplotlib/Agg — no Plotly/Kaleido/Chromium) ─────

def _mpl_color(value: str) -> str:
    """Convert a Plotly ``rgb()``/``rgba()`` colour to a matplotlib hex string."""
    text = str(value).strip()
    if text.lower().startswith("rgb"):
        try:
            inside = text[text.index("(") + 1: text.rindex(")")]
            r, g, b = (float(p) for p in inside.split(",")[:3])
            return "#{:02x}{:02x}{:02x}".format(
                max(0, min(255, int(round(r)))),
                max(0, min(255, int(round(g)))),
                max(0, min(255, int(round(b)))),
            )
        except (TypeError, ValueError):
            return "#666666"
    return text


def _mpl_date_numbers(values) -> np.ndarray:
    """datetime64/datetime/None sequence → matplotlib date numbers (NaN gaps)."""
    import datetime as _dt
    import matplotlib.dates as mdates

    out = np.full(len(values), np.nan, dtype=float)
    for i, v in enumerate(values):
        if v is None:
            continue
        if isinstance(v, np.datetime64):
            v = v.astype("datetime64[us]").astype(_dt.datetime)
        elif isinstance(v, _dt.date) and not isinstance(v, _dt.datetime):
            v = _dt.datetime(v.year, v.month, v.day)
        out[i] = mdates.date2num(v)
    return out


def _readable(color: str, min_luminance: float = 0.62) -> str:
    """Darken a near-white palette colour so a 1 px line stays visible."""
    hex_ = str(color).lstrip("#")
    if len(hex_) != 6:
        return color
    try:
        r, g, b = (int(hex_[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return color
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    if luminance <= min_luminance or luminance <= 0:
        return color
    scale = (min_luminance / luminance) * 0.94
    r, g, b = (max(0, min(255, int(round(channel * scale)))) for channel in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _finite_float(values) -> np.ndarray:
    """None-tolerant float conversion so gaps become NaN (a broken line)."""
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _legend_bottom(fig, fontsize: float, rows: int, *, with_title: bool = True,
                   xlabel: bool = True) -> float:
    """Fraction of the figure height to reserve below the axes for a legend.

    The legend is anchored at the figure's bottom edge, so the reserved space
    must cover the legend's own height (rows + optional title) plus the x tick
    labels/x label. Computed from the actual figure height so large "default"
    fonts cannot grow into the x label.
    """
    fig_h = max(float(fig.get_figheight()), 0.5)
    line_in = (float(fontsize) * 1.75) / 72.0
    legend_in = max(int(rows), 0) * line_in + (0.22 if with_title else 0.12)
    extra_in = 0.55 if xlabel else 0.18
    return float(min(0.45, max(0.14, (legend_in + extra_in) / fig_h)))


def _panel_axes(fig, n: int, *, has_header: bool, has_legend: bool,
                gap: float = 0.05, right: float = 0.975,
                bottom: Optional[float] = None, top_trim: float = 0.0):
    """Stacked full-width axes inside ``fig`` (top panel first)."""
    st = lcz_static
    left = 0.115
    top = st.figure_top(has_header) - float(top_trim)
    bottom = st.figure_bottom(has_legend) if bottom is None else float(bottom)
    n = max(int(n), 1)
    span = max(top - bottom - gap * (n - 1), 0.10)
    height = span / n
    axes = []
    for i in range(n):
        bottom_i = top - (i + 1) * height - i * gap
        axes.append(fig.add_axes([left, bottom_i, right - left, height]))
    return axes


def _static_ts_lines(
    panels: list,
    *,
    title: Optional[str],
    xlab: str,
    ylab: str,
    palette_name: str,
    style: str = "default",
    dpi: Optional[int] = None,
    caption: Optional[str] = None,
    lang: str = "en",
    isave: bool = False,
    save_extension: str = "html",
    filename: str = "lcz4r_ts_plot",
    daylight: Optional[tuple] = None,
):
    """Publication-quality time-series figure (one axes per facet)."""
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    st = lcz_static
    panels = list(panels) or [("", pl.DataFrame({"date": [], "var_interp": [], "station": []}))]
    n = len(panels)

    station_names: list[str] = []
    for _, sub in panels:
        for name in _partition_by_station(sub):
            if name not in station_names:
                station_names.append(name)
    colors = dict(zip(
        station_names,
        [_readable(_mpl_color(c)) for c in _station_colors(station_names, palette_name)],
    ))

    ncol = min(max(len(station_names), 1), 4)
    n_handles = len(station_names) + (1 if daylight is not None else 0)
    legend_rows = max(1, -(-n_handles // ncol)) if n_handles else 0
    height_mm = min(62.0 * n + 22.0 + 7.0 * max(legend_rows - 1, 0), 195.0)
    with st.style_context(style):
        fig, spec, _ = st.new_figure(style, height_mm=height_mm, dpi=dpi)
        axes = _panel_axes(
            fig, n, has_header=bool(title), has_legend=bool(n_handles),
            bottom=_legend_bottom(
                fig, max(5.5, spec.font_size - 2.5), legend_rows,
                with_title=bool(station_names),
            ),
            top_trim=0.03 if any(label for label, _ in panels) else 0.0,
        )
        label_size = max(6.0, spec.font_size - 1.5)
        tick_size = max(5.5, spec.font_size - 2.5)

        for idx, (ax, panel) in enumerate(zip(axes, panels)):
            label, sub = panel
            if label:
                ax.set_title(str(label), fontsize=label_size, loc="left", pad=2.5)

            if daylight is not None and not sub.is_empty():
                lat, lon = daylight
                unique_days = sorted({
                    _dt_mod.date(d.year, d.month, d.day)
                    for d in sub["date"].to_list() if d is not None
                })
                if len(unique_days) > _MAX_DAYLIGHT_RECTS // 2:
                    logger.warning(
                        "Daylight shading skipped in the static figure: %d days "
                        "would need too many spans (limit %d).",
                        len(unique_days), _MAX_DAYLIGHT_RECTS,
                    )
                else:
                    for day, intervals in zip(
                        unique_days,
                        _daylight_intervals_batch(lat, lon, unique_days),
                    ):
                        base = _dt_mod.datetime(day.year, day.month, day.day)
                        for rise_h, set_h in intervals:
                            ax.axvspan(
                                mdates.date2num(base + _dt_mod.timedelta(hours=rise_h)),
                                mdates.date2num(base + _dt_mod.timedelta(hours=set_h)),
                                color="#FFC832", alpha=0.20, lw=0, zorder=0,
                            )

            for name in _partition_by_station(sub):
                part = sub.filter(pl.col("station") == name).sort("date")
                if part.is_empty():
                    continue
                if part.height > _MAX_POINTS_PER_STATION:
                    part = part.gather_every(_DECIMATION)
                dates_np = part["date"].to_numpy()
                vals_np = part["var_interp"].to_numpy()
                d_plot, v_plot = _break_gaps_numpy(
                    dates_np, vals_np, gap_threshold_hours(dates_np)
                )
                ax.plot(
                    _mpl_date_numbers(d_plot), _finite_float(v_plot),
                    color=colors.get(name, "#666666"), lw=0.9, zorder=3,
                )

            ax.set_ylabel(ylab, fontsize=label_size)
            ax.grid(True, axis="y", lw=0.35, color="0.88", zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.tick_params(axis="both", labelsize=tick_size, length=3, width=0.6)
            if idx == n - 1:
                ax.set_xlabel(xlab, fontsize=label_size)
            else:
                ax.tick_params(axis="x", labelbottom=False)

        handles = [
            Line2D([], [], color=colors[name], lw=1.2, label=name)
            for name in station_names
        ]
        if daylight is not None:
            handles.append(Patch(facecolor="#FFC832", alpha=0.55, label="Daytime"))
        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.008),
                ncol=ncol,
                frameon=False,
                fontsize=tick_size,
                handlelength=1.4,
                columnspacing=1.1,
                title=lcz_msg("lcz_class_legend", lang) if station_names else None,
                title_fontsize=tick_size,
            )

        st.add_header(fig, title, None, spec)
        st.add_footer(fig, caption, None, spec)
        st.save_static(
            fig, filename, isave=isave, save_extension=save_extension,
            style=style, dpi=dpi, lang=lang,
        )
    return fig


def _static_ts_heatmap(
    df: pl.DataFrame,
    facet_var: str,
    *,
    title: Optional[str],
    xlab: str,
    ylab: str,
    style: str = "default",
    dpi: Optional[int] = None,
    caption: Optional[str] = None,
    lang: str = "en",
    isave: bool = False,
    save_extension: str = "html",
    filename: str = "lcz4r_ts_plot",
    colorscale: str = "Viridis",
):
    """Date × facet heatmap (time on x, one LCZ/facet per row)."""
    import matplotlib.dates as mdates

    st = lcz_static
    raw_facets = df[facet_var].unique().to_list()

    def _facet_key(value):
        text = str(value)
        try:
            return (0, float(text), "")
        except ValueError:
            return (1, 0.0, text)

    facets = [str(f) for f in sorted(raw_facets, key=_facet_key)]
    rows = []
    all_dates: set = set()
    for facet in facets:
        sub = df.filter(pl.col(facet_var).cast(pl.String) == facet).sort("date")
        dates = sub["date"].to_list()
        rows.append((facet, dates, sub["var_interp"].to_numpy()))
        all_dates.update(dates)

    dates_sorted = sorted(all_dates)
    col_of = {d: i for i, d in enumerate(dates_sorted)}
    matrix = np.full((len(facets), len(dates_sorted)), np.nan, dtype=float)
    for i, (_, dates, values) in enumerate(rows):
        for d, v in zip(dates, values):
            if v is not None:
                matrix[i, col_of[d]] = float(v)

    xnum = _mpl_date_numbers(dates_sorted)
    if xnum.size >= 2:
        edges = np.concatenate([
            [xnum[0] - (xnum[1] - xnum[0]) / 2.0],
            (xnum[:-1] + xnum[1:]) / 2.0,
            [xnum[-1] + (xnum[-1] - xnum[-2]) / 2.0],
        ])
    elif xnum.size == 1:
        edges = np.array([xnum[0] - 0.5, xnum[0] + 0.5])
    else:  # pragma: no cover - guarded by the caller's non-empty check
        edges = np.array([0.0, 1.0])

    finite = matrix[np.isfinite(matrix)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    height_mm = min(13.0 * max(len(facets), 1) + 45.0, 195.0)
    with st.style_context(style):
        fig, spec, _ = st.new_figure(style, height_mm=height_mm, dpi=dpi)
        axes = _panel_axes(fig, 1, has_header=bool(title), has_legend=False, right=0.88)
        ax = axes[0]

        mesh = ax.pcolormesh(
            edges, np.arange(len(facets) + 1), matrix, shading="flat",
            cmap=st.mpl_cmap(colorscale), vmin=vmin, vmax=vmax,
        )
        ax.set_yticks(np.arange(len(facets)) + 0.5)
        ax.set_yticklabels(facets, fontsize=max(5.5, spec.font_size - 2.5))
        ax.invert_yaxis()
        ax.set_xlabel(xlab, fontsize=max(6.0, spec.font_size - 1.5))
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.tick_params(axis="both", labelsize=max(5.5, spec.font_size - 2.5),
                       length=3, width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        st.add_colorbar(fig, ax, mesh, units=ylab, fontsize=spec.font_size)
        st.add_header(fig, title, None, spec)
        st.add_footer(fig, caption, None, spec)
        st.save_static(
            fig, filename, isave=isave, save_extension=save_extension,
            style=style, dpi=dpi, lang=lang,
        )
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
    renderer: Optional[str] = "static",
    dpi: Optional[int] = None,
) -> Union[go.Figure, pl.DataFrame, "Figure"]:
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
    renderer : {"static", "plotly"}, optional
        ``"static"`` (default, also ``None``/``"matplotlib"``/``"agg"``)
        returns a matplotlib ``Figure`` drawn with the Agg canvas — no Plotly,
        Kaleido, Chromium or browser needed — and saves a >=300 dpi PNG (or
        PDF/SVG/TIFF) via ``lcz_static.save_static``. ``"plotly"`` returns the
        interactive WebGL figure.
    dpi : int, optional
        Export resolution for the static renderer (never below 300 dpi).

    Returns
    -------
    matplotlib.figure.Figure, go.Figure or pl.DataFrame
        A matplotlib ``Figure`` for the default static renderer, a Plotly
        ``go.Figure`` for ``renderer="plotly"``, or the averaged DataFrame
        when ``iplot=False``.

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

    renderer = resolve_renderer(renderer)
    static = renderer == "static"
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
        if static:
            fig = _static_ts_lines(
                [("", avg.sort(["station", "date"]))],
                title=chart_title, xlab=x_label, ylab=y_label,
                palette_name=palette, style=style, dpi=dpi, caption=caption,
                lang=lang, isave=isave, save_extension=save_extension,
                daylight=(lat_mean, lon_mean),
            )
        else:
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
        if static:
            fig = _static_ts_lines(
                [(g, avg.filter(pl.col("_by") == g)) for g in groups],
                title=chart_title, xlab=x_label, ylab=y_label,
                palette_name=palette, style=style, dpi=dpi, caption=caption,
                lang=lang, isave=isave, save_extension=save_extension,
            )
        else:
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
            if static:
                fig = _static_ts_lines(
                    [("", avg.sort(["station", "date"]))],
                    title=chart_title, xlab=x_label, ylab=y_label,
                    palette_name=palette, style=style, dpi=dpi, caption=caption,
                    lang=lang, isave=isave, save_extension=save_extension,
                )
            else:
                fig = _plot_basic_line(avg, palette, chart_title, x_label, y_label, lang)
        else:
            facet_col = facet_plot.lower() if facet_plot.lower() in avg.columns else facet_plot
            if facet_col not in avg.columns:
                raise ValueError(
                    f"facet_plot={facet_col!r} is not an aggregated column; "
                    f"choose one of {sorted(avg.columns)}."
                )
            if avg.is_empty():
                raise ValueError("No data to plot as a heatmap.")
            if static:
                fig = _static_ts_heatmap(
                    avg, facet_col, title=chart_title, xlab=x_label, ylab=y_label,
                    style=style, dpi=dpi, caption=caption, lang=lang,
                    isave=isave, save_extension=save_extension,
                )
            else:
                fig = _plot_heatmap(
                    avg, facet_col, palette, chart_title, x_label, y_label, lang
                )

    if static:
        # Captions and saving are handled inside the static builders.
        return fig

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
