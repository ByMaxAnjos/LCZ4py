"""
lcz_uhi_intensity.py — Urban Heat Island intensity from LCZ-stratified stations.

Urban (LCZ 1–10) and rural (LCZ 11–16) stations are either chosen automatically
from the LCZ raster or supplied manually. UHI = T_urban − T_rural.

Thresholds follow the standard canopy-layer UHI definition: a timestamp is only
kept when *both* a valid urban and a valid rural observation exist, so a missing
rural reference never silently becomes a zero anomaly.

Rendering backends
------------------
- ``renderer="static"`` *(default)* — native matplotlib/Agg figure saved as a
  300 dpi PNG/PDF/SVG/TIFF. No Plotly, Kaleido, Chromium or browser needed.
- ``renderer="plotly"`` — the interactive figure.
"""

from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

try:
    from numba import njit  # noqa: F401 - optional acceleration, API parity
    HAS_NUMBA = True
except ImportError:  # pragma: no cover - depends on the local environment
    HAS_NUMBA = False

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, normalise_input_df, normalise_missing,
    select_by_date, assign_station_lcz, lcz_palette,
    require_nonempty, add_by_column, by_sorted_groups,
    _break_gaps_numpy, gap_threshold_hours,
)
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export
from LCZ4py._internal import lcz_static
from LCZ4py._internal.lcz_static import resolve_renderer

logger = logging.getLogger(__name__)

URBAN_CLASSES = [str(i) for i in range(1, 11)]
RURAL_CLASSES = [str(i) for i in range(11, 17)]

_TIME_FREQ = {"hour": "1h", "day": "1d", "month": "1mo", "year": "1y"}


# ── Time averaging ────────────────────────────────────────────────────────────

def _time_avg(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    if freq not in _TIME_FREQ:
        raise ValueError(f"time_freq must be one of {tuple(_TIME_FREQ)}, got {freq!r}")
    return (
        df.sort("date")
          .group_by_dynamic("date", every=_TIME_FREQ[freq], closed="left")
          .agg(pl.col("var_interp").mean())
    )


# ── Plotly figures ────────────────────────────────────────────────────────────

def _gapped(dates_np, values_np):
    """Break a line where observations are missing."""
    threshold_h = gap_threshold_hours(dates_np)
    return _break_gaps_numpy(dates_np, values_np, threshold_h)


def _plot_single(df: pl.DataFrame, title: str, xlab: str, ylab: str) -> go.Figure:
    fig = go.Figure()
    dates_np = df["date"].to_numpy()
    vals_np = df["uhi"].to_numpy()
    dates_p, vals_p = _gapped(dates_np, vals_np)
    fig.add_trace(go.Scatter(
        x=dates_p, y=vals_p,
        mode="lines", name="UHI",
        line=dict(color="#333333", width=1.5),
        hovertemplate=f"{ylab}: %{{y:.2f}}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=xlab, font=dict(size=14)),
                   tickfont=dict(size=12), gridcolor="#f0f0f0"),
        yaxis=dict(title=dict(text=ylab, font=dict(size=14)),
                   tickfont=dict(size=12), gridcolor="#f0f0f0",
                   zeroline=True, zerolinecolor="#aaaaaa"),
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

    u_d, u_v = _gapped(dates_np, df["urban"].to_numpy())
    r_d, r_v = _gapped(dates_np, df["rural"].to_numpy())
    h_d, h_v = _gapped(dates_np, df["uhi"].to_numpy())

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
        zeroline=True, zerolinecolor="#aaaaaa",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=xlab, font=dict(size=14)),
                   tickfont=dict(size=12), gridcolor="#f0f0f0"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=70, b=60),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.05),
    )
    return fig


def _dominant_lcz(df: pl.DataFrame, ids: Sequence[str]) -> Optional[str]:
    """Most frequent LCZ class among the given stations (``None`` if empty)."""
    if not ids:
        return None
    sub = df.filter(pl.col("station").is_in(list(ids)))
    if sub.is_empty():
        return None
    mode = sub["lcz"].drop_nulls().mode()
    return str(mode[0]) if mode.len() else None


def _wide_from_avg(avg: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:
    """Pivot long urban/rural means into a wide frame with a ``uhi`` column."""
    if avg.is_empty():
        raise ValueError(
            "No averaged observations are available — the date range and "
            "station filter removed every record."
        )
    wide = avg.pivot(index=index_cols, on="reference", values="var_interp")
    for col in ("urban", "rural"):
        if col not in wide.columns:
            logger.warning(
                "No '%s' observation in any time bin; UHI cannot be computed "
                "for this selection.", col,
            )
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))

    # Keep the urban/rural means, but only where *both* sides exist: a missing
    # rural reference must not be treated as a zero-temperature baseline.
    n_bins = wide.height
    wide = wide.with_columns(
        (pl.col("urban") - pl.col("rural")).round(2).alias("uhi")
    ).drop_nulls(subset=["urban", "rural"])
    dropped = n_bins - wide.height
    if dropped:
        logger.warning(
            "Dropped %d/%d time bin(s) where the urban or rural mean was "
            "missing (UHI needs both).", dropped, n_bins,
        )
    if wide.is_empty():
        raise ValueError(
            "No time bin has both an urban and a rural observation — cannot "
            "compute UHI intensity. Check the station selection and the "
            "date range."
        )
    return wide.sort(index_cols)


# ── Native static renderer (matplotlib/Agg — no Plotly/Kaleido/Chromium) ─────

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


def _finite_float(values) -> np.ndarray:
    """None-tolerant float conversion so gaps become NaN (a broken line)."""
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _break_date_column(sub: pl.DataFrame, col: str):
    """Gap-aware (dates, values) for one column of a UHI frame."""
    dates_np = sub["date"].to_numpy()
    values_np = np.asarray(sub[col].to_numpy(), dtype=float)
    return _gapped(dates_np, values_np)


def _legend_bottom(fig, fontsize: float, rows: int, *, with_title: bool = True,
                   xlabel: bool = True) -> float:
    """Fraction of the figure height to reserve below the axes for a legend."""
    fig_h = max(float(fig.get_figheight()), 0.5)
    line_in = (float(fontsize) * 1.75) / 72.0
    legend_in = max(int(rows), 0) * line_in + (0.22 if with_title else 0.12)
    extra_in = 0.55 if xlabel else 0.18
    return float(min(0.45, max(0.14, (legend_in + extra_in) / fig_h)))


def _static_uhi_figure(
    panels: list,
    *,
    group: bool,
    title: Optional[str],
    xlab: str,
    ylab: str,
    y2lab: str,
    u_color: str,
    r_color: str,
    style: str = "default",
    dpi: Optional[int] = None,
    caption: Optional[str] = None,
    lang: str = "en",
    isave: bool = False,
    save_extension: str = "html",
    filename: str = "lcz4r_uhi_plot",
):
    """Static UHI chart: UHI-only line, or urban/rural + UHI on twin axes."""
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    st = lcz_static
    panels = list(panels) or [("", pl.DataFrame({"date": [], "uhi": []}))]
    n = len(panels)
    height_mm = float(min(max(58.0 * n + 30.0, 80.0), 195.0))

    with st.style_context(style):
        fig, spec, _ = st.new_figure(style, height_mm=height_mm, dpi=dpi)
        left, right = 0.115, 0.955 if group else 0.975
        top = st.figure_top(bool(title)) - (
            0.03 if any(label for label, _ in panels) else 0.0
        )
        label_size = max(6.0, spec.font_size - 1.5)
        tick_size = max(5.5, spec.font_size - 2.5)
        bottom = (
            _legend_bottom(fig, tick_size, 1, with_title=False)
            if group else st.figure_bottom(False)
        )
        gap = 0.075 if group else 0.055
        span = max(top - bottom - gap * (n - 1), 0.10)
        height = span / n

        handles = None
        for idx in range(n):
            label, sub = panels[idx]
            bottom_i = top - (idx + 1) * height - idx * gap
            ax = fig.add_axes([left, bottom_i, right - left, height])
            ax2 = ax.twinx() if group else None

            if label:
                ax.set_title(str(label), fontsize=label_size, loc="left", pad=2.5)

            if not sub.is_empty():
                d_uhi, v_uhi = _break_date_column(sub, "uhi")
                x_uhi = _mpl_date_numbers(d_uhi)
                y_uhi = _finite_float(v_uhi)
                if group and ax2 is not None:
                    d_u, v_u = _break_date_column(sub, "urban")
                    d_r, v_r = _break_date_column(sub, "rural")
                    ax.plot(_mpl_date_numbers(d_u), _finite_float(v_u),
                            color=u_color, lw=1.1, zorder=3)
                    ax.plot(_mpl_date_numbers(d_r), _finite_float(v_r),
                            color=r_color, lw=1.1, zorder=3)
                    ax2.plot(x_uhi, y_uhi, color="#333333", lw=1.0,
                             ls=(0, (4, 2)), zorder=3)
                    ax2.axhline(0, color="0.55", lw=0.7, ls="--", zorder=2)
                    ax2.set_ylabel(y2lab, fontsize=label_size, color="#333333")
                    ax2.tick_params(axis="y", labelsize=tick_size,
                                    colors="#333333", length=3, width=0.6)
                    ax2.spines["right"].set_linewidth(0.8)
                    ax2.spines["top"].set_visible(False)
                    handles = [
                        Line2D([], [], color=u_color, lw=1.2, label="Urban"),
                        Line2D([], [], color=r_color, lw=1.2, label="Rural"),
                        Line2D([], [], color="#333333", lw=1.0, ls=(0, (4, 2)),
                               label="UHI"),
                    ]
                else:
                    ax.plot(x_uhi, y_uhi, color="#333333", lw=1.1, zorder=3)
                    ax.axhline(0, color="0.55", lw=0.7, ls="--", zorder=2)

            ax.set_ylabel(ylab if group else y2lab, fontsize=label_size)
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

        if group and handles:
            fig.legend(
                handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.008),
                ncol=3, frameon=False, fontsize=tick_size, handlelength=1.5,
                columnspacing=1.4,
            )

        st.add_header(fig, title, None, spec)
        st.add_footer(fig, caption, None, spec)
        st.save_static(
            fig, filename, isave=isave, save_extension=save_extension,
            style=style, dpi=dpi, lang=lang,
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
    renderer: Optional[str] = "static",
    dpi: Optional[int] = None,
) -> Union[go.Figure, pl.DataFrame, "Figure"]:
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
    time_freq : {"hour", "day", "month", "year"}
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
        ``go.Figure`` for ``renderer="plotly"``, or the wide DataFrame when
        ``iplot=False``. The wide DataFrame has columns
        ``date, urban, rural, uhi`` (+ ``_by`` when ``by=`` is used).

    Raises
    ------
    ValueError
        If no urban/rural pair can be formed, ``method="manual"`` is missing
        its station lists, or the selection leaves no usable time bin.
    """
    if data_frame is None:
        raise ValueError("data_frame is required.")

    renderer = resolve_renderer(renderer)
    static = renderer == "static"
    if time_freq not in _TIME_FREQ:
        raise ValueError(
            f"time_freq must be one of {tuple(_TIME_FREQ)}, got {time_freq!r}"
        )
    method_key = str(method).lower()
    if method_key not in ("lcz", "manual"):
        raise ValueError(f"method must be 'LCZ' or 'manual', got {method!r}")

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
    df = df.with_columns(pl.col("lcz").cast(pl.String))

    if method_key == "lcz":
        u_ids = df.filter(pl.col("lcz").is_in(URBAN_CLASSES))["station"].unique().to_list()
        r_ids = df.filter(pl.col("lcz").is_in(RURAL_CLASSES))["station"].unique().to_list()
        if not u_ids or not r_ids:
            raise ValueError(
                lcz_msg("no_urban_rural", lang)
                + f" Found urban={len(u_ids)}, rural={len(r_ids)} station(s) "
                f"among classes {sorted(df['lcz'].unique().to_list())}."
            )
    else:
        u_ids = [str(s) for s in (T_urban or [])]
        r_ids = [str(s) for s in (T_rural or [])]
        if not u_ids or not r_ids:
            raise ValueError(
                "method='manual' requires non-empty T_urban and T_rural station "
                f"lists (got {len(u_ids)} and {len(r_ids)})."
            )
        missing = [
            sid for sid in (u_ids + r_ids)
            if sid not in set(df["station"].unique().to_list())
        ]
        if missing:
            raise ValueError(
                "These T_urban/T_rural station IDs are not in the data (after "
                f"filtering and LCZ matching): {missing}"
            )

    overlap = set(u_ids) & set(r_ids)
    if overlap:
        logger.warning(
            "%d station(s) listed as both urban and rural; they are treated as "
            "urban: %s", len(overlap), sorted(overlap)[:10],
        )

    palette = lcz_palette()
    u_lcz = _dominant_lcz(df, u_ids)
    r_lcz = _dominant_lcz(df, r_ids)
    u_color = palette.get(u_lcz, "#CC3333") if u_lcz else "#CC3333"
    r_color = palette.get(r_lcz, "#3366CC") if r_lcz else "#3366CC"

    model_df = df.filter(pl.col("station").is_in(u_ids + r_ids)).with_columns(
        pl.when(pl.col("station").is_in(u_ids))
        .then(pl.lit("urban"))
        .otherwise(pl.lit("rural"))
        .alias("reference")
    )
    require_nonempty(model_df, "selected urban/rural observations")

    p_freq = _TIME_FREQ[time_freq]

    if by is not None:
        lat_mean = float(df["latitude"].mean())
        lon_mean = float(df["longitude"].mean())
        model_df = add_by_column(model_df, by, lat=lat_mean, lon=lon_mean)
        group_by_cols: Union[list[str], str] = ["reference", "_by"]
        pivot_index = ["date", "_by"]
    else:
        group_by_cols = "reference"
        pivot_index = ["date"]

    avg = (
        model_df.sort("date")
        .group_by_dynamic("date", every=p_freq, closed="left", group_by=group_by_cols)
        .agg(pl.col("var_interp").mean())
    )
    wide = _wide_from_avg(avg, pivot_index)

    if not iplot:
        return wide

    chart_title = title or lcz_msg("lcz_uhi_title", lang)
    x_label = xlab or lcz_msg("time_label", lang)
    y_label = ylab or lcz_msg("temperature_label", lang)
    y2_label = ylab2 or lcz_msg("uhi_label", lang)

    if by is not None:
        groups = by_sorted_groups(wide["_by"], by)
        n = max(len(groups), 1)
        if static:
            fig = _static_uhi_figure(
                [(g, wide.filter(pl.col("_by") == g).sort("date")) for g in groups],
                group=group, title=chart_title, xlab=x_label, ylab=y_label,
                y2lab=y2_label, u_color=u_color, r_color=r_color, style=style,
                dpi=dpi, caption=caption, lang=lang, isave=isave,
                save_extension=save_extension,
            )
        else:
            specs = [[{"secondary_y": True}]] * n if group else [[{}]] * n
            fig = make_subplots(
                rows=n, cols=1,
                subplot_titles=groups,
                shared_xaxes=False,
                vertical_spacing=0.05,
                specs=specs,
            )
            for i, g in enumerate(groups, 1):
                sub = wide.filter(pl.col("_by") == g).sort("date")
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
        if static:
            fig = _static_uhi_figure(
                [("", wide.sort("date"))],
                group=group, title=chart_title, xlab=x_label, ylab=y_label,
                y2lab=y2_label, u_color=u_color, r_color=r_color, style=style,
                dpi=dpi, caption=caption, lang=lang, isave=isave,
                save_extension=save_extension,
            )
        else:
            fig = (
                _plot_dual(wide, u_color, r_color, chart_title, x_label, y_label, y2_label)
                if group else
                _plot_single(wide, chart_title, x_label, y2_label)
            )

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
                           filename="lcz4r_uhi_plot", lang=lang)

    return fig


__all__ = ["lcz_uhi_intensity"]
