"""Visualize the relationship between LCZ classes and a gridded environmental
variable (ERA5 temperature, CHIRPS rainfall, pollution, drought indices...)
for the same city/region.

New function for this repo (no direct R equivalent) — climasus4r's
sus_grid_*() functions aggregate to whole municipalities, but the LCZ side
of this codebase (lcz_get_map, lcz_cal_area) works per-pixel on a single
city's raster. The natural join between the two is therefore in RASTER
SPACE: sample the environmental raster at each LCZ pixel and group by class
(the same zonal-stats-per-LCZ-class pattern lcz_cal_area already uses for
area). ``lcz_cal_indexes`` computes the same per-class summary as a table
instead of a plot; the ``lcz_grid_*`` downloaders already return data
pre-cropped to the LCZ grid, so no separate join step is needed.

Cartography
-----------
* Scatter plots carry an OLS fit with its equation, R² and p-value, plus a
  95% confidence band for the mean response. The fit is computed on the
  un-jittered class codes (jitter is display-only).
* Markers use the colorblind-safe LCZ palette (``inclusive=True``) and a
  cycling marker shape, so classes stay distinguishable without colour.
* The **default** renderer is native static matplotlib/Agg
  (``renderer="static"``, also ``None``/``"matplotlib"``/``"agg"``), built
  with the shared ``_internal/lcz_static`` helpers and exported at >=300 dpi
  without Plotly, Kaleido, Chromium or a browser. ``renderer="plotly"``
  keeps the previous interactive figures.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import stats as spstats

from LCZ4py._internal.lcz_parameters_data import (
    LCZ_COLORS, LCZ_COLORBLIND, LCZ_IDS, get_lcz_names,
)
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import get_style, save_plotly_figure
from LCZ4py._internal import lcz_static as _st

logger = logging.getLogger(__name__)

OUTPUT_DIR = "LCZ4r_output"

#: Marker shapes cycled by LCZ class — redundant (non-colour) encoding so the
#: scatter stays readable for colour-vision-deficient readers.
_MARKER_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "cross")

#: Aggregations accepted by the heatmap (pandas groupby names).
_AGG_FUNS = ("mean", "median", "sum", "min", "max")


# ── Raster input ──────────────────────────────────────────────────────────────

def _read_band(path: Union[str, Path]) -> tuple[np.ndarray, dict]:
    """Read band 1 as float64 with the file's nodata mapped to NaN."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raster not found: {p}")
    with rasterio.open(str(p)) as src:
        data = src.read(1, masked=True).astype("float64").filled(np.nan)
        profile = src.profile.copy()
    if profile.get("crs") is None:
        raise ValueError(
            f"Raster {p.name} has no CRS; cannot align it to the LCZ grid."
        )
    return data, profile


def _align_to_lcz(
    lcz_profile: dict,
    variable_path: Union[str, Path],
    resampling: str = "average",
) -> np.ndarray:
    """Reproject/resample the variable raster onto the LCZ raster's grid."""
    p = Path(variable_path)
    if not p.exists():
        raise FileNotFoundError(f"Raster not found: {p}")

    if resampling not in Resampling.__members__:
        raise ValueError(
            f"Invalid resampling method {resampling!r}. "
            f"Options: {', '.join(Resampling.__members__)}"
        )
    method = Resampling[resampling]

    with rasterio.open(str(p)) as src:
        if src.crs is None:
            raise ValueError(
                f"Raster {p.name} has no CRS; cannot align it to the LCZ grid."
            )
        same_grid = (
            src.crs == lcz_profile["crs"]
            and src.width == lcz_profile["width"]
            and src.height == lcz_profile["height"]
            and src.transform.almost_equals(lcz_profile["transform"], precision=1e-9)
        )
        if same_grid:
            return src.read(1, masked=True).astype("float64").filled(np.nan)

        dst = np.full(
            (lcz_profile["height"], lcz_profile["width"]), np.nan, dtype="float64",
        )
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=lcz_profile["transform"], dst_crs=lcz_profile["crs"],
            resampling=method,
            src_nodata=src.nodata, dst_nodata=np.nan,
            init_dest_nodata=True,
        )
        return dst


# ── Statistics ────────────────────────────────────────────────────────────────

@dataclass
class _Fit:
    """Simple linear fit with a 95% confidence band for the mean response."""
    slope: float
    intercept: float
    r2: float
    p_value: float
    n: int
    xs: np.ndarray
    yhat: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def equation(self) -> str:
        sign = "+" if self.intercept >= 0 else "−"
        return f"y = {self.slope:.3g}·x {sign} {abs(self.intercept):.3g}"


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "p = n/a"
    if p < 1e-4:
        return "p < 0.0001"
    return f"p = {p:.4g}"


def _linear_fit(x: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> Optional[_Fit]:
    """OLS fit of ``y ~ x`` plus the mean-response confidence band.

    Returns ``None`` when the fit is not defined (fewer than 3 points, or no
    spread in x).
    """
    if x.size < 3 or y.size != x.size:
        return None
    xbar = float(np.mean(x))
    sxx = float(np.sum((x - xbar) ** 2))
    if not np.isfinite(sxx) or sxx <= 0:
        return None
    if float(np.var(y)) <= 0:  # constant response: no regression to fit
        return None

    with np.errstate(invalid="ignore", divide="ignore"):
        res = spstats.linregress(x, y)
    if not (np.isfinite(res.slope) and np.isfinite(res.intercept)):
        return None

    yhat_all = res.intercept + res.slope * x
    sse = float(np.sum((y - yhat_all) ** 2))
    dof = x.size - 2
    s = np.sqrt(sse / dof) if dof > 0 else np.nan

    xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    yhat = res.intercept + res.slope * xs
    if np.isfinite(s):
        tcrit = float(spstats.t.ppf(1 - (1 - confidence) / 2, dof))
        se_mean = s * np.sqrt(1.0 / x.size + (xs - xbar) ** 2 / sxx)
        lower = yhat - tcrit * se_mean
        upper = yhat + tcrit * se_mean
    else:
        lower = upper = np.full(xs.shape, np.nan)

    return _Fit(
        slope=float(res.slope),
        intercept=float(res.intercept),
        r2=float(res.rvalue**2),
        p_value=float(res.pvalue),
        n=int(x.size),
        xs=xs,
        yhat=yhat,
        lower=lower,
        upper=upper,
    )


def _add_fit_traces(
    fig: go.Figure, x: np.ndarray, y: np.ndarray, color: str = "#1a1a1a",
) -> Optional[_Fit]:
    """Add the OLS line + 95% confidence band; return the fit."""
    fit = _linear_fit(x, y)
    if fit is None:
        logger.info("Not enough spread/points for a regression line.")
        return None

    fig.add_trace(go.Scatter(
        x=fit.xs, y=fit.lower, mode="lines", line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fit.xs, y=fit.upper, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(70, 70, 70, 0.18)",
        hoverinfo="skip", showlegend=False, name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=fit.xs, y=fit.yhat, mode="lines",
        line=dict(color=color, width=2.5),
        name="OLS fit", hoverinfo="skip",
    ))
    return fit


# ── Figure construction ───────────────────────────────────────────────────────

def _palette(inclusive: bool) -> list[str]:
    return LCZ_COLORBLIND if inclusive else LCZ_COLORS


def _axis_title(variable_name: str, variable_units: Optional[str]) -> str:
    if not variable_units:
        return variable_name
    unit = variable_units.strip()
    if unit.startswith("[") and unit.endswith("]"):
        return f"{variable_name} {unit}"
    return f"{variable_name} [{unit}]"


def _scatter_figure(
    df: pd.DataFrame,
    variable_name: str,
    palette: list[str],
    names: list[str],
    lang: str,
    variable_units: Optional[str],
    add_regression: bool,
    max_points: int,
) -> go.Figure:
    """Jittered per-pixel scatter of class code vs value, with OLS overlay."""
    fig = go.Figure()

    # Display subsample (the fit below still uses every valid pixel).
    n = len(df)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
        df_plot = df.iloc[idx].reset_index(drop=True)
    else:
        df_plot = df.reset_index(drop=True)

    rng = np.random.default_rng(0)
    x_base = df_plot["lcz_id"].to_numpy(dtype=float)
    jitter = rng.uniform(-0.22, 0.22, size=len(df_plot))

    for i, lcz_id in enumerate(sorted(df_plot["lcz_id"].unique())):
        mask = (df_plot["lcz_id"] == lcz_id).to_numpy()
        if not mask.any():
            continue
        name = names[int(lcz_id) - 1]
        fig.add_trace(go.Scattergl(
            x=x_base[mask] + jitter[mask],
            y=df_plot.loc[mask, variable_name].to_numpy(),
            mode="markers",
            marker=dict(
                color=palette[int(lcz_id) - 1],
                symbol=_MARKER_SYMBOLS[i % len(_MARKER_SYMBOLS)],
                size=8,
                line=dict(color="black", width=0.4),
            ),
            name=f"{lcz_id} · {name}",
            customdata=np.full(int(mask.sum()), f"LCZ {lcz_id} — {name}", dtype=object),
            hovertemplate=(
                "%{customdata}<br>"
                f"{variable_name}: %{{y:.3f}}<extra></extra>"
            ),
        ))

    fit = None
    if add_regression:
        fit = _add_fit_traces(
            fig, df["lcz_id"].to_numpy(dtype=float),
            df[variable_name].to_numpy(dtype=float),
        )

    present = sorted(df["lcz_id"].unique())
    fig.update_layout(
        xaxis=dict(
            title=dict(text=lcz_msg("lcz_code_label", lang)),
            tickmode="array",
            tickvals=present,
            ticktext=[str(int(v)) for v in present],
            range=[0.4, 17.6],
        ),
        yaxis=dict(title=dict(text=_axis_title(variable_name, variable_units))),
        legend=dict(title=dict(text=lcz_msg("lcz_class_legend", lang)),
                    itemsizing="constant"),
        margin=dict(l=70, r=30, t=80, b=60),
        width=900, height=640,
        hoverlabel=dict(bgcolor="white"),
    )

    if fit is not None:
        fig.add_annotation(
            text=(
                f"<b>{fit.equation()}</b><br>"
                f"R² = {fit.r2:.3f}<br>"
                f"{_format_p(fit.p_value)}<br>"
                f"n = {fit.n:,}"
            ),
            xref="paper", yref="paper", x=0.02, y=0.98,
            xanchor="left", yanchor="top", align="left",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#888888", borderwidth=1, borderpad=6,
            font=dict(size=11),
        )
    return fig


def _distribution_figure(
    df: pd.DataFrame,
    variable_name: str,
    plot_type: str,
    color_map: dict[str, str],
    order: list[str],
    variable_units: Optional[str],
    lang: str,
) -> go.Figure:
    plot_fn = px.box if plot_type == "box" else px.violin
    kwargs = dict(
        x="lcz_name", y=variable_name, color="lcz_name",
        color_discrete_map=color_map, category_orders={"lcz_name": order},
    )
    if plot_type == "box":
        kwargs["points"] = "outliers"
    else:
        # ``pointpos`` was removed from px.violin in Plotly 6; the default
        # jitter is the correct replacement and works on 5.x too.
        kwargs["box"] = True
    fig = plot_fn(df, **kwargs)
    fig.update_layout(
        xaxis_title=lcz_msg("lcz_code_label", lang),
        yaxis_title=_axis_title(variable_name, variable_units),
        showlegend=False,
        height=560,
    )
    return fig


def _heatmap_figure(
    df: pd.DataFrame,
    variable_name: str,
    order: list[str],
    agg_fun: str,
    variable_units: Optional[str],
    lang: str,
) -> go.Figure:
    summary = df.groupby("lcz_name")[variable_name].agg(agg_fun).reindex(order)
    present = [name for name in order if name in summary.index and pd.notna(summary[name])]
    values = summary.loc[present].to_numpy(dtype=float)

    fig = go.Figure(go.Heatmap(
        z=[values],
        x=present,
        colorscale="Cividis",  # perceptually uniform, colorblind-safe
        colorbar=dict(
            title=dict(text=_axis_title(variable_name, variable_units)),
            thickness=14,
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{agg_fun}({variable_name}) = %{{z:.3f}}<extra></extra>"
        ),
        text=[[f"{v:.2f}" for v in values]],
        texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        xaxis=dict(title=dict(text=lcz_msg("lcz_code_label", lang)), tickangle=-35),
        yaxis=dict(showticklabels=False),
        height=420,
    )
    return fig


def _finalize(
    fig: go.Figure,
    style: str,
    isave: bool,
    save_extension: str,
    filename: str,
    dpi: int,
    lang: str,
) -> go.Figure:
    """Apply the publication template and optionally export at ``dpi``."""
    spec = get_style(style, lang)
    fig.update_layout(template=spec.template)

    if not isave:
        return fig

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{filename}.{save_extension}")

    if save_extension == "html":
        fig.write_html(save_path, include_plotlyjs="cdn")
        logger.info("Saved: %s", os.path.abspath(save_path))
        return fig

    if dpi == spec.dpi or save_extension == "tiff":
        # Delegate to the shared exporter (also handles .tiff metadata).
        from LCZ4py._internal.lcz_theme import finalize_export
        return finalize_export(
            fig, style=style, isave=True, save_extension=save_extension,
            filename=filename, column="double", lang=lang,
        )

    width_mm = spec.width_mm.get("double", spec.width_mm["single"])
    height_mm = min(spec.max_height_mm, width_mm * 0.72)
    save_plotly_figure(
        fig,
        save_path,
        width_px=round(width_mm / 25.4 * dpi),
        height_px=round(height_mm / 25.4 * dpi),
        scale=1,
        save_extension=save_extension,
        lang=lang,
    )
    logger.info("Saved: %s (%d dpi)", os.path.abspath(save_path), dpi)
    return fig


# ── Native static charts (matplotlib/Agg — no Plotly/Kaleido/browser) ────────

#: matplotlib equivalents of the Plotly marker names in ``_MARKER_SYMBOLS``.
_MPL_MARKERS = {
    "circle": "o", "square": "s", "diamond": "D",
    "triangle-up": "^", "cross": "x",
}


def _style_axes(ax, spec, *, grid_axis: str = "y") -> None:
    """Consistent panel styling: small ticks, light grid, open frame."""
    ax.tick_params(axis="both", labelsize=max(5.5, spec.font_size - 2), length=2.5, width=0.6)
    ax.grid(axis=grid_axis, color="#e8e8e8", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _class_legend(fig, present: list[int], palette: list[str], names: list[str],
                  spec, *, lang: str, y: float = 0.02) -> None:
    """Marker legend below the panel, one entry per LCZ class."""
    from matplotlib.lines import Line2D

    if not present:
        return None
    fontsize = max(5.0, spec.font_size - 2)
    handles = [
        Line2D([], [], linestyle="",
               marker=_MPL_MARKERS[_MARKER_SYMBOLS[i % len(_MARKER_SYMBOLS)]],
               markersize=4.5, markerfacecolor=palette[int(c) - 1],
               markeredgecolor="black", markeredgewidth=0.3,
               label=f"{int(c)} · {names[int(c) - 1]}")
        for i, c in enumerate(present)
    ]
    return fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=min(3, max(2, len(handles))),
        frameon=False,
        fontsize=fontsize,
        handlelength=0.9,
        columnspacing=1.1,
        handletextpad=0.35,
        title=lcz_msg("lcz_class_legend", lang),
        title_fontsize=fontsize,
    )


def _draw_scatter_static(
    df: pd.DataFrame,
    variable_name: str,
    palette: list[str],
    names: list[str],
    variable_units: Optional[str],
    add_regression: bool,
    max_points: int,
    *,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Jittered per-pixel scatter of class code vs value, with OLS overlay."""
    fig, spec, _ = _st.new_figure(
        style, aspect=0.72, extra_height_mm=36.0, column="single", dpi=dpi,
    )
    ax = _st.add_axes(
        fig, bottom=0.39, top=_st.figure_top(bool(title or subtitle)),
        left=0.14, right=0.97,
    )

    n = len(df)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
        df_plot = df.iloc[idx].reset_index(drop=True)
    else:
        df_plot = df.reset_index(drop=True)

    rng = np.random.default_rng(0)
    x_base = df_plot["lcz_id"].to_numpy(dtype=float)
    jitter = rng.uniform(-0.22, 0.22, size=len(df_plot))
    present = sorted(int(v) for v in df_plot["lcz_id"].unique())

    for i, lcz_id in enumerate(present):
        mask = (df_plot["lcz_id"] == lcz_id).to_numpy()
        if not mask.any():
            continue
        marker = _MPL_MARKERS[_MARKER_SYMBOLS[i % len(_MARKER_SYMBOLS)]]
        marker_kwargs: dict = dict(s=11.0, alpha=0.85, c=palette[lcz_id - 1])
        if marker == "x":  # unfilled marker: edgecolour would be ignored
            marker_kwargs["linewidths"] = 0.6
        else:
            marker_kwargs["edgecolors"] = "black"
            marker_kwargs["linewidths"] = 0.25
        ax.scatter(
            x_base[mask] + jitter[mask],
            df_plot.loc[mask, variable_name].to_numpy(dtype=float),
            marker=marker, **marker_kwargs,
        )

    if add_regression:
        fit = _linear_fit(
            df["lcz_id"].to_numpy(dtype=float),
            df[variable_name].to_numpy(dtype=float),
        )
        if fit is None:
            logger.info("Not enough spread/points for a regression line.")
        else:
            ax.fill_between(
                fit.xs, fit.lower, fit.upper, color="#464646", alpha=0.18,
                linewidth=0, label="95% CI",
            )
            ax.plot(fit.xs, fit.yhat, color="#1a1a1a", linewidth=1.4, label="OLS fit")
            ax.annotate(
                (
                    f"{fit.equation()}\n"
                    f"R² = {fit.r2:.3f}\n"
                    f"{_format_p(fit.p_value)}\n"
                    f"n = {fit.n:,}"
                ),
                xy=(0.02, 0.98), xycoords="axes fraction",
                ha="left", va="top", fontsize=max(5.0, spec.font_size - 2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#888888", linewidth=0.5, alpha=0.9),
            )

    ax.set_xlim(0.4, 17.6)
    ax.set_xticks(present)
    ax.set_xticklabels([str(c) for c in present])
    ax.set_xlabel(lcz_msg("lcz_code_label", lang))
    ax.set_ylabel(_axis_title(variable_name, variable_units))
    _style_axes(ax, spec, grid_axis="both")

    _class_legend(fig, present, palette, names, spec, lang=lang, y=0.02)
    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _draw_distribution_static(
    df: pd.DataFrame,
    variable_name: str,
    plot_type: str,
    palette: list[str],
    names: list[str],
    variable_units: Optional[str],
    *,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Box (quartiles/whiskers/outliers) or violin distribution per class."""
    fig, spec, _ = _st.new_figure(style, aspect=0.68, column="single", dpi=dpi)
    ax = _st.add_axes(
        fig, bottom=0.19, top=_st.figure_top(bool(title or subtitle)),
        left=0.15, right=0.97,
    )

    present = sorted(int(v) for v in df["lcz_id"].unique())
    datasets = [
        df.loc[df["lcz_id"] == c, variable_name].dropna().to_numpy(dtype=float)
        for c in present
    ]
    positions = np.arange(len(present), dtype=float)
    colors = [palette[c - 1] for c in present]

    if plot_type == "box":
        result = ax.boxplot(
            datasets, positions=positions, widths=0.66, patch_artist=True,
            showmeans=True, manage_ticks=False,
            boxprops=dict(linewidth=0.5, edgecolor="black"),
            medianprops=dict(color="black", linewidth=0.8),
            whiskerprops=dict(linewidth=0.5),
            capprops=dict(linewidth=0.5),
            meanprops=dict(marker="o", markersize=2.6, markerfacecolor="white",
                           markeredgecolor="black", markeredgewidth=0.5),
            flierprops=dict(marker=".", markersize=1.8, markerfacecolor="#333333",
                            markeredgecolor="none", alpha=0.5),
        )
        for patch, color in zip(result["boxes"], colors):
            patch.set_facecolor(color)
    else:
        result = ax.violinplot(
            datasets, positions=positions, widths=0.80, showmeans=True,
            showextrema=False,
        )
        for body, color in zip(result["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_linewidth(0.5)
            body.set_alpha(0.85)
        if "cmeans" in result:
            result["cmeans"].set(color="black", linewidth=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(c) for c in present])
    ax.set_xlabel(lcz_msg("lcz_code_label", lang))
    ax.set_ylabel(_axis_title(variable_name, variable_units))
    _style_axes(ax, spec, grid_axis="y")

    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _draw_heatmap_static(
    df: pd.DataFrame,
    variable_name: str,
    order: list[str],
    agg_fun: str,
    variable_units: Optional[str],
    names: list[str],
    palette: list[str],
    *,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Single-row heatmap of the per-class aggregate (Cividis, colourblind-safe).

    Class **codes** label the axis (17 names never fit a 89-120 mm row) and the
    full names sit in a legend below; values are printed vertically so they
    cannot overprint in narrow cells.
    """
    from matplotlib.patches import Patch

    summary = df.groupby("lcz_name")[variable_name].agg(agg_fun).reindex(order)
    present = [name for name in order if name in summary.index and pd.notna(summary[name])]
    values = summary.loc[present].to_numpy(dtype=float)
    codes = [names.index(name) + 1 for name in present]

    fig, spec, _ = _st.new_figure(
        style, aspect=0.60, extra_height_mm=26.0, column="single", dpi=dpi,
    )
    ax = _st.add_axes(
        fig, bottom=0.40, top=_st.figure_top(bool(title or subtitle)),
        left=0.11, right=0.86,
    )

    image = ax.imshow(
        values[None, :], cmap=_st.mpl_cmap("cividis"), aspect="auto",
    )
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([str(c) for c in codes])
    ax.set_yticks([])
    ax.set_xlabel(lcz_msg("lcz_code_label", lang))
    ax.tick_params(axis="x", labelsize=max(5.0, spec.font_size - 2),
                   length=2.0, width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    # Value labels, rotated 90° so narrow cells stay legible.
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    span = (vmax - vmin) or 1.0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        ax.text(
            i, 0, f"{value:.2f}", ha="center", va="center", rotation=90,
            fontsize=max(4.5, spec.font_size - 2),
            color="white" if (value - vmin) / span > 0.55 else "black",
        )

    _st.add_colorbar(
        fig, ax, image, units=_axis_title(variable_name, variable_units),
        fontsize=max(5.5, spec.font_size - 2),
    )

    fontsize = max(5.0, spec.font_size - 2)
    fig.legend(
        handles=[
            Patch(facecolor=palette[c - 1], edgecolor="black", linewidth=0.3,
                  label=f"{c}: {names[c - 1]}")
            for c in codes
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=min(3, max(2, len(codes))), frameon=False, fontsize=fontsize,
        handlelength=1.1, columnspacing=1.1, handletextpad=0.4,
        title=lcz_msg("lcz_class_legend", lang), title_fontsize=fontsize,
    )
    _st.add_header(fig, title, subtitle, spec)
    _st.add_footer(fig, caption, None, spec)
    return fig


def _relationship_static(
    df: pd.DataFrame,
    plot_type: str,
    *,
    variable_name: str,
    palette: list[str],
    names: list[str],
    order: list[str],
    agg_fun: str,
    variable_units: Optional[str],
    add_regression: bool,
    max_points: int,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    lang: str,
    style: str,
    dpi: Optional[int],
):
    """Build the requested relationship chart with matplotlib/Agg."""
    with _st.style_context(style):
        if plot_type in ("box", "violin"):
            fig = _draw_distribution_static(
                df, variable_name, plot_type, palette, names, variable_units,
                title=title, subtitle=subtitle, caption=caption,
                lang=lang, style=style, dpi=dpi,
            )
        elif plot_type == "scatter":
            fig = _draw_scatter_static(
                df, variable_name, palette, names, variable_units,
                add_regression, max_points,
                title=title, subtitle=subtitle, caption=caption,
                lang=lang, style=style, dpi=dpi,
            )
        else:
            fig = _draw_heatmap_static(
                df, variable_name, order, agg_fun, variable_units,
                names, palette,
                title=title, subtitle=subtitle, caption=caption,
                lang=lang, style=style, dpi=dpi,
            )
    return fig


# ── Public API ────────────────────────────────────────────────────────────────

def plot_lcz_relationship(
    lcz_path: Union[str, Path],
    variable_path: Union[str, Path],
    variable_name: str = "value",
    plot_type: Literal["box", "violin", "scatter", "heatmap"] = "box",
    agg_fun: str = "mean",
    isave: bool = False,
    save_extension: str = "html",
    style: str = "default",
    lang: str = "en",
    inclusive: bool = False,
    variable_units: Optional[str] = None,
    add_regression: bool = True,
    resampling: str = "average",
    max_points: int = 20_000,
    dpi: int = 300,
    renderer: Optional[str] = "static",
) -> Union[go.Figure, "Figure"]:
    """Plot the distribution of a gridded variable across LCZ classes.

    Parameters
    ----------
    lcz_path : str or Path
        Path to an LCZ classification GeoTIFF (values 1-17), e.g. from
        ``lcz_get_map``.
    variable_path : str or Path
        Path to a single-band raster of the environmental variable (e.g. an
        ERA5/CHIRPS/pollution GeoTIFF for the same region/date). Reprojected
        and resampled onto the LCZ raster's grid if its CRS/resolution differ.
    variable_name : str
        Label for the variable, used as the value-axis title. Default "value".
    plot_type : {"box", "violin", "scatter", "heatmap"}
        "box"/"violin": distribution of variable values per LCZ class.
        "scatter": per-pixel value vs. LCZ class (jittered), with an OLS fit,
        its equation/R²/p-value and a 95% confidence band.
        "heatmap": the ``agg_fun`` of the variable per LCZ class as a
        single-row heatmap.
    agg_fun : str
        Aggregation used for "heatmap": "mean" (default), "median", "sum",
        "min", "max".
    isave : bool
        Save the figure to ``LCZ4r_output/``. Default False.
    save_extension : str
        File extension when ``isave``. Default "html" (Plotly renderer);
        "html" is coerced to "png" for the static renderer.
    style : str
        Publication style preset: 'default', 'nature', 'science', or
        'generic_bw'. Controls font, figure size (mm), DPI, and palette
        used when isave and save_extension != 'html'.
    lang : str
        Message language. Default "en".
    inclusive : bool
        Use the colorblind-safe LCZ palette (``LCZ_COLORBLIND``).
    variable_units : str, optional
        Units for the value axis, e.g. ``"°C"`` or ``"mm"``. Appended as
        ``"[unit]"`` unless the label already carries brackets.
    add_regression : bool
        Overlay the OLS fit, equation, R², p-value and confidence band on the
        scatter plot. Default True.
    resampling : str
        Rasterio resampling method used when the variable raster must be
        reprojected onto the LCZ grid. Default "average" (continuous data);
        use "nearest" for categorical variables.
    max_points : int
        Maximum number of points drawn in the scatter (display only — the
        regression always uses every valid pixel).
    dpi : int
        Resolution used for static exports. Default 300 (publication grade).
    renderer : {"static", "plotly"}, optional
        "static" (default, also ``None``/``"matplotlib"``/``"agg"``) returns a
        matplotlib ``Figure`` saved as native PNG/PDF/SVG/TIFF at >=300 dpi
        with no Plotly, Kaleido, Chromium or browser. "plotly" returns the
        previous interactive figure.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
        matplotlib ``Figure`` for the default static renderer, Plotly
        ``go.Figure`` when ``renderer="plotly"``.
    """
    if plot_type not in ("box", "violin", "scatter", "heatmap"):
        raise ValueError(lcz_msg(
            "plot_lcz_rel_invalid_type", lang, valid="box, violin, scatter, heatmap",
        ))
    # None / "matplotlib" / "agg" -> the native static renderer (default).
    renderer = _st.resolve_renderer(renderer)
    if agg_fun not in _AGG_FUNS:
        raise ValueError(
            f"agg_fun must be one of {_AGG_FUNS}, got {agg_fun!r}."
        )
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi!r}")

    lcz_arr, lcz_profile = _read_band(lcz_path)
    var_arr = _align_to_lcz(lcz_profile, variable_path, resampling=resampling)

    valid = np.isin(lcz_arr, LCZ_IDS) & np.isfinite(var_arr)
    if not valid.any():
        raise ValueError(lcz_msg("plot_lcz_rel_no_overlap", lang))

    lcz_codes = lcz_arr[valid].astype(int)
    values = var_arr[valid]
    names = get_lcz_names(lang)
    df = pd.DataFrame({
        "lcz_id": lcz_codes,
        "lcz_name": [names[c - 1] for c in lcz_codes],
        variable_name: values,
    })

    palette = _palette(inclusive)
    color_map = {names[i]: palette[i] for i in range(len(names))}
    order = [n for n in names if n in set(df["lcz_name"])]
    base_title = lcz_msg("cal_indexes_title", lang, variable=variable_name)
    filename = f"lcz_relationship_{plot_type}"

    # ── Native static renderer (default): matplotlib/Agg, no browser ────────
    if renderer == "static":
        fig = _relationship_static(
            df, plot_type,
            variable_name=variable_name, palette=palette, names=names,
            order=order, agg_fun=agg_fun, variable_units=variable_units,
            add_regression=add_regression, max_points=max_points,
            title=base_title, subtitle=None, caption=None,
            lang=lang, style=style, dpi=dpi,
        )
        _st.save_static(
            fig, filename, isave=isave, save_extension=save_extension,
            style=style, dpi=dpi, lang=lang,
        )
        return fig

    # ── Plotly renderer (opt-in): previous interactive figure ───────────────
    if plot_type in ("box", "violin"):
        fig = _distribution_figure(
            df, variable_name, plot_type, color_map, order, variable_units, lang,
        )
    elif plot_type == "scatter":
        fig = _scatter_figure(
            df, variable_name, palette, names, lang,
            variable_units, add_regression, max_points,
        )
    else:
        fig = _heatmap_figure(
            df, variable_name, order, agg_fun, variable_units, lang,
        )

    fig.update_layout(
        title=dict(
            text=base_title,
            x=0.5, xanchor="center",
        ),
    )

    return _finalize(
        fig, style, isave, save_extension,
        filename=filename, dpi=dpi, lang=lang,
    )


__all__ = ["plot_lcz_relationship"]
