"""
lcz_variogram.py — empirical semivariogram computation and theoretical model fitting.

Computes pairwise semivariances between stations, bins them into lag classes,
and fits Spherical, Exponential, or Gaussian theoretical variogram models.

Per-time-step empirical variograms are independent, so they are computed with
a bounded ``ThreadPoolExecutor`` (``max_workers``) while results are
re-assembled in deterministic (date) order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import polars as pl
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:  # ImportError or a broken plotly install
    go = None  # type: ignore[assignment]
    HAS_PLOTLY = False

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, extract_lcz_at_points, normalise_input_df,
    normalise_missing, select_by_date, utm_epsg_for,
)
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export, get_style

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 4)

# Local fallbacks for variogram keys that are not (yet) registered in
# ``i18n_messages.LCZ_MESSAGES``.  ``lcz_msg`` returns a
# "<missing i18n key: ...>" sentinel for unknown keys, which must never reach
# a publication figure, so we substitute an English default instead.
_FALLBACKS: dict[str, str] = {
    "variogram_title": "Empirical Semivariogram",
    "variogram_xlabel": "Lag distance [m]",
    "variogram_ylabel": "Semivariance",
    "variogram_no_data": "No valid observations available for the variogram.",
    "variogram_no_pairs": "Not enough station pairs to compute a variogram.",
    "variogram_complete": (
        "{model} variogram fitted — R²={r2}, nugget={nugget}, "
        "sill={sill}, range={range} m"
    ),
}


def _msg(key: str, lang: str, **kwargs) -> str:
    """``lcz_msg`` with an English fallback for not-yet-registered keys."""
    text = lcz_msg(key, lang, **kwargs)
    if text.startswith("<missing i18n key:"):
        template = _FALLBACKS.get(key, key)
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError):
            return template
    return text


# ── Theoretical variogram models ──────────────────────────────────────────────

def _spherical(h: np.ndarray, c0: float, c1: float, a: float) -> np.ndarray:
    """Spherical model: γ(h) = c0 + c1 * (1.5h/a − 0.5(h/a)³) for h ≤ a, else c0+c1."""
    h = np.asarray(h, dtype=np.float64)
    a = max(float(a), 1e-12)
    out = np.full(h.shape, c0 + c1, dtype=np.float64)
    mask = h <= a
    r = h[mask] / a
    out[mask] = c0 + c1 * (1.5 * r - 0.5 * r ** 3)
    return out


def _exponential(h: np.ndarray, c0: float, c1: float, a: float) -> np.ndarray:
    """Exponential model: γ(h) = c0 + c1 * (1 − exp(−3h/a))."""
    a = max(float(a), 1e-12)
    return c0 + c1 * (1.0 - np.exp(-3.0 * np.asarray(h, dtype=np.float64) / a))


def _gaussian(h: np.ndarray, c0: float, c1: float, a: float) -> np.ndarray:
    """Gaussian model: γ(h) = c0 + c1 * (1 − exp(−3h²/a²))."""
    a = max(float(a), 1e-12)
    return c0 + c1 * (1.0 - np.exp(-3.0 * np.asarray(h, dtype=np.float64) ** 2 / a ** 2))


_MODELS = {
    "Sph": _spherical,
    "Exp": _exponential,
    "Gau": _gaussian,
}

_MODEL_LABELS = {
    "Sph": "Spherical",
    "Exp": "Exponential",
    "Gau": "Gaussian",
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class VariogramResult:
    """Return type for :func:`lcz_variogram`.

    Attributes
    ----------
    df : pl.DataFrame
        Empirical variogram with columns: ``lag_center``, ``semivariance``,
        ``n_pairs``, ``model_fitted``.
    params : dict
        Fitted model parameters: ``nugget``, ``sill``, ``range``, ``model_type``.
    r2 : float
        Coefficient of determination of the fitted model.
    plot : go.Figure or None
        Plotly figure with empirical scatter and fitted curve (if plotly is installed).
    """
    df: pl.DataFrame
    params: dict
    r2: float
    plot: Optional[object] = field(default=None, repr=False)


_EMPTY_SCHEMA = {
    "lag_center": pl.Float64,
    "semivariance": pl.Float64,
    "n_pairs": pl.Float64,
    "model_fitted": pl.Float64,
}


def _empty_result() -> VariogramResult:
    return VariogramResult(
        df=pl.DataFrame(schema=_EMPTY_SCHEMA), params={}, r2=0.0, plot=None,
    )


# ── Pairwise computation helpers ──────────────────────────────────────────────

def _compute_empirical_variogram(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_lags: int,
    max_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute empirical semivariogram from station coordinates and values.

    Returns
    -------
    lag_centers : np.ndarray
        Mean distance per lag bin.
    semivariances : np.ndarray
        Mean semivariance per lag bin.
    n_pairs : np.ndarray
        Number of pairs per lag bin (empty bins keep the bin centre and a
        zero semivariance, with ``n_pairs == 0`` so the fit can ignore them).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y and z must have the same length.")
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]

    n_lags = int(max(1, n_lags))
    if len(x) < 2:
        raise ValueError("At least 2 valid stations are required for a variogram.")

    dists = pdist(np.column_stack([x, y]))
    diffs = pdist(z.reshape(-1, 1), metric="sqeuclidean")
    semivariances = 0.5 * diffs

    max_dist = float(max_dist)
    if not np.isfinite(max_dist) or max_dist <= 0:
        max_dist = float(dists.max()) if dists.size else 1.0
    if max_dist <= 0:
        raise ValueError("max_dist must be > 0 (all stations are coincident).")

    lag_edges = np.linspace(0.0, max_dist, n_lags + 1)
    # Vectorised binning (digitize returns 1..n_lags; clamp the right edge).
    bin_idx = np.digitize(dists, lag_edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, n_lags - 1)

    counts = np.bincount(bin_idx, minlength=n_lags).astype(np.float64)
    sum_gamma = np.bincount(bin_idx, weights=semivariances, minlength=n_lags)
    sum_dist = np.bincount(bin_idx, weights=dists, minlength=n_lags)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_gamma = np.where(counts > 0, sum_gamma / counts, 0.0)
        lag_centers = np.where(
            counts > 0, sum_dist / counts, (lag_edges[:-1] + lag_edges[1:]) / 2.0,
        )

    return lag_centers, mean_gamma, counts


def _fit_variogram_model(
    lag_centers: np.ndarray,
    semivariances: np.ndarray,
    n_pairs: np.ndarray,
    model: str,
) -> tuple[dict, float]:
    """Fit a theoretical variogram model via weighted least-squares.

    Returns
    -------
    params : dict
        ``{nugget, sill, range, model_type}``.
    r2 : float
        Coefficient of determination.

    Notes
    -----
    The fit never raises on convergence failure: it falls back to
    moment-style initial estimates and logs a warning.
    """
    fn = _MODELS.get(model)
    if fn is None:
        raise ValueError(
            f"Unknown model '{model}'. Choose from {list(_MODELS)}."
        )

    lag_centers = np.asarray(lag_centers, dtype=np.float64)
    semivariances = np.asarray(semivariances, dtype=np.float64)
    n_pairs = np.asarray(n_pairs, dtype=np.float64)

    valid = np.isfinite(semivariances) & np.isfinite(lag_centers) & (n_pairs > 0)
    x_fit = lag_centers[valid]
    y_fit = semivariances[valid]
    w = np.sqrt(np.maximum(n_pairs[valid], 1.0))

    if x_fit.size < 2:
        logger.warning(
            "Variogram fit needs at least 2 populated lag bins, got %d — "
            "returning moment estimates.", x_fit.size,
        )
        sill = float(np.var(y_fit)) if y_fit.size else 0.0
        return ({"nugget": 0.0, "sill": sill, "range": 1.0, "model_type": model}, 0.0)

    if np.allclose(y_fit, y_fit[0]):
        # Zero-variance field: a flat variogram is the only valid answer.
        logger.warning(
            "All lag-bin semivariances are identical (%.3g) — the variable is "
            "spatially constant; the variogram model is not identifiable.", y_fit[0],
        )
        return ({"nugget": float(y_fit[0]), "sill": 0.0, "range": float(x_fit.max()),
                 "model_type": model}, 0.0)

    # Initial parameter guesses
    c0_0 = float(np.min(y_fit))
    c1_0 = max(float(np.max(y_fit) - c0_0), 1e-9)
    a_0 = float(np.max(x_fit)) / 2.0 or 1.0

    popt = None
    try:
        popt, _ = curve_fit(
            fn, x_fit, y_fit, p0=[c0_0, c1_0, a_0],
            bounds=([0.0, 0.0, 1e-9], [np.inf, np.inf, np.inf]),
            sigma=1.0 / np.maximum(w, 1e-12), maxfev=20000,
        )
        if not np.all(np.isfinite(popt)):
            popt = None
    except Exception as exc:  # noqa: BLE001 - ValueError, RuntimeError, ...
        logger.warning(
            "Variogram model '%s' did not converge (%s: %s) — using moment "
            "estimates.", model, type(exc).__name__, exc,
        )
        popt = None

    if popt is None:
        c0, c1, a = c0_0, c1_0, a_0
    else:
        c0, c1, a = (float(popt[0]), float(popt[1]), float(popt[2]))

    predicted = fn(x_fit, c0, c1, a)
    ss_res = float(np.sum((y_fit - predicted) ** 2))
    ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"nugget": c0, "sill": c1, "range": a, "model_type": model}, r2


def _build_variogram_plot(
    lag_centers: np.ndarray,
    semivariances: np.ndarray,
    n_pairs: np.ndarray,
    params: dict,
    r2: float,
    lang: str,
    style: str = "default",
    units: Optional[str] = None,
) -> "go.Figure":
    """Build a publication-ready Plotly figure (empirical scatter + fitted model).

    Colours come from the active :class:`~LCZ4py._internal.lcz_theme.StyleSpec`
    preset (no hardcoded LCZ palette), the point size scales with the number of
    pairs in each bin, and the fitted parameters are reported in the legend.
    """
    spec = get_style(style, lang)
    c_empirical = spec.colorway[1] if len(spec.colorway) > 1 else "#1f4e79"
    c_fit = spec.colorway[7] if len(spec.colorway) > 7 else "#b22222"

    x_max = float(np.nanmax(lag_centers)) * 1.05 if len(lag_centers) else 1.0
    x_fit = np.linspace(0.0, x_max, 300)
    fn = _MODELS[params["model_type"]]
    y_fit = fn(x_fit, params["nugget"], params["sill"], params["range"])

    model_label = _MODEL_LABELS.get(params["model_type"], params["model_type"])
    var_label = f" [{units}]" if units else ""
    legend_text = (
        f"{model_label}: nugget={params['nugget']:.2f}{var_label}, "
        f"partial sill={params['sill']:.2f}{var_label}, "
        f"range={params['range']:.0f} m, R²={r2:.3f}"
    )

    # Marker size ∝ sqrt(pair count): denser bins are visually heavier.
    counts = np.asarray(n_pairs, dtype=np.float64)
    scale = np.sqrt(np.maximum(counts, 1.0))
    scale = scale / scale.max() if scale.max() > 0 else np.ones_like(scale)
    marker_sizes = 4.0 + 8.0 * scale

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lag_centers.tolist(),
        y=semivariances.tolist(),
        mode="markers",
        name="Empirical",
        marker=dict(size=marker_sizes.tolist(), color=c_empirical,
                    opacity=0.85, line=dict(width=1, color="white")),
        customdata=counts.tolist(),
        hovertemplate=(
            "Distance: %{x:.0f} m<br>Semivariance"
            + (f" [{units}]" if units else "")
            + ": %{y:.3f}<br>Pairs: %{customdata:.0f}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_fit.tolist(),
        y=y_fit.tolist(),
        mode="lines",
        name=legend_text,
        line=dict(width=2, dash="dash", color=c_fit),
        hovertemplate="Fitted %{y:.3f}<extra></extra>",
    ))
    # Total sill reference — makes the nugget/sill split readable.
    fig.add_hline(
        y=params["nugget"] + params["sill"], line_width=1, line_dash="dot",
        line_color="gray",
        annotation_text=f"total sill = {params['nugget'] + params['sill']:.2f}",
        annotation_position="top right",
        annotation_font_size=10,
    )

    fig.update_layout(
        title=dict(text=_msg("variogram_title", lang), x=0.5, xanchor="center"),
        xaxis_title=_msg("variogram_xlabel", lang),
        yaxis_title=_msg("variogram_ylabel", lang)
        + (f" [{units}]" if units else ""),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="lightgray", borderwidth=1),
        hovermode="closest",
    )
    fig.update_xaxes(range=[0, x_max])
    return fig


# ── Public API ────────────────────────────────────────────────────────────────

def lcz_variogram(
    x: Union[str, "os.PathLike", "rasterio.io.DatasetReader"],
    data_frame,
    var: str,
    station_id: str,
    *,
    sp_res: float = 100.0,
    model: str = "Sph",
    n_lags: int = 20,
    max_dist: Optional[float] = None,
    style: str = "default",
    lang: str = "en",
    year=None,
    month=None,
    day=None,
    hour=None,
    start=None,
    end=None,
    units: Optional[str] = None,
    isave: bool = False,
    save_extension: str = "html",
    max_workers: Optional[int] = None,
) -> VariogramResult:
    """Compute an empirical semivariogram and fit a theoretical model.

    Parameters
    ----------
    x : str, PathLike, or rasterio dataset
        LCZ raster used to determine UTM projection and extract LCZ classes.
    data_frame : pd.DataFrame or pl.DataFrame
        Observation table with lat/lon/date/var/station columns.
    var : str
        Column name for the meteorological variable.
    station_id : str
        Column name for the station identifier.
    sp_res : float
        Spatial resolution in metres.  Retained for API compatibility: the
        variogram is computed from station coordinates, not rasterised.
    model : {"Sph", "Exp", "Gau"}
        Theoretical variogram model to fit.
    n_lags : int
        Number of distance lag bins.
    max_dist : float, optional
        Maximum pairwise distance for the variogram (metres). If None, uses
        the 90th percentile of all inter-station distances.
    style : str
        Publication style preset: 'default', 'nature', 'science', or
        'generic_bw'. Controls font, figure size (mm), DPI, and palette
        used when isave and save_extension != 'html'.
    lang : str
        Language for log messages — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.
    year, month, day, hour : optional
        Date/time component filters.
    start, end : optional
        Date range filters.
    max_workers : int, optional
        Thread workers for the per-time-step empirical variograms
        (default ``min(8, cpu_count)``).
    units : str, optional
        Units of the interpolated variable (e.g. ``"°C"``, ``"m s-1"``), used
        in axis/colourbar labels and hover text.
    isave : bool
        Save the variogram figure to ``LCZ4r_output/lcz_variogram.<ext>``.
    save_extension : str
        ``"html"`` (interactive) or a static format such as ``"png"``/``"pdf"``.

    Returns
    -------
    VariogramResult
        Dataclass with fitted variogram DataFrame, model parameters, R², and plot.
    """
    if model not in _MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(_MODELS)}.")
    if not isinstance(n_lags, (int, np.integer)) or int(n_lags) < 1:
        raise ValueError("n_lags must be a positive integer.")
    if max_dist is not None and (not np.isfinite(max_dist) or max_dist <= 0):
        raise ValueError("max_dist must be a positive distance in metres.")

    df = normalise_input_df(
        data_frame.to_pandas() if isinstance(data_frame, pl.DataFrame) else data_frame,
        var=var, station_id=station_id,
    )
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null())

    if len(df) == 0:
        logger.warning(_msg("variogram_no_data", lang))
        return _empty_result()

    ds = load_lcz_raster(x)
    stations = df.unique(subset=["latitude", "longitude"])
    lcz_vals, mask = extract_lcz_at_points(
        ds, stations["longitude"].to_numpy(), stations["latitude"].to_numpy()
    )
    stations = stations.with_columns(
        pl.Series(name="lcz", values=np.where(mask, lcz_vals, 0)).cast(int)
    )

    # Project station coordinates to UTM
    epsg = utm_epsg_for(ds)
    from pyproj import Transformer
    tx = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    xs, ys = tx.transform(
        stations["longitude"].to_numpy(), stations["latitude"].to_numpy()
    )
    stations = stations.with_columns(x=xs, y=ys)
    stations = stations.filter(pl.col("lcz").is_between(1, 17))

    if stations.height < 3:
        logger.warning(
            "Only %d station(s) inside LCZ classes 1–17 — a variogram needs at "
            "least 3.", stations.height,
        )
        return _empty_result()

    df = df.join(
        stations.select(["latitude", "longitude", "lcz", "x", "y"]),
        on=["latitude", "longitude"], how="left",
    )
    df = df.filter(pl.col("lcz").is_between(1, 17))

    # ── Per-time-step empirical variograms (independent → thread pool) ──────
    groups: list[tuple[object, np.ndarray, np.ndarray]] = []
    for time_val, grp in df.group_by("date", maintain_order=True):
        if len(grp) < 3:
            continue
        coords = grp.select(["x", "y"]).to_numpy()
        vals = grp["var_interp"].to_numpy()
        if len(coords) < 3:
            continue
        groups.append((time_val, coords, vals))

    def _one(item):
        _label, coords, vals = item
        dists = pdist(coords)
        eff_max = float(max_dist) if max_dist is not None else float(
            np.percentile(dists, 90)
        )
        try:
            return _compute_empirical_variogram(
                coords[:, 0], coords[:, 1], vals, int(n_lags), eff_max
            )
        except ValueError as exc:
            logger.debug("Skipping a time step in the variogram: %s", exc)
            empty = np.zeros(int(n_lags))
            return empty, empty.copy(), empty.copy()

    if not groups:
        logger.warning(_msg("variogram_no_pairs", lang))
        return _empty_result()

    workers = DEFAULT_MAX_WORKERS if max_workers is None else max(1, int(max_workers))
    if workers > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            per_group = list(ex.map(_one, groups))  # deterministic order
    else:
        per_group = [_one(g) for g in groups]

    all_lags = [r[0] for r in per_group]
    all_gammas = [r[1] for r in per_group]
    all_counts = [r[2] for r in per_group]

    # Average across time groups (only bins that actually have pairs)
    stacked_gamma = np.vstack(all_gammas)
    stacked_lag = np.vstack(all_lags)
    stacked_counts = np.vstack(all_counts)
    have_pairs = stacked_counts > 0
    n_have = have_pairs.sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(
            n_have > 0,
            np.nansum(np.where(have_pairs, stacked_gamma, 0.0), axis=0)
            / np.maximum(n_have, 1),
            0.0,
        )
        lag_c = np.where(
            n_have > 0,
            np.nansum(np.where(have_pairs, stacked_lag, 0.0), axis=0)
            / np.maximum(n_have, 1),
            np.nanmean(stacked_lag, axis=0),
        )
    counts = np.sum(stacked_counts, axis=0)

    if not np.any(counts > 0):
        logger.warning(_msg("variogram_no_pairs", lang))
        return _empty_result()

    # Fit model
    params, r2 = _fit_variogram_model(lag_c, gamma, counts, model)
    fn = _MODELS[model]
    model_vals = fn(lag_c, params["nugget"], params["sill"], params["range"])

    result_df = pl.DataFrame({
        "lag_center": lag_c,
        "semivariance": gamma,
        "n_pairs": counts,
        "model_fitted": model_vals,
    })

    # Plot
    fig = None
    if HAS_PLOTLY:
        fig = _build_variogram_plot(
            lag_c, gamma, counts, params, r2, lang, style=style, units=units,
        )
        fig = finalize_export(
            fig, style=style, isave=isave, save_extension=save_extension,
            filename="lcz_variogram", lang=lang,
        )

    logger.info(
        _msg("variogram_complete", lang,
                model=_MODEL_LABELS.get(model, model), r2=f"{r2:.3f}",
                nugget=f"{params['nugget']:.2f}",
                sill=f"{params['sill']:.2f}",
                range=f"{params['range']:.0f}")
    )

    return VariogramResult(df=result_df, params=params, r2=r2, plot=fig)


__all__ = [
    "lcz_variogram", "VariogramResult", "HAS_PLOTLY", "DEFAULT_MAX_WORKERS",
]
