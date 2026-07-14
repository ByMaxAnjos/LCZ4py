"""
lcz_variogram.py — empirical semivariogram computation and theoretical model fitting.

Computes pairwise semivariances between stations, bins them into lag classes,
and fits Spherical, Exponential, or Gaussian theoretical variogram models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import polars as pl
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, extract_lcz_at_points, normalise_input_df,
    normalise_missing, select_by_date, utm_epsg_for,
)
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)


# ── Theoretical variogram models ──────────────────────────────────────────────

def _spherical(h: np.ndarray, c0: float, c1: float, a: float) -> np.ndarray:
    """Spherical model: γ(h) = c0 + c1 * (1.5h/a − 0.5(h/a)³) for h ≤ a, else c0+c1."""
    out = np.full_like(h, c0 + c1, dtype=np.float64)
    mask = h <= a
    r = h[mask] / a
    out[mask] = c0 + c1 * (1.5 * r - 0.5 * r ** 3)
    return out


def _exponential(h: np.ndarray, c0: float, c1: float, a: float) -> np.ndarray:
    """Exponential model: γ(h) = c0 + c1 * (1 − exp(−3h/a))."""
    return c0 + c1 * (1.0 - np.exp(-3.0 * h / a))


def _gaussian(h: np.ndarray, c0: float, c1: float, a: float) -> np.ndarray:
    """Gaussian model: γ(h) = c0 + c1 * (1 − exp(−3h²/a²))."""
    return c0 + c1 * (1.0 - np.exp(-3.0 * h ** 2 / a ** 2))


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
        Number of pairs per lag bin.
    """
    dists = pdist(np.column_stack([x, y]))
    diffs = pdist(z.reshape(-1, 1), metric="sqeuclidean")
    semivariances = 0.5 * diffs

    lag_edges = np.linspace(0, max_dist, n_lags + 1)
    lag_centers = np.empty(n_lags)
    mean_gamma = np.empty(n_lags)
    counts = np.empty(n_lags, dtype=int)

    for i in range(n_lags):
        lo, hi = lag_edges[i], lag_edges[i + 1]
        mask = (dists >= lo) & (dists < hi)
        counts[i] = mask.sum()
        if counts[i] > 0:
            lag_centers[i] = dists[mask].mean()
            mean_gamma[i] = semivariances[mask].mean()
        else:
            lag_centers[i] = (lo + hi) / 2
            mean_gamma[i] = 0.0

    return lag_centers, mean_gamma, counts.astype(float)


def _fit_variogram_model(
    lag_centers: np.ndarray,
    semivariances: np.ndarray,
    n_pairs: np.ndarray,
    model: str,
) -> tuple[dict, float]:
    """Fit a theoretical variogram model via least-squares.

    Returns
    -------
    params : dict
        ``{nugget, sill, range, model_type}``.
    r2 : float
        Coefficient of determination.
    """
    fn = _MODELS.get(model)
    if fn is None:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(_MODELS)}.")

    weights = np.sqrt(n_pairs)
    valid = weights > 0
    x_fit = lag_centers[valid]
    y_fit = semivariances[valid]
    w = weights[valid]

    # Initial parameter guesses
    c0_0 = float(y_fit.min()) if len(y_fit) else 0.0
    c1_0 = float(y_fit.max() - c0_0) if len(y_fit) else 1.0
    a_0 = float(x_fit.max()) if len(x_fit) else 1.0

    try:
        popt, _ = curve_fit(fn, x_fit, y_fit, p0=[c0_0, c1_0, a_0],
                            bounds=(0, np.inf), maxfev=10000, sigma=1.0 / w)
        c0, c1, a = popt
    except RuntimeError:
        c0, c1, a = c0_0, c1_0, a_0

    predicted = fn(x_fit, c0, c1, a)
    ss_res = np.sum((y_fit - predicted) ** 2)
    ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"nugget": c0, "sill": c1, "range": a, "model_type": model}, r2


def _build_variogram_plot(
    lag_centers: np.ndarray,
    semivariances: np.ndarray,
    n_pairs: np.ndarray,
    params: dict,
    r2: float,
    lang: str,
) -> "go.Figure":
    """Build a Plotly figure with empirical scatter + fitted model curve."""
    x_fit = np.linspace(0, float(lag_centers.max()) * 1.05, 200)
    fn = _MODELS[params["model_type"]]
    y_fit = fn(x_fit, params["nugget"], params["sill"], params["range"])

    model_label = _MODEL_LABELS.get(params["model_type"], params["model_type"])
    legend_text = (
        f"{model_label}: nugget={params['nugget']:.2f}, "
        f"sill={params['sill']:.2f}, range={params['range']:.0f} m, R²={r2:.3f}"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lag_centers.tolist(),
        y=semivariances.tolist(),
        mode="markers",
        name="Empirical",
        marker=dict(size=6, opacity=0.7),
        text=[f"Pairs: {int(n)}" for n in n_pairs],
        hovertemplate="Distance: %{x:.0f} m<br>Semivariance: %{y:.2f}<br>%{text}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_fit.tolist(),
        y=y_fit.tolist(),
        mode="lines",
        name=legend_text,
        line=dict(width=2, dash="dash"),
    ))
    fig.update_layout(
        title=lcz_msg("variogram_title", lang),
        xaxis_title=lcz_msg("variogram_xlabel", lang),
        yaxis_title=lcz_msg("variogram_ylabel", lang),
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )
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
    lang: str = "en",
    year=None,
    month=None,
    day=None,
    hour=None,
    start=None,
    end=None,
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
        Spatial resolution in metres (used for UTM extraction; not grid output).
    model : {"Sph", "Exp", "Gau"}
        Theoretical variogram model to fit.
    n_lags : int
        Number of distance lag bins.
    max_dist : float, optional
        Maximum pairwise distance for the variogram (metres). If None, uses
        half the maximum inter-station distance.
    lang : str
        Language for log messages — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.
    year, month, day, hour : optional
        Date/time component filters.
    start, end : optional
        Date range filters.

    Returns
    -------
    VariogramResult
        Dataclass with fitted variogram DataFrame, model parameters, R², and plot.
    """
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
        logger.warning(lcz_msg("variogram_no_data", lang))
        empty = pl.DataFrame({"lag_center": [], "semivariance": [], "n_pairs": [], "model_fitted": []})
        return VariogramResult(df=empty, params={}, r2=0.0, plot=None)

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

    df = df.join(
        stations.select(["latitude", "longitude", "lcz", "x", "y"]),
        on=["latitude", "longitude"], how="left",
    )
    df = df.filter(pl.col("lcz").is_between(1, 17))

    # Compute variogram per time group (daily aggregation)
    all_lags, all_gammas, all_counts = [], [], []

    for time_val, grp in df.group_by("date"):
        if len(grp) < 3:
            continue
        coords = grp.select(["x", "y"]).to_numpy()
        vals = grp["var_interp"].to_numpy()
        dists = pdist(coords)
        if len(dists) == 0:
            continue

        eff_max = max_dist if max_dist is not None else float(np.percentile(dists, 90))
        if eff_max <= 0:
            continue

        lag_c, gamma, n = _compute_empirical_variogram(
            coords[:, 0], coords[:, 1], vals, n_lags, eff_max
        )
        all_lags.append(lag_c)
        all_gammas.append(gamma)
        all_counts.append(n)

    if not all_lags:
        logger.warning(lcz_msg("variogram_no_pairs", lang))
        empty = pl.DataFrame({"lag_center": [], "semivariance": [], "n_pairs": [], "model_fitted": []})
        return VariogramResult(df=empty, params={}, r2=0.0, plot=None)

    # Average across time groups
    lag_c = np.mean(all_lags, axis=0)
    gamma = np.mean(all_gammas, axis=0)
    counts = np.sum(all_counts, axis=0)

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
        fig = _build_variogram_plot(lag_c, gamma, counts, params, r2, lang)

    logger.info(
        lcz_msg("variogram_complete", lang,
                model=_MODEL_LABELS.get(model, model), r2=f"{r2:.3f}",
                nugget=f"{params['nugget']:.2f}",
                sill=f"{params['sill']:.2f}",
                range=f"{params['range']:.0f}")
    )

    return VariogramResult(df=result_df, params=params, r2=r2, plot=fig)


__all__ = ["lcz_variogram", "VariogramResult"]
