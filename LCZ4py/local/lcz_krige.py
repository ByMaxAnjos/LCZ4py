"""
LCZ-aware kriging wrapper for spatial interpolation.

Accepts Polars DataFrames, PyArrow Tables, or Pandas DataFrames.
Optionally incorporates LCZ class membership as an external drift term
in Universal Kriging (External Drift Kriging).

``pykrige`` is an **optional** dependency: when it is not installed (or its
variogram fit / solver fails at runtime), kriging transparently falls back to
a pure-NumPy ordinary-kriging solver (:func:`_numpy_ok_points`) that fits the
same Spherical/Exponential/Gaussian/Stein models.  Importing this module never
fails because of a missing optional dependency.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
from typing import Literal, Optional

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# ── Optional dependencies ─────────────────────────────────────────────────────
try:  # pragma: no cover - depends on environment
    from numba import njit

    HAS_NUMBA = True
except Exception:  # ImportError, or a broken numba install
    HAS_NUMBA = False

try:  # pragma: no cover - depends on environment
    from pykrige.ok import OrdinaryKriging
    from pykrige.uk import UniversalKriging  # noqa: F401  (public re-export)

    HAS_PYKRIGE = True
except Exception:  # ImportError, or an incomplete/broken pykrige install
    OrdinaryKriging = None  # type: ignore[assignment]
    UniversalKriging = None  # type: ignore[assignment]
    HAS_PYKRIGE = False

# Backward-compatible private alias (imported by lcz_interp_map / eval /
# lcz_anomaly_map and possibly by user code).
_HAS_PYK = HAS_PYKRIGE

VgModel = Literal["Sph", "Exp", "Gau", "Ste"]
_PYK_MODEL = {"Sph": "spherical", "Exp": "exponential", "Gau": "gaussian", "Ste": "stein"}

DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 4)

# Minimum number of distinct stations for a meaningful variogram/kriging fit.
_MIN_KRIGE_POINTS = 3


@dataclass
class KrigeResult:
    """Output of :func:`krige_predict`."""

    prediction: np.ndarray   # (ny, nx) float32 grid
    variance: np.ndarray     # (ny, nx) float32 kriging variance
    transform: tuple         # (res_x, 0, xmin, 0, res_y, ymax) affine-like
    crs: str


def _as_float_array(values) -> np.ndarray:
    """Convert (possibly masked) pykrige output to a float array with NaN holes."""
    arr = np.ma.asarray(values, dtype=np.float64)
    return np.ma.filled(arr, np.nan)


def _resolve_workers(max_workers: Optional[int]) -> int:
    """Normalise a user-supplied worker count to a bounded positive int."""
    if max_workers is None:
        return DEFAULT_MAX_WORKERS
    try:
        n = int(max_workers)
    except (TypeError, ValueError):
        return DEFAULT_MAX_WORKERS
    if n < 1:
        return DEFAULT_MAX_WORKERS
    return max(1, min(n, 32))


def _fast_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


# Drift-grid assembly: Numba if available, else vectorised NumPy
if HAS_NUMBA:  # pragma: no cover - numba is optional
    @njit(fastmath=True)
    def _build_drift_grid(tree_indices: np.ndarray, lcz_arr: np.ndarray,
                          means: np.ndarray, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            idx = tree_indices[i]
            lcz_val = lcz_arr[idx]
            found = False
            for j in range(means.shape[0]):
                if means[j, 0] == lcz_val:
                    out[i] = means[j, 1]
                    found = True
                    break
            if not found:
                out[i] = 0.0
        return out
else:
    def _build_drift_grid(tree_indices: np.ndarray, lcz_arr: np.ndarray,
                          means: np.ndarray, n: int) -> np.ndarray:
        lcz_mean_map = {int(means[j, 0]): means[j, 1] for j in range(len(means))}
        return np.array([lcz_mean_map.get(int(lcz_arr[i]), 0.0) for i in tree_indices])


# ── DataFrame conversion helpers ──────────────────────────────────────────────

def _to_pandas(df):
    """Convert Polars / Arrow / dict-like input to a pandas DataFrame."""
    try:
        import pyarrow as pa

        if isinstance(df, pa.Table):
            return df.to_pandas()
    except Exception:  # pyarrow missing or conversion failure
        pass
    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
    except Exception:
        pass
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    import pandas as pd

    return pd.DataFrame(df)


def _xy_z_arrays(df, x_col: str, y_col: str, z_col: str):
    """Extract finite, de-duplicated coordinate/value arrays from ``df``.

    Coincident stations are averaged so the kriging matrix never becomes
    singular simply because two rows share one coordinate pair.
    """
    d = df.dropna(subset=[x_col, y_col, z_col]) if hasattr(df, "dropna") else df
    x = np.asarray(d[x_col], dtype=np.float64).ravel()
    y = np.asarray(d[y_col], dtype=np.float64).ravel()
    z = np.asarray(d[z_col], dtype=np.float64).ravel()

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not ok.all():
        x, y, z = x[ok], y[ok], z[ok]
    if x.size == 0:
        return x, y, z, np.zeros(0, dtype=bool)

    # De-duplicate identical coordinates (average the observed values).
    coords = np.column_stack([x, y])
    uniq, inverse = np.unique(coords, axis=0, return_inverse=True)
    if uniq.shape[0] != coords.shape[0]:
        sums = np.bincount(inverse, weights=z, minlength=uniq.shape[0])
        counts = np.bincount(inverse, minlength=uniq.shape[0])
        logger.warning(
            "Averaged %d duplicate station coordinate(s) before kriging.",
            coords.shape[0] - uniq.shape[0],
        )
        x, y = uniq[:, 0], uniq[:, 1]
        z = sums / np.maximum(counts, 1)
        # Map each original row onto its unique-coordinate index, for drift.
        return x, y, z, inverse

    return x, y, z, np.arange(coords.shape[0])


# ── Variogram models (shared by the NumPy fallback solver) ────────────────────

def _model_spherical(h: np.ndarray, c0: float, c1: float, a: float) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    a = max(float(a), 1e-12)
    out = np.full(h.shape, c0 + c1, dtype=np.float64)
    mask = h <= a
    r = h[mask] / a
    out[mask] = c0 + c1 * (1.5 * r - 0.5 * r ** 3)
    return out


def _model_exponential(h: np.ndarray, c0: float, c1: float, a: float) -> np.ndarray:
    a = max(float(a), 1e-12)
    return c0 + c1 * (1.0 - np.exp(-3.0 * np.asarray(h, dtype=np.float64) / a))


def _model_gaussian(h: np.ndarray, c0: float, c1: float, a: float) -> np.ndarray:
    a = max(float(a), 1e-12)
    return c0 + c1 * (1.0 - np.exp(-3.0 * (np.asarray(h, dtype=np.float64) ** 2) / a ** 2))


def _model_stein(h: np.ndarray, c0: float, c1: float, a: float) -> np.ndarray:
    a = max(float(a), 1e-12)
    return c0 + c1 * (1.0 - np.exp(-((3.0 * np.asarray(h, dtype=np.float64) / a) ** 2)))


_MODEL_FUNCS = {
    "Sph": _model_spherical,
    "Exp": _model_exponential,
    "Gau": _model_gaussian,
    "Ste": _model_stein,
}


def _fit_variogram_params(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    model: str = "Sph",
    nlags: int = 20,
) -> dict:
    """Least-squares fit of a theoretical variogram to station data.

    Used by the NumPy fallback solver.  Falls back to robust moment
    estimates when the non-linear fit does not converge.

    Returns
    -------
    dict
        ``{"nugget", "sill", "range", "model_type"}`` — ``sill`` is the
        *partial* sill (``c1``), so total sill = ``nugget + sill``.
    """
    from scipy.optimize import curve_fit
    from scipy.spatial.distance import pdist

    fn = _MODEL_FUNCS.get(model, _model_spherical)
    n = len(x)

    # Subsample very large station sets — the variogram shape is insensitive.
    if n > 1500:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=1500, replace=False))
        x, y, z = x[idx], y[idx], z[idx]
        n = 1500

    pts = np.column_stack([x, y])
    dists = pdist(pts)
    if dists.size == 0:
        return {"nugget": 0.0, "sill": float(np.var(z)) or 1.0,
                "range": 1.0, "model_type": model}
    semi = 0.5 * pdist(z.reshape(-1, 1), metric="sqeuclidean")

    max_dist = float(np.percentile(dists, 90))
    if not np.isfinite(max_dist) or max_dist <= 0:
        max_dist = float(dists.max())
    if not np.isfinite(max_dist) or max_dist <= 0:
        return {"nugget": 0.0, "sill": float(np.var(z)) or 1.0,
                "range": 1.0, "model_type": model}

    n_lags = int(max(3, min(int(nlags), max(3, n - 1))))
    edges = np.linspace(0.0, max_dist, n_lags + 1)
    lag_c, gamma, counts = [], [], []
    for i in range(n_lags):
        mask = (dists >= edges[i]) & (dists < edges[i + 1])
        if mask.any():
            lag_c.append(float(dists[mask].mean()))
            gamma.append(float(semi[mask].mean()))
            counts.append(float(mask.sum()))
    if not lag_c:
        return {"nugget": 0.0, "sill": float(np.var(z)) or 1.0,
                "range": max_dist / 2.0 or 1.0, "model_type": model}

    lag_c = np.asarray(lag_c)
    gamma = np.asarray(gamma)
    weights = np.sqrt(np.asarray(counts))

    c0_0 = float(np.min(gamma))
    c1_0 = max(float(np.max(gamma) - c0_0), 1e-6)
    a_0 = float(np.max(lag_c)) / 2.0 or 1.0

    params = None
    try:
        popt, _ = curve_fit(
            fn, lag_c, gamma, p0=[c0_0, c1_0, a_0],
            bounds=([0.0, 0.0, 1e-9], [np.inf, np.inf, np.inf]),
            sigma=1.0 / np.maximum(weights, 1e-9), maxfev=20000,
        )
        if np.all(np.isfinite(popt)):
            params = (float(popt[0]), float(popt[1]), float(popt[2]))
    except Exception as exc:  # noqa: BLE001 - curve_fit raises several types
        logger.debug("Variogram fit did not converge (%s); using moment estimates.", exc)

    if params is None:
        params = (c0_0, c1_0, a_0)
        logger.warning(
            "Variogram fit did not converge for model '%s' — using moment "
            "estimates (nugget=%.3g, sill=%.3g, range=%.3g).",
            model, *params,
        )

    return {"nugget": params[0], "sill": params[1], "range": params[2],
            "model_type": model}


# ── Pure-NumPy ordinary kriging ───────────────────────────────────────────────

def _numpy_ok_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    model: str = "Sph",
    nlags: int = 20,
    n_neighbors: int = 16,
    max_workers: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging at arbitrary query points, pure NumPy/SciPy.

    Solves one local kriging system per query point using the
    ``n_neighbors`` nearest stations (batched ``np.linalg.solve``), so memory
    stays bounded even for full-grid prediction.

    Returns
    -------
    (prediction, variance) : tuple of float64 arrays
    """
    n = len(x)
    if n == 0:
        raise ValueError("No valid stations provided to the kriging solver.")
    if qx.size == 0:
        return np.zeros(0), np.zeros(0)

    params = _fit_variogram_params(x, y, z, model, nlags)
    fn = _MODEL_FUNCS.get(params["model_type"], _model_spherical)
    nugget = params["nugget"]
    total_sill = params["nugget"] + params["sill"]
    vrange = params["range"]

    k = int(max(1, min(n_neighbors, n)))
    tree = cKDTree(np.column_stack([x, y]))

    def _gamma(h):
        return fn(h, nugget, params["sill"], vrange)

    # A nugget of exactly zero makes the matrix singular when a query point
    # coincides with a station; nudge it and keep the prediction exact via λ.
    noise = max(nugget, 1e-10 * max(total_sill, 1.0))

    qpts = np.column_stack([qx, qy])
    pred = np.empty(len(qpts), dtype=np.float64)
    var = np.empty(len(qpts), dtype=np.float64)

    chunk = 20_000
    slices = [(i, min(i + chunk, len(qpts))) for i in range(0, len(qpts), chunk)]

    def _solve_chunk(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Batched kriging solve with a per-point least-squares fallback."""
        try:
            return np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            logger.debug("Kriging system singular for a chunk; using lstsq.")
        out = np.empty(b.shape, dtype=np.float64)
        for i in range(A.shape[0]):
            out[i] = np.linalg.lstsq(A[i], b[i], rcond=None)[0]
        return out

    def _solve_slice(bounds):
        lo, hi = bounds
        pts = qpts[lo:hi]
        m = len(pts)
        dists, idx = tree.query(pts, k=k)
        if k == 1:
            dists = dists[:, None]
            idx = idx[:, None]
        neigh = np.column_stack([x[idx.ravel()], y[idx.ravel()]]).reshape(m, k, 2)
        d_nn = dists  # (m, k)

        # Semivariance between query point and neighbours.
        g0 = _gamma(d_nn)
        # Pairwise semivariance among neighbours.
        diff = neigh[:, :, None, :] - neigh[:, None, :, :]
        d_pair = np.sqrt(np.sum(diff ** 2, axis=-1))
        gamma_mat = _gamma(d_pair)

        A = np.empty((m, k + 1, k + 1), dtype=np.float64)
        A[:, :k, :k] = gamma_mat
        A[:, :k, k] = 1.0
        A[:, k, :k] = 1.0
        A[:, k, k] = 0.0

        # Lagrange multiplier stabilisation for coincident points.
        diag = np.arange(k)
        A[:, diag, diag] = gamma_mat[:, diag, diag] + noise

        b = np.empty((m, k + 1), dtype=np.float64)
        b[:, :k] = g0
        b[:, k] = 1.0

        lam = _solve_chunk(A, b)

        weights = lam[:, :k]
        mu = lam[:, k]
        weights = np.where(np.isfinite(weights), weights, 0.0)
        zs = z[idx]  # (m, k)
        p = np.sum(weights * zs, axis=1)
        v = np.sum(weights * g0, axis=1) + np.where(np.isfinite(mu), mu, 0.0)
        # Exact reproduction at coincident locations.
        exact = d_nn[:, 0] <= 1e-9
        if exact.any():
            p[exact] = zs[exact, 0]
            v[exact] = 0.0
        return lo, hi, p, np.maximum(v, 0.0)

    workers = _resolve_workers(max_workers)
    if workers > 1 and len(slices) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for lo, hi, p, v in ex.map(_solve_slice, slices):
                pred[lo:hi] = p
                var[lo:hi] = v
    else:
        for bounds in slices:
            lo, hi, p, v = _solve_slice(bounds)
            pred[lo:hi] = p
            var[lo:hi] = v

    return pred, var


def _grid_mesh(grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = np.meshgrid(np.asarray(grid_x, dtype=np.float64),
                         np.asarray(grid_y, dtype=np.float64))
    return gx.ravel(), gy.ravel()


def _validate_grid(grid_x, grid_y, caller: str) -> tuple[np.ndarray, np.ndarray]:
    gx = np.asarray(grid_x, dtype=np.float64).ravel()
    gy = np.asarray(grid_y, dtype=np.float64).ravel()
    if gx.size == 0 or gy.size == 0:
        raise ValueError(f"{caller}: grid_x and grid_y must be non-empty 1-D arrays.")
    if not (np.all(np.isfinite(gx)) and np.all(np.isfinite(gy))):
        raise ValueError(f"{caller}: grid coordinates must be finite.")
    return gx, gy


def _affine_from_grid(grid_x: np.ndarray, grid_y: np.ndarray) -> tuple:
    """North-up affine-ish tuple ``(res_x, 0, xmin, 0, -res_y, ymax)``."""
    gx, gy = np.asarray(grid_x, np.float64), np.asarray(grid_y, np.float64)
    res_x = float(np.abs(np.diff(gx)).mean()) if gx.size > 1 else 1.0
    res_y = float(np.abs(np.diff(gy)).mean()) if gy.size > 1 else 1.0
    xmin = float(gx.min()) - res_x / 2.0
    ymax = float(gy.max()) + res_y / 2.0
    return (res_x, 0.0, xmin, 0.0, -res_y, ymax)


def _lcz_drift_setup(x, y, z, lcz):
    """Fit the LCZ external-drift OLS term; returns (drift_means, residual)."""
    unique = np.unique(lcz)
    if unique.size < 2:
        return None, z
    dummies = (lcz[:, None] == unique[None, :]).astype(np.float64)
    X = np.column_stack([np.ones_like(z), dummies])
    beta = _fast_lstsq(X, z)
    residual = z - X @ beta
    means = {float(c): float(z[lcz == c].mean()) for c in unique}
    return means, residual


def _drift_at_points(means: dict, qx, qy, x, y, lcz, lcz_query=None,
                     fallback: Optional[float] = None) -> np.ndarray:
    """Drift value at each query point (raster class if given, else nearest station).

    LCZ classes that have no station of their own fall back to ``fallback``
    (the overall station mean) rather than 0 — a class with no observations
    must not be predicted as "zero degrees".
    """
    known = np.array(sorted(means), dtype=np.float64)
    if fallback is None:
        fallback = float(np.mean(list(means.values()))) if means else 0.0

    if lcz_query is not None:
        arr = np.asarray(lcz_query, dtype=np.float64).ravel()
        out = np.full(arr.shape, float(fallback), dtype=np.float64)
        for c, m in means.items():
            out[arr == c] = m
        return out

    tree = cKDTree(np.column_stack([x, y]))
    _, idx = tree.query(np.column_stack([qx, qy]), k=1)
    out = _build_drift_grid(
        idx, lcz,
        np.array([[c, m] for c, m in means.items()], dtype=np.float64),
        len(idx),
    )
    unknown = ~np.isin(lcz[idx], known)
    out[unknown] = float(fallback)
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def krige_predict(
    df,
    x_col: str,
    y_col: str,
    z_col: str,
    lcz_col: Optional[str],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    model: VgModel = "Sph",
    enable_lcz_drift: bool = True,
    nlags: int = 20,
    lcz_grid: Optional[np.ndarray] = None,
    n_neighbors: int = 16,
    crs: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> KrigeResult:
    """Predict a variable on a regular grid from point observations.

    Parameters
    ----------
    df : pl.DataFrame, pa.Table, or pd.DataFrame
        Point observations.
    x_col, y_col, z_col : str
        Column names for projected easting, northing, and the target variable.
    lcz_col : str or None
        LCZ class column (integer). ``None`` disables the drift term.
    grid_x, grid_y : np.ndarray
        1-D UTM grid coordinates (easting, northing).
    model : VgModel
        Variogram model — ``"Sph"``, ``"Exp"``, ``"Gau"``, or ``"Ste"``.
    enable_lcz_drift : bool
        If True and ``lcz_col`` is provided, use LCZ membership as external drift.
    nlags : int
        Number of variogram lags.
    lcz_grid : np.ndarray or None
        LCZ class values resampled to the prediction grid, shape ``(ny, nx)``.
        When provided, drift at grid points is read directly from the raster
        instead of approximated via KD-tree from station locations.
    max_workers : int, optional
        Thread workers used by the pure-NumPy fallback solver
        (default ``min(8, cpu_count)``).
    n_neighbors : int
        Neighbours per local kriging system in the NumPy fallback.
    crs : str, optional
        CRS label to record on the result.  Defaults to ``"EPSG:4326"`` for
        backward compatibility — pass the real CRS (e.g. the UTM EPSG code)
        when the grid is projected.

    Returns
    -------
    KrigeResult
    """
    if model not in _PYK_MODEL:
        raise ValueError(f"Unknown variogram model '{model}'. Choose from {list(_PYK_MODEL)}.")
    gx_1d, gy_1d = _validate_grid(grid_x, grid_y, "krige_predict")
    for name, axis in (("grid_x", gx_1d), ("grid_y", gy_1d)):
        if axis.size > 2:
            steps = np.diff(axis)
            if not np.allclose(steps, steps[0], rtol=1e-3, atol=0):
                logger.debug("krige_predict: %s is not evenly spaced.", name)

    pdf = _to_pandas(df)
    for col in (x_col, y_col, z_col):
        if col not in pdf.columns:
            raise KeyError(f"krige_predict: column '{col}' not found in input data.")
    x, y, z, _ = _xy_z_arrays(pdf, x_col, y_col, z_col)
    n = len(x)
    if n == 0:
        raise ValueError("krige_predict: no finite observations after dropping NaN.")

    lcz = None
    if lcz_col and lcz_col in pdf.columns and enable_lcz_drift:
        # Align with the de-duplicated station arrays via nearest-station join.
        lcz_map = (
            pdf[[x_col, y_col, lcz_col]]
            .dropna()
            .astype({x_col: float, y_col: float, lcz_col: float})
        )
        tree = cKDTree(lcz_map[[x_col, y_col]].to_numpy(dtype=np.float64))
        _, idx = tree.query(np.column_stack([x, y]), k=1)
        lcz = lcz_map[lcz_col].to_numpy(dtype=np.float64)[idx]
        if np.unique(lcz).size < 2:
            lcz = None

    if n < _MIN_KRIGE_POINTS:
        logger.warning(
            "Only %d distinct station(s) — kriging needs at least %d; "
            "using Inverse Distance Weighting instead.", n, _MIN_KRIGE_POINTS,
        )
        pred = idw_predict(x, y, z, gx_1d, gy_1d)
        return KrigeResult(
            prediction=pred.astype(np.float32),
            variance=np.full(pred.shape, np.nan, dtype=np.float32),
            transform=_affine_from_grid(gx_1d, gy_1d),
            crs=crs or "EPSG:4326",
        )

    # ── LCZ external drift: krige the OLS residual, then add the drift back ──
    drift_means = None
    z_krige = z
    if lcz is not None:
        drift_means, z_krige = _lcz_drift_setup(x, y, z, lcz)
        if drift_means is None:
            logger.warning(
                "LCZ drift requested but only one class present at stations; "
                "falling back to Ordinary Kriging."
            )

    pred = var = None
    if HAS_PYKRIGE:
        try:
            ok = OrdinaryKriging(
                x, y, z_krige,
                variogram_model=_PYK_MODEL[model],
                nlags=int(max(1, nlags)),
                exact_values=False,
            )
            z_pred, var_pred = ok.execute("grid", gx_1d, gy_1d)
            pred = _as_float_array(z_pred)
            var = _as_float_array(var_pred)
        except Exception as exc:  # noqa: BLE001 - pykrige raises many types
            logger.warning(
                "pykrige kriging failed (%s: %s) — falling back to the NumPy "
                "ordinary-kriging solver.", type(exc).__name__, exc,
            )
            pred = var = None

    if pred is None:
        qx, qy = _grid_mesh(gx_1d, gy_1d)
        pred_flat, var_flat = _numpy_ok_points(
            x, y, z_krige, qx, qy, model=model, nlags=nlags,
            n_neighbors=n_neighbors, max_workers=max_workers,
        )
        ny, nx = len(gy_1d), len(gx_1d)
        pred = pred_flat.reshape(ny, nx)
        var = var_flat.reshape(ny, nx)

    if drift_means is not None:
        if lcz_grid is not None:
            lg = np.asarray(lcz_grid).ravel()
            if lg.size != pred.size:
                raise ValueError(
                    "krige_predict: lcz_grid has %d cells but the prediction "
                    "grid has %d." % (lg.size, pred.size)
                )
            fallback = float(z.mean())
            drift_grid = np.full(lg.shape, fallback, dtype=np.float64)
            for c, m in drift_means.items():
                drift_grid[lg == int(c)] = m
            drift_grid = drift_grid.reshape(pred.shape)
        else:
            qx, qy = _grid_mesh(gx_1d, gy_1d)
            drift_grid = _drift_at_points(
                drift_means, qx, qy, x, y, lcz,
                fallback=float(z.mean()),
            ).reshape(pred.shape)
        pred = pred + drift_grid

    var = np.where(np.isfinite(var), np.maximum(var, 0.0), np.nan)

    return KrigeResult(
        prediction=np.asarray(pred, dtype=np.float32),
        variance=np.asarray(var, dtype=np.float32),
        transform=_affine_from_grid(gx_1d, gy_1d),
        crs=crs or "EPSG:4326",
    )


def krige_predict_points(
    df,
    x_col: str,
    y_col: str,
    z_col: str,
    lcz_col: Optional[str],
    points_x: np.ndarray,
    points_y: np.ndarray,
    model: VgModel = "Sph",
    enable_lcz_drift: bool = True,
    nlags: int = 20,
    lcz_points: Optional[np.ndarray] = None,
    n_neighbors: int = 16,
    max_workers: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Kriging prediction at arbitrary (non-grid) points — e.g. for LOOCV.

    Same variogram/back-end machinery as :func:`krige_predict`, but no full
    grid is materialised: cost scales with the number of query points.

    Returns
    -------
    (prediction, variance) : tuple of 1-D float64 arrays
    """
    px = np.asarray(points_x, dtype=np.float64).ravel()
    py = np.asarray(points_y, dtype=np.float64).ravel()
    if px.size != py.size:
        raise ValueError("points_x and points_y must have the same length.")
    if px.size == 0:
        return np.zeros(0), np.zeros(0)
    if not (np.all(np.isfinite(px)) and np.all(np.isfinite(py))):
        raise ValueError("Query point coordinates must be finite.")

    pdf = _to_pandas(df)
    x, y, z, _ = _xy_z_arrays(pdf, x_col, y_col, z_col)
    if len(x) == 0:
        raise ValueError("krige_predict_points: no finite observations.")

    lcz = None
    if lcz_col and lcz_col in pdf.columns and enable_lcz_drift:
        lcz_map = (
            pdf[[x_col, y_col, lcz_col]].dropna()
            .astype({x_col: float, y_col: float, lcz_col: float})
        )
        tree = cKDTree(lcz_map[[x_col, y_col]].to_numpy(dtype=np.float64))
        _, idx = tree.query(np.column_stack([x, y]), k=1)
        lcz = lcz_map[lcz_col].to_numpy(dtype=np.float64)[idx]
        if np.unique(lcz).size < 2:
            lcz = None

    if len(x) < _MIN_KRIGE_POINTS:
        pred = _idw_points(x, y, z, px, py, power=2.0, k=-1)
        return pred, np.full(pred.shape, np.nan)

    drift_means, z_krige = (None, z)
    if lcz is not None:
        drift_means, z_krige = _lcz_drift_setup(x, y, z, lcz)

    pred = var = None
    if HAS_PYKRIGE:
        try:
            ok = OrdinaryKriging(
                x, y, z_krige, variogram_model=_PYK_MODEL[model],
                nlags=int(max(1, nlags)), exact_values=False,
            )
            z_pred, var_pred = ok.execute("points", px, py)
            pred = _as_float_array(z_pred).ravel()
            var = _as_float_array(var_pred).ravel()
        except Exception as exc:  # noqa: BLE001
            logger.debug("pykrige point prediction failed (%s); using NumPy.", exc)
            pred = var = None

    if pred is None:
        pred, var = _numpy_ok_points(
            x, y, z_krige, px, py, model=model, nlags=nlags,
            n_neighbors=n_neighbors, max_workers=max_workers,
        )

    if drift_means is not None:
        pred = pred + _drift_at_points(
            drift_means, px, py, x, y, lcz, lcz_query=lcz_points,
        )

    return pred, np.where(np.isfinite(var), np.maximum(var, 0.0), np.nan)


# ── Scatter-to-grid estimators (always available, no optional deps) ───────────

def _clean_points(x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y and z must have the same length.")
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if x.size == 0:
        raise ValueError("No finite observation points provided.")

    # Average coincident stations — a duplicate coordinate makes RBF systems
    # singular and gives two stations double weight under IDW.
    coords = np.column_stack([x, y])
    uniq, inverse = np.unique(coords, axis=0, return_inverse=True)
    if uniq.shape[0] != coords.shape[0]:
        sums = np.bincount(inverse, weights=z, minlength=uniq.shape[0])
        counts = np.bincount(inverse, minlength=uniq.shape[0])
        logger.debug("Averaged %d coincident station(s).",
                     coords.shape[0] - uniq.shape[0])
        x, y = uniq[:, 0], uniq[:, 1]
        z = sums / np.maximum(counts, 1)
    return x, y, z


def _idw_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    power: float = 2.0,
    k: int = -1,
) -> np.ndarray:
    """Inverse Distance Weighting evaluated at flat query points."""
    x, y, z = _clean_points(x, y, z)
    qx = np.asarray(qx, dtype=np.float64).ravel()
    qy = np.asarray(qy, dtype=np.float64).ravel()
    if qx.size != qy.size:
        raise ValueError("Query coordinates must have the same length.")
    if qx.size == 0:
        return np.zeros(0, dtype=np.float64)

    power = float(power) if power and power > 0 else 2.0
    k_use = len(x) if (k is None or k < 1) else int(max(1, min(k, len(x))))
    tree = cKDTree(np.column_stack([x, y]))

    out = np.empty(qx.size, dtype=np.float64)
    chunk = 200_000
    for lo in range(0, qx.size, chunk):
        hi = min(lo + chunk, qx.size)
        pts = np.column_stack([qx[lo:hi], qy[lo:hi]])
        dists, idx = tree.query(pts, k=k_use)
        if k_use == 1:
            dists = dists[:, None]
            idx = idx[:, None]
        vals = z[idx]
        # Coincident query/station → return that observation exactly.
        exact = dists[:, 0] <= 1e-12
        safe = np.where(dists <= 0, 1e-12, dists)
        w = 1.0 / safe ** power
        w_sum = w.sum(axis=1, keepdims=True)
        pred = np.sum(w * vals, axis=1) / w_sum[:, 0]
        pred = np.where(exact, vals[:, 0], pred)
        out[lo:hi] = pred
    return out


def idw_predict(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    grid_x: np.ndarray, grid_y: np.ndarray,
    power: float = 2.0,
    k: int = -1,
) -> np.ndarray:
    """Scatter-to-grid via Inverse Distance Weighting.

    ``power=2`` is standard; higher values give sharper gradients near stations.
    ``k=-1`` uses all stations; set ``k=8`` or similar for local weighting.
    Coincident stations are handled exactly (no division by zero) and
    non-finite observations are dropped.
    """
    x, y, z = _clean_points(x, y, z)
    gx_1d, gy_1d = _validate_grid(grid_x, grid_y, "idw_predict")
    qx, qy = _grid_mesh(gx_1d, gy_1d)
    pred = _idw_points(x, y, z, qx, qy, power=power, k=k)
    return pred.reshape(len(gy_1d), len(gx_1d)).astype(np.float32)


def rbf_predict(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    grid_x: np.ndarray, grid_y: np.ndarray,
    kernel: str = "thin_plate_spline",
    smoothing: float = 0.0,
) -> np.ndarray:
    """Scatter-to-grid via Radial Basis Function (scipy, no extra dependencies).

    ``kernel="thin_plate_spline"`` is smooth, rotation-invariant, and well-suited
    for meteorological station data.  Set ``smoothing>0`` for noisy observations.

    The solver is condition-number-safe (coordinates are centred/scaled
    internally).  When the system is singular — too few or collinear stations,
    or a kernel whose polynomial degree the data cannot support — the estimate
    degrades gracefully to Inverse Distance Weighting instead of raising.
    """
    from scipy.interpolate import RBFInterpolator

    x, y, z = _clean_points(x, y, z)
    gx_1d, gy_1d = _validate_grid(grid_x, grid_y, "rbf_predict")
    ny, nx = len(gy_1d), len(gx_1d)
    qx, qy = _grid_mesh(gx_1d, gy_1d)

    def _fallback(reason: str) -> np.ndarray:
        logger.warning("RBF interpolation unavailable (%s) — using IDW instead.", reason)
        return _idw_points(x, y, z, qx, qy).reshape(ny, nx).astype(np.float32)

    if len(x) < 3:
        return _fallback("fewer than 3 valid stations")

    # Centre and scale the coordinates — RBF systems are very sensitive to
    # large UTM magnitudes.
    cx, cy = float(x.mean()), float(y.mean())
    scale = float(max(np.std(x), np.std(y), 1e-9))
    pts = np.column_stack([(x - cx) / scale, (y - cy) / scale])
    qpts = np.column_stack([(qx - cx) / scale, (qy - cy) / scale])

    try:
        rbf = RBFInterpolator(
            pts, z, kernel=kernel, smoothing=float(max(smoothing, 0.0)),
        )
        pred = np.asarray(rbf(qpts), dtype=np.float64)
        if not np.all(np.isfinite(pred)):
            raise FloatingPointError("non-finite RBF output")
    except Exception as exc:  # noqa: BLE001 - LinAlgError, ValueError, ...
        return _fallback(f"{type(exc).__name__}: {exc}")

    return pred.reshape(ny, nx).astype(np.float32)


__all__ = [
    "krige_predict",
    "krige_predict_points",
    "KrigeResult",
    "VgModel",
    "rbf_predict",
    "idw_predict",
    "HAS_PYKRIGE",
    "HAS_NUMBA",
    "DEFAULT_MAX_WORKERS",
]
