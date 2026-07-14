"""
LCZ-aware kriging wrapper for spatial interpolation.

Accepts Polars DataFrames, PyArrow Tables, or Pandas DataFrames.
Optionally incorporates LCZ class membership as an external drift term
in Universal Kriging (External Drift Kriging).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    from pykrige.ok import OrdinaryKriging
    from pykrige.uk import UniversalKriging
    _HAS_PYK = True
except ImportError:
    _HAS_PYK = False

from scipy.spatial import cKDTree

VgModel = Literal["Sph", "Exp", "Gau", "Ste"]
_PYK_MODEL = {"Sph": "spherical", "Exp": "exponential", "Gau": "gaussian", "Ste": "stein"}


@dataclass
class KrigeResult:
    """Output of :func:`krige_predict`."""
    prediction: np.ndarray   # (ny, nx) float32 grid
    variance: np.ndarray     # (ny, nx) float32 kriging variance
    transform: tuple         # (res_x, 0, xmin, 0, res_y, ymax) affine-like
    crs: str


def _fast_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


# Drift-grid assembly: Numba if available, else vectorised NumPy
if HAS_NUMBA:
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

    Returns
    -------
    KrigeResult
    """
    if not _HAS_PYK:
        raise ImportError("pykrige is required. Install with: pip install pykrige")

    # Zero-copy conversion to Pandas (PyKrige requires it)
    try:
        import pyarrow as pa
        if isinstance(df, pa.Table):
            df = df.to_pandas()
    except ImportError:
        pass
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
    except ImportError:
        pass

    df = df.dropna(subset=[x_col, y_col, z_col])
    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    z = df[z_col].to_numpy(dtype=np.float64)

    if enable_lcz_drift and lcz_col and lcz_col in df.columns and len(np.unique(df[lcz_col])) >= 2:
        lcz = df[lcz_col].to_numpy(dtype=np.float64)
        unique_lczs = np.unique(lcz)
        drift_cols = (lcz[:, None] == unique_lczs[None, :]).astype(np.float64)
        X = np.column_stack([np.ones_like(z), drift_cols])
        beta = _fast_lstsq(X, z)
        residual = z - X @ beta

        ok_resid = OrdinaryKriging(
            x, y, residual,
            variogram_model=_PYK_MODEL[model],
            nlags=nlags,
            exact_values=False,
        )
        z_pred, var_pred = ok_resid.execute("grid", grid_x, grid_y)

        per_class_mean = {c: float(z[lcz == c].mean()) for c in unique_lczs}

        if lcz_grid is not None:
            # Use actual raster LCZ classes at each grid point
            drift_grid = np.zeros(lcz_grid.shape, dtype=np.float64)
            for c, m in per_class_mean.items():
                drift_grid[lcz_grid == int(c)] = m
        else:
            # ponytail: KD-tree fallback when lcz_grid not supplied
            means_arr = np.array([[c, m] for c, m in per_class_mean.items()])
            tree = cKDTree(np.column_stack([x, y]))
            gx, gy = np.meshgrid(grid_x, grid_y)
            flat_pts = np.column_stack([gx.ravel(), gy.ravel()])
            _, idx = tree.query(flat_pts, k=1)
            drift_grid = _build_drift_grid(idx, lcz, means_arr, len(idx)).reshape(gx.shape)

        z_pred = np.asarray(z_pred) + drift_grid
    else:
        ok = OrdinaryKriging(
            x, y, z,
            variogram_model=_PYK_MODEL[model],
            nlags=nlags,
            exact_values=False,
        )
        z_pred, var_pred = ok.execute("grid", grid_x, grid_y)

    return KrigeResult(
        prediction=np.asarray(z_pred, dtype=np.float32),
        variance=np.asarray(var_pred, dtype=np.float32),
        transform=(1.0, 0.0, float(grid_x[0]), 0.0, 1.0, float(grid_y[0])),
        crs="EPSG:4326",
    )


def rbf_predict(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    grid_x: np.ndarray, grid_y: np.ndarray,
    kernel: str = "thin_plate_spline",
    smoothing: float = 0.0,
) -> np.ndarray:
    """Scatter-to-grid via Radial Basis Function (scipy, no extra dependencies).

    ``kernel="thin_plate_spline"`` is smooth, rotation-invariant, and well-suited
    for meteorological station data.  Set ``smoothing>0`` for noisy observations.
    """
    from scipy.interpolate import RBFInterpolator
    rbf = RBFInterpolator(
        np.column_stack([x, y]), z,
        kernel=kernel, smoothing=smoothing,
    )
    gx, gy = np.meshgrid(grid_x, grid_y)
    z_pred = rbf(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    return z_pred.astype(np.float32)


def idw_predict(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    grid_x: np.ndarray, grid_y: np.ndarray,
    power: float = 2.0,
    k: int = -1,
) -> np.ndarray:
    """Scatter-to-grid via Inverse Distance Weighting.

    ``power=2`` is standard; higher values give sharper gradients near stations.
    ``k=-1`` uses all stations; set ``k=8`` or similar for local weighting.
    """
    gx, gy = np.meshgrid(grid_x, grid_y)
    flat = np.column_stack([gx.ravel(), gy.ravel()])
    k_use = len(x) if k < 1 else min(k, len(x))
    dists, idx = cKDTree(np.column_stack([x, y])).query(flat, k=k_use)
    if k_use == 1:
        dists, idx = dists[:, None], idx[:, None]
    w = 1.0 / np.where(dists == 0, 1e-10, dists) ** power
    w /= w.sum(axis=1, keepdims=True)
    return (w * z[idx]).sum(axis=1).reshape(gx.shape).astype(np.float32)


__all__ = ["krige_predict", "KrigeResult", "VgModel", "rbf_predict", "idw_predict"]
