"""
lcz_interp_map_plus.py — ML-based spatial interpolation of air temperature.

Uses Random Forest or Multiple Linear Regression with urban morphological
parameters (building fraction, height, land fraction, roughness, etc.) as
features to predict air temperature across the LCZ map grid. Designed for
urban areas with low meteorological station density where traditional
kriging may perform poorly.

Workflow:
    1. Download urban parameter rasters via UrbanParameterProcessor
    2. Extract parameter values at station locations as training features
    3. Train ML model (RF or OLS) on temperature ~ urban parameters
    4. Apply model pixel-by-pixel across the LCZ grid
    5. Output GeoTIFF interpolation map
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import numpy as np
import polars as pl
import rasterio

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, normalise_input_df, normalise_missing,
    select_by_date, extract_lcz_at_points, utm_epsg_for, OUTPUT_DIR,
    add_by_column, by_sorted_groups,
)
from LCZ4py.local.lcz_interp_map import _resample_lcz_to_grid
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 4)

# ── Optional dependency: scikit-learn ────────────────────────────────────────
# ``ml_model="rf"`` uses a Random Forest when scikit-learn is importable and
# degrades to a NumPy ridge/OLS + external-drift model otherwise.  Importing
# this module must never fail because scikit-learn is absent.
try:  # pragma: no cover - depends on environment
    import sklearn  # noqa: F401  - availability probe only
    from sklearn.ensemble import RandomForestRegressor  # noqa: F401
    from sklearn.model_selection import cross_val_score  # noqa: F401

    HAS_SKLEARN = True
except Exception:  # ImportError, or a broken/partial sklearn install
    HAS_SKLEARN = False

# Only warn once per process — LOOCV calls the trainer dozens of times.
_WARNED: set[str] = set()


def _warn_once(key: str, message: str, *args) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message, *args)

MLModel = Literal["rf", "ols"]


@dataclass
class LCZInterpResult:
    """Structured output from :func:`lcz_interp_map_plus`.

    Attributes
    ----------
    path : str
        Path to the output GeoTIFF.
    feature_importance : dict or None
        Per-group feature importances ``{group_label: {feature: importance}}``.
    cv_scores : dict or None
        Per-group cross-validation scores ``{group_label: {"r2_mean": ..., "r2_std": ...}}``.
    model_type : str
        ``"rf"`` or ``"ols"``.
    n_stations : int
        Number of unique training stations.
    n_features : int
        Number of features used.
    groups : list
        Group labels (time steps or by-groups) that were interpolated.
    uncertainty_path : str or None
        Path to uncertainty GeoTIFF (RF std or OLS residual SE).
    pca_explained_variance : list[float] or None
        Per-component explained variance ratios (when ``use_pca=True``).
    """
    path: str
    feature_importance: Optional[dict] = None
    cv_scores: Optional[dict] = None
    model_type: str = "rf"
    n_stations: int = 0
    n_features: int = 0
    groups: list = field(default_factory=list)
    uncertainty_path: Optional[str] = None
    pca_explained_variance: Optional[list] = None

# Default urban parameters — core morphological variables that drive
# urban temperature variation.  Directional roughness parameters are
# excluded by default to keep download time and feature count manageable.
_DEFAULT_UCP_VARS: list[str] = [
    "elevation", "frc_esa", "hgt", "lb", "lc", "lf", "lp",
    "urban_frc", "cglc", "tree", "urban",
]


def _extract_lcz_params_at_stations(
    lcz_classes: np.ndarray,
    use_means_only: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Extract LCZ morphological parameters from the lookup table at station locations.

    Parameters
    ----------
    lcz_classes : 1-d array of int
        LCZ class values (1–17) at each station.
    use_means_only : bool
        If True, return only the 12 mean parameters (SVFmean, ARmean, etc.)
        to keep feature count manageable.  If False, return all 34 parameters.

    Returns
    -------
    features : (n, n_params) array
    names : list[str]
    """
    from LCZ4py.general.lcz_get_parameters import _get_parameter_lookup
    lookup, all_names = _get_parameter_lookup()
    safe = np.clip(lcz_classes.astype(int), 0, 17)
    params = lookup[safe]  # (n, n_params)

    if use_means_only:
        # Mean columns are the last 12: svf_mean, aspect_mean, BSF_mean, ...
        mean_idx = [all_names.index(n) for n in [
            "svf_mean", "BSF_mean", "ISF_mean", "PSF_mean", "TSF_mean",
            "HRE_mean", "TRC_mean", "SAD_mean", "SAL_mean", "AH_mean", "z0",
        ]]
        mean_names = [all_names[i] for i in mean_idx]
        return params[:, mean_idx].astype(np.float32), mean_names
    else:
        return params[:, 1:].astype(np.float32), all_names[1:]  # skip 'lcz' column


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_northup_grid(
    ds: rasterio.io.DatasetReader,
    sp_res: float = 100.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> tuple[np.ndarray, np.ndarray, "Affine", str]:
    """Build a guaranteed north-up UTM grid from the LCZ raster bounds.

    Unlike ``_make_grid_memsafe`` which delegates to the original transform,
    this function builds a fresh ``Affine`` from the geographic bounds —
    ensuring the output is always north-up regardless of the source
    raster's rotation or shear.

    Parameters
    ----------
    ds : rasterio dataset
        Open LCZ raster.
    sp_res : float
        Output pixel size in metres.
    bbox : tuple, optional
        User-defined bounding box ``(minx, miny, maxx, maxy)`` in the
        **geographic CRS** of the LCZ raster (typically EPSG:4326).
        If ``None``, the full raster extent is used.
    """
    from rasterio.transform import from_origin
    from pyproj import Transformer

    if not np.isfinite(sp_res) or sp_res <= 0:
        raise ValueError(f"sp_res must be a positive resolution in metres, got {sp_res!r}.")
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox must be (minx, miny, maxx, maxy).")
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            raise ValueError(
                "bbox must satisfy minx < maxx and miny < maxy, got "
                f"{tuple(bbox)!r}."
            )

    epsg = utm_epsg_for(ds)
    tx = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)

    if bbox is not None:
        xmin, ymin = tx.transform(bbox[0], bbox[1])
        xmax, ymax = tx.transform(bbox[2], bbox[3])
    else:
        bounds = ds.bounds
        xmin, ymin = tx.transform(bounds.left, bounds.bottom)
        xmax, ymax = tx.transform(bounds.right, bounds.top)

    nx = max(1, int(np.ceil((xmax - xmin) / sp_res)))
    ny = max(1, int(np.ceil((ymax - ymin) / sp_res)))
    grid_x = xmin + sp_res * (np.arange(nx) + 0.5)
    grid_y = ymax - sp_res * (np.arange(ny) + 0.5)
    # from_origin: origin is top-left corner (x=west, y=north), positive step
    transform = from_origin(xmin, ymax, sp_res, sp_res)
    return grid_x, grid_y, transform, epsg


def _check_missing_rasters(
    rasters: dict[str, np.ndarray],
    min_valid_fraction: float = 0.5,
) -> list[str]:
    """Identify rasters that are mostly zeros or NaN (likely failed downloads).

    Returns a list of variable names that fail the threshold and should be
    excluded from model training.
    """
    bad = []
    for name, arr in rasters.items():
        total = np.asarray(arr).size
        if total == 0:
            logger.warning("Raster '%s': empty array — excluding.", name)
            bad.append(name)
            continue
        arr = np.asarray(arr)
        finite = np.isfinite(arr)
        if not finite.any():
            logger.warning("Raster '%s': no finite pixels — excluding.", name)
            bad.append(name)
            continue
        # Valid = finite and not the 0 fill value used by failed downloads.
        nonzero = np.count_nonzero(finite & (arr != 0))
        valid_frac = nonzero / total
        if valid_frac < min_valid_fraction:
            logger.warning(
                "Raster '%s': only %.1f%% valid pixels — excluding.",
                name, valid_frac * 100,
            )
            bad.append(name)
    return bad


class _PCAResult:
    """Minimal PCA transform (NumPy SVD), API-compatible with sklearn's."""

    def __init__(self, mean: np.ndarray, components: np.ndarray,
                 explained_variance_ratio_: np.ndarray):
        self.mean_ = mean
        self.components_ = components
        self.explained_variance_ratio_ = explained_variance_ratio_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T


def _apply_pca(
    X_train: np.ndarray,
    X_predict: np.ndarray,
    n_components: int = 0,
) -> tuple[np.ndarray, np.ndarray, Optional[list[float]], object]:
    """Apply PCA dimensionality reduction.

    If ``n_components == 0``, use all components (no reduction).
    Returns ``(X_train_pca, X_predict_pca, explained_variance, pca_obj)``.

    Uses scikit-learn when available and a deterministic NumPy SVD otherwise,
    so ``use_pca=True`` never requires an optional dependency.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_predict = np.asarray(X_predict, dtype=np.float64)
    n_samples, n_features = X_train.shape
    if n_components <= 0:
        n_components = min(n_samples, n_features)
    n_components = int(max(1, min(n_components, n_features)))
    if n_components > n_samples:
        logger.warning(
            "n_pca_components=%d exceeds the %d training samples — clamped.",
            n_components, n_samples,
        )
        n_components = max(1, n_samples)

    if HAS_SKLEARN:
        from sklearn.decomposition import PCA as _PCA

        pca = _PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_predict_pca = pca.transform(X_predict)
        evr = pca.explained_variance_ratio_.tolist()
    else:
        mean = X_train.mean(axis=0)
        Xc = X_train - mean
        # full_matrices=False keeps memory bounded for wide feature matrices
        _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        components = vt[:n_components]
        var = s ** 2
        total = float(var.sum())
        evr = (var[:n_components] / total).tolist() if total > 0 else []
        pca = _PCAResult(mean, components, np.asarray(evr))
        X_train_pca = pca.transform(X_train)
        X_predict_pca = pca.transform(X_predict)

    logger.info("PCA: %d → %d components (%.1f%% variance retained)",
                n_features, len(evr), sum(evr) * 100)
    return X_train_pca, X_predict_pca, evr, pca


def _compute_ols_vif(X: np.ndarray) -> np.ndarray:
    """Variance Inflation Factor for each feature (OLS diagnostics).

    VIF > 5 suggests moderate multicollinearity; VIF > 10 is severe.
    """
    n_feat = X.shape[1]
    vifs = np.zeros(n_feat, dtype=np.float64)
    for j in range(n_feat):
        y = X[:, j]
        X_other = np.delete(X, j, axis=1)
        X_with_int = np.column_stack([np.ones(len(X_other)), X_other])
        beta, residuals, _, _ = np.linalg.lstsq(X_with_int, y, rcond=None)
        y_hat = X_with_int @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[j] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    return vifs


def _extract_raster_values_at_points(
    rasters: dict[str, "np.ndarray"],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    points_x: np.ndarray,
    points_y: np.ndarray,
) -> dict[str, np.ndarray]:
    """Bilinear extraction of raster values at station coordinates.

    Parameters
    ----------
    rasters : dict
        Mapping ``{name: 2-d numpy array}`` where the array is indexed
        as ``[row, col]`` with ``row=0`` at the top (north).
    grid_x, grid_y : 1-d arrays
        Cell-centre coordinates.  ``grid_y`` is descending (north first).
    points_x, points_y : 1-d arrays
        Station easting / northing in the same CRS as the grid.

    Returns
    -------
    dict
        ``{name: 1-d array}`` with one value per station, NaN where the
        point falls outside the grid.
    """
    from scipy.ndimage import map_coordinates

    ncols = len(grid_x)
    nrows = len(grid_y)
    res_x = grid_x[1] - grid_x[0] if ncols > 1 else 1.0
    # grid_y is descending → pixel row 0 at top
    res_y = grid_y[0] - grid_y[1] if nrows > 1 else 1.0

    # Convert world coords to pixel coords (continuous). grid_x[0]/grid_y[0]
    # are pixel *centres*, and map_coordinates treats integer coordinates as
    # array indices (i.e. pixel centres too), so a point exactly at
    # grid_x[0]/grid_y[0] must map to index 0 — no extra +/-0.5 offset.
    col_coords = (points_x - grid_x[0]) / res_x
    row_coords = (grid_y[0] - points_y) / res_y

    results = {}
    for name, arr in rasters.items():
        vals = map_coordinates(
            arr.astype(np.float64),
            [row_coords, col_coords],
            order=1,           # bilinear
            mode="constant",
            cval=np.nan,
        )
        # Mask points outside the raster
        outside = (col_coords < 0) | (col_coords >= ncols) | \
                  (row_coords < 0) | (row_coords >= nrows)
        vals[outside] = np.nan
        results[name] = vals.astype(np.float32)
    return results


def _build_feature_matrix(
    extracted: dict[str, np.ndarray],
) -> np.ndarray:
    """Stack extracted raster values into a 2-d feature matrix ``(n, p)``."""
    cols = [extracted[k] for k in sorted(extracted.keys())]
    return np.column_stack(cols)


class _RidgeFallback:
    """NumPy ridge regression used when scikit-learn is unavailable.

    API-compatible with the parts of the sklearn estimator interface this
    module uses (``fit``, ``predict``, ``feature_importances_``).  Features are
    standardised internally, so urban parameters on wildly different scales
    (heights in metres, fractions in 0–1) are comparable, and the ridge term
    keeps the normal equations solvable when features are collinear or n < p.
    """

    def __init__(self, alpha: float = 1e-6):
        self.alpha = float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RidgeFallback":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        self._xm = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        self._xs = scale
        Xs = (X - self._xm) / self._xs
        A = np.column_stack([np.ones(len(Xs)), Xs])
        n_par = A.shape[1]
        reg = np.sqrt(self.alpha) * np.eye(n_par)
        reg[0, 0] = 0.0  # never penalise the intercept
        beta = np.linalg.lstsq(
            np.vstack([A, reg]), np.concatenate([y, np.zeros(n_par)]), rcond=None,
        )[0]
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        imp = np.abs(self.coef_)
        total = float(imp.sum())
        self.feature_importances_ = (
            imp / total if total > 0 else np.full(len(imp), 1.0 / max(len(imp), 1))
        )
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("_RidgeFallback: call fit() before predict().")
        Xs = (np.asarray(X, dtype=np.float64) - self._xm) / self._xs
        return self.intercept_ + Xs @ self.coef_


def _cv_r2_numpy(X: np.ndarray, y: np.ndarray, k: int = 5,
                 seed: int = 42) -> tuple[float, float]:
    """Deterministic k-fold R² for the NumPy fallback model."""
    n = len(X)
    k = int(max(2, min(k, n)))
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    scores = []
    for test_idx in folds:
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)
        if len(train_idx) < 2 or len(test_idx) < 1:
            continue
        model = _RidgeFallback().fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        ss_res = float(np.sum((y[test_idx] - pred) ** 2))
        ss_tot = float(np.sum((y[test_idx] - y[test_idx].mean()) ** 2))
        if ss_tot > 0:
            scores.append(1.0 - ss_res / ss_tot)
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


def _train_model_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    n_jobs: int = -1,
):
    """Fit a Random Forest Regressor with sensible urban-climate defaults.

    Falls back to :class:`_RidgeFallback` — a standardised NumPy ridge
    regression — when scikit-learn is not installed, so ``ml_model="rf"``
    still produces a usable map instead of raising.
    """
    if not HAS_SKLEARN:
        _warn_once(
            "sklearn_fallback",
            "scikit-learn is not installed — ml_model='rf' falls back to a "
            "NumPy ridge/OLS model with external drift. For the full Random "
            "Forest, install scikit-learn: pip install scikit-learn",
        )
        model = _RidgeFallback().fit(X, y)
        if len(X) >= 5:
            mean, std = _cv_r2_numpy(X, y, k=min(5, len(X)))
            if np.isfinite(mean):
                logger.info(
                    "Fallback ridge cross-val R² = %.3f ± %.3f  (n=%d, features=%d)",
                    mean, std, len(X), X.shape[1],
                )
        else:
            logger.warning("Only %d samples — skipping cross-validation.", len(X))
        return model

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    rf = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=None,
        min_samples_split=max(2, len(X) // 10),
        min_samples_leaf=max(1, len(X) // 20),
        max_features="sqrt",
        n_jobs=int(n_jobs),
        random_state=42,
    )
    rf.fit(X, y)

    # Cross-validated R² to log (never fatal — tiny samples can fail it)
    if len(X) >= 5:
        try:
            cv = int(min(5, len(X)))
            scores = cross_val_score(rf, X, y, cv=cv, scoring="r2")
            logger.info(
                "RF cross-val R² = %.3f ± %.3f  (n=%d, features=%d)",
                scores.mean(), scores.std(), len(X), X.shape[1],
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            logger.debug("RF cross-validation unavailable (%s).", exc)
    else:
        logger.warning("Only %d samples — skipping cross-validation.", len(X))

    return rf


def _train_model_ols(
    X: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit Ordinary Least Squares via numpy (no sklearn dependency).

    Returns ``(coefficients, r_squared)``.
    """
    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    coeffs, residuals, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)

    y_pred = X_with_intercept @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    logger.info(
        "OLS R² = %.3f  (n=%d, features=%d)",
        r2, len(X), X.shape[1],
    )
    return coeffs, r2


def _predict_ols(X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply OLS coefficients to a feature matrix."""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    return X_with_intercept @ coeffs


def _resample_urban_to_grid(
    combined,  # xarray Dataset
    transform,  # Affine
    width: int,
    height: int,
    epsg: str,
    var_names: list[str],
) -> dict[str, np.ndarray]:
    """Resample each xarray variable onto the prediction grid via rasterio reproject.

    Returns ``{name: 2-d numpy array}`` with shape ``(height, width)`` matching
    the north-up prediction grid.
    """
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS

    dst_crs = CRS.from_string(epsg)
    out = {}
    for name in var_names:
        if name not in combined.data_vars:
            continue
        da = combined[name]
        # Ensure data is 2-D (squeeze extra dims)
        arr = da.values.squeeze()
        if arr.ndim != 2:
            continue
        # Build source CRS/transform from xarray coords
        src_crs = CRS.from_string(da.rio.crs.to_string()) if da.rio.crs else CRS.from_epsg(4326)
        src_transform = da.rio.transform()
        dst = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=arr.astype(np.float32),
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
        out[name] = dst
    return out


def _download_urban_params(
    lcz_path: str,
    stations_pdf,
    variables: Optional[list[str]] = None,
    process_ghsl: bool = True,
    process_wumpod: bool = True,
    process_vegetation: bool = True,
    process_directional: bool = False,
    n_workers: int = -1,
    lang: str = "en",
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Download and cache urban parameter rasters.

    Returns
    -------
    combined : xarray.Dataset
        Combined urban parameter dataset.
    var_names : list[str]
        Names of successfully downloaded variables.
    """
    from LCZ4py.general.lcz_get_ucp import lcz_get_ucp

    n_w = n_workers if n_workers > 0 else max(1, (os.cpu_count() or 4) - 1)
    logger.info(lcz_msg("interp_plus_download", lang, n=n_w))

    result = lcz_get_ucp(
        lcz_map=lcz_path,
        stations=stations_pdf,
        cache_dir="lcz4r_cache",
        n_workers=n_w,
        verbose=True,
        variables=variables,
        process_ghsl=process_ghsl,
        process_wumpod=process_wumpod,
        process_vegetation=process_vegetation,
        process_directional=process_directional,
        use_threads=True,
        fail_fast=False,
    )

    combined = result.get("combined_rasters")
    var_names = result.get("variable_list", [])

    if combined is None or not var_names:
        logger.error(lcz_msg("interp_plus_no_data", lang))
        return {}, []

    logger.info("Downloaded %d / %d urban parameter rasters.", len(combined.data_vars), len(var_names))
    return combined, list(var_names)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def lcz_interp_map_plus(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
    data_frame,
    var: str = "",
    station_id: str = "",
    *,
    ml_model: MLModel = "rf",
    sp_res: float = 100.0,
    tp_res: str = "hour",
    by: Optional[str] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,
    ucp_variables: Optional[list[str]] = None,
    n_estimators: int = 200,
    min_samples: int = 3,
    use_pca: bool = False,
    n_pca_components: int = 0,
    process_ghsl: bool = True,
    process_wumpod: bool = True,
    process_vegetation: bool = True,
    process_directional: bool = False,
    use_lcz_params: bool = False,
    isave: bool = False,
    n_jobs: int = -1,
    lang: str = "en",
    year=None, month=None, day=None, hour=None, start=None, end=None,
    rasters: Optional[dict] = None,
    max_workers: Optional[int] = None,
) -> Optional[LCZInterpResult]:
    """Interpolate air temperature onto a regular north-up grid using machine learning.

    Trains a Random Forest or Multiple Linear Regression model on urban
    morphological parameters and predicts temperature pixel-by-pixel across
    the LCZ map grid.  Designed for urban areas with **low station density**
    where kriging can produce unstable results.

    The output GeoTIFF is always in a projected UTM CRS with a north-up
    affine transform built from the geographic bounds, guaranteeing correct
    orientation regardless of the input raster's rotation or shear.

    Parameters
    ----------
    x : str, PathLike, or rasterio dataset
        LCZ raster (defines the interpolation grid extent and CRS).
    data_frame : pd.DataFrame or pl.DataFrame
        Observation table with lat/lon/date/var/station columns.
    var : str
        Column name for the meteorological variable (e.g. ``"airT"``).
    station_id : str
        Column name for the station identifier.
    ml_model : ``"rf"`` or ``"ols"``
        ``"rf"`` — Random Forest Regressor (default).  Handles non-linear
        relationships, robust to outliers, provides feature importances and
        prediction uncertainty (std of tree predictions).

        ``"ols"`` — Ordinary Least Squares (Multiple Linear Regression).
        Fast, interpretable, no extra dependencies.  Reports VIF for
        multicollinearity diagnostics.
    sp_res : float
        Output grid spatial resolution in metres.
    tp_res : ``"hour"`` or ``"day"``
        Temporal averaging frequency.
    by : str, optional
        Aggregate time steps within each temporal group before modelling,
        producing one raster band per group.
    bbox : tuple of 4 floats, optional
        User-defined bounding box ``(minx, miny, maxx, maxy)`` in the
        geographic CRS of the LCZ raster (typically EPSG:4326).
        Crops the prediction grid and urban rasters to this area.
    ucp_variables : list of str, optional
        Urban Characterization Parameters to download.
    n_estimators : int
        Number of trees in the Random Forest.
    min_samples : int
        Minimum valid stations required per group to fit a model.
    use_pca : bool
        Apply PCA dimensionality reduction before fitting.  Useful when
        features are highly correlated (many urban parameters).
    n_pca_components : int
        Number of PCA components to retain.  ``0`` = keep all
        (equivalent to ``use_pca=False``).
    process_ghsl, process_wumpod, process_vegetation, process_directional : bool
        Urban parameter download flags.
    use_lcz_params : bool
        Use LCZ morphological parameters from the lookup table as features.
    isave : bool
        Copy the output GeoTIFF to ``LCZ4r_output/lcz4r_interp_map_plus.tif``.
    n_jobs : int
        Worker processes (``-1`` = all CPU cores).
    lang : str
        Language for log messages — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.
    rasters : dict of str → 2-D array, optional
        Pre-computed urban-parameter rasters already resampled onto the output
        grid (``(len(grid_y), len(grid_x))``), keyed by feature name.  When
        given, no download is attempted — useful for offline runs and for
        supplying your own covariates.
    max_workers : int, optional
        Thread workers for the per-group model fits
        (default ``min(8, cpu_count)``); predictions are written sequentially
        to keep peak memory at one band.

    Returns
    -------
    LCZInterpResult or None
        Structured result with path, uncertainty path, feature importances,
        PCA explained variance, and metadata.  ``None`` if no data.
    """
    # ---- 1. Input normalisation -------------------------------------------
    if ml_model not in ("rf", "ols"):
        raise ValueError(f"Unknown ml_model '{ml_model}'. Choose from 'rf', 'ols'.")
    if int(min_samples) < 1:
        raise ValueError("min_samples must be >= 1.")

    df = normalise_input_df(
        data_frame.to_pandas() if isinstance(data_frame, pl.DataFrame) else data_frame,
        var=var, station_id=station_id,
    )
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null())

    if df.is_empty():
        logger.warning("No valid observations after filtering.")
        return None

    # ---- 2. Load LCZ raster and extract classes at stations ---------------
    ds = load_lcz_raster(x)
    stations = df.unique(subset=["latitude", "longitude"])
    lcz_vals, mask = extract_lcz_at_points(
        ds, stations["longitude"].to_numpy(), stations["latitude"].to_numpy(),
    )
    stations = stations.with_columns(
        pl.Series(name="lcz", values=np.where(mask, lcz_vals, 0)).cast(int),
    )

    # ---- 3. Build prediction grid (guaranteed north-up) -------------------
    grid_x, grid_y, transform, epsg = _make_northup_grid(ds, sp_res, bbox=bbox)
    from pyproj import Transformer
    tx = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    xs, ys = tx.transform(
        stations["longitude"].to_numpy(), stations["latitude"].to_numpy(),
    )
    stations = stations.with_columns(x=xs, y=ys)

    # Join coordinates back to the full dataframe
    df = df.join(
        stations.select(["latitude", "longitude", "lcz", "x", "y"]),
        on=["latitude", "longitude"],
        how="left",
    )
    df = df.filter(pl.col("lcz").is_between(1, 17))
    if df.is_empty():
        logger.warning(
            "No stations fall inside LCZ classes 1–17 of the supplied map — "
            "nothing to interpolate."
        )
        return None

    # Always resample LCZ to the prediction grid (for masking + optional use)
    lcz_grid = _resample_lcz_to_grid(ds, transform, len(grid_x), len(grid_y), epsg)
    valid_mask = (lcz_grid >= 1) & (lcz_grid <= 17)
    if not valid_mask.any():
        logger.warning(
            "The resampled LCZ grid has no valid classes — check that the LCZ "
            "raster CRS matches its bounds."
        )
        return None

    # ---- 4. Urban parameter rasters: caller-supplied or downloaded --------
    ny, nx = len(grid_y), len(grid_x)
    if rasters is not None:
        supplied = {}
        for name, arr in dict(rasters).items():
            arr = np.asarray(arr, dtype=np.float64)
            if arr.shape != (ny, nx):
                raise ValueError(
                    f"rasters['{name}'] has shape {arr.shape} but the prediction "
                    f"grid is {(ny, nx)}. Supply arrays already on the output grid."
                )
            supplied[str(name)] = arr
        if not supplied:
            logger.error(lcz_msg("interp_plus_no_data", lang))
            return None
        rasters_used = supplied
        downloaded_vars = list(supplied)
        logger.info("Using %d caller-supplied urban parameter raster(s).", len(supplied))
    else:
        variables_to_download = ucp_variables if ucp_variables is not None else _DEFAULT_UCP_VARS
        combined, downloaded_vars = _download_urban_params(
            lcz_path=str(x) if not isinstance(x, rasterio.io.DatasetReader) else x.name,
            stations_pdf=stations.select(["latitude", "longitude"]).to_pandas(),
            variables=variables_to_download,
            process_ghsl=process_ghsl,
            process_wumpod=process_wumpod,
            process_vegetation=process_vegetation,
            process_directional=process_directional,
            n_workers=n_jobs,
            lang=lang,
        )

        if not combined:
            logger.error(lcz_msg("interp_plus_no_data", lang))
            return None

        # Resample urban rasters onto the prediction grid (UTM, north-up)
        rasters_used = _resample_urban_to_grid(
            combined, transform, nx, ny, epsg, downloaded_vars,
        )
        if not rasters_used:
            logger.error(lcz_msg("interp_plus_no_data", lang))
            return None

    # ---- 5. Extract raster values at station locations ---------------------
    station_x = stations["x"].cast(pl.Float64).to_numpy()
    station_y = stations["y"].cast(pl.Float64).to_numpy()
    extracted = _extract_raster_values_at_points(
        rasters_used, grid_x, grid_y, station_x, station_y,
    )

    # Drop rasters that are mostly zeros/NaN (likely failed downloads)
    bad_rasters = _check_missing_rasters(
        {k: v for k, v in extracted.items() if not k.startswith("lcz_")},
    )
    for name in bad_rasters:
        extracted.pop(name, None)
    if not extracted and not use_lcz_params:
        logger.error(
            "Every urban parameter raster was empty or mostly nodata — cannot "
            "train a model. Check the download or pass `rasters=`."
        )
        return None

    # Optionally add LCZ morphological parameters as features
    lcz_param_names: list[str] = []
    if use_lcz_params:
        lcz_at_stations = stations["lcz"].to_numpy()
        lcz_feats, lcz_param_names = _extract_lcz_params_at_stations(lcz_at_stations)
        for j, pname in enumerate(lcz_param_names):
            extracted[f"lcz_{pname}"] = lcz_feats[:, j]

    logger.info("Features: %s (%d total)", sorted(extracted), len(extracted))

    # ---- 6. Temporal grouping ---------------------------------------------
    if by is not None:
        lat_mean = float(df["latitude"].mean())
        lon_mean = float(df["longitude"].mean())
        df = add_by_column(df, by, lat=lat_mean, lon=lon_mean)
        group_avg = (
            df.group_by(["station", "_by"])
              .agg([
                  pl.col("var_interp").mean(),
                  pl.col("x").first(),
                  pl.col("y").first(),
                  pl.col("lcz").first(),
              ])
        )
        groups = by_sorted_groups(df["_by"], by)
        group_specs = []
        for g in groups:
            sub = group_avg.filter(pl.col("_by") == g)
            if len(sub) < 2:
                logger.warning("Group '%s': only %d station(s) — skipped.", g, len(sub))
                continue
            group_specs.append((g, sub))
    else:
        freq = {"hour": "1h", "day": "1d"}.get(tp_res, "1h")
        if tp_res not in ("hour", "day"):
            logger.warning("Unknown tp_res '%s' — defaulting to hourly.", tp_res)
        avg = (
            df.sort("date")
              .group_by_dynamic("date", every=freq, group_by="station")
              .agg([
                  pl.col("var_interp").mean(),
                  pl.col("x").first(),
                  pl.col("y").first(),
                  pl.col("lcz").first(),
              ])
        )
        group_specs = []
        for t in avg["date"].unique().sort():
            sub = avg.filter(pl.col("date") == t)
            if len(sub) < 2:
                continue
            group_specs.append((t, sub))

    if not group_specs:
        logger.warning("No time step/group had at least 2 stations — nothing to model.")
        return None

    # ---- 7. Per-group station features ------------------------------------
    all_stations_xy = stations.select(["x", "y"]).unique().to_pandas()
    all_extracted = _extract_raster_values_at_points(
        rasters_used, grid_x, grid_y,
        all_stations_xy["x"].to_numpy(dtype=np.float64),
        all_stations_xy["y"].to_numpy(dtype=np.float64),
    )
    if use_lcz_params and lcz_param_names:
        st_lcz_map = stations.select(["x", "y", "lcz"]).unique().to_pandas()
        lcz_classes_all = st_lcz_map["lcz"].to_numpy()
        lcz_feats_all, _ = _extract_lcz_params_at_stations(lcz_classes_all)
        for j, pname in enumerate(lcz_param_names):
            all_extracted[f"lcz_{pname}"] = lcz_feats_all[:, j]

    # Build a lookup: pixel (row, col) -> feature row index (avoids float rounding)
    res_x = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
    res_y = grid_y[0] - grid_y[1] if len(grid_y) > 1 else 1.0
    pix_to_idx: dict[tuple[int, int], int] = {}
    for i in range(len(all_stations_xy)):
        col = int((float(all_stations_xy.iloc[i]["x"]) - grid_x[0]) / res_x)
        row = int((grid_y[0] - float(all_stations_xy.iloc[i]["y"])) / res_y)
        col = max(0, min(col, len(grid_x) - 1))
        row = max(0, min(row, len(grid_y) - 1))
        pix_to_idx[(row, col)] = i

    def _row_feature_index(row) -> Optional[int]:
        col = int((float(row["x"]) - grid_x[0]) / res_x)
        rrow = int((grid_y[0] - float(row["y"])) / res_y)
        col = max(0, min(col, len(grid_x) - 1))
        rrow = max(0, min(rrow, len(grid_y) - 1))
        return pix_to_idx.get((rrow, col))

    # ---- 8. Raster feature stack for pixel-wise prediction ----------------
    urban_names = sorted(n for n in extracted if not n.startswith("lcz_"))
    pixel_rasters: dict[str, np.ndarray] = {
        name: np.asarray(rasters_used[name], dtype=np.float32) for name in urban_names
    }

    if use_lcz_params and lcz_param_names:
        from LCZ4py.general.lcz_get_parameters import _get_parameter_lookup
        lookup, all_names = _get_parameter_lookup()
        safe = np.clip(lcz_grid.astype(int), 0, 17)
        lcz_pixel_params = lookup[safe]  # (ny, nx, n_params)
        mean_idx = [all_names.index(n) for n in [
            "svf_mean", "BSF_mean", "ISF_mean", "PSF_mean", "TSF_mean",
            "HRE_mean", "TRC_mean", "SAD_mean", "SAL_mean", "AH_mean", "z0",
        ]]
        for j, pname in enumerate(lcz_param_names):
            pixel_rasters[f"lcz_{pname}"] = lcz_pixel_params[:, :, mean_idx[j]]

    # Column order must match all_feature_names exactly — the training matrix
    # is built via all_extracted[f] lookups in this same order.
    all_feature_names = urban_names + [f"lcz_{n}" for n in lcz_param_names]
    if not all_feature_names:
        logger.error("No usable features available — cannot fit a model.")
        return None
    missing = [f for f in all_feature_names if f not in pixel_rasters]
    if missing:
        logger.error("Raster(s) missing for prediction: %s", missing)
        return None
    raster_stack = np.stack(
        [pixel_rasters[name] for name in all_feature_names], axis=-1,
    ).astype(np.float32)
    flat_features = raster_stack.reshape(-1, raster_stack.shape[-1])
    del raster_stack

    # ---- 9. Fit models per group (parallel), predict + write in order ------
    workers = DEFAULT_MAX_WORKERS if max_workers is None else max(1, int(max_workers))
    workers = max(1, min(workers, len(group_specs)))
    rf_n_jobs = -1 if workers == 1 else 1

    def _fit_group(spec):
        label, sub = spec
        train_rows = sub.to_pandas()
        X_list: list[list[float]] = []
        y_list: list[float] = []
        for _, row in train_rows.iterrows():
            idx = _row_feature_index(row)
            if idx is None:
                continue
            feat = [all_extracted[f][idx] for f in all_feature_names]
            if any(not np.isfinite(v) for v in feat):
                continue
            X_list.append(feat)
            y_list.append(float(row["var_interp"]))

        n_feat = len(all_feature_names)
        min_needed = max(int(min_samples), n_feat + 1) if ml_model == "ols" else int(min_samples)
        if len(X_list) < min_needed:
            logger.warning(lcz_msg(
                "interp_plus_few_stations", lang, group=str(label), n=len(X_list),
            ))
            return None
        if len(X_list) < 5:
            logger.warning(lcz_msg(
                "interp_plus_few_stations", lang, group=str(label), n=len(X_list),
            ))

        X_train = np.asarray(X_list, dtype=np.float32)
        y_train = np.asarray(y_list, dtype=np.float32)

        pca_obj = None
        pca_evr = None
        X_fit = X_train
        if use_pca and n_pca_components != 0:
            X_fit, _unused, pca_evr, pca_obj = _apply_pca(
                X_train, X_train, n_pca_components,
            )

        fit: dict = {
            "label": label, "pca_obj": pca_obj, "pca_evr": pca_evr,
            "n_train": len(X_list), "model": None, "coeffs": None,
        }

        if ml_model == "rf":
            model = _train_model_rf(
                X_fit, y_train, n_estimators=n_estimators, n_jobs=rf_n_jobs,
            )
            y_hat = model.predict(X_fit)
            fit["model"] = model
            fit["resid_se"] = float(np.sqrt(np.mean((y_train - y_hat) ** 2)))
            names = (all_feature_names if pca_obj is None
                     else [f"PC{i + 1}" for i in range(X_fit.shape[1])])
            importances = dict(zip(names, np.asarray(model.feature_importances_).ravel()))
            fit["importances"] = {k: float(v) for k, v in importances.items()}
            top5 = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]
            logger.info(
                "Group '%s' RF top-5 features: %s",
                label, [(f, f"{v:.3f}") for f, v in top5],
            )
        else:
            coeffs, r2 = _train_model_ols(X_fit, y_train)
            y_hat = _predict_ols(X_fit, coeffs)
            resid_se = float(np.sqrt(np.mean((y_train - y_hat) ** 2)))
            fit["coeffs"] = coeffs
            fit["resid_se"] = resid_se
            coef_names = ["intercept"] + (
                all_feature_names if pca_obj is None
                else [f"PC{i + 1}" for i in range(X_fit.shape[1])]
            )
            fit["importances"] = {
                k: float(v) for k, v in zip(coef_names, np.asarray(coeffs).ravel())
            }
            fit["r2"] = float(r2)
            fit["residual_se"] = resid_se
            if X_train.shape[0] > X_train.shape[1] + 1:
                vif = _compute_ols_vif(X_train)
                if np.nanmax(vif) > 10:
                    logger.warning(
                        "Group '%s': severe multicollinearity (max VIF = %.1f) — "
                        "consider use_pca=True.", label, float(np.nanmax(vif)),
                    )
            logger.info("Group '%s' OLS R² = %.3f, residual SE = %.3f",
                        label, r2, resid_se)
        return fit

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fitted_all = list(ex.map(_fit_group, group_specs))  # deterministic order
    else:
        fitted_all = [_fit_group(spec) for spec in group_specs]

    fitted = [f for f in fitted_all if f is not None]
    if not fitted:
        logger.error(
            "No group had enough valid stations to fit %s — no output written.",
            ml_model,
        )
        return None

    def _predict_flat(fit: dict) -> tuple[np.ndarray, np.ndarray]:
        n_pixels = flat_features.shape[0]
        pred = np.empty(n_pixels, dtype=np.float32)
        unc = np.empty(n_pixels, dtype=np.float32)
        chunk = 500_000
        for lo in range(0, n_pixels, chunk):
            hi = min(lo + chunk, n_pixels)
            Xc = flat_features[lo:hi]
            if fit["pca_obj"] is not None:
                Xc = fit["pca_obj"].transform(Xc).astype(np.float32)
            model = fit["model"]
            if model is not None:
                pred[lo:hi] = np.asarray(model.predict(Xc), dtype=np.float32)
                estimators = getattr(model, "estimators_", None)
                if estimators is not None:
                    tree_preds = np.stack(
                        [t.predict(Xc) for t in estimators], axis=0,
                    )
                    unc[lo:hi] = np.std(tree_preds, axis=0)
                else:
                    unc[lo:hi] = fit["resid_se"]
            else:
                pred[lo:hi] = _predict_ols(Xc, fit["coeffs"]).astype(np.float32)
                unc[lo:hi] = fit["resid_se"]
        return pred, unc

    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    unc_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": len(fitted),
        "dtype": "float32",
        "crs": epsg,
        "transform": transform,
        "compress": "lzw",
        "nodata": np.nan,
    }

    all_importances: dict = {}
    all_cv_scores: dict = {}
    completed_groups: list = []
    pca_evr_list: list[float] = []

    with rasterio.open(out_path, "w", **profile) as dst, \
         rasterio.open(unc_path, "w", **profile) as unc_dst:
        for band_idx, fit in enumerate(fitted, start=1):
            flat_pred, flat_unc = _predict_flat(fit)
            pred_grid = np.where(valid_mask, flat_pred.reshape(ny, nx), np.nan).astype(np.float32)
            unc_grid = np.where(valid_mask, flat_unc.reshape(ny, nx), np.nan).astype(np.float32)

            dst.write(pred_grid, band_idx)
            dst.set_band_description(band_idx, str(fit["label"]))
            unc_dst.write(unc_grid, band_idx)
            unc_dst.set_band_description(band_idx, str(fit["label"]))

            all_importances[str(fit["label"])] = fit["importances"]
            if ml_model == "ols":
                all_cv_scores[str(fit["label"])] = {
                    "r2": fit["r2"], "residual_se": fit["residual_se"],
                }
            if fit["pca_evr"]:
                pca_evr_list = fit["pca_evr"]
            completed_groups.append(fit["label"])

    # ---- 10. Save if requested -------------------------------------------
    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "lcz4r_interp_map_plus.tif")
        with rasterio.open(out_path) as src, \
             rasterio.open(save_path, "w", **src.profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)
                desc = src.descriptions[i - 1]
                if desc:
                    dst.set_band_description(i, desc)
        logger.info(lcz_msg("interp_plus_saved", lang, path=os.path.abspath(save_path)))

    logger.info(lcz_msg(
        "interp_plus_complete", lang, n=len(completed_groups), method=ml_model,
    ))
    return LCZInterpResult(
        path=out_path,
        feature_importance=all_importances or None,
        cv_scores=all_cv_scores or None,
        model_type=ml_model,
        n_stations=len(stations),
        n_features=len(all_feature_names),
        groups=completed_groups,
        uncertainty_path=unc_path if completed_groups else None,
        pca_explained_variance=pca_evr_list if pca_evr_list else None,
    )


# ---------------------------------------------------------------------------
# ML cross-validation
# ---------------------------------------------------------------------------

_EVAL_SCHEMA = {
    "station": pl.String,
    "group": pl.String,
    "observed": pl.Float64,
    "predicted": pl.Float64,
    "residual": pl.Float64,
    "cv_method": pl.String,
    "rmse": pl.Float64,
    "mae": pl.Float64,
    "r2": pl.Float64,
}


def _empty_eval_frame() -> pl.DataFrame:
    """Empty CV result with the documented schema (never a schema-less frame)."""
    return pl.DataFrame(schema=_EVAL_SCHEMA)


def lcz_interp_eval_plus(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
    data_frame,
    var: str = "",
    station_id: str = "",
    *,
    ml_model: MLModel = "rf",
    LOOCV: bool = True,
    split_ratio: float = 0.8,
    sp_res: float = 100.0,
    tp_res: str = "hour",
    by: Optional[str] = None,
    ucp_variables: Optional[list[str]] = None,
    n_estimators: int = 200,
    process_ghsl: bool = True,
    process_wumpod: bool = True,
    process_vegetation: bool = True,
    process_directional: bool = False,
    isave: bool = False,
    n_jobs: int = -1,
    lang: str = "en",
    year=None, month=None, day=None, hour=None, start=None, end=None,
    rasters: Optional[dict] = None,
    seed: Optional[int] = 42,
    max_workers: Optional[int] = None,
) -> pl.DataFrame:
    """Cross-validate ML temperature interpolation via LOOCV or hold-out.

    Same feature pipeline as :func:`lcz_interp_map_plus` but evaluates
    prediction accuracy by leaving stations out and predicting at their
    locations.

    Parameters
    ----------
    x : str, PathLike, or rasterio dataset
        LCZ raster (defines the interpolation grid extent and CRS).
    data_frame : pd.DataFrame or pl.DataFrame
        Observation table with lat/lon/date/var/station columns.
    var, station_id : str
        Column names for the meteorological variable and station identifier.
    ml_model : ``"rf"`` or ``"ols"``
        ML model type.
    LOOCV : bool
        If True, leave-one-station-out per group.  If False, random hold-out.
    split_ratio : float
        Fraction for training in hold-out mode (ignored when ``LOOCV=True``).
    sp_res : float
        Spatial resolution in metres.
    tp_res : ``"hour"`` or ``"day"``
        Temporal averaging frequency.
    by : str, optional
        Grouping variable (same as ``lcz_interp_map_plus``).
    ucp_variables : list of str, optional
        Urban parameters to download.
    n_estimators : int
        RF tree count (ignored for ``"ols"``).
    process_ghsl, process_wumpod, process_vegetation, process_directional : bool
        Urban parameter download flags.
    isave : bool
        Save result to ``LCZ4r_output/lcz4r_interp_eval_plus.csv``.
    n_jobs : int
        Worker count for urban parameter download.
    lang : str
        Language for log messages.
    rasters : dict of str → 2-D array, optional
        Pre-computed urban-parameter rasters already on the output grid; when
        given, no download is attempted.
    max_workers : int, optional
        Thread workers for independent cross-validation folds
        (default ``min(8, cpu_count)``).
    seed : int or None
        Seed for the hold-out split (deterministic by default).

    Returns
    -------
    pl.DataFrame
        Columns: station, group, observed, predicted, residual, rmse, mae, r2.
    """
    # ---- 1. Input normalisation -------------------------------------------
    if ml_model not in ("rf", "ols"):
        raise ValueError(f"Unknown ml_model '{ml_model}'. Choose from 'rf', 'ols'.")
    if not LOOCV and not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio!r}.")

    df = normalise_input_df(
        data_frame.to_pandas() if isinstance(data_frame, pl.DataFrame) else data_frame,
        var=var, station_id=station_id,
    )
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null())

    if df.is_empty():
        logger.warning("No valid observations after filtering.")
        return _empty_eval_frame()

    # ---- 2. Load LCZ + extract classes + build grid ----------------------
    ds = load_lcz_raster(x)
    stations = df.unique(subset=["latitude", "longitude"])
    lcz_vals, mask = extract_lcz_at_points(
        ds, stations["longitude"].to_numpy(), stations["latitude"].to_numpy(),
    )
    stations = stations.with_columns(
        pl.Series(name="lcz", values=np.where(mask, lcz_vals, 0)).cast(int),
    )

    grid_x, grid_y, transform, epsg = _make_northup_grid(ds, sp_res)
    from pyproj import Transformer
    tx = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    xs, ys = tx.transform(
        stations["longitude"].to_numpy(), stations["latitude"].to_numpy(),
    )
    stations = stations.with_columns(x=xs, y=ys)

    df = df.join(
        stations.select(["latitude", "longitude", "lcz", "x", "y"]),
        on=["latitude", "longitude"],
        how="left",
    )
    df = df.filter(pl.col("lcz").is_between(1, 17))
    if df.is_empty():
        logger.warning(
            "No stations fall inside LCZ classes 1–17 of the supplied map — "
            "nothing to validate."
        )
        return _empty_eval_frame()

    ny, nx = len(grid_y), len(grid_x)

    # ---- 3. Urban parameter rasters: caller-supplied or downloaded --------
    if rasters is not None:
        supplied = {}
        for name, arr in dict(rasters).items():
            arr = np.asarray(arr, dtype=np.float64)
            if arr.shape != (ny, nx):
                raise ValueError(
                    f"rasters['{name}'] has shape {arr.shape} but the prediction "
                    f"grid is {(ny, nx)}. Supply arrays already on the output grid."
                )
            supplied[str(name)] = arr
        if not supplied:
            logger.error(lcz_msg("interp_plus_no_data", lang))
            return _empty_eval_frame()
        rasters_used = supplied
        downloaded_vars = list(supplied)
    else:
        variables_to_download = ucp_variables if ucp_variables is not None else _DEFAULT_UCP_VARS
        combined, downloaded_vars = _download_urban_params(
            lcz_path=str(x) if not isinstance(x, rasterio.io.DatasetReader) else x.name,
            stations_pdf=stations.select(["latitude", "longitude"]).to_pandas(),
            variables=variables_to_download,
            process_ghsl=process_ghsl,
            process_wumpod=process_wumpod,
            process_vegetation=process_vegetation,
            process_directional=process_directional,
            n_workers=n_jobs,
            lang=lang,
        )

        if not combined:
            return _empty_eval_frame()

        # Resample urban rasters onto the prediction grid (UTM, north-up)
        rasters_used = _resample_urban_to_grid(
            combined, transform, nx, ny, epsg, downloaded_vars,
        )
        if not rasters_used:
            return _empty_eval_frame()

    # ---- 4. Extract features at all stations -----------------------------
    all_stations_xy = stations.select(["x", "y"]).unique().to_pandas()
    all_extracted = _extract_raster_values_at_points(
        rasters_used, grid_x, grid_y,
        all_stations_xy["x"].to_numpy(dtype=np.float64),
        all_stations_xy["y"].to_numpy(dtype=np.float64),
    )
    feature_names = sorted(all_extracted.keys())
    if not feature_names:
        logger.error("No usable features available — cannot cross-validate.")
        return _empty_eval_frame()

    # Pixel-index lookup
    res_x = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
    res_y = grid_y[0] - grid_y[1] if len(grid_y) > 1 else 1.0
    pix_to_idx: dict[tuple[int, int], int] = {}
    for i in range(len(all_stations_xy)):
        col = int((float(all_stations_xy.iloc[i]["x"]) - grid_x[0]) / res_x)
        row = int((grid_y[0] - float(all_stations_xy.iloc[i]["y"])) / res_y)
        col = max(0, min(col, len(grid_x) - 1))
        row = max(0, min(row, len(grid_y) - 1))
        pix_to_idx[(row, col)] = i

    # ---- 5. Temporal grouping --------------------------------------------
    if by is not None:
        lat_mean = float(df["latitude"].mean())
        lon_mean = float(df["longitude"].mean())
        df = add_by_column(df, by, lat=lat_mean, lon=lon_mean)
        group_avg = (
            df.group_by(["station", "_by"])
              .agg([
                  pl.col("var_interp").mean().alias("observed"),
                  pl.col("x").first(),
                  pl.col("y").first(),
                  pl.col("lcz").first(),
              ])
        )
        groups = by_sorted_groups(df["_by"], by)
        group_list = [(g, group_avg.filter(pl.col("_by") == g)) for g in groups
                      if len(group_avg.filter(pl.col("_by") == g)) >= 2]
    else:
        freq = {"hour": "1h", "day": "1d"}.get(tp_res, "1h")
        if tp_res not in ("hour", "day"):
            logger.warning("Unknown tp_res '%s' — defaulting to hourly.", tp_res)
        avg = (
            df.sort("date")
              .group_by_dynamic("date", every=freq, group_by="station")
              .agg([
                  pl.col("var_interp").mean().alias("observed"),
                  pl.col("x").first(),
                  pl.col("y").first(),
                  pl.col("lcz").first(),
              ])
        )
        group_list = [(t, avg.filter(pl.col("date") == t))
                      for t in avg["date"].unique().sort()
                      if len(avg.filter(pl.col("date") == t)) >= 2]

    if not group_list:
        logger.warning("No time step/group had at least 2 stations — nothing to validate.")
        return _empty_eval_frame()

    # ---- 6. Cross-validation per group (parallel, deterministic order) ----
    def _fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        if ml_model == "rf":
            model = _train_model_rf(
                X_tr, y_tr, n_estimators=n_estimators, n_jobs=1,
            )
            return np.asarray(model.predict(X_te), dtype=np.float64)
        coeffs, _ = _train_model_ols(X_tr, y_tr)
        return np.asarray(_predict_ols(X_te, coeffs), dtype=np.float64)

    def _eval_group(payload) -> list[dict]:
        group_idx, (label, group_df) = payload
        rows = group_df.to_pandas()
        X_all: list[list[float]] = []
        y_all: list[float] = []
        station_names: list[str] = []
        for _, row in rows.iterrows():
            col = int((float(row["x"]) - grid_x[0]) / res_x)
            rrow = int((grid_y[0] - float(row["y"])) / res_y)
            col = max(0, min(col, len(grid_x) - 1))
            rrow = max(0, min(rrow, len(grid_y) - 1))
            idx = pix_to_idx.get((rrow, col))
            if idx is None:
                continue
            feat = [all_extracted[f][idx] for f in feature_names]
            if any(not np.isfinite(v) for v in feat):
                continue
            X_all.append(feat)
            y_all.append(float(row["observed"]))
            station_names.append(str(row["station"]))

        out_rows: list[dict] = []
        if len(X_all) < 3:
            logger.warning(
                "Group '%s': only %d usable station(s) — skipped.", label, len(X_all),
            )
            return out_rows

        X_arr = np.asarray(X_all, dtype=np.float32)
        y_arr = np.asarray(y_all, dtype=np.float32)

        if LOOCV:
            for i in range(len(X_arr)):
                train_mask = np.arange(len(X_arr)) != i
                X_tr, y_tr = X_arr[train_mask], y_arr[train_mask]
                if len(X_tr) < 2:
                    continue
                try:
                    pred = float(_fit_predict(X_tr, y_tr, X_arr[i:i + 1])[0])
                except Exception as exc:  # noqa: BLE001 - fold-level failure
                    logger.warning(
                        "LOOCV fold failed for station %s (group '%s'): %s: %s",
                        station_names[i], label, type(exc).__name__, exc,
                    )
                    continue
                if not np.isfinite(pred):
                    continue
                out_rows.append({
                    "station": station_names[i],
                    "group": str(label),
                    "observed": float(y_arr[i]),
                    "predicted": pred,
                    "residual": float(y_arr[i]) - pred,
                    "cv_method": "loocv",
                })
        else:
            n = len(X_arr)
            group_seed = None if seed is None else int(seed) + int(group_idx)
            perm = np.random.default_rng(group_seed).permutation(n)
            n_train = max(2, int(n * split_ratio))
            train_idx, test_idx = perm[:n_train], perm[n_train:]
            if len(test_idx) < 1 or len(train_idx) < 2:
                return out_rows
            try:
                preds = _fit_predict(X_arr[train_idx], y_arr[train_idx], X_arr[test_idx])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Hold-out fit failed (group '%s'): %s: %s",
                    label, type(exc).__name__, exc,
                )
                return out_rows
            for j, idx in enumerate(test_idx):
                p = float(preds[j])
                if not np.isfinite(p):
                    continue
                out_rows.append({
                    "station": station_names[idx],
                    "group": str(label),
                    "observed": float(y_arr[idx]),
                    "predicted": p,
                    "residual": float(y_arr[idx]) - p,
                    "cv_method": "holdout",
                })
        return out_rows

    workers = DEFAULT_MAX_WORKERS if max_workers is None else max(1, int(max_workers))
    workers = min(workers, max(1, len(group_list)))
    payloads = list(enumerate(group_list))

    if workers > 1 and len(payloads) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            per_group = list(ex.map(_eval_group, payloads))  # deterministic order
    else:
        per_group = [_eval_group(p) for p in payloads]

    results = [row for group in per_group for row in group]
    if not results:
        logger.warning(
            "Cross-validation produced no results — check the number of usable "
            "stations per group and the urban-parameter rasters."
        )
        return _empty_eval_frame()

    out = pl.DataFrame(results)
    # Per-group RMSE and MAE (use np.sqrt to avoid Polars scalar issue)
    group_rmse = (
        out.group_by("group")
           .agg(np.sqrt((pl.col("residual") ** 2).mean()).alias("rmse"))
    )
    group_mae = (
        out.group_by("group")
           .agg(pl.col("residual").abs().mean().alias("mae"))
    )
    out = out.join(group_rmse, on="group", how="left")
    out = out.join(group_mae, on="group", how="left")
    # R² per group
    r2_per_group = (
        out.group_by("group")
           .agg([
               (1.0 - (pl.col("residual") ** 2).sum()
                / ((pl.col("observed") - pl.col("observed").mean()) ** 2).sum()
               ).alias("r2"),
           ])
    )
    out = out.join(r2_per_group, on="group", how="left")

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DIR, "lcz4r_interp_eval_plus.csv")
        out.write_csv(csv_path)
        logger.info(lcz_msg("save_output_path", lang, path=os.path.abspath(csv_path)))

    return out


__all__ = [
    "lcz_interp_map_plus", "lcz_interp_eval_plus", "LCZInterpResult", "MLModel",
    "_make_northup_grid", "_check_missing_rasters", "_apply_pca", "_compute_ols_vif",
    "HAS_SKLEARN", "DEFAULT_MAX_WORKERS",
]
