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
from LCZ4py.local.lcz_interp_map import _make_grid_memsafe, _resample_lcz_to_grid
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

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
        total = arr.size
        if total == 0:
            bad.append(name)
            continue
        valid = np.count_nonzero(arr) + np.count_nonzero(np.isnan(arr) == 0)
        # count non-zero, non-NaN pixels
        nonzero = np.count_nonzero(arr != 0)
        finite = np.count_nonzero(np.isfinite(arr))
        valid_frac = nonzero / total
        if valid_frac < min_valid_fraction:
            logger.warning("Raster '%s': only %.1f%% valid pixels — excluding.", name, valid_frac * 100)
            bad.append(name)
    return bad


def _apply_pca(
    X_train: np.ndarray,
    X_predict: np.ndarray,
    n_components: int = 0,
) -> tuple[np.ndarray, np.ndarray, Optional[list[float]], object]:
    """Apply PCA dimensionality reduction.

    If ``n_components == 0``, use all components (no reduction).
    Returns ``(X_train_pca, X_predict_pca, explained_variance, pca_obj)``.
    """
    from sklearn.decomposition import PCA

    if n_components <= 0:
        n_components = min(X_train.shape[0], X_train.shape[1])

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_predict_pca = pca.transform(X_predict)
    evr = pca.explained_variance_ratio_.tolist()
    logger.info("PCA: %d → %d components (%.1f%% variance retained)",
                X_train.shape[1], len(evr), sum(evr) * 100)
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


def _train_model_rf(
    X: np.ndarray, y: np.ndarray, n_estimators: int = 200,
):
    """Fit a Random Forest Regressor with sensible urban-climate defaults."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        raise ImportError(
            "scikit-learn is required for ml_model='rf'. "
            "Install with: pip install scikit-learn"
        )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=max(2, len(X) // 10),
        min_samples_leaf=max(1, len(X) // 20),
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X, y)

    # Cross-validated R² to log
    if len(X) >= 5:
        cv = min(5, len(X))
        scores = cross_val_score(rf, X, y, cv=cv, scoring="r2")
        logger.info(
            "RF cross-val R² = %.3f ± %.3f  (n=%d, features=%d)",
            scores.mean(), scores.std(), len(X), X.shape[1],
        )
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
    import xarray as xr

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

    Returns
    -------
    LCZInterpResult or None
        Structured result with path, uncertainty path, feature importances,
        PCA explained variance, and metadata.  ``None`` if no data.
    """
    # ---- 1. Input normalisation -------------------------------------------
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

    # Always resample LCZ to the prediction grid (for masking + optional use)
    lcz_grid = _resample_lcz_to_grid(ds, transform, len(grid_x), len(grid_y), epsg)
    valid_mask = (lcz_grid >= 1) & (lcz_grid <= 17)

    # ---- 4. Download urban parameter rasters ------------------------------
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
    rasters = _resample_urban_to_grid(
        combined, transform, len(grid_x), len(grid_y), epsg, downloaded_vars,
    )
    if not rasters:
        logger.error(lcz_msg("interp_plus_no_data", lang))
        return None

    # ---- 5. Extract raster values at station locations ---------------------
    station_x = stations["x"].cast(pl.Float64).to_numpy()
    station_y = stations["y"].cast(pl.Float64).to_numpy()
    extracted = _extract_raster_values_at_points(rasters, grid_x, grid_y, station_x, station_y)

    # Drop rasters that are mostly zeros/NaN (likely failed downloads)
    bad_rasters = _check_missing_rasters(
        {k: extracted[k] for k in extracted if not k.startswith("lcz_")},
    )
    for name in bad_rasters:
        extracted.pop(name, None)
    feature_names = sorted(extracted.keys())

    # Optionally add LCZ morphological parameters as features
    lcz_param_names: list[str] = []
    if use_lcz_params:
        lcz_at_stations = stations["lcz"].to_numpy()
        lcz_feats, lcz_param_names = _extract_lcz_params_at_stations(lcz_at_stations)
        for j, pname in enumerate(lcz_param_names):
            extracted[f"lcz_{pname}"] = lcz_feats[:, j]
        feature_names = sorted(extracted.keys())

    logger.info("Features: %s (%d total)", feature_names, len(feature_names))

    # ---- 6. Temporal grouping and per-group ML fitting ---------------------
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
                continue
            group_specs.append((g, sub))
    else:
        freq = {"hour": "1h", "day": "1d"}.get(tp_res, "1h")
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
        return None

    # ---- 7. Extract per-group station features from the full station set ---
    # Re-extract features for all unique station coordinates so we can
    # select the right rows for each group.
    all_stations_xy = stations.select(["x", "y"]).unique().to_pandas()
    all_extracted = _extract_raster_values_at_points(
        rasters, grid_x, grid_y,
        all_stations_xy["x"].to_numpy(dtype=np.float64),
        all_stations_xy["y"].to_numpy(dtype=np.float64),
    )
    # Add LCZ parameters at all unique station locations
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

    # ---- 8. Build raster feature grid for pixel-wise prediction -----------
    ny, nx = len(grid_y), len(grid_x)
    # Only use urban raster names (not lcz_ prefixed — those come from lookup)
    urban_names = [n for n in feature_names if not n.startswith("lcz_")]
    pixel_rasters: dict[str, np.ndarray] = {name: rasters[name] for name in urban_names}

    # Add LCZ morphological parameters at every pixel if requested
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

    # Rebuild feature_names to include both urban and LCZ params
    lcz_prefixed = [f"lcz_{n}" for n in lcz_param_names]
    all_feature_names = sorted(urban_names) + sorted(lcz_prefixed)
    # Column order must match all_feature_names exactly — the training
    # matrix below is built via all_extracted[f] lookups in this same order.
    raster_list = [pixel_rasters[name] for name in all_feature_names]
    raster_stack = np.stack(raster_list, axis=-1).astype(np.float32)
    n_total_features = raster_stack.shape[-1]
    flat_features = raster_stack.reshape(-1, n_total_features)

    # ---- 9. Fit + predict per group ---------------------------------------
    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    unc_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": len(group_specs),
        "dtype": "float32",
        "crs": epsg,
        "transform": transform,
        "compress": "lzw",
        "nodata": np.nan,
    }
    unc_profile = dict(profile)  # uncertainty band
    all_importances: dict = {}
    all_cv_scores: dict = {}
    completed_groups: list = []
    pca_evr_list: list[float] = []

    with rasterio.open(out_path, "w", **profile) as dst, \
         rasterio.open(unc_path, "w", **unc_profile) as unc_dst:
        for band_idx, (label, sub) in enumerate(group_specs, start=1):
            # Build training matrix for this group
            train_rows = sub.to_pandas()
            X_list = []
            y_list = []
            for _, row in train_rows.iterrows():
                col = int((float(row["x"]) - grid_x[0]) / res_x)
                rrow = int((grid_y[0] - float(row["y"])) / res_y)
                col = max(0, min(col, len(grid_x) - 1))
                rrow = max(0, min(rrow, len(grid_y) - 1))
                idx = pix_to_idx.get((rrow, col))
                if idx is None:
                    continue
                feat = [all_extracted[f][idx] for f in all_feature_names]
                if any(np.isnan(feat)):
                    continue
                X_list.append(feat)
                y_list.append(float(row["var_interp"]))

            n_feat = len(all_feature_names)
            min_needed = max(min_samples, n_feat + 1) if ml_model == "ols" else min_samples
            if len(X_list) < min_needed:
                logger.warning(lcz_msg("interp_plus_few_stations", lang, group=str(label), n=len(X_list)))
                continue
            if len(X_list) < 5:
                logger.warning(lcz_msg("interp_plus_few_stations", lang, group=str(label), n=len(X_list)))

            X_train = np.array(X_list, dtype=np.float32)
            y_train = np.array(y_list, dtype=np.float32)

            # ---- PCA (optional) ----
            pca_obj = None
            if use_pca and n_pca_components != 0:
                X_train, flat_features_pca, pca_evr, pca_obj = _apply_pca(
                    X_train, flat_features, n_pca_components,
                )
                pca_evr_list = pca_evr
            else:
                flat_features_pca = flat_features

            # ---- Train model ----
            if ml_model == "rf":
                model = _train_model_rf(X_train, y_train, n_estimators=n_estimators)
                flat_pred = model.predict(flat_features_pca).astype(np.float32)
                # Uncertainty: std of individual tree predictions
                tree_preds = np.stack(
                    [tree.predict(flat_features_pca) for tree in model.estimators_],
                    axis=0,
                )
                flat_unc = np.std(tree_preds, axis=0).astype(np.float32)
                importances = dict(zip(
                    all_feature_names if pca_obj is None else [f"PC{i+1}" for i in range(X_train.shape[1])],
                    model.feature_importances_,
                ))
                all_importances[str(label)] = {k: float(v) for k, v in importances.items()}
                top5 = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]
                logger.info(
                    "Group '%s' RF top-5 features: %s",
                    label, [(f, f"{v:.3f}") for f, v in top5],
                )
            else:  # "ols"
                coeffs, r2 = _train_model_ols(X_train, y_train)
                flat_pred = _predict_ols(flat_features_pca, coeffs).astype(np.float32)
                # Uncertainty: residual standard error
                y_hat_train = _predict_ols(X_train, coeffs)
                residual_se = float(np.sqrt(np.mean((y_train - y_hat_train) ** 2)))
                flat_unc = np.full(flat_pred.shape, residual_se, dtype=np.float32)
                coef_names = ["intercept"] + (
                    all_feature_names if pca_obj is None
                    else [f"PC{i+1}" for i in range(X_train.shape[1])]
                )
                all_importances[str(label)] = {k: float(v) for k, v in zip(coef_names, coeffs)}
                all_cv_scores[str(label)] = {"r2": r2, "residual_se": residual_se}
                logger.info("Group '%s' OLS R² = %.3f, residual SE = %.3f", label, r2, residual_se)

            # Reshape to (ny, nx) and apply valid mask
            pred_grid = flat_pred.reshape(ny, nx)
            pred_grid = np.where(valid_mask, pred_grid, np.nan).astype(np.float32)
            unc_grid = flat_unc.reshape(ny, nx)
            unc_grid = np.where(valid_mask, unc_grid, np.nan).astype(np.float32)

            dst.write(pred_grid, band_idx)
            dst.set_band_description(band_idx, str(label))
            unc_dst.write(unc_grid, band_idx)
            unc_dst.set_band_description(band_idx, str(label))
            completed_groups.append(label)

    # ---- 10. Save if requested -------------------------------------------
    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "lcz4r_interp_map_plus.tif")
        with rasterio.open(out_path) as src, \
             rasterio.open(save_path, "w", **src.profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)
        logger.info(lcz_msg("interp_plus_saved", lang, path=os.path.abspath(save_path)))

    logger.info(lcz_msg("interp_plus_complete", lang, n=len(completed_groups), method=ml_model))
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

    Returns
    -------
    pl.DataFrame
        Columns: station, group, observed, predicted, residual, rmse, mae, r2.
    """
    # ---- 1. Input normalisation -------------------------------------------
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
        return pl.DataFrame()

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
        on=["latitude", "longitude"], how="left",
    )
    df = df.filter(pl.col("lcz").is_between(1, 17))

    # ---- 3. Download urban parameters ------------------------------------
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
        return pl.DataFrame()

    # Resample urban rasters onto the prediction grid (UTM, north-up)
    rasters = _resample_urban_to_grid(
        combined, transform, len(grid_x), len(grid_y), epsg, downloaded_vars,
    )
    if not rasters:
        return pl.DataFrame()

    # ---- 4. Extract features at all stations -----------------------------
    all_stations_xy = stations.select(["x", "y"]).unique().to_pandas()
    all_extracted = _extract_raster_values_at_points(
        rasters, grid_x, grid_y,
        all_stations_xy["x"].to_numpy(dtype=np.float64),
        all_stations_xy["y"].to_numpy(dtype=np.float64),
    )
    feature_names = sorted(all_extracted.keys())

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

    # ---- 6. Cross-validation per group -----------------------------------
    results = []
    for label, group_df in group_list:
        rows = group_df.to_pandas()
        X_all = []
        y_all = []
        station_names = []
        for _, row in rows.iterrows():
            col = int((float(row["x"]) - grid_x[0]) / res_x)
            rrow = int((grid_y[0] - float(row["y"])) / res_y)
            col = max(0, min(col, len(grid_x) - 1))
            rrow = max(0, min(rrow, len(grid_y) - 1))
            idx = pix_to_idx.get((rrow, col))
            if idx is None:
                continue
            feat = [all_extracted[f][idx] for f in feature_names]
            if any(np.isnan(feat)):
                continue
            X_all.append(feat)
            y_all.append(float(row["observed"]))
            station_names.append(str(row["station"]))

        if len(X_all) < 3:
            continue

        X_arr = np.array(X_all, dtype=np.float32)
        y_arr = np.array(y_all, dtype=np.float32)

        if LOOCV:
            # Leave-one-station-out
            for i in range(len(X_arr)):
                train_mask = np.arange(len(X_arr)) != i
                X_tr, y_tr = X_arr[train_mask], y_arr[train_mask]
                if len(X_tr) < 2:
                    continue
                if ml_model == "rf":
                    try:
                        from sklearn.ensemble import RandomForestRegressor
                        m = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                        m.fit(X_tr, y_tr)
                        pred = float(m.predict(X_arr[i:i+1])[0])
                    except ImportError:
                        continue
                else:
                    coeffs, _ = _train_model_ols(X_tr, y_tr)
                    pred = float(_predict_ols(X_arr[i:i+1], coeffs)[0])
                obs = float(y_arr[i])
                results.append({
                    "station": station_names[i],
                    "group": str(label),
                    "observed": obs,
                    "predicted": pred,
                    "residual": obs - pred,
                    "cv_method": "loocv",
                })
        else:
            # Random hold-out
            n = len(X_arr)
            perm = np.random.permutation(n)
            n_train = max(2, int(n * split_ratio))
            train_idx, test_idx = perm[:n_train], perm[n_train:]
            if len(test_idx) < 1 or len(train_idx) < 2:
                continue
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_te, y_te = X_arr[test_idx], y_arr[test_idx]
            if ml_model == "rf":
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    m = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                    m.fit(X_tr, y_tr)
                    preds = m.predict(X_te)
                except ImportError:
                    continue
            else:
                coeffs, _ = _train_model_ols(X_tr, y_tr)
                preds = _predict_ols(X_te, coeffs)
            for j in range(len(test_idx)):
                results.append({
                    "station": station_names[test_idx[j]],
                    "group": str(label),
                    "observed": float(y_te[j]),
                    "predicted": float(preds[j]),
                    "residual": float(y_te[j] - preds[j]),
                    "cv_method": "holdout",
                })

    if not results:
        return pl.DataFrame()

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
]
