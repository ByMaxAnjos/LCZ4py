"""
lcz_interp_eval.py — cross-validation of spatial interpolation.

Supports two validation modes:
- LOOCV=True  : leave-one-station-out per time step (true LOOCV)
- LOOCV=False : random hold-out split at the station level

Returns a Polars DataFrame with observed, predicted, residual, RMSE, and MAE
per time step.

Prediction during validation is done **at the held-out station coordinates**
rather than by rasterising the full grid and sampling the nearest pixel: it is
orders of magnitude cheaper, and it removes the grid-resolution bias from the
error statistics.  Time steps are evaluated independently, so they run in a
bounded ``ThreadPoolExecutor`` while results keep deterministic order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
from typing import Optional, Union

import numpy as np
import polars as pl
import rasterio

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, normalise_input_df, normalise_missing,
    select_by_date, extract_lcz_at_points, utm_epsg_for, OUTPUT_DIR,
)
from LCZ4py.local.lcz_krige import (
    krige_predict_points, VgModel,
    _HAS_PYK,  # noqa: F401 - public compatibility alias (was pykrige availability)
    HAS_PYKRIGE, DEFAULT_MAX_WORKERS,
)
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

_RESULT_SCHEMA = {
    "station": pl.String,
    "date": pl.Datetime("ns"),
    "observed": pl.Float64,
    "predicted": pl.Float64,
    "residual": pl.Float64,
    "rmse": pl.Float64,
    "mae": pl.Float64,
}


def _empty_frame() -> pl.DataFrame:
    """Empty result with the documented schema (never a schema-less frame)."""
    return pl.DataFrame(schema=_RESULT_SCHEMA)


def _stratified_split(
    df: pl.DataFrame, ratio: float, seed: Optional[int] = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Random hold-out split — single seeded draw to avoid train/test overlap."""
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {ratio!r}.")
    rng = np.random.default_rng(seed)
    rand = pl.Series("_rand", rng.random(len(df)))
    flagged = df.with_columns(rand.alias("_rand")).with_columns(
        (pl.col("_rand") <= ratio).alias("_is_train")
    )
    train = flagged.filter(pl.col("_is_train")).drop(["_rand", "_is_train"])
    test = flagged.filter(~pl.col("_is_train")).drop(["_rand", "_is_train"])
    return train, test


def _predict_at_stations(
    train_df, px: float, py: float, vg_model: VgModel, use_lcz: bool,
    n_neighbors: int = 16,
) -> float:
    """Kriging prediction at one point from a station snapshot."""
    pred, _var = krige_predict_points(
        train_df, "x", "y", "z", "lcz" if use_lcz else None,
        np.array([px]), np.array([py]),
        model=vg_model, enable_lcz_drift=use_lcz, n_neighbors=n_neighbors,
    )
    return float(pred[0])


def lcz_interp_eval(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
    data_frame,
    var: str = "",
    station_id: str = "",
    *,
    LOOCV: bool = True,
    split_ratio: float = 0.8,
    sp_res: float = 100.0,
    tp_res: str = "hour",
    vg_model: VgModel = "Sph",
    isave: bool = False,
    LCZinterp: bool = True,
    lang: str = "en",
    year=None, month=None, day=None, hour=None, start=None, end=None,
    seed: Optional[int] = 42,
    n_neighbors: int = 16,
    max_workers: Optional[int] = None,
) -> pl.DataFrame:
    """Evaluate interpolation accuracy via cross-validation.

    Parameters
    ----------
    x : str, PathLike, or rasterio dataset
        LCZ raster (defines the interpolation CRS and LCZ classes).
    data_frame : pd.DataFrame or pl.DataFrame
        Observation table with lat/lon/date/var/station columns.
    var : str
        Column name for the meteorological variable.
    station_id : str
        Column name for the station identifier.
    LOOCV : bool
        If True, use leave-one-station-out cross-validation per time step.
        If False, use a random hold-out split (``split_ratio`` controls size).
    split_ratio : float
        Fraction of stations used for training in hold-out mode.
    sp_res : float
        Spatial resolution in metres.  Retained for API compatibility; the
        validator predicts at station points, so no grid is rasterised.
    tp_res : {"hour", "day"}
        Temporal averaging frequency.
    vg_model : VgModel
        Variogram model — ``"Sph"``, ``"Exp"``, ``"Gau"``, or ``"Ste"``.
    isave : bool
        Save the result table to ``LCZ4r_output/lcz4r_interp_eval_result.csv``.
    LCZinterp : bool
        Use LCZ membership as External Drift Kriging term.
    lang : str
        Language for log messages — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.
    max_workers : int, optional
        Thread workers for independent time steps
        (default ``min(8, cpu_count)``).
    seed : int or None
        Seed for the hold-out split (deterministic by default).  ``None`` uses
        fresh entropy.
    n_neighbors : int
        Neighbours per local kriging system when the NumPy solver is used.

    Returns
    -------
    pl.DataFrame
        Columns: station, date, observed, predicted, residual, rmse, mae.
        Empty (with the same schema) when no fold could be evaluated.
    """
    if not LOOCV and not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio!r}.")
    if not np.isfinite(sp_res) or sp_res <= 0:
        raise ValueError(f"sp_res must be a positive resolution in metres, got {sp_res!r}.")
    if not HAS_PYKRIGE:
        logger.warning(
            "pykrige is not installed — cross-validation uses the NumPy "
            "ordinary-kriging fallback."
        )

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
        logger.warning("No valid observations after filtering — nothing to validate.")
        return _empty_frame()

    ds = load_lcz_raster(x)
    stations = df.unique(subset=["latitude", "longitude"])
    lcz_vals, mask = extract_lcz_at_points(
        ds, stations["longitude"].to_numpy(), stations["latitude"].to_numpy()
    )
    stations = stations.with_columns(
        pl.Series(name="lcz", values=np.where(mask, lcz_vals, 0)).cast(int)
    )

    epsg = utm_epsg_for(ds)
    from pyproj import Transformer
    tx = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    xs, ys = tx.transform(stations["longitude"].to_numpy(), stations["latitude"].to_numpy())
    stations = stations.with_columns(x=xs, y=ys)

    df = df.join(
        stations.select(["latitude", "longitude", "lcz", "x", "y"]),
        on=["latitude", "longitude"], how="left",
    )
    df = df.filter(pl.col("lcz").is_between(1, 17))
    if df.is_empty():
        logger.warning(
            "No stations fall inside LCZ classes 1–17 of the supplied map — "
            "nothing to validate."
        )
        return _empty_frame()

    use_lcz = LCZinterp and df["lcz"].n_unique() >= 2
    if LCZinterp and not use_lcz:
        logger.warning(
            "LCZinterp=True but fewer than 2 LCZ classes are present at the "
            "stations — evaluating Ordinary Kriging."
        )

    freq = {"hour": "1h", "day": "1d"}.get(tp_res, "1h")
    if tp_res not in ("hour", "day"):
        logger.warning("Unknown tp_res '%s' — defaulting to hourly.", tp_res)
    # Keep per-station resolution so LOOCV can leave each station out
    avg = df.sort("date").group_by_dynamic("date", every=freq, group_by="station").agg([
        pl.col("var_interp").mean().alias("observed"),
        pl.col("x").first(),
        pl.col("y").first(),
        pl.col("lcz").first(),
    ])

    dates = avg["date"].unique().sort().to_list()
    slices = [avg.filter(pl.col("date") == d) for d in dates]
    rng_seed = seed

    def _evaluate_group(payload) -> list[dict]:
        date_val, time_slice, group_idx = payload
        rows: list[dict] = []
        if len(time_slice) < 2:
            return rows  # need at least 2 stations to interpolate

        if LOOCV:
            # Leave-one-station-out: for each station, train on the rest
            records = time_slice.to_dicts()
            for i, test_row in enumerate(records):
                train_rows = records[:i] + records[i + 1:]
                if len(train_rows) < 1:
                    continue
                train_pd = pl.DataFrame(train_rows).rename({"observed": "z"}).to_pandas()
                try:
                    pred = _predict_at_stations(
                        train_pd, float(test_row["x"]), float(test_row["y"]),
                        vg_model, use_lcz, n_neighbors=n_neighbors,
                    )
                except Exception as exc:  # noqa: BLE001 - fold-level failure
                    logger.warning(
                        "LOOCV failed for station %s at %s: %s: %s",
                        test_row["station"], date_val, type(exc).__name__, exc,
                    )
                    continue
                obs = float(test_row["observed"])
                if not np.isfinite(pred):
                    continue
                rows.append({
                    "station": test_row["station"],
                    "date": date_val,
                    "observed": obs,
                    "predicted": pred,
                    "residual": obs - pred,
                })
        else:
            # Random hold-out split at the station level
            group_seed = None if rng_seed is None else int(rng_seed) + int(group_idx)
            train, test = _stratified_split(time_slice, split_ratio, seed=group_seed)
            if test.is_empty() or len(train) < 2:
                return rows
            test_records = test.to_dicts()
            try:
                pred, _var = krige_predict_points(
                    train.rename({"observed": "z"}).to_pandas(),
                    "x", "y", "z", "lcz" if use_lcz else None,
                    np.array([float(r["x"]) for r in test_records]),
                    np.array([float(r["y"]) for r in test_records]),
                    model=vg_model, enable_lcz_drift=use_lcz,
                    n_neighbors=n_neighbors,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hold-out kriging failed at %s: %s: %s",
                               date_val, type(exc).__name__, exc)
                return rows
            for j, row in enumerate(test_records):
                p = float(pred[j])
                if not np.isfinite(p):
                    continue
                obs = float(row["observed"])
                rows.append({
                    "station": row["station"],
                    "date": date_val,
                    "observed": obs,
                    "predicted": p,
                    "residual": obs - p,
                })
        return rows

    payloads = [(d, s, i) for i, (d, s) in enumerate(zip(dates, slices))]
    workers = DEFAULT_MAX_WORKERS if max_workers is None else max(1, int(max_workers))
    workers = min(workers, max(1, len(payloads)))

    if workers > 1 and len(payloads) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            per_group = list(ex.map(_evaluate_group, payloads))  # deterministic order
    else:
        per_group = [_evaluate_group(p) for p in payloads]

    results = [row for group in per_group for row in group]
    empty_folds = sum(1 for group in per_group if not group)
    if empty_folds:
        logger.warning(
            "%d of %d time step(s) produced no evaluation rows (too few "
            "stations or all predictions non-finite).", empty_folds, len(payloads),
        )

    if not results:
        logger.warning(
            "Cross-validation produced no results — check station counts per "
            "time step (at least 2 co-located observations are required)."
        )
        return _empty_frame()

    out = pl.DataFrame(results).with_columns(pl.col("date").cast(pl.Datetime("ns")))
    out = out.with_columns([
        ((pl.col("residual") ** 2).mean().over("date").sqrt()).alias("rmse"),
        (pl.col("residual").abs().mean().over("date")).alias("mae"),
    ])

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DIR, "lcz4r_interp_eval_result.csv")
        out.write_csv(csv_path)
        logger.info(lcz_msg("save_output_path", lang, path=os.path.abspath(csv_path)))

    return out


__all__ = ["lcz_interp_eval"]
