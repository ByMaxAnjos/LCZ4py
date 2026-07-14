"""
lcz_interp_eval.py — cross-validation of spatial kriging interpolation.

Supports two validation modes:
- LOOCV=True  : leave-one-station-out per time step (true LOOCV)
- LOOCV=False : random hold-out split at the station level

Returns a Polars DataFrame with observed, predicted, residual, RMSE, and MAE
per time step.
"""

from __future__ import annotations
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
from LCZ4py.local.lcz_krige import krige_predict, VgModel, _HAS_PYK
from LCZ4py.local.lcz_interp_map import _make_grid_memsafe
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)


def _stratified_split(
    df: pl.DataFrame, ratio: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Random hold-out split — single random draw to avoid train/test overlap."""
    rand = pl.Series("_rand", np.random.rand(len(df)))
    flagged = df.with_columns(rand.alias("_rand")).with_columns(
        (pl.col("_rand") <= ratio).alias("_is_train")
    )
    train = flagged.filter(pl.col("_is_train")).drop(["_rand", "_is_train"])
    test = flagged.filter(~pl.col("_is_train")).drop(["_rand", "_is_train"])
    return train, test


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
) -> pl.DataFrame:
    """Evaluate kriging interpolation accuracy via cross-validation.

    Parameters
    ----------
    x : str, PathLike, or rasterio dataset
        LCZ raster (defines the interpolation grid extent and CRS).
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
        Spatial resolution of the kriging grid in metres.
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

    Returns
    -------
    pl.DataFrame
        Columns: station, date, observed, predicted, residual, rmse, mae.
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

    ds = load_lcz_raster(x)
    stations = df.unique(subset=["latitude", "longitude"])
    lcz_vals, mask = extract_lcz_at_points(
        ds, stations["longitude"].to_numpy(), stations["latitude"].to_numpy()
    )
    stations = stations.with_columns(
        pl.Series(name="lcz", values=np.where(mask, lcz_vals, 0)).cast(int)
    )

    grid_x, grid_y, _transform, epsg = _make_grid_memsafe(ds, sp_res)
    from pyproj import Transformer
    tx = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    xs, ys = tx.transform(stations["longitude"].to_numpy(), stations["latitude"].to_numpy())
    stations = stations.with_columns(x=xs, y=ys)

    df = df.join(
        stations.select(["latitude", "longitude", "lcz", "x", "y"]),
        on=["latitude", "longitude"], how="left",
    )
    df = df.filter(pl.col("lcz").is_between(1, 17))

    use_lcz = LCZinterp and df["lcz"].n_unique() >= 2

    freq = {"hour": "1h", "day": "1d"}.get(tp_res, "1h")
    # Keep per-station resolution so LOOCV can leave each station out
    avg = df.sort("date").group_by_dynamic("date", every=freq, group_by="station").agg([
        pl.col("var_interp").mean().alias("observed"),
        pl.col("x").first(),
        pl.col("y").first(),
        pl.col("lcz").first(),
    ])

    results = []
    for date_val in avg["date"].unique().sort():
        time_slice = avg.filter(pl.col("date") == date_val)
        if len(time_slice) < 2:
            continue  # need at least 2 stations to interpolate

        if LOOCV:
            # Leave-one-station-out: for each station, train on the rest
            for i in range(len(time_slice)):
                test_row = time_slice[i]
                train_rows = pl.concat([time_slice[:i], time_slice[i + 1:]])
                if len(train_rows) < 1:
                    continue
                try:
                    res = krige_predict(
                        train_rows.rename({"observed": "z"}).to_pandas(),
                        "x", "y", "z", "lcz" if use_lcz else None,
                        grid_x, grid_y, vg_model, use_lcz,
                    )
                    from scipy.spatial import cKDTree
                    gx, gy = np.meshgrid(grid_x, grid_y)
                    tree = cKDTree(np.column_stack([gx.ravel(), gy.ravel()]))
                    _, idx = tree.query([[test_row["x"][0], test_row["y"][0]]])
                    pred = float(res.prediction.ravel()[idx[0]])
                    obs = float(test_row["observed"][0])
                    results.append({
                        "station": test_row["station"][0],
                        "date": date_val,
                        "observed": obs,
                        "predicted": pred,
                        "residual": obs - pred,
                    })
                except Exception as exc:
                    logger.warning("LOOCV failed for station %s at %s: %s",
                                   test_row["station"][0], date_val, exc)
        else:
            # Random hold-out split at the station level
            train, test = _stratified_split(time_slice, split_ratio)
            if test.is_empty() or len(train) < 2:
                continue
            try:
                res = krige_predict(
                    train.rename({"observed": "z"}).to_pandas(),
                    "x", "y", "z", "lcz" if use_lcz else None,
                    grid_x, grid_y, vg_model, use_lcz,
                )
                from scipy.spatial import cKDTree
                gx, gy = np.meshgrid(grid_x, grid_y)
                tree = cKDTree(np.column_stack([gx.ravel(), gy.ravel()]))
                _, idx = tree.query(np.column_stack([test["x"].to_numpy(), test["y"].to_numpy()]))
                pred = res.prediction.ravel()[idx]
                for j, row in enumerate(test.iter_rows(named=True)):
                    obs = float(row["observed"])
                    p = float(pred[j])
                    results.append({
                        "station": row["station"],
                        "date": date_val,
                        "observed": obs,
                        "predicted": p,
                        "residual": obs - p,
                    })
            except Exception as exc:
                logger.warning("Hold-out kriging failed at %s: %s", date_val, exc)

    if not results:
        return pl.DataFrame()

    out = pl.DataFrame(results)
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
