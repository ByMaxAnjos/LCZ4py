"""
lcz_anomaly_map.py — spatial anomaly map via parallel kriging.

Pre-computes per-station anomalies in Polars, then distributes kriging
to a ProcessPoolExecutor using Arrow zero-copy serialization.
Output is a multi-band GeoTIFF (one band per time step).
"""

from __future__ import annotations
import logging
import os
import tempfile
from typing import Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import polars as pl
import rasterio

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, normalise_input_df, normalise_missing,
    select_by_date, extract_lcz_at_points, OUTPUT_DIR,
    add_by_column, by_sorted_groups,
)
from LCZ4py.local.lcz_krige import krige_predict, VgModel, _HAS_PYK
from LCZ4py.local.lcz_interp_map import _make_grid_memsafe, _interp_worker, _resample_lcz_to_grid
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)


def lcz_anomaly_map(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
    data_frame,
    var: str = "",
    station_id: str = "",
    *,
    sp_res: float = 100.0,
    tp_res: str = "hour",
    by: Optional[str] = None,
    method: str = "krige",
    vg_model: VgModel = "Sph",
    isave: bool = False,
    LCZinterp: bool = True,
    n_jobs: int = -1,
    lang: str = "en",
    year=None, month=None, day=None, hour=None, start=None, end=None,
) -> Optional[str]:
    """Compute and save an anomaly raster stack via parallel kriging.

    Anomaly is defined as each station's value minus the overall spatial mean
    for that time step. Each time step produces one band in the output GeoTIFF.

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
    sp_res : float
        Spatial resolution of the output grid in metres.
    tp_res : {"hour", "day"}
        Temporal averaging frequency.
    by : str, optional
        Aggregate all time steps within each temporal group before kriging,
        producing one anomaly band per group instead of per time step.
        Anomaly is computed relative to the within-group mean.
        Options: ``"year"``, ``"season"``, ``"seasonyear"``, ``"month"``,
        ``"monthyear"``, ``"weekday"``, ``"weekend"``, ``"site"``,
        ``"daylight"``, ``"dst"``.  ``None`` = one band per time step (default).
    method : {"krige", "rbf", "idw"}
        Interpolation method.

        ``"krige"`` (default) — Ordinary Kriging, or External Drift Kriging
        using LCZ class as a covariate when ``LCZinterp=True`` and at least
        2 distinct LCZ classes are present.  Requires ``pykrige``.

        ``"rbf"`` / ``"idw"`` — faster alternatives that ignore
        ``LCZinterp``; LCZ classes are not used.
    vg_model : VgModel
        Variogram model — ``"Sph"``, ``"Exp"``, ``"Gau"``, or ``"Ste"``.
    isave : bool
        Copy the temporary GeoTIFF to ``LCZ4r_output/lcz4r_anomaly_map.tif``.
    LCZinterp : bool
        Use LCZ class membership as External Drift Kriging term when at least
        2 distinct LCZ classes are present in the stations.
    n_jobs : int
        Worker processes (``-1`` = all CPU cores).
    lang : str
        Language for log messages — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.

    Returns
    -------
    str or None
        Path to the (temporary) output GeoTIFF, or ``None`` if no data.
    """
    if method == "krige" and not _HAS_PYK:
        raise ImportError("pykrige is required for method='krige'. Install with: pip install pykrige")

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

    grid_x, grid_y, transform, epsg = _make_grid_memsafe(ds, sp_res)
    from pyproj import Transformer
    tx = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    xs, ys = tx.transform(stations["longitude"].to_numpy(), stations["latitude"].to_numpy())
    stations = stations.with_columns(x=xs, y=ys)

    df = df.join(
        stations.select(["latitude", "longitude", "lcz", "x", "y"]),
        on=["latitude", "longitude"], how="left",
    )
    df = df.filter(pl.col("lcz").is_between(1, 17))

    # Evaluate LCZinterp flag once against all station classes (not per-row)
    use_lcz = LCZinterp and df["lcz"].n_unique() >= 2
    if method != "krige" and use_lcz:
        logger.warning(lcz_msg("lcz_ignored_non_krige", lang, method=method))
    # Always resample LCZ onto the grid: used as the drift term when use_lcz,
    # and always used to mask the output to the LCZ map's actual footprint
    # (pixels outside its coverage — nodata/class 0 — are not part of the map,
    # even though they fall inside its rectangular bounding box).
    lcz_grid = _resample_lcz_to_grid(ds, transform, len(grid_x), len(grid_y), epsg)
    valid_mask = (lcz_grid >= 1) & (lcz_grid <= 17)

    if by is not None:
        lat_mean = float(df["latitude"].mean())
        lon_mean = float(df["longitude"].mean())
        df = add_by_column(df, by, lat=lat_mean, lon=lon_mean)
        # Average per station per group, then compute within-group anomaly
        group_avg = (
            df.group_by(["station", "_by"])
              .agg([
                  pl.col("var_interp").mean().alias("var_mean"),
                  pl.col("x").first(),
                  pl.col("y").first(),
                  pl.col("lcz").first(),
              ])
        )
        group_means = (
            group_avg.group_by("_by")
                     .agg(pl.col("var_mean").mean().alias("group_mean"))
        )
        group_avg = group_avg.join(group_means, on="_by").with_columns(
            (pl.col("var_mean") - pl.col("group_mean")).alias("anomaly")
        )
        groups = by_sorted_groups(df["_by"], by)
        tasks = []
        for g in groups:
            sub = group_avg.filter(pl.col("_by") == g)
            if len(sub) < 2:
                continue
            sub_pd = sub.select([
                pl.col("x"), pl.col("y"),
                pl.col("anomaly").alias("z"), pl.col("lcz"),
            ]).to_pandas()
            table = pa.Table.from_pandas(sub_pd)
            sink = pa.BufferOutputStream()
            writer = pa.ipc.new_stream(sink, table.schema)
            writer.write_table(table)
            writer.close()
            tasks.append((sink.getvalue().to_pybytes(), g, use_lcz))
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
        # Anomaly = per-station value minus per-time-step spatial mean
        step_means = avg.group_by("date").agg(pl.col("var_interp").mean().alias("step_mean"))
        avg = avg.join(step_means, on="date").with_columns(
            (pl.col("var_interp") - pl.col("step_mean")).alias("anomaly")
        )
        tasks = []
        for t in avg["date"].unique().sort():
            sub = avg.filter(pl.col("date") == t)
            if len(sub) < 2:
                continue
            sub_pd = sub.select([
                pl.col("x"), pl.col("y"),
                pl.col("anomaly").alias("z"), pl.col("lcz"),
            ]).to_pandas()
            table = pa.Table.from_pandas(sub_pd)
            sink = pa.BufferOutputStream()
            writer = pa.ipc.new_stream(sink, table.schema)
            writer.write_table(table)
            writer.close()
            tasks.append((sink.getvalue().to_pybytes(), t, use_lcz))

    if not tasks:
        return None

    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    profile = {
        "driver": "GTiff", "height": len(grid_y), "width": len(grid_x),
        "count": len(tasks), "dtype": "float32", "crs": epsg,
        "transform": transform, "compress": "lzw", "nodata": np.nan,
    }
    max_workers = (os.cpu_count() or 4) if n_jobs == -1 else n_jobs

    with rasterio.open(out_path, "w", **profile) as dst:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_interp_worker, t[0], grid_x, grid_y, vg_model, t[2], method, lcz_grid): t[1]
                for t in tasks
            }
            for band_idx, future in enumerate(as_completed(futures), start=1):
                date_val = futures[future]
                band = np.where(valid_mask, future.result(), np.nan).astype(np.float32)
                dst.write(band, band_idx)
                dst.set_band_description(band_idx, str(date_val))

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "lcz4r_anomaly_map.tif")
        with rasterio.open(out_path) as src, rasterio.open(save_path, "w", **src.profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)
        logger.info(lcz_msg("save_output_path", lang, path=os.path.abspath(save_path)))

    return out_path


__all__ = ["lcz_anomaly_map"]
