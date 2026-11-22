"""
lcz_interp_map.py — memory-safe parallel spatial interpolation.

Bands are interpolated independently per time step (or per ``by=`` group) and
written to a GeoTIFF sequentially in deterministic task order, so large time
stacks never materialise in memory at once.

Parallelism uses a bounded ``ProcessPoolExecutor`` when the inputs are
picklable (Arrow IPC buffers, NumPy grids) and automatically degrades to a
bounded ``ThreadPoolExecutor`` — or to serial execution — if process start-up
or pickling fails for any reason.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import os
import tempfile
from typing import Optional, Union

import numpy as np
import pyarrow as pa
import polars as pl
import rasterio
from rasterio.transform import Affine

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, normalise_input_df, normalise_missing,
    select_by_date, extract_lcz_at_points, utm_epsg_for, OUTPUT_DIR,
    add_by_column, by_sorted_groups,
)
from LCZ4py.local.lcz_krige import (
    krige_predict, VgModel,
    _HAS_PYK,  # noqa: F401 - public compatibility alias (was pykrige availability)
    rbf_predict, idw_predict,
    HAS_PYKRIGE, DEFAULT_MAX_WORKERS,
)
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

# Hard ceiling on output grid size; beyond this a float32 band alone needs
# >1.5 GB and the request is almost certainly a mistake.
_MAX_GRID_PIXELS = 400_000_000


def _make_grid_memsafe(
    ds: rasterio.io.DatasetReader, sp_res_m: float = 100.0
) -> tuple[np.ndarray, np.ndarray, Affine, str]:
    """Build a north-up UTM grid covering the dataset extent at exactly ``sp_res_m``.

    ``grid_y`` is descending (row 0 = north) to match the north-up affine
    transform returned alongside it — interpolation results indexed by
    ``(grid_y, grid_x)`` can be written to a GeoTIFF with this transform
    with no vertical flip.  Coordinates are pixel centres spaced exactly
    ``sp_res_m`` apart, matching the transform's declared pixel size (a
    plain ``linspace`` over the bounds would silently round the spacing).
    """
    from pyproj import Transformer

    if not np.isfinite(sp_res_m) or sp_res_m <= 0:
        raise ValueError(f"sp_res must be a positive resolution in metres, got {sp_res_m!r}.")

    epsg = utm_epsg_for(ds)
    bounds = ds.bounds
    tx = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    xmin, ymin = tx.transform(bounds.left, bounds.bottom)
    xmax, ymax = tx.transform(bounds.right, bounds.top)
    if not all(np.isfinite([xmin, ymin, xmax, ymax])) or xmax <= xmin or ymax <= ymin:
        raise ValueError(
            "Could not derive a valid projected extent from the LCZ raster bounds."
        )

    nx = max(1, int(np.ceil((xmax - xmin) / sp_res_m)))
    ny = max(1, int(np.ceil((ymax - ymin) / sp_res_m)))
    if nx * ny > _MAX_GRID_PIXELS:
        raise ValueError(
            f"Requested grid {nx}x{ny} = {nx * ny:,} pixels exceeds the "
            f"{_MAX_GRID_PIXELS:,}-pixel safety limit. Increase sp_res."
        )
    if nx * ny > 25_000_000:
        logger.warning(
            "Large interpolation grid: %d x %d = %s pixels per band.",
            nx, ny, f"{nx * ny:,}",
        )

    grid_x = xmin + sp_res_m * (np.arange(nx) + 0.5)
    grid_y = ymax - sp_res_m * (np.arange(ny) + 0.5)
    transform = Affine(sp_res_m, 0, xmin, 0, -sp_res_m, ymax)
    return grid_x, grid_y, transform, epsg


def _resample_lcz_to_grid(
    ds: rasterio.io.DatasetReader,
    transform: Affine,
    width: int,
    height: int,
    epsg: str,
) -> np.ndarray:
    """Resample the LCZ raster onto the prediction grid via nearest-neighbour warp.

    Reuses the exact same ``transform``/shape as the output raster (from
    :func:`_make_grid_memsafe`) so LCZ classes line up pixel-for-pixel with
    the interpolated values — both for the External Drift Kriging term and
    for masking the output to the LCZ map's actual footprint.
    """
    from rasterio.warp import reproject, Resampling

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid target grid shape: {height} x {width}.")
    out = np.zeros((height, width), dtype=np.int32)
    reproject(
        source=rasterio.band(ds, 1),
        destination=out,
        dst_transform=transform,
        dst_crs=epsg,
        resampling=Resampling.nearest,
    )
    return out


def _interp_worker(
    arrow_bytes: bytes,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    model: VgModel,
    use_lcz: bool,
    method: str = "rbf",
    lcz_grid: Optional[np.ndarray] = None,
    max_workers: Optional[int] = None,
) -> np.ndarray:
    """Worker: deserialise Arrow bytes, run interpolation, return (ny, nx) grid.

    Never raises for data reasons: an empty task or a failed solver yields a
    NaN grid plus a warning, so one bad time step cannot abort the whole stack.
    """
    ny, nx = len(grid_y), len(grid_x)
    try:
        table = pa.ipc.open_stream(arrow_bytes).read_all()
        df = table.to_pandas()
    except Exception as exc:  # noqa: BLE001 - corrupt/unexpected payload
        logger.warning("Could not decode interpolation task (%s: %s).",
                       type(exc).__name__, exc)
        return np.full((ny, nx), np.nan, dtype=np.float32)

    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    z = df["z"].to_numpy(dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not ok.all():
        x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 2:
        logger.warning(
            "Skipping a time step: %d valid station value(s) (need >= 2).", len(x),
        )
        return np.full((ny, nx), np.nan, dtype=np.float32)

    try:
        if method == "krige":
            res = krige_predict(
                df, "x", "y", "z", "lcz" if use_lcz else None,
                grid_x, grid_y, model, use_lcz,
                lcz_grid=lcz_grid if use_lcz else None,
                max_workers=max_workers,
            )
            from scipy.ndimage import uniform_filter
            out = uniform_filter(res.prediction.astype(np.float32), size=3, mode="nearest")
            # Keep pixels that were nodata before smoothing.
            out = np.where(np.isfinite(res.prediction), out, np.nan).astype(np.float32)
            return out
        if method == "idw":
            return idw_predict(x, y, z, grid_x, grid_y)
        return rbf_predict(x, y, z, grid_x, grid_y)
    except Exception as exc:  # noqa: BLE001 - engine-level failure
        logger.warning("Interpolation failed (%s): %s: %s",
                       method, type(exc).__name__, exc)
        return np.full((ny, nx), np.nan, dtype=np.float32)


# backward-compat alias used by lcz_anomaly_map
_krige_worker = _interp_worker


def _pack_task(sub_pd, label, use_lcz):
    """Serialize one station snapshot to Arrow IPC for the workers."""
    table = pa.Table.from_pandas(sub_pd, preserve_index=False)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return (sink.getvalue().to_pybytes(), label, use_lcz)


def _run_tasks(tasks, grid_x, grid_y, model, method, lcz_grid, workers,
               use_processes: bool = False):
    """Execute interpolation tasks with thread → (optional) process → serial.

    Threads are the default: they cannot dead-lock on a missing
    ``if __name__ == "__main__"`` guard (the classic macOS/Windows spawn trap
    that makes a process pool re-import the caller's script recursively), and
    NumPy/SciPy/pykrige release the GIL for the heavy numerical work.
    ``use_processes=True`` opts into a process pool for CPU-bound workloads,
    falling back to threads and then to serial if start-up or pickling fails.

    Returns a list of ``(ny, nx)`` grids in the same order as ``tasks``.
    """
    # Only parallelise inside the worker when it runs alone; otherwise the
    # band-level pool already saturates the cores.
    inner_workers = None if len(tasks) == 1 else 1

    def _collect(executor):
        futures = [
            executor.submit(
                _interp_worker, payload, grid_x, grid_y, model, use_lcz, method,
                (lcz_grid if use_lcz else None), inner_workers,
            )
            for payload, _label, use_lcz in tasks
        ]
        return [f.result() for f in futures]

    if workers > 1 and len(tasks) > 1 and use_processes:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                return _collect(ex)
        except Exception as exc:  # noqa: BLE001 - pickling / spawn / OOM
            logger.warning(
                "Process-pool interpolation unavailable (%s: %s) — retrying "
                "with threads.", type(exc).__name__, exc,
            )

    if workers > 1 and len(tasks) > 1:
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                return _collect(ex)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Thread-pool interpolation failed (%s: %s) — running serially.",
                type(exc).__name__, exc,
            )

    return [
        _interp_worker(payload, grid_x, grid_y, model, use_lcz, method,
                       (lcz_grid if use_lcz else None), inner_workers)
        for payload, _label, use_lcz in tasks
    ]


def lcz_interp_map(
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
    use_processes: bool = False,
    max_workers: Optional[int] = None,
) -> Optional[str]:
    """Spatially interpolate a meteorological variable onto a regular grid.

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
        Output grid spatial resolution in metres.
    tp_res : {"hour", "day"}
        Temporal averaging frequency.
    by : str, optional
        Aggregate all time steps within each temporal group before interpolation,
        producing one raster band per group instead of per time step.
        Options: ``"year"``, ``"season"``, ``"seasonyear"``, ``"month"``,
        ``"monthyear"``, ``"weekday"``, ``"weekend"``, ``"site"``,
        ``"daylight"``, ``"dst"``.  ``None`` = one band per time step (default).
    method : {"krige", "rbf", "idw"}
        Interpolation method.

        ``"krige"`` (default) — Ordinary Kriging, or External Drift Kriging
        using LCZ class as a covariate when ``LCZinterp=True`` and at least
        2 distinct LCZ classes are present.  Matches the original R
        ``lcz_interp_map`` behavior.  Uses ``pykrige`` when installed and
        transparently falls back to the NumPy ordinary-kriging solver when it
        is not.

        ``"rbf"`` — Radial Basis Function thin-plate spline via
        ``scipy.interpolate.RBFInterpolator``.  Smooth, no variogram fitting,
        works well with 10–200 stations.  Ignores ``LCZinterp`` — LCZ classes
        are not used.

        ``"idw"`` — Inverse Distance Weighting.  Fast, deterministic, no
        extra dependencies.  Ignores ``LCZinterp`` — LCZ classes are not used.
    vg_model : VgModel
        Variogram model for ``method="krige"`` only —
        ``"Sph"``, ``"Exp"``, ``"Gau"``, or ``"Ste"``.
    isave : bool
        Copy the temporary GeoTIFF to ``LCZ4r_output/lcz4r_interp_map.tif``.
    LCZinterp : bool
        For ``method="krige"``: use LCZ class as External Drift Kriging term
        when at least 2 distinct LCZ classes are present.
    n_jobs : int
        Worker processes (``-1`` = all CPU cores).
    lang : str
        Language for log messages — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.
    max_workers : int, optional
        Explicit worker bound; overrides ``n_jobs`` when given
        (default ``min(8, cpu_count)``).
    use_processes : bool
        Opt into a ``ProcessPoolExecutor`` for the per-time-step work.  Threads
        are used by default because they cannot dead-lock on a script without
        an ``if __name__ == "__main__"`` guard; process start-up failures fall
        back to threads and then to serial execution.

    Returns
    -------
    str or None
        Path to the (temporary) output GeoTIFF, or ``None`` if no data.
    """
    if method not in ("krige", "rbf", "idw"):
        raise ValueError(f"Unknown method '{method}'. Choose from 'krige', 'rbf', 'idw'.")
    if method == "krige" and not HAS_PYKRIGE:
        logger.warning(
            "pykrige is not installed — method='krige' will use the NumPy "
            "ordinary-kriging fallback. Install pykrige for the reference "
            "implementation: pip install pykrige"
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
        logger.warning("No valid observations after filtering — nothing to interpolate.")
        return None

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
    if df.is_empty():
        logger.warning(
            "No stations fall inside LCZ classes 1–17 of the supplied map — "
            "nothing to interpolate."
        )
        return None

    # Evaluate LCZinterp flag once against all station classes (not per-row)
    use_lcz = LCZinterp and df["lcz"].n_unique() >= 2
    if method != "krige" and use_lcz:
        logger.warning(lcz_msg("lcz_ignored_non_krige", lang, method=method))
    elif method == "krige" and LCZinterp and not use_lcz:
        logger.warning(
            "LCZinterp=True but fewer than 2 LCZ classes are present at the "
            "stations — falling back to Ordinary Kriging."
        )

    # Always resample LCZ onto the grid: used as the drift term when use_lcz,
    # and always used to mask the output to the LCZ map's actual footprint
    # (pixels outside its coverage — nodata/class 0 — are not part of the map,
    # even though they fall inside its rectangular bounding box).
    lcz_grid = _resample_lcz_to_grid(ds, transform, len(grid_x), len(grid_y), epsg)
    valid_mask = (lcz_grid >= 1) & (lcz_grid <= 17)

    if not valid_mask.any():
        logger.warning(
            "The resampled LCZ grid has no valid classes — check that the LCZ "
            "raster CRS matches its bounds."
        )
        return None

    tasks: list[tuple[bytes, object, bool]] = []
    if by is not None:
        lat_mean = float(df["latitude"].mean())
        lon_mean = float(df["longitude"].mean())
        df = add_by_column(df, by, lat=lat_mean, lon=lon_mean)
        # Average var_interp per station per group → one kriging task per group
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
        for g in groups:
            sub = group_avg.filter(pl.col("_by") == g)
            if len(sub) < 2:
                logger.warning("Group '%s': only %d station(s) — skipped.", g, len(sub))
                continue
            sub_pd = sub.select([
                pl.col("x"), pl.col("y"),
                pl.col("var_interp").alias("z"), pl.col("lcz"),
            ]).to_pandas()
            tasks.append(_pack_task(sub_pd, g, use_lcz))
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
        for t in avg["date"].unique().sort():
            sub = avg.filter(pl.col("date") == t)
            if len(sub) < 2:
                continue
            sub_pd = sub.select([
                pl.col("x"), pl.col("y"),
                pl.col("var_interp").alias("z"), pl.col("lcz"),
            ]).to_pandas()
            tasks.append(_pack_task(sub_pd, t, use_lcz))

    if not tasks:
        logger.warning("No time step had at least 2 stations — nothing to interpolate.")
        return None

    if max_workers is not None:
        workers = max(1, int(max_workers))
    elif n_jobs == -1:
        workers = DEFAULT_MAX_WORKERS
    else:
        workers = max(1, int(n_jobs))
    workers = min(workers, len(tasks))

    grids = _run_tasks(tasks, grid_x, grid_y, vg_model, method, lcz_grid, workers,
                       use_processes=use_processes)

    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    profile = {
        "driver": "GTiff", "height": len(grid_y), "width": len(grid_x),
        "count": len(tasks), "dtype": "float32", "crs": epsg,
        "transform": transform, "compress": "lzw", "nodata": np.nan,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        for band_idx, (grid, (_payload, label, _u)) in enumerate(zip(grids, tasks), start=1):
            band = np.where(valid_mask, grid, np.nan).astype(np.float32)
            dst.write(band, band_idx)
            dst.set_band_description(band_idx, str(label))

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "lcz4r_interp_map.tif")
        with rasterio.open(out_path) as src, rasterio.open(save_path, "w", **src.profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)
            for i in range(1, src.count + 1):
                desc = src.descriptions[i - 1]
                if desc:
                    dst.set_band_description(i, desc)
        logger.info(lcz_msg("save_output_path", lang, path=os.path.abspath(save_path)))

    return out_path


__all__ = [
    "lcz_interp_map", "_make_grid_memsafe", "_resample_lcz_to_grid",
    "_interp_worker", "_krige_worker",
]
