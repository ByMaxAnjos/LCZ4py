"""
lcz_anomaly_map.py — spatial anomaly map via parallel kriging.

Pre-computes per-station anomalies in Polars, then distributes interpolation
over a thread pool using Arrow zero-copy serialization. Output is a multi-band
GeoTIFF (one band per time step), written in task order so band indices are
deterministic regardless of worker completion order.

Threads are the default because ``ProcessPoolExecutor`` requires an importable
``__main__`` under the macOS/Windows *spawn* start method, which ``python -c``,
the REPL and many notebooks do not have (the process pool used to die with
``BrokenProcessPool`` and silently redo everything serially). Pass
``executor="process"`` to opt in; the fallback chain is
process → thread → serial, so a half-ordered stack is never written.
"""

from __future__ import annotations
import logging
import os
import tempfile
from typing import Any, Optional, Union

import numpy as np
import pyarrow as pa
import polars as pl
import rasterio

from LCZ4py._internal.lcz_ts_utils import (
    load_lcz_raster, normalise_input_df, normalise_missing,
    select_by_date, assign_station_lcz, OUTPUT_DIR, require_nonempty,
    resolve_workers, add_by_column, by_sorted_groups,
)
from LCZ4py.local.lcz_krige import VgModel, _HAS_PYK
from LCZ4py.local.lcz_interp_map import (
    _make_grid_memsafe, _interp_worker, _resample_lcz_to_grid,
)
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

_VALID_METHODS = ("krige", "rbf", "idw")
_MAX_GRID_PIXELS = 50_000_000  # ~200 MB float32 per band

# Worker-process state, populated once by _init_worker instead of pickling the
# full prediction grid and LCZ raster with every task.
_WORKER_STATE: dict[str, Any] = {}


def _init_worker(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    vg_model: str,
    use_lcz: bool,
    method: str,
    lcz_grid: Optional[np.ndarray],
) -> None:  # pragma: no cover - runs in child processes
    _WORKER_STATE.update(
        grid_x=grid_x, grid_y=grid_y, vg_model=vg_model,
        use_lcz=use_lcz, method=method, lcz_grid=lcz_grid,
    )


def _worker_entry(payload: bytes) -> np.ndarray:  # pragma: no cover - child
    state = _WORKER_STATE
    return _interp_worker(
        payload,
        state["grid_x"],
        state["grid_y"],
        state["vg_model"],
        state["use_lcz"],
        state["method"],
        state["lcz_grid"],
    )


def _arrow_payload(sub: pl.DataFrame) -> bytes:
    """Serialise a station table to Arrow IPC bytes for the worker."""
    table = pa.Table.from_pandas(sub.select([
        pl.col("x"), pl.col("y"),
        pl.col("anomaly").alias("z"), pl.col("lcz"),
    ]).to_pandas())
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return sink.getvalue().to_pybytes()


def _main_importable() -> bool:
    """True when ``__main__`` is a real file (so spawn can re-import it).

    ``python -c``, the REPL and some notebook kernels have no importable
    ``__main__`` file, which is exactly what makes ``ProcessPoolExecutor``
    fail with ``BrokenProcessPool`` on macOS/Windows spawn.
    """
    import __main__

    main_file = getattr(__main__, "__file__", None)
    try:
        return bool(main_file) and os.path.exists(str(main_file))
    except OSError:  # pragma: no cover - exotic __main__ objects
        return False


def _write_band(
    dst: rasterio.io.DatasetWriter,
    band_idx: int,
    label: str,
    arr: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Mask to the LCZ footprint and write one band (ordered)."""
    band = np.where(valid_mask & np.isfinite(arr), arr, np.nan).astype(np.float32)
    dst.write(band, band_idx)
    dst.set_band_description(band_idx, label)


def _interpolate_into(
    dst: rasterio.io.DatasetWriter,
    payloads: list[bytes],
    labels: list[str],
    *,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    vg_model: str,
    use_lcz: bool,
    method: str,
    lcz_grid: np.ndarray,
    valid_mask: np.ndarray,
    n_workers: int,
    executor: str,
) -> int:
    """Interpolate every payload and write bands 1..N in task order.

    Execution strategy (safe by default):

    1. ``executor="thread"`` (default) → ``ThreadPoolExecutor``. No pickling,
       no spawn, safe from ``python -c`` / notebooks / unguarded scripts.
    2. ``executor="process"`` → ``ProcessPoolExecutor`` **only** when
       ``__main__`` is an importable file; otherwise it logs and uses threads.
    3. Any pool failure falls back to threads, and finally to a serial loop.

    Bands are always written from index 1, so a failed attempt cannot leave a
    half-ordered stack behind.
    """
    n_tasks = len(payloads)

    def _run_pool(pool) -> int:
        futures = [pool.submit(_worker_entry, p) for p in payloads]
        for band_idx, (label, future) in enumerate(zip(labels, futures), start=1):
            _write_band(dst, band_idx, label, future.result(), valid_mask)
        return n_tasks

    # Worker threads read the shared module-level state (no pickling).
    _init_worker(grid_x, grid_y, vg_model, use_lcz, method, lcz_grid)

    if n_workers > 1 and n_tasks > 1:
        if executor == "process":
            if not _main_importable():
                logger.warning(
                    "executor='process' requested, but __main__ is not an "
                    "importable file (python -c / REPL / notebook kernel) — "
                    "using threads to avoid BrokenProcessPool."
                )
            else:
                try:
                    from concurrent.futures import ProcessPoolExecutor

                    with ProcessPoolExecutor(
                        max_workers=min(n_workers, n_tasks),
                        initializer=_init_worker,
                        initargs=(grid_x, grid_y, vg_model, use_lcz, method, lcz_grid),
                    ) as pool:
                        return _run_pool(pool)
                except Exception as exc:  # noqa: BLE001 - BrokenProcessPool & friends
                    logger.warning(
                        "Process pool failed (%s: %s); retrying with threads.",
                        type(exc).__name__, exc,
                    )

        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(n_workers, n_tasks)) as pool:
                return _run_pool(pool)
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.warning(
                "Thread pool failed (%s: %s); computing serially.",
                type(exc).__name__, exc,
            )

    # Serial path (also the normal path for a single task).
    for band_idx, (label, payload) in enumerate(zip(labels, payloads), start=1):
        arr = _interp_worker(
            payload, grid_x, grid_y, vg_model, use_lcz, method, lcz_grid,
        )
        _write_band(dst, band_idx, label, arr, valid_mask)
    return n_tasks



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
    max_workers: Optional[int] = None,
    executor: str = "thread",
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
        Spatial resolution of the output grid in metres (must be > 0).
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
        Worker processes (``-1`` = all CPU cores, capped at 8).
    lang : str
        Language for log messages — ``"en"``, ``"pt"``, ``"es"``, or ``"zh"``.
    year, month, day, hour : int or sequence, optional
        Date/time component filters.
    start, end : str or datetime, optional
        Inclusive date range bounds.
    max_workers : int, optional
        Explicit worker count, overriding ``n_jobs`` (clamped to 1-8).  Added
        at the end of the signature for backward compatibility.
    executor : {"thread", "process"}
        Parallel backend. ``"thread"`` (default) uses
        :class:`concurrent.futures.ThreadPoolExecutor`, which needs no
        pickling/spawn and is safe from ``python -c``, the REPL, notebooks and
        scripts without an ``if __name__ == "__main__"`` guard.
        ``"process"`` uses a process pool **only** when ``__main__`` is an
        importable file (required by the macOS/Windows *spawn* start method);
        otherwise it logs a warning and uses threads. Any pool failure
        degrades to threads, then to a serial loop.

    Returns
    -------
    str or None
        Path to the (temporary) output GeoTIFF, or ``None`` when fewer than
        two stations share any time step (nothing can be interpolated).

    Raises
    ------
    ImportError
        If ``method="krige"`` and pykrige is not installed.
    ValueError
        On an unknown method, a non-positive ``sp_res``, missing columns, or
        when no station maps to LCZ class 1-17.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    if method == "krige" and not _HAS_PYK:
        raise ImportError("pykrige is required for method='krige'. Install with: pip install pykrige")
    if not np.isfinite(sp_res) or sp_res <= 0:
        raise ValueError(f"sp_res must be a positive number of metres, got {sp_res!r}")
    if tp_res not in ("hour", "day"):
        raise ValueError(f"tp_res must be 'hour' or 'day', got {tp_res!r}")
    if vg_model not in ("Sph", "Exp", "Gau", "Ste"):
        raise ValueError(
            f"vg_model must be one of ('Sph', 'Exp', 'Gau', 'Ste'), got {vg_model!r}"
        )
    if executor not in ("thread", "process"):
        raise ValueError(f"executor must be 'thread' or 'process', got {executor!r}")
    if data_frame is None:
        raise ValueError("data_frame is required.")

    df = normalise_input_df(data_frame, var=var, station_id=station_id)
    df = select_by_date(
        normalise_missing(df, ("var_interp",)),
        year=year, month=month, day=day, hour=hour, start=start, end=end,
    )
    df = df.filter(pl.col("var_interp").is_not_null())
    require_nonempty(df, "observations")

    ds = load_lcz_raster(x)
    df = assign_station_lcz(df, ds)
    require_nonempty(df, "observations at stations inside the LCZ raster")

    grid_x, grid_y, transform, epsg = _make_grid_memsafe(ds, float(sp_res))
    n_grid = int(len(grid_x) * len(grid_y))
    if n_grid > _MAX_GRID_PIXELS:
        logger.warning(
            "Interpolation grid is %d x %d (%.1f M pixels, %.0f MB per band) — "
            "increase sp_res to reduce memory.",
            len(grid_x), len(grid_y), n_grid / 1e6, n_grid * 4 / 1e6,
        )

    from pyproj import Transformer
    tx = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    xs, ys = tx.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    df = df.with_columns([
        pl.Series("x", np.asarray(xs, dtype=np.float64)),
        pl.Series("y", np.asarray(ys, dtype=np.float64)),
    ])

    # Evaluate LCZinterp flag once against all station classes (not per-row)
    use_lcz = LCZinterp and df["lcz"].n_unique() >= 2
    if method != "krige" and use_lcz:
        logger.warning(lcz_msg("lcz_ignored_non_krige", lang, method=method))
        use_lcz = False
    # Always resample LCZ onto the grid: used as the drift term when use_lcz,
    # and always used to mask the output to the LCZ map's actual footprint
    # (pixels outside its coverage — nodata/class 0 — are not part of the map,
    # even though they fall inside its rectangular bounding box).
    lcz_grid = _resample_lcz_to_grid(ds, transform, len(grid_x), len(grid_y), epsg)
    valid_mask = (lcz_grid >= 1) & (lcz_grid <= 17)
    if not valid_mask.any():
        raise ValueError(
            "The LCZ raster reprojected onto the interpolation grid has no "
            "pixels in classes 1-17 — check its CRS and coverage."
        )

    payloads: list[bytes] = []
    labels: list[str] = []

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
        for g in by_sorted_groups(df["_by"], by):
            sub = group_avg.filter(pl.col("_by") == g).drop_nulls("anomaly")
            if len(sub) < 2:
                logger.warning(
                    "Skipping by-group %r: %d station(s) with data (need >= 2 "
                    "for interpolation).", g, len(sub),
                )
                continue
            payloads.append(_arrow_payload(sub))
            labels.append(str(g))
    else:
        freq = {"hour": "1h", "day": "1d"}[tp_res]
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
        ).drop_nulls("anomaly")

        partitions = avg.partition_by("date", as_dict=True)
        for key in sorted(partitions, key=lambda k: (k[0] if isinstance(k, tuple) else k)):
            sub = partitions[key]
            if len(sub) < 2:
                logger.warning(
                    "Skipping time step %s: %d station(s) with data (need >= 2 "
                    "for interpolation).", key, len(sub),
                )
                continue
            payloads.append(_arrow_payload(sub))
            labels.append(str(key[0] if isinstance(key, tuple) else key))

    if not payloads:
        logger.warning(
            "No time step or group has at least 2 stations with data — "
            "nothing to interpolate."
        )
        return None

    n_workers = resolve_workers(n_jobs, max_workers)
    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    profile = {
        "driver": "GTiff", "height": len(grid_y), "width": len(grid_x),
        "count": len(payloads), "dtype": "float32", "crs": epsg,
        "transform": transform, "compress": "lzw", "nodata": np.nan,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        _interpolate_into(
            dst, payloads, labels,
            grid_x=grid_x, grid_y=grid_y, vg_model=vg_model, use_lcz=use_lcz,
            method=method, lcz_grid=lcz_grid, valid_mask=valid_mask,
            n_workers=n_workers, executor=executor,
        )
        dst.update_tags(
            method=method, vg_model=str(vg_model), variable=var or "var",
            anomaly_reference=("within-group mean" if by is not None
                               else "per-time-step spatial mean"),
            sp_res_m=str(sp_res),
        )

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "lcz4r_anomaly_map.tif")
        with rasterio.open(out_path) as src, rasterio.open(save_path, "w", **src.profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)
                desc = src.descriptions[i - 1]
                if desc:
                    dst.set_band_description(i, desc)
            dst.update_tags(**src.tags())
        logger.info(lcz_msg("save_output_path", lang, path=os.path.abspath(save_path)))

    return out_path


__all__ = ["lcz_anomaly_map"]
