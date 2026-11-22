"""
LCZ4r adaptive crop / mask strategy — Python equivalent of the R
``.lcz_crop_mask_adaptive`` dispatcher.

Three-tier strategy based on study-area size:

* **Tier 1** (< 500 000 km²) — standard crop + mask
* **Tier 2** (500k – 5M km²) — simplify boundary, then tiled parallel mask
* **Tier 3** (> 5 000 000 km²) — N×N tile grid → parallel crop+mask → mosaic

Parallelism uses :mod:`concurrent.futures` (always available, fork+spawn
safe) and :mod:`multiprocessing` is **not** required. All tile work runs on a
bounded :class:`~concurrent.futures.ThreadPoolExecutor` — rasterio releases
the GIL while decompressing, so threads keep memory low and avoid per-worker
GDAL handle duplication (a real problem for ``/vsicurl/`` sources).

Mosaic ordering is deterministic: tiles are collected in grid order (not
completion order) and merged bottom-up/left-right, so repeated runs on the
same input produce byte-identical output.

Public entry point
------------------
>>> from LCZ4py._internal.adaptive_crop_mask import lcz_crop_mask_adaptive
>>> lcz_ras = lcz_crop_mask_adaptive(lcz_global, study_area, lang="en")
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping, shape

from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

# ── Thresholds (km²) ──────────────────────────────────────────────────────────
LARGE_AREA_KM2: float = 500_000.0      # Tier 2 starts here
XLARGE_AREA_KM2: float = 5_000_000.0   # Tier 3 starts here
TILE_SIDE: int = 3                     # 3×3 = 9 tiles
TIER2_TILE_SIDE: int = 4               # simplified boundary → 4×4 tiles

# Upper bound on parallel raster reads. Beyond ~8 concurrent ``/vsicurl/``
# windows the remote server (Zenodo) starts rate-limiting, and local disks
# are already saturated, so more threads only add contention.
_MAX_PARALLEL_WORKERS: int = 8


def default_max_workers() -> int:
    """Default thread-pool size: ``min(8, cpu_count)``."""
    return max(1, min(_MAX_PARALLEL_WORKERS, os.cpu_count() or 4))


def coerce_max_workers(max_workers: Optional[int]) -> int:
    """Validate/normalise a user-supplied ``max_workers`` value."""
    if max_workers is None:
        return default_max_workers()
    try:
        n = int(max_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_workers must be a positive integer") from exc
    if n < 1:
        raise ValueError("max_workers must be a positive integer")
    return n


# ── Geometry helpers ──────────────────────────────────────────────────────────

def ensure_wgs84(
    study_area: gpd.GeoDataFrame,
    *,
    context: str = "study_area",
    repair: bool = True,
) -> gpd.GeoDataFrame:
    """Return ``study_area`` in EPSG:4326 with valid, non-empty polygons.

    Degenerate geometries (self-intersections from poorly digitised
    boundaries, empty MultiPolygons from a failed intersection, ...) make
    ``rasterio.mask`` fail with an opaque PROJ/GEOS error, so they are repaired
    up-front instead of silently producing a nodata mask.
    """
    if not isinstance(study_area, gpd.GeoDataFrame):
        raise TypeError(
            f"{context} must be a GeoDataFrame with polygon geometry, got "
            f"{type(study_area).__name__}"
        )
    if study_area.crs is None:
        # A GeoDataFrame without a CRS is almost always lon/lat in this
        # package (geocoded boundaries, hand-built ROIs). Assume WGS84 rather
        # than failing, but say so.
        logger.warning("%s has no CRS; assuming EPSG:4326.", context)
        study_area = study_area.set_crs("EPSG:4326")
    elif not _crs_is_wgs84(study_area.crs):
        study_area = study_area.to_crs("EPSG:4326")

    study_area = study_area[study_area.geometry.notna()]
    study_area = study_area[~study_area.geometry.is_empty]
    if study_area.empty:
        raise ValueError(f"{context} contains no non-empty geometry.")

    if repair and not bool(study_area.geometry.is_valid.all()):
        bad = int((~study_area.geometry.is_valid).sum())
        logger.warning("%s has %d invalid geometries; repairing them.", context, bad)
        repaired = study_area.copy()
        if hasattr(repaired.geometry, "make_valid"):
            repaired["geometry"] = repaired.geometry.make_valid()
        else:  # pragma: no cover - shapely < 2.0
            repaired["geometry"] = repaired.geometry.buffer(0)
        repaired = repaired[~repaired.geometry.is_empty]
        if repaired.empty:
            raise ValueError(
                f"{context} contains only invalid geometry that could not be repaired."
            )
        study_area = repaired

    return study_area


def _crs_is_wgs84(crs) -> bool:
    """True when ``crs`` is EPSG:4326 (tolerant of PROJ string spellings)."""
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg == 4326:
        return True
    try:
        return crs == rasterio.crs.CRS.from_epsg(4326)
    except Exception:
        return False


def _validate_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Validate a WGS84 bounding box; raise a clear error when degenerate."""
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox must be 4 finite numbers, got {bbox!r}") from exc
    if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
        raise ValueError(f"bbox contains non-finite values: {bbox!r}")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            f"bbox is degenerate (must be xmin<xmax and ymin<ymax): {bbox!r}"
        )
    if xmin < -180 or xmax > 180 or ymin < -90 or ymax > 90:
        raise ValueError(f"bbox is outside WGS84 bounds: {bbox!r}")
    return xmin, ymin, xmax, ymax


# ── Area helper ───────────────────────────────────────────────────────────────

def study_area_km2(study_area: gpd.GeoDataFrame) -> float:
    """Compute study-area size in km² (geodesic, EPSG:4326 input)."""
    study_area = ensure_wgs84(study_area, context="study_area")

    # Equal-area projection is exact for whole continents but undefined at the
    # poles / antimeridian, so fall back to a latitude-corrected degree
    # approximation instead of crashing on circumpolar ROIs.
    try:
        eq_area = study_area.to_crs("EPSG:6933")  # cylindrical equal-area
        return float(eq_area.geometry.union_all().area) / 1e6
    except Exception as exc:  # noqa: BLE001 - PROJ failure on extreme latitudes
        logger.debug("Equal-area reprojection failed (%s); using degree approximation.", exc)

    geom = study_area.geometry.union_all()
    minx, miny, maxx, maxy = geom.bounds
    lat_c = math.radians((miny + maxy) / 2.0)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * max(math.cos(lat_c), 1e-6)
    approx = (maxx - minx) * km_per_deg_lon * (maxy - miny) * km_per_deg_lat
    return float(max(approx, 0.0))


# ── Fork-safe raster helper ────────────────────────────────────────────────────

@dataclass
class DiskInfo:
    """Local-file handle for a raster, fork-safe."""
    path: str
    is_temp: bool


def ensure_on_disk(raster_path: str) -> DiskInfo:
    """Return a local file path for the given raster.

    Rasterio datasets accept both local paths and remote ``/vsicurl/`` URLs.
    The returned path is safe to pass into worker processes because each
    worker will open its own read-only handle.
    """
    if raster_path and raster_path != "":
        return DiskInfo(path=raster_path, is_temp=False)

    # Caller didn't pass a path; nothing we can do.
    raise ValueError("raster_path must be a non-empty string")


@contextmanager
def _open_4326(raster_path: str):
    """Open a raster, transparently reprojecting to EPSG:4326 if needed.

    Mirrors the WarpedVRT guard in ``_lcz_map_engine._stream_and_clip_cog``
    so masking against WGS84 study-area geometry is always correct.
    """
    with rasterio.open(raster_path) as src:
        if src.crs is not None and not _crs_is_wgs84(src.crs):
            with WarpedVRT(src, crs="EPSG:4326") as vrt:
                yield vrt
        else:
            yield src


def _write_raster_atomic(path: str, array: np.ndarray, profile: dict) -> str:
    """Write ``array`` to ``path`` atomically (temp file + ``os.replace``)."""
    dst_dir = os.path.dirname(path) or "."
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=dst_dir)
    os.close(fd)
    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(array)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return path


def _temp_raster_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    return path


def _geometry_overlaps_raster(src, study_area: gpd.GeoDataFrame) -> bool:
    """Fast bbox-level overlap test between a study area and an open raster."""
    try:
        left, bottom, right, top = src.bounds
    except Exception:
        return True  # can't tell → let rasterio try
    sxmin, symin, sxmax, symax = study_area.total_bounds
    return not (sxmax < left or sxmin > right or symax < bottom or symin > top)


def _profile_from(src, array: np.ndarray, transform) -> dict:
    """Build a single-band output profile from an open dataset + array."""
    return {
        "driver": "GTiff",
        "dtype": src.dtypes[0] if src.count else array.dtype.name,
        "nodata": src.nodata if src.nodata is not None else 0,
        "width": int(array.shape[-1]),
        "height": int(array.shape[-2]),
        "count": 1,
        "crs": src.crs or "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
        "BIGTIFF": "IF_SAFER",
    }


# ── Tile-grid builder ──────────────────────────────────────────────────────────

def make_tile_grid(
    bbox: tuple[float, float, float, float],
    n_side: int = TILE_SIDE,
) -> gpd.GeoDataFrame:
    """Create an N×N grid of bounding-box polygons (WGS84).

    Tiles are emitted row-major from south-west, which keeps downstream
    ordering deterministic.
    """
    try:
        n_side = int(n_side)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_side must be a positive integer") from exc
    if n_side < 1:
        raise ValueError("n_side must be a positive integer")

    xmin, ymin, xmax, ymax = _validate_bbox(bbox)
    xs = np.linspace(xmin, xmax, n_side + 1)
    ys = np.linspace(ymin, ymax, n_side + 1)

    tiles = []
    for j in range(n_side):
        for i in range(n_side):
            tiles.append(box(xs[i], ys[j], xs[i + 1], ys[j + 1]))
    return gpd.GeoDataFrame(geometry=tiles, crs="EPSG:4326")


# ── Per-tile worker ───────────────────────────────────────────────────────────

def _process_tile(args: tuple[str, dict, dict, dict]) -> Optional[str]:
    """Worker: crop+mask a single tile. Returns the path of the temp output.

    Parameters are passed as plain dicts so the worker is picklable under
    both fork and spawn start methods (the parent uses threads, but keeping
    this pickle-friendly costs nothing and keeps the door open for process
    pools).
    """
    raster_path, tile_geom, study_geom, meta = args
    tile_geom = shape(tile_geom)
    study_geom = shape(study_geom)

    # 1. Intersect tile with study area (saves the mask step on empty tiles).
    try:
        inter = tile_geom.intersection(study_geom)
    except Exception as exc:  # noqa: BLE001 - GEOS topology failure
        logger.warning("Skipping tile: tile/study-area intersection failed (%s).", exc)
        return None
    if inter.is_empty:
        return None

    # 2. Open raster windowed on the tile bounds — never loads the full file.
    tb = tile_geom.bounds  # (xmin, ymin, xmax, ymax)
    try:
        with _open_4326(raster_path) as src:
            # src is always EPSG:4326 here (native or WarpedVRT), same as tb.
            try:
                win = from_bounds(*tb, src.transform)
                win = win.intersection(Window(0, 0, src.width, src.height))
            except rasterio.errors.WindowError:
                # Tile lies outside this raster's extent (e.g. a study area
                # spanning two products) — a normal no-data case.
                logger.debug("Tile outside raster extent; skipping.")
                return None
            if win.width <= 0 or win.height <= 0:
                return None

            win = Window(
                int(round(win.col_off)), int(round(win.row_off)),
                max(1, int(round(win.width))), max(1, int(round(win.height))),
            )
            win = win.intersection(Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                return None

            # Read the windowed data (single band for LCZ maps).
            fill = src.nodata if src.nodata is not None else 0
            data = src.read(1, window=win, boundless=True, fill_value=fill)

            # Update transform for the windowed read.
            win_transform = rasterio.windows.transform(win, src.transform)
            out_meta = src.meta.copy()
            out_meta.update({
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": win_transform,
            })

            # 3. Mask against the study geometry clipped to this tile.
            try:
                with rasterio.io.MemoryFile() as memfile, memfile.open(**out_meta) as mem_ds:
                    mem_ds.write(data, 1)
                    masked, masked_transform = rio_mask(
                        mem_ds,
                        [mapping(inter)], crop=True, all_touched=False, nodata=0,
                    )
            except Exception:  # noqa: BLE001
                # Window entirely outside the geometry (or collinear edge) —
                # not an error, just nothing to contribute to the mosaic.
                logger.debug("Tile produced no masked pixels; skipping.")
                return None

            tile_meta = _profile_from(src, masked, masked_transform)
    except Exception as exc:  # noqa: BLE001 - one bad tile must not kill the run
        logger.warning("Tile read failed (%s: %s); skipping tile.", type(exc).__name__, exc)
        return None

    # 4. Write tile to a tempfile for the parent to mosaic.
    return _write_raster_atomic(_temp_raster_path(), masked, tile_meta)


# ── Tier-3 tiled crop / mask ──────────────────────────────────────────────────

def _crop_tiled(
    raster_path: str,
    study_area: gpd.GeoDataFrame,
    lang: str = "en",
    verbose: bool = True,
    max_workers: Optional[int] = None,
    n_side: int = TILE_SIDE,
    simplify: bool = False,
) -> Optional[str]:
    """Parallel tiled crop+mask. Returns path to a merged GeoTIFF."""
    study_area = ensure_wgs84(study_area, context="study_area")

    if simplify:
        simplified = study_area.copy()
        simplified["geometry"] = simplified.geometry.simplify(
            tolerance=0.005, preserve_topology=True
        )
        simplified = simplified[~simplified.geometry.is_empty]
        if simplified.empty:
            raise ValueError("Study area collapsed to nothing after simplification.")
        study_area = simplified

    bbox = _validate_bbox(tuple(study_area.total_bounds))
    grid = make_tile_grid(bbox, n_side=n_side)

    # Identify non-empty tiles via vector intersection.
    union_geom = study_area.geometry.union_all()
    if union_geom.is_empty:
        return None
    non_empty_mask = grid.geometry.intersects(union_geom)
    non_empty_idx = np.where(np.asarray(non_empty_mask))[0]
    n_tiles = int(len(non_empty_idx))
    if n_tiles == 0:
        return None

    workers = coerce_max_workers(max_workers)
    workers = min(workers, n_tiles)

    if verbose:
        logger.info(lcz_msg("tiling_start", lang, n=n_tiles, cores=workers))

    # The per-tile worker derives its own output profile from the opened
    # source, so no template raster open is needed here (that used to cost an
    # extra HTTP round-trip for remote ``/vsicurl/`` sources).
    template_meta: dict = {}

    args_list = [
        (raster_path, mapping(grid.geometry.iloc[int(idx)]), mapping(union_geom), template_meta)
        for idx in non_empty_idx
    ]

    # Ordered collection: index-aligned with ``args_list`` so the mosaic is
    # deterministic regardless of completion order.
    results: list[Optional[str]] = [None] * len(args_list)
    valid_paths: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lcz-tile") as ex:
            futures = [ex.submit(_process_tile, a) for a in args_list]
            for pos, fut in enumerate(futures):
                try:
                    results[pos] = fut.result()
                except Exception as exc:  # noqa: BLE001 - never lose other tiles
                    logger.warning(
                        "Tile %d failed (%s: %s); skipping.", pos, type(exc).__name__, exc
                    )
                    results[pos] = None
        valid_paths = [p for p in results if p]
    finally:
        pass

    keep_path: Optional[str] = None
    try:
        if not valid_paths:
            return None

        if verbose:
            logger.info(lcz_msg("tiling_mosaic", lang, n=len(valid_paths)))

        if len(valid_paths) == 1:
            # Single surviving tile: hand it to the caller, don't delete it.
            keep_path = valid_paths[0]
            return keep_path

        # Mosaic the valid tiles using rasterio + numpy (avoids the rio.merge
        # deprecation path).
        sources = [rasterio.open(p) for p in valid_paths]
        try:
            mosaic, mosaic_transform = _merge_arrays(sources, valid_paths)
            out_meta = sources[0].meta.copy()
            out_meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": mosaic_transform,
                "compress": "lzw",
                "BIGTIFF": "IF_SAFER",
            })
            out_path = _temp_raster_path()
            keep_path = out_path
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(mosaic)
            return out_path
        finally:
            for s in sources:
                s.close()
    finally:
        # Tile temporaries were consumed by the mosaic; delete them so
        # repeated large-area calls don't fill the temp directory.
        for p in valid_paths:
            if p == keep_path:
                continue
            try:
                os.remove(p)
            except OSError:
                pass


def _merge_arrays(sources: list, paths: Optional[list[str]] = None) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Merge multiple rasters into one (deterministic first-valid-wins).

    Sources are ordered by position (south→north, west→east) rather than by
    completion order, and an already-written pixel is never overwritten, so
    the same tiles always produce the same mosaic.
    """
    if not sources:
        raise ValueError("No rasters to merge.")
    if paths is not None and len(paths) != len(sources):
        raise ValueError("paths and sources must have the same length")

    # Deterministic ordering by (bottom, left) edge.
    order = sorted(range(len(sources)), key=lambda i: (sources[i].bounds.bottom, sources[i].bounds.left))
    ordered = [sources[i] for i in order]

    minx = min(s.bounds.left for s in ordered)
    miny = min(s.bounds.bottom for s in ordered)
    maxx = max(s.bounds.right for s in ordered)
    maxy = max(s.bounds.top for s in ordered)

    res_x = min(abs(s.res[0]) for s in ordered) or 1.0
    res_y = min(abs(s.res[1]) for s in ordered) or 1.0

    width = max(1, int(round((maxx - minx) / res_x)))
    height = max(1, int(round((maxy - miny) / res_y)))
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    count = ordered[0].count
    dtype = ordered[0].dtypes[0]
    nodata = ordered[0].nodata
    fill = nodata if nodata is not None else 0

    out = np.full((count, height, width), fill, dtype=dtype)
    filled = np.zeros((height, width), dtype=bool)

    for s in ordered:
        win = from_bounds(*s.bounds, transform)
        col0 = int(round(win.col_off))
        row0 = int(round(win.row_off))
        if col0 >= width or row0 >= height or col0 < -(s.width) or row0 < -(s.height):
            continue
        chunk = s.read(window=Window(0, 0, s.width, s.height), boundless=False)
        h = min(chunk.shape[1], height - row0)
        w = min(chunk.shape[2], width - col0)
        if h <= 0 or w <= 0:
            continue
        # Trim negative offsets (tile partially west/south of the mosaic).
        src_r0 = max(0, -row0)
        src_c0 = max(0, -col0)
        dst_r0 = max(0, row0)
        dst_c0 = max(0, col0)
        h = min(chunk.shape[1] - src_r0, height - dst_r0)
        w = min(chunk.shape[2] - src_c0, width - dst_c0)
        if h <= 0 or w <= 0:
            continue

        sub = chunk[:, src_r0:src_r0 + h, src_c0:src_c0 + w]
        target = out[:, dst_r0:dst_r0 + h, dst_c0:dst_c0 + w]
        target_filled = filled[dst_r0:dst_r0 + h, dst_c0:dst_c0 + w]

        if nodata is None:
            valid = np.ones(sub.shape[1:], dtype=bool)
        else:
            valid = sub[0] != nodata
        write_mask = valid & ~target_filled
        if not write_mask.any():
            continue
        for band in range(count):
            np.copyto(target[band], sub[band], where=write_mask)
        filled[dst_r0:dst_r0 + h, dst_c0:dst_c0 + w] |= write_mask

    return out, transform


# ── Adaptive entry point ──────────────────────────────────────────────────────

def lcz_crop_mask_adaptive(
    raster_path: str,
    study_area: gpd.GeoDataFrame,
    lang: str = "en",
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> Optional[str]:
    """Adaptive crop and mask dispatcher.

    Parameters
    ----------
    raster_path : str
        Path (local or ``/vsicurl/...``) to the global LCZ raster.
    study_area : gpd.GeoDataFrame
        Study area polygon in WGS84 (other CRSs are reprojected).
    lang : str
        Language code for progress messages.
    verbose : bool
        If True, emit progress messages via :mod:`logging`.
    max_workers : int, optional
        Thread-pool size for tiled crop/mask work. Defaults to
        ``min(8, os.cpu_count())``.

    Returns
    -------
    str or None
        Path to the cropped/masked GeoTIFF, or ``None`` if the region lies
        outside the raster extent.
    """
    if not raster_path:
        raise ValueError("raster_path must be a non-empty string")
    workers = coerce_max_workers(max_workers)

    study_area = ensure_wgs84(study_area, context="study_area")
    area_km2 = study_area_km2(study_area)

    # ── Tier 3: tiled parallel ────────────────────────────────────────────────
    if area_km2 > XLARGE_AREA_KM2:
        if verbose:
            logger.info(lcz_msg("large_area_tiling", lang,
                                area=round(area_km2 / 1e6, 1)))
        result = _crop_tiled(
            raster_path, study_area, lang=lang, verbose=verbose,
            max_workers=workers, n_side=TILE_SIDE,
        )
        if result is None:
            return None
        return result

    # ── Tier 2: simplify + parallel tiled mask ────────────────────────────────
    if area_km2 > LARGE_AREA_KM2:
        if verbose:
            logger.info(lcz_msg(
                "large_area_simplify", lang,
                area=f"{round(area_km2):,}",
            ))
        result = _crop_tiled(
            raster_path, study_area, lang=lang, verbose=verbose,
            max_workers=workers, n_side=TIER2_TILE_SIDE, simplify=True,
        )
        if result is None:
            return None
        return result

    # ── Tier 1: standard ─────────────────────────────────────────────────────
    with _open_4326(raster_path) as src:
        # A study area entirely outside the raster is a normal, non-fatal
        # outcome (e.g. a European map asked for a Brazilian city) — report it
        # as "no data" rather than an opaque rasterio window error.
        if not _geometry_overlaps_raster(src, study_area):
            logger.info("Study area does not overlap the raster extent.")
            return None
        try:
            masked, transform = rio_mask(
                src, study_area.geometry.apply(mapping).tolist(),
                crop=True, all_touched=False, nodata=src.nodata if src.nodata is not None else 0,
            )
        except ValueError as exc:
            if "do not overlap" in str(exc):
                logger.info("Study area does not overlap the raster extent.")
                return None
            raise RuntimeError(lcz_msg("large_raster_error", lang)) from exc
        except Exception as exc:
            raise RuntimeError(lcz_msg("large_raster_error", lang)) from exc
        out_meta = _profile_from(src, masked, transform)

    out_path = _temp_raster_path()
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(masked)
    return out_path


__all__ = [
    "LARGE_AREA_KM2",
    "XLARGE_AREA_KM2",
    "TILE_SIDE",
    "TIER2_TILE_SIDE",
    "study_area_km2",
    "make_tile_grid",
    "lcz_crop_mask_adaptive",
    "default_max_workers",
    "coerce_max_workers",
    "ensure_wgs84",
    "ensure_on_disk",
    "DiskInfo",
]
