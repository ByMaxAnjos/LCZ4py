"""
LCZ4r adaptive crop / mask strategy — Python equivalent of the R
``.lcz_crop_mask_adaptive`` dispatcher.

Three-tier strategy based on study-area size:

* **Tier 1** (< 500 000 km²) — standard crop + mask
* **Tier 2** (500k – 5M km²) — simplify boundary + multi-thread mask
* **Tier 3** (> 5 000 000 km²) — N×N tile grid → parallel crop+mask → mosaic

Parallelism uses :mod:`concurrent.futures` (always available, fork+spawn
safe) and :mod:`multiprocessing` is **not** required.

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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping, shape

from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

# ── Thresholds (km²) ──────────────────────────────────────────────────────────
LARGE_AREA_KM2: float = 500_000.0      # Tier 2 starts here
XLARGE_AREA_KM2: float = 5_000_000.0   # Tier 3 starts here
TILE_SIDE: int = 3                     # 3×3 = 9 tiles


# ── Area helper ───────────────────────────────────────────────────────────────

def study_area_km2(study_area: gpd.GeoDataFrame) -> float:
    """Compute study-area size in km² (geodesic, EPSG:4326 input)."""
    if study_area.crs is None or study_area.crs.to_epsg() != 4326:
        study_area = study_area.to_crs(4326)
    # Reproject to an equal-area CRS for accurate area measurement.
    eq_area = study_area.to_crs("EPSG:6933")  # cylindrical equal-area
    return float(eq_area.geometry.union_all().area) / 1e6


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
        if src.crs and src.crs.to_epsg() != 4326:
            with WarpedVRT(src, crs="EPSG:4326") as vrt:
                yield vrt
        else:
            yield src


# ── Tile-grid builder ──────────────────────────────────────────────────────────

def make_tile_grid(bbox: tuple[float, float, float, float], n_side: int = TILE_SIDE) -> gpd.GeoDataFrame:
    """Create an N×N grid of bounding-box polygons (WGS84)."""
    xmin, ymin, xmax, ymax = bbox
    xs = np.linspace(xmin, xmax, n_side + 1)
    ys = np.linspace(ymin, ymax, n_side + 1)

    tiles = []
    for i in range(n_side):
        for j in range(n_side):
            tiles.append(box(xs[i], ys[j], xs[i + 1], ys[j + 1]))
    return gpd.GeoDataFrame(geometry=tiles, crs="EPSG:4326")


# ── Per-tile worker ───────────────────────────────────────────────────────────

def _process_tile(args: tuple[str, dict, dict, dict]) -> Optional[str]:
    """Worker: crop+mask a single tile. Returns the path of the temp output.

    Parameters are passed as plain dicts so the worker is pickle-friendly
    under both fork and spawn start methods.
    """
    raster_path, tile_geom, study_geom, meta = args
    tile_geom = shape(tile_geom)
    study_geom = shape(study_geom)

    # 1. Intersect tile with study area (saves the mask step on empty tiles).
    inter = tile_geom.intersection(study_geom)
    if inter.is_empty:
        return None

    # 2. Open raster windowed on the tile bounds — never loads the full file.
    tb = tile_geom.bounds  # (xmin, ymin, xmax, ymax)
    with _open_4326(raster_path) as src:
        # src is always EPSG:4326 here (native or WarpedVRT), same as tb.
        win = from_bounds(*tb, src.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if win.width <= 0 or win.height <= 0:
            return None

        # Read the windowed data (single band for LCZ maps).
        data = src.read(1, window=win, boundless=True, fill_value=src.nodata or 0)

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
            # If the window is entirely outside the geometry, skip.
            return None

    # 4. Write tile to a tempfile for the parent to mosaic.
    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    out_meta = meta.copy()
    out_meta.update({
        "height": masked.shape[1],
        "width": masked.shape[2],
        "transform": masked_transform,
    })
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(masked)
    return out_path


# ── Tier-3 tiled crop / mask ──────────────────────────────────────────────────

def _crop_tiled(raster_path: str, study_area: gpd.GeoDataFrame,
                lang: str = "en", verbose: bool = True) -> Optional[str]:
    """Tier-3 parallel tiled crop+mask. Returns path to a merged GeoTIFF."""
    if study_area.crs is None or study_area.crs.to_epsg() != 4326:
        study_area = study_area.to_crs(4326)

    bbox = study_area.total_bounds  # (xmin, ymin, xmax, ymax)
    grid = make_tile_grid(tuple(bbox), n_side=TILE_SIDE)

    # Identify non-empty tiles via vector intersection.
    union_geom = study_area.geometry.union_all()
    non_empty_mask = grid.geometry.intersects(union_geom)
    non_empty_idx = np.where(non_empty_mask.values)[0]
    n_tiles = int(len(non_empty_idx))

    n_cores = max(1, (os.cpu_count() or 2) - 1)
    n_cores = min(n_cores, n_tiles) if n_tiles else 1

    if verbose:
        logger.info(lcz_msg("tiling_start", lang, n=n_tiles, cores=n_cores))

    # Read template metadata once.
    with _open_4326(raster_path) as src:
        meta = src.meta.copy()

    # Pick executor: ProcessPoolExecutor survives GIL; ThreadPoolExecutor is
    # safer for rasterio (no fork issues) but slower for pure-Python masking.
    # rasterio releases the GIL during decompression so threads are fine.
    Executor = ThreadPoolExecutor if n_tiles <= 4 else ProcessPoolExecutor

    # Build arg tuples (already-copied dicts/lists are pickle-friendly).
    args_list = []
    for idx in non_empty_idx:
        tile_geom = grid.geometry.iloc[int(idx)]
        args_list.append((
            raster_path,
            mapping(tile_geom),
            mapping(union_geom),
            {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
             for k, v in meta.items()},
        ))

    valid_paths: list[str] = []
    try:
        with Executor(max_workers=n_cores) as ex:
            futures = [ex.submit(_process_tile, a) for a in args_list]
            for fut in as_completed(futures):
                p = fut.result()
                if p is not None:
                    valid_paths.append(p)
    finally:
        pass

    if not valid_paths:
        return None

    if verbose:
        logger.info(lcz_msg("tiling_mosaic", lang, n=len(valid_paths)))

    if len(valid_paths) == 1:
        return valid_paths[0]

    # Mosaic the valid tiles using rasterio + numpy (avoids the rio.merge
    # deprecation path).
    sources = [rasterio.open(p) for p in valid_paths]
    try:
        mosaic, mosaic_transform = _merge_arrays(sources)
        out_meta = sources[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform,
        })
        out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(mosaic)
        return out_path
    finally:
        for s in sources:
            s.close()


def _merge_arrays(sources: list) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Merge multiple rasters into one (last-write-wins per pixel)."""
    import rasterio
    from rasterio.transform import from_bounds as tb_from_bounds

    bounds = src_crs = None
    for s in sources:
        b = s.bounds
        if bounds is None:
            bounds = list(b)
        else:
            bounds[0] = min(bounds[0], b[0])
            bounds[1] = min(bounds[1], b[1])
            bounds[2] = max(bounds[2], b[2])
            bounds[3] = max(bounds[3], b[3])
        src_crs = s.crs

    # Use the resolution of the first source.
    res_x, res_y = sources[0].res
    width = int(math.ceil((bounds[2] - bounds[0]) / res_x))
    height = int(math.ceil((bounds[3] - bounds[1]) / abs(res_y)))
    transform = tb_from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    out = np.zeros((sources[0].count, height, width), dtype=sources[0].dtypes[0])
    for s in sources:
        win = from_bounds(*s.bounds, transform)
        win = win.intersection(rasterio.windows.Window(0, 0, width, height))
        if win.width <= 0 or win.height <= 0:
            continue
        chunk = s.read(window=win, boundless=True, fill_value=0)
        # Last write wins (matches R's mosaic(fun="first") in single-band case).
        row_off = int(round(win.row_off))
        col_off = int(round(win.col_off))
        h = int(round(win.height))
        w = int(round(win.width))
        out[:, row_off:row_off + h, col_off:col_off + w] = chunk

    return out, transform


# ── Adaptive entry point ──────────────────────────────────────────────────────

def lcz_crop_mask_adaptive(
    raster_path: str,
    study_area: gpd.GeoDataFrame,
    lang: str = "en",
    verbose: bool = True,
) -> Optional[str]:
    """Adaptive crop and mask dispatcher.

    Parameters
    ----------
    raster_path : str
        Path (local or ``/vsicurl/...``) to the global LCZ raster.
    study_area : gpd.GeoDataFrame
        Study area polygon in WGS84.
    lang : str
        Language code for progress messages.
    verbose : bool
        If True, emit progress messages via :mod:`logging`.

    Returns
    -------
    str or None
        Path to the cropped/masked GeoTIFF, or ``None`` if the region lies
        outside the raster extent.
    """
    if study_area.crs is None or study_area.crs.to_epsg() != 4326:
        study_area = study_area.to_crs(4326)

    area_km2 = study_area_km2(study_area)

    # ── Tier 3: tiled parallel ────────────────────────────────────────────────
    if area_km2 > XLARGE_AREA_KM2:
        if verbose:
            logger.info(lcz_msg("large_area_tiling", lang,
                                area=round(area_km2 / 1e6, 1)))
        result = _crop_tiled(raster_path, study_area, lang=lang, verbose=verbose)
        if result is None:
            raise RuntimeError(lcz_msg("large_raster_error", lang))
        return result

    # ── Tier 2: simplify + multi-thread ──────────────────────────────────────
    if area_km2 > LARGE_AREA_KM2:
        if verbose:
            logger.info(lcz_msg(
                "large_area_simplify", lang,
                area=f"{round(area_km2):,}",
            ))

        simplified = study_area.copy()
        simplified["geometry"] = simplified.geometry.simplify(
            tolerance=0.005, preserve_topology=True
        )

        # Use rasterio's threaded read with a simplified geometry mask.
        with _open_4326(raster_path) as src:
            try:
                masked, transform = rio_mask(
                    src, simplified.geometry.apply(mapping).tolist(),
                    crop=True, all_touched=False, nodata=src.nodata or 0,
                )
            except Exception as exc:
                raise RuntimeError(lcz_msg("large_raster_error", lang)) from exc

        out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
        out_meta = src.meta.copy()
        out_meta.update({
            "height": masked.shape[1],
            "width": masked.shape[2],
            "transform": transform,
        })
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(masked)
        return out_path

    # ── Tier 1: standard ─────────────────────────────────────────────────────
    with _open_4326(raster_path) as src:
        try:
            masked, transform = rio_mask(
                src, study_area.geometry.apply(mapping).tolist(),
                crop=True, all_touched=False, nodata=src.nodata or 0,
            )
        except Exception as exc:
            raise RuntimeError(lcz_msg("large_raster_error", lang)) from exc

    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    out_meta = src.meta.copy()
    out_meta.update({
        "height": masked.shape[1],
        "width": masked.shape[2],
        "transform": transform,
    })
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(masked)
    return out_path


__all__ = [
    "LARGE_AREA_KM2",
    "XLARGE_AREA_KM2",
    "TILE_SIDE",
    "study_area_km2",
    "make_tile_grid",
    "lcz_crop_mask_adaptive",
]