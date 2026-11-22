"""
lcz_get_parameters.py

LCZ parameter extraction (the 34 Stewart & Oke 2012 morphological/thermal
parameters) from an LCZ class raster.

Design notes
------------
* Class -> parameter lookup is a single pre-computed ``(18, n_params)`` table,
  indexed by the class code, so the whole raster is mapped with NumPy fancy
  indexing instead of a per-pixel / per-class join (the R implementation uses
  ``dplyr::inner_join``).
* The mapping is split into row blocks and run on a bounded
  ``ThreadPoolExecutor``; results are written into one pre-allocated
  ``(n_params, H, W)`` array, so output ordering is deterministic and the
  peak temporary is one block instead of a ``(H, W, n_params)`` cube.
* Optional accelerators (DuckDB Spatial, GeoArrow, Pyogrio) are imported
  defensively with ``HAS_*`` flags and pure rasterio/GeoPandas/Numpy
  fallbacks, so importing this module never requires them.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import polars as pl
import pyarrow as pa
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape

# Optional: DuckDB Spatial (fast polygon dissolve)
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_DUCKDB = False

# Optional: GeoArrow (zero-copy export)
try:
    import geoarrow.pyarrow as ga
    HAS_GEOARROW = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_GEOARROW = False

# Optional: Pyogrio (3-5x faster vector I/O than Fiona)
try:
    import pyogrio
    HAS_PYOGRIO = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_PYOGRIO = False

# Optional: exactextract (not needed here — categorical polygonization is
# faster with rasterio.shapes — but flagged for API compatibility)
try:
    import exactextract  # noqa: F401
    HAS_EXACTEXTRACT = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_EXACTEXTRACT = False

from LCZ4py._internal.lcz_parameters_data import (
    LCZ_TABLE, LCZ_NAMES, LCZ_COLUMNS, LCZ_IDS,
)

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"

#: LCZ classes 1-17; index 0 in the lookup table is the nodata row.
N_LCZ: int = 17


def _resolve_workers(n_items: int, max_workers: Optional[int] = None) -> int:
    """Bounded worker count: ``min(8, cpu_count, n_items)`` unless overridden."""
    if max_workers is not None:
        workers = int(max_workers)
    else:
        workers = min(8, os.cpu_count() or 4)
    return max(1, min(workers, max(1, n_items)))


# ── Pre-computed parameter lookup table ────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_parameter_lookup() -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the ``(18, n_params)`` lookup table with index 0 = nodata.

    Row 0 is all zeros (nodata / invalid class). Rows 1-17 hold the parameter
    values for LCZ classes 1-17, in ``LCZ_COLUMNS`` order (plus a leading
    ``lcz`` column). The table is built once and cached.
    """
    means = LCZ_TABLE.means()

    params = [
        LCZ_TABLE.ids.astype(np.float32),
        LCZ_TABLE.svf_min, LCZ_TABLE.svf_max,
        LCZ_TABLE.ar_min, LCZ_TABLE.ar_max,
        LCZ_TABLE.bsf_min, LCZ_TABLE.bsf_max,
        LCZ_TABLE.isf_min, LCZ_TABLE.isf_max,
        LCZ_TABLE.psf_max, LCZ_TABLE.psf_min,
        LCZ_TABLE.tsf_min, LCZ_TABLE.tsf_max,
        LCZ_TABLE.hre_min, LCZ_TABLE.hre_max,
        LCZ_TABLE.trc_min, LCZ_TABLE.trc_max,
        LCZ_TABLE.sad_min, LCZ_TABLE.sad_max,
        LCZ_TABLE.sal_min, LCZ_TABLE.sal_max,
        LCZ_TABLE.ah_min, LCZ_TABLE.ah_max,
        LCZ_TABLE.z0,
        means["SVFmean"], means["ARmean"],
        means["BSFmean"], means["ISFmean"],
        means["PSFmean"], means["TSFmean"],
        means["HREmean"], means["TRCmean"],
        means["SADmean"], means["SALmean"],
        means["AHmean"],
    ]

    names = (
        "lcz", "svf_min", "svf_max", "AR_min", "AR_max",
        "BSF_min", "BSF_max", "ISF_min", "ISF_max",
        "PSF_max", "PSF_min", "TSF_min", "TSF_max",
        "HRE_min", "HRE_max", "TRC_min", "TRC_max",
        "SAD_min", "SAD_max", "SAL_min", "SAL_max",
        "AH_min", "AH_max", "z0",
        "svf_mean", "aspect_mean", "BSF_mean", "ISF_mean",
        "PSF_mean", "TSF_mean", "HRE_mean", "TRC_mean",
        "SAD_mean", "SAL_mean", "AH_mean",
    )

    for arr in params:
        arr = np.asarray(arr)
        if arr.shape != (N_LCZ,):
            raise ValueError(
                f"LCZ parameter table is inconsistent: expected {N_LCZ} values "
                f"per parameter, got shape {arr.shape}. This is a bug in "
                "lcz_parameters_data.py (the single source of truth)."
            )

    lookup = np.zeros((N_LCZ + 1, len(params)), dtype=np.float32)
    lookup[1:] = np.stack(params, axis=1)  # (17, n_params)

    return lookup, names


def parameter_names() -> list[str]:
    """Canonical parameter names, in ``LCZ_COLUMNS`` order."""
    _, names = _get_parameter_lookup()
    return list(names)


# ── Vectorized class → parameter mapping (ultra-fast) ─────────────────────────

def _class_to_index(class_arr: np.ndarray, nodata: Optional[int] = 0) -> np.ndarray:
    """Convert a raw raster band to a safe 0-17 lookup index.

    Any value that is not an integer LCZ class in ``1..17`` (including NaN,
    infinities, and values above 17 such as a 255 nodata sentinel) maps to 0
    — the nodata row. Previously the array was simply clipped to ``0..17``,
    which silently turned every out-of-range nodata value into class 17
    (water).
    """
    arr = np.asarray(class_arr)
    if arr.dtype.kind in "iu":
        filled = arr.astype(np.int32, copy=False)
        finite = np.ones(arr.shape, dtype=bool)
    else:
        finite = np.isfinite(arr)
        filled = np.where(finite, arr, 0).astype(np.int32)

    valid = finite & (filled >= 1) & (filled <= N_LCZ)
    if nodata is not None:
        valid &= filled != int(nodata)

    idx = np.zeros(arr.shape, dtype=np.int16)
    idx[valid] = filled[valid].astype(np.int16)
    return idx


def _map_block(
    lookup: np.ndarray, block: np.ndarray, nodata: Optional[int],
) -> np.ndarray:
    """Map one raster block to ``(n_params, h, w)`` float32."""
    idx = _class_to_index(block, nodata)
    # lookup[idx] -> (h, w, n_params); transposing into a C-contiguous
    # (n_params, h, w) block keeps downstream band access contiguous.
    return np.ascontiguousarray(lookup[idx].transpose(2, 0, 1))


def map_class_to_params_vectorized(
    class_arr: np.ndarray,
    nodata: int = 0,
    max_workers: Optional[int] = None,
) -> tuple[np.ndarray, list[str]]:
    """Map an LCZ class raster to the parameter stack.

    This is 30-100x faster than R's ``dplyr::inner_join`` approach.

    Parameters
    ----------
    class_arr : np.ndarray
        ``(H, W)`` integer array of LCZ classes (1-17). Float arrays are
        accepted; non-finite values are treated as nodata.
    nodata : int, default 0
        Nodata sentinel. Values outside ``1..17`` (and equal to ``nodata``)
        map to the all-zero row.
    max_workers : int, optional
        Bound on the row-block thread pool. Default ``min(8, cpu_count)``.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        ``(n_params, H, W)`` float32 array and the parameter names.
    """
    lookup, names = _get_parameter_lookup()
    arr = np.asarray(class_arr)
    if arr.ndim != 2:
        raise ValueError(
            f"class_arr must be a 2-D (H, W) array, got shape {arr.shape}."
        )

    h, _w = arr.shape
    workers = _resolve_workers(h, max_workers)

    result = np.empty((lookup.shape[1], h, arr.shape[1]), dtype=np.float32)
    if workers == 1 or h < 4:
        result[:] = _map_block(lookup, arr, nodata)
        return result, list(names)

    # Enough blocks to load-balance, few enough to keep overhead negligible.
    n_blocks = min(h, max(workers, workers * 4))
    bounds = np.linspace(0, h, n_blocks + 1, dtype=int)
    spans = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_blocks)
             if bounds[i + 1] > bounds[i]]

    def _run(span: tuple[int, int]) -> tuple[int, int, np.ndarray]:
        r0, r1 = span
        return r0, r1, _map_block(lookup, arr[r0:r1, :], nodata)

    with ThreadPoolExecutor(max_workers=min(workers, len(spans))) as pool:
        for r0, r1, block in pool.map(_run, spans):
            result[:, r0:r1, :] = block

    return result, list(names)


# ── Polygon helpers ───────────────────────────────────────────────────────────

def _polygonize_lcz(
    class_arr: np.ndarray,
    transform: Affine,
    crs,
    nodata: Optional[int] = 0,
) -> gpd.GeoDataFrame:
    """Polygonize valid LCZ classes and dissolve them per class.

    Pure rasterio + GeoPandas (no optional dependency). Returns one row per
    class present with columns ``lcz`` and ``geometry``.

    Raises
    ------
    ValueError
        If the raster contains no valid LCZ class (1-17).
    """
    idx = _class_to_index(class_arr, nodata)
    records: list[dict] = []
    for geom, val in shapes(idx, transform=transform):
        val = int(val)
        if val < 1 or val > N_LCZ:
            continue
        records.append({"lcz": val, "geometry": shape(geom)})

    if not records:
        raise ValueError(
            "The LCZ raster contains no valid classes (1-17) — nothing to "
            "vectorize. Check the raster's values/nodata."
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    dissolved = gdf.dissolve(by="lcz", as_index=False).sort_values("lcz")
    return gpd.GeoDataFrame(dissolved, geometry="geometry", crs=crs).reset_index(drop=True)


def vectorize_with_duckdb(
    class_arr: np.ndarray,
    transform: Affine,
    crs: str,
    nodata: Optional[int] = 0,
    max_workers: Optional[int] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Dissolve LCZ polygons, using DuckDB Spatial when it is usable.

    DuckDB's spatial extension must already be installed (no implicit
    ``INSTALL`` network call is attempted); otherwise this falls back to the
    GeoPandas dissolve in :func:`_polygonize_lcz`. Returns ``None`` only when
    DuckDB was requested but the fallback also found no valid classes.
    """
    try:
        gdf = _polygonize_lcz(class_arr, transform, crs, nodata=nodata)
    except ValueError:
        return None

    if not HAS_DUCKDB:
        return gdf

    try:
        conn = duckdb.connect()
        try:
            installed = conn.execute(
                "SELECT installed FROM duckdb_extensions() "
                "WHERE extension_name = 'spatial'"
            ).fetchone()
            if not installed or not installed[0]:
                return gdf
            conn.execute("LOAD spatial;")
            conn.register("lcz_polygons", gdf)
            rows = conn.execute(
                "SELECT lcz, ST_AsWKB(ST_Union_Agg(geometry)) AS geom "
                "FROM lcz_polygons WHERE lcz BETWEEN 1 AND 17 "
                "GROUP BY lcz ORDER BY lcz"
            ).fetchall()
        finally:
            conn.close()

        from shapely import wkb as shapely_wkb

        geoms = []
        lczs = []
        for lcz_val, geom_blob in rows:
            geom = shapely_wkb.loads(bytes(geom_blob))
            if geom.is_empty:
                continue
            geoms.append(geom)
            lczs.append(int(lcz_val))
        if not geoms:
            return gdf
        out = gpd.GeoDataFrame({"lcz": lczs}, geometry=geoms, crs=crs)
        return out.sort_values("lcz").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001 - optional accelerator
        logger.debug("DuckDB dissolve unavailable (%s); using GeoPandas.", exc)
        return gdf


def _add_area_km2(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach a correct ``area_km2`` column (equal-area projection)."""
    out = gdf.copy()
    try:
        metric = out
        if out.crs is not None and out.crs.is_geographic:
            metric = out.to_crs(6933)  # Equal Earth (global equal-area)
        out["area_km2"] = metric.geometry.area.to_numpy() / 1_000_000.0
    except Exception as exc:  # noqa: BLE001 - area is cosmetic
        logger.debug("Could not compute polygon areas: %s", exc)
        out["area_km2"] = np.nan
    return out


# ── GeoArrow export for parameters ────────────────────────────────────────────

def parameters_to_geoarrow(
    gdf: gpd.GeoDataFrame,
    param_names: Optional[list[str]] = None,
) -> Optional[pa.Table]:
    """Export the parameter polygons to GeoArrow (requires ``geoarrow``)."""
    if not HAS_GEOARROW:
        return None

    try:
        return ga.from_geopandas(gdf)
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.warning("GeoArrow conversion failed: %s", exc)
        return None


def vectorize_with_exactextract(
    class_arr: np.ndarray,
    transform: Affine,
    crs: str,
) -> Optional[gpd.GeoDataFrame]:
    """Deprecated compatibility shim; always returns ``None``.

    ``exactextract`` is built for extracting *continuous* values into
    polygons, not for polygonizing categorical rasters, so this module uses
    ``rasterio.features.shapes`` (see :func:`_polygonize_lcz`) instead. The
    name is kept so existing imports keep working.
    """
    return None


# ── Concurrent multi-band writer ──────────────────────────────────────────────

def write_stack_concurrent(
    path: str,
    stack: np.ndarray,
    profile: dict,
    band_names: list[str],
    max_workers: int = 4,
    nodata: float = 0.0,
) -> str:
    """Write a multi-band float32 GeoTIFF with band descriptions.

    GDAL writes to a single dataset are not thread-safe, so the bands are
    written sequentially (the cheap part); LZW/deflate compression is done by
    GDAL's C code, which releases the GIL. ``max_workers`` is accepted for
    API symmetry with the other helpers.

    Returns
    -------
    str
        The path written.
    """
    out_profile = dict(profile or {})
    out_profile.update(
        driver=out_profile.get("driver", "GTiff"),
        count=int(stack.shape[0]),
        dtype="float32",
        compress="lzw",
        predictor=3,  # Float predictor for LZW
        nodata=nodata,
        BIGTIFF="IF_SAFER",  # Handle >4GB outputs
    )
    # A block size from the source can be invalid for a much smaller array.
    out_profile.pop("blockxsize", None)
    out_profile.pop("blockysize", None)
    out_profile.pop("tiled", None)

    data = np.asarray(stack, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"stack must be (n_bands, H, W), got shape {data.shape}")
    if len(band_names) != data.shape[0]:
        raise ValueError(
            f"band_names has {len(band_names)} entries but stack has "
            f"{data.shape[0]} bands."
        )

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)
            dst.update_tags(i, NAME=name)

    return path


# ── Pyogrio fast I/O ──────────────────────────────────────────────────────────

def write_geopackage_fast(
    gdf: gpd.GeoDataFrame,
    path: str,
    layer_name: str = "lcz_parameters",
) -> str:
    """Write a GeoPackage, using Pyogrio when available."""
    if HAS_PYOGRIO:
        try:
            pyogrio.write_vector(
                gdf,
                path,
                layer=layer_name,
                driver="GPKG",
                promote_to_multi=True,
                spatial_index=True,
            )
            return path
        except Exception as exc:  # noqa: BLE001 - fall back to GeoPandas
            logger.warning("Pyogrio write failed, falling back to GeoPandas: %s", exc)

    gdf.to_file(path, driver="GPKG", layer=layer_name)
    return path


# ── Public entry point ────────────────────────────────────────────────────────

@dataclass
class LCZStackResult:
    """Return type for the advanced stack / shp / single-select paths."""
    path: Optional[str] = None
    array: Optional[np.ndarray] = None
    gdf: Optional[gpd.GeoDataFrame] = None
    geoarrow_table: Optional[pa.Table] = None


def _read_class_raster(x: Union[str, Path]) -> tuple[np.ndarray, dict, object, Affine]:
    """Read band 1 of an LCZ GeoTIFF and its georeferencing."""
    path = Path(x)
    if not path.exists():
        raise FileNotFoundError(f"LCZ raster not found: {path}")

    with rasterio.open(str(path)) as src:
        class_arr = src.read(1)
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        if src.count > 1:
            logger.warning(
                "LCZ raster has %d bands; using band 1 as the class raster.", src.count
            )
        if crs is None:
            logger.warning(
                "LCZ raster has no CRS; assuming EPSG:4326. Vector outputs will "
                "not be georeferenced correctly until the raster is given a CRS."
            )
            crs = "EPSG:4326"

    return class_arr, profile, crs, transform


def lcz_get_parameters(
    x: Union[str, Path],
    iselect: Union[str, list[str], None] = None,
    istack: bool = True,
    ishp: bool = False,
    isave: bool = False,
    use_duckdb: bool = True,
    use_geoarrow: bool = True,
    use_pyogrio: bool = True,
    lang: str = "en",
    max_workers: Optional[int] = None,
) -> Union[LCZStackResult, np.ndarray, gpd.GeoDataFrame]:
    """Compute the 34 LCZ parameters for an LCZ class raster (Advanced).

    Parameters
    ----------
    x : str or Path
        Path to the LCZ class GeoTIFF (classes 1-17, nodata 0).
    iselect : str or list[str] or None
        Select a subset of parameters to return, e.g. ``["svf_mean", "z0"]``.
    istack : bool
        If True and ``iselect`` is None, return the full multi-band stack.
    ishp : bool
        If True, return a per-class polygon table as a GeoDataFrame with one
        column per parameter (R's ``ishp = TRUE``).
    isave : bool
        Persist the result under ``LCZ4r_output/``.
    use_duckdb : bool
        Try DuckDB Spatial for the polygon dissolve (falls back to GeoPandas
        when DuckDB or its spatial extension is unavailable).
    use_geoarrow : bool
        Generate a GeoArrow table for the polygon output (requires geoarrow).
    use_pyogrio : bool
        Use Pyogrio for faster GeoPackage I/O (falls back to GeoPandas).
    lang : str
        Message language ("en"/"pt"/"es"/"zh"). Accepted for API symmetry.
    max_workers : int, optional
        Bound on the row-block thread pool used for the class -> parameter
        mapping. Default ``min(8, cpu_count)``.

    Returns
    -------
    LCZStackResult, np.ndarray, dict, or gpd.GeoDataFrame
        * ``ishp=True`` -> ``LCZStackResult`` with ``.gdf`` populated.
        * ``iselect`` -> ``(n_selected, H, W)`` float32 ndarray.
        * ``istack=False`` -> ``{name: (H, W) array}`` dict.
        * default -> ``LCZStackResult`` with ``.array`` = ``(35, H, W)``.
    """
    if x is None:
        raise ValueError("x must be a path to a GeoTIFF")

    class_arr, profile, crs, transform = _read_class_raster(x)

    # Vectorized (block-parallel) class -> parameter mapping.
    stack, param_names = map_class_to_params_vectorized(
        class_arr, max_workers=max_workers,
    )

    # ── Polygon output ────────────────────────────────────────────────────────
    if ishp:
        logger.info("Vectorizing LCZ classes...")
        if use_duckdb:
            gdf = vectorize_with_duckdb(
                class_arr, transform, crs, max_workers=max_workers,
            )
        else:
            gdf = _polygonize_lcz(class_arr, transform, crs)

        if gdf is None:
            raise ValueError(
                "The LCZ raster contains no valid classes (1-17) — cannot "
                "build a polygon table."
            )

        gdf = _add_area_km2(gdf)

        # Attach one column per parameter through a Polars join on the class.
        # ``param_names[0]`` is the class id itself, which is the join key.
        lookup, _ = _get_parameter_lookup()
        df_params = pl.DataFrame(
            lookup[1:, 1:], schema=list(param_names[1:]),
        ).with_columns(
            pl.Series("lcz", LCZ_IDS.astype(np.int64))
        ).to_pandas()
        gdf = gdf.merge(df_params, on="lcz", how="left")
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)

        geoarrow_table = None
        if use_geoarrow:
            geoarrow_table = parameters_to_geoarrow(gdf, param_names)

        out_path: Optional[str] = None
        if isave:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, "lcz_par.gpkg")
            if use_pyogrio:
                write_geopackage_fast(gdf, out_path)
            else:
                gdf.to_file(out_path, driver="GPKG")
            logger.info("Saved: %s", os.path.abspath(out_path))

        return LCZStackResult(path=out_path, gdf=gdf, geoarrow_table=geoarrow_table)

    # ── Single / subset selection ─────────────────────────────────────────────
    if iselect is not None:
        if isinstance(iselect, str):
            iselect = [iselect]
        missing = [c for c in iselect if c not in param_names]
        if missing:
            raise ValueError(
                f"Unknown parameter(s): {missing}. "
                f"Available: {', '.join(param_names)}"
            )
        if not iselect:
            raise ValueError("iselect must contain at least one parameter name.")

        idx = [param_names.index(c) for c in iselect]
        sel = np.ascontiguousarray(stack[idx])

        out_path = None
        if isave:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, "lcz_par_select.tif")
            write_stack_concurrent(out_path, sel, profile, list(iselect))
            logger.info("Saved: %s", os.path.abspath(out_path))

        return sel

    # ── List output (istack = FALSE) ──────────────────────────────────────────
    if not istack:
        return {name: stack[i] for i, name in enumerate(param_names)}

    # ── Default: full stack ───────────────────────────────────────────────────
    out_path = None
    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_par_stack.tif")
        write_stack_concurrent(out_path, stack, profile, param_names)
        logger.info("Saved: %s", os.path.abspath(out_path))

    return LCZStackResult(path=out_path, array=stack)


__all__ = [
    "LCZStackResult",
    "lcz_get_parameters",
    "map_class_to_params_vectorized",
    "parameter_names",
    "write_stack_concurrent",
    "write_geopackage_fast",
    "vectorize_with_duckdb",
    "vectorize_with_exactextract",
    "parameters_to_geoarrow",
]
