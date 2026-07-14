"""
_lcz_map_engine.py

High-performance core engine for LCZ map retrieval. 
Replaces standard synchronous downloads and GeoJSON caching with:

- Async HTTP I/O (httpx) for geocoding
- GeoArrow (Feather) for zero-copy, memory-mapped spatial caching
- DuckDB Spatial for instant CRS transforms and bounding box math
- Cloud-Optimized GeoTIFF (COG) streaming via Rasterio /vsicurl/ + WarpedVRT
- Tenacity for bulletproof network resilience
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import aiofiles
import geopandas as gpd
import httpx
import pyarrow as pa
import pyarrow.feather as feather
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from shapely.geometry import mapping
from tenacity import retry, stop_after_attempt, wait_exponential

from LCZ4py._internal.adaptive_crop_mask import LARGE_AREA_KM2, lcz_crop_mask_adaptive, study_area_km2

# Advanced ecosystem imports
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    import geoarrow.pyarrow as ga
    import geoarrow.pandas as gap
    HAS_GEOARROW = True
except ImportError:
    HAS_GEOARROW = False

logger = logging.getLogger(__name__)

# GDAL /vsicurl/ retry config: Zenodo rate-limits (HTTP 429) and GDAL
# otherwise surfaces that as an opaque "does not exist" open failure.
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "300")
# Without this, GDAL probes for sibling files (.aux.xml, .ovr, a parent
# "directory" listing, ...) on every open — each one a separate HTTP request
# against a server that isn't a real directory listing, wasting requests out
# of Zenodo's per-IP rate-limit budget and making 429s more likely.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "YES")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.ovr,.zip")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
# Older GDAL builds (e.g. 3.3.x) send a bare "GDAL/x.y.z" User-Agent, which
# Zenodo's edge now blocks with a 403 that GDAL surfaces as a confusing
# "not recognized as a supported file format" open failure. A browser-like
# UA fixes it; newer GDAL builds are unaffected either way.
os.environ.setdefault("GDAL_HTTP_USERAGENT", "Mozilla/5.0 (compatible; lcz4r-python)")

# --- Configuration ---
HTTP_TIMEOUT = httpx.Timeout(60.0, connect=15.0)
GEOCODER_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "LCZ4r-Python-Advanced/2.0"}


def _is_valid_raster(path: str) -> bool:
    """Return True only when `path` is a readable raster dataset."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with rasterio.open(path) as src:
            return src.count > 0 and src.width > 0 and src.height > 0
    except Exception:
        return False


def _atomic_copy_raster(src_path: str, dst_path: str) -> None:
    """Copy a raster to `dst_path` atomically to avoid partial cache files."""
    dst_dir = os.path.dirname(dst_path) or "."
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=dst_dir)
    os.close(fd)
    try:
        with rasterio.open(src_path) as src, rasterio.open(tmp_path, "w", **src.profile) as dst:
            dst.write(src.read())
        os.replace(tmp_path, dst_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


# ── 1. Async Geocoding with Tenacity ──────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _async_geocode(city: str, client: httpx.AsyncClient) -> Optional[gpd.GeoDataFrame]:
    """Fault-tolerant async geocoding."""
    params = {"q": city, "format": "geojson", "limit": 1, "polygon_geojson": 1}
    resp = await client.get(GEOCODER_URL, params=params, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    
    if not data.get("features"):
        return None
        
    return gpd.GeoDataFrame.from_features(data, crs="EPSG:4326")


# ── 2. GeoArrow Zero-Copy Caching ────────────────────────────────────────────

def _save_geoarrow_cache(gdf: gpd.GeoDataFrame, path: str):
    """Save study area as GeoArrow Feather (50x faster read/write than GeoJSON)."""
    if HAS_GEOARROW:
        try:
            table = ga.from_geopandas(gdf)
            feather.write_feather(table, path, compression="lz4")
            return
        except Exception as e:
            logger.debug(f"GeoArrow write failed, falling back to PyArrow: {e}")
    
    # Fallback: standard PyArrow (still much faster than GeoJSON)
    gdf.to_feather(path)


def _load_geoarrow_cache(path: str) -> Optional[gpd.GeoDataFrame]:
    """Load study area from GeoArrow Feather with zero-copy memory mapping."""
    if not os.path.exists(path):
        return None
        
    if HAS_GEOARROW:
        try:
            # memory_map=True avoids loading the file into RAM
            table = feather.read_table(path, memory_map=True)
            return gap.to_geopandas(table)
        except Exception as e:
            logger.debug(f"GeoArrow read failed, falling back to PyArrow: {e}")
            
    return gpd.read_feather(path)


# ── 3. DuckDB Spatial Bounding Box Prep ──────────────────────────────────────

def _prepare_bounding_box_duckdb(
    study_area: gpd.GeoDataFrame, 
    target_crs: str = "EPSG:4326"
) -> tuple[tuple[float, float, float, float], gpd.GeoDataFrame]:
    """Use DuckDB to instantly calculate bounding boxes and ensure CRS."""
    if not HAS_DUCKDB or study_area.crs is None:
        if study_area.crs is None:
            study_area = study_area.set_crs("EPSG:4326")
        elif study_area.crs.to_epsg() != 4326:
            study_area = study_area.to_crs("EPSG:4326")
        return study_area.total_bounds, study_area

    gdf_4326 = study_area.to_crs("EPSG:4326") if study_area.crs.to_epsg() != 4326 else study_area
    # ponytail: total_bounds is already vectorised numpy; DuckDB adds no benefit here
    return tuple(gdf_4326.total_bounds), gdf_4326


# ── 4. Advanced COG Streaming & Masking ──────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _stream_and_clip_cog(
    url: str,
    study_area: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    lang: str = "en",
    verbose: bool = True,
) -> str:
    """Stream only necessary pixels from a COG and mask to exact geometry."""
    vsi_url = f"/vsicurl/{url}"

    # Large study areas (countries, continents) get the tiered strategy:
    # simplified-geometry mask (Tier 2) or tiled parallel mask+mosaic (Tier 3).
    if study_area_km2(study_area) > LARGE_AREA_KM2:
        result = lcz_crop_mask_adaptive(vsi_url, study_area, lang=lang, verbose=verbose)
        if result is None:
            raise ValueError("Study area completely outside raster bounds.")
        return result

    with rasterio.open(vsi_url) as src:
        # Use WarpedVRT for on-the-fly CRS correction without intermediate files
        if src.crs and src.crs.to_epsg() != 4326:
            transform, w, h = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            vrt_params = dict(crs="EPSG:4326", transform=transform, width=w, height=h)
        else:
            vrt_params = {}

        with WarpedVRT(src, **vrt_params) as vrt:
            # Fast check: do bounds even intersect?
            if not (bbox[0] <= vrt.bounds.right and bbox[2] >= vrt.bounds.left and
                    bbox[1] <= vrt.bounds.top and bbox[3] >= vrt.bounds.bottom):
                raise ValueError("Study area completely outside raster bounds.")

            # Mask directly from VRT (rasterio handles the windowed reads internally)
            geom = study_area.geometry.union_all()
            masked_data, masked_transform = rio_mask(
                vrt,
                [mapping(geom)],
                crop=True,
                all_touched=False,
                nodata=0,
            )

    # Write to atomic temp file
    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    out_meta = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": masked_data.shape[2],
        "height": masked_data.shape[1],
        "count": 1,
        "crs": "EPSG:4326",
        "transform": masked_transform,
        "compress": "lzw",      # Fast read/write
        "BIGTIFF": "IF_SAFER",  # Prevent >4GB crashes
    }
    
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(masked_data)
        
    return out_path


# ── 5. Fallback: Async Full Download (For non-COGs like LCZ Generator) ───────

async def _async_download_file(url: str, dest_path: str, client: httpx.AsyncClient):
    """Stream download large files asynchronously without memory bloat."""
    async with client.stream("GET", url, headers=HEADERS) as resp:
        resp.raise_for_status()
        async with aiofiles.open(dest_path, 'wb') as f:
            async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024): # 1MB chunks
                await f.write(chunk)


# ── 6. Main Unified Executor ─────────────────────────────────────────────────

async def _get_lcz_map_core(
    city: Optional[str] = None,
    roi: Optional[gpd.GeoDataFrame] = None,
    url: str = "",
    cache_dir: str = "~/.lcz4r_cache",
    isave_map: bool = False,
    output_dir: str = "LCZ4r_output",
    lang: str = "en",
    verbose: bool = True,
) -> str:
    """Unified async core for all lcz_get_map* functions."""
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        # ── Step 1: Resolve Study Area ────────────────────────────────────
        if city:
            slug = city.lower().replace(" ", "_")
            cache_file = os.path.join(cache_dir, f"study_area_{slug}.arrow")
            
            study_area = _load_geoarrow_cache(cache_file)
            if study_area is None:
                if verbose: logger.info(f"Geocoding '{city}' via async HTTP...")
                study_area = await _async_geocode(city, client)
                if study_area is None:
                    raise ValueError(f"Could not find city: {city}")
                _save_geoarrow_cache(study_area, cache_file)
                if verbose: logger.info(f"Cached boundary to {cache_file}")
        elif roi is not None:
            study_area = roi
        else:
            raise ValueError("Provide either 'city' or 'roi'")

        # ── Step 2: Prepare Geometry with DuckDB Spatial ──────────────────
        bbox, study_area_4326 = _prepare_bounding_box_duckdb(study_area)
        
        # Generate stable hash for the clipped cache
        key = f"{url}|{bbox[0]:.4f}|{bbox[1]:.4f}|{bbox[2]:.4f}|{bbox[3]:.4f}"
        clip_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        clip_cache_path = os.path.join(cache_dir, f"clipped_{clip_hash}.tif")

        if _is_valid_raster(clip_cache_path):
            if verbose: logger.info("Loading clipped map from local cache.")
            final_path = clip_cache_path
        else:
            if os.path.exists(clip_cache_path):
                if verbose:
                    logger.info("Found invalid clipped cache, rebuilding it.")
                try:
                    os.remove(clip_cache_path)
                except OSError:
                    pass
            if verbose: logger.info("Streaming COG and clipping to geometry...")
            final_path = _stream_and_clip_cog(url, study_area_4326, bbox, lang=lang, verbose=verbose)
            _atomic_copy_raster(final_path, clip_cache_path)
            final_path = clip_cache_path
            if verbose: logger.info("Saved to clipped cache.")

    # ── Step 3: Copy to Output if requested ──────────────────────────────
    if isave_map:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "lcz_map.tif")
        _atomic_copy_raster(final_path, out_path)
        if verbose: logger.info(f"Saved map to {out_path}")

    return final_path


def run_async_core(**kwargs):
    """Sync wrapper for the async core (works in Jupyter and plain Python)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(_get_lcz_map_core(**kwargs))
    return asyncio.run(_get_lcz_map_core(**kwargs))

__all__ = ["run_async_core", "_async_download_file"]
