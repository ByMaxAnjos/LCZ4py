"""
_lcz_downloader_advanced.py

High-performance async downloader with:
- httpx for async HTTP/2 streaming
- Rasterio /vsicurl/ for Cloud-Optimized GeoTIFF (COG) direct streaming
- DuckDB Spatial for fast bounding box operations
- Async caching with atomic writes
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
import httpx
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds as _from_bounds
from rasterio.vrt import WarpedVRT
from shapely.geometry import box, mapping
from rasterio.mask import mask as rio_mask

# DuckDB Spatial
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

logger = logging.getLogger(__name__)

# Custom retry and timeout settings for robustness
HTTP_TIMEOUT = httpx.Timeout(300.0, connect=30.0)
HTTP_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=5)

# GDAL /vsicurl/ retry config: Zenodo rate-limits (HTTP 429) and GDAL
# otherwise surfaces that as an opaque "does not exist" open failure.
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "300")
# Older GDAL builds send a bare "GDAL/x.y.z" User-Agent, which Zenodo's edge
# now blocks with a 403 that GDAL surfaces as an opaque open failure.
os.environ.setdefault("GDAL_HTTP_USERAGENT", "Mozilla/5.0 (compatible; lcz4r-python)")


def _is_valid_raster(path: str) -> bool:
    """Return True only when `path` exists and is a readable raster."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with rasterio.open(path) as src:
            return src.count > 0 and src.width > 0 and src.height > 0
    except Exception:
        return False


def _atomic_copy_raster(src_path: str, dst_path: str) -> None:
    """Copy a raster atomically to prevent partial cache files."""
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


def _open_vsicurl(vsi_path: str) -> rasterio.DatasetReader:
    """Open a /vsicurl/ dataset with a clearer error on failure."""
    try:
        return rasterio.open(vsi_path)
    except rasterio.errors.RasterioIOError as e:
        raise rasterio.errors.RasterioIOError(
            f"Could not open remote raster {vsi_path!r}. The source server may be "
            f"rate-limiting or temporarily unavailable; wait a bit and retry. "
            f"Original error: {e}"
        ) from e


# ── Async Nominatim Geocoder ──────────────────────────────────────────────────

async def _geocode_async(
    city: str,
    client: httpx.AsyncClient,
) -> Optional[gpd.GeoDataFrame]:
    """Async geocoding using Nominatim with proper rate limiting."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": city,
        "format": "geojson",
        "limit": 1,
        "polygon_geojson": 1,
    }
    headers = {"User-Agent": "LCZ4r-Python/2.0 Advanced"}
    
    try:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get("features"):
            return None
            
        gdf = gpd.GeoDataFrame.from_features(data, crs="EPSG:4326")
        return gdf
    except Exception as e:
        logger.error(f"Geocoding failed for {city}: {e}")
        return None


# ── Atomic async file writer ──────────────────────────────────────────────────

async def _atomic_write_async(
    path: str,
    data: bytes,
) -> None:
    """Write file atomically to prevent cache corruption."""
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    
    temp_path = f"{path}.tmp.{os.getpid()}"
    try:
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(data)
        os.replace(temp_path, path)  # Atomic on POSIX/Windows
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


# ── COG Streaming with Rasterio ───────────────────────────────────────────────

def stream_cog_window(
    url: str,
    bounds: tuple[float, float, float, float],
    target_crs: str = "EPSG:4326",
    max_pixels: int = 10_000_000,
) -> tuple[np.ndarray, dict]:
    """Stream only the necessary window from a Cloud-Optimized GeoTIFF.
    
    This never downloads the full file, only the header + required tiles.
    """
    # Use /vsicurl/ for true streaming
    vsi_url = f"/vsicurl/{url}"
    
    with _open_vsicurl(vsi_url) as src:
        # Calculate intersection window
        src_bounds = src.bounds
        
        # Clip bounds to source extent
        clipped_bounds = (
            max(bounds[0], src_bounds.left),
            max(bounds[1], src_bounds.bottom),
            min(bounds[2], src_bounds.right),
            min(bounds[3], src_bounds.top),
        )
        
        if clipped_bounds[0] >= clipped_bounds[2] or clipped_bounds[1] >= clipped_bounds[3]:
            raise ValueError("Bounds do not intersect the raster")
        
        # Handle CRS mismatch using WarpedVRT (no intermediate file)
        if src.crs is None or str(src.crs) != target_crs:
            with WarpedVRT(src, crs=target_crs, resampling=rasterio.enums.Resampling.nearest) as vrt:
                return _read_window_from_dataset(vrt, clipped_bounds, max_pixels)
        else:
            return _read_window_from_dataset(src, clipped_bounds, max_pixels)


def _read_window_from_dataset(
    src,
    bounds: tuple[float, float, float, float],
    max_pixels: int,
) -> tuple[np.ndarray, dict]:
    """Read a window from a rasterio dataset with optional downsampling."""
    window = rasterio.windows.from_bounds(*bounds, src.transform)
    window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
    
    h, w = window.height, window.width
    if h * w > max_pixels:
        scale = (max_pixels / (h * w)) ** 0.5
        out_shape = (max(1, int(h * scale)), max(1, int(w * scale)))
    else:
        out_shape = (max(1, int(h)), max(1, int(w)))

    arr = src.read(
        1,
        window=window,
        out_shape=out_shape,
        resampling=rasterio.enums.Resampling.nearest,
        boundless=True,
        fill_value=src.nodata or 0,
    )

    # Compute transform that matches the actual output pixel size.
    # rasterio.windows.transform gives source resolution — wrong when downsampled.
    left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
    transform = _from_bounds(
        left, bottom, right, top,
        out_shape[1], out_shape[0],
    )
    
    return arr, {
        "transform": transform,
        "crs": src.crs,
        "height": arr.shape[0],
        "width": arr.shape[1],
        "nodata": src.nodata or 0,
    }


# ── Fast crop and mask ────────────────────────────────────────────────────────

def crop_mask_fast(
    raster_path_or_url: str,
    study_area: gpd.GeoDataFrame,
    nodata: int = 0,
) -> str:
    """Crop and mask a raster, handling both local and remote COGs."""
    if study_area.crs is None:
        study_area = study_area.set_crs("EPSG:4326")
    elif study_area.crs.to_epsg() != 4326:
        study_area = study_area.to_crs("EPSG:4326")
    
    # Handle remote URLs via /vsicurl/
    vsi_path = raster_path_or_url
    if raster_path_or_url.startswith("http"):
        vsi_path = f"/vsicurl/{raster_path_or_url}"
    
    geom = study_area.geometry.union_all()
    
    with _open_vsicurl(vsi_path) as src:
        # Use WarpedVRT if CRS mismatch
        if src.crs and str(src.crs) != "EPSG:4326":
            with WarpedVRT(src, crs="EPSG:4326") as vrt:
                return _apply_mask(vrt, geom, nodata)
        else:
            return _apply_mask(src, geom, nodata)


def _apply_mask(src, geom, nodata: int) -> str:
    """Apply mask and write to temp file."""
    try:
        masked, transform = rio_mask(
            src,
            [mapping(geom)],
            crop=True,
            all_touched=False,
            nodata=nodata,
        )
    except ValueError:
        # Empty intersection
        raise ValueError("Study area does not intersect the raster")

    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    out_meta = src.meta.copy()
    out_meta.update({
        "height": masked.shape[1],
        "width": masked.shape[2],
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    })
    
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(masked)
        
    return out_path


# ── Public Async Downloader ───────────────────────────────────────────────────

async def lcz_get_map_async(
    city: Optional[str] = None,
    roi: Optional[gpd.GeoDataFrame] = None,
    url: str = "https://zenodo.org/records/8419340/files/lcz_filter_v3.tif?download=1",
    cache_dir: str = "~/.lcz4r_cache",
    isave_map: bool = False,
) -> str:
    """Async map downloader with COG streaming.
    
    Parameters
    ----------
    city : str, optional
        City name to geocode.
    roi : gpd.GeoDataFrame, optional
        Custom ROI.
    url : str
        URL to the LCZ raster (should be COG for best performance).
    cache_dir : str
        Cache directory.
    isave_map : bool
        Save to LCZ4r_output.
    
    Returns
    -------
    str
        Path to the clipped GeoTIFF.
    """
    cache_dir = os.path.expanduser(cache_dir)
    
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, limits=HTTP_LIMITS) as client:
        # Resolve study area
        if city is not None:
            study_area = await _geocode_async(city, client)
            if study_area is None:
                raise ValueError(f"City not found: {city}")
            slug = city.lower().replace(" ", "_")
        elif roi is not None:
            study_area = roi.to_crs("EPSG:4326")
            slug = "roi"
        else:
            raise ValueError("Provide either city or roi")
        
        # Cache clipped result
        bbox = study_area.total_bounds
        key = f"{slug}|{bbox}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        clip_path = os.path.join(cache_dir, f"clipped_{digest}.tif")
        
        if _is_valid_raster(clip_path):
            logger.info("Cache hit: %s", clip_path)
            return clip_path
        if os.path.exists(clip_path):
            try:
                os.remove(clip_path)
            except OSError:
                pass
        
        # Stream and crop
        logger.info("Streaming COG and cropping...")
        try:
            # Try direct COG streaming first
            arr, profile = stream_cog_window(url, tuple(bbox))
            
            # Mask to exact boundary
            geom = study_area.geometry.union_all()
            from rasterio.features import rasterize
            mask_arr = rasterize(
                [(geom, 1)],
                out_shape=arr.shape,
                transform=profile["transform"],
                fill=0,
                dtype=np.uint8,
            )
            arr = np.where(mask_arr == 1, arr, 0)
            
            # Write to cache
            os.makedirs(cache_dir, exist_ok=True)
            out_meta = {
                "driver": "GTiff",
                "dtype": "uint8",
                "nodata": 0,
                "width": arr.shape[1],
                "height": arr.shape[0],
                "count": 1,
                "crs": profile["crs"],
                "transform": profile["transform"],
                "compress": "lzw",
            }
            temp_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
            with rasterio.open(temp_path, "w", **out_meta) as dst:
                dst.write(arr.astype(np.uint8), 1)
            _atomic_copy_raster(temp_path, clip_path)
            try:
                os.remove(temp_path)
            except OSError:
                pass
                
        except Exception as e:
            logger.warning(f"COG streaming failed ({e}), falling back to /vsicurl/ mask")
            clip_path = crop_mask_fast(url, study_area)
    
    if isave_map:
        from LCZ4py.general.lcz_plot_map import OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "lcz_map.tif")
        _atomic_copy_raster(clip_path, save_path)
    
    return clip_path


# Synchronous wrapper for backward compatibility
def lcz_get_map_fast(
    city: Optional[str] = None,
    roi: Optional[gpd.GeoDataFrame] = None,
    url: str = "https://zenodo.org/records/8419340/files/lcz_filter_v3.tif?download=1",
    cache_dir: str = "~/.lcz4r_cache",
    isave_map: bool = False,
) -> str:
    """Synchronous wrapper for the async downloader."""
    return asyncio.run(
        lcz_get_map_async(city, roi, url, cache_dir, isave_map)
    )


__all__ = [
    "lcz_get_map_async", 
    "lcz_get_map_fast", 
    "stream_cog_window",
    "crop_mask_fast"
]
