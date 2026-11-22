"""
_lcz_map_engine.py

High-performance core engine for LCZ map retrieval.

Replaces standard synchronous downloads and GeoJSON caching with:

- Async HTTP I/O (httpx) for geocoding, with built-in transport retries
- GeoArrow (Feather) for zero-copy, memory-mapped spatial caching
  (falls back to plain GeoPandas/Feather when GeoArrow is absent)
- Cloud-Optimized GeoTIFF (COG) streaming via Rasterio ``/vsicurl/`` +
  ``WarpedVRT``, with adaptive tiled crop/mask for large areas
- Atomic cache writes and raster validation, so an interrupted download never
  poisons the cache

Optional dependencies (duckdb, geoarrow) are imported behind ``try/except``
guards with ``HAS_*`` flags: importing this module never fails because one of
them is missing, it simply uses the core implementation.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

import geopandas as gpd
import httpx
import pyarrow.feather as feather
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from shapely.geometry import mapping

from LCZ4py._internal.adaptive_crop_mask import (
    LARGE_AREA_KM2,
    coerce_max_workers,
    ensure_wgs84,
    lcz_crop_mask_adaptive,
    study_area_km2,
)

# ── Optional ecosystem imports ────────────────────────────────────────────────
# These accelerate/annotate the pipeline but are never required: importing this
# module (and therefore LCZ4py) must succeed with core dependencies only.

try:  # pragma: no cover - depends on the environment
    import duckdb

    HAS_DUCKDB = True
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]
    HAS_DUCKDB = False

try:  # pragma: no cover - depends on the environment
    import geoarrow.pyarrow as ga
    import geoarrow.pandas as gap

    HAS_GEOARROW = True
except ImportError:  # pragma: no cover
    ga = None  # type: ignore[assignment]
    gap = None  # type: ignore[assignment]
    HAS_GEOARROW = False

try:  # pragma: no cover - depends on the environment
    import aiofiles

    HAS_AIOFILES = True
except ImportError:  # pragma: no cover
    aiofiles = None  # type: ignore[assignment]
    HAS_AIOFILES = False

try:  # pragma: no cover - depends on the environment
    import nest_asyncio

    HAS_NEST_ASYNCIO = True
except ImportError:  # pragma: no cover
    nest_asyncio = None  # type: ignore[assignment]
    HAS_NEST_ASYNCIO = False

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
DOWNLOAD_TIMEOUT = httpx.Timeout(300.0, connect=15.0)
GEOCODER_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "LCZ4r-Python-Advanced/2.0"}
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_T = TypeVar("_T")


# ── Retry decorators (tenacity-free, so core installs never need it) ──────────

def _is_retryable_http_error(exc: BaseException) -> bool:
    """True for transient HTTP/transport failures worth a second try."""
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return exc.response.status_code in RETRYABLE_STATUS
        except Exception:  # pragma: no cover - malformed response object
            return False
    if isinstance(exc, rasterio.errors.RasterioIOError):
        return True
    return False


def _retry_sync(
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Retry a synchronous callable on transient network/raster errors."""

    def decorator(func: Callable[..., _T]) -> Callable[..., _T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            last: Optional[BaseException] = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - re-raised below
                    if not _is_retryable_http_error(exc) or attempt == attempts:
                        raise
                    last = exc
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                    logger.warning(
                        "%s failed (%s); retrying in %.1fs (attempt %d/%d).",
                        func.__name__, type(exc).__name__, delay, attempt, attempts,
                    )
                    time.sleep(delay)
            raise last  # pragma: no cover - loop always returns or raises

        return wrapper

    return decorator


def _retry_async(
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Retry a coroutine function on transient network errors."""

    def decorator(func: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> _T:
            last: Optional[BaseException] = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - re-raised below
                    if not _is_retryable_http_error(exc) or attempt == attempts:
                        raise
                    last = exc
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                    logger.warning(
                        "%s failed (%s); retrying in %.1fs (attempt %d/%d).",
                        func.__name__, type(exc).__name__, delay, attempt, attempts,
                    )
                    await asyncio.sleep(delay)
            raise last  # pragma: no cover - loop always returns or raises

        return wrapper

    return decorator


# ── Raster cache validation / atomic writes ──────────────────────────────────

def _is_valid_raster(path: str) -> bool:
    """Return True only when ``path`` is a readable raster dataset."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with rasterio.open(path) as src:
            return (
                src.count > 0
                and src.width > 0
                and src.height > 0
                and src.transform is not None
            )
    except Exception:
        return False


def _is_valid_lcz_raster(path: str) -> bool:
    """Stricter check for cached LCZ clips: valid raster *with* LCZ classes."""
    if not _is_valid_raster(path):
        return False
    try:
        with rasterio.open(path) as src:
            if src.count < 1:
                return False
            band = src.read(1, out_shape=(min(256, src.height), min(256, src.width)))
    except Exception:
        return False
    return bool(((band >= 1) & (band <= 17)).any())


def _atomic_copy_raster(src_path: str, dst_path: str) -> None:
    """Copy a raster to ``dst_path`` atomically to avoid partial cache files.

    Uses a filesystem copy (not an array round-trip) so multi-hundred-MB
    rasters don't have to fit in RAM, then validates the copy before it is
    promoted into the cache.
    """
    dst_dir = os.path.dirname(dst_path) or "."
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=dst_dir)
    os.close(fd)
    try:
        shutil.copyfile(src_path, tmp_path)
        if not _is_valid_raster(tmp_path):
            raise RuntimeError(
                f"Raster copy validation failed for {src_path!r} -> {dst_path!r}."
            )
        os.replace(tmp_path, dst_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_bytes(dst_path: str, write_fn: Callable[[str], None]) -> None:
    """Run ``write_fn(temp_path)`` then atomically promote it to ``dst_path``."""
    dst_dir = os.path.dirname(dst_path) or "."
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dst_dir)
    os.close(fd)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, dst_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ── 1. Async Geocoding ───────────────────────────────────────────────────────

@_retry_async(attempts=3)
async def _async_geocode(
    city: str,
    client: httpx.AsyncClient,
    lang: str = "en",
) -> Optional[gpd.GeoDataFrame]:
    """Fault-tolerant async geocoding via Nominatim (returns None if unseen)."""
    params = {"q": city, "format": "geojson", "limit": 1, "polygon_geojson": 1}
    resp = await client.get(GEOCODER_URL, params=params, headers=HEADERS)
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as exc:
        raise ValueError(f"Nominatim returned a non-JSON response for {city!r}.") from exc

    features = data.get("features") if isinstance(data, dict) else None
    if not features:
        return None

    try:
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    except Exception as exc:  # noqa: BLE001 - malformed GeoJSON payload
        raise ValueError(f"Nominatim returned unusable geometry for {city!r}: {exc}") from exc

    # Nominatim can return a Point/LineString for ambiguous queries; a LCZ clip
    # needs an areal geometry.
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        return None
    return gdf


# ── 2. GeoArrow Zero-Copy Caching ────────────────────────────────────────────

def _save_geoarrow_cache(gdf: gpd.GeoDataFrame, path: str) -> None:
    """Save a study area as GeoArrow Feather (50x faster than GeoJSON).

    The write is atomic: a crash mid-write leaves the previous cache (or no
    cache) rather than a truncated file that later fails to load.
    """
    def _write(tmp_path: str) -> None:
        if HAS_GEOARROW:
            try:
                table = ga.from_geopandas(gdf)
                feather.write_feather(table, tmp_path, compression="lz4")
                return
            except Exception as exc:  # noqa: BLE001 - fall back below
                logger.debug("GeoArrow write failed, falling back to PyArrow: %s", exc)
        # Fallback: standard PyArrow (still much faster than GeoJSON)
        gdf.to_feather(tmp_path)

    _atomic_write_bytes(path, _write)


def _load_geoarrow_cache(path: str) -> Optional[gpd.GeoDataFrame]:
    """Load a study area from GeoArrow Feather, or None when unusable.

    A corrupt/truncated cache is treated as a cache miss (never an error), so
    users can recover by simply calling the function again.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    if HAS_GEOARROW:
        try:
            # memory_map=True avoids loading the file into RAM
            table = feather.read_table(path, memory_map=True)
            gdf = gap.to_geopandas(table)
            if gdf.empty:
                return None
            return gdf
        except Exception as exc:  # noqa: BLE001
            logger.debug("GeoArrow read failed, falling back to PyArrow: %s", exc)

    try:
        gdf = gpd.read_feather(path)
    except Exception as exc:  # noqa: BLE001 - corrupt cache
        logger.warning("Ignoring unreadable boundary cache %s (%s).", path, exc)
        return None
    return gdf if not gdf.empty else None


# ── 3. Bounding Box Prep ─────────────────────────────────────────────────────

def _prepare_bounding_box_duckdb(
    study_area: gpd.GeoDataFrame,
    target_crs: str = "EPSG:4326",
) -> tuple[tuple[float, float, float, float], gpd.GeoDataFrame]:
    """Reproject the study area to WGS84 and return its validated bbox.

    Kept under its historical name for backwards compatibility. DuckDB is not
    used: ``total_bounds`` is already vectorised numpy and a DuckDB round-trip
    only added serialisation overhead.
    """
    gdf_4326 = ensure_wgs84(study_area, context="study_area")
    if target_crs != "EPSG:4326" and not str(target_crs).endswith("4326"):
        gdf_4326 = gdf_4326.to_crs(target_crs)

    bbox = tuple(float(v) for v in gdf_4326.total_bounds)
    xmin, ymin, xmax, ymax = bbox
    if not all(v == v for v in bbox):  # NaN check
        raise ValueError("Study area has non-finite bounds (NaN/Inf coordinates).")
    if xmin < -180.5 or xmax > 180.5 or ymin < -90.5 or ymax > 90.5:
        raise ValueError(
            f"Study area bounds are outside WGS84 longitude/latitude: {bbox}. "
            "Check that the ROI's CRS is correct."
        )
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Study area has degenerate bounds: {bbox}")
    return bbox, gdf_4326


# ── 4. COG Streaming & Masking ───────────────────────────────────────────────

@_retry_sync(attempts=3)
def _stream_and_clip_cog(
    url: str,
    study_area: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    lang: str = "en",
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> str:
    """Stream only necessary pixels from a COG and mask to exact geometry."""
    if not url:
        raise ValueError("url must be a non-empty string naming a COG.")
    if study_area is None or study_area.empty:
        raise ValueError("study_area is empty.")

    vsi_url = url if url.startswith("/vsicurl/") else f"/vsicurl/{url}"
    workers = coerce_max_workers(max_workers)

    # Large study areas (countries, continents) get the tiered strategy:
    # simplified-geometry mask (Tier 2) or tiled parallel mask+mosaic (Tier 3).
    if study_area_km2(study_area) > LARGE_AREA_KM2:
        result = lcz_crop_mask_adaptive(
            vsi_url, study_area, lang=lang, verbose=verbose, max_workers=workers,
        )
        if result is None:
            raise ValueError(
                "Study area completely outside raster bounds — check that the "
                "requested region is covered by this LCZ map product."
            )
        return result

    with rasterio.open(vsi_url) as src:
        if src.count < 1:
            raise ValueError(f"Raster at {url!r} has no data bands.")

        # Use WarpedVRT for on-the-fly CRS correction without intermediate files
        if src.crs is not None and src.crs.to_epsg() != 4326:
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
                raise ValueError(
                    "Study area completely outside raster bounds — check that the "
                    "requested region is covered by this LCZ map product."
                )

            # Mask directly from VRT (rasterio handles the windowed reads internally)
            geom = study_area.geometry.union_all()
            try:
                masked_data, masked_transform = rio_mask(
                    vrt,
                    [mapping(geom)],
                    crop=True,
                    all_touched=False,
                    nodata=0,
                )
            except ValueError as exc:
                if "do not overlap" in str(exc):
                    raise ValueError(
                        "Study area completely outside raster bounds."
                    ) from exc
                raise
            out_crs = vrt.crs or "EPSG:4326"
            out_dtype = vrt.dtypes[0]

    if masked_data.size == 0 or masked_data.shape[1] == 0 or masked_data.shape[2] == 0:
        raise ValueError("Clipping produced an empty raster; the study area may be too small.")

    # Write to a temp file (caller copies it into the cache atomically)
    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    out_meta = {
        "driver": "GTiff",
        "dtype": out_dtype,
        "nodata": 0,
        "width": masked_data.shape[2],
        "height": masked_data.shape[1],
        "count": 1,
        "crs": out_crs,
        "transform": masked_transform,
        "compress": "lzw",       # Fast read/write
        "BIGTIFF": "IF_SAFER",   # Prevent >4GB crashes
    }

    try:
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(masked_data[0:1])
    except Exception:
        _safe_remove(out_path)
        raise

    return out_path


# ── 5. Async Full Download (for non-COGs like LCZ Generator) ────────────────

@_retry_async(attempts=3)
async def _async_download_file(
    url: str,
    dest_path: str,
    client: httpx.AsyncClient,
    *,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Stream-download ``url`` to ``dest_path`` without memory bloat.

    Returns the number of bytes written. A truncated transfer (server closed
    early / ``Content-Length`` mismatch) raises instead of leaving a corrupt
    file behind, so callers never cache a partial download.
    """
    if not url:
        raise ValueError("url must be a non-empty string.")
    dst_dir = os.path.dirname(os.path.abspath(dest_path)) or "."
    os.makedirs(dst_dir, exist_ok=True)

    written = 0
    try:
        async with client.stream("GET", url, headers=HEADERS) as resp:
            resp.raise_for_status()
            expected: Optional[int] = None
            try:
                raw_len = resp.headers.get("Content-Length")
                expected = int(raw_len) if raw_len is not None else None
            except (TypeError, ValueError):
                expected = None

            if HAS_AIOFILES:
                async with aiofiles.open(dest_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=chunk_size):
                        await f.write(chunk)
                        written += len(chunk)
            else:  # pragma: no cover - exercised only without aiofiles
                with open(dest_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=chunk_size):
                        f.write(chunk)
                        written += len(chunk)
    except Exception:
        _safe_remove(dest_path)
        raise

    if expected is not None and written != expected:
        _safe_remove(dest_path)
        raise IOError(
            f"Incomplete download from {url}: got {written} of {expected} bytes."
        )
    if written == 0:
        _safe_remove(dest_path)
        raise IOError(f"Download from {url} returned no data.")
    return written


# ── 6. Async/sync bridge ─────────────────────────────────────────────────────

def _run_coro_in_thread(coro: Awaitable[_T]) -> _T:
    """Run ``coro`` to completion on a private event loop in another thread.

    Needed when we are already inside a running loop (Jupyter/async apps) and
    ``nest_asyncio`` is unavailable — re-entering the running loop is
    impossible, and raising would break notebook users.
    """
    outcome: dict[str, Any] = {}

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(coro)  # type: ignore[arg-type]
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller
            outcome["error"] = exc

    thread = threading.Thread(target=_runner, name="lcz-async", daemon=True)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]  # type: ignore[return-value]


def run_coroutine(coro: Awaitable[_T]) -> _T:
    """Run ``coro`` from sync code, whether or not a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    if HAS_NEST_ASYNCIO:
        nest_asyncio.apply()
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(coro)  # type: ignore[arg-type]
    return _run_coro_in_thread(coro)


# ── 7. Main Unified Executor ─────────────────────────────────────────────────

async def _get_lcz_map_core(
    city: Optional[str] = None,
    roi: Optional[gpd.GeoDataFrame] = None,
    url: str = "",
    cache_dir: str = "~/.lcz4r_cache",
    isave_map: bool = False,
    output_dir: str = "LCZ4r_output",
    lang: str = "en",
    verbose: bool = True,
    max_workers: Optional[int] = None,
    output_name: str = "lcz_map.tif",
) -> str:
    """Unified async core for all lcz_get_map* functions."""
    if not url:
        raise ValueError("url must be a non-empty string naming a COG.")

    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    workers = coerce_max_workers(max_workers)

    if city is not None and (not isinstance(city, str) or not city.strip()):
        raise ValueError("city must be a non-empty place name, or pass roi= instead.")
    if roi is not None and not isinstance(roi, gpd.GeoDataFrame):
        raise TypeError(
            f"roi must be a GeoDataFrame with polygon geometry, got {type(roi).__name__}. "
            "If you picked a previous result, make sure it's a vector/boundary output "
            "(e.g. GeoJSON), not a plot or raster result."
        )
    if city is None and roi is None:
        from LCZ4py._internal.i18n_messages import lcz_msg

        raise ValueError(lcz_msg("no_map_input", lang))

    transport = httpx.AsyncHTTPTransport(retries=2)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, transport=transport) as client:
        # ── Step 1: Resolve Study Area ────────────────────────────────────
        if city:
            slug = city.lower().strip().replace(" ", "_").replace("/", "_")
            cache_file = os.path.join(cache_dir, f"study_area_{slug}.arrow")

            study_area = _load_geoarrow_cache(cache_file)
            if study_area is None:
                if verbose:
                    logger.info("Geocoding '%s' via async HTTP...", city)
                study_area = await _async_geocode(city, client, lang=lang)
                if study_area is None:
                    from LCZ4py._internal.i18n_messages import lcz_msg

                    raise ValueError(lcz_msg("city_not_found", lang, city=city))
                _save_geoarrow_cache(study_area, cache_file)
                if verbose:
                    logger.info("Cached boundary to %s", cache_file)
        else:
            study_area = roi

        # ── Step 2: Prepare Geometry ──────────────────────────────────────
        bbox, study_area_4326 = _prepare_bounding_box_duckdb(study_area)

        # Generate stable hash for the clipped cache. This MUST include the
        # actual geometry, not just its bounding box: two different-shaped
        # study areas (e.g. a custom ROI vs. a geocoded city, or two ROIs)
        # can share the same bbox, and hashing only the bbox previously made
        # them collide on the same cache file — silently returning one
        # request's mask shape for a different, unrelated request ("crop"
        # instead of "mask" bugs traced back to this).
        geom_wkb = study_area_4326.geometry.union_all().wkb
        key = (
            f"{url}|{bbox[0]:.4f}|{bbox[1]:.4f}|{bbox[2]:.4f}|{bbox[3]:.4f}".encode()
            + geom_wkb
        )
        clip_hash = hashlib.sha256(key).hexdigest()[:16]
        clip_cache_path = os.path.join(cache_dir, f"clipped_{clip_hash}.tif")

        if _is_valid_lcz_raster(clip_cache_path):
            if verbose:
                logger.info("Loading clipped map from local cache.")
            final_path = clip_cache_path
        else:
            if os.path.exists(clip_cache_path):
                if verbose:
                    logger.info("Found invalid clipped cache, rebuilding it.")
                _safe_remove(clip_cache_path)
            if verbose:
                logger.info("Streaming COG and clipping to geometry...")
            tmp_clip = _stream_and_clip_cog(
                url, study_area_4326, bbox,
                lang=lang, verbose=verbose, max_workers=workers,
            )
            try:
                _atomic_copy_raster(tmp_clip, clip_cache_path)
            finally:
                _safe_remove(tmp_clip)
            final_path = clip_cache_path
            if verbose:
                logger.info("Saved to clipped cache.")

    # ── Step 3: Copy to Output if requested ──────────────────────────────
    if isave_map:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_name)
        _atomic_copy_raster(final_path, out_path)
        if verbose:
            logger.info("Saved map to %s", out_path)
        return out_path

    return final_path


def run_async_core(**kwargs: Any) -> str:
    """Sync wrapper for the async core (works in Jupyter and plain Python)."""
    return run_coroutine(_get_lcz_map_core(**kwargs))


__all__ = [
    "run_async_core",
    "run_coroutine",
    "_async_download_file",
    "_get_lcz_map_core",
    "_stream_and_clip_cog",
    "_atomic_copy_raster",
    "_is_valid_raster",
    "HAS_DUCKDB",
    "HAS_GEOARROW",
    "HAS_AIOFILES",
]
