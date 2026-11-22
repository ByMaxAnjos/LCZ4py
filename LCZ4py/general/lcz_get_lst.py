"""
lcz_get_lst.py

Land Surface Temperature (LST) time series from Microsoft Planetary Computer,
cropped to an existing LCZ map's footprint.

Two sources, both keyless/anonymous:
- ``source="goes"``: GOES-R ABI-L2-LST (CONUS). Not STAC-registered — read
  directly from Azure Blob Storage (goeseuwest) using a public SAS token.
  Regular geostationary-projection grid; reprojected onto the LCZ grid.
- ``source="sentinel3"``: Sentinel-3 SLSTR LST, STAC-searchable on Planetary
  Computer. Curvilinear per-pixel lat/lon grid; regridded via nearest-
  neighbor (scipy.spatial.cKDTree, the same tool used in lcz_krige.py).
- ``source="both"``: per day, try GOES first (finer native resolution over
  the Americas), fall back to Sentinel-3.

Output is a multi-band GeoTIFF (one band per date, band description = ISO
date), float32 with NaN nodata, pixel-aligned to the input LCZ map's grid —
consistent with the ``LCZStackResult`` convention in lcz_get_parameters.py.

Only core dependencies are required at import time. ``pystac-client`` /
``planetary-computer`` (Sentinel-3) and ``xarray`` / ``rioxarray`` (scene
regridding) are optional and only raise an actionable ``ImportError`` when the
feature that needs them is called.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union
from urllib.parse import quote

import geopandas as gpd
import httpx
import numpy as np
import pyproj
import rasterio
from scipy.spatial import cKDTree

try:  # optional at import time; only scene regridding needs it
    import xarray as xr

    HAS_XARRAY = True
except ImportError:  # pragma: no cover - exercised by the no-optional-deps test
    xr = None  # type: ignore[assignment]
    HAS_XARRAY = False

try:
    import rioxarray  # noqa: F401 - registers the .rio accessor

    HAS_RIOXARRAY = True
except ImportError:  # pragma: no cover
    rioxarray = None  # type: ignore[assignment]
    HAS_RIOXARRAY = False

try:
    import pystac_client
    import planetary_computer

    HAS_STAC = True
except ImportError:
    pystac_client = None  # type: ignore[assignment]
    planetary_computer = None  # type: ignore[assignment]
    HAS_STAC = False

from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"
DEFAULT_CACHE_DIR = "~/.lcz4r_cache"

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL3_LST_COLLECTION = "sentinel-3-slstr-lst-l2-netcdf"

GOES_ACCOUNT = "goeseuwest"
GOES_SATELLITE_CONTAINERS = {
    "goes16": ("noaa-goes16", "G16"),
    "goes17": ("noaa-goes17", "G17"),
    "goes18": ("noaa-goes18", "G18"),
}
# ponytail: rough CONUS bounding box, not the true (irregular) ABI-L2-LSTC
# swath outline. Good enough to reject obviously-out-of-view requests
# (e.g. Berlin); refine with the exact scan geometry if edge cases matter.
GOES_CONUS_BBOX = (-135.0, 14.0, -60.0, 55.0)

# ~1km SLSTR footprint: reject nearest-neighbour matches further than ~0.02 deg
SENTINEL3_MAX_MATCH_DEG = 0.02
# Extra margin (degrees) kept around the LCZ bbox when pre-filtering the
# curvilinear SLSTR swath before building the KD-tree.
_SENTINEL3_BBOX_MARGIN_DEG = 0.1

# NetCDF/HDF5 engines tried in order: fast pure-pip reader first, then the
# classic netCDF4 backend, then whatever xarray picks by default.
_NETCDF_ENGINES: tuple[Optional[str], ...] = ("h5netcdf", "netcdf4", None)

DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 4)

_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=20.0)
_HTTP_HEADERS = {"User-Agent": "LCZ4r-Python/2.0 (lcz_get_lst)"}
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_HTTP_ATTEMPTS = 4
_DOWNLOAD_ATTEMPTS = 3

_CLIENT_LOCK = threading.Lock()
_HTTP_CLIENT: Optional[httpx.Client] = None

_SAS_TOKEN_CACHE: dict[tuple[str, str], tuple[str, datetime]] = {}
_SAS_TOKEN_LOCK = threading.Lock()


@dataclass
class LCZLSTResult:
    """Return type for lcz_get_lst."""

    path: Optional[str] = None
    array: Optional[np.ndarray] = None
    dates: Optional[list[str]] = None
    units: Optional[str] = None
    gdf: Optional[gpd.GeoDataFrame] = None
    geoarrow_table: Optional[object] = None


@dataclass
class _TargetGrid:
    shape: tuple[int, int]  # (H, W)
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    valid_mask: np.ndarray  # True where the LCZ map has data
    bbox_wgs84: tuple[float, float, float, float]


# ── Optional dependency / HTTP plumbing ───────────────────────────────────────


def _require_xarray() -> None:
    """Raise an actionable error only when a scene needs xarray/rioxarray."""
    if HAS_XARRAY and HAS_RIOXARRAY:
        return
    missing = []
    if not HAS_XARRAY:
        missing.append("xarray")
    if not HAS_RIOXARRAY:
        missing.append("rioxarray")
    raise ImportError(
        "Reading and regridding GOES/Sentinel-3 NetCDF scenes requires "
        f"{' and '.join(missing)} — install with "
        f"'pip install {' '.join(missing)}' (or 'pip install LCZ4py[lst]')."
    )


def _get_client() -> httpx.Client:
    """Return the shared HTTP client (created once, reused across threads)."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _CLIENT_LOCK:
            if _HTTP_CLIENT is None:
                _HTTP_CLIENT = httpx.Client(
                    timeout=_HTTP_TIMEOUT,
                    headers=_HTTP_HEADERS,
                    follow_redirects=True,
                    transport=httpx.HTTPTransport(retries=_HTTP_ATTEMPTS - 1),
                )
    return _HTTP_CLIENT


def close_http_client() -> None:
    """Close the module-level HTTP client (called automatically at exit)."""
    global _HTTP_CLIENT
    with _CLIENT_LOCK:
        client, _HTTP_CLIENT = _HTTP_CLIENT, None
    if client is not None:
        try:
            client.close()
        except Exception:  # pragma: no cover - best effort at interpreter exit
            pass


atexit.register(close_http_client)


def _backoff(attempt: int, base: float = 0.6) -> float:
    return min(20.0, base * (2 ** attempt))


def _retry_call(func, *, attempts: int = 2, base_delay: float = 1.0):
    """Call ``func`` with a small retry, for transient STAC/HTTP failures."""
    last_exc: Optional[BaseException] = None
    for attempt in range(max(1, int(attempts))):
        try:
            return func()
        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001 - retried, or re-raised below
            last_exc = exc
            if attempt + 1 < int(attempts):
                time.sleep(_backoff(attempt, base=base_delay))
    raise last_exc  # type: ignore[misc]


def _request(
    method: str,
    url: str,
    *,
    attempts: int = _HTTP_ATTEMPTS,
    timeout: Optional[httpx.Timeout] = None,
    **kwargs: Any,
) -> httpx.Response:
    """HTTP request with bounded retries and exponential backoff."""
    attempts = max(1, int(attempts))
    last_exc: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            resp = _get_client().request(
                method, url, timeout=timeout or _HTTP_TIMEOUT, **kwargs
            )
            if resp.status_code in _RETRY_STATUS:
                resp.close()
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code} for {url}",
                    request=resp.request,
                    response=resp,
                )
            else:
                resp.raise_for_status()
                return resp
        except (
            httpx.TransportError,
            httpx.TimeoutException,
            httpx.HTTPStatusError,
        ) as exc:
            last_exc = exc
        if attempt + 1 < attempts:
            time.sleep(_backoff(attempt))
    raise RuntimeError(f"request failed after {attempts} attempts: {url}") from last_exc


def _unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _cleanup_temp(path: Optional[str]) -> None:
    """Remove a file created by :func:`_download_to_temp` (and its temp dir)."""
    if not path:
        return
    parent = os.path.dirname(path)
    _unlink(path)
    base = os.path.basename(parent)
    if base.startswith("lcz_lst_") and os.path.isdir(parent):
        shutil.rmtree(parent, ignore_errors=True)


# ── LCZ map grid / geometry helpers ───────────────────────────────────────────


def _load_target_grid(x: Union[str, Path]) -> _TargetGrid:
    """Read an LCZ classification GeoTIFF and describe its grid."""
    path = os.path.expanduser(str(x))
    if not os.path.exists(path):
        raise FileNotFoundError(f"LCZ map not found: {path}")

    with rasterio.open(path) as src:
        class_arr = src.read(1)
        transform = src.transform
        crs = src.crs
        shape_hw = (src.height, src.width)
        bounds = src.bounds
        if src.count > 1:
            logger.debug("LCZ map has %d bands; using band 1 only.", src.count)

    if crs is None:
        logger.warning(
            "LCZ map %s has no CRS; assuming EPSG:4326 (WGS84).", path
        )
        crs = rasterio.crs.CRS.from_epsg(4326)

    # Integer/float class codes: 0/255/nodata and NaN all fail these
    # comparisons and correctly end up as invalid pixels.
    with np.errstate(invalid="ignore"):
        valid_mask = (class_arr >= 1) & (class_arr <= 17)
    if not np.any(valid_mask):
        logger.warning(
            "LCZ map %s contains no pixels in classes 1-17; the LST stack will "
            "be entirely NaN.",
            path,
        )

    if crs.to_epsg() == 4326:
        bbox_wgs84 = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    else:
        transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        x0, y0 = transformer.transform(bounds.left, bounds.bottom)
        x1, y1 = transformer.transform(bounds.right, bounds.top)
        bbox_wgs84 = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    return _TargetGrid(shape_hw, transform, crs, valid_mask, bbox_wgs84)


def _target_lonlat_grid(grid: _TargetGrid) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-center (lon, lat) for every cell of the target grid, as (H, W) arrays."""
    h, w = grid.shape
    t = grid.transform

    if t.b == 0 and t.d == 0:
        # North-up grid: pixel centers are a simple affine function of the
        # indices, which is orders of magnitude cheaper than transform.xy for
        # large grids (and this is called once per scene otherwise).
        xs = t.c + (np.arange(w) + 0.5) * t.a
        ys = t.f + (np.arange(h) + 0.5) * t.e
        xx, yy = np.meshgrid(xs, ys)
    else:
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        xx, yy = rasterio.transform.xy(t, rows.ravel(), cols.ravel())
        xx, yy = np.asarray(xx).reshape(h, w), np.asarray(yy).reshape(h, w)

    if grid.crs.to_epsg() != 4326:
        transformer = pyproj.Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        xx, yy = transformer.transform(xx, yy)
        xx, yy = np.asarray(xx), np.asarray(yy)

    return xx.astype(np.float64, copy=False), yy.astype(np.float64, copy=False)


def _rio_template(grid: _TargetGrid) -> "xr.DataArray":
    """Empty DataArray carrying the target grid's crs/transform, for reproject_match."""
    _require_xarray()
    template = xr.DataArray(
        np.zeros(grid.shape, dtype="float32"),
        dims=("y", "x"),
    )
    template = template.rio.write_crs(grid.crs)
    template = template.rio.write_transform(grid.transform)
    return template


def _parse_iso_date(value: Any, name: str) -> date:
    """Parse an ISO ``YYYY-MM-DD`` string (or a full ISO datetime)."""
    text = str(value).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an ISO date ('YYYY-MM-DD'), got {value!r}"
        ) from exc


def _daterange(start_date: str, end_date: str) -> list[date]:
    start = _parse_iso_date(start_date, "start_date")
    end = _parse_iso_date(end_date, "end_date")
    if end < start:
        raise ValueError("end_date must not be before start_date")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _cache_key(
    source: str,
    satellite: str,
    grid: _TargetGrid,
    start_date: str,
    end_date: str,
    target_hour: float,
) -> str:
    bbox = ",".join(f"{v:.4f}" for v in grid.bbox_wgs84)
    raw = f"{source}|{satellite}|{bbox}|{start_date}|{end_date}|{target_hour}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _coerce_workers(max_workers: Any) -> int:
    try:
        n = int(max_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_workers must be a positive integer") from exc
    if n < 1:
        raise ValueError("max_workers must be a positive integer")
    return n


def _is_valid_raster(path: Union[str, Path]) -> bool:
    """Return True only when ``path`` is a readable raster dataset."""
    path = str(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with rasterio.open(path) as src:
            return src.count > 0 and src.width > 0 and src.height > 0
    except Exception:
        return False


def _atomic_copy_raster(src_path: Union[str, Path], dst_path: Union[str, Path]) -> None:
    """Copy a raster to ``dst_path`` atomically to avoid partial files."""
    dst_path = str(dst_path)
    dst_dir = os.path.dirname(dst_path) or "."
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=dst_dir)
    os.close(fd)
    try:
        with rasterio.open(str(src_path)) as src, rasterio.open(
            tmp_path, "w", **src.profile
        ) as dst:
            dst.write(src.read())
            dst.update_tags(**src.tags())
        os.replace(tmp_path, dst_path)
    except Exception:
        _unlink(tmp_path)
        raise


# ── GOES ingestion ────────────────────────────────────────────────────────────


def _parse_sas_expiry(raw: Any) -> datetime:
    """Parse the ``msft:expiry`` timestamp into an aware UTC datetime."""
    now = datetime.now(timezone.utc)
    if not raw:
        return now + timedelta(minutes=10)
    try:
        expiry = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    except ValueError:
        return now + timedelta(minutes=10)
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return expiry.astimezone(timezone.utc)


def _get_sas_token(account: str, container: str, *, force_refresh: bool = False) -> str:
    """Return a (cached, thread-safe) public SAS token for an Azure container."""
    key = (account, container)
    now = datetime.now(timezone.utc)

    if not force_refresh:
        with _SAS_TOKEN_LOCK:
            cached = _SAS_TOKEN_CACHE.get(key)
            if cached and cached[1] > now:
                return cached[0]

    # Refresh under the lock so a burst of worker threads issues one request.
    with _SAS_TOKEN_LOCK:
        cached = _SAS_TOKEN_CACHE.get(key)
        if not force_refresh and cached and cached[1] > now:
            return cached[0]

        try:
            resp = _request(
                "GET",
                f"{PC_STAC_URL}/sas/v1/token/{account}/{container}",
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"Could not obtain a SAS token for {account}/{container}: {exc}"
            ) from exc

        token = data.get("token")
        if not token:
            raise RuntimeError(
                f"Planetary Computer returned no SAS token for {account}/{container}."
            )
        # Refresh two minutes early so no in-flight read uses an expired token.
        expiry = _parse_sas_expiry(data.get("msft:expiry")) - timedelta(minutes=2)
        _SAS_TOKEN_CACHE[key] = (str(token), expiry)
        return str(token)


def _list_blobs(
    account: str,
    container: str,
    prefix: str,
    token: str,
    *,
    max_pages: int = 50,
    timeout: float = 30.0,
) -> list[str]:
    """List blob names under ``prefix`` (paginated, retried)."""
    names: list[str] = []
    marker = ""
    base = f"https://{account}.blob.core.windows.net/{container}"
    seen_markers: set[str] = set()

    for _ in range(max_pages):
        params = f"restype=container&comp=list&prefix={quote(prefix, safe='/')}"
        if marker:
            params += f"&marker={quote(marker, safe='')}"
        url = f"{base}?{params}&{token.lstrip('&')}"
        xml = _request(
            "GET", url, timeout=httpx.Timeout(timeout, connect=10.0)
        ).text

        names.extend(re.findall(r"<Name>([^<]+)</Name>", xml))
        match = re.search(r"<NextMarker>([^<]*)</NextMarker>", xml)
        marker = match.group(1).strip() if match else ""
        if not marker or marker in seen_markers:
            break
        seen_markers.add(marker)
    else:  # pragma: no cover - defensive: pagination did not terminate
        logger.warning(
            "Blob listing for %s/%s hit the page cap (%d); results may be truncated.",
            container,
            prefix,
            max_pages,
        )

    return names


def _download_to_temp(
    url: str, suffix: str, *, timeout: float = 300.0
) -> str:
    """Stream ``url`` to a fresh temp file with retries; return the local path."""
    dir_path = tempfile.mkdtemp(prefix="lcz_lst_")
    path = os.path.join(dir_path, f"scene{suffix}")
    attempts = _DOWNLOAD_ATTEMPTS
    last_exc: Optional[BaseException] = None

    for attempt in range(attempts):
        written = 0
        try:
            with _get_client().stream(
                "GET", url, timeout=httpx.Timeout(timeout, connect=30.0)
            ) as resp:
                resp.raise_for_status()
                with open(path, "wb") as fh:
                    for chunk in resp.iter_bytes(1024 * 256):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)
            if written == 0:
                raise RuntimeError("empty response body")
            return path
        except Exception as exc:  # noqa: BLE001 - retried below
            last_exc = exc
            _unlink(path)
            if attempt + 1 < attempts:
                time.sleep(_backoff(attempt, base=1.0))

    shutil.rmtree(dir_path, ignore_errors=True)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last_exc


def _goes_in_view(grid: _TargetGrid) -> bool:
    x0, y0, x1, y1 = grid.bbox_wgs84
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0, bx1, by1 = GOES_CONUS_BBOX
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def _pick_goes_hour_file(
    container: str,
    sat_code: str,
    day: date,
    target_hour: float,
) -> Optional[str]:
    """Closest available GOES LSTC file for ``day`` around ``target_hour``."""
    token = _get_sas_token(GOES_ACCOUNT, container)
    doy = day.timetuple().tm_yday
    base_hour = int(round(target_hour))

    for offset in (0, -1, 1, -2, 2, -3, 3):
        hour = base_hour + offset
        if not (0 <= hour <= 23):
            continue
        prefix = f"ABI-L2-LSTC/{day.year}/{doy:03d}/{hour:02d}/"
        try:
            files = _list_blobs(GOES_ACCOUNT, container, prefix, token)
        except Exception as exc:  # noqa: BLE001 - try the next candidate hour
            logger.debug("GOES listing failed for %s (hour %02d): %s", day, hour, exc)
            continue
        files = sorted(f for f in files if sat_code in f)
        if files:
            return files[0]
    return None


# ── NetCDF/HDF5 opening with engine fallback ─────────────────────────────────


def _open_dataset(path: str, engine: Optional[str]):
    """Open a NetCDF/HDF5 file with one engine (caller closes the dataset)."""
    _require_xarray()
    if engine is None:
        return xr.open_dataset(path)
    return xr.open_dataset(path, engine=engine)


def _first_dataarray(ds, candidates: Sequence[str]):
    for name in candidates:
        if name in ds.variables:
            return ds[name]
    available = ", ".join(sorted(map(str, ds.variables)))
    raise KeyError(
        f"none of {list(candidates)} found in the scene (available: {available})"
    )


def _drop_singleton_dims(da):
    """Drop size-1 dimensions (e.g. a ``time`` axis) from a scene DataArray."""
    squeeze = [d for d in da.dims if da.sizes[d] == 1]
    if squeeze:
        da = da.isel({d: 0 for d in squeeze}, drop=True)
    return da


def _scene_2d(da):
    """Return a 2-D scene, raising a clear error for unsupported shapes."""
    da = _drop_singleton_dims(da)
    if da.ndim != 2:
        raise ValueError(
            f"scene variable is {da.ndim}-D {da.dims}; expected a 2-D raster"
        )
    return da


def _goes_ds_to_grid(ds, grid: _TargetGrid) -> np.ndarray:
    """Regrid one open GOES ABI-L2-LST dataset onto the LCZ grid."""
    _require_xarray()

    if "goes_imager_projection" not in ds.variables:
        raise ValueError("not a GOES ABI-L2 LST file (no 'goes_imager_projection')")
    proj = dict(ds["goes_imager_projection"].attrs)
    required = (
        "perspective_point_height",
        "semi_major_axis",
        "semi_minor_axis",
        "longitude_of_projection_origin",
    )
    missing = [k for k in required if k not in proj]
    if missing:
        raise ValueError(f"GOES projection attributes missing: {missing}")

    height = float(proj["perspective_point_height"])
    lst = _first_dataarray(ds, ("LST", "lst", "LST_in", "lst_in"))

    if "DQF" in ds.variables:
        dqf = ds["DQF"]
        try:
            lst = lst.where(dqf == 0)  # keep only good-quality retrievals
        except Exception:  # noqa: BLE001 - shapes may differ in odd products
            logger.debug("GOES DQF mask could not be aligned; using all pixels.")
    lst = lst.where(np.isfinite(lst))
    lst = _scene_2d(lst)

    x = np.asarray(ds["x"].values, dtype="float64") * height
    y = np.asarray(ds["y"].values, dtype="float64") * height
    da = lst.assign_coords(x=("x", x), y=("y", y))

    geos_crs = pyproj.CRS.from_dict(
        {
            "proj": "geos",
            "h": height,
            "a": float(proj["semi_major_axis"]),
            "b": float(proj["semi_minor_axis"]),
            "lon_0": float(proj["longitude_of_projection_origin"]),
            "sweep": proj.get("sweep_angle_axis", "x"),
            "units": "m",
        }
    )
    da = da.rio.write_crs(geos_crs)
    da.rio.write_transform(inplace=True)
    da = da.rio.write_nodata(np.nan)

    regridded = da.rio.reproject_match(
        _rio_template(grid), resampling=rasterio.enums.Resampling.bilinear
    )
    return np.asarray(regridded.values, dtype=np.float32)


def _goes_scene_to_grid(nc_path: str, grid: _TargetGrid) -> np.ndarray:
    """Read a downloaded GOES scene, falling back between NetCDF engines."""
    _require_xarray()
    errors: list[str] = []
    for engine in _NETCDF_ENGINES:
        ds = None
        try:
            ds = _open_dataset(nc_path, engine)
            return _goes_ds_to_grid(ds, grid)
        except Exception as exc:  # noqa: BLE001 - try the next engine
            errors.append(f"{engine or 'default'}: {exc}")
        finally:
            if ds is not None:
                try:
                    ds.close()
                except Exception:  # pragma: no cover - best effort
                    pass
    raise RuntimeError(
        "Could not read GOES scene (tried engines: " + "; ".join(errors) + ")"
    )


def _ingest_goes(
    grid: _TargetGrid,
    satellite: str,
    start_date: str,
    end_date: str,
    target_hour: float,
    lang: str,
    verbose: bool,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, np.ndarray]:
    """Fetch/regrid one GOES scene per day, in parallel, ordered by date."""
    if not _goes_in_view(grid):
        raise ValueError(lcz_msg("lst_source_out_of_bounds", lang))
    _require_xarray()

    container, sat_code = GOES_SATELLITE_CONTAINERS[satellite]
    days = _daterange(start_date, end_date)
    workers = min(_coerce_workers(max_workers), max(1, len(days)))

    # Fail fast with an actionable error if the container token is unavailable.
    _get_sas_token(GOES_ACCOUNT, container)

    def _fetch_day(day: date) -> tuple[str, Optional[np.ndarray], Optional[str]]:
        day_key = day.isoformat()
        try:
            blob_name = _pick_goes_hour_file(container, sat_code, day, target_hour)
        except Exception as exc:  # noqa: BLE001 - day is skipped, not fatal
            return day_key, None, f"listing failed: {exc}"
        if blob_name is None:
            return day_key, None, "no scene for the requested hour"

        tmp_path: Optional[str] = None
        try:
            token = _get_sas_token(GOES_ACCOUNT, container)
            url = (
                f"https://{GOES_ACCOUNT}.blob.core.windows.net/"
                f"{container}/{blob_name}?{token}"
            )
            tmp_path = _download_to_temp(url, ".nc")
            arr = _goes_scene_to_grid(tmp_path, grid)
        except Exception as exc:  # noqa: BLE001 - day is skipped, not fatal
            return day_key, None, str(exc)
        finally:
            _cleanup_temp(tmp_path)

        if not np.isfinite(arr).any():
            return day_key, None, "scene has no valid pixels over this area"
        return day_key, arr, None

    out: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lcz-goes") as pool:
        for day_key, arr, reason in pool.map(_fetch_day, days):
            if arr is None:
                if verbose:
                    logger.warning(
                        "%s (%s)",
                        lcz_msg("lst_day_skipped", lang, date=day_key),
                        reason,
                    )
                continue
            out[day_key] = arr

    return dict(sorted(out.items()))


# ── Sentinel-3 ingestion ──────────────────────────────────────────────────────


def _sentinel3_ds_to_grid(
    lst_ds,
    geo_ds,
    grid: _TargetGrid,
    target_lonlat: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Nearest-neighbour regrid of a curvilinear SLSTR swath onto the LCZ grid."""
    lst_da = _first_dataarray(lst_ds, ("LST", "lst", "LST_in", "lst_in"))
    lon_da = _first_dataarray(
        geo_ds, ("longitude_in", "longitude", "lon_in", "lon")
    )
    lat_da = _first_dataarray(geo_ds, ("latitude_in", "latitude", "lat_in", "lat"))

    lst_raw = np.asarray(lst_da.values)
    lst = lst_raw.astype(np.float64, copy=False)
    lon = np.asarray(lon_da.values, dtype=np.float64)
    lat = np.asarray(lat_da.values, dtype=np.float64)

    # Some SLSTR products are integer-encoded with scale/offset that the
    # chosen NetCDF engine may not have decoded.
    if lst_raw.dtype.kind in "iu":
        scale = float(lst_da.attrs.get("scale_factor", 1.0))
        offset = float(lst_da.attrs.get("add_offset", 0.0))
        lst = lst * scale + offset

    # A leading singleton axis (e.g. ``time``) is harmless; anything else
    # means the LST and geodetic grids cannot be paired.
    if lst.ndim == 3 and lst.shape[0] == 1:
        lst = lst[0]

    lst, lon, lat = lst.ravel(), lon.ravel(), lat.ravel()
    if not (lst.size == lon.size == lat.size):
        raise ValueError(
            "SLSTR LST and geodetic grids have different shapes "
            f"({lst.size} vs {lon.size}/{lat.size} samples)"
        )

    minx, miny, maxx, maxy = grid.bbox_wgs84
    with np.errstate(invalid="ignore"):
        valid = (
            np.isfinite(lst)
            & np.isfinite(lon)
            & np.isfinite(lat)
            & (lon >= minx - _SENTINEL3_BBOX_MARGIN_DEG)
            & (lon <= maxx + _SENTINEL3_BBOX_MARGIN_DEG)
            & (lat >= miny - _SENTINEL3_BBOX_MARGIN_DEG)
            & (lat <= maxy + _SENTINEL3_BBOX_MARGIN_DEG)
        )
    if not valid.any():
        return np.full(grid.shape, np.nan, dtype=np.float32)

    if target_lonlat is None:
        target_lonlat = _target_lonlat_grid(grid)
    tgt_lon, tgt_lat = target_lonlat

    tree = cKDTree(np.column_stack([lon[valid], lat[valid]]))
    dist, idx = tree.query(
        np.column_stack([tgt_lon.ravel(), tgt_lat.ravel()]), k=1, workers=-1
    )

    result = lst[valid][idx]
    result = np.where(dist > SENTINEL3_MAX_MATCH_DEG, np.nan, result)
    return np.asarray(result.reshape(grid.shape), dtype=np.float32)


def _sentinel3_scene_to_grid(
    lst_nc_path: str,
    geo_nc_path: str,
    grid: _TargetGrid,
    target_lonlat: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Read a downloaded SLSTR pair, falling back between NetCDF engines."""
    _require_xarray()
    errors: list[str] = []
    for engine in _NETCDF_ENGINES:
        lst_ds = geo_ds = None
        try:
            lst_ds = _open_dataset(lst_nc_path, engine)
            geo_ds = _open_dataset(geo_nc_path, engine)
            return _sentinel3_ds_to_grid(lst_ds, geo_ds, grid, target_lonlat)
        except Exception as exc:  # noqa: BLE001 - try the next engine
            errors.append(f"{engine or 'default'}: {exc}")
        finally:
            for ds in (geo_ds, lst_ds):
                if ds is not None:
                    try:
                        ds.close()
                    except Exception:  # pragma: no cover - best effort
                        pass
    raise RuntimeError(
        "Could not read Sentinel-3 scene (tried engines: " + "; ".join(errors) + ")"
    )


def _search_sentinel3_items(
    grid: _TargetGrid, start_date: str, end_date: str, n_days: int, lang: str
) -> list:
    """Paginated STAC search for Sentinel-3 LST items covering the bbox."""
    if not HAS_STAC:
        raise ImportError(
            "source='sentinel3' requires pystac-client and planetary-computer "
            "(pip install pystac-client planetary-computer, or "
            "'pip install LCZ4py[lst]')."
        )

    def _do_search() -> list:
        catalog = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace
        )
        search = catalog.search(
            collections=[SENTINEL3_LST_COLLECTION],
            bbox=list(grid.bbox_wgs84),
            datetime=f"{start_date}/{end_date}",
        )
        # pystac-client follows the catalog's next-links; cap the total so
        # a continent-scale bbox over a long date range cannot page forever.
        limit = max(200, n_days * 20)
        collected: list = []
        for item in search.items():
            collected.append(item)
            if len(collected) >= limit:
                logger.warning(
                    "Sentinel-3 search hit the %d-item cap; later days may be "
                    "missing. Narrow the date range or AOI.",
                    limit,
                )
                break
        return collected

    try:
        items = _retry_call(_do_search, attempts=2)
    except ImportError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
        raise RuntimeError(
            f"Sentinel-3 STAC search failed on Planetary Computer: {exc}"
        ) from exc

    def _sort_key(item):
        dt = getattr(item, "datetime", None)
        return (dt is None, dt.timestamp() if dt else 0.0, str(item.id))

    items.sort(key=_sort_key)
    return items


def _ingest_sentinel3(
    grid: _TargetGrid,
    start_date: str,
    end_date: str,
    lang: str,
    verbose: bool,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, np.ndarray]:
    """Fetch/regrid one Sentinel-3 scene per day, in parallel, ordered by date."""
    if not HAS_STAC:
        raise ImportError(
            "source='sentinel3' requires pystac-client and planetary-computer "
            "(pip install pystac-client planetary-computer, or "
            "'pip install LCZ4py[lst]')."
        )
    _require_xarray()

    days = _daterange(start_date, end_date)
    items = _search_sentinel3_items(grid, start_date, end_date, len(days), lang)

    by_day: dict[str, list] = {}
    for item in items:
        dt = getattr(item, "datetime", None)
        if dt is None:
            continue
        by_day.setdefault(dt.date().isoformat(), []).append(item)

    # Expensive once: pixel-center lon/lat of the target grid (reused per day).
    target_lonlat = _target_lonlat_grid(grid) if by_day else None
    workers = min(_coerce_workers(max_workers), max(1, len(days)))

    def _fetch_day(day: date) -> tuple[str, Optional[np.ndarray], Optional[str]]:
        day_key = day.isoformat()
        candidates = by_day.get(day_key)
        if not candidates:
            return day_key, None, "no STAC item for this day"

        item = candidates[0]
        available = getattr(item, "assets", {}) or {}
        lst_key = next(
            (k for k in ("lst-in", "lst_in", "LST_in") if k in available), None
        )
        geo_key = next(
            (k for k in ("slstr-geodetic-in", "slstr_geodetic_in") if k in available),
            None,
        )
        if lst_key is None or geo_key is None:
            return (
                day_key,
                None,
                f"missing LST/geodetic assets (available: {', '.join(sorted(available))})",
            )

        lst_path = geo_path = None
        try:
            lst_path = _download_to_temp(available[lst_key].href, ".nc")
            geo_path = _download_to_temp(available[geo_key].href, ".nc")
            arr = _sentinel3_scene_to_grid(lst_path, geo_path, grid, target_lonlat)
        except Exception as exc:  # noqa: BLE001 - day is skipped, not fatal
            return day_key, None, str(exc)
        finally:
            _cleanup_temp(lst_path)
            _cleanup_temp(geo_path)

        if not np.isfinite(arr).any():
            return day_key, None, "scene has no valid pixels over this area"
        return day_key, arr, None

    out: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lcz-s3") as pool:
        for day_key, arr, reason in pool.map(_fetch_day, days):
            if arr is None:
                if verbose:
                    logger.warning(
                        "%s (%s)",
                        lcz_msg("lst_day_skipped", lang, date=day_key),
                        reason,
                    )
                continue
            out[day_key] = arr

    return dict(sorted(out.items()))


# ── Stack assembly / writing ──────────────────────────────────────────────────


def _write_lst_stack(
    path: str,
    stack: np.ndarray,
    dates: list[str],
    grid: _TargetGrid,
    units: str = "K",
) -> None:
    """Write an (n_dates, H, W) stack as a multi-band GeoTIFF, atomically."""
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3-D (n_dates, H, W), got shape {stack.shape}")
    if stack.shape[0] != len(dates):
        raise ValueError(
            f"stack has {stack.shape[0]} bands but {len(dates)} dates were given"
        )
    if tuple(stack.shape[1:]) != tuple(grid.shape):
        raise ValueError(
            f"stack grid {stack.shape[1:]} does not match the LCZ grid {grid.shape}"
        )

    path = os.path.abspath(str(path))
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=out_dir)
    os.close(fd)

    profile = {
        "driver": "GTiff",
        "height": grid.shape[0],
        "width": grid.shape[1],
        "count": stack.shape[0],
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": np.nan,
        "compress": "lzw",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(stack)
            dst.update_tags(units=units, variable="LST")
            for i, d in enumerate(dates, start=1):
                dst.set_band_description(i, str(d))
        os.replace(tmp_path, path)
    except Exception:
        _unlink(tmp_path)
        raise


def _kelvin_to_celsius(arr: np.ndarray) -> np.ndarray:
    return arr - 273.15


def _read_cached_stack(cache_path: str) -> tuple[np.ndarray, list[str]]:
    """Read a cached LST stack, raising on corrupt/empty files."""
    with rasterio.open(cache_path) as src:
        if src.count < 1:
            raise ValueError("cached LST stack has no bands")
        array = src.read()
        desc = list(src.descriptions)
    dates = [d or f"band_{i + 1}" for i, d in enumerate(desc)]
    if len(dates) != array.shape[0]:
        dates = [f"band_{i + 1}" for i in range(array.shape[0])]
    return array, dates


# ── Public API ────────────────────────────────────────────────────────────────


def lcz_get_lst(
    x: Union[str, Path],
    source: str,
    start_date: str,
    end_date: str,
    satellite: str = "goes16",
    target_hour: float = 13.0,
    units: str = "C",
    isave: bool = False,
    cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    lang: str = "en",
    verbose: bool = True,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> LCZLSTResult:
    """Download a daily LST time-series stack cropped to an LCZ map's footprint.

    Parameters
    ----------
    x : str or Path
        Path to an existing LCZ map GeoTIFF (e.g. from ``lcz_get_map``).
    source : {"goes", "sentinel3", "both"}
        "goes" (Americas only, ~2km CONUS grid), "sentinel3" (global, ~1km),
        or "both" (GOES where available, Sentinel-3 fallback per day).
    start_date, end_date : str
        ISO dates ("YYYY-MM-DD"), inclusive.
    satellite : str
        GOES satellite to use when source includes "goes": "goes16"/"goes17"/"goes18".
    target_hour : float
        Local-solar hour to target for the GOES daily pick (avoids mixing
        day/night pixels across the stack). Ignored for Sentinel-3.
    units : {"C", "K"}
        Output temperature units — Celsius (default) or Kelvin (the sources'
        native unit). The on-disk cache always stores Kelvin, so switching
        ``units`` between calls never triggers a re-download.
    isave : bool
        Also copy the stack to LCZ4r_output/lcz_lst_stack.tif.
    cache : bool
        Reuse a previously downloaded stack for the same (source, satellite,
        bbox, date range, target_hour).
    cache_dir : str
        Cache directory. Default ``~/.lcz4r_cache``.
    lang : str
        Message language ("en"/"pt"/"es"/"zh").
    verbose : bool
        Log per-day skip warnings.
    max_workers : int
        Maximum concurrent per-day scene downloads/regrids. Default
        ``min(8, os.cpu_count())``.

    Returns
    -------
    LCZLSTResult
        ``array`` is (n_days, H, W) float32, in the requested ``units``, with
        NaN outside the LCZ map's valid pixels or wherever no usable scene
        was found; ``dates[i]`` gives the ISO date for band ``i``.
        ``path`` (when ``cache=True``) always points at the Kelvin cache
        file, regardless of ``units`` — read ``result.array``/the ``isave``
        copy for data in the requested units.
    """
    source = str(source).strip().lower()
    if source not in ("goes", "sentinel3", "both"):
        raise ValueError("source must be 'goes', 'sentinel3', or 'both'")

    satellite = str(satellite).strip().lower()
    if satellite not in GOES_SATELLITE_CONTAINERS:
        raise ValueError(
            f"satellite must be one of {list(GOES_SATELLITE_CONTAINERS)}"
        )

    units = str(units).strip().upper()
    if units not in ("C", "K"):
        raise ValueError("units must be 'C' or 'K'")

    try:
        target_hour = float(target_hour)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_hour must be a number between 0 and 24") from exc
    if not (0.0 <= target_hour < 24.0):
        raise ValueError("target_hour must be between 0 and 24 (exclusive)")

    workers = _coerce_workers(max_workers)
    _daterange(start_date, end_date)  # validate dates before touching the network

    grid = _load_target_grid(x)

    cache_path = ""
    if cache:
        cache_root = os.path.expanduser(str(cache_dir))
        key = _cache_key(source, satellite, grid, start_date, end_date, target_hour)
        cache_path = os.path.join(cache_root, f"lst_stack_{key}.tif")

    array: Optional[np.ndarray] = None
    dates: list[str] = []

    if cache and cache_path and _is_valid_raster(cache_path):
        try:
            array, dates = _read_cached_stack(cache_path)
            if array.shape[1:] != grid.shape:
                raise ValueError(
                    f"cached stack is {array.shape[1:]} but the LCZ grid is "
                    f"{grid.shape}; re-downloading"
                )
            if verbose:
                logger.info("Using cached LST stack: %s", cache_path)
        except Exception as exc:  # noqa: BLE001 - corrupt cache is re-downloaded
            logger.warning("Ignoring unreadable LST cache %s: %s", cache_path, exc)
            array = None

    if array is None:
        scenes: dict[str, np.ndarray] = {}

        if source in ("goes", "both"):
            try:
                scenes.update(
                    _ingest_goes(
                        grid,
                        satellite,
                        start_date,
                        end_date,
                        target_hour,
                        lang,
                        verbose,
                        workers,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - "both" falls back per day
                if source == "goes":
                    raise
                if verbose:
                    logger.warning(
                        "GOES unavailable for this request (%s); "
                        "falling back to Sentinel-3.",
                        exc,
                    )

        if source in ("sentinel3", "both"):
            try:
                s3_scenes = _ingest_sentinel3(
                    grid, start_date, end_date, lang, verbose, workers
                )
            except Exception as exc:  # noqa: BLE001 - keep GOES data if we have it
                if not scenes:
                    raise
                if verbose:
                    logger.warning("Sentinel-3 unavailable (%s); using GOES only.", exc)
                s3_scenes = {}
            for day, arr in s3_scenes.items():
                scenes.setdefault(day, arr)  # GOES takes priority when both cover a day

        if not scenes:
            raise ValueError(
                lcz_msg(
                    "lst_no_scenes_found",
                    lang,
                    source=source,
                    start_date=start_date,
                    end_date=end_date,
                )
            )

        dates = sorted(scenes)
        stack = np.stack([scenes[d] for d in dates]).astype(np.float32, copy=False)
        array = np.where(grid.valid_mask[None, :, :], stack, np.nan).astype(np.float32)

        if cache and cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                _write_lst_stack(cache_path, array, dates, grid, units="K")
            except Exception as exc:  # noqa: BLE001 - caching is best-effort
                logger.warning("Could not write the LST cache %s: %s", cache_path, exc)

    # `array` (from cache or freshly assembled) is always Kelvin at this point.
    if units == "C":
        array = _kelvin_to_celsius(array)

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_lst_stack.tif")
        _write_lst_stack(out_path, array, dates, grid, units=units)
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    return LCZLSTResult(
        path=cache_path if cache and cache_path and os.path.exists(cache_path) else None,
        array=array,
        dates=dates,
        units=units,
    )


__all__ = ["lcz_get_lst", "LCZLSTResult"]
