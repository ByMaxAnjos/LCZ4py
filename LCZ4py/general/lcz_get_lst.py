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
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import httpx
import numpy as np
import pyproj
import rasterio
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr
from rasterio.features import shapes
from scipy.spatial import cKDTree
from shapely.geometry import shape

try:
    import pystac_client
    import planetary_computer
    HAS_STAC = True
except ImportError:
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

_SAS_TOKEN_CACHE: dict[tuple[str, str], tuple[str, datetime]] = {}


@dataclass
class LCZLSTResult:
    """Return type for lcz_get_lst."""
    path: Optional[str] = None
    array: Optional[np.ndarray] = None
    dates: Optional[list[str]] = None
    units: Optional[str] = None
    gdf: Optional[gpd.GeoDataFrame] = None
    geoarrow_table: Optional[object] = None


# ── LCZ map grid / geometry helpers ───────────────────────────────────────────

@dataclass
class _TargetGrid:
    shape: tuple[int, int]      # (H, W)
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    valid_mask: np.ndarray      # True where the LCZ map has data
    bbox_wgs84: tuple[float, float, float, float]


def _load_target_grid(x: Union[str, Path]) -> _TargetGrid:
    with rasterio.open(str(x)) as src:
        class_arr = src.read(1)
        transform = src.transform
        crs = src.crs or rasterio.crs.CRS.from_epsg(4326)
        shape_hw = (src.height, src.width)
        bounds = src.bounds

    valid_mask = (class_arr >= 1) & (class_arr <= 17)

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
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xs, ys = rasterio.transform.xy(grid.transform, rows.ravel(), cols.ravel())
    xs, ys = np.asarray(xs), np.asarray(ys)

    if grid.crs.to_epsg() != 4326:
        transformer = pyproj.Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        xs, ys = transformer.transform(xs, ys)

    return xs.reshape(h, w), ys.reshape(h, w)


def _rio_template(grid: _TargetGrid) -> xr.DataArray:
    """Empty DataArray carrying the target grid's crs/transform, for reproject_match."""
    template = xr.DataArray(
        np.zeros(grid.shape, dtype="float32"),
        dims=("y", "x"),
    )
    template = template.rio.write_crs(grid.crs)
    template = template.rio.write_transform(grid.transform)
    return template


def _daterange(start_date: str, end_date: str) -> list[date]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("end_date must not be before start_date")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _cache_key(source: str, satellite: str, grid: _TargetGrid,
                start_date: str, end_date: str, target_hour: float) -> str:
    bbox = ",".join(f"{v:.4f}" for v in grid.bbox_wgs84)
    raw = f"{source}|{satellite}|{bbox}|{start_date}|{end_date}|{target_hour}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── GOES ingestion ────────────────────────────────────────────────────────────

def _get_sas_token(account: str, container: str) -> str:
    key = (account, container)
    cached = _SAS_TOKEN_CACHE.get(key)
    if cached and cached[1] > datetime.utcnow():
        return cached[0]

    resp = httpx.get(
        f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}",
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data["token"]
    expiry = datetime.fromisoformat(data["msft:expiry"].replace("Z", "+00:00")).replace(tzinfo=None)
    # refresh a little early
    _SAS_TOKEN_CACHE[key] = (token, expiry - timedelta(minutes=2))
    return token


def _list_blobs(account: str, container: str, prefix: str, token: str) -> list[str]:
    names: list[str] = []
    marker = ""
    base = f"https://{account}.blob.core.windows.net/{container}"
    while True:
        params = f"restype=container&comp=list&prefix={prefix}&{token}"
        if marker:
            params += f"&marker={marker}"
        resp = httpx.get(f"{base}?{params}", timeout=30)
        resp.raise_for_status()
        xml = resp.text
        import re
        names.extend(re.findall(r"<Name>([^<]+)</Name>", xml))
        m = re.search(r"<NextMarker>([^<]*)</NextMarker>", xml)
        marker = m.group(1) if m else ""
        if not marker:
            break
    return names


def _goes_in_view(grid: _TargetGrid) -> bool:
    x0, y0, x1, y1 = grid.bbox_wgs84
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0, bx1, by1 = GOES_CONUS_BBOX
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def _pick_goes_hour_file(
    container: str, sat_code: str, day: date, target_hour: float,
) -> Optional[str]:
    token = _get_sas_token(GOES_ACCOUNT, container)
    doy = day.timetuple().tm_yday
    base_hour = int(round(target_hour))

    for offset in [0, -1, 1, -2, 2, -3, 3]:
        hour = base_hour + offset
        if not (0 <= hour <= 23):
            continue
        prefix = f"ABI-L2-LSTC/{day.year}/{doy:03d}/{hour:02d}/"
        files = _list_blobs(GOES_ACCOUNT, container, prefix, token)
        files = [f for f in files if sat_code in f]
        if files:
            return files[0]
    return None


def _download_to_temp(url: str, suffix: str) -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="lcz_lst_"), f"scene{suffix}")
    with httpx.stream("GET", url, timeout=120) as resp:
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    return path


def _goes_scene_to_grid(nc_path: str, grid: _TargetGrid) -> np.ndarray:
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    proj = ds["goes_imager_projection"].attrs
    height = float(proj["perspective_point_height"])

    lst = ds["LST"].where(ds["DQF"] == 0)  # keep only good-quality retrievals
    da = lst.assign_coords(x=ds["x"].values * height, y=ds["y"].values * height)

    geos_crs = pyproj.CRS.from_dict({
        "proj": "geos",
        "h": height,
        "a": float(proj["semi_major_axis"]),
        "b": float(proj["semi_minor_axis"]),
        "lon_0": float(proj["longitude_of_projection_origin"]),
        "sweep": proj["sweep_angle_axis"],
        "units": "m",
    })
    da = da.rio.write_crs(geos_crs)
    da.rio.write_transform(inplace=True)

    template = _rio_template(grid)
    regridded = da.rio.reproject_match(template, resampling=rasterio.enums.Resampling.bilinear)
    return regridded.values.astype(np.float32)


def _ingest_goes(
    grid: _TargetGrid, satellite: str, start_date: str, end_date: str,
    target_hour: float, lang: str, verbose: bool,
) -> dict[str, np.ndarray]:
    if not _goes_in_view(grid):
        raise ValueError(lcz_msg("lst_source_out_of_bounds", lang))

    container, sat_code = GOES_SATELLITE_CONTAINERS[satellite]
    out: dict[str, np.ndarray] = {}

    for day in _daterange(start_date, end_date):
        blob_name = _pick_goes_hour_file(container, sat_code, day, target_hour)
        if blob_name is None:
            if verbose:
                logger.warning(lcz_msg("lst_day_skipped", lang, date=day.isoformat()))
            continue
        try:
            token = _get_sas_token(GOES_ACCOUNT, container)
            url = f"https://{GOES_ACCOUNT}.blob.core.windows.net/{container}/{blob_name}?{token}"
            nc_path = _download_to_temp(url, ".nc")
            arr = _goes_scene_to_grid(nc_path, grid)
        except Exception as e:
            logger.warning(f"GOES scene failed for {day}: {e}")
            continue
        if np.isnan(arr).all():
            if verbose:
                logger.warning(lcz_msg("lst_day_skipped", lang, date=day.isoformat()))
            continue
        out[day.isoformat()] = arr

    return out


# ── Sentinel-3 ingestion ──────────────────────────────────────────────────────

def _sentinel3_scene_to_grid(
    lst_nc_path: str, geo_nc_path: str, grid: _TargetGrid,
) -> np.ndarray:
    lst_ds = xr.open_dataset(lst_nc_path, engine="h5netcdf")
    geo_ds = xr.open_dataset(geo_nc_path, engine="h5netcdf")

    lst = lst_ds["LST"].values.ravel()
    lon = geo_ds["longitude_in"].values.ravel()
    lat = geo_ds["latitude_in"].values.ravel()

    valid = ~np.isnan(lst)
    if not valid.any():
        return np.full(grid.shape, np.nan, dtype=np.float32)

    tree = cKDTree(np.column_stack([lon[valid], lat[valid]]))
    tgt_lon, tgt_lat = _target_lonlat_grid(grid)
    dist, idx = tree.query(
        np.column_stack([tgt_lon.ravel(), tgt_lat.ravel()]), k=1
    )

    # ~1km SLSTR footprint: reject matches further than ~0.02 deg (~2km) away
    result = lst[valid][idx]
    result = np.where(dist > 0.02, np.nan, result)
    return result.reshape(grid.shape).astype(np.float32)


def _ingest_sentinel3(
    grid: _TargetGrid, start_date: str, end_date: str, lang: str, verbose: bool,
) -> dict[str, np.ndarray]:
    if not HAS_STAC:
        raise ImportError(
            "source='sentinel3' requires pystac-client and planetary-computer "
            "(pip install pystac-client planetary-computer)"
        )

    catalog = pystac_client.Client.open(
        PC_STAC_URL, modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[SENTINEL3_LST_COLLECTION],
        bbox=list(grid.bbox_wgs84),
        datetime=f"{start_date}/{end_date}",
    )
    items = list(search.items())

    by_day: dict[str, list] = {}
    for item in items:
        day_key = item.datetime.date().isoformat()
        by_day.setdefault(day_key, []).append(item)

    out: dict[str, np.ndarray] = {}
    for day in _daterange(start_date, end_date):
        day_key = day.isoformat()
        candidates = by_day.get(day_key)
        if not candidates:
            if verbose:
                logger.warning(lcz_msg("lst_day_skipped", lang, date=day_key))
            continue
        item = candidates[0]
        try:
            lst_path = _download_to_temp(item.assets["lst-in"].href, ".nc")
            geo_path = _download_to_temp(item.assets["slstr-geodetic-in"].href, ".nc")
            arr = _sentinel3_scene_to_grid(lst_path, geo_path, grid)
        except Exception as e:
            logger.warning(f"Sentinel-3 scene failed for {day_key}: {e}")
            continue
        if np.isnan(arr).all():
            if verbose:
                logger.warning(lcz_msg("lst_day_skipped", lang, date=day_key))
            continue
        out[day_key] = arr

    return out


# ── Stack assembly / writing ──────────────────────────────────────────────────

def _write_lst_stack(
    path: str, stack: np.ndarray, dates: list[str], grid: _TargetGrid, units: str = "K",
) -> None:
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
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)
        dst.update_tags(units=units, variable="LST")
        for i, d in enumerate(dates, start=1):
            dst.set_band_description(i, d)


def _kelvin_to_celsius(arr: np.ndarray) -> np.ndarray:
    return arr - 273.15


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
    if source not in ("goes", "sentinel3", "both"):
        raise ValueError("source must be 'goes', 'sentinel3', or 'both'")
    if satellite not in GOES_SATELLITE_CONTAINERS:
        raise ValueError(f"satellite must be one of {list(GOES_SATELLITE_CONTAINERS)}")
    if units not in ("C", "K"):
        raise ValueError("units must be 'C' or 'K'")

    grid = _load_target_grid(x)

    cache_dir = os.path.expanduser(cache_dir)
    key = _cache_key(source, satellite, grid, start_date, end_date, target_hour)
    cache_path = os.path.join(cache_dir, f"lst_stack_{key}.tif")

    if cache and os.path.exists(cache_path):
        with rasterio.open(cache_path) as src:
            array = src.read()
            dates = [src.descriptions[i] for i in range(src.count)]
        if verbose:
            logger.info(f"Using cached LST stack: {cache_path}")
    else:
        scenes: dict[str, np.ndarray] = {}
        if source in ("goes", "both"):
            try:
                scenes.update(_ingest_goes(
                    grid, satellite, start_date, end_date, target_hour, lang, verbose,
                ))
            except ValueError:
                if source == "goes":
                    raise
                if verbose:
                    logger.warning("GOES out of view for this area; using Sentinel-3 only.")

        if source == "sentinel3" or (source == "both"):
            s3_scenes = _ingest_sentinel3(grid, start_date, end_date, lang, verbose)
            for day, arr in s3_scenes.items():
                scenes.setdefault(day, arr)  # GOES takes priority when both cover a day

        if not scenes:
            raise ValueError(lcz_msg(
                "lst_no_scenes_found", lang,
                source=source, start_date=start_date, end_date=end_date,
            ))

        dates = sorted(scenes)
        array = np.stack([
            np.where(grid.valid_mask, scenes[d], np.nan) for d in dates
        ]).astype(np.float32)

        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            _write_lst_stack(cache_path, array, dates, grid, units="K")  # cache is always Kelvin

    # `array` (from cache or freshly assembled) is always Kelvin at this point.
    if units == "C":
        array = _kelvin_to_celsius(array)

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_lst_stack.tif")
        _write_lst_stack(out_path, array, dates, grid, units=units)
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    return LCZLSTResult(path=cache_path if cache else None, array=array, dates=dates, units=units)


__all__ = ["lcz_get_lst", "LCZLSTResult"]
