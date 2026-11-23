"""
lcz_get_meta_tree.py

Meta/WRI global Canopy Height Map v2 (CHMv2) downloader.

Source: https://registry.opendata.aws/dataforgood-fb-forestsv2/ (CC BY 4.0).
Public S3 bucket, anonymous HTTPS access — no AWS credentials needed. Tiles
are ~1.2 m Cloud-Optimized GeoTIFFs (EPSG:3857, uint8 = canopy height in
metres) named by their zoom-10 Bing quadkey, so only the tiles touching the
study area are streamed via /vsicurl/ (with overviews when ``res`` > 1.2 m).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import tempfile
from typing import List, Optional

import geopandas as gpd
import httpx
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge

from LCZ4py._internal._lcz_map_engine import (
    HTTP_TIMEOUT,
    _async_geocode,
    _atomic_copy_raster,
    _load_geoarrow_cache,
    _safe_remove,
    _save_geoarrow_cache,
    run_coroutine,
)

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "~/.lcz4r_cache"
CHM_URL = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3/chm"
CHM_ZOOM = 10
NODATA = 255  # outside study area / no tile (heights are 0-~100 m)


def _quadkeys(west: float, south: float, east: float, north: float, z: int = CHM_ZOOM) -> List[str]:
    """Bing quadkeys of the Web-Mercator tiles covering a lon/lat bbox."""
    n = 2 ** z

    def tile(lon: float, lat: float):
        lat = max(min(lat, 85.0511), -85.0511)
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        return min(max(x, 0), n - 1), min(max(y, 0), n - 1)

    x0, y0 = tile(west, north)
    x1, y1 = tile(east, south)
    keys = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            keys.append("".join(
                str(((x >> i) & 1) | (((y >> i) & 1) << 1)) for i in range(z - 1, -1, -1)
            ))
    return keys


async def _resolve_study_area(city: Optional[str], roi, cache_dir: str, lang: str, verbose: bool):
    if city is None:
        return roi
    slug = city.lower().strip().replace(" ", "_").replace("/", "_")
    cache_file = os.path.join(cache_dir, f"study_area_{slug}.arrow")
    study_area = _load_geoarrow_cache(cache_file)
    if study_area is None:
        if verbose:
            logger.info("Geocoding '%s'...", city)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            study_area = await _async_geocode(city, client, lang=lang)
        if study_area is None:
            from LCZ4py._internal.i18n_messages import lcz_msg

            raise ValueError(lcz_msg("city_not_found", lang, city=city))
        _save_geoarrow_cache(study_area, cache_file)
    return study_area


def _build_chm(study_area: gpd.GeoDataFrame, res: Optional[float], verbose: bool) -> str:
    """Mosaic the intersecting CHM tiles, mask to the study area, write a temp GeoTIFF."""
    sa_4326 = study_area.set_crs(4326) if study_area.crs is None else study_area.to_crs(4326)
    keys = _quadkeys(*sa_4326.total_bounds)
    if verbose:
        logger.info("Streaming %d CHMv2 tile(s)...", len(keys))

    sources = []
    for k in keys:
        try:
            sources.append(rasterio.open(f"/vsicurl/{CHM_URL}/{k}.tif"))
        except rasterio.errors.RasterioIOError:
            pass  # ponytail: no tile = no coverage (ocean/ice); skipped silently
    if not sources:
        raise ValueError("No Meta CHMv2 tiles cover this study area.")

    geom = sa_4326.to_crs(3857).geometry.union_all()
    try:
        arr, transform = merge(sources, bounds=geom.bounds, res=res, nodata=NODATA)
    finally:
        for s in sources:
            s.close()

    arr = arr[0]
    inside = rasterize([(geom, 1)], out_shape=arr.shape, transform=transform, fill=0, dtype=np.uint8)
    arr[inside == 0] = NODATA

    out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    with rasterio.open(
        out_path, "w", driver="GTiff", dtype="uint8", count=1, nodata=NODATA,
        width=arr.shape[1], height=arr.shape[0], crs="EPSG:3857", transform=transform,
        compress="deflate", tiled=True,
    ) as dst:
        dst.write(arr, 1)
    return out_path


def lcz_get_meta_tree(
    city: Optional[str] = None,
    roi: Optional[gpd.GeoDataFrame] = None,
    res: Optional[float] = 10.0,
    isave_map: bool = False,
    cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    lang: str = "en",
    verbose: bool = True,
) -> str:
    """Download the Meta/WRI Canopy Height Map v2 clipped to a city or ROI.

    Parameters
    ----------
    city : str, optional
        Place name resolved through OpenStreetMap/Nominatim.
    roi : GeoDataFrame, optional
        Custom study-area polygon(s). Provide either ``city`` or ``roi``.
    res : float or None
        Output pixel size in metres (EPSG:3857). Default 10 m. ``None`` keeps
        the native ~1.2 m — large: ~700k pixels per km².
    isave_map : bool
        Also copy the GeoTIFF into ``LCZ4r_output/meta_tree_height.tif``.
    cache : bool
        Cache the geocoded boundary and the clipped raster under ``cache_dir``.
    cache_dir : str
        Cache directory (default ``~/.lcz4r_cache``).
    lang : {"en", "pt", "es", "zh"}
        Message language.
    verbose : bool
        Log progress messages.

    Returns
    -------
    str
        Path to a one-band uint8 GeoTIFF (EPSG:3857): canopy height in metres,
        255 = nodata.

    Notes
    -----
    Data: Meta & WRI (2026), CHMv2, https://registry.opendata.aws/dataforgood-fb-forestsv2/
    (CC BY 4.0). Anonymous access, no credentials.
    """
    from LCZ4py.general.lcz_get_map import _validate_inputs

    _validate_inputs(city, roi, lang)
    cache_dir = os.path.expanduser(cache_dir) if cache else tempfile.mkdtemp()
    os.makedirs(cache_dir, exist_ok=True)

    study_area = run_coroutine(_resolve_study_area(city, roi, cache_dir, lang, verbose))
    sa_4326 = study_area.set_crs(4326) if study_area.crs is None else study_area.to_crs(4326)
    key = f"{CHM_URL}|{res}".encode() + sa_4326.geometry.union_all().wkb
    # "clipped_" prefix so lcz_clear_cache() removes it too.
    final_path = os.path.join(cache_dir, f"clipped_chm_{hashlib.sha256(key).hexdigest()[:16]}.tif")

    if os.path.exists(final_path):
        if verbose:
            logger.info("Loading canopy height map from local cache.")
    else:
        tmp = _build_chm(sa_4326, res, verbose)
        try:
            _atomic_copy_raster(tmp, final_path)
        finally:
            _safe_remove(tmp)

    if isave_map:
        os.makedirs("LCZ4r_output", exist_ok=True)
        out_path = os.path.join("LCZ4r_output", "meta_tree_height.tif")
        _atomic_copy_raster(final_path, out_path)
        return out_path
    return final_path


__all__ = ["lcz_get_meta_tree"]
