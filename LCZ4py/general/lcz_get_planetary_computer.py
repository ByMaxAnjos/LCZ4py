"""
lcz_get_planetary_computer.py

Generic Microsoft Planetary Computer downloader, cropped to an existing LCZ
map's footprint — same pattern as lcz_get_lst.py, generalized to any
keyless/anonymous, globally-covering STAC collection (Sentinel-2, Landsat,
ESA WorldCover, biodiversity intactness, elevation, surface water, ...).

All Planetary Computer STAC collections are readable without an API key
(assets are signed on the fly via ``planetary_computer.sign_inplace``, the
same mechanism ``lcz_get_lst`` uses for Sentinel-3). ``PC_COLLECTIONS`` is a
small, curated shortcut registry for common global/free collections; any
other Planetary Computer collection id also works if you pass ``assets``
explicitly (use :func:`lcz_list_pc_assets` to discover asset keys first).

Output is a multi-band GeoTIFF (one band per requested asset), float32 with
NaN nodata, pixel-aligned to the input LCZ map's grid — consistent with the
``LCZLSTResult``/``LCZStackResult`` conventions elsewhere in this codebase.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import rioxarray  # noqa: F401 - registers the .rio accessor

try:
    import pystac_client
    import planetary_computer
    HAS_STAC = True
except ImportError:
    HAS_STAC = False

from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py.general.lcz_get_lst import _load_target_grid, _rio_template
from LCZ4py._internal._lcz_map_engine import _atomic_copy_raster, _is_valid_raster

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"
DEFAULT_CACHE_DIR = "~/.lcz4r_cache"

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ── Curated shortcuts for common global, keyless-access collections ──────────
# Not exhaustive (Planetary Computer hosts 100+ collections) — pass the raw
# collection id + `assets=` for anything not listed here, and use
# `lcz_list_pc_assets` to discover its asset keys first.
PC_COLLECTIONS: dict[str, dict] = {
    "sentinel-2-l2a": {
        "collection_id": "sentinel-2-l2a",
        "assets": ["B04", "B03", "B02", "B08"],
        "time_varying": True,
        "cloud_cover_property": "eo:cloud_cover",
        "resampling": "bilinear",
        "description": "Sentinel-2 L2A surface reflectance, 10-60m, global, ~5-day revisit.",
    },
    "landsat": {
        "collection_id": "landsat-c2-l2",
        "assets": ["red", "green", "blue", "nir08"],
        "time_varying": True,
        "cloud_cover_property": "eo:cloud_cover",
        "resampling": "bilinear",
        "description": "Landsat Collection 2 Level-2 surface reflectance, 30m, global, ~16-day revisit.",
    },
    "worldcover": {
        "collection_id": "esa-worldcover",
        "assets": ["map"],
        "time_varying": False,
        "cloud_cover_property": None,
        "resampling": "nearest",  # categorical land-cover codes
        "description": "ESA WorldCover 10m global land cover (2020/2021), static.",
    },
    "biodiversity": {
        "collection_id": "io-biodiversity",
        "assets": ["data"],
        "time_varying": False,
        "cloud_cover_property": None,
        "resampling": "bilinear",
        "description": "Global Biodiversity Intactness Index (Vizzuality/UNEP-WCMC, ~300m), static.",
    },
    "landcover-cci": {
        "collection_id": "esa-cci-lc",
        "assets": ["lccs_class"],
        "time_varying": True,
        "cloud_cover_property": None,
        "resampling": "nearest",  # categorical land-cover codes
        "description": "ESA CCI global land cover, annual, 300m (1992-2020).",
    },
    "elevation": {
        "collection_id": "nasadem",
        "assets": ["elevation"],
        "time_varying": False,
        "cloud_cover_property": None,
        "resampling": "bilinear",
        "description": "NASADEM global elevation, 30m, static.",
    },
    "surface-water": {
        "collection_id": "jrc-gsw",
        "assets": ["occurrence"],
        "time_varying": False,
        "cloud_cover_property": None,
        "resampling": "bilinear",
        "description": "JRC Global Surface Water occurrence, 30m, static.",
    },
}


@dataclass
class LCZPCResult:
    """Return type for lcz_get_planetary_computer."""
    path: Optional[str] = None
    array: Optional[np.ndarray] = None
    bands: Optional[list[str]] = None
    collection: Optional[str] = None
    item_ids: Optional[list[str]] = None
    gdf: Optional[gpd.GeoDataFrame] = None
    geoarrow_table: Optional[object] = None


# ── Collection resolution ─────────────────────────────────────────────────────

def _resolve_collection(
    collection: str, assets: Optional[list[str]], lang: str,
) -> tuple[str, list[str], Optional[str], bool, str]:
    """Return (collection_id, assets, cloud_cover_property, time_varying, resampling)."""
    cfg = PC_COLLECTIONS.get(collection)
    if cfg is not None:
        return (
            cfg["collection_id"],
            list(assets) if assets else list(cfg["assets"]),
            cfg["cloud_cover_property"],
            cfg["time_varying"],
            cfg["resampling"],
        )

    # Not a known shortcut — treat literally as a Planetary Computer collection
    # id. We can't guess its asset keys or whether it's time-varying, so
    # require the caller to be explicit (discoverable via lcz_list_pc_assets).
    if not assets:
        raise ValueError(lcz_msg(
            "pc_unknown_collection", lang,
            collection=collection, known=", ".join(sorted(PC_COLLECTIONS)),
        ))
    return collection, list(assets), None, True, "bilinear"


def lcz_list_pc_assets(collection: str, x: Union[str, Path]) -> dict[str, str]:
    """List available asset keys (and titles) for a Planetary Computer
    collection over an LCZ map's area.

    Handy for picking ``assets=`` when ``collection`` isn't one of the
    built-in shortcuts in :data:`PC_COLLECTIONS`.

    Parameters
    ----------
    collection : str
        A key in :data:`PC_COLLECTIONS`, or a raw Planetary Computer
        collection id.
    x : str or Path
        Path to an existing LCZ map GeoTIFF, used only for its bounding box.

    Returns
    -------
    dict
        ``{asset_key: title}`` from the first matching item.
    """
    if not HAS_STAC:
        raise ImportError(
            "lcz_list_pc_assets requires pystac-client and planetary-computer "
            "(pip install pystac-client planetary-computer)"
        )
    cfg = PC_COLLECTIONS.get(collection)
    collection_id = cfg["collection_id"] if cfg else collection

    grid = _load_target_grid(x)
    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=[collection_id], bbox=list(grid.bbox_wgs84), max_items=1)
    items = list(search.items())
    if not items:
        raise ValueError(f"No items found for collection '{collection_id}' over this area.")
    return {key: (asset.title or "") for key, asset in items[0].assets.items()}


# ── Item search / selection ───────────────────────────────────────────────────

def _search_items(
    collection_id: str, grid, start_date: Optional[str], end_date: Optional[str],
    time_varying: bool, cloud_cover_property: Optional[str], max_cloud_cover: Optional[float],
    max_items: int, lang: str,
) -> list:
    if not HAS_STAC:
        raise ImportError(
            "lcz_get_planetary_computer requires pystac-client and planetary-computer "
            "(pip install pystac-client planetary-computer)"
        )

    if time_varying and bool(start_date) != bool(end_date):
        raise ValueError("Provide both start_date and end_date, or neither.")

    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)

    search_kwargs: dict = {
        "collections": [collection_id],
        "bbox": list(grid.bbox_wgs84),
        # Candidate pool bigger than max_items so best-by-cloud-cover sorting
        # below has something to choose from; still server-side bounded.
        "max_items": max(50, max_items * 5),
    }
    if time_varying and start_date and end_date:
        search_kwargs["datetime"] = f"{start_date}/{end_date}"
    if cloud_cover_property and max_cloud_cover is not None:
        search_kwargs["query"] = {cloud_cover_property: {"lt": max_cloud_cover}}

    items = list(catalog.search(**search_kwargs).items())
    if not items:
        raise ValueError(lcz_msg(
            "pc_no_items_found", lang, collection=collection_id,
            date_range=f" between {start_date} and {end_date}" if start_date else "",
        ))

    def sort_key(item):
        cloud = item.properties.get(cloud_cover_property, 0.0) if cloud_cover_property else 0.0
        # Fallback for items without a datetime (some static collections):
        # sort_key still needs a comparable value, so treat as "oldest".
        ts = item.datetime.timestamp() if item.datetime else 0.0
        return (cloud, -ts)

    items.sort(key=sort_key)
    return items[:max_items]


def _clip_to_grid_bbox(da, grid):
    """Crop a just-opened asset to the target grid's bbox, in the asset's own CRS.

    Must run before reproject_match — see the caller for why.
    """
    minx, miny, maxx, maxy = grid.bbox_wgs84
    src_crs = da.rio.crs
    if src_crs is not None and src_crs.to_epsg() != 4326:
        transformer = pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
        xs, ys = transformer.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
    return da.rio.clip_box(minx, miny, maxx, maxy, auto_expand=True)


def _pick_overview_level(href: str, grid) -> Optional[int]:
    """Pick the finest COG overview whose resolution is still >= the target
    grid's, so reproject_match warps thousands of pixels instead of millions.

    clip_box alone only bounds the *extent* read, not the *resolution* — a
    Sentinel-2 band is a native 10m/px COG, and reproject_match onto a
    ~100m/px LCZ grid was measured taking 50s/band at full resolution vs.
    ~4s picking the matching overview. Returns None (native resolution) on
    any lookup failure or when the target is already at/above native res.
    """
    try:
        with rasterio.open(href) as src:
            overviews = src.overviews(1)
            if not overviews:
                return None
            native_res_x = src.res[0]
            src_crs = src.crs
    except Exception:
        return None

    tgt_res = abs(grid.transform.a)
    if grid.crs.to_epsg() == 4326:
        lat = (grid.bbox_wgs84[1] + grid.bbox_wgs84[3]) / 2
        tgt_res_m = tgt_res * 111_320 * max(0.1, math.cos(math.radians(lat)))
    else:
        tgt_res_m = tgt_res
    native_res_m = native_res_x * 111_320 if (src_crs and src_crs.to_epsg() == 4326) else native_res_x

    decimation_needed = tgt_res_m / native_res_m
    if decimation_needed <= 1:
        return None

    level = None
    for i, dec in enumerate(overviews):
        if dec <= decimation_needed:
            level = i
        else:
            break
    return level


def _fetch_asset_mosaic(
    items: list, asset: str, grid, template, resampling_enum,
    collection_id: str, lang: str, verbose: bool,
) -> tuple[np.ndarray, list[str]]:
    """Merge one asset across candidate items onto the target grid.

    Items are already sorted best-first (lowest cloud cover / most recent);
    each contributes only where the mosaic still has gaps (first-valid-wins),
    stopping early once the grid is fully covered.
    """
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    used_ids: list[str] = []

    for item in items:
        if np.isfinite(out).all():
            break
        if asset not in item.assets:
            continue

        href = item.assets[asset].href  # signed already (catalog modifier=sign_inplace)
        try:
            overview_level = _pick_overview_level(href, grid)
            with rioxarray.open_rasterio(href, masked=True, overview_level=overview_level) as src:
                da = src
                if "band" in da.dims:
                    da = da.isel(band=0, drop=True)
                # Planetary Computer assets are often whole-tile/global COGs
                # (e.g. a 36000x36000px ESA WorldCover tile, or a 10980x10980
                # 10m-native Sentinel-2 band). clip_box bounds the *extent*
                # read; overview_level (picked above) bounds the *resolution*
                # read — without it, reproject_match onto a coarser LCZ grid
                # still warps from full native resolution over the network
                # (measured: 50s/band vs. ~4s with the matching overview).
                da = _clip_to_grid_bbox(da, grid)
                regridded = da.rio.reproject_match(template, resampling=resampling_enum)
                vals = regridded.values.astype(np.float32)
        except Exception as e:
            logger.warning(f"Planetary Computer asset '{asset}' failed for item {item.id}: {e}")
            continue

        fill_mask = np.isnan(out) & ~np.isnan(vals)
        if not fill_mask.any():
            continue
        out[fill_mask] = vals[fill_mask]
        used_ids.append(item.id)

    if not used_ids:
        available = ", ".join(sorted(items[0].assets)) if items else ""
        raise ValueError(lcz_msg(
            "pc_asset_missing", lang, asset=asset, collection=collection_id, available=available,
        ))
    if verbose and np.isnan(out).any():
        logger.info(lcz_msg("pc_asset_partial", lang, asset=asset, n_items=len(used_ids)))

    return out, used_ids


# ── Stack assembly / writing ──────────────────────────────────────────────────

def _write_pc_stack(
    path: str, stack: np.ndarray, band_names: list[str], grid,
    collection_id: str, item_ids: list[str],
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
        dst.update_tags(collection=collection_id, items=",".join(item_ids))
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def _cache_key(
    collection_id: str, assets: list[str], grid,
    start_date: Optional[str], end_date: Optional[str],
    max_cloud_cover: Optional[float], max_items: int,
) -> str:
    bbox = ",".join(f"{v:.4f}" for v in grid.bbox_wgs84)
    raw = (
        f"{collection_id}|{','.join(assets)}|{bbox}|{start_date}|{end_date}"
        f"|{max_cloud_cover}|{max_items}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Public API ────────────────────────────────────────────────────────────────

def lcz_get_planetary_computer(
    x: Union[str, Path],
    collection: str,
    assets: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_cloud_cover: Optional[float] = 30.0,
    cloud_cover_property: Optional[str] = None,
    max_items: int = 10,
    resampling: Optional[str] = None,
    isave: bool = False,
    cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    lang: str = "en",
    verbose: bool = True,
) -> LCZPCResult:
    """Download Microsoft Planetary Computer products cropped to an LCZ map's footprint.

    Works with any keyless/anonymous, global Planetary Computer STAC
    collection: pass a shortcut key from ``PC_COLLECTIONS`` (e.g.
    ``"sentinel-2-l2a"``, ``"biodiversity"``, ``"worldcover"``), or any raw
    Planetary Computer collection id plus ``assets=`` (see
    :func:`lcz_list_pc_assets` to discover asset keys).

    Parameters
    ----------
    x : str or Path
        Path to an existing LCZ map GeoTIFF (e.g. from ``lcz_get_map``).
    collection : str
        A key in ``PC_COLLECTIONS``, or a raw Planetary Computer collection id.
    assets : list of str, optional
        Asset/band keys to extract, one output band per asset. Defaults to
        the shortcut's curated list; required if ``collection`` is not a
        known shortcut.
    start_date, end_date : str, optional
        ISO dates ("YYYY-MM-DD"), inclusive. Required together for
        time-varying collections (ignored for static ones like WorldCover or
        NASADEM); if omitted, the most recent matching imagery is used.
    max_cloud_cover : float, optional
        Maximum ``eo:cloud_cover`` percentage for optical collections that
        expose it (Sentinel-2, Landsat). Ignored for collections without a
        cloud-cover property. Default 30.
    cloud_cover_property : str, optional
        Override the STAC property name used for cloud filtering (rarely
        needed — shortcuts already set this).
    max_items : int
        Maximum number of STAC items merged per asset (best-cloud/most-recent
        first, first-valid-wins mosaic) to fill gaps across tile boundaries.
        Default 10.
    resampling : str, optional
        Name of a ``rasterio.enums.Resampling`` member (e.g. "bilinear",
        "nearest", "cubic"). Defaults to the shortcut's recommendation
        ("nearest" for categorical land-cover collections, "bilinear"
        otherwise); unknown collections default to "bilinear".
    isave : bool
        Also copy the stack to ``LCZ4r_output/lcz_pc_<collection>_stack.tif``.
    cache : bool
        Reuse a previously downloaded stack for the same (collection, assets,
        bbox, date range, cloud filter, max_items).
    cache_dir : str
        Cache directory. Default ``~/.lcz4r_cache``.
    lang : str
        Message language ("en"/"pt"/"es"/"zh").
    verbose : bool
        Log progress and partial-coverage warnings.

    Returns
    -------
    LCZPCResult
        ``array`` is (n_assets, H, W) float32, NaN outside the LCZ map's
        valid pixels or wherever no scene covered a pixel; ``bands[i]``
        names the asset for band ``i``; ``item_ids`` lists the STAC items
        that contributed to the mosaic.

    Examples
    --------
    >>> result = lcz_get_planetary_computer(
    ...     "LCZ4r_output/lcz_map.tif", collection="sentinel-2-l2a",
    ...     start_date="2024-06-01", end_date="2024-08-31", max_cloud_cover=20,
    ... )
    >>> result.bands
    ['B04', 'B03', 'B02', 'B08']

    >>> # Static, global, no dates needed:
    >>> bio = lcz_get_planetary_computer("LCZ4r_output/lcz_map.tif", collection="biodiversity")
    """
    if max_items < 1:
        raise ValueError("max_items must be >= 1")

    grid = _load_target_grid(x)
    collection_id, resolved_assets, cfg_cloud_prop, time_varying, default_resampling = (
        _resolve_collection(collection, assets, lang)
    )
    effective_cloud_prop = cloud_cover_property or cfg_cloud_prop

    resampling = resampling or default_resampling
    resampling_enum = getattr(rasterio.enums.Resampling, resampling, None)
    if resampling_enum is None:
        raise ValueError(f"Unknown resampling method: {resampling!r}")

    cache_dir = os.path.expanduser(cache_dir)
    key = _cache_key(collection_id, resolved_assets, grid, start_date, end_date,
                      max_cloud_cover, max_items)
    safe_id = collection_id.replace("/", "_")
    cache_path = os.path.join(cache_dir, f"pc_{safe_id}_{key}.tif")

    if cache and _is_valid_raster(cache_path):
        with rasterio.open(cache_path) as src:
            array = src.read()
            bands = [src.descriptions[i] or resolved_assets[i] for i in range(src.count)]
            tags = src.tags()
        item_ids = tags.get("items", "").split(",") if tags.get("items") else []
        if verbose:
            logger.info(f"Using cached Planetary Computer stack: {cache_path}")
    else:
        if verbose:
            logger.info(f"Searching '{collection_id}' on Planetary Computer...")
        items = _search_items(
            collection_id, grid, start_date, end_date, time_varying,
            effective_cloud_prop, max_cloud_cover, max_items, lang,
        )
        if verbose:
            logger.info(f"Found {len(items)} candidate item(s); fetching {len(resolved_assets)} asset(s).")

        template = _rio_template(grid)
        band_arrays: list[np.ndarray] = []
        item_ids_used: set[str] = set()
        for asset in resolved_assets:
            arr, used_ids = _fetch_asset_mosaic(
                items, asset, grid, template, resampling_enum, collection_id, lang, verbose,
            )
            band_arrays.append(arr)
            item_ids_used.update(used_ids)

        array = np.stack(band_arrays).astype(np.float32)
        array = np.where(grid.valid_mask[None, :, :], array, np.nan).astype(np.float32)
        bands = resolved_assets
        item_ids = sorted(item_ids_used)

        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            _write_pc_stack(cache_path, array, bands, grid, collection_id, item_ids)

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"lcz_pc_{safe_id}_stack.tif")
        if cache and os.path.exists(cache_path):
            _atomic_copy_raster(cache_path, out_path)
        else:
            _write_pc_stack(out_path, array, bands, grid, collection_id, item_ids)
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    return LCZPCResult(
        path=cache_path if cache else None,
        array=array,
        bands=bands,
        collection=collection_id,
        item_ids=item_ids,
    )


__all__ = ["lcz_get_planetary_computer", "lcz_list_pc_assets", "LCZPCResult", "PC_COLLECTIONS"]
