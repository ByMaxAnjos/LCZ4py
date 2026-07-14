"""Shared helpers for the LCZ-map-cropped ``lcz_grid_*`` downloaders
(``lcz_grid_chirps``, ``lcz_grid_era5``, ``lcz_grid_pdsi``,
``lcz_grid_pollution_ghap``, ``lcz_grid_pollution_merra2``).

Each of these downloads a gridded climate/environmental raster and crops +
reprojects it onto an existing LCZ classification GeoTIFF's grid (from
``lcz_get_map``) — the same ``_TargetGrid``/``reproject_match`` pattern used
by ``lcz_get_lst.py`` and ``lcz_get_planetary_computer.py``, rather than
aggregating to municipality polygons.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import rioxarray  # noqa: F401 - registers the .rio accessor
from rasterio.enums import Resampling

from LCZ4py.general.lcz_get_lst import _TargetGrid, _load_target_grid, _rio_template

# Re-exported for the lcz_grid_* modules that only need the grid, not the
# rest of lcz_get_lst's GOES/Sentinel-3 machinery.
load_target_grid = _load_target_grid
rio_template = _rio_template

DEFAULT_GRID_CACHE_DIR = "~/.lcz4r_cache/grid"


@dataclass
class LCZGridResult:
    """Return type shared by the raster ``lcz_grid_*`` downloaders.

    ``array`` is (n_bands, H, W) float32, NaN outside the LCZ map's valid
    pixels (classes 1-17). ``bands`` labels each band (an ISO date for
    single-variable sources, or ``"{variable}_{date}"`` for multi-variable
    ones) and has the same length as ``array``'s first axis.
    """
    path: Optional[str] = None
    array: Optional[np.ndarray] = None
    bands: Optional[list[str]] = None
    variables: Optional[list[str]] = None
    units: Optional[str] = None
    gdf: Optional[gpd.GeoDataFrame] = None
    geoarrow_table: Optional[object] = None


def cache_key(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def crop_reproject_to_grid(
    raster_path: Union[str, Path],
    grid: _TargetGrid,
    template=None,
    resampling: str = "average",
    band: int = 1,
) -> np.ndarray:
    """Crop a raster to ``grid``'s bbox and reproject/resample it onto
    ``grid``'s exact shape/transform/CRS, masked to the LCZ map's valid
    pixels. ``raster_path`` may be a local path or a ``/vsi...`` URL."""
    if template is None:
        template = _rio_template(grid)

    resampling_enum = getattr(Resampling, resampling, Resampling.average)

    da = rioxarray.open_rasterio(str(raster_path), masked=True)
    if "band" in da.dims:
        da = da.isel(band=band - 1, drop=True)

    minx, miny, maxx, maxy = grid.bbox_wgs84
    src_crs = da.rio.crs
    if src_crs is not None and src_crs.to_epsg() != 4326:
        transformer = pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
        xs, ys = transformer.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
    da = da.rio.clip_box(minx, miny, maxx, maxy, auto_expand=True)

    regridded = da.rio.reproject_match(template, resampling=resampling_enum)
    arr = regridded.values.astype("float32")
    return np.where(grid.valid_mask, arr, np.nan)


def crop_reproject_dataarray(da, grid: _TargetGrid, template=None, resampling: str = "average") -> np.ndarray:
    """Same as ``crop_reproject_to_grid`` but for an already-open xarray
    ``DataArray`` (e.g. one time slice of a multi-day NetCDF) instead of a
    raster path. Expects lat/lon (or latitude/longitude) coordinates."""
    if template is None:
        template = _rio_template(grid)
    resampling_enum = getattr(Resampling, resampling, Resampling.average)

    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"
    da = da.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    if not da.rio.crs:
        da = da.rio.write_crs("EPSG:4326", inplace=False)

    minx, miny, maxx, maxy = grid.bbox_wgs84
    da = da.rio.clip_box(minx, miny, maxx, maxy, auto_expand=True)
    regridded = da.rio.reproject_match(template, resampling=resampling_enum)
    arr = regridded.values.astype("float32")
    return np.where(grid.valid_mask, arr, np.nan)


def write_grid_stack(
    path: str,
    array: np.ndarray,
    bands: list[str],
    grid: _TargetGrid,
    variable: str = "",
    units: str = "",
) -> None:
    profile = {
        "driver": "GTiff",
        "height": grid.shape[0],
        "width": grid.shape[1],
        "count": array.shape[0],
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": np.nan,
        "compress": "lzw",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
        dst.update_tags(variable=variable, units=units)
        for i, label in enumerate(bands, start=1):
            dst.set_band_description(i, label)


def read_grid_stack(path: str) -> tuple[np.ndarray, list[str]]:
    with rasterio.open(path) as src:
        array = src.read()
        bands = [src.descriptions[i] for i in range(src.count)]
    return array, bands
