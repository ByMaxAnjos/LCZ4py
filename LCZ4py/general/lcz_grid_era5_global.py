"""Download global ERA5-Land monthly-mean climate variables, cropped to an
LCZ map's footprint, via the Copernicus Climate Data Store (CDS).

Unlike ``lcz_grid_era5`` (Latin America only, no auth, daily), this covers
**any region on Earth** but requires a free CDS API key
(https://cds.climate.copernicus.eu -> "API keys" on your profile page), set
via ``cds_key``/``cds_url`` args, ``CDSAPI_KEY``/``CDSAPI_URL`` env vars, or
a ``~/.cdsapirc`` file. The CDS request itself is subset to the LCZ map's
bbox (plus a small margin), so only a small area is ever transferred —
never the whole globe.

Dataset: ``reanalysis-era5-land-monthly-means``, ``product_type =
monthly_averaged_reanalysis`` (Muñoz Sabater, 2019, doi:10.24381/cds.68d2bb30).
Monthly grain only — for daily resolution over Latin America use
``lcz_grid_era5`` instead.

Output is a multi-band GeoTIFF (one band per variable x month), float32
with NaN nodata, pixel-aligned to the input LCZ map's grid — the same
``LCZGridResult`` convention used by ``lcz_get_lst``/``lcz_get_planetary_computer``.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

from LCZ4py._internal._lcz_grid_raster_base import (
    DEFAULT_GRID_CACHE_DIR,
    LCZGridResult,
    cache_key,
    crop_reproject_dataarray,
    load_target_grid,
    read_grid_stack,
    rio_template,
    write_grid_stack,
)
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

OUTPUT_DIR = "LCZ4r_output"
_VALID_RESAMPLING = ("average", "bilinear", "nearest")
_CDS_DATASET = "reanalysis-era5-land-monthly-means"
_BBOX_MARGIN_DEG = 0.25  # > the ~0.1deg native grid, so edge pixels have neighbors for reproject_match

# alias -> (CDS variable name, short name inside the returned NetCDF, unit-conversion fn)
_VAR_MAP = {
    "t2m":  ("2m_temperature", "t2m", lambda a: a - 273.15),
    "td2m": ("2m_dewpoint_temperature", "d2m", lambda a: a - 273.15),
    "u10":  ("10m_u_component_of_wind", "u10", lambda a: a),
    "v10":  ("10m_v_component_of_wind", "v10", lambda a: a),
    "sp":   ("surface_pressure", "sp", lambda a: a / 100),
    "tp":   ("total_precipitation", "tp", lambda a: a * 1000),
    "skt":  ("skin_temperature", "skt", lambda a: a - 273.15),
}


def _cds_client(cds_key: Optional[str], cds_url: Optional[str]):
    if cds_key:
        return cdsapi.Client(url=cds_url or "https://cds.climate.copernicus.eu/api", key=cds_key)
    return cdsapi.Client()  # reads ~/.cdsapirc


def _cds_retrieve(request: dict, target_path: str, cds_key: Optional[str], cds_url: Optional[str]) -> None:
    """Thin wrapper around ``cdsapi.Client.retrieve`` — isolated so tests can
    monkeypatch it instead of hitting the real CDS queue."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    client = _cds_client(cds_key, cds_url)
    client.retrieve(_CDS_DATASET, request, target_path)


def _area_from_grid(grid) -> list[float]:
    """CDS ``area`` is [north, west, south, east], with a small margin so
    edge pixels have real neighbors for average/bilinear reproject_match."""
    minx, miny, maxx, maxy = grid.bbox_wgs84
    m = _BBOX_MARGIN_DEG
    return [min(maxy + m, 90.0), max(minx - m, -180.0), max(miny - m, -90.0), min(maxx + m, 180.0)]


def lcz_grid_era5_global(
    x: Union[str, Path],
    years: list[int],
    months: Optional[list[int]] = None,
    vars: Union[str, list[str]] = ("t2m", "tp"),
    resampling: str = "average",
    cds_key: Optional[str] = None,
    cds_url: Optional[str] = None,
    isave: bool = False,
    cache: bool = True,
    cache_dir: str = os.path.join(DEFAULT_GRID_CACHE_DIR, "era5_global"),
    lang: str = "en",
    verbose: bool = True,
) -> LCZGridResult:
    """Download global ERA5-Land monthly means cropped to an LCZ map's footprint.

    Parameters
    ----------
    x : str or Path
        Path to an existing LCZ map GeoTIFF (e.g. from ``lcz_get_map``), for
        ANY region on Earth (unlike ``lcz_grid_era5``, which is Latin
        America only).
    years : list[int]
        Years to import, 1950-present.
    months : list[int], optional
        Months 1-12. Default all 12.
    vars : str or list[str]
        Variable alias(es), or "all". Options: t2m, td2m, u10, v10, sp, tp,
        skt. Default ("t2m", "tp"). (No t2m_max/t2m_min: the CDS monthly-means
        product only offers the monthly average, not daily extremes.)
    resampling : str
        Resampling for regridding onto the LCZ grid: "average" (default),
        "bilinear", or "nearest".
    cds_key, cds_url : str, optional
        Copernicus CDS API credentials. Default from ``CDSAPI_KEY``/
        ``CDSAPI_URL`` env vars, or a ``~/.cdsapirc`` file if neither is set.
    isave : bool
        Also copy the stack to ``LCZ4r_output/lcz_grid_era5_global.tif``.
    cache : bool
        Reuse a previously built stack for the same (years, months, vars,
        LCZ map bbox), and previously downloaded raw NetCDFs.
    cache_dir : str
        Root cache directory. Default ``~/.lcz4r_cache/grid/era5_global``.
    lang : str
        Message language. Default "en".
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    LCZGridResult
        ``array`` is (n_bands, H, W) float32, NaN outside the LCZ map's
        valid pixels; ``bands[i]`` is ``"{variable}_{year}-{month}-01"``.

    Notes
    -----
    Unit conversions applied: temperature/skin temperature K->C, precipitation
    m->mm, pressure Pa->hPa. Wind components left as m/s. Monthly grain only.
    """
    if not HAS_CDSAPI:
        raise ImportError(lcz_msg("grid_era5_global_no_cdsapi", lang))

    if not years:
        raise ValueError(lcz_msg("grid_era5_missing_years", lang))
    years = sorted(set(int(y) for y in years))
    current_year = date.today().year
    bad_years = [y for y in years if y < 1950 or y > current_year]
    if bad_years:
        raise ValueError(lcz_msg("grid_era5_invalid_years_range", lang, bad=", ".join(map(str, bad_years))))

    months = sorted(set(int(m) for m in (months or range(1, 13))))
    if any(m < 1 or m > 12 for m in months):
        raise ValueError(lcz_msg("grid_era5_invalid_months", lang))

    if vars == "all":
        vars = list(_VAR_MAP.keys())
    elif isinstance(vars, str):
        vars = [vars]
    bad_vars = [v for v in vars if v not in _VAR_MAP]
    if bad_vars:
        raise ValueError(lcz_msg("grid_era5_invalid_vars", lang, bad=", ".join(bad_vars), valid=", ".join(_VAR_MAP)))

    if resampling not in _VALID_RESAMPLING:
        raise ValueError(lcz_msg("grid_raster_invalid_resampling", lang, valid=", ".join(_VALID_RESAMPLING)))

    cds_key = cds_key or os.environ.get("CDSAPI_KEY", "")
    cds_url = cds_url or os.environ.get("CDSAPI_URL", "")
    has_rc = os.path.exists(os.path.expanduser("~/.cdsapirc"))
    if not has_rc and not cds_key:
        raise ValueError(lcz_msg("grid_era5_global_no_auth", lang))

    grid = load_target_grid(x)
    cache_dir = os.path.expanduser(cache_dir)

    key = cache_key("era5_global", years, months, sorted(vars), grid.bbox_wgs84, resampling)
    stack_path = os.path.join(cache_dir, "stack", f"era5_global_stack_{key}.tif")

    if cache and os.path.exists(stack_path):
        if verbose:
            print(lcz_msg("grid_raster_cache_hit", lang, name="ERA5-Land (global/CDS)"))
        array, bands = read_grid_stack(stack_path)
    else:
        cds_vars = sorted({_VAR_MAP[v][0] for v in vars})
        area = _area_from_grid(grid)
        raw_key = cache_key(cds_vars, area)

        manifest = [
            dict(year=yr, month=mo,
                 cache_path=os.path.join(cache_dir, "raw", f"era5land_monthly_{yr:04d}{mo:02d}_{raw_key}.nc"))
            for yr in years for mo in months
        ]

        if verbose:
            print(lcz_msg("grid_era5_global_download_start", lang, n_files=len(manifest)))
        for m in manifest:
            if cache and os.path.exists(m["cache_path"]) and os.path.getsize(m["cache_path"]) > 0:
                if verbose:
                    print(lcz_msg("grid_raster_cache_hit", lang, name=f"{m['year']}-{m['month']:02d}"))
                continue
            request = {
                "product_type": "monthly_averaged_reanalysis",
                "variable": cds_vars,
                "year": str(m["year"]),
                "month": f"{m['month']:02d}",
                "time": "00:00",
                "area": area,
                "format": "netcdf",
            }
            try:
                _cds_retrieve(request, m["cache_path"], cds_key, cds_url)
            except Exception as exc:
                logger.warning(lcz_msg("grid_era5_global_download_warn", lang, year=m["year"], month=m["month"]) + f" ({exc})")

        template = rio_template(grid)
        layers: dict[str, np.ndarray] = {}
        for m in manifest:
            if not os.path.exists(m["cache_path"]) or os.path.getsize(m["cache_path"]) == 0:
                logger.warning(lcz_msg("grid_era5_global_skip_missing", lang, year=m["year"], month=m["month"]))
                continue
            ds = xr.open_dataset(m["cache_path"])
            for alias in vars:
                _, short_name, conv_fn = _VAR_MAP[alias]
                if short_name not in ds.data_vars:
                    continue
                da = ds[short_name]
                if "time" in da.dims:
                    da = da.isel(time=0, drop=True)
                if "valid_time" in da.dims:
                    da = da.isel(valid_time=0, drop=True)
                try:
                    arr = conv_fn(crop_reproject_dataarray(da, grid, template, resampling))
                except Exception as exc:
                    logger.warning(lcz_msg("grid_era5_global_extract_warn", lang, year=m["year"], month=m["month"]) + f" ({exc})")
                    continue
                layers[f"{alias}_{m['year']:04d}-{m['month']:02d}-01"] = arr
            ds.close()

        if not layers:
            raise RuntimeError(lcz_msg("grid_era5_no_data", lang))

        bands = sorted(layers)
        array = np.stack([layers[b] for b in bands]).astype(np.float32)
        if cache:
            write_grid_stack(stack_path, array, bands, grid, variable=",".join(vars), units="mixed")

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_grid_era5_global.tif")
        write_grid_stack(out_path, array, bands, grid, variable=",".join(vars), units="mixed")
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    if verbose:
        print(lcz_msg("grid_raster_done", lang, name="ERA5-Land (global/CDS)", n_bands=len(bands)))

    return LCZGridResult(path=stack_path if cache else None, array=array, bands=bands, variables=list(vars), units="mixed")
