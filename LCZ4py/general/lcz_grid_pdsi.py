"""Download Palmer Drought Severity Index (PDSI), cropped to an LCZ map's footprint.

PDSI itself is precomputed upstream (TerraClimate or NOAA PSL/Dai) — this
function only downloads and crops/reprojects it onto the LCZ grid.

Sources
-------
- "terraclimate" (default): Abatzoglou et al. (2018), doi:10.1038/sdata.2017.191,
  1/24 deg (~4km), 1950-2025, one NetCDF file per year.
- "noaa_psl": Dai (2011), doi:10.1029/2010JD015541, 2.5 deg, 1850-2018, single file.

Output is a multi-band GeoTIFF (one band per month), float32 with NaN
nodata, pixel-aligned to the input LCZ map's grid — the same
``LCZGridResult`` convention used by ``lcz_get_lst``/``lcz_get_planetary_computer``.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr

from LCZ4py._internal._lcz_grid_base import download_file
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
_VALID_SOURCES = ("terraclimate", "noaa_psl")
_VALID_RESAMPLING = ("average", "bilinear", "nearest")
_SOURCE_YEAR_RANGE = {"terraclimate": (1950, 2025), "noaa_psl": (1850, 2018)}
_SOURCE_VAR = {"terraclimate": "PDSI", "noaa_psl": "pdsi"}
_TERRACLIMATE_URL = "https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_PDSI_{year:04d}.nc"
_NOAA_PSL_URL = "https://downloads.psl.noaa.gov/Datasets/dai_pdsi/pdsi.mon.mean.selfcalibrated.nc"


def _extract_year(nc_path: str, source: str, year: int, months: list[int], grid, template, resampling: str) -> dict[str, np.ndarray]:
    ds = xr.open_dataset(nc_path)
    da = ds[_SOURCE_VAR[source]]

    if source == "noaa_psl":
        da = da.sel(time=da["time.year"] == year)
    da = da.sel(time=da["time.month"].isin(months))

    out = {}
    if da.sizes.get("time", 0) > 0:
        for t in da["time"].values:
            arr = crop_reproject_dataarray(da.sel(time=t), grid, template, resampling)
            label = pd.Timestamp(t).replace(day=1).date().isoformat()
            out[label] = arr
    ds.close()
    return out


def lcz_grid_pdsi(
    x: Union[str, Path],
    years: Optional[list[int]] = None,
    months: Optional[list[int]] = None,
    source: str = "terraclimate",
    resampling: str = "average",
    isave: bool = False,
    cache: bool = True,
    cache_dir: str = os.path.join(DEFAULT_GRID_CACHE_DIR, "pdsi"),
    lang: str = "en",
    verbose: bool = True,
) -> LCZGridResult:
    """Download Palmer Drought Severity Index cropped to an LCZ map's footprint.

    Parameters
    ----------
    x : str or Path
        Path to an existing LCZ map GeoTIFF (e.g. from ``lcz_get_map``).
    years : list[int], optional
        Years to download. TerraClimate: 1950-2025. NOAA PSL: 1850-2018.
        Defaults to the last two complete years.
    months : list[int], optional
        Months 1-12. Default all 12.
    source : str
        "terraclimate" (default, ~4km) or "noaa_psl" (2.5 deg, longer record).
    resampling : str
        Resampling for regridding onto the LCZ grid: "average" (default),
        "bilinear", or "nearest".
    isave : bool
        Also copy the stack to ``LCZ4r_output/lcz_grid_pdsi.tif``.
    cache : bool
        Reuse a previously built stack for the same (source, years, months,
        LCZ map bbox), and previously downloaded raw NetCDFs.
    cache_dir : str
        Root cache directory. Default ``~/.lcz4r_cache/grid/pdsi``.
    lang : str
        Message language. Default "en".
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    LCZGridResult
        ``array`` is (n_months, H, W) float32, NaN outside the LCZ map's
        valid pixels; ``bands[i]`` gives the ISO date (first-of-month).

    Notes
    -----
    PDSI classification: >=+4 extremely wet, +2..+4 wet, -2..+2 near normal,
    -2..-4 moderate/severe drought, <=-4 extreme drought.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(lcz_msg("grid_pdsi_invalid_source", lang, valid=", ".join(_VALID_SOURCES)))
    if resampling not in _VALID_RESAMPLING:
        raise ValueError(lcz_msg("grid_raster_invalid_resampling", lang, valid=", ".join(_VALID_RESAMPLING)))

    min_year, max_year = _SOURCE_YEAR_RANGE[source]
    current_year = date.today().year
    if years is None:
        years = list(range(current_year - 2, current_year))
        years = [min(y, max_year) for y in years]
        if verbose:
            print(lcz_msg("grid_pdsi_default_years", lang, years=f"{years[0]}-{years[-1]}"))
    else:
        years = sorted(set(int(y) for y in years))
        bad = [y for y in years if y < min_year or y > max_year]
        if bad:
            raise ValueError(lcz_msg("grid_pdsi_invalid_years_range", lang, bad=", ".join(map(str, bad))))

    months = sorted(set(int(m) for m in (months or range(1, 13))))
    if any(m < 1 or m > 12 for m in months):
        raise ValueError(lcz_msg("grid_pdsi_invalid_months", lang))

    grid = load_target_grid(x)
    cache_dir = os.path.expanduser(cache_dir)

    key = cache_key("pdsi", source, years, months, grid.bbox_wgs84, resampling)
    stack_path = os.path.join(cache_dir, "stack", f"pdsi_stack_{key}.tif")

    if cache and os.path.exists(stack_path):
        if verbose:
            print(lcz_msg("grid_raster_cache_hit", lang, name="PDSI"))
        array, bands = read_grid_stack(stack_path)
    else:
        if source == "terraclimate":
            manifest = [dict(
                year=yr, filename=f"TerraClimate_PDSI_{yr:04d}.nc",
                url=_TERRACLIMATE_URL.format(year=yr),
                cache_path=os.path.join(cache_dir, source, f"TerraClimate_PDSI_{yr:04d}.nc"),
            ) for yr in years]
        else:
            manifest = [dict(
                year=years[0], filename="pdsi.mon.mean.selfcalibrated.nc", url=_NOAA_PSL_URL,
                cache_path=os.path.join(cache_dir, source, "pdsi.mon.mean.selfcalibrated.nc"),
            )]

        if verbose:
            print(lcz_msg("grid_pdsi_download_start", lang, n_files=len(manifest), source=source))

        seen_files = {}
        for m in manifest:
            seen_files.setdefault(m["filename"], (m["url"], m["cache_path"]))
        for filename, (url, cp) in seen_files.items():
            download_file(url, cp, use_cache=cache, verbose=verbose)

        template = rio_template(grid)
        layers: dict[str, np.ndarray] = {}
        # terraclimate: one manifest entry per year. noaa_psl: a single file
        # covering all years, queried once per requested year.
        year_entries = (
            [(m, m["year"]) for m in manifest] if source == "terraclimate"
            else [(manifest[0], yr) for yr in years]
        )
        for m, yr in year_entries:
            if not os.path.exists(m["cache_path"]) or os.path.getsize(m["cache_path"]) == 0:
                logger.warning(lcz_msg("grid_pdsi_skip_missing", lang, filename=m["filename"]))
                continue
            try:
                yearly = _extract_year(m["cache_path"], source, yr, months, grid, template, resampling)
            except Exception as exc:
                logger.warning(lcz_msg("grid_pdsi_extract_warn", lang, filename=m["filename"]) + f" ({exc})")
                continue
            layers.update(yearly)

        if not layers:
            raise RuntimeError(lcz_msg("grid_pdsi_no_data", lang))

        bands = sorted(layers)
        array = np.stack([layers[b] for b in bands]).astype(np.float32)
        if cache:
            write_grid_stack(stack_path, array, bands, grid, variable="pdsi", units="index")

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_grid_pdsi.tif")
        write_grid_stack(out_path, array, bands, grid, variable="pdsi", units="index")
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    if verbose:
        print(lcz_msg("grid_raster_done", lang, name="PDSI", n_bands=len(bands)))

    return LCZGridResult(path=stack_path if cache else None, array=array, bands=bands, variables=["pdsi"], units="index")
