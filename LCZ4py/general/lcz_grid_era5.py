"""Download ERA5-Land daily climate variables, cropped to an LCZ map's footprint.

Source: Saldanha, R. — ERA5-Land Daily Aggregates for Latin America
(1950-2025), ~10km, hosted on Zenodo (CC-BY 4.0), no authentication required.
https://zenodo.org/doi/10.5281/zenodo.10013254

Coverage is genuinely Latin-America-only (confirmed against a downloaded
file: lon -118.57..-34.07, lat -56.75..33.35) — an LCZ map outside that
window raises a clear error pointing at ``lcz_grid_era5_global`` instead of
failing deep inside a "no data" extraction error. That sibling function
covers the whole globe via the Copernicus Climate Data Store, but requires
a free CDS API key.

Output is a multi-band GeoTIFF (one band per variable x day), float32 with
NaN nodata, pixel-aligned to the input LCZ map's grid — the same
``LCZGridResult`` convention used by ``lcz_get_lst``/``lcz_get_planetary_computer``.
"""

from __future__ import annotations

import calendar
import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional, Union

import numpy as np
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
_VALID_RESAMPLING = ("average", "bilinear", "nearest")
# ponytail: rough bbox of the Saldanha Latin-America mirror's real extent
# (measured from a downloaded file), not its true (likely irregular) land
# mask. Good enough to reject obviously-out-of-coverage requests (e.g.
# Berlin) before a confusing "no data" error surfaces from deep inside xarray.
_LATAM_BBOX = (-119.0, -57.0, -34.0, 34.0)

# Zenodo record IDs for ERA5-Land Latin America, by year (1950-2025)
_ERA5_ZENODO_IDS = {
    1950: 10013255, 1951: 10013696, 1952: 10013781, 1953: 10014198, 1954: 10014369,
    1955: 10014474, 1956: 10014693, 1957: 10014722, 1958: 10014754, 1959: 10014771,
    1960: 10014790, 1961: 10020497, 1962: 10020520, 1963: 10020530, 1964: 10020539,
    1965: 10020552, 1966: 10020600, 1967: 10020663, 1968: 10020679, 1969: 10020690,
    1970: 10020859, 1971: 10021122, 1972: 10021300, 1973: 10021667, 1974: 10021706,
    1975: 10021943, 1976: 10021943, 1977: 10022017, 1978: 10022061, 1979: 10022145,
    1980: 10022315, 1981: 10022536, 1982: 10022546, 1983: 10022561, 1984: 10022571,
    1985: 10022579, 1986: 10022589, 1987: 10022593, 1988: 10022607, 1989: 10022632,
    1990: 10022641, 1991: 10032814, 1992: 10032859, 1993: 10033251, 1994: 10033276,
    1995: 10033306, 1996: 10033353, 1997: 10033755, 1998: 10033835, 1999: 10033983,
    2000: 10033995, 2001: 10034036, 2002: 10034077, 2003: 10034110, 2004: 10034145,
    2005: 10034179, 2006: 10034204, 2007: 10034283, 2008: 10034323, 2009: 10034370,
    2010: 10034386, 2011: 10034412, 2012: 10034443, 2013: 10034494, 2014: 10034523,
    2015: 10034541, 2016: 10034598, 2017: 10034630, 2018: 10036123, 2019: 10036132,
    2020: 10036153, 2021: 10036162, 2022: 10036168, 2023: 10889682, 2024: 15748090,
    2025: 18256859,
}

# alias -> (Zenodo indicator, agg label, unit-conversion fn)
_VAR_MAP = {
    "t2m":     ("2m_temperature", "mean", lambda a: a - 273.15),
    "t2m_max": ("2m_temperature", "max", lambda a: a - 273.15),
    "t2m_min": ("2m_temperature", "min", lambda a: a - 273.15),
    "td2m":    ("2m_dewpoint_temperature", "mean", lambda a: a - 273.15),
    "u10":     ("10m_u_component_of_wind", "mean", lambda a: a),
    "v10":     ("10m_v_component_of_wind", "mean", lambda a: a),
    "sp":      ("surface_pressure", "mean", lambda a: a / 100),
    "tp":      ("total_precipitation", "sum", lambda a: a * 1000),
}


def _nc_filename(indicator: str, year: int, month: int, agg: str) -> str:
    last_day = calendar.monthrange(year, month)[1]
    return f"{indicator}_{year:04d}-{month:02d}-01_{year:04d}-{month:02d}-{last_day:02d}_day_{agg}.nc"


def _zenodo_url(year: int, filename: str) -> str:
    record_id = _ERA5_ZENODO_IDS.get(year)
    if record_id is None:
        raise ValueError(f"No Zenodo record ID found for year {year}.")
    return f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"


def _in_latam(grid) -> bool:
    x0, y0, x1, y1 = grid.bbox_wgs84
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0, bx1, by1 = _LATAM_BBOX
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def _extract_daily_layers(nc_path: str, grid, template, resampling: str, conv_fn, start_date: date) -> dict[str, np.ndarray]:
    ds = xr.open_dataset(nc_path)
    var_name = next(iter(ds.data_vars))
    da = ds[var_name]

    time_name = "time" if "time" in da.coords else None
    n_layers = da.sizes.get(time_name, 1) if time_name else 1

    dates = None
    if time_name is not None:
        try:
            import pandas as pd
            dates = pd.to_datetime(da[time_name].values)
            if len(dates) != n_layers or dates.isna().any():
                dates = None
        except Exception:
            dates = None
    if dates is None:
        import pandas as pd
        dates = pd.date_range(start_date, periods=n_layers, freq="D")

    out = {}
    iterator = da[time_name].values if time_name else [None]
    for t, d in zip(iterator, dates):
        day_da = da.sel({time_name: t}) if time_name else da
        arr = conv_fn(crop_reproject_dataarray(day_da, grid, template, resampling))
        out[d.date().isoformat()] = arr
    ds.close()
    return out


def lcz_grid_era5(
    x: Union[str, Path],
    years: list[int],
    months: Optional[list[int]] = None,
    vars: Union[str, list[str]] = ("t2m", "tp"),
    resampling: str = "average",
    isave: bool = False,
    cache: bool = True,
    cache_dir: str = os.path.join(DEFAULT_GRID_CACHE_DIR, "era5"),
    lang: str = "en",
    verbose: bool = True,
) -> LCZGridResult:
    """Download ERA5-Land daily climate variables cropped to an LCZ map's footprint.

    Parameters
    ----------
    x : str or Path
        Path to an existing LCZ map GeoTIFF (e.g. from ``lcz_get_map``).
    years : list[int]
        Years to import, 1950-2025.
    months : list[int], optional
        Months 1-12. Default all 12.
    vars : str or list[str]
        Variable alias(es), or "all". Options: t2m, t2m_max, t2m_min, td2m,
        u10, v10, sp, tp. Default ("t2m", "tp").
    resampling : str
        Resampling for regridding onto the LCZ grid: "average" (default),
        "bilinear", or "nearest".
    isave : bool
        Also copy the stack to ``LCZ4r_output/lcz_grid_era5.tif``.
    cache : bool
        Reuse a previously built stack for the same (years, months, vars,
        LCZ map bbox), and previously downloaded raw NetCDFs.
    cache_dir : str
        Root cache directory. Default ``~/.lcz4r_cache/grid/era5``.
    lang : str
        Message language. Default "en".
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    LCZGridResult
        ``array`` is (n_bands, H, W) float32, NaN outside the LCZ map's
        valid pixels; ``bands[i]`` is ``"{variable}_{date}"``.

    Notes
    -----
    Unit conversions applied: temperature K->C, precipitation m->mm,
    pressure Pa->hPa. Wind components left as m/s.

    Coverage is Latin America only (raises ``ValueError`` otherwise) — use
    ``lcz_grid_era5_global`` for any other region (requires a free CDS API key).
    """
    if not years:
        raise ValueError(lcz_msg("grid_era5_missing_years", lang))
    years = sorted(set(int(y) for y in years))
    bad_years = [y for y in years if y < 1950 or y > 2025]
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

    grid = load_target_grid(x)
    if not _in_latam(grid):
        raise ValueError(lcz_msg("grid_era5_out_of_coverage", lang))
    cache_dir = os.path.expanduser(cache_dir)

    key = cache_key("era5", years, months, sorted(vars), grid.bbox_wgs84, resampling)
    stack_path = os.path.join(cache_dir, "stack", f"era5_stack_{key}.tif")

    if cache and os.path.exists(stack_path):
        if verbose:
            print(lcz_msg("grid_raster_cache_hit", lang, name="ERA5-Land"))
        array, bands = read_grid_stack(stack_path)
    else:
        # dedupe by (indicator, agg_label, year, month) so t2m_max/t2m_min
        # sharing one Zenodo file aren't downloaded twice.
        file_manifest: dict[tuple, dict] = {}
        for yr in years:
            for mo in months:
                for alias in vars:
                    indicator, agg_label, _ = _VAR_MAP[alias]
                    fkey = (indicator, agg_label, yr, mo)
                    if fkey in file_manifest:
                        continue
                    filename = _nc_filename(indicator, yr, mo, agg_label)
                    file_manifest[fkey] = dict(
                        year=yr, month=mo, filename=filename, url=_zenodo_url(yr, filename),
                        cache_path=os.path.join(cache_dir, str(yr), filename),
                    )

        if verbose:
            print(lcz_msg("grid_era5_download_start", lang, n_files=len(file_manifest)))
        for entry in file_manifest.values():
            download_file(entry["url"], entry["cache_path"], use_cache=cache, verbose=verbose)

        template = rio_template(grid)

        layers: dict[str, np.ndarray] = {}
        for alias in vars:
            indicator, agg_label, conv_fn = _VAR_MAP[alias]
            for yr in years:
                for mo in months:
                    entry = file_manifest[(indicator, agg_label, yr, mo)]
                    if not os.path.exists(entry["cache_path"]) or os.path.getsize(entry["cache_path"]) == 0:
                        logger.warning(lcz_msg("grid_era5_skip_missing", lang, filename=entry["filename"]))
                        continue
                    try:
                        daily = _extract_daily_layers(
                            entry["cache_path"], grid, template, resampling, conv_fn, date(yr, mo, 1),
                        )
                    except Exception as exc:
                        logger.warning(lcz_msg("grid_era5_extract_warn", lang, file=entry["filename"]) + f" ({exc})")
                        continue
                    for d, arr in daily.items():
                        layers[f"{alias}_{d}"] = arr

        if not layers:
            raise RuntimeError(lcz_msg("grid_era5_no_data", lang))

        bands = sorted(layers)
        array = np.stack([layers[b] for b in bands]).astype(np.float32)
        if cache:
            write_grid_stack(stack_path, array, bands, grid, variable=",".join(vars), units="mixed")

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_grid_era5.tif")
        write_grid_stack(out_path, array, bands, grid, variable=",".join(vars), units="mixed")
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    if verbose:
        print(lcz_msg("grid_raster_done", lang, name="ERA5-Land", n_bands=len(bands)))

    return LCZGridResult(path=stack_path if cache else None, array=array, bands=bands, variables=list(vars), units="mixed")
