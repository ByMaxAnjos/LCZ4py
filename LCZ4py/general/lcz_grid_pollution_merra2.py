"""Download NASA MERRA-2 aerosol/pollution reanalysis, cropped to an LCZ
map's footprint.

Requires a free NASA Earthdata Login (https://urs.earthdata.nasa.gov), set
via EARTHDATA_USER/EARTHDATA_PASSWORD env vars, explicit args, or a .netrc
file.

Reference: Gelaro et al. (2017), doi:10.1175/JCLI-D-16-0758.1.

Note: for resolution="daily" this still requests the monthly-mean
M2TMNXAER file (the hourly M2T1NXAER path is documented but not implemented
upstream), so "daily" and "monthly" download the same file and emit one
monthly-dated band. Ported as-is for R/Python parity.

Output is a multi-band GeoTIFF (one band per pollutant x month), float32
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
_VALID_POLLUTANTS = ("pm25", "aod", "so2")
_VALID_RESOLUTIONS = ("monthly", "daily")
_VALID_RESAMPLING = ("average", "bilinear", "nearest")

_VAR_MAP = {
    "pm25": {
        "nc_vars": ["DUSMASS25", "SSSMASS25", "BCSMASS", "OCSMASS", "SO4SMASS"],
        "derive": lambda d: (d["DUSMASS25"] + d["SSSMASS25"] + d["BCSMASS"] + 1.4 * d["OCSMASS"] + d["SO4SMASS"]) * 1e9,
    },
    "aod": {
        "nc_vars": ["TOTEXTTAU"],
        "derive": lambda d: d["TOTEXTTAU"],
    },
    "so2": {
        "nc_vars": ["SO2SMASS"],
        "derive": lambda d: d["SO2SMASS"] * 1e9,
    },
}


def _version_code(year: int) -> str:
    if year < 1992:
        return "100"
    if year < 2001:
        return "200"
    if year < 2011:
        return "300"
    return "400"


def _file_info(year: int, month: int) -> tuple[str, str]:
    ver = _version_code(year)
    fname = f"MERRA2_{ver}.tavgM_2d_aer_Nx.{year:04d}{month:02d}01.nc4"
    base = f"https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2TMNXAER.5.12.4/{year:04d}/"
    return fname, base + fname


def _download_auth(url: str, cache_path: str, use_cache: bool, user: str, password: str, netrc_path: Optional[str], verbose: bool) -> bool:
    import httpx

    if use_cache and os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        if verbose:
            print(f"Cache found: {os.path.basename(cache_path)}")
        return True

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        auth = None if netrc_path else (user, password)
        client_kwargs = {"follow_redirects": True, "timeout": 120.0}
        if netrc_path:
            client_kwargs["trust_env"] = True  # relies on NETRC env var / ~/.netrc
        with httpx.Client(auth=auth, **client_kwargs) as client:
            resp = client.get(url)
            resp.raise_for_status()
            with open(cache_path, "wb") as f:
                f.write(resp.content)
        if verbose:
            print(f"Downloaded: {os.path.basename(cache_path)}")
        return True
    except Exception as exc:
        if os.path.exists(cache_path):
            os.remove(cache_path)
        logger.warning("Failed to download %s: %s", url, exc)
        return False


def _extract_period(nc_path: str, pollutant: str, grid, template, resampling: str) -> np.ndarray:
    vmap = _VAR_MAP[pollutant]
    ds = xr.open_dataset(nc_path)
    arrs = {}
    for v in vmap["nc_vars"]:
        da = ds[v]
        extra_dims = [d for d in da.dims if d not in ("lat", "lon", "latitude", "longitude")]
        if extra_dims:
            da = da.mean(dim=extra_dims)
        arrs[v] = crop_reproject_dataarray(da, grid, template, resampling)
    ds.close()
    return vmap["derive"](arrs).astype("float32")


def lcz_grid_pollution_merra2(
    x: Union[str, Path],
    pollutants: list[str] = ("pm25", "aod"),
    resolution: str = "monthly",
    years: Optional[list[int]] = None,
    months: Optional[list[int]] = None,
    resampling: str = "average",
    earthdata_user: Optional[str] = None,
    earthdata_pass: Optional[str] = None,
    netrc_path: Optional[str] = None,
    isave: bool = False,
    cache: bool = True,
    cache_dir: str = os.path.join(DEFAULT_GRID_CACHE_DIR, "merra2"),
    lang: str = "en",
    verbose: bool = True,
) -> LCZGridResult:
    """Download MERRA-2 aerosol/pollution data cropped to an LCZ map's footprint.

    Parameters
    ----------
    x : str or Path
        Path to an existing LCZ map GeoTIFF (e.g. from ``lcz_get_map``).
    pollutants : list[str]
        "pm25", "aod", "so2". Default ("pm25", "aod").
    resolution : str
        "monthly" (default) or "daily". so2 is monthly-only.
    years : list[int], optional
        Years 1980-present. Defaults to the last two complete years.
    months : list[int], optional
        Months 1-12. Default all 12.
    resampling : str
        Resampling for regridding onto the LCZ grid: "average" (default),
        "bilinear", or "nearest".
    earthdata_user, earthdata_pass : str, optional
        NASA Earthdata Login credentials. Default from EARTHDATA_USER /
        EARTHDATA_PASSWORD env vars.
    netrc_path : str, optional
        Path to a .netrc file; takes precedence over user/pass if given.
    isave : bool
        Also copy the stack to ``LCZ4r_output/lcz_grid_merra2.tif``.
    cache : bool
        Reuse a previously built stack for the same (pollutants, years,
        months, LCZ map bbox), and previously downloaded raw NetCDFs.
    cache_dir : str
        Root cache directory. Default ``~/.lcz4r_cache/grid/merra2``.
    lang : str
        Message language. Default "en".
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    LCZGridResult
        ``array`` is (n_bands, H, W) float32, NaN outside the LCZ map's
        valid pixels; ``bands[i]`` is ``"{pollutant}_{date}"``.

    Notes
    -----
    PM2.5 = (DUSMASS25 + SSSMASS25 + BCSMASS + 1.4*OCSMASS + SO4SMASS) * 1e9
    (kg/m3 -> ug/m3, organic carbon scaled 1.4x for organic matter).
    """
    pollutants = list(pollutants)
    bad_p = [p for p in pollutants if p not in _VALID_POLLUTANTS]
    if bad_p:
        raise ValueError(lcz_msg("grid_merra2_invalid_pollutants", lang, bad=", ".join(bad_p)))
    if resolution not in _VALID_RESOLUTIONS:
        raise ValueError(lcz_msg("grid_merra2_invalid_resolution", lang))
    if resampling not in _VALID_RESAMPLING:
        raise ValueError(lcz_msg("grid_raster_invalid_resampling", lang, valid=", ".join(_VALID_RESAMPLING)))

    if "so2" in pollutants and resolution == "daily":
        logger.warning(lcz_msg("grid_merra2_so2_monthly_only", lang))
        pollutants = [p for p in pollutants if p != "so2"]
        if not pollutants:
            raise ValueError(lcz_msg("grid_merra2_no_pollutants", lang))

    current_year = date.today().year
    if years is None:
        years = [current_year - 2, current_year - 1]
    else:
        years = [int(y) for y in years]
        bad_years = [y for y in years if y < 1980 or y > current_year]
        if bad_years:
            raise ValueError(lcz_msg("grid_merra2_invalid_years_range", lang, bad=", ".join(map(str, bad_years))))

    months = sorted(set(int(m) for m in (months or range(1, 13))))
    if any(m < 1 or m > 12 for m in months):
        raise ValueError(lcz_msg("grid_merra2_invalid_months", lang))

    earthdata_user = earthdata_user or os.environ.get("EARTHDATA_USER", "")
    earthdata_pass = earthdata_pass or os.environ.get("EARTHDATA_PASSWORD", "")
    has_netrc = netrc_path and os.path.exists(netrc_path)
    if not has_netrc and not (earthdata_user and earthdata_pass):
        raise ValueError(lcz_msg("grid_merra2_no_auth", lang))

    grid = load_target_grid(x)
    cache_dir = os.path.expanduser(cache_dir)

    key = cache_key("merra2", sorted(pollutants), resolution, years, months, grid.bbox_wgs84, resampling)
    stack_path = os.path.join(cache_dir, "stack", f"merra2_stack_{key}.tif")

    if cache and os.path.exists(stack_path):
        if verbose:
            print(lcz_msg("grid_raster_cache_hit", lang, name="MERRA-2"))
        array, bands = read_grid_stack(stack_path)
    else:
        # ── Build manifest (one file per year x month, shared across pollutants) ──
        manifest = []
        for yr in years:
            for mo in months:
                filename, url = _file_info(yr, mo)
                manifest.append(dict(year=yr, month=mo, filename=filename, url=url,
                                      cache_nc=os.path.join(cache_dir, filename)))

        unique_files = {m["filename"]: (m["url"], m["cache_nc"]) for m in manifest}
        if verbose:
            print(lcz_msg("grid_merra2_download_start", lang, n_files=len(unique_files)))
        for filename, (url, cache_nc) in unique_files.items():
            _download_auth(url, cache_nc, cache, earthdata_user, earthdata_pass, netrc_path, verbose)

        template = rio_template(grid)
        layers: dict[str, np.ndarray] = {}
        for m in manifest:
            if not os.path.exists(m["cache_nc"]) or os.path.getsize(m["cache_nc"]) == 0:
                logger.warning(lcz_msg("grid_merra2_skip_missing", lang, filename=m["filename"]))
                continue
            for p in pollutants:
                label = f"{p}_{m['year']:04d}-{m['month']:02d}-01"
                try:
                    layers[label] = _extract_period(m["cache_nc"], p, grid, template, resampling)
                except Exception as exc:
                    logger.warning(lcz_msg("grid_merra2_extract_warn", lang, filename=m["filename"]) + f" ({exc})")

        if not layers:
            raise RuntimeError(lcz_msg("grid_merra2_no_data", lang))

        bands = sorted(layers)
        array = np.stack([layers[b] for b in bands]).astype(np.float32)
        if cache:
            write_grid_stack(stack_path, array, bands, grid, variable=",".join(pollutants), units="mixed")

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_grid_merra2.tif")
        write_grid_stack(out_path, array, bands, grid, variable=",".join(pollutants), units="mixed")
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    if verbose:
        print(lcz_msg("grid_raster_done", lang, name="MERRA-2", n_bands=len(bands)))

    return LCZGridResult(path=stack_path if cache else None, array=array, bands=bands, variables=list(pollutants), units="mixed")
