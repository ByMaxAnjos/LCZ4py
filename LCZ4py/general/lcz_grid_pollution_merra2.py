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

Monthly files are downloaded in parallel (bounded by ``max_workers``) with a
single reused HTTP client; each file is then opened once to derive every
requested pollutant (PM2.5 needs five variables from the same file).
"""

from __future__ import annotations

import logging
import netrc as netrc_module
import os
from datetime import date
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import numpy as np

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:  # pragma: no cover - depends on the environment
    xr = None  # type: ignore[assignment]
    HAS_XARRAY = False

from LCZ4py._internal._lcz_grid_base import (
    download_many,
    normalize_int_list,
    resolve_max_workers,
)
from LCZ4py._internal._lcz_grid_raster_base import (
    DEFAULT_GRID_CACHE_DIR,
    LCZGridResult,
    cache_key,
    crop_reproject_dataarray,
    load_target_grid,
    make_template_factory,
    try_read_grid_stack,
    write_grid_stack,
)
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

OUTPUT_DIR = "LCZ4r_output"
_VALID_POLLUTANTS = ("pm25", "aod", "so2")
_VALID_RESOLUTIONS = ("monthly", "daily")
_VALID_RESAMPLING = ("average", "bilinear", "nearest")
_FIRST_YEAR = 1980

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

_SPATIAL_NAMES = ("lat", "lon", "latitude", "longitude", "x", "y")


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


def _resolve_auth(
    url: str,
    user: Optional[str],
    password: Optional[str],
    netrc_path: Optional[str],
) -> Optional[tuple[str, str]]:
    """Build an ``(user, password)`` pair from a .netrc file or explicit creds."""
    if netrc_path:
        expanded = os.path.expanduser(netrc_path)
        if not os.path.exists(expanded):
            logger.warning("netrc file not found: %s", expanded)
        else:
            try:
                host = urlparse(url).hostname or ""
                parsed = netrc_module.netrc(expanded)
                # Try the exact host first, then the shared Earthdata login
                # host, then the only entry in the file (common when users
                # keep a single `machine urs.earthdata.nasa.gov` block).
                candidates = [host, "urs.earthdata.nasa.gov"]
                entry = None
                for candidate in candidates:
                    if not candidate:
                        continue
                    entry = parsed.authenticators(candidate)
                    if entry:
                        break
                if entry is None and len(parsed.hosts) == 1:
                    entry = next(iter(parsed.hosts.values()))
                if entry and entry[0] and entry[2]:
                    return (entry[0], entry[2])
                logger.warning(
                    "No .netrc credentials for %r in %s; falling back to "
                    "EARTHDATA_USER/EARTHDATA_PASSWORD.",
                    host, expanded,
                )
            except Exception as exc:  # noqa: BLE001 - malformed netrc
                logger.warning("Could not read netrc file %s: %s", expanded, exc)

    if user and password:
        return (user, password)
    return None


def _download_auth(
    url: str,
    cache_path: str,
    use_cache: bool,
    user: str,
    password: str,
    netrc_path: Optional[str],
    verbose: bool,
) -> bool:
    """Earthdata-authenticated download (atomic, retried, cache-aware).

    Kept as a thin wrapper for backward compatibility; batch downloads go
    through :func:`download_many` with a shared authenticated client.
    """
    if use_cache and os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        if verbose:
            print(f"Cache found: {os.path.basename(cache_path)}")
        return True

    from LCZ4py._internal._lcz_grid_base import download_file

    auth = _resolve_auth(url, user, password, netrc_path)
    return download_file(
        url, cache_path, use_cache=use_cache, verbose=verbose, auth=auth,
    )


def _extract_period(
    nc_path: str,
    pollutant: str,
    grid,
    template,
    resampling: str,
) -> np.ndarray:
    """Backward-compatible single-pollutant wrapper around :func:`_extract_file`."""
    arrays = _extract_file(nc_path, [pollutant], grid, lambda: template, resampling)
    return arrays[pollutant]


def _extract_file(
    nc_path: str,
    pollutants: list[str],
    grid,
    template_factory,
    resampling: str,
) -> dict[str, np.ndarray]:
    """Derive every requested pollutant from one MERRA-2 file (single open)."""
    if not HAS_XARRAY:
        raise ImportError(
            "Reading MERRA-2 NetCDF files requires 'xarray'. "
            "Install it with: pip install xarray"
        )

    needed = sorted({v for p in pollutants for v in _VAR_MAP[p]["nc_vars"]})
    derived_inputs: dict[str, np.ndarray] = {}

    ds = xr.open_dataset(nc_path)
    try:
        missing = [v for v in needed if v not in ds.data_vars]
        if missing:
            raise ValueError(
                f"Variable(s) {', '.join(missing)} not found in "
                f"{os.path.basename(nc_path)} (available: "
                f"{', '.join(map(str, ds.data_vars)) or 'none'})."
            )
        for var in needed:
            da = ds[var]
            extra_dims = [d for d in da.dims if d not in _SPATIAL_NAMES]
            if extra_dims:
                # e.g. (time, lev) on an aerosol file: monthly/column mean,
                # exactly as the R implementation does.
                da = da.mean(dim=extra_dims)
            derived_inputs[var] = crop_reproject_dataarray(
                da, grid, template_factory(), resampling,
            )
    finally:
        ds.close()

    out: dict[str, np.ndarray] = {}
    for p in pollutants:
        out[p] = np.asarray(_VAR_MAP[p]["derive"](derived_inputs), dtype="float32")
    return out


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
    max_workers: Optional[int] = None,
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
    max_workers : int, optional
        Threads used for parallel downloads. Default
        ``min(8, os.cpu_count() or 4)``.

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
    pollutants = [pollutants] if isinstance(pollutants, str) else list(pollutants)
    pollutants = [str(p).lower() for p in pollutants]
    pollutants = list(dict.fromkeys(pollutants))
    bad_p = [p for p in pollutants if p not in _VALID_POLLUTANTS]
    if bad_p:
        raise ValueError(lcz_msg(
            "grid_merra2_invalid_pollutants", lang, bad=", ".join(bad_p),
        ))
    if not pollutants:
        raise ValueError(lcz_msg("grid_merra2_no_pollutants", lang))
    if resolution not in _VALID_RESOLUTIONS:
        raise ValueError(lcz_msg("grid_merra2_invalid_resolution", lang))
    if resampling not in _VALID_RESAMPLING:
        raise ValueError(lcz_msg(
            "grid_raster_invalid_resampling", lang, valid=", ".join(_VALID_RESAMPLING),
        ))

    if "so2" in pollutants and resolution == "daily":
        logger.warning(lcz_msg("grid_merra2_so2_monthly_only", lang))
        pollutants = [p for p in pollutants if p != "so2"]
        if not pollutants:
            raise ValueError(lcz_msg("grid_merra2_no_pollutants", lang))

    current_year = date.today().year
    if years is None:
        years = [current_year - 2, current_year - 1]
    years = sorted(set(normalize_int_list(years, "years")))
    if not years:
        raise ValueError(lcz_msg(
            "grid_merra2_invalid_years_range", lang, bad="<empty>",
        ))
    bad_years = [y for y in years if y < _FIRST_YEAR or y > current_year]
    if bad_years:
        raise ValueError(lcz_msg(
            "grid_merra2_invalid_years_range", lang,
            bad=", ".join(map(str, bad_years)),
        ))

    months = sorted(set(normalize_int_list(months, "months", default=range(1, 13))))
    if any(m < 1 or m > 12 for m in months):
        raise ValueError(lcz_msg("grid_merra2_invalid_months", lang))

    earthdata_user = earthdata_user or os.environ.get("EARTHDATA_USER", "")
    earthdata_pass = earthdata_pass or os.environ.get("EARTHDATA_PASSWORD", "")
    netrc_path = os.path.expanduser(netrc_path) if netrc_path else None
    has_netrc = bool(netrc_path and os.path.exists(netrc_path))
    if not has_netrc and not (earthdata_user and earthdata_pass):
        raise ValueError(lcz_msg("grid_merra2_no_auth", lang))

    grid = load_target_grid(x)
    cache_dir = os.path.expanduser(cache_dir)

    key = cache_key("merra2", sorted(pollutants), resolution, years, months, grid.bbox_wgs84, resampling)
    stack_path = os.path.join(cache_dir, "stack", f"merra2_stack_{key}.tif")

    array: Optional[np.ndarray] = None
    bands: list[str] = []
    if cache:
        cached = try_read_grid_stack(stack_path)
        if cached is not None:
            if verbose:
                print(lcz_msg("grid_raster_cache_hit", lang, name="MERRA-2"))
            array, bands = cached

    if array is None:
        # ── Build manifest (one file per year x month, shared across pollutants) ──
        manifest = []
        for yr in years:
            for mo in months:
                filename, url = _file_info(yr, mo)
                manifest.append(dict(
                    year=yr, month=mo, filename=filename, url=url,
                    cache_nc=os.path.join(cache_dir, filename),
                ))

        unique_files = {m["filename"]: (m["url"], m["cache_nc"]) for m in manifest}
        if verbose:
            print(lcz_msg("grid_merra2_download_start", lang, n_files=len(unique_files)))

        workers = resolve_max_workers(max_workers, len(unique_files))
        auth = _resolve_auth(manifest[0]["url"], earthdata_user, earthdata_pass, netrc_path)
        download_many(
            unique_files, use_cache=cache, verbose=verbose, max_workers=workers,
            auth=auth, progress_desc="MERRA-2",
        )

        template = make_template_factory(grid)
        layers: dict[str, np.ndarray] = {}
        for m in manifest:
            if not os.path.exists(m["cache_nc"]) or os.path.getsize(m["cache_nc"]) == 0:
                logger.warning(lcz_msg(
                    "grid_merra2_skip_missing", lang, filename=m["filename"],
                ))
                continue
            try:
                derived = _extract_file(
                    m["cache_nc"], pollutants, grid, template, resampling,
                )
            except Exception as exc:  # noqa: BLE001 - reported per file
                logger.warning(
                    lcz_msg("grid_merra2_extract_warn", lang, filename=m["filename"])
                    + f" ({exc})"
                )
                continue
            for p, arr in derived.items():
                layers[f"{p}_{m['year']:04d}-{m['month']:02d}-01"] = arr

        if not layers:
            raise RuntimeError(lcz_msg("grid_merra2_no_data", lang))

        bands = sorted(layers)
        array = np.stack([layers[b] for b in bands]).astype(np.float32)
        if cache:
            write_grid_stack(
                stack_path, array, bands, grid,
                variable=",".join(pollutants), units="mixed",
            )

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_grid_merra2.tif")
        write_grid_stack(
            out_path, array, bands, grid,
            variable=",".join(pollutants), units="mixed",
        )
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    if verbose:
        print(lcz_msg("grid_raster_done", lang, name="MERRA-2", n_bands=len(bands)))

    return LCZGridResult(
        path=stack_path if cache else None,
        array=array, bands=bands, variables=list(pollutants), units="mixed",
    )
