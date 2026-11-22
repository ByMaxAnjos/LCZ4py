"""Download GHAP (GlobalHighAirPollutants) high-resolution pollution rasters,
cropped to an LCZ map's footprint.

GHAP (Wei et al., doi:10.5281/zenodo.10800980) provides AI-generated
ground-level pollutant fields: PM2.5 (1km, daily/monthly/annual, 2017-2022),
O3 (10km, annual, 2000-2020), CO (1km, annual, 2019-2022). Zenodo, no auth.

Output is a multi-band GeoTIFF (one band per pollutant x period), float32
with NaN nodata, pixel-aligned to the input LCZ map's grid — the same
``LCZGridResult`` convention used by ``lcz_get_lst``/``lcz_get_planetary_computer``.

Files are downloaded in parallel; extraction stays serial because the
NetCDF/HDF5 reader is not thread-safe.
"""

from __future__ import annotations

import glob
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Union

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
_VALID_POLLUTANTS = ("pm25", "o3", "co")
_VALID_RESOLUTIONS = ("daily", "monthly", "annual")
_VALID_RESAMPLING = ("average", "bilinear", "nearest")
_ANNUAL_ONLY = ("o3", "co")

_RECORDS = {
    "pm25": {"monthly": "10800980", "annual": "10800980",
              "daily": {2017: "10801181", 2018: "10795801", 2019: "10799037",
                        2020: "10800555", 2021: "10799203", 2022: "10795662"}},
    "o3": {"annual": "10208188"},
    "co": {"annual": "14207363"},
}
_AVAIL_YEARS = {
    "pm25": {"daily": range(2017, 2023), "monthly": range(2017, 2023), "annual": range(2017, 2023)},
    "o3": {"annual": range(2000, 2021)},
    "co": {"annual": range(2019, 2023)},
}
_RES_CODE = {"pm25": "1K", "o3": "1K", "co": "1K"}
_P_FNAME = {"pm25": "PM2.5", "o3": "O3", "co": "CO"}

def _file_info(pollutant: str, resolution: str, year: int, month: Optional[int] = None) -> tuple[str, str]:
    p_fname = _P_FNAME[pollutant]
    res_code = _RES_CODE[pollutant]
    if resolution == "daily" and month is not None:
        ym = f"{year:04d}{month:02d}"
        filename = f"GHAP_{p_fname}_D{res_code}_{ym}_V1.zip"
        record_id = _RECORDS[pollutant]["daily"].get(year)
        if not record_id:
            raise ValueError(f"No daily Zenodo record for {pollutant} year {year}.")
    elif resolution == "monthly" and month is not None:
        ym = f"{year:04d}{month:02d}"
        filename = f"GHAP_{p_fname}_M{res_code}_{ym}_V1.nc"
        record_id = _RECORDS[pollutant]["monthly"]
    else:
        filename = f"GHAP_{p_fname}_Y{res_code}_{year:04d}_V1.nc"
        record_id = _RECORDS[pollutant]["annual"]
    url = f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"
    return filename, url


def _read_nc_array(nc_path: str, grid, template, resampling: str) -> np.ndarray:
    """Read the first variable of a GHAP NetCDF and put it on the LCZ grid."""
    if not HAS_XARRAY:
        raise ImportError(
            "Reading GHAP NetCDF files requires 'xarray'. "
            "Install it with: pip install xarray"
        )

    ds = xr.open_dataset(nc_path)
    try:
        if not ds.data_vars:
            raise ValueError(f"No variables found in {os.path.basename(nc_path)}.")
        da = ds[next(iter(ds.data_vars))]
        # Collapse extra (non-spatial, non-time) dims and the time axis.
        for dim in ("time", "valid_time"):
            if dim in da.dims:
                da = da.isel({dim: 0}, drop=True)
        return crop_reproject_dataarray(da, grid, template, resampling)
    finally:
        ds.close()


def _extract_daily_zip(
    zip_path: str,
    grid,
    template_factory,
    resampling: str,
    lang: str = "en",
) -> dict[str, np.ndarray]:
    """Extract every daily NetCDF from a GHAP monthly zip onto the LCZ grid."""
    if not HAS_XARRAY:
        raise ImportError(
            "Reading GHAP NetCDF files requires 'xarray'. "
            "Install it with: pip install xarray"
        )

    out: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="ghap_") as tmp_dir:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = info.filename
                # Guard against path traversal in the archive.
                if name.startswith("/") or ".." in Path(name).parts:
                    logger.warning("Skipping unsafe archive member %r", name)
                    continue
                zf.extract(info, tmp_dir)

        nc_files = sorted(glob.glob(os.path.join(tmp_dir, "**", "*.nc"), recursive=True))
        if not nc_files:
            logger.warning(lcz_msg(
                "grid_ghap_zip_no_nc", lang, filename=os.path.basename(zip_path),
            ))
            return out

        for nc_path in nc_files:
            try:
                bn = os.path.splitext(os.path.basename(nc_path))[0]
                date_parts = [p for p in bn.split("_") if len(p) == 8 and p.isdigit()]
                day_label = (
                    f"{date_parts[0][:4]}-{date_parts[0][4:6]}-{date_parts[0][6:8]}"
                    if date_parts else bn
                )
                out[day_label] = _read_nc_array(
                    nc_path, grid, template_factory(), resampling,
                )
            except Exception as exc:  # noqa: BLE001 - reported per member
                logger.warning(
                    "Could not process %s: %s", os.path.basename(nc_path), exc,
                )
    return out


def _build_manifest(
    pollutants: list[str],
    resolution: str,
    years: Optional[list[int]],
    months: list[int],
    cache_dir: str,
    lang: str,
    verbose: bool,
) -> list[dict]:
    """One entry per file to download, with per-pollutant year resolution."""
    manifest: list[dict] = []
    for p in pollutants:
        p_res = "annual" if (p in _ANNUAL_ONLY or (resolution == "daily" and p != "pm25")) else resolution
        if p_res != resolution and verbose:
            print(lcz_msg("grid_ghap_fallback_annual", lang, pollutant=p.upper()))

        avail_years = list(_AVAIL_YEARS[p][p_res])
        if years is None:
            req_years = avail_years
        else:
            req_years = [int(y) for y in years]
            bad_y = [y for y in req_years if y not in avail_years]
            if bad_y:
                logger.warning(lcz_msg(
                    "grid_ghap_years_unavail", lang, pollutant=p.upper(),
                    bad_years=", ".join(map(str, bad_y)),
                    avail=f"{avail_years[0]}-{avail_years[-1]}",
                ))
                req_years = [y for y in req_years if y in avail_years]

        for yr in req_years:
            if p_res == "daily":
                for mo in months:
                    filename, url = _file_info(p, "daily", yr, mo)
                    manifest.append(dict(
                        pollutant=p, resolution="daily", year=yr, month=mo,
                        filename=filename, url=url,
                        cache_nc=os.path.join(cache_dir, p, "daily", filename),
                    ))
            elif p_res == "monthly":
                for mo in months:
                    filename, url = _file_info(p, "monthly", yr, mo)
                    manifest.append(dict(
                        pollutant=p, resolution="monthly", year=yr, month=mo,
                        filename=filename, url=url,
                        cache_nc=os.path.join(cache_dir, p, filename),
                    ))
            else:
                filename, url = _file_info(p, "annual", yr)
                manifest.append(dict(
                    pollutant=p, resolution="annual", year=yr, month=None,
                    filename=filename, url=url,
                    cache_nc=os.path.join(cache_dir, p, filename),
                ))
    return manifest


def lcz_grid_pollution_ghap(
    x: Union[str, Path],
    pollutants: Union[str, list[str]] = "pm25",
    resolution: str = "monthly",
    years: Optional[list[int]] = None,
    months: Optional[list[int]] = None,
    resampling: str = "average",
    isave: bool = False,
    cache: bool = True,
    cache_dir: str = os.path.join(DEFAULT_GRID_CACHE_DIR, "ghap"),
    lang: str = "en",
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> LCZGridResult:
    """Download GHAP pollution rasters cropped to an LCZ map's footprint.

    Parameters
    ----------
    x : str or Path
        Path to an existing LCZ map GeoTIFF (e.g. from ``lcz_get_map``).
    pollutants : str or list[str]
        "pm25", "o3", "co", or "all". Default "pm25". ("no2" not yet public.)
    resolution : str
        "daily" (PM2.5 only), "monthly" (PM2.5 only), or "annual" (all).
        O3/CO silently fall back to "annual".
    years : list[int], optional
        Defaults to all available years per pollutant/resolution.
    months : list[int], optional
        Months 1-12, used for daily/monthly. Default all 12.
    resampling : str
        Resampling for regridding onto the LCZ grid: "average" (default),
        "bilinear", or "nearest".
    isave : bool
        Also copy the stack to ``LCZ4r_output/lcz_grid_ghap.tif``.
    cache : bool
        Reuse a previously built stack for the same (pollutants, resolution,
        years, months, LCZ map bbox), and previously downloaded raw files.
    cache_dir : str
        Root cache directory. Default ``~/.lcz4r_cache/grid/ghap``.
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
    """
    if pollutants == "all":
        pollutants = list(_VALID_POLLUTANTS)
    elif isinstance(pollutants, str):
        pollutants = [pollutants]
    else:
        pollutants = [str(p).lower() for p in pollutants]

    pollutants = [str(p).lower() for p in pollutants]
    if "no2" in pollutants:
        logger.warning(lcz_msg("grid_ghap_no2_unavailable", lang))
        pollutants = [p for p in pollutants if p != "no2"]
    bad_p = [p for p in pollutants if p not in _VALID_POLLUTANTS]
    if bad_p:
        raise ValueError(lcz_msg(
            "grid_ghap_invalid_pollutants", lang, bad=", ".join(bad_p),
        ))
    if not pollutants:
        raise ValueError(lcz_msg("grid_ghap_no_pollutants", lang))
    # De-duplicate while keeping the caller's order.
    pollutants = list(dict.fromkeys(pollutants))

    if resolution not in _VALID_RESOLUTIONS:
        raise ValueError(lcz_msg("grid_ghap_invalid_resolution", lang))
    if resampling not in _VALID_RESAMPLING:
        raise ValueError(lcz_msg(
            "grid_raster_invalid_resampling", lang, valid=", ".join(_VALID_RESAMPLING),
        ))

    months = sorted(set(normalize_int_list(months, "months", default=range(1, 13))))
    if any(m < 1 or m > 12 for m in months):
        raise ValueError(lcz_msg("grid_ghap_invalid_months", lang))

    if years is not None:
        years = sorted(set(normalize_int_list(years, "years")))

    grid = load_target_grid(x)
    cache_dir = os.path.expanduser(cache_dir)

    # Resolve years *before* computing the cache key so an explicit year list
    # and the implicit "all available years" never share a cache entry.
    manifest = _build_manifest(pollutants, resolution, years, months, cache_dir, lang, verbose)
    if not manifest:
        raise ValueError(lcz_msg("grid_ghap_no_data_to_download", lang))

    key = cache_key(
        "ghap",
        tuple(sorted(m["filename"] for m in manifest)),
        grid.bbox_wgs84,
        resampling,
    )
    stack_path = os.path.join(cache_dir, "stack", f"ghap_stack_{key}.tif")

    array: Optional[np.ndarray] = None
    bands: list[str] = []
    if cache:
        cached = try_read_grid_stack(stack_path)
        if cached is not None:
            if verbose:
                print(lcz_msg("grid_raster_cache_hit", lang, name="GHAP"))
            array, bands = cached

    if array is None:
        if verbose:
            print(lcz_msg("grid_ghap_download_start", lang, n_files=len(manifest)))

        downloads = {
            m["filename"]: (m["url"], m["cache_nc"])
            for m in manifest
        }
        workers = resolve_max_workers(max_workers, len(downloads))
        available = download_many(
            downloads, use_cache=cache, verbose=verbose,
            max_workers=workers, progress_desc="GHAP",
        )

        template = make_template_factory(grid)
        layers: dict[str, np.ndarray] = {}
        for m in manifest:
            if not available.get(m["filename"], False):
                logger.warning(lcz_msg(
                    "grid_ghap_skip_missing", lang, filename=m["filename"],
                ))
                continue
            try:
                if m["cache_nc"].endswith(".zip"):
                    daily = _extract_daily_zip(
                        m["cache_nc"], grid, template, resampling, lang,
                    )
                    for d, arr in daily.items():
                        layers[f"{m['pollutant']}_{d}"] = arr
                else:
                    arr = _read_nc_array(m["cache_nc"], grid, template(), resampling)
                    label = f"{m['year']:04d}-{(m['month'] or 1):02d}-01"
                    layers[f"{m['pollutant']}_{label}"] = arr
            except Exception as exc:  # noqa: BLE001 - reported per file
                logger.warning(
                    lcz_msg("grid_ghap_extract_warn", lang, filename=m["filename"])
                    + f" ({exc})"
                )
                continue

        if not layers:
            raise RuntimeError(lcz_msg("grid_ghap_no_data", lang))

        bands = sorted(layers)
        array = np.stack([layers[b] for b in bands]).astype(np.float32)
        if cache:
            write_grid_stack(
                stack_path, array, bands, grid,
                variable=",".join(pollutants), units="ug_m3",
            )

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_grid_ghap.tif")
        write_grid_stack(
            out_path, array, bands, grid,
            variable=",".join(pollutants), units="ug_m3",
        )
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    if verbose:
        print(lcz_msg("grid_raster_done", lang, name="GHAP", n_bands=len(bands)))

    return LCZGridResult(
        path=stack_path if cache else None,
        array=array, bands=bands, variables=list(pollutants), units="ug_m3",
    )
