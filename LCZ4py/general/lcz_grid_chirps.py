"""Download CHIRPS v2.0 precipitation data, cropped to an LCZ map's footprint.

CHIRPS (Funk et al. 2015, doi:10.1038/sdata.2015.66) is a quasi-global 0.05°
(~5 km) daily rainfall dataset from UCSB CHC, 1981-present. No authentication
required. Source: https://data.chc.ucsb.edu/products/CHIRPS-2.0/

Output is a multi-band GeoTIFF (one band per period), float32 with NaN
nodata, pixel-aligned to the input LCZ map's grid — the same
``LCZGridResult`` convention used by ``lcz_get_lst``/``lcz_get_planetary_computer``.

Downloads and per-file crops run in a bounded thread pool
(``max_workers``), while band order stays deterministic (chronological).
"""

from __future__ import annotations

import calendar
import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional, Union

import numpy as np

from LCZ4py._internal._lcz_grid_base import (
    download_many,
    normalize_int_list,
    parallel_ordered,
    resolve_max_workers,
)
from LCZ4py._internal._lcz_grid_raster_base import (
    DEFAULT_GRID_CACHE_DIR,
    LCZGridResult,
    cache_key,
    crop_reproject_to_grid,
    load_target_grid,
    make_template_factory,
    try_read_grid_stack,
    write_grid_stack,
)
from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

OUTPUT_DIR = "LCZ4r_output"
_CHC_BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"
_VALID_RESOLUTIONS = ("monthly", "daily", "annual")
_VALID_RESAMPLING = ("average", "bilinear", "nearest")
_MAX_YEAR_ANNUAL = 2024
_FIRST_YEAR = 1981

#: Warn (but keep going) above this many daily requests — these are one file
#: per day, so a multi-year daily request is thousands of downloads.
_DAILY_MANIFEST_WARN = 400


def _file_info(resolution: str, year: int, month: Optional[int], day: Optional[int]) -> tuple[str, str]:
    """Return ``(filename, url)`` for one CHIRPS product."""
    if resolution == "annual":
        filename = f"chirps-v2.0.{year:04d}.tif"
        url = f"{_CHC_BASE}/global_annual/tifs/{filename}"
    elif resolution == "monthly":
        filename = f"chirps-v2.0.{year:04d}.{month:02d}.tif.gz"
        url = f"{_CHC_BASE}/global_monthly/tifs/{filename}"
    else:
        filename = f"chirps-v2.0.{year:04d}.{month:02d}.{day:02d}.tif.gz"
        url = f"{_CHC_BASE}/global_daily/tifs/p05/{year:04d}/{filename}"
    return filename, url


def _vsi_path(path: str) -> str:
    return f"/vsigzip/{path}" if path.endswith(".gz") else path


def _build_manifest(
    resolution: str,
    years: list[int],
    months: list[int],
) -> list[dict]:
    """One entry per period, sorted chronologically."""
    manifest: list[dict] = []
    if resolution == "annual":
        for yr in years:
            filename, url = _file_info("annual", yr, None, None)
            manifest.append(dict(filename=filename, url=url, date_val=date(yr, 1, 1)))
    elif resolution == "monthly":
        for yr in years:
            for mo in months:
                filename, url = _file_info("monthly", yr, mo, None)
                manifest.append(dict(filename=filename, url=url, date_val=date(yr, mo, 1)))
    else:
        for yr in years:
            for mo in months:
                n_days = calendar.monthrange(yr, mo)[1]
                for dy in range(1, n_days + 1):
                    filename, url = _file_info("daily", yr, mo, dy)
                    manifest.append(dict(filename=filename, url=url, date_val=date(yr, mo, dy)))
    manifest.sort(key=lambda m: m["date_val"])
    return manifest


def lcz_grid_chirps(
    x: Union[str, Path],
    resolution: str = "monthly",
    years: Optional[list[int]] = None,
    months: Optional[list[int]] = None,
    resampling: str = "average",
    isave: bool = False,
    cache: bool = True,
    cache_dir: str = os.path.join(DEFAULT_GRID_CACHE_DIR, "chirps"),
    lang: str = "en",
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> LCZGridResult:
    """Download CHIRPS precipitation cropped to an LCZ map's footprint.

    Parameters
    ----------
    x : str or Path
        Path to an existing LCZ map GeoTIFF (e.g. from ``lcz_get_map``).
    resolution : str
        "monthly" (default, mm/month), "daily" (mm/day), or "annual" (mm/year).
    years : list[int], optional
        Years to download (1981-present, or 1981-2024 for annual). Defaults
        to the last two complete years.
    months : list[int], optional
        Months 1-12 to include (ignored for annual). Default all 12.
    resampling : str
        Resampling for regridding onto the LCZ grid: "average" (default,
        best for a coarse-to-fine rainfall field), "bilinear", or "nearest".
    isave : bool
        Also copy the stack to ``LCZ4r_output/lcz_grid_chirps.tif``.
    cache : bool
        Reuse a previously built stack for the same (resolution, years,
        months, LCZ map bbox), and previously downloaded raw rasters.
    cache_dir : str
        Root cache directory. Default ``~/.lcz4r_cache/grid/chirps``.
    lang : str
        Message language. Default "en".
    verbose : bool
        Print progress. Default True.
    max_workers : int, optional
        Threads used for parallel downloads/extraction. Default
        ``min(8, os.cpu_count() or 4)``.

    Returns
    -------
    LCZGridResult
        ``array`` is (n_periods, H, W) float32 mm, NaN outside the LCZ map's
        valid pixels; ``bands[i]`` gives the ISO date (first-of-period) for
        band ``i``.

    Examples
    --------
    >>> from LCZ4py import lcz_get_map, lcz_grid_chirps
    >>> grid_result = lcz_get_map("Rio de Janeiro", isave=True)
    >>> res = lcz_grid_chirps(grid_result.path, years=[2020], months=[1, 2])
    >>> res.array.shape  # (2, H, W)
    """
    if resolution not in _VALID_RESOLUTIONS:
        raise ValueError(lcz_msg(
            "grid_chirps_invalid_resolution", lang,
            bad=resolution, valid=", ".join(_VALID_RESOLUTIONS),
        ))
    if resampling not in _VALID_RESAMPLING:
        raise ValueError(lcz_msg(
            "grid_raster_invalid_resampling", lang, valid=", ".join(_VALID_RESAMPLING),
        ))

    current_year = date.today().year
    max_year = _MAX_YEAR_ANNUAL if resolution == "annual" else current_year

    explicit_years = years is not None
    if not explicit_years:
        years = [current_year - 2, current_year - 1]
        if verbose:
            print(lcz_msg(
                "grid_chirps_default_years", lang,
                years=f"{years[0]}-{years[-1]}",
            ))

    years = sorted(set(normalize_int_list(years, "years")))
    if not years:
        raise ValueError(lcz_msg("grid_chirps_no_data_params", lang))

    if not explicit_years:
        # Defaults may point past the published range (CHIRPS annual lags a
        # year or two); clamp them instead of hard-failing a default call.
        clamped = sorted({min(max(y, _FIRST_YEAR), max_year) for y in years})
        if clamped != years:
            logger.warning(
                "Clamping default year(s) %s to the published range %d-%d.",
                sorted(set(years) - set(clamped)), _FIRST_YEAR, max_year,
            )
        years = clamped

    bad = [y for y in years if y < _FIRST_YEAR or y > max_year]
    if bad:
        raise ValueError(lcz_msg(
            "grid_chirps_invalid_years_range", lang,
            bad=", ".join(map(str, bad)), max_year=max_year,
        ))

    months = sorted(set(normalize_int_list(months, "months", default=range(1, 13))))
    if any(m < 1 or m > 12 for m in months):
        raise ValueError(lcz_msg("grid_chirps_invalid_months", lang))

    grid = load_target_grid(x)
    cache_dir = os.path.expanduser(cache_dir)

    # ── Build manifest ──────────────────────────────────────────────────────
    manifest = _build_manifest(resolution, years, months)
    if not manifest:
        raise ValueError(lcz_msg("grid_chirps_no_data_params", lang))
    if verbose and len(manifest) > _DAILY_MANIFEST_WARN:
        logger.warning(
            "%d CHIRPS files requested (%s); this may take a while.",
            len(manifest), resolution,
        )

    key = cache_key("chirps", resolution, years, months, grid.bbox_wgs84, resampling)
    stack_path = os.path.join(cache_dir, "stack", f"chirps_stack_{key}.tif")

    array: Optional[np.ndarray] = None
    bands: list[str] = []
    if cache:
        cached = try_read_grid_stack(stack_path)
        if cached is not None:
            if verbose:
                print(lcz_msg("grid_raster_cache_hit", lang, name="CHIRPS"))
            array, bands = cached

    if array is None:
        if verbose:
            print(lcz_msg("grid_chirps_download_start", lang, n_files=len(manifest)))

        workers = resolve_max_workers(max_workers, len(manifest))
        raw_paths = {
            m["filename"]: (
                m["url"],
                os.path.join(cache_dir, resolution, m["filename"]),
            )
            for m in manifest
        }
        available = download_many(
            raw_paths,
            use_cache=cache,
            verbose=verbose,
            max_workers=workers,
            progress_desc="CHIRPS",
        )

        template = make_template_factory(grid)

        def _extract(entry: dict) -> Optional[tuple[str, np.ndarray]]:
            if not available.get(entry["filename"], False):
                logger.warning(lcz_msg(
                    "grid_chirps_skip_missing", lang, filename=entry["filename"],
                ))
                return None
            raw_path = raw_paths[entry["filename"]][1]
            if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
                logger.warning(lcz_msg(
                    "grid_chirps_skip_missing", lang, filename=entry["filename"],
                ))
                return None
            try:
                arr = crop_reproject_to_grid(
                    _vsi_path(raw_path), grid, template(), resampling,
                    # CHIRPS encodes missing data as -9999 without always
                    # declaring it, which would poison "average" resampling.
                    nodata_below=-999.0,
                )
            except Exception as exc:  # noqa: BLE001 - reported per file
                logger.warning(
                    lcz_msg("grid_chirps_extract_warn", lang, filename=entry["filename"])
                    + f" ({exc})"
                )
                return None
            if not np.isfinite(arr).any():
                logger.warning(
                    "No valid pixels for %s inside the LCZ footprint.",
                    entry["filename"],
                )
            return entry["date_val"].isoformat(), arr

        extracted = parallel_ordered(
            _extract, manifest, max_workers=workers, default=None, label="chirps",
        )

        layer_pairs = [item for item in extracted if item is not None]
        if not layer_pairs:
            raise RuntimeError(lcz_msg("grid_chirps_no_data", lang))

        bands = [label for label, _ in layer_pairs]
        array = np.stack([arr for _, arr in layer_pairs]).astype(np.float32)
        if cache:
            write_grid_stack(
                stack_path, array, bands, grid,
                variable="rainfall_chirps_mm", units="mm",
            )

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_grid_chirps.tif")
        write_grid_stack(
            out_path, array, bands, grid,
            variable="rainfall_chirps_mm", units="mm",
        )
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    if verbose:
        print(lcz_msg("grid_raster_done", lang, name="CHIRPS", n_bands=len(bands)))

    return LCZGridResult(
        path=stack_path if cache else None,
        array=array, bands=bands,
        variables=["rainfall_chirps_mm"], units="mm",
    )
