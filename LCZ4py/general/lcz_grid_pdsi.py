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

Year files are downloaded in parallel; each NetCDF is opened and clipped to
the LCZ bbox once, then only the requested monthly slices are reprojected.
The single-file NOAA PSL product is likewise opened once for all years.
"""

from __future__ import annotations

import logging
import os
from datetime import date
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
    clip_dataarray_block,
    load_target_grid,
    make_template_factory,
    reproject_to_grid,
    try_read_grid_stack,
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
_NOAA_PSL_FILENAME = "pdsi.mon.mean.selfcalibrated.nc"


def _extract_periods(
    nc_path: str,
    source: str,
    years: list[int],
    months: list[int],
    grid,
    template_factory,
    resampling: str,
) -> dict[str, np.ndarray]:
    """Extract the requested months of one PDSI NetCDF as ``{ISO date: array}``."""
    if not HAS_XARRAY:
        raise ImportError(
            "Reading PDSI NetCDF files requires 'xarray'. "
            "Install it with: pip install xarray"
        )

    import pandas as pd

    ds = xr.open_dataset(nc_path)
    try:
        var = _SOURCE_VAR[source]
        if var not in ds.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in {os.path.basename(nc_path)} "
                f"(available: {', '.join(map(str, ds.data_vars)) or 'none'})."
            )
        da = ds[var]
        if "time" not in da.dims:
            raise ValueError(
                f"Expected a 'time' dimension in {os.path.basename(nc_path)}, "
                f"got {tuple(da.dims)}."
            )

        try:
            mask = da["time.month"].isin(months)
            if source == "noaa_psl":
                mask = mask & da["time.year"].isin(years)
            selected = da.sel(time=mask)
        except KeyError as exc:  # pragma: no cover - non-datetime time axis
            raise ValueError(
                f"Could not index the time axis of {os.path.basename(nc_path)}: {exc}"
            ) from exc

        if int(selected.sizes.get("time", 0)) == 0:
            return {}

        block = clip_dataarray_block(selected, grid)
        if block is None:
            return {}

        out: dict[str, np.ndarray] = {}
        for i in range(int(block.sizes["time"])):
            stamp = block["time"].values[i]
            layer = block.isel(time=i, drop=True)
            arr = reproject_to_grid(layer, grid, template_factory(), resampling)
            label = pd.Timestamp(stamp).replace(day=1).date().isoformat()
            out[label] = arr
        return out
    finally:
        ds.close()


def _extract_year(
    nc_path: str,
    source: str,
    year: int,
    months: list[int],
    grid,
    template,
    resampling: str,
) -> dict[str, np.ndarray]:
    """Backward-compatible single-year wrapper around :func:`_extract_periods`."""
    return _extract_periods(
        nc_path, source, [year], months, grid, lambda: template, resampling,
    )


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
    max_workers: Optional[int] = None,
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
    max_workers : int, optional
        Threads used for parallel downloads. Default
        ``min(8, os.cpu_count() or 4)``.

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
        raise ValueError(lcz_msg(
            "grid_pdsi_invalid_source", lang, valid=", ".join(_VALID_SOURCES),
        ))
    if resampling not in _VALID_RESAMPLING:
        raise ValueError(lcz_msg(
            "grid_raster_invalid_resampling", lang, valid=", ".join(_VALID_RESAMPLING),
        ))

    min_year, max_year = _SOURCE_YEAR_RANGE[source]
    current_year = date.today().year

    explicit_years = years is not None
    if not explicit_years:
        years = [current_year - 2, current_year - 1]
        if verbose:
            print(lcz_msg(
                "grid_pdsi_default_years", lang,
                years=f"{years[0]}-{years[-1]}",
            ))

    years = sorted(set(normalize_int_list(years, "years")))
    if not years:
        raise ValueError(lcz_msg(
            "grid_pdsi_invalid_years_range", lang, bad="<empty>",
        ))

    if not explicit_years:
        clamped = sorted({min(max(y, min_year), max_year) for y in years})
        if clamped != years:
            logger.warning(
                "Clamping default year(s) to the %s record (%d-%d).",
                source, min_year, max_year,
            )
        years = clamped

    bad = [y for y in years if y < min_year or y > max_year]
    if bad:
        raise ValueError(lcz_msg(
            "grid_pdsi_invalid_years_range", lang, bad=", ".join(map(str, bad)),
        ))

    months = sorted(set(normalize_int_list(months, "months", default=range(1, 13))))
    if any(m < 1 or m > 12 for m in months):
        raise ValueError(lcz_msg("grid_pdsi_invalid_months", lang))

    grid = load_target_grid(x)
    cache_dir = os.path.expanduser(cache_dir)

    key = cache_key("pdsi", source, years, months, grid.bbox_wgs84, resampling)
    stack_path = os.path.join(cache_dir, "stack", f"pdsi_stack_{key}.tif")

    array: Optional[np.ndarray] = None
    bands: list[str] = []
    if cache:
        cached = try_read_grid_stack(stack_path)
        if cached is not None:
            if verbose:
                print(lcz_msg("grid_raster_cache_hit", lang, name="PDSI"))
            array, bands = cached

    if array is None:
        if source == "terraclimate":
            manifest = [
                dict(
                    year=yr,
                    filename=f"TerraClimate_PDSI_{yr:04d}.nc",
                    url=_TERRACLIMATE_URL.format(year=yr),
                    cache_path=os.path.join(cache_dir, source, f"TerraClimate_PDSI_{yr:04d}.nc"),
                )
                for yr in years
            ]
        else:
            manifest = [
                dict(
                    year=years[0],
                    filename=_NOAA_PSL_FILENAME,
                    url=_NOAA_PSL_URL,
                    cache_path=os.path.join(cache_dir, source, _NOAA_PSL_FILENAME),
                )
            ]

        if verbose:
            print(lcz_msg(
                "grid_pdsi_download_start", lang,
                n_files=len(manifest), source=source,
            ))

        unique_files: dict[str, tuple[str, str]] = {}
        for m in manifest:
            unique_files.setdefault(m["filename"], (m["url"], m["cache_path"]))
        workers = resolve_max_workers(max_workers, len(unique_files))
        download_many(
            unique_files, use_cache=cache, verbose=verbose,
            max_workers=workers, progress_desc="PDSI",
        )

        template = make_template_factory(grid)
        layers: dict[str, np.ndarray] = {}
        # terraclimate: one manifest entry per year. noaa_psl: a single file
        # covering all years, queried once for every requested year.
        if source == "terraclimate":
            year_entries = [(m, [m["year"]]) for m in manifest]
        else:
            year_entries = [(manifest[0], years)]

        for m, file_years in year_entries:
            if not os.path.exists(m["cache_path"]) or os.path.getsize(m["cache_path"]) == 0:
                logger.warning(lcz_msg(
                    "grid_pdsi_skip_missing", lang, filename=m["filename"],
                ))
                continue
            try:
                periods = _extract_periods(
                    m["cache_path"], source, file_years, months,
                    grid, template, resampling,
                )
            except Exception as exc:  # noqa: BLE001 - reported per file
                logger.warning(
                    lcz_msg("grid_pdsi_extract_warn", lang, filename=m["filename"])
                    + f" ({exc})"
                )
                continue
            layers.update(periods)

        if not layers:
            raise RuntimeError(lcz_msg("grid_pdsi_no_data", lang))

        bands = sorted(layers)
        array = np.stack([layers[b] for b in bands]).astype(np.float32)
        if cache:
            write_grid_stack(
                stack_path, array, bands, grid,
                variable="pdsi", units="index",
            )

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_grid_pdsi.tif")
        write_grid_stack(
            out_path, array, bands, grid,
            variable="pdsi", units="index",
        )
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    if verbose:
        print(lcz_msg("grid_raster_done", lang, name="PDSI", n_bands=len(bands)))

    return LCZGridResult(
        path=stack_path if cache else None,
        array=array, bands=bands, variables=["pdsi"], units="index",
    )
