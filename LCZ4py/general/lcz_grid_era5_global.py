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

``cdsapi`` is an optional dependency: importing this module never fails
without it, and calling the function raises a clear ``ImportError``.
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:  # pragma: no cover - depends on the environment
    xr = None  # type: ignore[assignment]
    HAS_XARRAY = False

try:
    import cdsapi

    HAS_CDSAPI = True
except ImportError:  # pragma: no cover - optional dependency
    cdsapi = None  # type: ignore[assignment]
    HAS_CDSAPI = False

from LCZ4py._internal._lcz_grid_base import (
    DEFAULT_BACKOFF,
    DEFAULT_MAX_RETRIES,
    normalize_int_list,
    parallel_ordered,
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
_VALID_RESAMPLING = ("average", "bilinear", "nearest")
_CDS_DATASET = "reanalysis-era5-land-monthly-means"
_BBOX_MARGIN_DEG = 0.25  # > the ~0.1deg native grid, so edge pixels have neighbors for reproject_match
_FIRST_YEAR = 1950

#: The new CDS rejects large bursts of concurrent requests; keep this modest
#: even when the caller asks for more workers.
_CDS_MAX_CONCURRENCY = 4

# alias -> (CDS variable name, short name inside the returned NetCDF, unit-conversion fn)
_VAR_MAP: dict[str, tuple[str, str, Callable[[np.ndarray], np.ndarray]]] = {
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
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    client = _cds_client(cds_key, cds_url)
    client.retrieve(_CDS_DATASET, request, target_path)


def _area_from_grid(grid) -> list[float]:
    """CDS ``area`` is [north, west, south, east], with a small margin so
    edge pixels have real neighbors for average/bilinear reproject_match."""
    minx, miny, maxx, maxy = grid.bbox_wgs84
    if minx > maxx:
        # Antimeridian-crossing extent: CDS cannot express it, request the
        # full longitude range instead of silently clipping to nothing.
        logger.warning("LCZ footprint crosses the antimeridian; requesting the full longitude range.")
        minx, maxx = -180.0, 180.0
    m = _BBOX_MARGIN_DEG
    return [min(maxy + m, 90.0), max(minx - m, -180.0), max(miny - m, -90.0), min(maxx + m, 180.0)]


_ZIP_MAGIC = b"PK\x03\x04"


def _msg_month(key: str, lang: str, year: int, month: int) -> str:
    """Render a year/month message.

    Several i18n templates carry a ``{month:02d}`` format spec, which the
    package-wide placeholder regex (``{name}``) does not expand; substitute it
    here so logs show ``2020-01`` instead of ``2020-{month:02d}``.
    """
    text = lcz_msg(key, lang, year=year, month=month)
    return text.replace("{month:02d}", f"{month:02d}")


def _unwrap_cds_payload(path: str) -> str:
    """CDS may deliver a ZIP even when ``format="netcdf"`` was requested.

    The current Copernicus Data Stores service returns
    ``<hash>.zip`` containing ``data_stream-*.nc``; without unwrapping, every
    ``xr.open_dataset`` fails with an opaque "no matching IO backend" error.
    Returns ``path`` unchanged when the payload is already a NetCDF.
    """
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head != _ZIP_MAGIC:
        return path

    with zipfile.ZipFile(path) as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        nc_names = [n for n in names if n.lower().endswith((".nc", ".nc4"))] or names
        if not nc_names:
            raise ValueError(
                f"The CDS response for {os.path.basename(path)} is a ZIP with no data member."
            )
        member = sorted(nc_names)[0]
        tmp_out = path + ".unzipped"
        with zf.open(member) as src, open(tmp_out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    os.replace(tmp_out, path)
    logger.info("Unwrapped CDS ZIP payload -> %s", os.path.basename(member))
    return path


def _read_month(
    nc_path: str,
    grid,
    template_factory,
    resampling: str,
    aliases: list[str],
) -> dict[str, np.ndarray]:
    """Extract every requested variable from one monthly CDS NetCDF."""
    if not HAS_XARRAY:
        raise ImportError(
            "Reading ERA5-Land NetCDF files requires 'xarray'. "
            "Install it with: pip install xarray"
        )

    out: dict[str, np.ndarray] = {}
    ds = xr.open_dataset(nc_path)
    try:
        for alias in aliases:
            _, short_name, conv_fn = _VAR_MAP[alias]
            if short_name not in ds.data_vars:
                logger.warning(
                    "Variable %r missing from %s (available: %s).",
                    short_name, os.path.basename(nc_path),
                    ", ".join(map(str, ds.data_vars)) or "none",
                )
                continue
            da = ds[short_name]
            for dim in ("time", "valid_time"):
                if dim in da.dims:
                    da = da.isel({dim: 0}, drop=True)
            try:
                arr = crop_reproject_dataarray(da, grid, template_factory(), resampling)
            except Exception as exc:  # noqa: BLE001 - reported per variable
                logger.warning("Could not regrid %r: %s", short_name, exc)
                continue
            out[alias] = np.asarray(conv_fn(arr), dtype="float32")
        return out
    finally:
        ds.close()


def _retrieve_one(
    entry: dict,
    cds_vars: list[str],
    area: list[float],
    cds_key: Optional[str],
    cds_url: Optional[str],
    lang: str,
    max_retries: int,
    backoff: float,
) -> bool:
    """Download one monthly file atomically, retrying transient CDS errors."""
    import time

    final_path = entry["cache_path"]
    tmp_path = final_path + ".part"
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": cds_vars,
        "year": str(entry["year"]),
        "month": f"{entry['month']:02d}",
        "time": "00:00",
        "area": area,
        "format": "netcdf",
    }

    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            _cds_retrieve(request, tmp_path, cds_key, cds_url)
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                raise OSError("CDS returned an empty file")
            _unwrap_cds_payload(tmp_path)
            os.replace(tmp_path, final_path)
            return True
        except Exception as exc:  # noqa: BLE001 - classified below
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:  # pragma: no cover - best effort
                    pass
            if attempt >= attempts:
                logger.warning(
                    _msg_month(
                        "grid_era5_global_download_warn", lang,
                        entry["year"], entry["month"],
                    ) + f" ({exc})"
                )
                return False
            delay = backoff * (2 ** (attempt - 1))
            logger.debug(
                "Retrying CDS request %s-%02d in %.1fs: %s",
                entry["year"], entry["month"], delay, exc,
            )
            time.sleep(delay)
    return False  # pragma: no cover - loop always returns


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
    max_workers: Optional[int] = None,
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
    max_workers : int, optional
        Concurrent CDS requests. Default ``min(8, os.cpu_count() or 4)``,
        capped at 4 to stay within CDS queue limits.

    Returns
    -------
    LCZGridResult
        ``array`` is (n_bands, H, W) float32, NaN outside the LCZ map's
        valid pixels; ``bands[i]`` is ``"{variable}_{year}-{month}-01"``.

    Notes
    -----
    Unit conversions applied: temperature/skin temperature K->C, precipitation
    m->mm, pressure Pa->hPa. Wind components left m/s. Monthly grain only.
    """
    if not HAS_CDSAPI:
        raise ImportError(lcz_msg("grid_era5_global_no_cdsapi", lang))
    if not HAS_XARRAY:
        raise ImportError(
            "lcz_grid_era5_global requires 'xarray'. Install it with: pip install xarray"
        )

    years = sorted(set(normalize_int_list(years, "years")))
    if not years:
        raise ValueError(lcz_msg("grid_era5_missing_years", lang))
    current_year = date.today().year
    bad_years = [y for y in years if y < _FIRST_YEAR or y > current_year]
    if bad_years:
        raise ValueError(lcz_msg(
            "grid_era5_invalid_years_range", lang,
            bad=", ".join(map(str, bad_years)),
        ))

    months = sorted(set(normalize_int_list(months, "months", default=range(1, 13))))
    if any(m < 1 or m > 12 for m in months):
        raise ValueError(lcz_msg("grid_era5_invalid_months", lang))

    if vars == "all":
        vars = list(_VAR_MAP.keys())
    elif isinstance(vars, str):
        vars = [vars]
    else:
        vars = list(vars)
    vars = list(dict.fromkeys(vars))  # keep order, drop duplicates
    bad_vars = [v for v in vars if v not in _VAR_MAP]
    if bad_vars:
        raise ValueError(lcz_msg(
            "grid_era5_invalid_vars", lang,
            bad=", ".join(bad_vars), valid=", ".join(_VAR_MAP),
        ))
    if not vars:
        raise ValueError(lcz_msg("grid_era5_invalid_vars", lang, bad="", valid=", ".join(_VAR_MAP)))

    if resampling not in _VALID_RESAMPLING:
        raise ValueError(lcz_msg(
            "grid_raster_invalid_resampling", lang, valid=", ".join(_VALID_RESAMPLING),
        ))

    cds_key = cds_key or os.environ.get("CDSAPI_KEY", "")
    cds_url = cds_url or os.environ.get("CDSAPI_URL", "")
    has_rc = os.path.exists(os.path.expanduser("~/.cdsapirc"))
    if not has_rc and not cds_key:
        raise ValueError(lcz_msg("grid_era5_global_no_auth", lang))

    grid = load_target_grid(x)
    cache_dir = os.path.expanduser(cache_dir)

    key = cache_key("era5_global", years, months, sorted(vars), grid.bbox_wgs84, resampling)
    stack_path = os.path.join(cache_dir, "stack", f"era5_global_stack_{key}.tif")

    array: Optional[np.ndarray] = None
    bands: list[str] = []
    if cache:
        cached = try_read_grid_stack(stack_path)
        if cached is not None:
            if verbose:
                print(lcz_msg("grid_raster_cache_hit", lang, name="ERA5-Land (global/CDS)"))
            array, bands = cached

    if array is None:
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

        pending = []
        for m in manifest:
            if cache and os.path.exists(m["cache_path"]) and os.path.getsize(m["cache_path"]) > 0:
                if verbose:
                    print(lcz_msg("grid_raster_cache_hit", lang, name=f"{m['year']}-{m['month']:02d}"))
                continue
            pending.append(m)

        if pending:
            workers = min(
                _CDS_MAX_CONCURRENCY,
                resolve_max_workers(max_workers, len(pending)),
            )
            parallel_ordered(
                lambda entry: _retrieve_one(
                    entry, cds_vars, area, cds_key, cds_url, lang,
                    DEFAULT_MAX_RETRIES, DEFAULT_BACKOFF,
                ),
                pending,
                max_workers=workers,
                default=False,
                label="cds",
            )

        template = make_template_factory(grid)
        layers: dict[str, np.ndarray] = {}
        # CDS NetCDF reads are serialized (HDF5 is not thread-safe); each file
        # is opened only once for all requested variables.
        for m in manifest:
            if not os.path.exists(m["cache_path"]) or os.path.getsize(m["cache_path"]) == 0:
                logger.warning(_msg_month(
                    "grid_era5_global_skip_missing", lang,
                    m["year"], m["month"],
                ))
                continue
            try:
                # Heal raw files cached as ZIPs by older versions of this
                # function (the CDS payload format changed upstream).
                _unwrap_cds_payload(m["cache_path"])
            except Exception as exc:  # noqa: BLE001 - reported with the file
                logger.warning(
                    _msg_month(
                        "grid_era5_global_extract_warn", lang,
                        m["year"], m["month"],
                    ) + f" ({exc})"
                )
                continue
            try:
                derived = _read_month(
                    m["cache_path"], grid, template, resampling, vars,
                )
            except Exception as exc:  # noqa: BLE001 - reported per file
                logger.warning(
                    _msg_month(
                        "grid_era5_global_extract_warn", lang,
                        m["year"], m["month"],
                    ) + f" ({exc})"
                )
                continue
            for alias, arr in derived.items():
                layers[f"{alias}_{m['year']:04d}-{m['month']:02d}-01"] = arr

        if not layers:
            raise RuntimeError(lcz_msg("grid_era5_no_data", lang))

        bands = sorted(layers)
        array = np.stack([layers[b] for b in bands]).astype(np.float32)
        if cache:
            write_grid_stack(
                stack_path, array, bands, grid,
                variable=",".join(vars), units="mixed",
            )

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_grid_era5_global.tif")
        write_grid_stack(
            out_path, array, bands, grid,
            variable=",".join(vars), units="mixed",
        )
        if verbose:
            logger.info("Saved: %s", os.path.abspath(out_path))

    if verbose:
        print(lcz_msg("grid_raster_done", lang, name="ERA5-Land (global/CDS)", n_bands=len(bands)))

    return LCZGridResult(
        path=stack_path if cache else None,
        array=array, bands=bands, variables=list(vars), units="mixed",
    )
