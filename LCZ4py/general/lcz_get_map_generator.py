"""
lcz_get_map_generator.py

Download from the LCZ Generator platform.
Uses async chunked streaming to prevent memory bloat, validates the transfer
before caching, and only extracts the requested band.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Optional

import httpx
import numpy as np
import rasterio

from LCZ4py._internal._lcz_map_engine import (
    DOWNLOAD_TIMEOUT,
    _async_download_file,
    _atomic_copy_raster,
    _is_valid_raster,
    run_coroutine,
)

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"

_FACTSHEET_ID_RE = re.compile(r"^[0-9A-Za-z._-]+$")
_VALID_BANDS = ("lcz", "lczFilter")


def _validate_generator_inputs(
    id: str,
    band: str,
    save_extension: str,
) -> str:
    """Validate identifiers and normalise the output extension."""
    if id is None or not isinstance(id, str) or not id.strip():
        raise ValueError(
            "Provide a correct ID from LCZ Factsheet "
            "(the long hexadecimal hash, e.g. '3110e623fbe4e73b1cde55f0e9832c4f5640ac21')."
        )
    id = id.strip()
    if not _FACTSHEET_ID_RE.match(id):
        raise ValueError(
            f"LCZ Generator ID {id!r} contains unexpected characters. "
            "Copy it verbatim from the factsheet URL."
        )
    if band not in _VALID_BANDS:
        raise ValueError(f"band must be one of {_VALID_BANDS}, got {band!r}")

    ext = (save_extension or "tif").lower().lstrip(".")
    if ext in ("tif", "tiff"):
        ext = "tif"
    else:
        raise ValueError(
            f"save_extension must be 'tif' for LCZ Generator maps, got {save_extension!r}"
        )
    return ext


async def _get_generator_map_core(
    id: str,
    band: str = "lczFilter",
    isave_map: bool = False,
    roi=None,  # Can optionally clip to ROI if provided
    output_dir: str = OUTPUT_DIR,
    lang: str = "en",
    verbose: bool = True,
):
    id = id.strip()
    url = f"https://lcz-generator.rub.de/factsheets/{id}/{id}.tif"

    # Download to temp file asynchronously (LCZ Gen files are usually small, not COG)
    temp_full_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    temp_band_path: Optional[str] = None

    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            if verbose:
                logger.info("Streaming map from LCZ Generator...")
            size = await _async_download_file(url, temp_full_path, client)
        if verbose:
            logger.info("Downloaded %.1f MB from LCZ Generator.", size / 1e6)

        if not _is_valid_raster(temp_full_path):
            raise ValueError(
                "The downloaded file is not a readable GeoTIFF. Check the "
                "factsheet ID — the LCZ Generator may not have published this map yet."
            )

        # Extract band
        band_index = 1 if band == "lcz" else 2
        temp_band_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name

        with rasterio.open(temp_full_path) as src:
            if band_index > src.count:
                logger.warning(
                    "'%s' band not present (raster has %d band(s)); using band %d instead.",
                    band, src.count, src.count,
                )
                band_index = src.count
            arr = src.read(band_index)
            profile = src.profile.copy()
            profile.update(
                count=1,
                dtype="uint8",
                nodata=0,
                compress="lzw",
                BIGTIFF="IF_SAFER",
            )
            # Class codes are 1-17; anything else is nodata, not a class.
            if arr.dtype.kind == "f":
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.clip(arr, 0, 255).astype("uint8")
            arr[(arr < 1) | (arr > 17)] = 0

            with rasterio.open(temp_band_path, "w", **profile) as dst:
                dst.write(arr, 1)
    finally:
        # Cleanup the full stack as soon as the band is extracted
        try:
            os.remove(temp_full_path)
        except OSError:
            pass

    if isave_map:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "lcz_map_generator.tif")
        _atomic_copy_raster(temp_band_path, save_path)
        try:
            os.remove(temp_band_path)
        except OSError:
            pass
        if verbose:
            logger.info("Saved to %s", save_path)
        return save_path

    return temp_band_path


def lcz_get_map_generator(
    id: str = "3110e623fbe4e73b1cde55f0e9832c4f5640ac21",
    band: str = "lczFilter",
    isave_map: bool = False,
    save_extension: str = "tif",
    output_dir: str = OUTPUT_DIR,
    lang: str = "en",
    verbose: bool = True,
) -> str:
    """Download an LCZ Generator map.

    Parameters
    ----------
    id : str
        Factsheet ID of the map on https://lcz-generator.rub.de/factsheets/.
    band : {"lcz", "lczFilter"}
        ``"lczFilter"`` (default, second band) is the filtered map; ``"lcz"``
        is band 1 of the factsheet GeoTIFF.
    isave_map : bool
        Copy the single-band map into ``output_dir/lcz_map_generator.tif``.
    save_extension : str
        Only ``"tif"``/``"tiff"`` is supported (the source is a GeoTIFF);
        anything else raises ``ValueError``.
    output_dir : str
        Directory used when ``isave_map=True``.
    lang : {"en", "pt", "es", "zh"}
        Message language (reserved; diagnostics are logged in English here).
    verbose : bool
        Log download/extraction progress.

    Returns
    -------
    str
        Path to a single-band GeoTIFF (LCZ classes 1-17, 0 = nodata).
    """
    _validate_generator_inputs(id, band, save_extension)

    return run_coroutine(
        _get_generator_map_core(
            id=id,
            band=band,
            isave_map=isave_map,
            output_dir=output_dir,
            lang=lang,
            verbose=verbose,
        )
    )


__all__ = ["lcz_get_map_generator"]
