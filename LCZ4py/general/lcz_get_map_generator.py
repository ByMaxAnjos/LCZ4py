"""
lcz_get_map_generator.py

Download from LCZ Generator platform.
Uses async chunked streaming to prevent memory bloat.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Optional

import httpx
import rasterio
from rasterio.vrt import WarpedVRT

from LCZ4py._internal._lcz_map_engine import _stream_and_clip_cog, _async_download_file, _atomic_copy_raster
import aiofiles

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"

async def _get_generator_map_core(
    id: str,
    band: str = "lczFilter",
    isave_map: bool = False,
    roi=None, # Can optionally clip to ROI if provided
):
    if id is None:
        raise ValueError("Provide a correct ID from LCZ Factsheet.")
    if band not in ("lcz", "lczFilter"):
        raise ValueError("band must be either 'lcz' or 'lczFilter'")

    url = f"https://lcz-generator.rub.de/factsheets/{id}/{id}.tif"
    
    # Download to temp file asynchronously (LCZ Gen files are usually small, not COG)
    temp_full_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        logger.info("Streaming map from LCZ Generator...")
        await _async_download_file(url, temp_full_path, client)

    # Extract band
    band_index = 1 if band == "lcz" else 2
    temp_band_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    
    with rasterio.open(temp_full_path) as src:
        if band_index > src.count:
            band_index = src.count # Fallback
            
        arr = src.read(band_index)
        profile = src.profile.copy()
        profile.update(count=1, dtype='uint8')
        
        with rasterio.open(temp_band_path, "w", **profile) as dst:
            dst.write(arr, 1)
            
    os.remove(temp_full_path) # Cleanup full stack

    if isave_map:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "lcz_map_generator.tif")
        _atomic_copy_raster(temp_band_path, save_path)
        logger.info(f"Saved to {save_path}")
        return save_path

    return temp_band_path


def lcz_get_map_generator(
    id: str = "3110e623fbe4e73b1cde55f0e9832c4f5640ac21",
    band: str = "lczFilter",
    isave_map: bool = False,
    save_extension: str = "tif",
) -> str:
    """Download an LCZ Generator map.
    
    Advanced Features:
    - Async chunked download (prevents RAM exhaustion)
    - Single-band extraction without loading full stack into memory
    """
    return asyncio.run(_get_generator_map_core(id=id, band=band, isave_map=isave_map))

__all__ = ["lcz_get_map_generator"]
