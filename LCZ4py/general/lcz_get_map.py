"""
lcz_get_map.py

Global LCZ map downloader.
"""
from __future__ import annotations
import logging
from typing import Optional
import geopandas as gpd
from LCZ4py._internal._lcz_map_engine import run_async_core

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "~/.lcz4r_cache"
GLOBAL_URL = "https://zenodo.org/records/8419340/files/lcz_filter_v3.tif?download=1"

def lcz_get_map(
    city: Optional[str] = None,
    roi: Optional[gpd.GeoDataFrame] = None,
    isave_map: bool = False,
    cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    lang: str = "en",
    verbose: bool = True,
) -> str:
    """Download the global LCZ map clipped to a city or ROI.
    
    Advanced Features:
    - Streams via /vsicurl/ (downloads only required pixels)
    - GeoArrow Feather caching for boundaries
    - DuckDB Spatial for bounding box math
    """
    if not cache:
        # If cache is disabled, use a temporary directory
        import tempfile
        cache_dir = tempfile.mkdtemp()
        
    return run_async_core(
        city=city,
        roi=roi,
        url=GLOBAL_URL,
        cache_dir=cache_dir,
        isave_map=isave_map,
        lang=lang,
        verbose=verbose,
    )

def lcz_clear_cache(cache_dir: Optional[str] = None) -> int:
    """Remove all cached study areas and clipped maps."""
    import os
    cache_dir = os.path.expanduser(cache_dir or DEFAULT_CACHE_DIR)
    if not os.path.isdir(cache_dir):
        return 0
    deleted = 0
    for fn in os.listdir(cache_dir):
        if fn.startswith(("study_area_", "clipped_")) and (fn.endswith(".arrow") or fn.endswith(".tif")):
            try:
                os.remove(os.path.join(cache_dir, fn))
                deleted += 1
            except OSError:
                pass
    return deleted

__all__ = ["lcz_get_map", "lcz_clear_cache"]