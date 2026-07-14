"""
lcz_get_map_usa.py

CONUS LCZ map downloader.
"""
from __future__ import annotations
from typing import Optional
import geopandas as gpd
from LCZ4py._internal._lcz_map_engine import run_async_core

USA_URL = "https://zenodo.org/records/10835692/files/CONUS_LCZ_map_NLCD_v1.0_epsg4326.tif?download=1"
DEFAULT_CACHE_DIR = "~/.lcz4r_cache"

def lcz_get_map_usa(
    city: Optional[str] = None,
    roi: Optional[gpd.GeoDataFrame] = None,
    isave_map: bool = False,
    isave_usa: bool = False, # Maintained for API parity
    cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    lang: str = "en",
    verbose: bool = True,
) -> str:
    """Download the CONUS LCZ map (Demuzere et al. 2020).
    
    Advanced Features:
    - Streams via /vsicurl/ (downloads only required pixels)
    - GeoArrow Feather caching for boundaries
    - DuckDB Spatial for bounding box math
    """
    if not cache:
        import tempfile
        cache_dir = tempfile.mkdtemp()
        
    return run_async_core(
        city=city,
        roi=roi,
        url=USA_URL,
        cache_dir=cache_dir,
        isave_map=isave_map,
        lang=lang,
        verbose=verbose,
    )

__all__ = ["lcz_get_map_usa"]