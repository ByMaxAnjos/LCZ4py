"""
lcz_get_map_euro.py

European LCZ map downloader.
"""
from __future__ import annotations
from typing import Optional
import geopandas as gpd
from LCZ4py._internal._lcz_map_engine import run_async_core

EURO_URL = "https://zenodo.org/records/10835692/files/EU_LCZ_map.tiff?download=1"
DEFAULT_CACHE_DIR = "~/.lcz4r_cache"

def lcz_get_map_euro(
    city: Optional[str] = None,
    roi: Optional[gpd.GeoDataFrame] = None,
    isave_map: bool = False,
    isave_euro: bool = False,  # Maintained for API parity
    cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    lang: str = "en",
    verbose: bool = True,
) -> str:
    """Download the European LCZ map (Demuzere et al. 2019).
    
    Advanced Features:
    - Streams via /vsicurl/ (downloads only required pixels)
    - GeoArrow Feather caching for boundaries
    - DuckDB Spatial for bounding box math
    """
    if not cache:
        import tempfile
        cache_dir = tempfile.mkdtemp()
        
    # Note: isave_euro is ignored in the advanced version to prevent downloading 
    # the whole 50MB file. The COG streaming makes saving the full file redundant.
    return run_async_core(
        city=city,
        roi=roi,
        url=EURO_URL,
        cache_dir=cache_dir,
        isave_map=isave_map,
        lang=lang,
        verbose=verbose,
    )

__all__ = ["lcz_get_map_euro"]