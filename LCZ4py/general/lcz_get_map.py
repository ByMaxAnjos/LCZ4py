"""
lcz_get_map.py

Global LCZ map downloader.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

import geopandas as gpd

from LCZ4py._internal._lcz_map_engine import run_async_core

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "~/.lcz4r_cache"
GLOBAL_URL = "https://zenodo.org/records/8419340/files/lcz_filter_v3.tif?download=1"


def _validate_inputs(city: Optional[str], roi: Optional[gpd.GeoDataFrame], lang: str) -> None:
    """Fail fast (and clearly) on unusable city/ROI arguments."""
    from LCZ4py._internal.i18n_messages import lcz_msg

    if city is None and roi is None:
        raise ValueError(lcz_msg("no_map_input", lang))
    if city is not None:
        if not isinstance(city, str) or not city.strip():
            raise ValueError("city must be a non-empty place name, e.g. city='Berlin'.")
    if roi is not None and not isinstance(roi, gpd.GeoDataFrame):
        raise TypeError(
            f"roi must be a GeoDataFrame with polygon geometry, got {type(roi).__name__}. "
            "If you picked a previous result, make sure it's a vector/boundary output "
            "(e.g. GeoJSON), not a plot or raster result."
        )


def lcz_get_map(
    city: Optional[str] = None,
    roi: Optional[gpd.GeoDataFrame] = None,
    isave_map: bool = False,
    cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    lang: str = "en",
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> str:
    """Download the global LCZ map clipped to a city or ROI.

    Parameters
    ----------
    city : str, optional
        Place name resolved through OpenStreetMap/Nominatim.
    roi : GeoDataFrame, optional
        Custom study-area polygon(s). Provide either ``city`` or ``roi``.
    isave_map : bool
        Also copy the clipped GeoTIFF into ``LCZ4r_output/lcz_map.tif``.
    cache : bool
        Cache the geocoded boundary and the clipped raster under
        ``cache_dir`` (default ``~/.lcz4r_cache``).
    cache_dir : str
        Cache directory. Ignored (a temporary directory is used) when
        ``cache=False``.
    lang : {"en", "pt", "es", "zh"}
        Message language.
    verbose : bool
        Log progress messages.
    max_workers : int, optional
        Thread-pool size for large-area tiled crop/mask work. Defaults to
        ``min(8, os.cpu_count())``.

    Returns
    -------
    str
        Path to the clipped GeoTIFF (one band, LCZ classes 1-17, 0 = nodata).

    Notes
    -----
    Streaming via ``/vsicurl/`` means only the pixels covering your study area
    are fetched — the ~1 GB global file is never downloaded in full.
    """
    _validate_inputs(city, roi, lang)

    if not cache:
        # If cache is disabled, use a temporary directory
        cache_dir = tempfile.mkdtemp()

    return run_async_core(
        city=city,
        roi=roi,
        url=GLOBAL_URL,
        cache_dir=cache_dir,
        isave_map=isave_map,
        lang=lang,
        verbose=verbose,
        max_workers=max_workers,
    )


def lcz_clear_cache(cache_dir: Optional[str] = None) -> int:
    """Remove all cached study areas and clipped maps.

    Returns the number of deleted files.
    """
    cache_dir = os.path.expanduser(cache_dir or DEFAULT_CACHE_DIR)
    if not os.path.isdir(cache_dir):
        return 0
    deleted = 0
    for fn in os.listdir(cache_dir):
        if fn.startswith(("study_area_", "clipped_")) and fn.endswith((".arrow", ".tif")):
            try:
                os.remove(os.path.join(cache_dir, fn))
                deleted += 1
            except OSError:
                pass
    return deleted


__all__ = ["lcz_get_map", "lcz_clear_cache"]
