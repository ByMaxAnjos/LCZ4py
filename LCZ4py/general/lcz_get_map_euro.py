"""
lcz_get_map_euro.py

European LCZ map downloader.
"""

from __future__ import annotations

import logging
import tempfile
from typing import Optional

import geopandas as gpd

from LCZ4py._internal._lcz_map_engine import run_async_core

logger = logging.getLogger(__name__)

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
    max_workers: Optional[int] = None,
) -> str:
    """Download the European LCZ map (Demuzere et al. 2019).

    Parameters
    ----------
    city : str, optional
        Place name resolved through OpenStreetMap/Nominatim.
    roi : GeoDataFrame, optional
        Custom study-area polygon(s). Provide either ``city`` or ``roi``.
    isave_map : bool
        Also copy the clipped GeoTIFF into ``LCZ4r_output/lcz_map.tif``.
    isave_euro : bool
        Retained for API parity with LCZ4r. Ignored: COG streaming means the
        full continental file is never downloaded, so saving it separately
        would only duplicate the source.
    cache : bool
        Cache the geocoded boundary and the clipped raster.
    cache_dir : str
        Cache directory (default ``~/.lcz4r_cache``).
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
        Path to the clipped GeoTIFF.

    Raises
    ------
    ValueError
        If the study area is outside the European map extent — use
        :func:`lcz_get_map` for other regions.
    """
    from LCZ4py.general.lcz_get_map import _validate_inputs

    _validate_inputs(city, roi, lang)

    if not cache:
        cache_dir = tempfile.mkdtemp()

    return run_async_core(
        city=city,
        roi=roi,
        url=EURO_URL,
        cache_dir=cache_dir,
        isave_map=isave_map,
        lang=lang,
        verbose=verbose,
        max_workers=max_workers,
    )


__all__ = ["lcz_get_map_euro"]
