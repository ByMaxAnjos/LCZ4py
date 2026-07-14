"""Shared download helper for the raster ``lcz_grid_*`` family (crop+mask
onto an LCZ map's grid — see ``_lcz_grid_raster_base.py`` for the rest of
that shared machinery).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def download_file(url: str, cache_path: str, use_cache: bool = True, verbose: bool = True) -> bool:
    """Download ``url`` to ``cache_path`` if not already cached. Returns success."""
    import httpx

    if use_cache and os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        if verbose:
            logger.info("Cache hit: %s", os.path.basename(cache_path))
        return True

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = cache_path + ".part"
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
        os.replace(tmp_path, cache_path)
        if verbose:
            logger.info("Downloaded: %s", os.path.basename(cache_path))
        return True
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        logger.warning("Failed to download %s: %s", url, exc)
        return False
