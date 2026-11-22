"""Shared helpers for the LCZ-map-cropped ``lcz_grid_*`` downloaders
(``lcz_grid_chirps``, ``lcz_grid_era5``, ``lcz_grid_pdsi``,
``lcz_grid_pollution_ghap``, ``lcz_grid_pollution_merra2``).

Each of these downloads a gridded climate/environmental raster and crops +
reprojects it onto an existing LCZ classification GeoTIFF's grid (from
``lcz_get_map``) — the same ``_TargetGrid``/``reproject_match`` pattern used
by ``lcz_get_lst.py`` and ``lcz_get_planetary_computer.py``, rather than
aggregating to municipality polygons.

The grid helpers are defined here (instead of importing them from
``lcz_get_lst``) so this module stays importable without the optional
Planetary-Computer / xarray stack: only the reprojection functions need
``rioxarray``, and they raise a clear ``ImportError`` when it is missing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError

try:  # core dependency, but keep the import guard for degraded installs
    import rioxarray  # noqa: F401 - registers the .rio accessor

    HAS_RIOXARRAY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_RIOXARRAY = False

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:  # pragma: no cover - depends on the environment
    xr = None  # type: ignore[assignment]
    HAS_XARRAY = False

try:  # rioxarray >= 0.15
    from rioxarray.exceptions import NoDataInBounds
except Exception:  # pragma: no cover - very old rioxarray

    class NoDataInBounds(Exception):  # type: ignore[no-redef]
        """Fallback when rioxarray does not expose its exception type."""


logger = logging.getLogger(__name__)

DEFAULT_GRID_CACHE_DIR = "~/.lcz4r_cache/grid"

#: Resampling names the ``lcz_grid_*`` functions accept.
VALID_RESAMPLING = ("average", "bilinear", "nearest", "cubic", "mode", "max", "min", "med")

#: LCZ classes with data (1-17); 0 is nodata, 11-17 are natural/water classes.
LCZ_MIN_CLASS = 1
LCZ_MAX_CLASS = 17


# ── Target grid ───────────────────────────────────────────────────────────────

@dataclass
class _TargetGrid:
    """Geometry of the LCZ map a ``lcz_grid_*`` result is aligned to."""

    shape: tuple[int, int]      # (H, W)
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    valid_mask: np.ndarray      # True where the LCZ map has data
    bbox_wgs84: tuple[float, float, float, float]

    @property
    def height(self) -> int:
        return int(self.shape[0])

    @property
    def width(self) -> int:
        return int(self.shape[1])

    @property
    def is_geographic(self) -> bool:
        try:
            return bool(self.crs.is_geographic)
        except Exception:  # pragma: no cover - exotic CRS objects
            return False

    @property
    def n_valid_pixels(self) -> int:
        return int(np.count_nonzero(self.valid_mask))


def _load_target_grid(x: Union[str, Path]) -> _TargetGrid:
    """Read an LCZ classification GeoTIFF into a :class:`_TargetGrid`.

    Raises
    ------
    FileNotFoundError
        If ``x`` does not exist (with the resolved absolute path).
    ValueError
        If the raster has no valid LCZ pixel (classes 1-17).
    """
    path = Path(os.path.expanduser(str(x)))
    if not path.exists():
        raise FileNotFoundError(f"LCZ map not found: {path.resolve()}")

    try:
        with rasterio.open(str(path)) as src:
            if src.count < 1:
                raise ValueError(f"LCZ map has no raster band: {path}")
            class_arr = src.read(1)
            transform = src.transform
            crs = src.crs or rasterio.crs.CRS.from_epsg(4326)
            shape_hw = (src.height, src.width)
            bounds = src.bounds
    except RasterioIOError as exc:
        raise ValueError(f"Could not open the LCZ map {path}: {exc}") from exc

    valid_mask = (
        np.isfinite(class_arr)
        & (class_arr >= LCZ_MIN_CLASS)
        & (class_arr <= LCZ_MAX_CLASS)
    )
    if not valid_mask.any():
        raise ValueError(
            f"LCZ map {path} has no valid LCZ class pixels (expected classes "
            f"{LCZ_MIN_CLASS}-{LCZ_MAX_CLASS})."
        )

    if crs.to_epsg() == 4326:
        bbox_wgs84 = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    else:
        transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        # Sample all four corners: a projected rectangle is not necessarily
        # axis-aligned once reprojected.
        xs, ys = transformer.transform(
            [bounds.left, bounds.right, bounds.left, bounds.right],
            [bounds.bottom, bounds.bottom, bounds.top, bounds.top],
        )
        bbox_wgs84 = (min(xs), min(ys), max(xs), max(ys))

    return _TargetGrid(shape_hw, transform, crs, valid_mask, bbox_wgs84)


#: Backwards-compatible public alias.
load_target_grid = _load_target_grid


def cache_key(*parts: object) -> str:
    """Stable 16-char cache key for a set of request parameters."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Optional-dependency guards ────────────────────────────────────────────────

def _require_raster_stack() -> None:
    if not HAS_XARRAY or not HAS_RIOXARRAY:
        raise ImportError(
            "Cropping/reprojecting gridded data onto an LCZ map requires "
            "'xarray' and 'rioxarray'. Install them with: pip install xarray rioxarray"
        )


def rio_template(grid: _TargetGrid):
    """Empty ``DataArray`` carrying ``grid``'s crs/transform, for ``reproject_match``.

    Raises
    ------
    ImportError
        If ``xarray``/``rioxarray`` are unavailable.
    """
    _require_raster_stack()
    template = xr.DataArray(
        np.zeros(grid.shape, dtype="float32"),
        dims=("y", "x"),
    )
    template = template.rio.write_crs(grid.crs)
    template = template.rio.write_transform(grid.transform)
    return template


class TemplateFactory:
    """Thread-safe ``rio_template`` provider.

    ``xarray`` objects are not safe to mutate from several threads, so each
    worker gets its own template (built lazily, once per thread).
    """

    def __init__(self, grid: _TargetGrid):
        self._grid = grid
        self._local = threading.local()

    def __call__(self):
        template = getattr(self._local, "template", None)
        if template is None:
            template = rio_template(self._grid)
            self._local.template = template
        return template


def make_template_factory(grid: _TargetGrid) -> TemplateFactory:
    """Return a thread-safe factory of reprojection templates for ``grid``."""
    return TemplateFactory(grid)


# ── Raster → grid ─────────────────────────────────────────────────────────────

def _resampling_enum(name: str) -> Resampling:
    enum = getattr(Resampling, name, None)
    if enum is None:
        logger.warning("Unknown resampling %r; falling back to 'average'.", name)
        return Resampling.average
    return enum


def sanitize_array(arr: np.ndarray) -> np.ndarray:
    """Cast to float32 and replace masked/inf/nodata sentinels by NaN."""
    if isinstance(arr, np.ma.MaskedArray):
        out = np.ma.asarray(arr, dtype="float32").filled(np.nan)
    else:
        out = np.asarray(arr, dtype="float32")
    # R/nodata sentinels and +/-inf both mean "missing".
    bad = ~np.isfinite(out) | (out <= -9999.0)
    if bad.any():
        out = np.where(bad, np.nan, out)
    return out


def _prepare_spatial(da):
    """Ensure ``da`` has x/y spatial dims and a CRS (assume WGS84 if unset)."""
    dims = set(da.dims)
    if not {"x", "y"}.issubset(dims):
        lat_name = next((n for n in ("lat", "latitude", "y") if n in dims), None)
        lon_name = next((n for n in ("lon", "longitude", "x") if n in dims), None)
        if lat_name is None or lon_name is None:
            raise ValueError(
                "Could not identify latitude/longitude dimensions on the data "
                f"(dims: {tuple(da.dims)}). Expected lat/lon or latitude/longitude."
            )
        da = da.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)

    if da.rio.crs is None:
        logger.debug("Source has no CRS; assuming EPSG:4326 (lon/lat).")
        da = da.rio.write_crs("EPSG:4326", inplace=False)
    return da


def _spatial_dims(da) -> tuple[str, str]:
    """``(x_dim, y_dim)`` names for ``da`` (x/y or lon/lat after preparation)."""
    try:
        return str(da.rio.x_dim), str(da.rio.y_dim)
    except Exception:  # pragma: no cover - defensive
        return "x", "y"


def _drop_singleton_dims(da):
    """Drop size-1 dimensions that are neither spatial nor the time axis."""
    x_dim, y_dim = _spatial_dims(da)
    keep = {x_dim, y_dim, "time", "valid_time"}
    extra = [d for d in da.dims if d not in keep and da.sizes[d] == 1]
    if extra:
        da = da.isel({d: 0 for d in extra}, drop=True)
    return da


def _bbox_in_crs(grid: _TargetGrid, src_crs) -> tuple[float, float, float, float]:
    """``grid.bbox_wgs84`` expressed in ``src_crs``."""
    minx, miny, maxx, maxy = grid.bbox_wgs84
    try:
        if src_crs is None or src_crs.to_epsg() == 4326:
            return minx, miny, maxx, maxy
    except Exception:  # pragma: no cover - exotic CRS objects
        return minx, miny, maxx, maxy

    transformer = pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    xs, ys = transformer.transform(
        [minx, maxx, minx, maxx],
        [miny, miny, maxy, maxy],
    )
    return min(xs), min(ys), max(xs), max(ys)


def _clip_box(da, grid: _TargetGrid, src_crs=None):
    """Clip ``da`` to ``grid``'s bbox; ``None`` when nothing overlaps."""
    minx, miny, maxx, maxy = grid.bbox_wgs84
    if minx > maxx:
        # Antimeridian-crossing LCZ map: clip_box cannot express a wrapped
        # box, so keep the full extent and let reproject_match crop.
        logger.debug("LCZ bbox crosses the antimeridian; skipping clip_box.")
        return da

    x0, y0, x1, y1 = _bbox_in_crs(grid, src_crs if src_crs is not None else da.rio.crs)

    # Some sources store longitudes as 0..360 (e.g. older ERA5/CHIRPS
    # mirrors). Translate the requested box instead of silently clipping to
    # nothing.
    x_dim, _ = _spatial_dims(da)
    if x_dim in da.coords:
        try:
            if float(da[x_dim].max()) > 180.0 and x0 < 0.0:
                x0, x1 = x0 % 360.0, x1 % 360.0
                if x1 < x0:  # box wrapped across the seam
                    logger.debug("0-360 longitude grid: skipping clip_box.")
                    return da
                if x1 > 360.0:
                    return da
        except (TypeError, ValueError):  # pragma: no cover - non-numeric coords
            pass

    try:
        clipped = da.rio.clip_box(x0, y0, x1, y1, auto_expand=True)
    except (NoDataInBounds, ValueError, RasterioIOError) as exc:
        logger.debug("clip_box found no overlap with the LCZ grid: %s", exc)
        return None

    if clipped.size == 0 or not clipped.dims:
        return None
    spatial = [d for d in clipped.dims if d in _spatial_dims(clipped)]
    if any(clipped.sizes[d] == 0 for d in spatial):
        return None
    return clipped


def reproject_to_grid(da, grid: _TargetGrid, template=None, resampling: str = "average") -> np.ndarray:
    """Reproject an already-clipped single-layer ``DataArray`` onto ``grid``.

    Returns float32 with NaN outside the LCZ map's valid pixels.
    """
    _require_raster_stack()
    if template is None:
        template = rio_template(grid)

    regridded = da.rio.reproject_match(template, resampling=_resampling_enum(resampling))
    arr = sanitize_array(regridded.values)
    return np.where(grid.valid_mask, arr, np.nan)


def clip_dataarray_block(da, grid: _TargetGrid):
    """Prepare and clip a (possibly time-stacked) ``DataArray`` to ``grid``.

    Used to clip **once** per file and then reproject each time slice, instead
    of repeating the clip for every day/month.
    """
    _require_raster_stack()
    da = _drop_singleton_dims(_prepare_spatial(da))
    return _clip_box(da, grid)


def crop_reproject_to_grid(
    raster_path: Union[str, Path],
    grid: _TargetGrid,
    template=None,
    resampling: str = "average",
    band: int = 1,
    nodata_below: Optional[float] = None,
) -> np.ndarray:
    """Crop a raster to ``grid``'s bbox and reproject/resample it onto
    ``grid``'s exact shape/transform/CRS, masked to the LCZ map's valid
    pixels. ``raster_path`` may be a local path or a ``/vsi...`` URL.

    ``nodata_below`` additionally masks values at or below that threshold
    **before** resampling — needed when a source encodes missing data as a
    numeric sentinel (e.g. CHIRPS ``-9999``) without declaring it as nodata,
    which would otherwise contaminate averaged pixels.

    Returns an all-NaN array (instead of raising) when the raster does not
    overlap the grid at all.
    """
    _require_raster_stack()
    if template is None:
        template = rio_template(grid)

    da = rioxarray.open_rasterio(str(raster_path), masked=True)
    try:
        if "band" in da.dims:
            if band > da.sizes["band"]:
                raise ValueError(
                    f"{raster_path} has {da.sizes['band']} band(s); band={band} requested."
                )
            da = da.isel(band=band - 1, drop=True)
        elif da.ndim > 2:
            da = da.isel({da.dims[0]: 0}, drop=True)

        da = _prepare_spatial(da)
        if nodata_below is not None:
            da = da.where(da > nodata_below)
        clipped = _clip_box(da, grid)
        if clipped is None:
            logger.warning(
                "No overlap between %s and the LCZ map footprint.",
                os.path.basename(str(raster_path)),
            )
            return np.full(grid.shape, np.nan, dtype="float32")
        return reproject_to_grid(clipped, grid, template, resampling)
    finally:
        try:
            da.close()
        except Exception:  # pragma: no cover - best effort
            pass


def crop_reproject_dataarray(da, grid: _TargetGrid, template=None, resampling: str = "average") -> np.ndarray:
    """Same as :func:`crop_reproject_to_grid` but for an already-open xarray
    ``DataArray`` (e.g. one time slice of a multi-day NetCDF) instead of a
    raster path. Expects lat/lon (or latitude/longitude) coordinates.
    """
    _require_raster_stack()
    if template is None:
        template = rio_template(grid)

    da = _drop_singleton_dims(_prepare_spatial(da))
    x_dim, y_dim = _spatial_dims(da)
    extra = [d for d in da.dims if d not in (x_dim, y_dim)]
    if extra:
        raise ValueError(
            "crop_reproject_dataarray expects a single 2-D layer; got extra "
            f"dimension(s) {extra}. Select a slice first (e.g. .isel(time=0))."
        )

    clipped = _clip_box(da, grid)
    if clipped is None:
        logger.warning("No overlap between the data and the LCZ map footprint.")
        return np.full(grid.shape, np.nan, dtype="float32")
    return reproject_to_grid(clipped, grid, template, resampling)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class LCZGridResult:
    """Return type shared by the raster ``lcz_grid_*`` downloaders.

    ``array`` is (n_bands, H, W) float32, NaN outside the LCZ map's valid
    pixels (classes 1-17). ``bands`` labels each band (an ISO date for
    single-variable sources, or ``"{variable}_{date}"`` for multi-variable
    ones) and has the same length as ``array``'s first axis.
    """
    path: Optional[str] = None
    array: Optional[np.ndarray] = None
    bands: Optional[list[str]] = None
    variables: Optional[list[str]] = None
    units: Optional[str] = None
    gdf: Optional[Any] = None
    geoarrow_table: Optional[object] = None


# ── Stack I/O ─────────────────────────────────────────────────────────────────

def write_grid_stack(
    path: str,
    array: np.ndarray,
    bands: list[str],
    grid: _TargetGrid,
    variable: str = "",
    units: str = "",
) -> None:
    """Write a (n_bands, H, W) stack as a LZW-compressed, NaN-nodata GeoTIFF.

    The write is atomic: data goes to ``<path>.part`` and is renamed into
    place only after the file is closed, so an interrupted run never leaves a
    half-written cache file behind.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"array must be 2-D or 3-D (bands, H, W); got shape {arr.shape}")
    if arr.shape[0] != len(bands):
        raise ValueError(
            f"array has {arr.shape[0]} band(s) but {len(bands)} label(s) were given."
        )
    if arr.shape[1:] != tuple(grid.shape):
        raise ValueError(
            f"array spatial shape {arr.shape[1:]} does not match the grid {tuple(grid.shape)}."
        )

    path = os.path.expanduser(str(path))
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".part"

    profile = {
        "driver": "GTiff",
        "height": grid.shape[0],
        "width": grid.shape[1],
        "count": arr.shape[0],
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": np.nan,
        "compress": "lzw",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(arr.astype("float32", copy=False))
            dst.update_tags(variable=variable, units=units)
            for i, label in enumerate(bands, start=1):
                dst.set_band_description(i, str(label))
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:  # pragma: no cover - best effort
                pass
        raise


def read_grid_stack(path: str) -> tuple[np.ndarray, list[str]]:
    """Read a stack written by :func:`write_grid_stack`.

    Missing band descriptions fall back to ``band_1``/``band_2``/… so callers
    never get ``None`` as a band label.
    """
    with rasterio.open(str(path)) as src:
        array = src.read()
        bands = [
            src.descriptions[i] or f"band_{i + 1}"
            for i in range(src.count)
        ]
    if array.ndim == 2:  # pragma: no cover - defensive
        array = array[np.newaxis, ...]
    return array, bands


def try_read_grid_stack(
    path: str,
    expected_bands: Optional[int] = None,
) -> Optional[tuple[np.ndarray, list[str]]]:
    """Read a cached stack, returning ``None`` when it is unusable.

    Corrupt/truncated cache files (interrupted older runs, disk-full) are
    treated as a cache miss instead of crashing the request.
    """
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        array, bands = read_grid_stack(path)
    except Exception as exc:  # noqa: BLE001 - corrupt cache is a cache miss
        logger.warning("Ignoring unreadable cache file %s: %s", path, exc)
        return None
    if expected_bands is not None and array.shape[0] != expected_bands:
        logger.warning(
            "Ignoring cache file %s: %d band(s), expected %d.",
            path, array.shape[0], expected_bands,
        )
        return None
    return array, bands
