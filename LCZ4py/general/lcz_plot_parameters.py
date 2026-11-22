"""
lcz_plot_parameters.py

LCZ parameter visualization — maps, facets, and (via ``lcz_cal_indices``)
scatter/radar/correlation charts.

- Plotly WebGL/heatmap rendering for every parameter band
- Optional Datashader rendering for very large rasters (falls back to Plotly)
- Colorbars carry the parameter name and its unit range
- Colour-blind-friendly palette sets (sequential, no red/green pairs)
- Consistent typography and >=300 dpi static export via ``lcz_theme``

Optional dependencies (datashader, geoarrow, xarray/rioxarray) are imported
behind ``try/except`` guards: importing this module never fails when they are
missing.
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Union, Literal

import numpy as np
import polars as pl
import pyarrow as pa
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# xarray / rioxarray (optional: only needed for lcz_get_ucp Dataset output)
try:  # pragma: no cover - depends on the environment
    import xarray as xr

    HAS_XARRAY = True
except ImportError:  # pragma: no cover
    xr = None  # type: ignore[assignment]
    HAS_XARRAY = False

try:  # pragma: no cover - depends on the environment
    import rioxarray  # noqa: F401 - registers the .rio accessor on xr.Dataset

    HAS_RIOXARRAY = True
except ImportError:  # pragma: no cover
    HAS_RIOXARRAY = False

# Datashader (optional; Plotly heatmaps are the fallback)
try:  # pragma: no cover - depends on the environment
    import datashader as ds
    import datashader.transfer_functions as tf
    import colorcet as cc  # noqa: F401 - datashader's default palette provider

    HAS_DATASHADER = True
except ImportError:  # pragma: no cover
    ds = None  # type: ignore[assignment]
    tf = None  # type: ignore[assignment]
    HAS_DATASHADER = False

# GeoArrow (optional)
try:  # pragma: no cover - depends on the environment
    import geoarrow.pyarrow as ga
    import geoarrow.pandas as gap  # noqa: F401

    HAS_GEOARROW = True
except ImportError:  # pragma: no cover
    ga = None  # type: ignore[assignment]
    HAS_GEOARROW = False

from LCZ4py._internal.lcz_parameters_data import (
    PARAM_NAMES,
    PARAM_UNITS,
    PARAM_PALETTE_DEFAULT,
    PARAM_PALETTE_INCLUSIVE,
)
from LCZ4py._internal.lcz_theme import (
    METBREWER_PALETTES as _METBREWER_PALETTES,
    add_map_furniture,
)
from LCZ4py.general.lcz_plot_map import _add_caption, _finalize_publication

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"

# Fallbacks that are valid MetBrewer keys (guards against typos in the
# parameters table, e.g. "Pissaro" instead of "Pissarro").
_FALLBACK_PALETTE = "Archambault"
_FALLBACK_PALETTE_INCLUSIVE = "Hokusai2"
_PALETTE_ALIASES = {"pissaro": "Pissarro"}


def _resolve_palette(name: Optional[str], inclusive: bool = False) -> str:
    """Map a palette name to a known MetBrewer key (typo/case tolerant)."""
    if not name:
        return _FALLBACK_PALETTE_INCLUSIVE if inclusive else _FALLBACK_PALETTE
    if name in _METBREWER_PALETTES:
        return name
    alias = _PALETTE_ALIASES.get(name.lower())
    if alias and alias in _METBREWER_PALETTES:
        return alias
    for key in _METBREWER_PALETTES:
        if key.lower() == name.lower():
            return key
    logger.debug("Unknown palette %r; using %s.", name, _FALLBACK_PALETTE)
    return _FALLBACK_PALETTE_INCLUSIVE if inclusive else _FALLBACK_PALETTE


@lru_cache(maxsize=32)
def _palette_to_plotly_colorscale(name: str, n: int = 256) -> list[tuple[float, str]]:
    """Convert a MetBrewer palette to a Plotly colorscale with interpolation.

    Interpolation is linear (``np.interp``) rather than cubic: cubic splines
    overshoot beyond [0, 1] on short palettes, which produced clipped/banded
    colourbars for parameters with narrow ranges.
    """
    palette = _METBREWER_PALETTES.get(_resolve_palette(name), _METBREWER_PALETTES[_FALLBACK_PALETTE])
    n = max(2, int(n))

    rgb_colors = []
    for hex_color in palette:
        h = hex_color.lstrip("#")
        rgb_colors.append(tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)))

    x = np.linspace(0.0, 1.0, len(rgb_colors))
    x_new = np.linspace(0.0, 1.0, n)

    colorscale: list[tuple[float, str]] = []
    for i in range(n):
        pos = x_new[i]
        rgb = [float(np.interp(pos, x, [c[channel] for c in rgb_colors])) for channel in range(3)]
        r, g, b = (min(255, max(0, int(round(v * 255)))) for v in rgb)
        colorscale.append((float(pos), f"rgb({r},{g},{b})"))

    return colorscale


def _palette_to_datashader_cmap(name: str) -> list[tuple[float, tuple]]:
    """Convert palette to Datashader-compatible colormap."""
    palette = _METBREWER_PALETTES.get(_resolve_palette(name), _METBREWER_PALETTES[_FALLBACK_PALETTE])
    rgb_colors = []
    for hex_color in palette:
        h = hex_color.lstrip("#")
        rgb_colors.append(tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)))

    # Create evenly spaced positions
    return [(i / (len(rgb_colors) - 1), c) for i, c in enumerate(rgb_colors)]


def _normalise_param_key(value: str) -> str:
    """Case/underscore-insensitive key, e.g. ``SVFmean`` == ``svf_mean``."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


@lru_cache(maxsize=1)
def _canonical_param_names() -> tuple[str, ...]:
    """Canonical band names in ``lcz_get_parameters`` order (may be empty).

    Band 0 is the LCZ class band, so a 3-D ``(n_params, H, W)`` array returned
    by ``lcz_get_parameters`` maps slice-for-slice onto this list.
    """
    try:
        from LCZ4py.general.lcz_get_parameters import parameter_names

        return tuple(parameter_names())
    except Exception as exc:  # noqa: BLE001 - name lookup is a convenience only
        logger.debug("Cannot read canonical parameter names: %s", exc)
        return ()


# Spellings that do not reduce to a canonical band name on their own
# (``lcz_get_parameters`` calls the aspect-ratio mean ``aspect_mean``).
_PARAM_EXTRA_ALIASES: dict[str, str] = {
    "armean": "aspectmean",
    "aspectratiomean": "aspectmean",
    "lczclass": "lcz",
}

# Canonical band name -> PARAM_NAMES code, for labels/units only
# (``lcz_get_parameters`` calls the aspect-ratio mean ``aspect_mean``).
_PARAM_DISPLAY_ALIASES: dict[str, str] = {
    "aspectmean": "ARmean",
}


@lru_cache(maxsize=1)
def _param_alias_map() -> dict[str, int]:
    """Map accepted parameter spellings to their canonical band index.

    Keys are normalised spellings of (a) the canonical ``lcz_get_parameters``
    band names (``svf_mean``, ``z0`` …), (b) the ``PARAM_NAMES`` short codes
    (``SVFmean``) and (c) their human-readable labels ("Mean Sky View
    Factor"). Values index the canonical band order (0 = LCZ class band).
    """
    canonical = _canonical_param_names()
    aliases: dict[str, int] = {
        _normalise_param_key(name): i for i, name in enumerate(canonical)
    }
    for alias, target in _PARAM_EXTRA_ALIASES.items():
        if target in aliases:
            aliases[alias] = aliases[target]

    for code, label in PARAM_NAMES.items():
        idx = aliases.get(_normalise_param_key(code))
        if idx is not None:
            aliases.setdefault(_normalise_param_key(label), idx)
    return aliases


def _param_code(name: str) -> str:
    """Resolve any accepted spelling to the ``PARAM_NAMES`` code (or *name*)."""
    if name in PARAM_NAMES:
        return name
    norm = _normalise_param_key(name)
    for code in PARAM_NAMES:
        if _normalise_param_key(code) == norm:
            return code
    if norm in _PARAM_EXTRA_ALIASES:
        target = _PARAM_EXTRA_ALIASES[norm]
        for code in PARAM_NAMES:
            if _normalise_param_key(code) == target:
                return code
    if norm in _PARAM_DISPLAY_ALIASES:
        return _PARAM_DISPLAY_ALIASES[norm]
    return name


def _resolve_ndarray_param(
    arr: np.ndarray,
    selector: Union[int, str, None],
    name_fallback: str,
) -> tuple[np.ndarray, str]:
    """Pick a 2-D slice out of a ``(n_params, H, W)`` array.

    Parameters
    ----------
    arr : np.ndarray
        ``(H, W)`` (returned unchanged) or ``(n_params, H, W)``.
    selector : int or str or None
        Slice index or parameter name (``"svf_mean"``, ``"SVFmean"``, …).
        ``None`` keeps slice 0 (the LCZ class band of an
        ``lcz_get_parameters`` stack).
    name_fallback : str
        Name used when the array is already 2-D.

    Returns
    -------
    (slice_2d, slice_name)

    Raises
    ------
    ValueError
        On an unknown selector, or an array with more than 3 dimensions.
    """
    arr = np.asarray(arr)
    if arr.ndim <= 2:
        return arr, name_fallback
    if arr.ndim != 3:
        raise ValueError(
            "ndarray input must be 2-D (H, W) or 3-D (n_params, H, W); got "
            f"{arr.ndim}-D with shape {arr.shape}."
        )

    n_slices = int(arr.shape[0])
    canonical = list(_canonical_param_names())
    slice_names = (
        canonical[:n_slices] if len(canonical) >= n_slices
        else canonical + [f"band_{i}" for i in range(len(canonical), n_slices)]
    )

    if selector is None:
        index = 0
    elif isinstance(selector, (int, np.integer)):
        index = int(selector)
        if not -n_slices <= index < n_slices:
            raise ValueError(
                f"Parameter index {index} is out of range for {n_slices} slice(s)."
            )
        index %= n_slices
    else:
        index = _param_alias_map().get(_normalise_param_key(selector), -1)
        if index == -1:
            raise ValueError(
                f"Unknown parameter {selector!r} for a {n_slices}-slice array. "
                f"Pass an integer index (0-{n_slices - 1}) or one of: "
                f"{', '.join(slice_names)}"
            )
        if not 0 <= index < n_slices:
            raise ValueError(
                f"Parameter {selector!r} is band {index} of the canonical "
                f"lcz_get_parameters stack, but this array only has "
                f"{n_slices} slice(s) (0-{n_slices - 1}). Pass an integer "
                f"index or one of: {', '.join(slice_names)}"
            )
    return arr[index], slice_names[index]


def _ndarray_sources(
    arr: np.ndarray,
    iselect: Union[str, int, Sequence[Union[str, int]], None],
    all_params: bool,
    has_class_band: bool = True,
) -> list[tuple[np.ndarray, str]]:
    """Build ``[(2-D array, band name), …]`` for ndarray input.

    A 3-D ``(n_params, H, W)`` array — exactly what ``lcz_get_parameters``
    returns — is indexed by ``iselect`` (parameter name or integer index), or
    expanded slice by slice with ``all_params=True`` (skipping the leading LCZ
    class band when ``has_class_band`` is True, mirroring file input).
    """
    arr = np.asarray(arr)
    if arr.ndim < 3:
        if all_params:
            raise ValueError(
                "all_params=True requires a multi-band raster or a 3-D "
                f"(n_params, H, W) array; got a {arr.ndim}-D array."
            )
        return [(arr, iselect if isinstance(iselect, str) else "band_1")]

    n_slices = int(arr.shape[0])
    canonical = list(_canonical_param_names())
    slice_names = (
        canonical[:n_slices] if len(canonical) >= n_slices
        else canonical + [f"band_{i}" for i in range(len(canonical), n_slices)]
    )

    if all_params:
        start = 1 if has_class_band else 0
        return [(arr[i], slice_names[i]) for i in range(start, n_slices)]

    if iselect is None:
        raise ValueError(
            "iselect is required for 3-D ndarray input — pass a parameter "
            f"name/index or set all_params=True. Available: {slice_names}"
        )

    requested = (
        [iselect] if isinstance(iselect, (str, int, np.integer)) else list(iselect)
    )
    sources: list[tuple[np.ndarray, str]] = []
    for item in requested:
        slice_arr, slice_name = _resolve_ndarray_param(arr, item, str(item))
        sources.append((slice_arr, slice_name))
    return sources


def _param_label(name: str) -> str:
    """Human-readable parameter label including its unit/range."""
    code = _param_code(name)
    label = PARAM_NAMES.get(code, name)
    unit = (PARAM_UNITS.get(code, "") or "").strip()
    return f"{label} {unit}".strip()


# ── Fast raster reader ───────────────────────────────────────────────────────

class FastRasterReader:
    """High-performance raster reader with optional downsampling."""

    def __init__(
        self,
        path: Union[str, Path, rasterio.io.DatasetReader],
        max_pixels: int = 10_000_000,
        use_mmap: bool = True,
        resampling: Resampling = Resampling.bilinear,
    ):
        if int(max_pixels) < 1:
            raise ValueError("max_pixels must be a positive integer")
        self.max_pixels = int(max_pixels)
        self.use_mmap = use_mmap
        self.resampling = resampling
        self._path = str(path) if not hasattr(path, "read") else None
        self._dataset = path if hasattr(path, "read") else None
        self._opened_here = False

    def __enter__(self):
        if self._dataset is None:
            self._dataset = rasterio.open(self._path)
            self._opened_here = True
        return self

    def __exit__(self, *args):
        # Never close a dataset the caller handed us.
        if self._opened_here and self._dataset is not None:
            self._dataset.close()
            self._dataset = None
            self._opened_here = False

    def read_band(
        self,
        band: int = 1,
        window: Optional[rasterio.windows.Window] = None,
        out_shape: Optional[tuple[int, int]] = None,
    ) -> tuple[np.ndarray, dict]:
        """Read a single band with optimal strategy."""
        src = self._dataset
        if src is None:
            raise RuntimeError("FastRasterReader must be used as a context manager.")
        h, w = src.height, src.width

        if band < 1 or band > src.count:
            raise ValueError(f"band {band} out of range (raster has {src.count} band(s))")

        # Source pixel dimensions (a window narrows the read).
        win_h, win_w = (int(window.height), int(window.width)) if window is not None else (h, w)

        # Calculate target dimensions
        if out_shape is None and window is None and (h * w) > self.max_pixels:
            scale = (self.max_pixels / (h * w)) ** 0.5
            out_shape = (max(1, int(h * scale)), max(1, int(w * scale)))
        elif out_shape is None and window is not None and (win_h * win_w) > self.max_pixels:
            scale = (self.max_pixels / (win_h * win_w)) ** 0.5
            out_shape = (max(1, int(win_h * scale)), max(1, int(win_w * scale)))

        # Read with optional downsampling
        arr = src.read(
            band,
            window=window,
            out_shape=out_shape,
            resampling=self.resampling,
        )

        # Transform/dimensions must describe the *read* (window + downsample).
        if window is not None:
            transform = rasterio.windows.transform(window, src.transform)
            full_h, full_w = win_h, win_w
        else:
            transform = src.transform
            full_h, full_w = h, w
        if out_shape is not None and (out_shape[0] != full_h or out_shape[1] != full_w):
            transform = transform * transform.scale(full_w / arr.shape[1], full_h / arr.shape[0])

        profile = {
            "transform": transform,
            "crs": src.crs,
            "height": arr.shape[0],
            "width": arr.shape[1],
            "dtype": str(arr.dtype),
            "nodata": src.nodata,
        }
        return arr, profile


def read_with_warped_vrt(
    path: str,
    target_crs: str = "EPSG:3857",
    max_pixels: int = 10_000_000,
) -> tuple[np.ndarray, dict]:
    """Read raster with on-the-fly reprojection via WarpedVRT (no temp files)."""
    with rasterio.open(path) as src:
        if src.crs is None:
            # Nothing to reproject from: return the native grid instead of
            # failing inside PROJ.
            logger.warning("Raster has no CRS; returning native grid (no reprojection).")
            h, w = src.height, src.width
            if h * w > max_pixels:
                scale = (max_pixels / (h * w)) ** 0.5
                out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))
            else:
                out_h, out_w = h, w
            arr = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
            return arr, {
                "transform": src.transform * src.transform.scale(w / out_w, h / out_h),
                "crs": src.crs,
                "height": arr.shape[0],
                "width": arr.shape[1],
            }

        h, w = src.height, src.width
        if h * w > max_pixels:
            scale = (max_pixels / (h * w)) ** 0.5
            out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))
        else:
            out_h, out_w = h, w

        # Destination transform for the *requested* output size — passing the
        # full-resolution transform with a downsampled width/height (the old
        # behaviour) silently shifted the output by a fraction of a pixel.
        dst_transform, _, _ = calculate_default_transform(
            src.crs, target_crs, out_w, out_h, *src.bounds
        )

        with WarpedVRT(
            src,
            crs=target_crs,
            resampling=Resampling.bilinear,
            transform=dst_transform,
            width=out_w,
            height=out_h,
        ) as vrt:
            arr = vrt.read(1)
            return arr, {
                "transform": vrt.transform,
                "crs": vrt.crs,
                "height": vrt.height,
                "width": vrt.width,
            }


# ── Datashader-based rendering for massive rasters ───────────────────────────

def _render_with_datashader(
    arr: np.ndarray,
    transform,
    palette_name: str,
    extent: Optional[tuple] = None,
) -> go.Image:
    """Render raster using Datashader for WebGL performance."""
    if not HAS_DATASHADER:  # pragma: no cover - guarded by callers
        raise ImportError(
            "Datashader is required for this rendering path. "
            "Install it with 'pip install datashader', or use renderer='plotly'."
        )
    if not HAS_XARRAY:  # pragma: no cover - datashader pulls xarray in
        raise ImportError("Datashader rendering requires xarray.")

    h, w = arr.shape

    # Create coordinate arrays from transform
    if extent is None:
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + w * transform.a
        ymin = ymax + h * transform.e
        extent = (xmin, xmax, ymin, ymax)

    # Create xarray DataArray
    x = np.linspace(extent[0], extent[1], w)
    y = np.linspace(extent[2], extent[3], h)
    da = xr.DataArray(arr, coords=[("y", y), ("x", x)], name="value")

    # Create canvas
    cvs = ds.Canvas(
        plot_width=min(w, 4096),  # Limit for WebGL
        plot_height=min(h, 4096),
        x_range=(extent[0], extent[1]),
        y_range=(extent[2], extent[3]),
    )

    # Aggregate and shade
    agg = cvs.raster(da, upsample_method="linear")
    cmap = _palette_to_datashader_cmap(palette_name)
    img = tf.shade(agg, cmap=cmap, how="linear")

    # Convert to Plotly
    rgba = img.to_numpy()
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        # Plotly expects an opaque RGBA image; transparent pixels become white.
        alpha = rgba[..., 3:4].astype(float) / 255.0
        rgb = rgba[..., :3].astype(float) * alpha + 255.0 * (1.0 - alpha)
        rgba = np.concatenate([rgb, np.full_like(rgb[..., :1], 255.0)], axis=2).astype(np.uint8)
    return go.Image(
        z=rgba,
        x0=extent[0],
        y0=extent[2],
        dx=(extent[1] - extent[0]) / rgba.shape[1],
        dy=(extent[3] - extent[2]) / rgba.shape[0],
        hovertemplate="x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>",
    )


def _colorbar_stub(
    colorscale: list[tuple[float, str]],
    zmin: float,
    zmax: float,
    title: str,
) -> go.Scatter:
    """Invisible colourbar carrier for image traces (Datashader path)."""
    return go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            color=[zmin],
            showscale=True,
            colorbar=dict(
                title=dict(text=title or None, side="right", font=dict(size=11)),
                thickness=16,
                len=0.75,
                tickfont=dict(size=10),
                outlinewidth=0.8,
                outlinecolor="black",
            ),
        ),
        showlegend=False,
        hoverinfo="skip",
    )


# ── Plotly interactive plot ──────────────────────────────────────────────────

def _plot_one_interactive(
    arr: np.ndarray,
    profile: dict,
    name: str,
    inclusive: bool,
    use_datashader: bool = False,
    palette_override: Optional[str] = None,
    param: Optional[Union[int, str]] = None,
) -> go.Figure:
    """Create a Plotly figure for one parameter band.

    ``arr`` may be a 2-D ``(H, W)`` array or a 3-D ``(n_params, H, W)`` array
    as returned by ``lcz_get_parameters``. For 3-D input the rendered slice is
    selected by ``param`` (integer index or parameter name such as
    ``"svf_mean"``/``"SVFmean"``); when ``param`` is None, ``name`` is used,
    falling back to slice 0 (the LCZ class band) with a warning.
    """
    arr = np.asarray(arr)
    if arr.ndim >= 3:
        selector = param if param is not None else name
        try:
            arr, resolved = _resolve_ndarray_param(arr, selector, name)
        except ValueError:
            logger.warning(
                "3-D array with %d slice(s) cannot be matched to %r; rendering "
                "slice 0. Pass param=<index|name> to choose explicitly.",
                arr.shape[0], selector,
            )
            arr, resolved = arr[0], name
        name = resolved or name

    if arr.ndim != 2:
        raise ValueError(
            f"Cannot plot a {arr.ndim}-D array; expected 2-D (H, W) or "
            "3-D (n_params, H, W)."
        )

    palette_map = PARAM_PALETTE_INCLUSIVE if inclusive else PARAM_PALETTE_DEFAULT
    palette_name = _resolve_palette(
        palette_override or palette_map.get(_param_code(name)), inclusive=inclusive
    )
    colorscale = _palette_to_plotly_colorscale(palette_name)

    # Mask nodata
    arr = np.asarray(arr, dtype=float)
    masked = np.where((arr == 0) | np.isnan(arr) | np.isinf(arr), np.nan, arr)

    # Calculate extent
    h, w = arr.shape
    transform = profile.get("transform")
    if transform is not None:
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + w * transform.a
        ymin = ymax + h * transform.e
        extent = (xmin, xmax, ymin, ymax)
        # flipud: raster row 0 is north, but Plotly y0=ymin (south) — align
        # them (same convention as lcz_plot_map._create_continuous_map).
        # The no-transform fallback below already compensates via an
        # inverted extent, so it must NOT be flipped again.
        arr = np.flipud(arr)
        masked = np.flipud(masked)
    else:
        extent = (0, w, h, 0)

    # Choose rendering strategy based on size
    h, w = arr.shape
    use_ds = HAS_DATASHADER and (use_datashader or (h * w > 4_000_000))

    finite = masked[np.isfinite(masked)]
    if finite.size:
        zmin, zmax = float(np.min(finite)), float(np.max(finite))
        if math.isclose(zmin, zmax):
            pad = abs(zmin) * 0.01 or 0.5
            zmin, zmax = zmin - pad, zmax + pad
    else:
        zmin, zmax = 0.0, 1.0
        logger.warning("Parameter %r has no finite values; plotting an empty map.", name)

    fig = go.Figure()
    colorbar_title = _param_label(name)

    if use_ds and transform is not None:
        try:
            # Use Datashader for massive rasters
            img_trace = _render_with_datashader(arr, transform, palette_name, extent)
            fig.add_trace(img_trace)
            fig.add_trace(_colorbar_stub(colorscale, zmin, zmax, colorbar_title))
        except Exception as exc:  # noqa: BLE001 - fall back to Plotly
            logger.warning("Datashader rendering failed (%s); using Plotly heatmap.", exc)
            use_ds = False

    if not (use_ds and transform is not None):
        # Direct Plotly heatmap
        fig.add_trace(go.Heatmap(
            z=masked,
            x0=extent[0],
            y0=extent[2],
            dx=(extent[1] - extent[0]) / w,
            dy=(extent[3] - extent[2]) / h,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=True,
            colorbar=dict(
                title=dict(text=colorbar_title or None, side="right", font=dict(size=11)),
                thickness=16,
                len=0.75,
                tickfont=dict(size=10),
                outlinewidth=0.8,
                outlinecolor="black",
            ),
            hovertemplate=(
                f"<b>{PARAM_NAMES.get(name, name)}</b><br>"
                "Value: %{z:.4g}<br>"
                "X: %{x:.5f}<br>"
                "Y: %{y:.5f}<extra></extra>"
            ),
            opacity=1.0,
        ))

    # Configure layout
    full_title = PARAM_NAMES.get(_param_code(name), name)
    fig.update_layout(
        title=dict(
            text=full_title,
            font=dict(size=16, color="black"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False,
            ticks="",
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False,
            ticks="",
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=10, r=90, t=55, b=25),
        width=1000,
        height=800,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# ── Facet layout ─────────────────────────────────────────────────────────────

def _build_facet_figure(
    panels: Sequence[tuple[str, np.ndarray, dict]],
    inclusive: bool,
    ncols: Optional[int] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    lang: str = "en",
) -> go.Figure:
    """Combine several parameter maps into one faceted panel figure.

    Each facet gets its own colourbar and a shared typography, so multi-
    parameter figures are print-ready instead of a stack of separate images.
    """
    if not panels:
        raise ValueError("No parameter panels to facet.")
    n = len(panels)
    ncols = int(ncols) if ncols else (1 if n == 1 else 2 if n <= 4 else 3)
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))

    titles = [PARAM_NAMES.get(name, name) for name, _, _ in panels]
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=titles,
        horizontal_spacing=0.16,
        vertical_spacing=0.10 if nrows > 1 else 0.0,
    )

    for idx, (name, arr, profile) in enumerate(panels):
        row, col = divmod(idx, ncols)
        arr = np.asarray(arr, dtype=float)
        masked = np.where((arr == 0) | np.isnan(arr) | np.isinf(arr), np.nan, arr)
        transform = profile.get("transform")
        h, w = arr.shape
        if transform is not None:
            xmin = transform.c
            ymax = transform.f
            xmax = xmin + w * transform.a
            ymin = ymax + h * transform.e
            extent = (xmin, xmax, ymin, ymax)
            masked = np.flipud(masked)
        else:
            extent = (0, w, h, 0)

        finite = masked[np.isfinite(masked)]
        if finite.size:
            zmin, zmax = float(np.min(finite)), float(np.max(finite))
            if math.isclose(zmin, zmax):
                pad = abs(zmin) * 0.01 or 0.5
                zmin, zmax = zmin - pad, zmax + pad
        else:
            zmin, zmax = 0.0, 1.0

        palette_map = PARAM_PALETTE_INCLUSIVE if inclusive else PARAM_PALETTE_DEFAULT
        palette_name = _resolve_palette(palette_map.get(name), inclusive=inclusive)
        palette_override = profile.get("palette")
        if palette_override:
            palette_name = _resolve_palette(palette_override, inclusive=inclusive)

        axis_suffix = "" if idx == 0 else str(idx + 1)
        # Plotly puts every colorbar at x=1.02 *paper* fraction by default, so
        # in a subplot grid the first panel's colorbar lands on top of the
        # second panel. Anchor each one just outside its own panel instead.
        try:
            x_domain = getattr(fig.layout, f"xaxis{axis_suffix}").domain
            y_domain = getattr(fig.layout, f"yaxis{axis_suffix}").domain
            cb_x = float(x_domain[1]) + 0.012
            cb_y = (float(y_domain[0]) + float(y_domain[1])) / 2.0
            cb_len = max(0.2, (float(y_domain[1]) - float(y_domain[0])) * 0.9)
        except Exception:  # pragma: no cover - layout always exposes domains
            cb_x, cb_y, cb_len = 1.02, 0.5, 0.9

        fig.add_trace(
            go.Heatmap(
                z=masked,
                x0=extent[0],
                y0=extent[2],
                dx=(extent[1] - extent[0]) / w,
                dy=(extent[3] - extent[2]) / h,
                colorscale=_palette_to_plotly_colorscale(palette_name),
                zmin=zmin,
                zmax=zmax,
                showscale=True,
                colorbar=dict(
                    x=cb_x,
                    y=cb_y,
                    xanchor="left",
                    yanchor="middle",
                    # The panel title already names the parameter, so the
                    # colourbar only needs its unit/range — a full label
                    # rotated here collides with its own tick labels.
                    title=dict(
                        text=(PARAM_UNITS.get(_param_code(name), "") or None),
                        side="right",
                        font=dict(size=9),
                    ),
                    thickness=10,
                    len=cb_len,
                    tickfont=dict(size=8),
                    outlinewidth=0.6,
                    outlinecolor="black",
                ),
                hovertemplate=(
                    f"<b>{PARAM_NAMES.get(name, name)}</b><br>Value: %{{z:.4g}}"
                    "<extra></extra>"
                ),
            ),
            row=row + 1,
            col=col + 1,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                         ticks="", showline=False, row=row + 1, col=col + 1)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                         ticks="", showline=False,
                         scaleanchor=f"x{axis_suffix}", scaleratio=1,
                         row=row + 1, col=col + 1)

    title_text = title or lcz_msg_or("lcz_map_title", lang)
    if subtitle:
        title_text = f"{title_text}<br><sup>{subtitle}</sup>"
    height = 300 * nrows + 80
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center", font=dict(size=16)),
        margin=dict(l=10, r=90, t=70, b=30),
        width=1000,
        height=max(400, height),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    _add_caption(fig, caption)
    return fig


def lcz_msg_or(key: str, lang: str) -> str:
    """``lcz_msg`` wrapper tolerant of missing keys (keeps plotting alive)."""
    from LCZ4py._internal.i18n_messages import lcz_msg

    try:
        return lcz_msg(key, lang)
    except Exception:  # pragma: no cover - defensive
        return "LCZ parameters"


# ── GeoArrow integration for spatial metadata ─────────────────────────────────

def create_geoarrow_metadata(
    profile: dict,
    arr_shape: tuple[int, int],
) -> Optional[pa.Table]:
    """Create a GeoArrow table with the raster footprint geometry."""
    if not HAS_GEOARROW:
        return None

    transform = profile.get("transform")
    if transform is None:
        return None

    h, w = arr_shape
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + w * transform.a
    ymin = ymax + h * transform.e

    # Create bounding box as GeoArrow polygon
    bbox_coords = np.array([
        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]
    ])

    try:
        schema = ga.schema(["geometry:geometry[polygon]"])
        arrays = [
            pa.array([bbox_coords.ravel().tolist()], type=pa.list_(pa.float64(), 10))
        ]
        return pa.Table.from_arrays(arrays, schema=schema)
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort
        logger.debug("GeoArrow metadata creation failed: %s", exc)
        return None


def _finalize_figure(
    fig: go.Figure,
    name: str,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    isave: bool,
    save_extension: str,
    style: str = "default",
    lang: str = "en",
    dpi: Optional[int] = None,
) -> go.Figure:
    """Apply shared title/caption annotations, then delegate export to lcz_theme."""
    if title and not subtitle:
        fig.update_layout(title=dict(text=title))
    elif title and subtitle:
        fig.update_layout(title=dict(text=f"{title}<br><sup>{subtitle}</sup>"))
    _add_caption(fig, caption)

    return _finalize_publication(
        fig, style=style, isave=isave, save_extension=save_extension,
        filename=name, dpi=dpi, lang=lang,
    )


def _plot_from_dataset(
    ds_: "xr.Dataset",
    iselect: Union[str, Sequence[str], None],
    all_params: bool,
    inclusive: bool,
    isave: bool,
    save_extension: str,
    use_datashader: bool,
    title: Optional[str],
    subtitle: Optional[str],
    caption: Optional[str],
    renderer: Literal["plotly", "datashader", "auto"],
    style: str = "default",
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
    lang: str = "en",
    dpi: Optional[int] = None,
    facet: bool = False,
    ncols: Optional[int] = None,
) -> Union[go.Figure, list[go.Figure]]:
    """Plot parameters straight from an lcz_get_ucp ``combined_rasters`` Dataset.

    Unlike the file-based path, every data variable is a real parameter (no
    leading LCZ-class band to skip), and bands are addressed by name instead
    of position.
    """
    if not HAS_XARRAY or not HAS_RIOXARRAY:  # pragma: no cover - guarded upstream
        raise ImportError(
            "Plotting xarray Datasets requires xarray and rioxarray. "
            "Install them with 'pip install xarray rioxarray'."
        )

    if all_params:
        band_names = list(ds_.data_vars)
    elif iselect is not None:
        band_names = [iselect] if isinstance(iselect, str) else list(iselect)
        missing = [n for n in band_names if n not in ds_.data_vars]
        if missing:
            raise ValueError(f"Parameter(s) not found in dataset: {missing}")
    else:
        raise ValueError("Provide either iselect='Param' or all_params=True")
    if not band_names:
        raise ValueError("The dataset contains no data variables to plot.")

    transform = ds_.rio.transform()
    crs = ds_.rio.crs

    panels = []
    for name in band_names:
        arr = np.squeeze(np.asarray(ds_[name].values, dtype=float))
        if arr.ndim != 2:
            raise ValueError(
                f"Variable {name!r} is not 2-D after squeezing (shape {arr.shape})."
            )
        panels.append((name, arr, {"transform": transform, "crs": crs}))

    if len(panels) == 1:
        name, arr, profile = panels[0]
        use_ds = (renderer == "datashader") or (
            renderer == "auto" and use_datashader and arr.size > 4_000_000
        )
        fig = _plot_one_interactive(arr, profile, name, inclusive, use_datashader=use_ds)
        if transform is not None:
            h, w = arr.shape
            bounds = (
                transform.c,
                transform.f + h * transform.e,
                transform.c + w * transform.a,
                transform.f,
            )
            add_map_furniture(
                fig, bounds=bounds, crs=crs,
                add_scalebar=add_scalebar, add_north_arrow=add_north_arrow,
            )
        return _finalize_figure(
            fig, name, title, subtitle, caption, isave, save_extension,
            style=style, lang=lang, dpi=dpi,
        )

    if facet:
        fig = _build_facet_figure(
            panels, inclusive=inclusive, ncols=ncols,
            title=title, subtitle=subtitle, caption=caption, lang=lang,
        )
        return _finalize_figure(
            fig, "lcz_parameters_facet", None, None, None, isave, save_extension,
            style=style, lang=lang, dpi=dpi,
        )

    figs = []
    for name, arr, profile in panels:
        use_ds = (renderer == "datashader") or (
            renderer == "auto" and use_datashader and arr.size > 4_000_000
        )
        fig = _plot_one_interactive(arr, profile, name, inclusive, use_datashader=use_ds)
        figs.append(_finalize_figure(
            fig, name, title, subtitle, caption, isave, save_extension,
            style=style, lang=lang, dpi=dpi,
        ))
    return figs


# ── Public entry point ────────────────────────────────────────────────────────

def _resolve_band_indices(
    descriptions: Sequence[Optional[str]],
    iselect: Union[str, Sequence[str], None],
    all_params: bool,
    has_class_band: bool,
) -> tuple[list[int], list[str]]:
    """Resolve 1-based band indices + display names from band descriptions.

    ``all_params=True`` returns every band except a leading class band (if
    ``has_class_band``). ``iselect`` looks each requested name up against
    ``descriptions`` directly — this fixes a prior bug where ``iselect``
    band indices were guessed positionally (1, 2, 3, ...) instead of
    actually matching the requested name to its real band position.
    """
    count = len(descriptions)
    start = 2 if (all_params and has_class_band) else 1

    if all_params:
        indices = list(range(start, count + 1))
        if not indices:
            raise ValueError(
                f"No parameter bands to plot (raster has {count} band(s), "
                f"has_class_band={has_class_band})."
            )
        names = [descriptions[i - 1] or f"band_{i}" for i in indices]
        return indices, names

    if iselect is None:
        raise ValueError("Provide either iselect='Param' or all_params=True")
    requested = [iselect] if isinstance(iselect, str) else list(iselect)
    if not requested:
        raise ValueError("iselect is empty; name at least one parameter band.")

    indices = []
    for name in requested:
        try:
            indices.append(descriptions.index(name) + 1)
        except ValueError:
            raise ValueError(
                f"No band with description {name!r}. Available: {list(descriptions)}"
            )
    return indices, requested


def lcz_plot_parameters(
    x: Union[str, Path, rasterio.io.DatasetReader, np.ndarray, dict, "xr.Dataset"],
    iselect: Union[str, Sequence[str], None] = None,
    all_params: bool = False,
    has_class_band: bool = True,
    chart_type: Literal["map", "scatter", "radar", "correlation"] = "map",
    lcz_map: Optional[Union[str, Path]] = None,
    index_x: Optional[str] = None,
    index_y: Optional[str] = None,
    size_by: Optional[str] = None,
    lcz_classes: Optional[list[int]] = None,
    normalize_radar: bool = True,
    inclusive: bool = False,
    isave: bool = False,
    save_extension: str = "html",
    style: str = "default",
    max_pixels: int = 10_000_000,
    use_datashader: bool = True,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    renderer: Literal["plotly", "datashader", "auto"] = "auto",
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
    lang: str = "en",
    facet: bool = False,
    ncols: Optional[int] = None,
    dpi: Optional[int] = None,
) -> Union[go.Figure, list[go.Figure], dict[str, go.Figure]]:
    """Plot LCZ parameter rasters with publication-quality Plotly figures.

    Parameters
    ----------
    x : str, Path, rasterio dataset, numpy array, dict, or xarray.Dataset
        Source data. A file path / rasterio dataset is read band-by-band by
        matching ``iselect`` names against the file's band descriptions (or
        every band via ``all_params``). A numeric ``ndarray`` may be 2-D
        ``(H, W)`` or 3-D ``(n_params, H, W)`` — the latter is exactly what
        ``lcz_get_parameters`` returns and is interpreted in that same band
        order (slice 0 = LCZ class band, then one slice per parameter).
        Select a slice with ``iselect`` (parameter name such as
        ``"svf_mean"``/``"SVFmean"``/``"Mean Sky View Factor"``, or an integer
        index) or expand all slices with ``all_params=True``. A ``dict`` is
        the result of ``lcz_get_ucp(..., stations=None)`` — its
        ``'combined_rasters'`` Dataset is plotted directly, with every
        variable treated as a real parameter (no band skipped). An
        ``xarray.Dataset`` (e.g. ``result['combined_rasters']``) is accepted
        the same way; this form requires ``xarray`` (and ``rioxarray`` for
        georeferencing), and raises a clear ImportError at call time if they
        are not installed. For ``chart_type`` other than ``"map"``, ``x``
        must be a path to an ``lcz_get_indices`` GeoTIFF stack.
    iselect : str, int, sequence of str/int, or None
        Parameter names to plot — matched against the raster's band
        descriptions (e.g. ``"NDVI"`` for an ``lcz_get_indices`` stack, or
        a morphological parameter name for a ``lcz_get_parameters`` stack).
        For 3-D ndarray input, an integer selects a slice and a name is
        resolved case/underscore-insensitively against the canonical
        parameter names. For ``chart_type`` other than ``"map"``, this
        instead restricts which indices feed the chart (defaults to every
        band in ``x``).
    all_params : bool
        Plot every parameter band. For 3-D ndarray input this renders one
        figure per slice (minus the class band, see ``has_class_band``).
        Ignored for ``chart_type != "map"``.
    has_class_band : bool
        Whether band 1 is a leading LCZ-class band to skip when
        ``all_params=True`` — True for ``lcz_get_parameters`` output
        (default), False for stacks with no class band (e.g.
        ``lcz_get_indices`` or ``lcz_get_planetary_computer`` output).
        Ignored when ``iselect`` is given (bands are matched by name
        regardless of position). For a 3-D ndarray it controls whether slice 0
        (the LCZ class band) is skipped by ``all_params=True``.
    chart_type : {"map", "scatter", "radar", "correlation"}
        "map" (default): the spatial heatmap behavior described above.
        "scatter": per-pixel ``index_x`` vs ``index_y``, colored by LCZ
        class, optionally sized by ``size_by``. "radar": one polar trace
        per LCZ class (or ``lcz_classes``) across the requested indices.
        "correlation": Pearson correlation matrix between the requested
        indices. These three delegate to ``lcz_cal_indices`` (which is
        where their statistics/plotting logic actually lives — this is a
        convenience entry point, not a second implementation) and require
        ``lcz_map``.
    lcz_map : str or Path, optional
        Path to the LCZ class GeoTIFF (classes 1-17) that ``x`` was
        computed on the same grid of. Required for ``chart_type != "map"``.
    index_x, index_y : str, optional
        Required for ``chart_type="scatter"``.
    size_by : str, optional
        Optional third index controlling scatter marker size.
    lcz_classes : list of int, optional
        Restrict the radar chart to these LCZ classes.
    normalize_radar : bool
        Min-max normalize each index's radar axis across the plotted
        classes, so indices with very different natural ranges stay
        visually comparable. Default True.
    inclusive : bool
        Use the colorblind-friendly palette set.
    isave : bool
        Save each rendered figure.
    save_extension : str
        "html" for interactive, "png"/"pdf"/"svg"/"tiff" for static.
    style : str
        Publication style preset: "default", "nature", "science", or
        "generic_bw". Controls font, figure size (mm), DPI, and palette
        used when ``isave`` and ``save_extension`` != "html".
    max_pixels : int
        Down-sample threshold. Ignored for ``chart_type != "map"``.
    use_datashader : bool
        Use Datashader for large rasters when installed. Ignored for
        ``chart_type != "map"``; falls back to Plotly when unavailable.
    title, subtitle, caption : str, optional
        Figure annotations.
    renderer : {"plotly", "datashader", "auto"}
        Rendering backend. Ignored for ``chart_type != "map"``.
    add_scalebar, add_north_arrow : bool
        Add a scale bar / north arrow to the rendered map. Only applies to
        ``chart_type="map"`` (skipped silently for geographic CRSs).
    lang : str
        Message language ("en"/"pt"/"es"/"zh"). Only used for
        ``chart_type != "map"``.
    facet : bool
        Combine every selected parameter into a single faceted figure
        (one panel + colourbar per parameter) instead of returning one
        figure per band. Default False preserves the historic return type.
    ncols : int, optional
        Column count for ``facet=True``. Defaults to 2 (up to 4 panels) or
        3 (more).
    dpi : int, optional
        Resolution for static exports (default >= 300 dpi).

    Returns
    -------
    plotly.graph_objects.Figure, list thereof, or (unused here) dict thereof

    Examples
    --------
    >>> from LCZ4py.general.lcz_get_indices import lcz_get_indices
    >>> idx = lcz_get_indices(pc_result, indices=["NDVI", "NDBI"])
    >>> lcz_plot_parameters(idx.path, iselect="NDVI")
    >>> lcz_plot_parameters(idx.path, all_params=True, has_class_band=False, facet=True)
    """
    if chart_type != "map":
        if lcz_map is None:
            raise ValueError(f"chart_type={chart_type!r} requires lcz_map (the LCZ class GeoTIFF)")
        if not isinstance(x, (str, Path)):
            raise ValueError(
                f"chart_type={chart_type!r} requires x to be a path to an "
                f"lcz_get_indices GeoTIFF stack, got {type(x)}"
            )

        # These three chart kinds are LCZ-class-conditioned statistics, not
        # spatial maps — that logic already lives in lcz_cal_indices, so
        # delegate rather than re-implement it here. Local import: avoids
        # paying the cost (and any future circularity risk) for the common
        # chart_type="map" path, which never touches this module.
        from LCZ4py.general.lcz_cal_indices import lcz_cal_indices

        requested = [iselect] if isinstance(iselect, str) else list(iselect) if iselect else None
        result = lcz_cal_indices(
            lcz_map, x, indices=requested, plot_type=chart_type,
            index_x=index_x, index_y=index_y, size_by=size_by,
            lcz_classes=lcz_classes, normalize_radar=normalize_radar,
            inclusive=inclusive, isave=isave, save_extension=save_extension,
            title=title, subtitle=subtitle, caption=caption, lang=lang,
        )
        return result.fig

    if x is None:
        raise ValueError("x must be a raster path / dataset / array")
    if renderer not in ("plotly", "datashader", "auto"):
        raise ValueError(
            f"renderer must be 'plotly', 'datashader' or 'auto', got {renderer!r}"
        )
    if (save_extension or "").lower().lstrip(".") not in (
        "html", "png", "pdf", "svg", "tiff", "jpg", "jpeg", "webp"
    ):
        raise ValueError(
            f"Unsupported save_extension {save_extension!r} "
            "(use html, png, pdf, svg, tiff, jpg or webp)."
        )
    save_extension = (save_extension or "html").lower().lstrip(".")

    # Accept the dict returned by lcz_get_ucp() directly
    if isinstance(x, dict):
        x = x.get("combined_rasters")
        if x is None:
            raise ValueError(
                "lcz_get_ucp result has no 'combined_rasters' to plot — "
                "check 'failed_variables' in the result"
            )

    if HAS_XARRAY and isinstance(x, xr.Dataset):
        return _plot_from_dataset(
            x, iselect=iselect, all_params=all_params, inclusive=inclusive,
            isave=isave, save_extension=save_extension, use_datashader=use_datashader,
            title=title, subtitle=subtitle, caption=caption, renderer=renderer,
            style=style, add_scalebar=add_scalebar, add_north_arrow=add_north_arrow,
            lang=lang, dpi=dpi, facet=facet, ncols=ncols,
        )
    if not HAS_XARRAY and hasattr(x, "data_vars") and hasattr(x, "attrs"):
        # An xarray-like object was passed but xarray is not installed. Raise
        # the actionable error here instead of an AttributeError further down.
        raise ImportError(
            "Plotting an xarray.Dataset requires xarray and rioxarray. "
            "Install them with 'pip install xarray rioxarray', or pass a "
            "raster path / numpy array instead."
        )


    # Determine which bands to render
    user_dataset = None
    ndarray_sources: Optional[list[tuple[np.ndarray, str]]] = None
    if isinstance(x, np.ndarray):
        # 2-D (H, W) or 3-D (n_params, H, W) — the latter is exactly what
        # lcz_get_parameters returns (band 0 = LCZ class band).
        ndarray_sources = _ndarray_sources(x, iselect, all_params, has_class_band)
        band_indices, band_names = [], []
    else:
        if not (all_params or iselect is not None):
            raise ValueError("Provide either iselect='Param' or all_params=True")
        if isinstance(x, (str, Path)):
            with rasterio.open(str(x)) as src:
                descriptions = list(src.descriptions)
        else:
            user_dataset = x
            descriptions = list(x.descriptions)
        band_indices, band_names = _resolve_band_indices(
            descriptions, iselect, all_params, has_class_band,
        )

    panels: list[tuple[str, np.ndarray, dict]] = []

    if ndarray_sources is not None:
        for arr, name in ndarray_sources:
            arr = np.asarray(arr, dtype=float)
            panels.append((
                name,
                arr,
                {"transform": None, "crs": None,
                 "height": arr.shape[0], "width": arr.shape[1]},
            ))
    else:
        for band_idx, name in zip(band_indices, band_names):
            # Read data with optimal strategy
            if user_dataset is not None:
                # Caller-owned dataset: read without taking ownership of the handle.
                reader = FastRasterReader(user_dataset, max_pixels=max_pixels)
                arr, profile = reader.read_band(band=band_idx)
            else:
                with FastRasterReader(x, max_pixels=max_pixels) as reader:
                    arr, profile = reader.read_band(band=band_idx)
            panels.append((name, arr, profile))

    if facet:
        fig = _build_facet_figure(
            panels, inclusive=inclusive, ncols=ncols,
            title=title, subtitle=subtitle, caption=caption, lang=lang,
        )
        return _finalize_figure(
            fig, "lcz_parameters_facet", None, None, None, isave, save_extension,
            style=style, lang=lang, dpi=dpi,
        )

    figures = []
    for name, arr, profile in panels:
        # Determine rendering strategy
        use_ds = (renderer == "datashader") or (
            renderer == "auto" and use_datashader and
            arr.shape[0] * arr.shape[1] > 4_000_000
        )

        # Create interactive figure
        fig = _plot_one_interactive(
            arr, profile, name, inclusive, use_datashader=use_ds
        )

        transform = profile.get("transform")
        if transform is not None:
            h, w = arr.shape
            bounds = (
                transform.c,
                transform.f + h * transform.e,
                transform.c + w * transform.a,
                transform.f,
            )
            add_map_furniture(
                fig, bounds=bounds, crs=profile.get("crs"),
                add_scalebar=add_scalebar, add_north_arrow=add_north_arrow,
            )

        figures.append(_finalize_figure(
            fig, name, title, subtitle, caption, isave, save_extension,
            style=style, lang=lang, dpi=dpi,
        ))

    return figures if len(figures) > 1 else figures[0]


__all__ = [
    "lcz_plot_parameters",
    "FastRasterReader",
    "read_with_warped_vrt",
    "create_geoarrow_metadata",
]
