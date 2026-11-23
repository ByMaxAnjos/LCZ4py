"""Native static (matplotlib/Agg) rendering backend for LCZ4py figures.

This module is the **default** rendering path for every figure-producing
function in the package. It draws with matplotlib's Agg canvas and writes
PNG/PDF/SVG/TIFF directly, so it never needs Plotly, Kaleido, Chromium, a
browser, or a network connection.

Plotly remains available as an opt-in renderer (``renderer="plotly"``); the
functions build their Plotly figure as before in that case.

Public helpers
--------------
``resolve_renderer``       normalise ``"static"``/``"plotly"``/``"maplibre"``
``new_figure``             figure sized from a journal style preset
``save_static``            validated, 300 dpi, atomic-ish save
``style_context``          matplotlib ``rc_context`` for a style preset
``plot_lcz_raster``        imshow with the LCZ class colormap
``plot_continuous_raster`` imshow with a continuous colormap
``apply_map_axes``         metric/geographic axes, limits, aspect
``add_map_furniture_mpl``  scale bar + north arrow
``class_legend``           ordered LCZ legend
``add_colorbar``           color bar with units
``add_header`` / ``add_footer``  title/subtitle and caption/attribution
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional

import numpy as np

from .lcz_parameters_data import (
    LCZ_COLORBLIND,
    LCZ_COLORS,
    get_lcz_names,
)
from .lcz_theme import _nice_scale_length, get_style

logger = logging.getLogger(__name__)

OUTPUT_DIR = "LCZ4r_output"
MM_PER_INCH = 25.4
MIN_STATIC_DPI = 300
STATIC_EXTENSIONS = {
    "png", "pdf", "svg", "tif", "tiff", "jpg", "jpeg", "webp",
}

_RENDERER_ALIASES = {
    "static": "static",
    "matplotlib": "static",
    "mpl": "static",
    "agg": "static",
    "native": "static",
    "plotly": "plotly",
    "interactive": "plotly",
    "maplibre": "maplibre",
    "mapbox": "maplibre",
    "basemap": "maplibre",
}


def resolve_renderer(renderer: Optional[str] = None) -> str:
    """Normalise a renderer name.

    ``None`` means the native static renderer (the package default). Aliases
    such as ``"matplotlib"``/``"agg"`` map to ``"static"``.
    """
    if renderer is None:
        return "static"
    key = str(renderer).strip().lower()
    try:
        return _RENDERER_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unknown renderer {renderer!r}; use 'static' (default), 'plotly' "
            "or 'maplibre'."
        ) from None


def is_static(renderer: Optional[str]) -> bool:
    """True when ``renderer`` resolves to the native static backend."""
    return resolve_renderer(renderer) == "static"


def validate_extension(save_extension: str, renderer: str = "static") -> str:
    """Return a usable file extension, coercing HTML for the static backend."""
    ext = (save_extension or "png").strip().lower().lstrip(".")
    if ext == "html":
        if renderer == "static":
            logger.info("Static renderer cannot write HTML; saving PNG instead.")
            return "png"
        return "html"
    if ext not in STATIC_EXTENSIONS:
        raise ValueError(
            f"Unsupported static save_extension {save_extension!r}; use "
            f"{', '.join(sorted(STATIC_EXTENSIONS))}."
        )
    return ext


@contextmanager
def style_context(style: str = "default"):
    """Context manager applying a journal style preset's matplotlib rcParams."""
    import matplotlib.pyplot as plt

    spec = get_style(style)
    with plt.rc_context(spec.mpl_rcparams):
        yield spec


def target_dpi(style: str = "default", dpi: Optional[int] = None) -> int:
    """Effective export DPI: never below 300 for static output."""
    spec = get_style(style)
    return max(int(dpi) if dpi else 0, MIN_STATIC_DPI, int(spec.dpi))


def figure_dims(
    style: str = "default",
    *,
    width_mm: Optional[float] = None,
    height_mm: Optional[float] = None,
    aspect: Optional[float] = None,
    extra_height_mm: float = 0.0,
    column: Optional[str] = None,
):
    """Return ``(spec, width_mm, height_mm)`` for a static figure.

    ``aspect`` is height/width of the data area; ``extra_height_mm`` reserves
    room for a legend/caption. Height is capped at the style's max height.
    """
    spec = get_style(style)
    if width_mm is None:
        if column and column in spec.width_mm:
            width_mm = float(spec.width_mm[column])
        else:
            width_mm = float(spec.width_mm.get("single", 120))
    if height_mm is not None:
        height_mm = float(height_mm)
    elif aspect is not None:
        height_mm = float(width_mm) * float(aspect)
    else:
        height_mm = float(width_mm) * 0.62
    height_mm = min(float(height_mm) + float(extra_height_mm), float(spec.max_height_mm))
    return spec, float(width_mm), float(height_mm)


def new_figure(
    style: str = "default",
    *,
    width_mm: Optional[float] = None,
    height_mm: Optional[float] = None,
    aspect: Optional[float] = None,
    extra_height_mm: float = 0.0,
    column: Optional[str] = None,
    dpi: Optional[int] = None,
):
    """Create a white matplotlib figure sized from a style preset.

    Returns ``(fig, spec, dpi)``. Callers should build and save inside
    :func:`style_context` so fonts/rcParams are applied at draw time.
    """
    import matplotlib.pyplot as plt

    spec, w_mm, h_mm = figure_dims(
        style, width_mm=width_mm, height_mm=height_mm, aspect=aspect,
        extra_height_mm=extra_height_mm, column=column,
    )
    dpi = target_dpi(style, dpi)
    fig = plt.figure(
        figsize=(w_mm / MM_PER_INCH, h_mm / MM_PER_INCH),
        dpi=dpi,
        facecolor="white",
    )
    return fig, spec, dpi


def add_axes(
    fig,
    *,
    bottom: float = 0.14,
    top: float = 0.94,
    left: float = 0.10,
    right: float = 0.95,
    **kwargs,
):
    """Add a full-width axes with explicit fractional margins."""
    ax = fig.add_axes([left, bottom, max(right - left, 0.05), max(top - bottom, 0.05)])
    if kwargs:
        ax.set(**kwargs)
    return ax


def save_static(
    fig,
    filename: str,
    *,
    isave: bool = False,
    save_extension: str = "png",
    style: str = "default",
    dpi: Optional[int] = None,
    lang: str = "en",
) -> Optional[str]:
    """Save a matplotlib figure at >=300 dpi; return the path or ``None``."""
    if not isave:
        return None
    ext = validate_extension(save_extension, "static")
    out_dpi = target_dpi(style, dpi)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{filename}.{ext}")
    fig.savefig(
        path,
        dpi=out_dpi,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05,
    )
    logger.info("Saved: %s", os.path.abspath(path))
    return path


# ── Colormaps ────────────────────────────────────────────────────────────────

def mpl_cmap(name: str = "viridis"):
    """Resolve a Plotly/Matplotlib colormap name to a matplotlib colormap."""
    import matplotlib

    key = (name or "viridis").strip()
    try:
        cmap = matplotlib.colormaps[key]
    except KeyError:
        aliases = {
            "rdbu_r": "RdBu_r", "rdbu": "RdBu", "viridis": "viridis",
            "cividis": "cividis", "blues": "Blues", "greens": "Greens",
            "plasma": "plasma", "inferno": "inferno", "magma": "magma",
            "turbo": "turbo", "ylgnbu": "YlGnBu", "spectral": "Spectral",
            "rdylbu": "RdYlBu", "rdylbu_r": "RdYlBu_r", "coolwarm": "coolwarm",
            "greys": "Greys", "ylorrd": "YlOrRd", "ylgn": "YlGn",
        }
        cmap = matplotlib.colormaps.get(aliases.get(key.lower(), "viridis"),
                                        matplotlib.colormaps["viridis"])
    cmap = cmap.copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    return cmap


def lcz_class_cmap(inclusive: bool = False):
    """Return ``(cmap, norm)`` for LCZ classes 1-17 with transparent nodata."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    cmap = ListedColormap(list(colors), name="lcz")
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    norm = BoundaryNorm(np.arange(0.5, 18.5, 1.0), cmap.N)
    return cmap, norm


def plot_lcz_raster(
    ax,
    arr: np.ndarray,
    extent: tuple,
    *,
    inclusive: bool = False,
    interpolation: str = "nearest",
):
    """Draw an LCZ class raster on ``ax`` (row 0 north) and return the image."""
    cmap, norm = lcz_class_cmap(inclusive)
    masked = np.ma.masked_where((arr < 1) | (arr > 17), arr)
    return ax.imshow(
        masked,
        extent=(extent[0], extent[1], extent[2], extent[3]),
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation=interpolation,
        aspect="auto",
    )


def plot_continuous_raster(
    ax,
    arr: np.ndarray,
    extent: tuple,
    *,
    colorscale: str = "RdBu_r",
    interpolation: str = "nearest",
):
    """Draw a continuous raster on ``ax`` and return the image."""
    return ax.imshow(
        arr,
        extent=(extent[0], extent[1], extent[2], extent[3]),
        origin="upper",
        cmap=mpl_cmap(colorscale),
        interpolation=interpolation,
        aspect="auto",
    )


# ── Map furniture / axes ────────────────────────────────────────────────────

def apply_map_axes(ax, extent: tuple, crs=None) -> bool:
    """Format axes for a metric/geographic map; return ``is_geographic``."""
    from matplotlib.ticker import FuncFormatter

    xmin, xmax, ymin, ymax = extent
    is_geo = bool(getattr(crs, "is_geographic", False))
    if is_geo:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        fmt = FuncFormatter(lambda v, _: f"{v / 1000:.1f}")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel("Easting [km]")
        ax.set_ylabel("Northing [km]")
    ax.tick_params(labelsize=7, length=3, width=0.6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.grid(False)
    return is_geo


def metric_width(extent: tuple, crs=None) -> float:
    """Map width in metres, converting feet-based projected CRSs."""
    factor = 1.0
    units = (getattr(crs, "linear_units", "") or "").lower()
    if "us survey" in units:
        factor = 1200.0 / 3937.0
    elif "foot" in units or "feet" in units:
        factor = 0.3048
    return (float(extent[1]) - float(extent[0])) * factor


def _draw_scalebar(ax, map_width_m: float, is_geographic: bool) -> None:
    """Two-tone scale bar in axes coordinates (metric maps only)."""
    if is_geographic or map_width_m <= 0:
        return
    bar_m = _nice_scale_length(map_width_m)
    span = min(bar_m / map_width_m, 0.40)
    half = span / 2.0
    x0, y0 = 0.03, 0.055
    ax.plot([x0, x0 + half], [y0, y0], transform=ax.transAxes,
            color="black", lw=6, solid_capstyle="butt", zorder=20)
    ax.plot([x0 + half, x0 + span], [y0, y0], transform=ax.transAxes,
            color="white", lw=6, solid_capstyle="butt", zorder=20)
    ax.plot([x0, x0 + span], [y0, y0], transform=ax.transAxes,
            color="black", lw=0.7, zorder=21)
    label = f"{bar_m / 1000:.0f} km" if bar_m >= 1000 else f"{bar_m:.0f} m"
    ax.text(
        x0 + half, y0 + 0.018, label, transform=ax.transAxes,
        ha="center", va="bottom", fontsize=7, zorder=22,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
    )


def _draw_north_arrow(ax) -> None:
    """North arrow in the upper-right corner."""
    ax.annotate(
        "N", xy=(0.95, 0.93), xytext=(0.95, 0.81), xycoords="axes fraction",
        ha="center", va="center", fontsize=9, fontweight="bold", zorder=22,
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5,
                        shrinkA=0, shrinkB=0),
    )


# Public aliases (the underscore names avoid shadowing inside the composite
# helper below, where the boolean flags share the public names).
add_scalebar = _draw_scalebar
add_north_arrow = _draw_north_arrow


def add_map_furniture_mpl(
    ax,
    extent: tuple,
    crs=None,
    *,
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
) -> bool:
    """Axis labels + scale bar + north arrow; return ``is_geographic``."""
    is_geo = apply_map_axes(ax, extent, crs)
    if add_scalebar:
        _draw_scalebar(ax, metric_width(extent, crs), is_geo)
    if add_north_arrow and not is_geo:
        _draw_north_arrow(ax)
    return is_geo


# ── Legends / colorbars / text ──────────────────────────────────────────────

def class_legend(
    fig,
    arr: np.ndarray,
    *,
    inclusive: bool = False,
    lang: str = "en",
    ncol: int = 6,
    fontsize: float = 7,
    bbox_to_anchor: tuple = (0.5, 0.02),
    title: Optional[str] = None,
):
    """Ordered LCZ class legend below the map; returns the legend or ``None``."""
    from matplotlib.patches import Patch

    valid = arr[(arr >= 1) & (arr <= 17)]
    if valid.size == 0:
        return None
    present = [int(v) for v in np.unique(valid)]
    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    names = get_lcz_names(lang)
    handles = [
        Patch(facecolor=colors[c - 1], edgecolor="black", linewidth=0.3,
              label=f"{c}: {names[c - 1]}")
        for c in present
    ]
    return fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=min(int(ncol), max(2, len(handles))),
        frameon=False,
        fontsize=fontsize,
        handlelength=1.1,
        columnspacing=1.0,
        handletextpad=0.5,
        title=title,
        title_fontsize=fontsize,
    )


def add_colorbar(
    fig,
    ax,
    image,
    *,
    units: str = "",
    fontsize: float = 7,
    **kwargs,
):
    """Attach a vertical color bar with a units-bearing label."""
    cb = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, shrink=0.92, **kwargs)
    cb.set_label(units or "Value", fontsize=fontsize)
    cb.ax.tick_params(labelsize=max(6, fontsize - 1))
    cb.outline.set_linewidth(0.6)
    return cb


def add_header(
    fig,
    title: Optional[str],
    subtitle: Optional[str],
    spec,
    *,
    y: float = 0.965,
):
    """Title/subtitle above the axes using the style's font sizes."""
    if title:
        fig.text(
            0.5, y, title, ha="center", va="top",
            fontsize=spec.font_size + 2, fontweight="bold",
        )
    if subtitle:
        offset = 0.035 if title else 0.0
        fig.text(
            0.5, y - offset, subtitle, ha="center", va="top",
            fontsize=spec.font_size, style="italic",
        )


def add_footer(
    fig,
    caption: Optional[str],
    attribution: Optional[str],
    spec,
    *,
    y: float = 0.012,
):
    """Caption (left) and credit/attribution (right) below the axes."""
    small = max(5.5, spec.font_size - 1.5)
    if caption:
        fig.text(0.01, y, caption, ha="left", va="bottom",
                 fontsize=small, color="#333333")
    if attribution:
        fig.text(0.99, y, attribution, ha="right", va="bottom",
                 fontsize=small, color="#444444")


def figure_top(has_header: bool) -> float:
    """Fractional top for the axes given whether a header is drawn."""
    return 0.88 if has_header else 0.94


def figure_bottom(has_legend: bool) -> float:
    """Fractional bottom for the axes given whether a legend is drawn."""
    return 0.26 if has_legend else 0.14


__all__ = [
    "MIN_STATIC_DPI",
    "STATIC_EXTENSIONS",
    "resolve_renderer",
    "is_static",
    "validate_extension",
    "style_context",
    "target_dpi",
    "figure_dims",
    "new_figure",
    "add_axes",
    "save_static",
    "mpl_cmap",
    "lcz_class_cmap",
    "plot_lcz_raster",
    "plot_continuous_raster",
    "apply_map_axes",
    "metric_width",
    "add_scalebar",
    "add_north_arrow",
    "add_map_furniture_mpl",
    "class_legend",
    "add_colorbar",
    "add_header",
    "add_footer",
    "figure_top",
    "figure_bottom",
]
