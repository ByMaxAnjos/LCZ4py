"""Shared publication-style presets and figure export/finalization helpers.

Every ``lcz_plot_*``/``lcz_cal_*`` function delegates its ``isave``/
``save_extension`` handling to :func:`finalize_export` instead of duplicating
its own ``write_html``/``write_image``/``savefig`` block. ``style=`` picks a
:class:`StyleSpec` preset (font, size in mm, DPI, palette) so exported
figures match a target journal's author guidelines.

Presets are backed by each journal's published figure-preparation guide, not
guessed:

* Nature — https://www.nature.com/nature/for-authors/formatting-guide
  (89 mm single column / 183 mm double column, max height 170 mm,
  Helvetica/Arial, 5-7 pt body text)
* Science — https://spj.science.org/pb-assets/SPJ/CustomPages/Misc/SPJ_Figure_Preparation_Guide-1691522222.pdf
  (86 mm / 121 mm / 184 mm columns, Helvetica, ~7 pt text, 5 pt minimum)

``"default"`` reproduces today's look (unchanged, backward-compat anchor)
and ``"generic_bw"`` is a neutral, colorblind/grayscale-safe fallback for
journals without a specific house style.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from .i18n_messages import lcz_msg
from .lcz_parameters_data import LCZ_COLORS, LCZ_COLORBLIND

logger = logging.getLogger(__name__)

OUTPUT_DIR = "LCZ4r_output"
MM_PER_INCH = 25.4

# ── MetBrewer palettes as Plotly colorscales ──────────────────────────────────
# Relocated from lcz_plot_parameters.py — this is a style concern, not a
# parameters-plot concern. Re-exported so existing importers keep working.

METBREWER_PALETTES: dict[str, list[str]] = {
    "Archambault": ["#88a0dc", "#381a61", "#7c4b73", "#ed68ed", "#fcffa4"],
    "Greek":       ["#3b1c3a", "#6a2263", "#9b2c77", "#d0577e", "#ed8a87"],
    "VanGogh1":    ["#1c3a5e", "#4a6f9a", "#88aac2", "#cde0e8", "#f7f3e7"],
    "VanGogh2":    ["#2a3f5c", "#566985", "#8e9cad", "#c3cbd2", "#f0e9d6"],
    "VanGogh3":    ["#3a4a4a", "#6c7b7b", "#9ea9a9", "#cdd2cd", "#f0e8d8"],
    "Hokusai2":    ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261"],
    "Hokusai3":    ["#1d3557", "#457b9d", "#a8dadc", "#f1faee", "#e63946"],
    "Pissarro":    ["#2c4e6e", "#5b7c99", "#a3b9c4", "#e0dccd", "#f1e9d2"],
    "Tam":         ["#40004b", "#762a83", "#9970ab", "#c2a5cf", "#e7d4e8"],
    "Renoir":      ["#2c1d3a", "#5a3d5e", "#8d5b8a", "#c98bb0", "#fcd5ce"],
    "Manet":       ["#2c2c4a", "#5b5b75", "#8c8ca1", "#bcbccf", "#ece5d4"],
    "Demuth":      ["#2d2d3e", "#5b5b6e", "#8d8d9a", "#bdbdc1", "#e8e0d2"],
    "Troy":        ["#3a2f4a", "#665375", "#9582a1", "#c4b9cc", "#e8e0d2"],
    "Ingres":      ["#1c1c2e", "#3a3a55", "#5b5b75", "#8888a0", "#bcbccf"],
    "Cassatt1":    ["#a37b6a", "#c89e88", "#e1c1a8", "#ecd5b5", "#f0d5a8"],
    "Cassatt2":    ["#3d405b", "#5d6b87", "#9bb1c8", "#cbd5dc", "#ece5d4"],
}


@dataclass(frozen=True)
class StyleSpec:
    """A publication style preset."""

    name: str
    font_family: str
    font_size: int
    width_mm: dict[str, float]
    max_height_mm: float
    dpi: int
    colorway: list[str]
    template: go.layout.Template
    mpl_rcparams: dict = field(default_factory=dict)


def _make_template(font_family: str, font_size: int, colorway: list[str]) -> go.layout.Template:
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=font_family, size=font_size, color="black"),
            colorway=colorway,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showline=True, linecolor="black", linewidth=1, mirror=False,
                       ticks="outside", showgrid=False, zeroline=False),
            yaxis=dict(showline=True, linecolor="black", linewidth=1, mirror=False,
                       ticks="outside", showgrid=False, zeroline=False),
            margin=dict(l=60, r=20, t=50, b=50),
        )
    )


_DEFAULT_MPL_RC = {}

_JOURNAL_MPL_RC = {
    "font.family": "sans-serif",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
}


def _build_presets() -> dict[str, StyleSpec]:
    default_colorway = LCZ_COLORS
    return {
        # Today's look, unchanged — the backward-compat anchor.
        "default": StyleSpec(
            name="default",
            font_family="Arial, sans-serif",
            font_size=12,
            width_mm={"single": 120, "double": 250},
            max_height_mm=200,
            dpi=150,
            colorway=default_colorway,
            template=go.layout.Template(layout=go.Layout(template="plotly_white")),
            mpl_rcparams=_DEFAULT_MPL_RC,
        ),
        "nature": StyleSpec(
            name="nature",
            font_family="Arial, Helvetica, sans-serif",
            font_size=7,
            width_mm={"single": 89, "double": 183, "1.5col": 136},
            max_height_mm=170,
            dpi=300,
            colorway=default_colorway,
            template=_make_template("Arial, Helvetica, sans-serif", 7, default_colorway),
            mpl_rcparams={**_JOURNAL_MPL_RC, "font.sans-serif": ["Arial", "Helvetica"],
                          "font.size": 7},
        ),
        "science": StyleSpec(
            name="science",
            font_family="Helvetica, Arial, sans-serif",
            font_size=7,
            width_mm={"single": 86, "double": 121, "3col": 184},
            max_height_mm=230,
            dpi=300,
            colorway=default_colorway,
            template=_make_template("Helvetica, Arial, sans-serif", 7, default_colorway),
            mpl_rcparams={**_JOURNAL_MPL_RC, "font.sans-serif": ["Helvetica", "Arial"],
                          "font.size": 7},
        ),
        # Neutral, colorblind/grayscale-safe fallback for journals without a
        # specific house style.
        "generic_bw": StyleSpec(
            name="generic_bw",
            font_family="Arial, Helvetica, sans-serif",
            font_size=8,
            width_mm={"single": 90, "double": 180},
            max_height_mm=200,
            dpi=300,
            colorway=LCZ_COLORBLIND,
            template=_make_template("Arial, Helvetica, sans-serif", 8, LCZ_COLORBLIND),
            mpl_rcparams={**_JOURNAL_MPL_RC, "font.sans-serif": ["Arial", "Helvetica"],
                          "font.size": 8},
        ),
    }


_PRESETS: dict[str, StyleSpec] = _build_presets()


def get_style(style: str, lang: str = "en") -> StyleSpec:
    """Look up a :class:`StyleSpec` preset by name.

    Raises
    ------
    ValueError
        If ``style`` is not a known preset name.
    """
    spec = _PRESETS.get(style)
    if spec is None:
        raise ValueError(
            lcz_msg("invalid_style", lang, style=style, valid=", ".join(_PRESETS))
        )
    return spec


def _mm_to_px(mm: float, dpi: int) -> int:
    return round(mm / MM_PER_INCH * dpi)


def finalize_export(
    fig,
    *,
    style: str = "default",
    isave: bool = False,
    save_extension: str = "html",
    filename: str = "lcz_plot",
    column: str = "single",
    lang: str = "en",
):
    """Apply a publication style template to ``fig`` and optionally save it.

    Works for both Plotly ``go.Figure`` objects and matplotlib ``Figure``
    objects (the latter for ``plot_grid_only.py``, the only matplotlib-based
    plot in the package).
    """
    spec = get_style(style, lang)

    if isinstance(fig, go.Figure):
        fig.update_layout(template=spec.template)
        if isave:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            save_path = os.path.join(OUTPUT_DIR, f"{filename}.{save_extension}")
            if save_extension == "html":
                fig.write_html(save_path, include_plotlyjs="cdn")
            else:
                width_mm = spec.width_mm.get(column, spec.width_mm["single"])
                width_px = _mm_to_px(width_mm, spec.dpi)
                height_px = _mm_to_px(min(spec.max_height_mm, width_mm * 0.75), spec.dpi)
                if save_extension == "tiff":
                    png_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
                    fig.write_image(png_path, width=width_px, height=height_px,
                                     scale=1)
                    from PIL import Image
                    Image.open(png_path).save(save_path, dpi=(spec.dpi, spec.dpi))
                else:
                    fig.write_image(save_path, width=width_px, height=height_px,
                                     scale=1)
            logger.info("Saved: %s", os.path.abspath(save_path))
        return fig

    # matplotlib path
    import matplotlib.pyplot as plt

    with plt.rc_context(spec.mpl_rcparams):
        fig.canvas.draw()
        if isave:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            save_path = os.path.join(OUTPUT_DIR, f"{filename}.{save_extension}")
            fig.savefig(save_path, dpi=spec.dpi, bbox_inches="tight")
            logger.info("Saved: %s", os.path.abspath(save_path))
    return fig


def _nice_scale_length(map_width_m: float) -> float:
    """Pick a round-number ground distance for a scale bar (~1/5 of map width)."""
    target = map_width_m / 5
    if target <= 0:
        return 1.0
    magnitude = 10 ** int(np.floor(np.log10(target))) if target >= 1 else 1
    for mult in (1, 2, 5, 10):
        candidate = mult * magnitude
        if candidate >= target:
            return float(candidate)
    return float(10 * magnitude)


def add_map_furniture(
    fig: go.Figure,
    bounds: tuple[float, float, float, float],
    crs=None,
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
) -> go.Figure:
    """Add a scale bar and north arrow to a Plotly map figure, in place.

    Parameters
    ----------
    fig : go.Figure
        Map figure to annotate (paper-relative shapes/annotations are added).
    bounds : (xmin, ymin, xmax, ymax)
        Map extent in projected, metric units (e.g. EPSG:3857/UTM). If the
        CRS is geographic (degrees), the scale bar is skipped since ground
        distance cannot be computed without reprojection.
    crs : optional
        CRS of ``bounds`` (pyproj-compatible or rasterio CRS). Used only to
        detect geographic (lon/lat) CRSs so the north arrow/scale bar are
        skipped when not meaningful.
    """
    xmin, ymin, xmax, ymax = bounds
    is_geographic = False
    if crs is not None:
        try:
            is_geographic = bool(getattr(crs, "is_geographic", False))
        except Exception:
            is_geographic = False

    if add_scalebar and not is_geographic:
        map_width_m = xmax - xmin
        if map_width_m > 0:
            bar_m = _nice_scale_length(map_width_m)
            bar_frac = min(bar_m / map_width_m, 0.3)
            label = f"{bar_m/1000:.0f} km" if bar_m >= 1000 else f"{bar_m:.0f} m"
            x0, x1, y0 = 0.03, 0.03 + bar_frac, 0.04
            fig.add_shape(type="line", xref="paper", yref="paper",
                          x0=x0, x1=x1, y0=y0, y1=y0,
                          line=dict(color="black", width=3))
            fig.add_annotation(xref="paper", yref="paper",
                                x=(x0 + x1) / 2, y=y0 + 0.02,
                                text=label, showarrow=False,
                                font=dict(size=10, color="black"))
    elif add_scalebar and is_geographic:
        logger.debug("Skipping scale bar: geographic (lon/lat) CRS, reproject to a metric CRS first.")

    if add_north_arrow and not is_geographic:
        # ax/ay default to "pixel" ref (relative to x/y) — "paper" is not a
        # valid axref/ayref value, so the arrow tail is offset in pixels.
        fig.add_annotation(xref="paper", yref="paper",
                            x=0.96, y=0.90, ax=0, ay=40,
                            text="N", showarrow=True, arrowhead=2,
                            arrowsize=1.2, arrowwidth=2, arrowcolor="black",
                            font=dict(size=12, color="black"))
    elif add_north_arrow and is_geographic:
        logger.debug("Skipping north arrow: not a standard north-up projected CRS.")

    return fig
