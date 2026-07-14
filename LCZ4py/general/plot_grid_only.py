"""Plot a spatial grid/mesh (vector polygons or points) over a basemap,
optionally colored by an attribute.

New function for this repo (no direct R equivalent) — a lightweight
counterpart to lcz_plot_map.py for visualizing the raw grid/mesh geometry
itself (e.g. municipality boundaries, GHAP/ERA5 grid cells, LCZ zone
polygons) rather than a rendered raster.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt

try:
    import contextily as cx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

OUTPUT_DIR = "LCZ4r_output"


def plot_grid_only(
    grid: "gpd.GeoDataFrame",
    color_by: Optional[str] = None,
    cmap: str = "viridis",
    edgecolor: str = "black",
    linewidth: float = 0.3,
    alpha: float = 0.7,
    add_basemap: bool = True,
    basemap_source: Optional[object] = None,
    figsize: tuple[float, float] = (10, 10),
    title: Optional[str] = None,
    isave: bool = False,
    save_extension: str = "png",
    lang: str = "en",
) -> "plt.Figure":
    """Plot a grid/mesh GeoDataFrame over a contextily basemap.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        Vector grid/mesh to plot (polygons or points) — municipality
        boundaries, GHAP/ERA5/CHIRPS grid cells, LCZ zone polygons, etc.
    color_by : str, optional
        Column name to color features by. None (default) plots a single
        uncolored fill.
    cmap : str
        Matplotlib colormap for ``color_by``. Default "viridis".
    edgecolor : str
        Polygon outline color. Default "black".
    linewidth : float
        Polygon outline width. Default 0.3.
    alpha : float
        Fill opacity. Default 0.7.
    add_basemap : bool
        Overlay an OpenStreetMap basemap via contextily. Default True.
        Silently skipped if contextily is not installed.
    basemap_source : contextily provider, optional
        Custom tile source (e.g. ``contextily.providers.CartoDB.Positron``).
        Defaults to ``contextily.providers.OpenStreetMap.Mapnik``.
    figsize : tuple[float, float]
        Figure size in inches. Default (10, 10).
    title : str, optional
        Plot title.
    isave : bool
        Save the figure to ``LCZ4r_output/``. Default False.
    save_extension : str
        File extension when ``isave``. Default "png".
    lang : str
        Message language. Default "en".

    Returns
    -------
    matplotlib.figure.Figure
    """
    if color_by is not None and color_by not in grid.columns:
        raise ValueError(lcz_msg("plot_grid_bad_color_by", lang, col=color_by, avail=", ".join(grid.columns)))

    fig, ax = plt.subplots(figsize=figsize)

    grid_3857 = grid.to_crs(3857)
    plot_kwargs = dict(ax=ax, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
    if color_by is not None:
        grid_3857.plot(column=color_by, cmap=cmap, legend=True, **plot_kwargs)
    else:
        grid_3857.plot(facecolor="none" if grid_3857.geom_type.iloc[0] != "Point" else "steelblue", **plot_kwargs)

    if add_basemap:
        if HAS_CONTEXTILY:
            source = basemap_source or cx.providers.OpenStreetMap.Mapnik
            cx.add_basemap(ax, source=source, crs=grid_3857.crs)
        else:
            logger.warning(lcz_msg("plot_grid_no_contextily", lang))

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"lcz_plot_grid.{save_extension}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", os.path.abspath(out_path))

    return fig
