"""Visualize the relationship between LCZ classes and a gridded environmental
variable (ERA5 temperature, CHIRPS rainfall, pollution, drought indices...)
for the same city/region.

New function for this repo (no direct R equivalent) — climasus4r's
sus_grid_*() functions aggregate to whole municipalities, but the LCZ side
of this codebase (lcz_get_map, lcz_cal_area) works per-pixel on a single
city's raster. The natural join between the two is therefore in RASTER
SPACE: sample the environmental raster at each LCZ pixel and group by class
(the same zonal-stats-per-LCZ-class pattern lcz_cal_area already uses for
area). ``lcz_cal_indexes`` computes the same per-class summary as a table
instead of a plot; the ``lcz_grid_*`` downloaders already return data
pre-cropped to the LCZ grid, so no separate join step is needed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from LCZ4py._internal.lcz_parameters_data import LCZ_COLORS, LCZ_IDS, get_lcz_names
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export

logger = logging.getLogger(__name__)

OUTPUT_DIR = "LCZ4r_output"


def _read_band(path: Union[str, Path]) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(1).astype("float64"), src.profile


def _align_to_lcz(lcz_profile: dict, variable_path: Union[str, Path]) -> np.ndarray:
    """Reproject/resample the variable raster onto the LCZ raster's grid."""
    with rasterio.open(variable_path) as src:
        if (src.crs == lcz_profile["crs"] and src.transform == lcz_profile["transform"]
                and src.width == lcz_profile["width"] and src.height == lcz_profile["height"]):
            return src.read(1).astype("float64")

        dst = np.full((lcz_profile["height"], lcz_profile["width"]), np.nan, dtype="float64")
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=lcz_profile["transform"], dst_crs=lcz_profile["crs"],
            resampling=Resampling.average,
        )
        return dst


def plot_lcz_relationship(
    lcz_path: Union[str, Path],
    variable_path: Union[str, Path],
    variable_name: str = "value",
    plot_type: Literal["box", "violin", "scatter", "heatmap"] = "box",
    agg_fun: str = "mean",
    isave: bool = False,
    save_extension: str = "html",
    style: str = "default",
    lang: str = "en",
) -> go.Figure:
    """Plot the distribution of a gridded variable across LCZ classes.

    Parameters
    ----------
    lcz_path : str or Path
        Path to an LCZ classification GeoTIFF (values 1-17), e.g. from
        ``lcz_get_map``.
    variable_path : str or Path
        Path to a single-band raster of the environmental variable (e.g. an
        ERA5/CHIRPS/pollution GeoTIFF for the same region/date). Reprojected
        and resampled onto the LCZ raster's grid if its CRS/resolution differ.
    variable_name : str
        Label for the variable, used as the value-axis title. Default "value".
    plot_type : {"box", "violin", "scatter", "heatmap"}
        "box"/"violin": distribution of variable values per LCZ class.
        "scatter": per-pixel value vs. LCZ class (jittered).
        "heatmap": mean variable value per LCZ class as a single-row heatmap.
    agg_fun : str
        Aggregation used for "heatmap": "mean" (default), "median", "sum".
    isave : bool
        Save the figure to ``LCZ4r_output/``. Default False.
    save_extension : str
        File extension when ``isave``. Default "html".
    style : str
        Publication style preset: 'default', 'nature', 'science', or
        'generic_bw'. Controls font, figure size (mm), DPI, and palette
        used when isave and save_extension != 'html'.
    lang : str
        Message language. Default "en".

    Returns
    -------
    go.Figure
    """
    lcz_arr, lcz_profile = _read_band(lcz_path)
    var_arr = _align_to_lcz(lcz_profile, variable_path)

    valid = np.isin(lcz_arr, LCZ_IDS) & ~np.isnan(var_arr)
    if not valid.any():
        raise ValueError(lcz_msg("plot_lcz_rel_no_overlap", lang))

    lcz_codes = lcz_arr[valid].astype(int)
    values = var_arr[valid]
    names = get_lcz_names(lang)
    df = pd.DataFrame({
        "lcz_id": lcz_codes,
        "lcz_name": [names[c - 1] for c in lcz_codes],
        variable_name: values,
    })

    color_map = {names[i]: LCZ_COLORS[i] for i in range(len(names))}
    order = [n for n in names if n in df["lcz_name"].unique()]

    if plot_type in ("box", "violin"):
        plot_fn = px.box if plot_type == "box" else px.violin
        fig = plot_fn(df, x="lcz_name", y=variable_name, color="lcz_name",
                      color_discrete_map=color_map, category_orders={"lcz_name": order})
        fig.update_layout(xaxis_title="LCZ class", yaxis_title=variable_name, showlegend=False)
    elif plot_type == "scatter":
        fig = px.strip(df, x="lcz_name", y=variable_name, color="lcz_name",
                        color_discrete_map=color_map, category_orders={"lcz_name": order})
        fig.update_layout(xaxis_title="LCZ class", yaxis_title=variable_name, showlegend=False)
    elif plot_type == "heatmap":
        summary = df.groupby("lcz_name")[variable_name].agg(agg_fun).reindex(order)
        fig = go.Figure(go.Heatmap(z=[summary.values], x=summary.index.tolist(),
                                    colorscale="RdYlBu_r", colorbar=dict(title=variable_name)))
        fig.update_layout(xaxis_title="LCZ class", yaxis=dict(showticklabels=False))
    else:
        raise ValueError(lcz_msg("plot_lcz_rel_invalid_type", lang, valid="box, violin, scatter, heatmap"))

    fig.update_layout(title=f"{variable_name} by LCZ class")

    fig = finalize_export(fig, style=style, isave=isave, save_extension=save_extension,
                           filename=f"lcz_relationship_{plot_type}", lang=lang)

    return fig
