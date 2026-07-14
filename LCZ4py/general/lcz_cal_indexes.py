"""Summarize a gridded climate/environmental variable by LCZ class.

The per-pixel counterpart to ``lcz_cal_area`` (which summarizes *area* per
LCZ class): here the input is a variable already cropped/reprojected onto
an LCZ map's grid — e.g. the ``array`` from ``lcz_grid_chirps``,
``lcz_grid_era5``, ``lcz_grid_pdsi``, ``lcz_grid_pollution_ghap``, or
``lcz_grid_pollution_merra2`` — and the output is one row of summary
statistics (n_pixels, mean, std, min, max, median) per LCZ class.

``plot_lcz_relationship`` is the per-pixel distribution counterpart
(box/violin/scatter/heatmap of every valid pixel); this function is the
aggregated summary-table counterpart, following ``lcz_cal_area``'s
Polars group-by pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import rasterio
import plotly.graph_objects as go

from LCZ4py._internal.lcz_parameters_data import LCZ_COLORS, LCZ_COLORBLIND, get_lcz_names
from LCZ4py._internal._lcz_grid_raster_base import LCZGridResult
from LCZ4py._internal.i18n_messages import lcz_msg

OUTPUT_DIR = "LCZ4r_output"


@dataclass
class LCZIndexesResult:
    """Return type of lcz_cal_indexes."""
    df: pl.DataFrame
    fig: Optional[go.Figure] = None


def _select_band(grid_result: LCZGridResult, band: Optional[Union[int, str]]) -> np.ndarray:
    array = grid_result.array
    if band is None:
        return array  # (n_bands, H, W) -> flattened across all bands below
    if isinstance(band, str):
        if band not in grid_result.bands:
            raise ValueError(f"band {band!r} not found. Available: {grid_result.bands}")
        band = grid_result.bands.index(band)
    return array[band][None, ...]


def _build_summary(classes: np.ndarray, values: np.ndarray) -> pl.DataFrame:
    df = pl.DataFrame({"lcz": classes.astype(np.int16), "value": values.astype(np.float64)}).lazy()
    return (
        df.group_by("lcz")
        .agg([
            pl.len().alias("n_pixels"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().alias("std"),
            pl.col("value").min().alias("min"),
            pl.col("value").max().alias("max"),
            pl.col("value").median().alias("median"),
        ])
        .filter((pl.col("lcz") >= 1) & (pl.col("lcz") <= 17))
        .sort("lcz")
        .collect()
    )


def _attach_metadata(df: pl.DataFrame, inclusive: bool, lang: str) -> pl.DataFrame:
    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    names_df = pl.DataFrame({"lcz": list(range(1, 18)), "lcz_name": get_lcz_names(lang), "color": colors})
    return df.join(names_df, on="lcz", how="left")


def _plot_bar(df: pl.DataFrame, variable_name: str, lang: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["lcz"].cast(str),
        y=df["mean"],
        error_y=dict(type="data", array=df["std"].fill_null(0)),
        marker_color=df["color"].to_list(),
        marker_line_color="black",
        marker_line_width=0.5,
        customdata=df.select(["lcz_name", "n_pixels"]),
        hovertemplate=(
            "<b>LCZ %{x}</b><br>%{customdata[0]}<br>"
            f"{variable_name}: " + "%{y:.2f}<br>n pixels: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(text=lcz_msg("cal_indexes_title", lang, variable=variable_name), x=0.5, xanchor="center"),
        xaxis=dict(title="LCZ code", gridcolor="white"),
        yaxis=dict(title=variable_name, gridcolor="#f0f0f0"),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        margin=dict(l=80, r=40, t=60, b=60), width=1000, height=600,
    )
    return fig


def lcz_cal_indexes(
    x: Union[str, Path],
    grid_result: LCZGridResult,
    band: Optional[Union[int, str]] = None,
    variable_name: Optional[str] = None,
    inclusive: bool = False,
    iplot: bool = True,
    isave: bool = False,
    save_extension: str = "html",
    lang: str = "en",
) -> Union[LCZIndexesResult, pl.DataFrame]:
    """Summarize a gridded variable (already aligned to an LCZ map) by class.

    Parameters
    ----------
    x : str or Path
        Path to the LCZ map GeoTIFF the grid data was cropped to (e.g. the
        same ``x`` passed to ``lcz_grid_chirps``/``lcz_grid_era5``/etc.).
    grid_result : LCZGridResult
        The result of an ``lcz_grid_*`` raster call.
    band : int or str, optional
        Which band to summarize: an index, a label from ``grid_result.bands``,
        or None (default) to pool every band's valid pixels together.
    variable_name : str, optional
        Label for the variable. Defaults to ``grid_result.variables`` joined
        by "+", or "value".
    inclusive : bool
        Use the colorblind-friendly LCZ palette.
    iplot : bool
        If False, return only the summary DataFrame (no figure).
    isave : bool
        Save the figure + CSV to ``LCZ4r_output/``.
    save_extension : str
        "html" (default, interactive) or a static image extension.
    lang : str
        Message language. Default "en".

    Returns
    -------
    LCZIndexesResult or pl.DataFrame
        ``df`` has one row per LCZ class present: lcz, lcz_name, color,
        n_pixels, mean, std, min, max, median.
    """
    with rasterio.open(str(x)) as src:
        classes = src.read(1)

    values = _select_band(grid_result, band)
    n_bands = values.shape[0]
    tiled_classes = np.broadcast_to(classes, (n_bands,) + classes.shape)

    valid = (tiled_classes >= 1) & (tiled_classes <= 17) & np.isfinite(values)
    if not valid.any():
        raise ValueError(lcz_msg("cal_indexes_no_overlap", lang))

    df = _build_summary(tiled_classes[valid], values[valid])
    df = _attach_metadata(df, inclusive, lang)

    variable_name = variable_name or (
        "+".join(grid_result.variables) if grid_result.variables else "value"
    )

    if not iplot:
        return df

    fig = _plot_bar(df, variable_name, lang)

    if isave:
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.write_html(os.path.join(OUTPUT_DIR, f"lcz_cal_indexes.{save_extension}")) if save_extension == "html" \
            else fig.write_image(os.path.join(OUTPUT_DIR, f"lcz_cal_indexes.{save_extension}"))
        df.write_csv(os.path.join(OUTPUT_DIR, "lcz_cal_indexes.csv"))

    return LCZIndexesResult(df=df, fig=fig)
