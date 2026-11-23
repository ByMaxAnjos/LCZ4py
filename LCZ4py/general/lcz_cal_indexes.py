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

Rendering
---------
The **default** renderer is native static matplotlib/Agg
(``renderer="static"``, also ``None``/``"matplotlib"``/``"agg"``): a
publication-quality mean ± SD bar chart per LCZ class, saved at >=300 dpi
without Plotly, Kaleido, Chromium or a browser. ``renderer="plotly"`` keeps
the previous interactive figure.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
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
from LCZ4py._internal.lcz_theme import finalize_export
from LCZ4py._internal import lcz_static as _st

OUTPUT_DIR = "LCZ4r_output"


def _resolve_workers(n_items: int, max_workers: Optional[int] = None) -> int:
    """Bounded worker count: ``min(8, cpu_count, n_items)`` unless overridden."""
    if max_workers is not None:
        workers = int(max_workers)
    else:
        workers = min(8, os.cpu_count() or 4)
    return max(1, min(workers, max(1, n_items)))


@dataclass
class LCZIndexesResult:
    """Return type of lcz_cal_indexes.

    ``fig`` is a matplotlib ``Figure`` for the default static renderer and a
    Plotly ``go.Figure`` for ``renderer="plotly"``.
    """
    df: pl.DataFrame
    fig: Optional[object] = None
    renderer: str = "static"

    def show(self) -> None:
        """Display the figure in the active frontend (matplotlib or Plotly)."""
        if self.fig is None:
            return
        if type(self.fig).__module__.startswith("matplotlib"):
            import matplotlib.pyplot as plt

            plt.show()
        else:
            self.fig.show()


def _select_band(
    grid_result: LCZGridResult, band: Optional[Union[int, str]],
) -> np.ndarray:
    """Return a ``(1, H, W)`` or ``(n_bands, H, W)`` float array."""
    array = grid_result.array
    if array is None:
        raise ValueError(
            "grid_result.array is None — pass the LCZGridResult returned by an "
            "lcz_grid_* raster call (or set .array from its saved GeoTIFF)."
        )
    array = np.asarray(array)
    if array.ndim == 2:  # tolerate a single 2-D band
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(
            f"grid_result.array must be (n_bands, H, W); got shape {array.shape}."
        )

    if band is None:
        return array

    if isinstance(band, str):
        bands = list(grid_result.bands or [])
        if band not in bands:
            raise ValueError(f"band {band!r} not found. Available: {bands}")
        band = bands.index(band)

    try:
        idx = int(band)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"band must be an int or a label from {list(grid_result.bands or [])}, "
            f"got {band!r}."
        ) from exc
    if idx < 0 or idx >= array.shape[0]:
        raise IndexError(
            f"band index {idx} out of range for {array.shape[0]} band(s)."
        )
    return array[idx][None, ...]


def _build_summary(classes: np.ndarray, values: np.ndarray) -> pl.DataFrame:
    """Group valid (class, value) pairs and compute the summary statistics."""
    df = pl.DataFrame({
        "lcz": np.asarray(classes, dtype=np.int16),
        "value": np.asarray(values, dtype=np.float64),
    }).lazy()
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


def _axis_title(variable_name: str, units: Optional[str]) -> str:
    """``'rainfall_mm [mm]'`` unless the label already carries units."""
    if not units:
        return variable_name
    return f"{variable_name} [{units.strip('[]')}]"


def _plot_bar(
    df: pl.DataFrame, variable_name: str, lang: str, units: Optional[str] = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["lcz"].cast(str),
        y=df["mean"],
        error_y=dict(type="data", array=df["std"].fill_null(0), color="#444444", thickness=1),
        marker_color=df["color"].to_list(),
        marker_line_color="black",
        marker_line_width=0.5,
        customdata=df.select(["lcz_name", "n_pixels", "std"]),
        hovertemplate=(
            "<b>LCZ %{x}</b> — %{customdata[0]}<br>"
            f"mean: %{{y:.3f}}<br>"
            "SD: %{customdata[2]:.3f}<br>"
            "n pixels: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(
            text=lcz_msg("cal_indexes_title", lang, variable=variable_name),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title=dict(text=lcz_msg("lcz_code_label", lang)), gridcolor="white"),
        yaxis=dict(title=dict(text=_axis_title(variable_name, units)),
                   gridcolor="#f0f0f0", zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=90, r=40, t=70, b=60), width=1000, height=600,
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


def _plot_bar_static(
    df: pl.DataFrame,
    variable_name: str,
    lang: str,
    *,
    units: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    style: str = "default",
    dpi: Optional[int] = None,
):
    """Mean per LCZ class with a ±1 SD whisker, drawn with matplotlib/Agg."""
    with _st.style_context(style):
        fig, spec, _ = _st.new_figure(style, aspect=0.62, column="single", dpi=dpi)
        ax = _st.add_axes(
            fig, bottom=0.20, top=_st.figure_top(bool(title or subtitle)),
            left=0.15, right=0.97,
        )

        x = np.arange(df.height, dtype=float)
        mean = np.nan_to_num(df["mean"].to_numpy().astype(float), nan=0.0)
        std = np.nan_to_num(
            df["std"].fill_null(0.0).to_numpy().astype(float), nan=0.0,
        )
        ax.bar(
            x, mean, yerr=std, width=0.74,
            color=df["color"].to_list(), edgecolor="black", linewidth=0.45,
            error_kw=dict(elinewidth=0.6, capsize=1.8, ecolor="#333333"),
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in df["lcz"].to_list()])
        ax.set_xlabel(lcz_msg("lcz_code_label", lang))
        ax.set_ylabel(_axis_title(variable_name, units))
        ax.tick_params(axis="both", labelsize=max(6.0, spec.font_size - 2),
                       length=3, width=0.6)
        ax.grid(axis="y", color="#e8e8e8", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        _st.add_header(
            fig, title or lcz_msg("cal_indexes_title", lang, variable=variable_name),
            subtitle, spec,
        )
        _st.add_footer(fig, caption, None, spec)
    return fig


def _valid_mask_per_band(
    values: np.ndarray, max_workers: Optional[int] = None,
) -> np.ndarray:
    """``(n_bands, H, W)`` boolean finite mask, computed band-parallel.

    Numpy releases the GIL for these reductions, so a small bounded thread
    pool genuinely overlaps the per-band scans.
    """
    n_bands = values.shape[0]
    workers = _resolve_workers(n_bands, max_workers)
    if workers == 1 or n_bands == 1:
        return np.isfinite(values)

    mask = np.empty(values.shape, dtype=bool)
    with ThreadPoolExecutor(max_workers=min(workers, n_bands)) as pool:
        for i, band_mask in enumerate(pool.map(
            lambda i: np.isfinite(values[i]), range(n_bands)
        )):
            mask[i] = band_mask
    return mask


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
    units: Optional[str] = None,
    style: str = "default",
    max_workers: Optional[int] = None,
    dpi: Optional[int] = None,
    renderer: Optional[str] = "static",
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
        "png" (default for the static renderer), "pdf", "svg", "tiff", or
        "html" for the interactive Plotly renderer ("html" is coerced to
        "png" for the static renderer).
    lang : str
        Message language. Default "en".
    units : str, optional
        Variable units for the y-axis title. Defaults to
        ``grid_result.units`` when set.
    style : str
        Publication style preset ("default"/"nature"/"science"/"generic_bw").
    max_workers : int, optional
        Bound on the thread pool used to scan the bands for valid pixels.
        Default ``min(8, cpu_count)``.
    dpi : int, optional
        Resolution for static exports. Defaults to 300 dpi (or the style
        preset's own resolution if higher).
    renderer : {"static", "plotly"}, optional
        "static" (default, also ``None``/``"matplotlib"``/``"agg"``) returns a
        matplotlib ``Figure`` saved as native PNG/PDF/SVG/TIFF with no Plotly,
        Kaleido, Chromium or browser. "plotly" returns the previous
        interactive figure.

    Returns
    -------
    LCZIndexesResult or pl.DataFrame
        ``df`` has one row per LCZ class present: lcz, lcz_name, color,
        n_pixels, mean, std, min, max, median. ``fig`` is a matplotlib
        ``Figure`` for the default static renderer (Plotly ``go.Figure`` when
        ``renderer="plotly"``).
    """
    renderer = _st.resolve_renderer(renderer)
    lcz_path = Path(x)
    if not lcz_path.exists():
        raise FileNotFoundError(f"LCZ map not found: {lcz_path}")

    with rasterio.open(str(lcz_path)) as src:
        classes = src.read(1)

    values = _select_band(grid_result, band)
    if values.shape[1:] != classes.shape:
        raise ValueError(
            f"Grid data shape {values.shape[1:]} does not match LCZ map shape "
            f"{classes.shape}. The lcz_grid_* result must be cropped to this "
            "exact LCZ map (pass the same x used to download it)."
        )

    n_bands = values.shape[0]
    mask = _valid_mask_per_band(values, max_workers=max_workers)
    class_mask = np.isfinite(classes) & (classes >= 1) & (classes <= 17)
    valid = mask & class_mask[None, ...]

    if not valid.any():
        raise ValueError(lcz_msg("cal_indexes_no_overlap", lang))

    tiled_classes = np.broadcast_to(classes, values.shape)
    df = _build_summary(tiled_classes[valid], values[valid])
    df = _attach_metadata(df, inclusive, lang)
    if df.height == 0:  # pragma: no cover - valid.any() above already guards
        raise ValueError(lcz_msg("cal_indexes_no_overlap", lang))

    variable_name = variable_name or (
        "+".join(grid_result.variables) if grid_result.variables else "value"
    )
    units = units if units is not None else getattr(grid_result, "units", None)

    if not iplot:
        if isave:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            df.write_csv(os.path.join(OUTPUT_DIR, "lcz_cal_indexes.csv"))
        return df

    if renderer == "static":
        fig = _plot_bar_static(
            df, variable_name, lang, units=units, style=style, dpi=dpi,
        )
        _st.save_static(
            fig, "lcz_cal_indexes", isave=isave,
            save_extension=save_extension, style=style, dpi=dpi, lang=lang,
        )
        if isave:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            df.write_csv(os.path.join(OUTPUT_DIR, "lcz_cal_indexes.csv"))
        return LCZIndexesResult(df=df, fig=fig, renderer=renderer)

    fig = _plot_bar(df, variable_name, lang, units=units)

    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        finalize_export(
            fig,
            style=style,
            isave=True,
            save_extension=save_extension,
            filename="lcz_cal_indexes",
            column="double",
            lang=lang,
        )
        df.write_csv(os.path.join(OUTPUT_DIR, "lcz_cal_indexes.csv"))
    else:
        # Apply the publication template even when not saving.
        finalize_export(fig, style=style, isave=False, lang=lang)

    return LCZIndexesResult(df=df, fig=fig, renderer=renderer)


__all__ = ["lcz_cal_indexes", "LCZIndexesResult"]
