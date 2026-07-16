"""
lcz_uhi_surface.py

Surface Urban Heat Island (SUHI) intensity from Land Surface Temperature
(``lcz_get_lst`` output), stratified by LCZ class — the *surface* (satellite
LST) counterpart to ``lcz_uhi_intensity.py``'s *canopy-layer* (station air
temperature) UHI. The distinction is standard in the literature (Voogt &
Oke 2003, "Thermal remote sensing of urban climates") — SUHI and canopy UHI
are different physical quantities and are not interchangeable.

Methods (selected from peer-reviewed literature, scoped to what a
multi-date LST raster + LCZ class map can actually support without a
separate rural-boundary polygon or PySAL-style vector weights):

- ``"urban_rural"`` — classic urban-minus-rural mean LST difference
  (Voogt & Oke 2003; rural reference excludes water, following the
  background-percentile convention of Imhoff et al. 2010, "Remote sensing
  of the urban heat island effect across biomes").
- ``"percentile"`` — same, but the rural reference is a low percentile of
  rural-class LST rather than its mean — a more conservative "coolest
  plausible background" baseline (Imhoff et al. 2010). CAVEAT verified on
  real Sentinel-3 data over Rio de Janeiro: this method (and, less
  severely, ``"hotspot"``) is sensitive to unmasked cloud-contaminated
  cold outliers — ``lcz_get_lst``'s Sentinel-3 path applies no per-pixel
  quality/cloud filtering (unlike its GOES path, which does filter on
  ``DQF``), so a low percentile can land on cloud, not clear-sky rural
  land, producing a spuriously huge "SUHI" on contaminated dates (observed:
  ~26°C vs. a normal ~1-4°C on clean dates in the same 6-day series).
  Inspect ``n_rural``/the raw rural LST distribution before trusting a
  single date, or prefer ``percentile`` >= 25 over very low percentiles.
- ``"lcz"`` — per-LCZ-class ΔT_LCZ_X-D: each class's mean LST minus the
  reference LCZ D ("Low plants", class 14) — the WUDAPT/LCZ convention
  used by Chakraborty & Lee (2019), Bechtel et al., and this package's own
  ``lcz_cal_indices``-style per-class comparison, applied to LST.
- ``"utfvi"`` — Urban Thermal Field Variance Index, Liu & Zhang (2011):
  ``(LST - LST_mean) / LST_mean`` in Kelvin (absolute temperature — unlike
  difference-based methods, this ratio is NOT unit-invariant, so inputs in
  Celsius are converted internally), classified into the six standard
  ecological-effect categories (excellent..worst).
- ``"hotspot"`` — Getis-Ord Gi* spatial hotspot detection (Ord & Getis
  1995) applied directly to the LST raster via a local moving window (no
  vector weights matrix needed) — classifies pixels as significant
  hot/cold spots at a chosen confidence level.
- ``"transect"`` — LST/SUHI anomaly binned by distance from the nearest
  urban pixel, the classic urban-rural gradient profile plot (e.g. Zhou et
  al.'s buffer-ring approach), computed via a Euclidean distance transform
  instead of concentric vector buffers.
- ``"persistence"`` — a novel visualization for this package: per-pixel
  fraction of observed dates on which that pixel's LST exceeded the day's
  rural reference by more than a threshold. Standard SUHI maps show a
  single date's snapshot; this instead answers "how *reliably* hot is this
  location across the whole observation record", which only makes sense
  because ``lcz_get_lst`` already returns a multi-date stack.
- ``"all"`` — every method above, as a dict of results.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import polars as pl
import rasterio
from scipy import stats as spstats
from scipy.ndimage import distance_transform_edt, uniform_filter

import plotly.graph_objects as go

from LCZ4py.general.lcz_cal_area import _attach_metadata
from LCZ4py.general.lcz_get_lst import LCZLSTResult
from LCZ4py._internal.lcz_parameters_data import LCZ_COLORS, LCZ_COLORBLIND
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"

# LCZ 1-10 built-up, 11-16 natural/rural reference (17=water excluded from the
# rural baseline — same convention as lcz_uhi_intensity.py's RURAL_CLASSES and
# the Imhoff et al. 2010 practice of excluding water/mountains from the rural
# background).
URBAN_CLASSES: list[int] = list(range(1, 11))
RURAL_CLASSES: list[int] = list(range(11, 17))
REFERENCE_LCZ: int = 14  # LCZ D, "Low plants" — Chakraborty & Lee 2019 / WUDAPT convention

_UTFVI_CATEGORIES: list[tuple[float, str]] = [
    (0.0, "excellent"), (0.005, "good"), (0.01, "normal"),
    (0.015, "bad"), (0.02, "worse"), (float("inf"), "worst"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_classes(lcz_map: Union[str, Path]) -> np.ndarray:
    with rasterio.open(str(lcz_map)) as src:
        return src.read(1)


def _load_lst_stack(
    lst_x: Union[str, Path, LCZLSTResult],
) -> tuple[np.ndarray, list[str], str, dict]:
    """Return (array, dates, units, profile) read from disk.

    Always re-reads from the on-disk GeoTIFF so units/crs/transform are
    guaranteed consistent with what was actually written (an
    ``LCZLSTResult.path`` is always Kelvin regardless of the ``units`` the
    caller originally requested from ``lcz_get_lst`` — see its docstring —
    so the file's own "units" tag, not the caller's assumption, is what
    this function trusts).
    """
    if isinstance(lst_x, LCZLSTResult):
        if not lst_x.path or not os.path.exists(lst_x.path):
            raise ValueError(
                "LCZLSTResult has no on-disk raster to read (path is None or "
                "missing) — call lcz_get_lst with cache=True (the default) "
                "or isave=True, or pass a raster path directly."
            )
        lst_x = lst_x.path

    with rasterio.open(str(lst_x)) as src:
        array = src.read()
        dates = [src.descriptions[i] or f"band_{i + 1}" for i in range(src.count)]
        units = src.tags().get("units", "K")
        profile = src.profile.copy()

    return array, dates, units, profile


def _to_kelvin(arr: np.ndarray, units: str) -> np.ndarray:
    return arr + 273.15 if units == "C" else arr


# ── Method: urban_rural / percentile ──────────────────────────────────────────

def _urban_rural_series(
    lst_stack: np.ndarray, classes: np.ndarray, dates: list[str],
    percentile: Optional[float] = None,
) -> pl.DataFrame:
    urban_mask = np.isin(classes, URBAN_CLASSES)
    rural_mask = np.isin(classes, RURAL_CLASSES)

    rows = []
    for i, d in enumerate(dates):
        band = lst_stack[i]
        u = band[urban_mask & np.isfinite(band)]
        r = band[rural_mask & np.isfinite(band)]
        if len(u) == 0 or len(r) == 0:
            continue
        u_mean = float(u.mean())
        r_ref = float(np.percentile(r, percentile)) if percentile is not None else float(r.mean())
        rows.append({
            "date": d, "urban": u_mean, "rural": r_ref, "suhi": u_mean - r_ref,
            "n_urban": int(len(u)), "n_rural": int(len(r)),
        })

    if not rows:
        return pl.DataFrame(schema={
            "date": pl.String, "urban": pl.Float64, "rural": pl.Float64,
            "suhi": pl.Float64, "n_urban": pl.Int64, "n_rural": pl.Int64,
        })
    return pl.DataFrame(rows)


# ── Method: lcz (ΔT_LCZ_X-D) ─────────────────────────────────────────────────

def _lcz_delta_table(
    lst_stack: np.ndarray, classes: np.ndarray, dates: list[str], reference_lcz: int,
) -> pl.DataFrame:
    lcz_flat = classes.ravel()
    valid_lcz = (lcz_flat >= 1) & (lcz_flat <= 17)

    rows = []
    for i, d in enumerate(dates):
        band_flat = lst_stack[i].ravel()
        finite = np.isfinite(band_flat)
        ref_vals = band_flat[valid_lcz & finite & (lcz_flat == reference_lcz)]
        if len(ref_vals) == 0:
            continue
        ref_mean = float(ref_vals.mean())
        for cls in range(1, 18):
            cls_vals = band_flat[valid_lcz & finite & (lcz_flat == cls)]
            if len(cls_vals) == 0:
                continue
            cls_mean = float(cls_vals.mean())
            rows.append({
                "date": d, "lcz": cls, "mean_lst": cls_mean,
                "delta_t": cls_mean - ref_mean, "count": int(len(cls_vals)),
            })

    if not rows:
        return pl.DataFrame(schema={
            "date": pl.String, "lcz": pl.Int16, "mean_lst": pl.Float64,
            "delta_t": pl.Float64, "count": pl.Int64,
        })
    return pl.DataFrame(rows).with_columns(pl.col("lcz").cast(pl.Int16))


# ── Method: utfvi ──────────────────────────────────────────────────────────────

def _utfvi_stack(lst_stack_k: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """(LST - LST_mean) / LST_mean per date, Kelvin required (Liu & Zhang 2011)."""
    out = np.full_like(lst_stack_k, np.nan, dtype=np.float32)
    for i in range(lst_stack_k.shape[0]):
        band = lst_stack_k[i]
        finite = valid_mask & np.isfinite(band)
        if not finite.any():
            continue
        tmean = float(band[finite].mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = np.where(finite, (band - tmean) / tmean, np.nan)
    return out


def _classify_utfvi(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    for upper, label in _UTFVI_CATEGORIES:
        if value < upper:
            return label
    return "worst"


_classify_utfvi_vec = np.vectorize(_classify_utfvi, otypes=[object])


# ── Method: hotspot (Getis-Ord Gi*, raster, no vector weights needed) ────────

def _getis_ord_gi_star(x: np.ndarray, valid: np.ndarray, window: int = 3) -> np.ndarray:
    """Ord & Getis (1995) Gi* over a square moving window, binary contiguity
    weights (1 inside the window incl. self, 0 outside) — the raster
    equivalent of a fixed-distance spatial weights matrix, computed via
    ``scipy.ndimage.uniform_filter`` instead of a PySAL-style sparse W
    (this repo has no pysal/esda dependency; a dense/sparse weights matrix
    would be infeasible for a full LST raster's pixel count anyway).
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")

    finite = valid & np.isfinite(x)
    if finite.sum() < 2:
        return np.full(x.shape, np.nan, dtype=np.float32)

    x_filled = np.where(finite, x, 0.0).astype(np.float64)
    valid_f = finite.astype(np.float64)
    area = window * window

    n_local = uniform_filter(valid_f, size=window, mode="constant", cval=0.0) * area
    sum_local = uniform_filter(x_filled, size=window, mode="constant", cval=0.0) * area

    n_global = float(finite.sum())
    x_bar = float(x[finite].mean())
    s = float(x[finite].std(ddof=0))

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = s * n_local * np.sqrt(np.maximum(n_global - n_local, 0) / (n_global - 1))
        gi = (sum_local - x_bar * n_local) / denom

    gi = np.where(finite & (n_local > 1) & (s > 0), gi, np.nan)
    return gi.astype(np.float32)


def _classify_hotspot(z: np.ndarray, confidence: float) -> np.ndarray:
    """Hot/Cold/Not significant at the given two-tailed confidence level
    (e.g. 0.90/0.95/0.99 -> z >= 1.645/1.960/2.576, the standard Gi*
    reporting thresholds)."""
    z_thr = spstats.norm.ppf(0.5 + confidence / 2)
    out = np.full(z.shape, "", dtype=object)
    out[np.isnan(z)] = "N/A"
    out[(z >= z_thr)] = "Hot spot"
    out[(z <= -z_thr)] = "Cold spot"
    out[(out == "") ] = "Not significant"
    return out


# ── Method: transect (urban-rural gradient profile) ─────────────────────────

def _pixel_size_km(transform: rasterio.Affine, crs: rasterio.crs.CRS, center_lat: float) -> float:
    """Approximate pixel size in km — geodesic for EPSG:4326, else assume meters."""
    if crs and crs.to_epsg() == 4326:
        deg = abs(transform.a)
        return deg * 111.32 * max(0.1, math.cos(math.radians(center_lat)))
    return abs(transform.a) / 1000.0


def _transect_table(
    lst_stack: np.ndarray, classes: np.ndarray, dates: list[str],
    transform: rasterio.Affine, crs: rasterio.crs.CRS, n_bins: int = 10,
) -> pl.DataFrame:
    urban_mask = np.isin(classes, URBAN_CLASSES)
    rural_mask = np.isin(classes, RURAL_CLASSES)
    valid_mask = (classes >= 1) & (classes <= 17)

    h, w = classes.shape
    center_row, center_col = h // 2, w // 2
    center_lat = rasterio.transform.xy(transform, center_row, center_col)[1]
    px_km = _pixel_size_km(transform, crs, center_lat)

    # Distance (km) from the nearest urban pixel, for every non-urban pixel.
    dist_px = distance_transform_edt(~urban_mask)
    dist_km = dist_px * px_km

    max_dist = float(dist_km[rural_mask].max()) if rural_mask.any() else 0.0
    if max_dist <= 0:
        return pl.DataFrame(schema={
            "date": pl.String, "bin_km": pl.Float64, "mean_anomaly": pl.Float64, "count": pl.Int64,
        })
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(dist_km, bin_edges) - 1, 0, n_bins - 1)

    rows = []
    for i, d in enumerate(dates):
        band = lst_stack[i]
        rural_vals = band[rural_mask & np.isfinite(band)]
        if len(rural_vals) == 0:
            continue
        rural_ref = float(rural_vals.mean())
        anomaly = band - rural_ref
        for b in range(n_bins):
            sel = (bin_idx == b) & rural_mask & np.isfinite(band) & valid_mask
            if not sel.any():
                continue
            rows.append({
                "date": d, "bin_km": float(bin_centers[b]),
                "mean_anomaly": float(anomaly[sel].mean()), "count": int(sel.sum()),
            })

    if not rows:
        return pl.DataFrame(schema={
            "date": pl.String, "bin_km": pl.Float64, "mean_anomaly": pl.Float64, "count": pl.Int64,
        })
    return pl.DataFrame(rows)


# ── Method: persistence (novel) ──────────────────────────────────────────────

def _suhi_persistence(
    lst_stack: np.ndarray, classes: np.ndarray, dates: list[str], threshold: float = 1.0,
) -> np.ndarray:
    rural_mask = np.isin(classes, RURAL_CLASSES)
    h, w = classes.shape
    exceed_count = np.zeros((h, w), dtype=np.float32)
    valid_count = np.zeros((h, w), dtype=np.float32)

    for i in range(len(dates)):
        band = lst_stack[i]
        rural_vals = band[rural_mask & np.isfinite(band)]
        if len(rural_vals) == 0:
            continue
        rural_ref = float(rural_vals.mean())
        finite = np.isfinite(band)
        valid_count += finite
        exceed_count += finite & ((band - rural_ref) > threshold)

    with np.errstate(divide="ignore", invalid="ignore"):
        persistence = np.where(valid_count > 0, exceed_count / valid_count, np.nan)
    return (persistence * 100).astype(np.float32)  # percent of dates


def _persistence_by_class(persistence: np.ndarray, classes: np.ndarray) -> pl.DataFrame:
    rows = []
    for cls in range(1, 18):
        vals = persistence[(classes == cls) & np.isfinite(persistence)]
        if len(vals) == 0:
            continue
        rows.append({"lcz": cls, "persistence": float(vals.mean()), "n": int(len(vals))})
    if not rows:
        return pl.DataFrame(schema={"lcz": pl.Int16, "persistence": pl.Float64, "n": pl.Int64})
    return pl.DataFrame(rows).with_columns(pl.col("lcz").cast(pl.Int16))


# ── Date-gap-aware line chart (daily resolution, mirrors lcz_uhi_intensity's
# hourly _break_gaps_numpy but binned in days since lcz_get_lst is daily) ────

def _break_date_gaps(dates: list[str], values: list[float]) -> tuple[list, list]:
    if len(dates) <= 2:
        return dates, values
    d = np.array(dates, dtype="datetime64[D]")
    diffs = np.diff(d).astype(np.float64)
    pos = diffs[diffs > 0]
    threshold = max(2.0 * float(np.median(pos)), 1.5) if len(pos) else 2.0

    out_d, out_v = [], []
    prev = 0
    for idx in np.where(diffs > threshold)[0]:
        out_d.extend(dates[prev:idx + 1])
        out_v.extend(values[prev:idx + 1])
        out_d.append(None)
        out_v.append(None)
        prev = idx + 1
    out_d.extend(dates[prev:])
    out_v.extend(values[prev:])
    return out_d, out_v


# ── Chart builders ─────────────────────────────────────────────────────────────

def _plot_timeseries(df: pl.DataFrame, y_col: str, title: str, ylabel: str, lang: str) -> go.Figure:
    dates = df["date"].to_list()
    values = df[y_col].to_list()
    d, v = _break_date_gaps(dates, values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d, y=v, mode="lines+markers",
        line=dict(color="#333333", width=1.5), marker=dict(size=5),
        hovertemplate=f"{ylabel}: %{{y:.2f}}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=lcz_msg("time_label", lang), font=dict(size=14)), gridcolor="#f0f0f0"),
        yaxis=dict(title=dict(text=ylabel, font=dict(size=14)), gridcolor="#f0f0f0"),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        margin=dict(l=80, r=40, t=70, b=60),
        hovermode="x unified", hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    return fig


def _plot_value_by_lcz_bar(
    df: pl.DataFrame, value_col: str, lang: str, inclusive: bool, ylab: str, title: str,
) -> go.Figure:
    """Mean ± 95% CI bar per LCZ class (CI collapses to 0 when there's only
    one row per class, e.g. a single-date lcz-method table)."""
    if "date" in df.columns:
        agg = (
            df.group_by("lcz")
            .agg([
                pl.col(value_col).mean().alias("mean_value"),
                pl.col(value_col).std().alias("std_value"),
                pl.len().alias("n"),
            ])
            .sort("lcz")
        )
    else:
        agg = df.rename({value_col: "mean_value"}).with_columns(
            std_value=pl.lit(0.0), n=pl.lit(1),
        ).sort("lcz")

    agg = _attach_metadata(agg, inclusive, lang=lang)

    n = agg["n"].to_numpy()
    std = agg["std_value"].fill_null(0.0).to_numpy()
    se = std / np.sqrt(np.maximum(n, 1))
    tcrit = spstats.t.ppf(0.975, np.maximum(n - 1, 1))
    err = np.where(n > 1, tcrit * se, 0.0)

    fig = go.Figure(go.Bar(
        x=agg["lcz"].cast(str), y=agg["mean_value"],
        marker_color=agg["color"].to_list(), marker_line_color="black", marker_line_width=0.5,
        error_y=dict(type="data", array=err.tolist()),
        customdata=agg.select(["lcz_name", "n"]),
        hovertemplate=(
            f"<b>LCZ %{{x}}</b> — %{{customdata[0]}}<br>{ylab}: %{{y:.2f}}<br>n=%{{customdata[1]}}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=lcz_msg("lcz_code_label", lang), font=dict(size=14))),
        yaxis=dict(title=dict(text=ylab, font=dict(size=14)), gridcolor="#f0f0f0"),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        margin=dict(l=80, r=40, t=60, b=60),
        width=1000, height=600,
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    return fig


def _plot_spatial_heatmap(
    arr: np.ndarray, transform: rasterio.Affine, title: str,
    colorscale: str = "RdBu_r", zmid: Optional[float] = None,
    zmin: Optional[float] = None, zmax: Optional[float] = None,
    colorbar_title: str = "", categories: Optional[np.ndarray] = None,
) -> go.Figure:
    h, w = arr.shape
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + w * transform.a
    ymin = ymax + h * transform.e

    z = np.flipud(arr)
    customdata = np.flipud(categories) if categories is not None else None
    hover_extra = "<br>%{customdata}" if categories is not None else ""

    fig = go.Figure(go.Heatmap(
        z=z, x0=xmin, y0=ymin,
        dx=(xmax - xmin) / w, dy=(ymax - ymin) / h,
        colorscale=colorscale, zmid=zmid, zmin=zmin, zmax=zmax,
        colorbar=dict(title=dict(text=colorbar_title)),
        customdata=customdata,
        hovertemplate="value: %{z:.3f}" + hover_extra + "<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=40, t=50, b=0),
        width=1000, height=800,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


_UTFVI_CATEGORY_ORDER = ["excellent", "good", "normal", "bad", "worse", "worst"]
_UTFVI_CATEGORY_COLORS = {
    "excellent": "#2ca02c", "good": "#8fbc3f", "normal": "#f4d03f",
    "bad": "#f39c12", "worse": "#e67e22", "worst": "#c0392b",
}


def _plot_utfvi_category_bar(cat_counts: pl.DataFrame, lang: str) -> go.Figure:
    present = [c for c in _UTFVI_CATEGORY_ORDER if c in cat_counts["category"].to_list()]
    cat_counts = cat_counts.filter(pl.col("category").is_in(present))
    order_idx = {c: i for i, c in enumerate(present)}
    cat_counts = cat_counts.with_columns(
        pl.col("category").replace_strict(order_idx).alias("_order")
    ).sort("_order")

    fig = go.Figure(go.Bar(
        x=cat_counts["category"], y=cat_counts["percent"],
        marker_color=[_UTFVI_CATEGORY_COLORS[c] for c in cat_counts["category"]],
        marker_line_color="black", marker_line_width=0.5,
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=lcz_msg("uhi_utfvi_category_title", lang), font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=lcz_msg("uhi_utfvi_category_label", lang))),
        yaxis=dict(title=dict(text=lcz_msg("uhi_percent_pixels_label", lang)), gridcolor="#f0f0f0"),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        margin=dict(l=60, r=40, t=60, b=60),
        width=800, height=500,
    )
    return fig


def _plot_transect(df: pl.DataFrame, lang: str) -> go.Figure:
    fig = go.Figure()
    dates = sorted(df["date"].unique().to_list())

    if len(dates) <= 8:
        for d in dates:
            sub = df.filter(pl.col("date") == d).sort("bin_km")
            fig.add_trace(go.Scatter(
                x=sub["bin_km"], y=sub["mean_anomaly"], mode="lines+markers", name=d,
                hovertemplate=f"{d}<br>%{{x:.1f}} km: %{{y:.2f}}<extra></extra>",
            ))
    else:
        agg = df.group_by("bin_km").agg(pl.col("mean_anomaly").mean()).sort("bin_km")
        fig.add_trace(go.Scatter(
            x=agg["bin_km"], y=agg["mean_anomaly"], mode="lines+markers", name="mean",
            hovertemplate="%{x:.1f} km: %{y:.2f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=dict(text=lcz_msg("uhi_transect_title", lang), font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=lcz_msg("uhi_distance_label", lang), font=dict(size=14)), gridcolor="#f0f0f0"),
        yaxis=dict(title=dict(text=lcz_msg("uhi_delta_label", lang), font=dict(size=14)), gridcolor="#f0f0f0"),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        margin=dict(l=80, r=40, t=60, b=60),
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    return fig


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class LCZUHISurfaceResult:
    """Return type of lcz_uhi_surface for a single method."""
    df: Optional[pl.DataFrame] = None
    fig: Optional[Union[go.Figure, dict[str, go.Figure]]] = None
    array: Optional[np.ndarray] = None


_METHODS = {
    "urban_rural", "percentile", "lcz", "utfvi", "hotspot", "transect", "persistence", "all",
}


def _save_result(
    name: str, result: LCZUHISurfaceResult, save_extension: str,
    style: str = "default", lang: str = "en",
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figs = result.fig if isinstance(result.fig, dict) else ({"_": result.fig} if result.fig else {})
    for key, f in figs.items():
        suffix = "" if key == "_" else f"_{key}"
        finalize_export(f, style=style, isave=True, save_extension=save_extension,
                         filename=f"lcz4r_uhi_surface_{name}{suffix}", lang=lang)
    if result.df is not None and not result.df.is_empty():
        result.df.write_csv(os.path.join(OUTPUT_DIR, f"lcz4r_uhi_surface_{name}.csv"))


def lcz_uhi_surface(
    lcz_map: Union[str, Path],
    lst_x: Union[str, Path, LCZLSTResult],
    method: Literal[
        "urban_rural", "percentile", "lcz", "utfvi", "hotspot", "transect", "persistence", "all",
    ] = "urban_rural",
    reference_lcz: int = REFERENCE_LCZ,
    percentile: float = 10.0,
    hotspot_window: int = 3,
    hotspot_confidence: float = 0.95,
    persistence_threshold: float = 1.0,
    n_bins: int = 10,
    date_index: Optional[int] = None,
    iplot: bool = True,
    isave: bool = False,
    save_extension: str = "html",
    style: str = "default",
    inclusive: bool = False,
    title: Optional[str] = None,
    caption: Optional[str] = None,
    lang: str = "en",
) -> Union[LCZUHISurfaceResult, dict[str, LCZUHISurfaceResult], pl.DataFrame, dict[str, pl.DataFrame]]:
    """Surface Urban Heat Island (SUHI) intensity from LST, stratified by LCZ.

    See the module docstring for the literature backing each ``method``.

    Parameters
    ----------
    lcz_map : str or Path
        Path to the LCZ class GeoTIFF (classes 1-17) that ``lst_x`` was
        computed on the same grid of.
    lst_x : str, Path, or LCZLSTResult
        Output of ``lcz_get_lst`` — a path to its GeoTIFF stack, or the
        result object directly (must have a valid ``.path``, i.e.
        ``cache=True`` (default) or ``isave=True`` was used).
    method : {"urban_rural", "percentile", "lcz", "utfvi", "hotspot", "transect", "persistence", "all"}
        "urban_rural": per-date mean(LST\\|urban) - mean(LST\\|rural), a
        time series. "percentile": same, but the rural reference is the
        ``percentile``-th percentile of rural LST rather than its mean.
        "lcz": per-date, per-class ΔT_LCZ_X-D against ``reference_lcz``
        (default LCZ 14, "Low plants"), bar chart averaged across dates.
        "utfvi": (LST-mean)/mean in Kelvin, classified into six ecological
        categories, for the date at ``date_index`` (default: most recent) —
        a spatial map + a category-frequency bar. "hotspot": Getis-Ord Gi*
        on the LST raster at ``date_index``, classified at
        ``hotspot_confidence``. "transect": LST anomaly (vs. the day's
        rural reference) binned into ``n_bins`` distance-from-urban bins,
        the classic urban-rural gradient profile. "persistence": per-pixel
        % of dates where LST exceeded the day's rural reference by more
        than ``persistence_threshold`` — a map plus a per-class summary.
        "all": every method above, returned as a dict keyed by method name.
    reference_lcz : int
        Reference LCZ class for the "lcz" method. Default 14 (Low plants).
    percentile : float
        Percentile (0-100) of rural LST used as the reference for the
        "percentile" method. Default 10.
    hotspot_window : int
        Odd window size (pixels) for the "hotspot" method's local Gi*.
        Default 3 (a 3x3 neighborhood).
    hotspot_confidence : float
        Two-tailed confidence level for hotspot significance (e.g. 0.90,
        0.95, 0.99). Default 0.95.
    persistence_threshold : float
        Degrees above the day's rural reference counted as "exceeding" for
        the "persistence" method. Default 1.0.
    n_bins : int
        Number of distance bins for the "transect" method. Default 10.
    date_index : int, optional
        Which LST band (0-based) to use for the single-date "utfvi"/
        "hotspot" methods. Defaults to the most recent date.
    iplot : bool
        If False, skip chart building and return only the statistics
        DataFrame (or, for "all", a dict of DataFrames).
    isave : bool
        Save chart(s) + CSV to ``LCZ4r_output/``.
    save_extension : str
        "html" for interactive, "png"/"pdf" for static.
    style : str
        Publication style preset: 'default', 'nature', 'science', or
        'generic_bw'. Controls font, figure size (mm), DPI, and palette
        used when isave and save_extension != 'html'.
    inclusive : bool
        Use the colorblind-friendly LCZ palette (bar charts only).
    title, caption : str, optional
        Figure annotations.
    lang : str
        Message language ("en"/"pt"/"es"/"zh").

    Returns
    -------
    LCZUHISurfaceResult, dict[str, LCZUHISurfaceResult], pl.DataFrame, or dict[str, pl.DataFrame]
        A single result for a specific ``method``, or (for ``"all"``) a
        dict keyed by method name. If ``iplot=False``, each result
        collapses to its bare ``pl.DataFrame`` (no chart built).

    Examples
    --------
    >>> from LCZ4py.general.lcz_get_lst import lcz_get_lst
    >>> lst = lcz_get_lst("lcz_map.tif", source="sentinel3",
    ...                    start_date="2024-07-01", end_date="2024-07-10")
    >>> suhi = lcz_uhi_surface("lcz_map.tif", lst, method="lcz")
    >>> suhi.df.sort("delta_t", descending=True)
    >>> suhi.fig.show()

    >>> everything = lcz_uhi_surface("lcz_map.tif", lst, method="all")
    >>> everything["persistence"].fig["map"].show()
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")

    classes = _load_classes(lcz_map)
    lst_stack, dates, units, profile = _load_lst_stack(lst_x)

    if classes.shape != lst_stack.shape[1:]:
        raise ValueError(lcz_msg(
            "uhi_shape_mismatch", lang, lcz_shape=classes.shape, lst_shape=lst_stack.shape[1:],
        ))

    valid_mask = (classes >= 1) & (classes <= 17)
    transform = profile["transform"]
    di = date_index if date_index is not None else len(dates) - 1

    def _finalize(name: str, df: pl.DataFrame, fig) -> LCZUHISurfaceResult:
        if title or caption:
            figs = fig if isinstance(fig, dict) else ({"_": fig} if fig else {})
            for f in figs.values():
                if title:
                    f.update_layout(title=dict(text=title))
                if caption:
                    f.add_annotation(
                        text=caption, xref="paper", yref="paper", x=0.5, y=-0.08,
                        showarrow=False, font=dict(size=10, color="gray"),
                    )
        result = LCZUHISurfaceResult(df=df, fig=fig)
        if isave:
            _save_result(name, result, save_extension, style=style, lang=lang)
        return result

    def _run(name: str) -> LCZUHISurfaceResult:
        if name == "urban_rural":
            df = _urban_rural_series(lst_stack, classes, dates)
            fig = _plot_timeseries(
                df, "suhi", lcz_msg("uhi_surface_title", lang), lcz_msg("uhi_delta_label", lang), lang,
            ) if iplot else None
            return _finalize(name, df, fig)

        if name == "percentile":
            df = _urban_rural_series(lst_stack, classes, dates, percentile=percentile)
            fig = _plot_timeseries(
                df, "suhi", lcz_msg("uhi_surface_percentile_title", lang, p=percentile),
                lcz_msg("uhi_delta_label", lang), lang,
            ) if iplot else None
            return _finalize(name, df, fig)

        if name == "lcz":
            df = _lcz_delta_table(lst_stack, classes, dates, reference_lcz)
            fig = _plot_value_by_lcz_bar(
                df, "delta_t", lang, inclusive, lcz_msg("uhi_delta_label", lang),
                lcz_msg("uhi_lcz_title", lang, ref=reference_lcz),
            ) if iplot else None
            return _finalize(name, df, fig)

        if name == "utfvi":
            lst_k = _to_kelvin(lst_stack, units)
            utfvi_stack = _utfvi_stack(lst_k, valid_mask)
            band = utfvi_stack[di]
            finite = valid_mask & np.isfinite(band)
            cats = _classify_utfvi_vec(band)
            counts = pl.Series(cats[finite]).value_counts()
            total = int(finite.sum())
            df = counts.rename({"": "category"} if "" in counts.columns else {}).with_columns(
                (pl.col("count") / total * 100).alias("percent")
            ).sort("category")
            array = np.where(valid_mask, band, np.nan)
            if iplot:
                fig = {
                    "map": _plot_spatial_heatmap(
                        array, transform, lcz_msg("uhi_utfvi_title", lang, date=dates[di]),
                        colorscale="RdYlGn_r", zmid=0.0, colorbar_title="UTFVI",
                    ),
                    "bar": _plot_utfvi_category_bar(df, lang),
                }
            else:
                fig = None
            result = _finalize(name, df, fig)
            result.array = array
            return result

        if name == "hotspot":
            band = lst_stack[di]
            gi = _getis_ord_gi_star(band, valid_mask, window=hotspot_window)
            cats = _classify_hotspot(gi, hotspot_confidence)
            finite = valid_mask & np.isfinite(gi)
            counts = pl.Series(cats[finite]).value_counts()
            total = int(finite.sum())
            df = counts.rename({"": "category"} if "" in counts.columns else {}).with_columns(
                (pl.col("count") / total * 100).alias("percent")
            ).sort("category")
            array = np.where(valid_mask, gi, np.nan)
            fig = _plot_spatial_heatmap(
                array, transform, lcz_msg("uhi_hotspot_title", lang, date=dates[di]),
                colorscale="RdBu_r", zmid=0.0, colorbar_title="Gi* z-score",
                categories=np.where(valid_mask, cats, ""),
            ) if iplot else None
            result = _finalize(name, df, fig)
            result.array = array
            return result

        if name == "transect":
            df = _transect_table(lst_stack, classes, dates, transform, profile.get("crs"), n_bins=n_bins)
            fig = _plot_transect(df, lang) if iplot else None
            return _finalize(name, df, fig)

        # "persistence"
        persistence = _suhi_persistence(lst_stack, classes, dates, threshold=persistence_threshold)
        array = np.where(valid_mask, persistence, np.nan)
        df = _persistence_by_class(array, classes)
        if iplot:
            fig = {
                "map": _plot_spatial_heatmap(
                    array, transform, lcz_msg("uhi_persistence_title", lang, thr=persistence_threshold),
                    colorscale="YlOrRd", zmin=0, zmax=100, colorbar_title="% of dates",
                ),
                "bar": _plot_value_by_lcz_bar(
                    df, "persistence", lang, inclusive, lcz_msg("uhi_persistence_label", lang),
                    lcz_msg("uhi_persistence_by_class_title", lang),
                ),
            }
        else:
            fig = None
        result = _finalize(name, df, fig)
        result.array = array
        return result

    if method == "all":
        results = {name: _run(name) for name in _METHODS - {"all"}}
        return results if iplot else {name: r.df for name, r in results.items()}

    result = _run(method)
    return result if iplot else result.df


__all__ = ["lcz_uhi_surface", "LCZUHISurfaceResult", "URBAN_CLASSES", "RURAL_CLASSES", "REFERENCE_LCZ"]
