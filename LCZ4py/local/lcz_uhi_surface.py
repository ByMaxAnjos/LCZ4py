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
- ``"all"`` — every method above, as a dict of results (in a deterministic
  order).

All per-class reductions use ``np.bincount`` on a precomputed class index
instead of one full-raster boolean mask per class per date.
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

from LCZ4py.general.lcz_get_lst import LCZLSTResult
from LCZ4py._internal.lcz_parameters_data import (
    LCZ_COLORS, LCZ_COLORBLIND, get_lcz_names,
)
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal.lcz_theme import finalize_export, add_map_furniture

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
_UTFVI_LABELS: list[str] = [label for _, label in _UTFVI_CATEGORIES]
_UTFVI_EDGES: np.ndarray = np.array([0.0, 0.005, 0.01, 0.015, 0.02])


# ── Small i18n helpers for labels missing from the central table ─────────────

_EXTRA_MESSAGES: dict[str, dict[str, str]] = {
    "longitude_label": {
        "en": "Longitude [°]", "pt": "Longitude [°]",
        "es": "Longitud [°]", "zh": "经度 [°]",
    },
    "latitude_label": {
        "en": "Latitude [°]", "pt": "Latitude [°]",
        "es": "Latitud [°]", "zh": "纬度 [°]",
    },
    "easting_label": {
        "en": "Easting [m]", "pt": "Este [m]", "es": "Este [m]", "zh": "东向 [m]",
    },
    "northing_label": {
        "en": "Northing [m]", "pt": "Norte [m]", "es": "Norte [m]", "zh": "北向 [m]",
    },
    "no_valid_pixels": {
        "en": "No valid pixels for this method",
        "pt": "Nenhum pixel válido para este método",
        "es": "Ningún píxel válido para este método",
        "zh": "此方法没有有效像元",
    },
}


def _xmsg(key: str, lang: str, **kwargs) -> str:
    if key in _EXTRA_MESSAGES:
        table = _EXTRA_MESSAGES[key]
        template = table.get(lang, table["en"])
        for k, v in kwargs.items():
            template = template.replace("{" + k + "}", str(v))
        return template
    return lcz_msg(key, lang, **kwargs)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_classes(lcz_map: Union[str, Path]) -> np.ndarray:
    """Read the single-band LCZ class raster as an int16 array."""
    path = str(lcz_map)
    if not os.path.exists(path):
        raise FileNotFoundError(f"LCZ raster not found: {path}")
    with rasterio.open(path) as src:
        if src.count < 1:
            raise ValueError(f"LCZ raster has no bands: {path}")
        classes = src.read(1)
        crs = src.crs
    if crs is None:
        logger.warning("LCZ raster %s has no CRS; assuming it matches the LST grid.", path)
    if np.issubdtype(classes.dtype, np.floating):
        classes = np.where(np.isfinite(classes), np.rint(classes), 0).astype(np.int16)
    else:
        classes = classes.astype(np.int16)
    return classes


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

    path = str(lst_x)
    if not os.path.exists(path):
        raise FileNotFoundError(f"LST raster not found: {path}")

    with rasterio.open(path) as src:
        if src.count < 1:
            raise ValueError(f"LST raster has no bands: {path}")
        array = src.read()
        dates = [src.descriptions[i] or f"band_{i + 1}" for i in range(src.count)]
        units = (src.tags().get("units") or "K").strip()
        profile = src.profile.copy()

    if array.ndim != 3:
        raise ValueError(f"Expected a multi-band LST stack, got shape {array.shape}")
    return array, dates, units, profile


def _to_kelvin(arr: np.ndarray, units: str) -> np.ndarray:
    """Convert Celsius input to Kelvin (the units UTFVI requires)."""
    key = str(units).strip().lower().replace("°", "")
    if key in ("c", "celsius", "degc"):
        return arr + 273.15
    if key not in ("k", "kelvin"):
        logger.warning("Unknown LST units %r; assuming Kelvin.", units)
    return arr


def _band_mean(band: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Finite mean of *band* over *mask* (``None`` when nothing is valid)."""
    values = band[mask]
    finite = np.isfinite(values)
    if not finite.any():
        return None
    return float(values[finite].mean())


# ── Method: urban_rural / percentile ──────────────────────────────────────────

def _urban_rural_series(
    lst_stack: np.ndarray, classes: np.ndarray, dates: list[str],
    percentile: Optional[float] = None,
) -> pl.DataFrame:
    """Per-date urban mean minus rural mean (or rural percentile)."""
    urban_mask = np.isin(classes, URBAN_CLASSES)
    rural_mask = np.isin(classes, RURAL_CLASSES)

    schema = {
        "date": pl.String, "urban": pl.Float64, "rural": pl.Float64,
        "suhi": pl.Float64, "n_urban": pl.Int64, "n_rural": pl.Int64,
    }
    if not urban_mask.any():
        raise ValueError(
            "The LCZ raster contains no urban pixels (classes 1-10), so no "
            "surface UHI can be computed."
        )
    if not rural_mask.any():
        raise ValueError(
            "The LCZ raster contains no rural pixels (classes 11-16, water "
            "excluded), so there is no reference for the surface UHI."
        )

    rows = []
    skipped: list[str] = []
    for i, d in enumerate(dates):
        band = lst_stack[i]
        u_values = band[urban_mask]
        r_values = band[rural_mask]
        u_finite = np.isfinite(u_values)
        r_finite = np.isfinite(r_values)
        if not u_finite.any() or not r_finite.any():
            skipped.append(d)
            continue
        u_mean = float(u_values[u_finite].mean())
        r_ref = (
            float(np.percentile(r_values[r_finite], percentile))
            if percentile is not None else float(r_values[r_finite].mean())
        )
        rows.append({
            "date": d, "urban": u_mean, "rural": r_ref, "suhi": u_mean - r_ref,
            "n_urban": int(u_finite.sum()), "n_rural": int(r_finite.sum()),
        })

    if skipped:
        logger.warning(
            "%d/%d date(s) had no valid urban and/or rural LST and were "
            "skipped: %s", len(skipped), len(dates), skipped[:5],
        )
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema_overrides=schema)


# ── Method: lcz (ΔT_LCZ_X-D) ─────────────────────────────────────────────────

def _lcz_delta_table(
    lst_stack: np.ndarray, classes: np.ndarray, dates: list[str], reference_lcz: int,
) -> pl.DataFrame:
    """Per-date, per-class mean LST and ΔT against ``reference_lcz``.

    Uses ``np.bincount`` over a precomputed class index — one pass per date
    instead of one full-raster mask per class per date.
    """
    schema = {
        "date": pl.String, "lcz": pl.Int16, "mean_lst": pl.Float64,
        "delta_t": pl.Float64, "count": pl.Int64,
    }
    cls_flat = classes.ravel().astype(np.int32)
    codes = np.where((cls_flat >= 1) & (cls_flat <= 17), cls_flat, 0)

    frame_rows = []
    missing_ref: list[str] = []
    for i, d in enumerate(dates):
        band_flat = lst_stack[i].ravel()
        finite = np.isfinite(band_flat)
        if not finite.any():
            missing_ref.append(d)
            continue
        weights = np.where(finite, band_flat, 0.0)
        counts = np.bincount(codes, weights=finite.astype(np.float64), minlength=18)
        sums = np.bincount(codes, weights=weights, minlength=18)

        ref_count = counts[reference_lcz]
        if ref_count <= 0:
            missing_ref.append(d)
            continue
        ref_mean = float(sums[reference_lcz] / ref_count)

        for cls in range(1, 18):
            if counts[cls] <= 0:
                continue
            cls_mean = float(sums[cls] / counts[cls])
            frame_rows.append({
                "date": d, "lcz": cls, "mean_lst": cls_mean,
                "delta_t": cls_mean - ref_mean, "count": int(counts[cls]),
            })

    if missing_ref:
        logger.warning(
            "%d/%d date(s) had no valid pixels for reference LCZ %d and were "
            "skipped: %s", len(missing_ref), len(dates), reference_lcz,
            missing_ref[:5],
        )
    if not frame_rows:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame(frame_rows)
          .with_columns([
              pl.col("lcz").cast(pl.Int16),
              pl.col("count").cast(pl.Int64),
          ])
          .sort(["date", "lcz"])
    )


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
        if not np.isfinite(tmean) or tmean == 0.0:
            logger.warning("UTFVI: band %d has a zero/undefined LST mean; skipped.", i)
            continue
        out[i] = np.where(finite, (band - tmean) / tmean, np.nan)
    return out


def _classify_utfvi(value: float) -> str:
    """Single-value UTFVI class (kept for API compatibility)."""
    if not np.isfinite(value):
        return "N/A"
    for upper, label in _UTFVI_CATEGORIES:
        if value < upper:
            return label
    return "worst"


def _classify_utfvi_array(values: np.ndarray) -> np.ndarray:
    """Vectorised UTFVI classification (``np.searchsorted``, no Python loop)."""
    arr = np.asarray(values, dtype=np.float64)
    idx = np.searchsorted(_UTFVI_EDGES, np.nan_to_num(arr, nan=0.0), side="right")
    idx = np.clip(idx, 0, len(_UTFVI_LABELS) - 1)
    out = np.asarray(_UTFVI_LABELS, dtype=object)[idx]
    out = np.where(np.isfinite(arr), out, "N/A")
    return out


# Retained: some callers rely on the old vectorize helper name.
_classify_utfvi_vec = _classify_utfvi_array


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

    gi = np.where(finite & (n_local > 1) & (s > 0) & np.isfinite(gi), gi, np.nan)
    return gi.astype(np.float32)


def _classify_hotspot(z: np.ndarray, confidence: float) -> np.ndarray:
    """Hot/Cold/Not significant at the given two-tailed confidence level
    (e.g. 0.90/0.95/0.99 -> z >= 1.645/1.960/2.576, the standard Gi*
    reporting thresholds)."""
    z_thr = spstats.norm.ppf(0.5 + confidence / 2)
    out = np.full(z.shape, "Not significant", dtype=object)
    out[z >= z_thr] = "Hot spot"
    out[z <= -z_thr] = "Cold spot"
    out[~np.isfinite(z)] = "N/A"
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
    """LST anomaly vs. the rural reference, binned by distance from urban."""
    schema = {
        "date": pl.String, "bin_km": pl.Float64,
        "mean_anomaly": pl.Float64, "count": pl.Int64,
    }
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    urban_mask = np.isin(classes, URBAN_CLASSES)
    rural_mask = np.isin(classes, RURAL_CLASSES)
    valid_mask = (classes >= 1) & (classes <= 17)
    if not urban_mask.any():
        raise ValueError(
            "The LCZ raster contains no urban pixels (classes 1-10), so the "
            "urban-rural transect is undefined."
        )
    if not rural_mask.any():
        raise ValueError(
            "The LCZ raster contains no rural pixels (classes 11-16), so the "
            "urban-rural transect is undefined."
        )

    h, w = classes.shape
    center_row, center_col = h // 2, w // 2
    center_lat = rasterio.transform.xy(transform, center_row, center_col)[1]
    px_km = _pixel_size_km(transform, crs, center_lat)

    # Distance (km) from the nearest urban pixel, for every non-urban pixel.
    dist_px = distance_transform_edt(~urban_mask)
    dist_km = dist_px * px_km

    max_dist = float(dist_km[rural_mask].max()) if rural_mask.any() else 0.0
    if max_dist <= 0:
        return pl.DataFrame(schema=schema)
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(dist_km, bin_edges) - 1, 0, n_bins - 1)

    # Precompute the pixel selection once; only the LST values change per date.
    selected = rural_mask & valid_mask
    selected_flat = selected.ravel()
    bins_selected = bin_idx.ravel()[selected_flat]

    rows = []
    skipped: list[str] = []
    for i, d in enumerate(dates):
        values = lst_stack[i].ravel()[selected_flat]
        finite = np.isfinite(values)
        if not finite.any():
            skipped.append(d)
            continue
        rural_ref = float(values[finite].mean())
        anomaly = values[finite] - rural_ref
        bins_f = bins_selected[finite]
        sums = np.bincount(bins_f, weights=anomaly, minlength=n_bins)
        counts = np.bincount(bins_f, minlength=n_bins)
        for b in range(n_bins):
            if counts[b] == 0:
                continue
            rows.append({
                "date": d, "bin_km": float(bin_centers[b]),
                "mean_anomaly": float(sums[b] / counts[b]), "count": int(counts[b]),
            })

    if skipped:
        logger.warning(
            "%d/%d date(s) had no valid rural LST for the transect: %s",
            len(skipped), len(dates), skipped[:5],
        )
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema_overrides=schema)


# ── Method: persistence (novel) ──────────────────────────────────────────────

def _suhi_persistence(
    lst_stack: np.ndarray, classes: np.ndarray, dates: list[str], threshold: float = 1.0,
) -> np.ndarray:
    rural_mask = np.isin(classes, RURAL_CLASSES)
    h, w = classes.shape
    exceed_count = np.zeros((h, w), dtype=np.float32)
    valid_count = np.zeros((h, w), dtype=np.float32)
    if not rural_mask.any():
        raise ValueError(
            "The LCZ raster contains no rural pixels (classes 11-16), so a "
            "rural-referenced persistence map cannot be built."
        )

    for i in range(len(dates)):
        band = lst_stack[i]
        rural_vals = band[rural_mask]
        rural_finite = np.isfinite(rural_vals)
        if not rural_finite.any():
            continue
        rural_ref = float(rural_vals[rural_finite].mean())
        finite = np.isfinite(band)
        valid_count += finite
        exceed_count += finite & ((band - rural_ref) > threshold)

    with np.errstate(divide="ignore", invalid="ignore"):
        persistence = np.where(valid_count > 0, exceed_count / valid_count, np.nan)
    return (persistence * 100).astype(np.float32)  # percent of dates


def _persistence_by_class(persistence: np.ndarray, classes: np.ndarray) -> pl.DataFrame:
    """Mean persistence per LCZ class via a single ``np.bincount`` pass."""
    schema = {"lcz": pl.Int16, "persistence": pl.Float64, "n": pl.Int64}
    codes = classes.ravel().astype(np.int32)
    values = persistence.ravel()
    finite = np.isfinite(values) & (codes >= 1) & (codes <= 17)
    if not finite.any():
        return pl.DataFrame(schema=schema)

    safe_codes = np.where(finite, codes, 0)
    counts = np.bincount(safe_codes, minlength=18)
    sums = np.bincount(safe_codes, weights=np.where(finite, values, 0.0), minlength=18)

    rows = [
        {"lcz": cls, "persistence": float(sums[cls] / counts[cls]), "n": int(counts[cls])}
        for cls in range(1, 18) if counts[cls] > 0
    ]
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema_overrides=schema)


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

def _lcz_meta_frame(lang: str, inclusive: bool) -> pl.DataFrame:
    """LCZ lookup (Int32 key) with localised names and palette colours."""
    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    return pl.DataFrame({
        "lcz": pl.Series(list(range(1, 18)), dtype=pl.Int32),
        "lcz_name": pl.Series(get_lcz_names(lang), dtype=pl.String),
        "color": pl.Series(list(colors)[:17], dtype=pl.String),
    })


def _plot_timeseries(df: pl.DataFrame, y_col: str, title: str, ylabel: str, lang: str) -> go.Figure:
    fig = go.Figure()
    if df.is_empty():
        logger.warning("_plot_timeseries: %s is empty; returning an empty chart.", y_col)
        fig.update_layout(title=dict(text=title))
        return fig

    dates = df["date"].to_list()
    values = df[y_col].to_list()
    d, v = _break_date_gaps(dates, values)

    fig.add_trace(go.Scatter(
        x=d, y=v, mode="lines+markers",
        line=dict(color="#333333", width=1.5), marker=dict(size=5),
        hovertemplate=f"{ylabel}: %{{y:.2f}}<extra></extra>",
        name=ylabel,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=lcz_msg("time_label", lang), font=dict(size=14)),
                   gridcolor="#f0f0f0"),
        yaxis=dict(title=dict(text=ylabel, font=dict(size=14)), gridcolor="#f0f0f0",
                   zeroline=True, zerolinecolor="#aaaaaa"),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        margin=dict(l=80, r=40, t=70, b=60),
        hovermode="x unified", hoverlabel=dict(bgcolor="white", font=dict(size=13)),
        showlegend=False,
    )
    return fig


def _plot_value_by_lcz_bar(
    df: pl.DataFrame, value_col: str, lang: str, inclusive: bool, ylab: str, title: str,
) -> go.Figure:
    """Mean ± 95% CI bar per LCZ class (CI collapses to 0 when there's only
    one row per class, e.g. a single-date lcz-method table)."""
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col!r} not found; available: {df.columns}")

    if df.is_empty():
        logger.warning("_plot_value_by_lcz_bar: %s is empty; returning an empty chart.", value_col)
        fig = go.Figure()
        fig.update_layout(title=dict(text=title))
        return fig

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
        agg = df.rename({value_col: "mean_value"}).select(["lcz", "mean_value"]).with_columns(
            std_value=pl.lit(0.0, dtype=pl.Float64), n=pl.lit(1, dtype=pl.Int64),
        ).sort("lcz")

    meta = _lcz_meta_frame(lang, inclusive)
    agg = agg.with_columns(pl.col("lcz").cast(pl.Int32)).join(meta, on="lcz", how="left")

    n = agg["n"].to_numpy()
    std = agg["std_value"].fill_null(0.0).to_numpy()
    se = std / np.sqrt(np.maximum(n, 1))
    tcrit = spstats.t.ppf(0.975, np.maximum(n - 1, 1))
    err = np.where(n > 1, tcrit * se, 0.0)

    fig = go.Figure(go.Bar(
        x=agg["lcz"].cast(pl.String), y=agg["mean_value"],
        marker_color=agg["color"].fill_null("#888888").to_list(),
        marker_line_color="black", marker_line_width=0.5,
        error_y=dict(type="data", array=err.tolist()),
        customdata=agg.select(["lcz_name", "n"]),
        hovertemplate=(
            f"<b>LCZ %{{x}}</b> — %{{customdata[0]}}<br>{ylab}: %{{y:.2f}}"
            "<br>n=%{customdata[1]}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=lcz_msg("lcz_code_label", lang), font=dict(size=14))),
        yaxis=dict(title=dict(text=ylab, font=dict(size=14)), gridcolor="#f0f0f0",
                   zeroline=True, zerolinecolor="#aaaaaa"),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        margin=dict(l=80, r=40, t=70, b=60),
        width=1000, height=600,
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    return fig


def _plot_spatial_heatmap(
    arr: np.ndarray, transform: rasterio.Affine, title: str,
    colorscale: str = "RdBu_r", zmid: Optional[float] = None,
    zmin: Optional[float] = None, zmax: Optional[float] = None,
    colorbar_title: str = "", categories: Optional[np.ndarray] = None,
    crs: Optional[rasterio.crs.CRS] = None,
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
        colorbar=dict(title=dict(text=colorbar_title, side="right")),
        customdata=customdata,
        hovertemplate="value: %{z:.3f}" + hover_extra + "<extra></extra>",
    ))

    geographic = True
    if crs is not None:
        try:
            geographic = bool(crs.is_geographic)
        except Exception:  # noqa: BLE001 - exotic CRS objects
            geographic = True
    xlab = _xmsg("longitude_label", "en") if geographic else _xmsg("easting_label", "en")
    ylab = _xmsg("latitude_label", "en") if geographic else _xmsg("northing_label", "en")

    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=xlab), showgrid=False, zeroline=False),
        yaxis=dict(title=dict(text=ylab), showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=70, r=40, t=70, b=60),
        width=1000, height=800,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    # North arrow + scale bar only make sense on a projected, metric grid.
    add_map_furniture(
        fig,
        bounds=(xmin, ymin, xmax, ymax),
        crs=crs,
        add_scalebar=not geographic,
        add_north_arrow=not geographic,
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
        title=dict(text=lcz_msg("uhi_utfvi_category_title", lang), font=dict(size=20),
                   x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=lcz_msg("uhi_utfvi_category_label", lang))),
        yaxis=dict(title=dict(text=lcz_msg("uhi_percent_pixels_label", lang)),
                   gridcolor="#f0f0f0"),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        margin=dict(l=60, r=40, t=70, b=60),
        width=800, height=500,
    )
    return fig


def _plot_transect(df: pl.DataFrame, lang: str) -> go.Figure:
    fig = go.Figure()
    if df.is_empty():
        logger.warning("_plot_transect: no bins to plot.")
        fig.update_layout(title=dict(text=lcz_msg("uhi_transect_title", lang)))
        return fig

    dates = sorted(df["date"].unique().to_list())

    if len(dates) <= 8:
        partitions = df.partition_by("date", as_dict=True)
        for key, sub in sorted(partitions.items(), key=lambda kv: str(kv[0])):
            d = str(key[0]) if isinstance(key, tuple) else str(key)
            sub = sub.sort("bin_km")
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
        title=dict(text=lcz_msg("uhi_transect_title", lang), font=dict(size=20),
                   x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text=lcz_msg("uhi_distance_label", lang), font=dict(size=14)),
                   gridcolor="#f0f0f0"),
        yaxis=dict(title=dict(text=lcz_msg("uhi_delta_label", lang), font=dict(size=14)),
                   gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#aaaaaa"),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        margin=dict(l=80, r=40, t=70, b=60),
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
        legend=dict(title=dict(text=lcz_msg("time_label", lang))),
    )
    return fig


def _category_counts(cats: np.ndarray, valid: np.ndarray) -> pl.DataFrame:
    """Category frequencies (%) with a deterministic schema."""
    present = cats[valid]
    total = int(present.size)
    if total == 0:
        raise ValueError(_xmsg("no_valid_pixels", "en"))
    labels, counts = np.unique(present.astype(str), return_counts=True)
    return pl.DataFrame({
        "category": pl.Series(labels.tolist(), dtype=pl.String),
        "count": pl.Series(counts.tolist(), dtype=pl.Int64),
    }).with_columns(
        (pl.col("count") / total * 100).alias("percent")
    ).sort("category")


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class LCZUHISurfaceResult:
    """Return type of lcz_uhi_surface for a single method."""
    df: Optional[pl.DataFrame] = None
    fig: Optional[Union[go.Figure, dict[str, go.Figure]]] = None
    array: Optional[np.ndarray] = None


_METHOD_ORDER: tuple[str, ...] = (
    "urban_rural", "percentile", "lcz", "utfvi", "hotspot",
    "transect", "persistence",
)
_METHODS = frozenset(_METHOD_ORDER + ("all",))


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

    Raises
    ------
    ValueError
        On an unknown method, mismatched LCZ/LST grids, an out-of-range
        ``date_index``/``percentile``/``reference_lcz``/``n_bins``/
        ``hotspot_window``, or when the requested method has no valid pixels.

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
        raise ValueError(f"method must be one of {sorted(_METHODS)}, got {method!r}")

    # ── Parameter validation (fail fast, with actionable messages) ─────────
    if not isinstance(reference_lcz, (int, np.integer)) or not 1 <= int(reference_lcz) <= 17:
        raise ValueError(f"reference_lcz must be an integer in 1-17, got {reference_lcz!r}")
    if not 0 <= float(percentile) <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile!r}")
    if not 0.5 <= float(hotspot_confidence) < 1.0:
        raise ValueError(
            f"hotspot_confidence must be in [0.5, 1.0), got {hotspot_confidence!r}"
        )
    if int(hotspot_window) < 3 or int(hotspot_window) % 2 == 0:
        raise ValueError(f"hotspot_window must be an odd integer >= 3, got {hotspot_window!r}")
    if int(n_bins) < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins!r}")
    if not np.isfinite(persistence_threshold):
        raise ValueError(
            f"persistence_threshold must be finite, got {persistence_threshold!r}"
        )
    if date_index is not None and (not isinstance(date_index, (int, np.integer))
                                   or int(date_index) < 0):
        raise ValueError(f"date_index must be a non-negative integer, got {date_index!r}")

    classes = _load_classes(lcz_map)
    lst_stack, dates, units, profile = _load_lst_stack(lst_x)

    if classes.shape != lst_stack.shape[1:]:
        raise ValueError(lcz_msg(
            "uhi_shape_mismatch", lang, lcz_shape=classes.shape, lst_shape=lst_stack.shape[1:],
        ))

    valid_mask = (classes >= 1) & (classes <= 17)
    if not valid_mask.any():
        raise ValueError(
            "The LCZ raster has no pixels in classes 1-17 — check that it is an "
            "LCZ class map (0 = nodata)."
        )
    transform = profile["transform"]
    crs = profile.get("crs")

    n_bands = len(dates)
    di = n_bands - 1 if date_index is None else int(date_index)
    if not 0 <= di < n_bands:
        raise ValueError(
            f"date_index={di} is out of range: the LST stack has {n_bands} "
            f"band(s) (valid indices 0-{n_bands - 1})."
        )

    def _finalize(name: str, df: Optional[pl.DataFrame], fig,
                  strict: bool = True) -> LCZUHISurfaceResult:
        if strict and (df is None or df.is_empty()):
            raise ValueError(
                f"method={name!r} produced no rows. The LCZ raster may lack the "
                "classes this method needs (urban 1-10 / rural 11-16 / "
                f"reference LCZ {reference_lcz}), or the LST stack may be all "
                "nodata for the selected dates."
            )
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

    def _run(name: str, strict: bool = True) -> LCZUHISurfaceResult:
        if name == "urban_rural":
            df = _urban_rural_series(lst_stack, classes, dates)
            fig = _plot_timeseries(
                df, "suhi", lcz_msg("uhi_surface_title", lang), lcz_msg("uhi_delta_label", lang), lang,
            ) if iplot else None
            return _finalize(name, df, fig, strict)

        if name == "percentile":
            df = _urban_rural_series(lst_stack, classes, dates, percentile=percentile)
            fig = _plot_timeseries(
                df, "suhi", lcz_msg("uhi_surface_percentile_title", lang, p=percentile),
                lcz_msg("uhi_delta_label", lang), lang,
            ) if iplot else None
            return _finalize(name, df, fig, strict)

        if name == "lcz":
            df = _lcz_delta_table(lst_stack, classes, dates, reference_lcz)
            fig = _plot_value_by_lcz_bar(
                df, "delta_t", lang, inclusive, lcz_msg("uhi_delta_label", lang),
                lcz_msg("uhi_lcz_title", lang, ref=reference_lcz),
            ) if iplot else None
            return _finalize(name, df, fig, strict)

        if name == "utfvi":
            lst_k = _to_kelvin(lst_stack, units)
            utfvi_stack = _utfvi_stack(lst_k, valid_mask)
            band = utfvi_stack[di]
            finite = valid_mask & np.isfinite(band)
            if not finite.any():
                raise ValueError(
                    f"UTFVI has no valid pixels for band {di} ({dates[di]!r})."
                )
            cats = _classify_utfvi_array(band)
            df = _category_counts(cats, finite)
            array = np.where(valid_mask, band, np.nan)
            if iplot:
                fig = {
                    "map": _plot_spatial_heatmap(
                        array, transform, lcz_msg("uhi_utfvi_title", lang, date=dates[di]),
                        colorscale="RdYlGn_r", zmid=0.0, colorbar_title="UTFVI",
                        crs=crs,
                    ),
                    "bar": _plot_utfvi_category_bar(df, lang),
                }
            else:
                fig = None
            result = _finalize(name, df, fig, strict)
            result.array = array
            return result

        if name == "hotspot":
            band = lst_stack[di]
            gi = _getis_ord_gi_star(band, valid_mask, window=int(hotspot_window))
            cats = _classify_hotspot(gi, float(hotspot_confidence))
            finite = valid_mask & np.isfinite(gi)
            if not finite.any():
                raise ValueError(
                    f"Getis-Ord Gi* has no valid pixels for band {di} "
                    f"({dates[di]!r}) — the band may be constant or all-nodata."
                )
            df = _category_counts(cats, finite)
            array = np.where(valid_mask, gi, np.nan)
            fig = _plot_spatial_heatmap(
                array, transform, lcz_msg("uhi_hotspot_title", lang, date=dates[di]),
                colorscale="RdBu_r", zmid=0.0, colorbar_title="Gi* z-score",
                categories=np.where(valid_mask, cats, ""),
                crs=crs,
            ) if iplot else None
            result = _finalize(name, df, fig, strict)
            result.array = array
            return result

        if name == "transect":
            df = _transect_table(lst_stack, classes, dates, transform, crs, n_bins=int(n_bins))
            fig = _plot_transect(df, lang) if iplot else None
            return _finalize(name, df, fig, strict)

        # "persistence"
        persistence = _suhi_persistence(
            lst_stack, classes, dates, threshold=float(persistence_threshold),
        )
        array = np.where(valid_mask, persistence, np.nan)
        df = _persistence_by_class(array, classes)
        if iplot:
            fig = {
                "map": _plot_spatial_heatmap(
                    array, transform, lcz_msg("uhi_persistence_title", lang, thr=persistence_threshold),
                    colorscale="YlOrRd", zmin=0, zmax=100, colorbar_title="% of dates",
                    crs=crs,
                ),
                "bar": _plot_value_by_lcz_bar(
                    df, "persistence", lang, inclusive, lcz_msg("uhi_persistence_label", lang),
                    lcz_msg("uhi_persistence_by_class_title", lang),
                ),
            }
        else:
            fig = None
        result = _finalize(name, df, fig, strict)
        result.array = array
        return result

    if method == "all":
        results: dict[str, LCZUHISurfaceResult] = {}
        for name in _METHOD_ORDER:
            try:
                results[name] = _run(name, strict=False)
            except ValueError as exc:
                # One inapplicable method (e.g. no rural pixels) must not kill
                # the whole batch, but it must not pass silently either.
                logger.warning("method=%r skipped: %s", name, exc)
                results[name] = _finalize(
                    name,
                    pl.DataFrame({"date": pl.Series([], dtype=pl.String)}),
                    None,
                )
        return results if iplot else {name: r.df for name, r in results.items()}

    result = _run(method)
    return result if iplot else result.df


__all__ = ["lcz_uhi_surface", "LCZUHISurfaceResult", "URBAN_CLASSES", "RURAL_CLASSES", "REFERENCE_LCZ"]
