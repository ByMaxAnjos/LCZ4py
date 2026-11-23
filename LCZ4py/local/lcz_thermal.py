"""
lcz_thermal.py

Thermal comfort and anthropogenic heat estimation for LCZ analysis.

- lcz_utci: Universal Thermal Climate Index (Fiala et al. 2012, ISO 7243)
- lcz_anthropogenic_heat: Anthropogenic heat flux Q_F (W/m²) from LCZ classes

Both functions draw with the native static matplotlib/Agg backend by default
(``renderer="static"``) and save a >=300 dpi PNG/PDF/SVG/TIFF, so they need no
Plotly, Kaleido, Chromium or browser. ``renderer="plotly"`` keeps the original
interactive figures unchanged.

The UTCI implementation here is a *simplified polynomial screening model*
(centred 2nd-order expansion), not the official 6th-order UTCI regression
table shipped by the UTCI operational code — use it for relative comparisons
between LCZ classes, not as a certified comfort assessment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np

import plotly.graph_objects as go

from LCZ4py._internal.lcz_parameters_data import get_lcz_names
from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py._internal import lcz_static
from LCZ4py._internal.lcz_static import resolve_renderer
from LCZ4py._internal.lcz_theme import finalize_export, add_map_furniture

logger = logging.getLogger(__name__)

# ── UTCI thermal stress categories ───────────────────────────────────────────

UTCI_CATEGORIES: list[tuple[float, str, str]] = [
    (-999.0, "extreme_cold_stress", "Extreme cold stress"),
    (-40.0, "very_strong_cold", "Very strong cold stress"),
    (-27.0, "strong_cold", "Strong cold stress"),
    (-13.0, "moderate_cold", "Moderate cold stress"),
    (0.0, "slight_cold", "Slight cold stress"),
    (9.0, "no_thermal_stress", "No thermal stress"),
    (26.0, "moderate_heat", "Moderate heat stress"),
    (32.0, "strong_heat", "Strong heat stress"),
    (38.0, "very_strong_heat", "Very strong heat stress"),
    (46.0, "extreme_heat", "Extreme heat stress"),
]

UTCI_CATEGORY_COLORS = {
    "Extreme cold stress": "#0000FF",
    "Very strong cold stress": "#0066FF",
    "Strong cold stress": "#3399FF",
    "Moderate cold stress": "#66CCFF",
    "Slight cold stress": "#CCFFCC",
    "No thermal stress": "#33CC33",
    "Moderate heat stress": "#FFCC00",
    "Strong heat stress": "#FF6600",
    "Very strong heat stress": "#FF3300",
    "Extreme heat stress": "#CC0000",
}

_UTCI_BOUNDS = np.array([cat[0] for cat in UTCI_CATEGORIES], dtype=np.float64)
_UTCI_LABELS = [cat[2] for cat in UTCI_CATEGORIES]
_UTCI_CATEGORY_INDEX = {label: i for i, label in enumerate(_UTCI_LABELS)}
_UTCI_NA_LABEL = "N/A"


# ── UTCI polynomial coefficients (simplified UTKK-style expansion) ───────────

def _utci_poly(ta: np.ndarray, tmrt: np.ndarray, vas: np.ndarray, es: np.ndarray) -> np.ndarray:
    """Compute a UTCI approximation from a centred 2nd-order expansion.

    This is a numerically stable *screening* approximation of the UTCI
    (Fiala et al. 2012) response surface. It is monotonic in the expected
    directions (warmer air / higher radiant load / higher humidity raise UTCI,
    wind lowers it) and is internally consistent, but it is **not** the
    official operative-temperature regression table.

    Parameters
    ----------
    ta : Air temperature (°C).
    tmrt : Mean radiant temperature (°C).
    vas : Wind speed at 10 m (m/s).
    es : Water vapor pressure (hPa).

    Returns
    -------
    UTCI values (°C).
    """
    # Center variables to improve numerical stability
    ta_c = ta - 25.0   # centered at 25°C
    dt = (tmrt - ta)    # radiant temperature difference
    dt_c = np.clip(dt, -50, 50)  # clamp to valid range
    va_c = np.clip(vas, 0, 30)
    es_c = es - 15.0    # centered at ~15 hPa

    utci = (
        25.0  # base temperature
        + 0.8467 * ta_c
        + 0.5222 * dt_c
        + 0.0281 * es_c
        - 0.0072 * va_c
        + 0.0014 * ta_c**2
        + 0.0008 * dt_c**2
        - 0.0001 * es_c**2
        + 0.0001 * va_c**2
        + 0.0012 * ta_c * dt_c
        - 0.0003 * ta_c * va_c
        - 0.0002 * dt_c * va_c
        + 0.0001 * es_c * va_c
    )
    return utci


def _utci_category(values: np.ndarray) -> np.ndarray:
    """Classify UTCI values into thermal stress categories.

    Bins are closed at the lower edge, matching the UTCI convention. Values
    that are NaN/inf are labelled ``"N/A"`` (they used to be left as
    uninitialised object cells).

    Parameters
    ----------
    values : np.ndarray
        UTCI values (°C).

    Returns
    -------
    np.ndarray
        Category labels (object dtype of strings).
    """
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.ravel()
    with np.errstate(invalid="ignore"):
        idx = np.searchsorted(_UTCI_BOUNDS, flat, side="right") - 1
    idx = np.clip(idx, 0, len(_UTCI_LABELS) - 1)
    out = np.asarray(_UTCI_LABELS, dtype=object)[idx]
    out = np.where(np.isfinite(flat), out, _UTCI_NA_LABEL)
    return out.reshape(arr.shape)


# ── Water vapor pressure from humidity ────────────────────────────────────────

def _saturation_vapor_pressure(ta: np.ndarray) -> np.ndarray:
    """Buck equation for saturation vapor pressure (hPa).

    The denominator is guarded so extreme (unphysical) temperatures cannot
    raise a division-by-zero / overflow warning.

    Parameters
    ----------
    ta : np.ndarray
        Air temperature (°C).

    Returns
    -------
    np.ndarray
        Saturation vapor pressure (hPa).
    """
    ta = np.asarray(ta, dtype=np.float64)
    denom = np.maximum(257.14 + ta, 1e-6)
    return 6.1121 * np.exp((18.678 - ta / 234.5) * ta / denom)


def _vapor_pressure(ta: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Actual water vapor pressure from relative humidity.

    Parameters
    ----------
    ta : np.ndarray
        Air temperature (°C).
    rh : np.ndarray
        Relative humidity (%).

    Returns
    -------
    np.ndarray
        Water vapor pressure (hPa).
    """
    return _saturation_vapor_pressure(ta) * np.clip(rh, 0.0, 100.0) / 100.0


# ── Mean radiant temperature estimation ──────────────────────────────────────

def _estimate_tmrt_from_svf(
    ta: np.ndarray,
    svf: np.ndarray,
) -> np.ndarray:
    """Estimate mean radiant temperature from sky view factor.

    Simplified approximation: Tmrt ≈ Ta + (1 - SVF) * 15.
    In urban canyons (low SVF), Tmrt is warmer than Ta due to
    longwave trapping by building surfaces.

    Parameters
    ----------
    ta : np.ndarray
        Air temperature (°C).
    svf : np.ndarray
        Sky view factor (0–1).

    Returns
    -------
    np.ndarray
        Estimated mean radiant temperature (°C).
    """
    return ta + (1.0 - np.clip(svf, 0.0, 1.0)) * 15.0


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class UTCIResult:
    """Return type for lcz_utci.

    Attributes
    ----------
    values : UTCI index values (°C), flattened to 1-D.
    categories : Stress labels, same shape as ``values``.
    stats : Summary statistics (mean/min/max/std/percentiles/category %).
    plot : matplotlib ``Figure`` for the default static renderer, Plotly
        ``go.Figure`` for ``renderer="plotly"`` (map for 2-D input, histogram
        for 1-D).
    renderer : Which renderer produced ``plot`` ("static" or "plotly").
    """

    values: np.ndarray
    categories: np.ndarray
    stats: dict
    plot: Optional[object] = None
    renderer: str = "static"

    @property
    def fig(self):
        """Alias for ``plot`` (matches ``LCZPlotResult.fig`` elsewhere)."""
        return self.plot


@dataclass
class AnthropogenicHeatResult:
    """Return type for lcz_anthropogenic_heat.

    Attributes
    ----------
    array : Q_F values (W/m²), same shape as the input class array.
    stats : Per-class and area-weighted summary statistics.
    plot : matplotlib ``Figure`` for the default static renderer, Plotly
        ``go.Figure`` for ``renderer="plotly"`` (heatmap for 2-D input, bar
        chart for 1-D).
    renderer : Which renderer produced ``plot`` ("static" or "plotly").
    """

    array: np.ndarray
    stats: dict
    plot: Optional[object] = None
    renderer: str = "static"

    @property
    def fig(self):
        """Alias for ``plot`` (matches ``LCZPlotResult.fig`` elsewhere)."""
        return self.plot


# ── Public entry points ───────────────────────────────────────────────────────

def _broadcast_thermal_inputs(
    air_temp, wind_speed, relative_humidity, mean_radiant_temp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Broadcast the four thermal inputs to a common shape, with validation."""
    try:
        ta, vas, rh = np.broadcast_arrays(
            np.asarray(air_temp, dtype=np.float64),
            np.asarray(wind_speed, dtype=np.float64),
            np.asarray(relative_humidity, dtype=np.float64),
        )
    except ValueError as exc:
        raise ValueError(
            "air_temp, wind_speed and relative_humidity must have compatible "
            f"shapes; got {np.shape(air_temp)}, {np.shape(wind_speed)}, "
            f"{np.shape(relative_humidity)}"
        ) from exc

    tmrt = None
    if mean_radiant_temp is not None:
        try:
            tmrt = np.broadcast_to(
                np.asarray(mean_radiant_temp, dtype=np.float64), ta.shape
            ).copy()
        except ValueError as exc:
            raise ValueError(
                "mean_radiant_temp must broadcast to the shape of air_temp "
                f"({ta.shape}); got {np.shape(mean_radiant_temp)}"
            ) from exc
    return ta, vas, rh, tmrt


def lcz_utci(
    air_temp: Union[float, np.ndarray],
    wind_speed: Union[float, np.ndarray],
    relative_humidity: Union[float, np.ndarray],
    mean_radiant_temp: Optional[Union[float, np.ndarray]] = None,
    *,
    lc_z: Optional[np.ndarray] = None,
    output: Literal["category", "index"] = "category",
    lang: str = "en",
    transform: Optional[Any] = None,
    crs: Optional[Any] = None,
    style: str = "default",
    isave: bool = False,
    save_extension: str = "png",
    dpi: Optional[int] = None,
    renderer: Optional[str] = "static",
) -> UTCIResult:
    """Compute the Universal Thermal Climate Index (UTCI).

    Uses a simplified polynomial approximation of the UTCI response surface
    (after Fiala et al. 2012 / ISO 7243) — see :func:`_utci_poly`. When
    ``mean_radiant_temp`` is not provided, it is estimated from the LCZ sky
    view factor if ``lc_z`` is given, otherwise assumed equal to air
    temperature.

    Parameters
    ----------
    air_temp : float or np.ndarray
        Air temperature (°C).
    wind_speed : float or np.ndarray
        Wind speed at 10 m height (m/s). Clipped to 0-30 m/s.
    relative_humidity : float or np.ndarray
        Relative humidity (%). Clipped to 0-100 %.
    mean_radiant_temp : float or np.ndarray, optional
        Mean radiant temperature (°C). If None, estimated from LCZ SVF
        (requires lc_z) or assumed equal to air_temp.
    lc_z : np.ndarray, optional
        2D array of LCZ classes (1–17). Used to look up sky view factor
        when mean_radiant_temp is None. Nodata (0 or non-finite) yields NaN
        UTCI rather than a made-up class.
    output : {"category", "index"}
        Selects the primary map rendering: discrete stress categories
        (default) or a continuous UTCI index map. Both are always returned in
        the result object.
    lang : str
        Language for category labels ("en", "pt", "es", "zh").
    transform : rasterio.Affine, optional
        Georeferencing of the 2-D input, so the map can be drawn with real
        coordinates, a scale bar and a north arrow. Added for publication
        figures; ignored for 1-D input.
    crs : rasterio CRS, optional
        CRS matching ``transform``. Axis labels and map furniture adapt to
        geographic vs. projected CRS.
    style : str
        Publication style preset: "default", "nature", "science", or
        "generic_bw" (see :mod:`LCZ4py._internal.lcz_theme`).
    isave : bool
        Save the figure under ``LCZ4r_output/``.
    save_extension : str
        Static output format: "png" (default), "pdf", "svg", "tiff"/"tif",
        "jpg"/"jpeg" or "webp". "html" is coerced to "png" for the static
        renderer and only meaningful for ``renderer="plotly"``.
    dpi : int, optional
        Export resolution; never below 300 dpi for static output.
    renderer : {"static", "plotly"}, optional
        "static" (default, also ``None``/``"matplotlib"``/``"agg"``) returns a
        matplotlib ``Figure`` and saves a native raster/vector file without
        Plotly or Kaleido. "plotly" returns the interactive Plotly figure.

    Returns
    -------
    UTCIResult
        values: UTCI index (°C), categories: stress labels,
        stats: summary statistics, plot: figure (static by default).

    Raises
    ------
    ValueError
        On incompatible input shapes, an unknown ``output`` value, a ``lc_z``
        that does not match ``air_temp``, or a non-finite air temperature.
    """
    if output not in ("category", "index"):
        raise ValueError(f"output must be 'category' or 'index', got {output!r}")
    if lang not in ("en", "pt", "es", "zh"):
        lang = "en"
    renderer = resolve_renderer(renderer)

    ta, vas, rh, tmrt = _broadcast_thermal_inputs(
        air_temp, wind_speed, relative_humidity, mean_radiant_temp
    )

    n_out_of_range = int(np.sum(np.isfinite(rh) & ((rh < 0) | (rh > 100))))
    if n_out_of_range:
        logger.warning(
            "%d relative-humidity value(s) outside 0-100%% were clipped.",
            n_out_of_range,
        )
    n_wind = int(np.sum(np.isfinite(vas) & (vas < 0)))
    if n_wind:
        logger.warning("%d negative wind-speed value(s) were clipped to 0.", n_wind)
    vas = np.clip(vas, 0.0, 30.0)

    # Estimate mean radiant temperature if not provided
    if tmrt is None and lc_z is not None:
        lcz_arr = np.asarray(lc_z)
        try:
            lcz_b = np.broadcast_to(lcz_arr, ta.shape)
        except ValueError as exc:
            raise ValueError(
                f"lc_z must broadcast to air_temp shape {ta.shape}; "
                f"got {lcz_arr.shape}"
            ) from exc
        lookup = _get_lookup_table()
        _, names = _get_parameter_lookup_cached()
        svf_idx = names.index("svf_mean")
        lcz_safe = np.clip(np.nan_to_num(lcz_b, nan=0.0).astype(np.int32), 0, 17)
        svf = lookup[lcz_safe, svf_idx].astype(np.float64)
        svf = np.where((lcz_safe >= 1) & np.isfinite(svf), svf, np.nan)
        tmrt = _estimate_tmrt_from_svf(ta, svf)
    elif tmrt is None:
        tmrt = ta.copy()

    es = _vapor_pressure(ta, rh)
    utci_values = np.asarray(_utci_poly(ta, tmrt, vas, es), dtype=np.float64).ravel()
    categories = np.asarray(_utci_category(utci_values)).ravel()

    valid = np.isfinite(utci_values)
    utci_valid = utci_values[valid]
    cat_valid = categories[valid]
    n_valid = int(valid.sum())

    cat_pcts = {
        label: (
            float(np.sum(cat_valid == label) / n_valid * 100) if n_valid else 0.0
        )
        for label in _UTCI_LABELS
    }

    if n_valid:
        stats = {
            "mean": float(np.mean(utci_valid)),
            "min": float(np.min(utci_valid)),
            "max": float(np.max(utci_valid)),
            "std": float(np.std(utci_valid)),
            "p05": float(np.percentile(utci_valid, 5)),
            "p50": float(np.percentile(utci_valid, 50)),
            "p95": float(np.percentile(utci_valid, 95)),
            "n_valid": n_valid,
            "category_percentages": cat_pcts,
        }
    else:
        logger.warning("lcz_utci: no finite UTCI values were produced.")
        stats = {
            "mean": float("nan"), "min": float("nan"), "max": float("nan"),
            "std": float("nan"), "p05": float("nan"), "p50": float("nan"),
            "p95": float("nan"), "n_valid": 0,
            "category_percentages": cat_pcts,
        }

    fig = _plot_utci_map(
        utci_values.reshape(ta.shape) if ta.ndim else utci_values,
        categories.reshape(ta.shape) if ta.ndim else categories,
        lang,
        mode=output,
        transform=transform,
        crs=crs,
        style=style,
        renderer=renderer,
        dpi=dpi,
        isave=isave,
        save_extension=save_extension,
    )

    return UTCIResult(
        values=utci_values,
        categories=categories,
        stats=stats,
        plot=fig,
        renderer=renderer,
    )


def _coerce_class_array(lcz_classes) -> np.ndarray:
    """Convert an LCZ class array to int32, mapping nodata/NaN to 0."""
    arr = np.asarray(lcz_classes)
    if arr.dtype == object or np.issubdtype(arr.dtype, np.floating):
        arr_f = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(arr_f)
        rounded = np.where(finite, np.rint(np.nan_to_num(arr_f, nan=0.0)), 0)
        out = rounded.astype(np.int32)
        n_nan = int((~finite).sum())
        if n_nan:
            logger.info(
                "lcz_anthropogenic_heat: %d non-finite class value(s) treated "
                "as nodata (0).", n_nan,
            )
        return out
    return arr.astype(np.int32)


def lcz_anthropogenic_heat(
    lcz_classes: np.ndarray,
    *,
    method: Literal["simple", "detailed"] = "simple",
    params: Optional[dict] = None,
    lang: str = "en",
    transform: Optional[Any] = None,
    crs: Optional[Any] = None,
    style: str = "default",
    isave: bool = False,
    save_extension: str = "png",
    dpi: Optional[int] = None,
    renderer: Optional[str] = "static",
) -> AnthropogenicHeatResult:
    """Estimate anthropogenic heat flux Q_F (W/m²) from LCZ classes.

    Parameters
    ----------
    lcz_classes : np.ndarray
        2D or 1D array of LCZ class codes (1–17). Nodata (0) and non-finite
        values yield Q_F = 0 (and are excluded from the statistics).
    method : {"simple", "detailed"}
        "simple": literature-derived Q_F ranges per class (midpoint).
        "detailed": energy balance from LCZ parameters (BSF, ISF, HRE).
    params : dict, optional
        Custom Q_F overrides for "simple" method. Keys are LCZ class
        codes (int), values are (min, max) tuples in W/m².
    lang : str
        Language for i18n messages.
    transform : rasterio.Affine, optional
        Georeferencing for the 2-D heatmap (publication maps).
    crs : rasterio CRS, optional
        CRS matching ``transform``.
    style : str
        Publication style preset (see :mod:`LCZ4py._internal.lcz_theme`).
    isave : bool
        Save the figure under ``LCZ4r_output/``.
    save_extension : str
        Static output format: "png" (default), "pdf", "svg", "tiff"/"tif",
        "jpg"/"jpeg" or "webp" ("html" is coerced to "png" for static).
    dpi : int, optional
        Export resolution; never below 300 dpi for static output.
    renderer : {"static", "plotly"}, optional
        "static" (default) returns a matplotlib ``Figure`` and saves a native
        file with no Plotly/Kaleido dependency; "plotly" returns the original
        interactive figure.

    Returns
    -------
    AnthropogenicHeatResult
        array: Q_F values (W/m²), stats: per-class and area-weighted
        means, plot: static heatmap (or bar chart for 1-D input) by default.

    Raises
    ------
    ValueError
        On an unknown ``method``, malformed ``params``, or empty input.
    """
    classes = _coerce_class_array(lcz_classes)
    if classes.size == 0:
        raise ValueError("lcz_classes is empty.")
    renderer = resolve_renderer(renderer)
    original_shape = classes.shape
    flat = classes.ravel()

    # Default Q_F ranges from literature (W/m²)
    default_qf_ranges: dict[int, tuple[float, float]] = {
        1: (60, 120),    # Compact highrise
        2: (40, 90),     # Compact midrise
        3: (25, 60),     # Compact lowrise
        4: (30, 70),     # Open highrise
        5: (20, 50),     # Open midrise
        6: (15, 40),     # Open lowrise
        7: (10, 25),     # Lightweight lowrise
        8: (15, 35),     # Large lowrise
        9: (8, 20),      # Sparsely built
        10: (50, 200),   # Heavy industry
        11: (0, 5),      # Dense trees
        12: (0, 5),      # Scattered trees
        13: (0, 5),      # Bush, scrub
        14: (0, 5),      # Low plants
        15: (0, 5),      # Bare rock or paved
        16: (0, 5),      # Bare soil or sand
        17: (0, 5),      # Water
    }

    if params:
        for key, value in params.items():
            try:
                cls = int(key)
                lo, hi = (float(v) for v in value)
            except (TypeError, ValueError):
                raise ValueError(
                    "params must map LCZ class codes (int) to (min, max) "
                    f"tuples, got {key!r}: {value!r}"
                ) from None
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
                raise ValueError(
                    f"params[{key!r}] must be a finite (min <= max) pair, "
                    f"got ({lo}, {hi})"
                )
            if not 1 <= cls <= 17:
                raise ValueError(f"params key {cls} is not an LCZ class (1-17).")
            default_qf_ranges[cls] = (lo, hi)

    if method == "simple":
        qf = _qf_simple(flat, default_qf_ranges)
    elif method == "detailed":
        qf = _qf_detailed(flat)
    else:
        raise ValueError(f"method must be 'simple' or 'detailed', got {method!r}")

    qf_array = qf.reshape(original_shape)
    stats = _build_qf_stats(qf, flat, default_qf_ranges, lang)
    fig = _plot_qf_heatmap(
        qf_array, classes, lang, transform=transform, crs=crs, style=style,
        renderer=renderer, dpi=dpi, isave=isave,
        save_extension=save_extension,
    )

    return AnthropogenicHeatResult(
        array=qf_array,
        stats=stats,
        plot=fig,
        renderer=renderer,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_parameter_lookup_cached() -> tuple[np.ndarray, list[str]]:
    """Import and cache the parameter lookup from lcz_get_parameters."""
    from LCZ4py.general.lcz_get_parameters import _get_parameter_lookup
    return _get_parameter_lookup()


def _get_lookup_table() -> np.ndarray:
    """Get the (18, n_params) lookup table."""
    table, _ = _get_parameter_lookup_cached()
    return table


def _qf_simple(
    flat: np.ndarray,
    qf_ranges: dict[int, tuple[float, float]],
) -> np.ndarray:
    """Compute Q_F as the midpoint of each LCZ class range (lookup table)."""
    lut = np.zeros(18, dtype=np.float64)
    for cls, (lo, hi) in qf_ranges.items():
        if 1 <= int(cls) <= 17:
            lut[int(cls)] = (float(lo) + float(hi)) / 2.0
    safe = np.where((flat >= 1) & (flat <= 17), flat, 0)
    return lut[safe]


def _qf_detailed(flat: np.ndarray) -> np.ndarray:
    """Compute Q_F from LCZ morphological parameters.

    Q_F = Q_F,bldg + Q_F,traffic + Q_F,people

    - Q_F,bldg = BSF_mean/100 * HRE_mean * 5.0  (W/m²)
    - Q_F,traffic = ISF_mean/100 * 10.0  (W/m²)
    - Q_F,people = urban_fraction * 5.0  (W/m², simplified)
    """
    table, names = _get_parameter_lookup_cached()
    try:
        bsf_idx = names.index("BSF_mean")
        isf_idx = names.index("ISF_mean")
        hre_idx = names.index("HRE_mean")
    except ValueError as exc:
        raise ValueError(
            f"Parameter lookup is missing {exc.args[0]!r}; available: {names}"
        ) from exc

    safe = np.clip(flat, 0, 17)
    bsf = table[safe, bsf_idx]   # %
    isf = table[safe, isf_idx]   # %
    hre = table[safe, hre_idx]   # m

    # Building energy: surface fraction × height × 5 W/m³
    qf_bldg = (bsf / 100.0) * hre * 5.0

    # Traffic: impervious fraction × 10 W/m²
    qf_traffic = (isf / 100.0) * 10.0

    # People: simplified from urban fraction (BSF + ISF proxy)
    urban_frac = np.clip((bsf + isf) / 200.0, 0.0, 1.0)
    qf_people = urban_frac * 5.0

    return qf_bldg + qf_traffic + qf_people


def _build_qf_stats(
    qf: np.ndarray,
    flat: np.ndarray,
    qf_ranges: dict[int, tuple[float, float]],
    lang: str,
) -> dict:
    """Build summary statistics for Q_F output (one bincount pass)."""
    valid = (flat >= 1) & (flat <= 17)
    qf_valid = qf[valid]
    codes = np.where(valid, flat, 0)

    counts = np.bincount(codes, minlength=18)
    sums = np.bincount(codes, weights=np.where(valid, qf, 0.0), minlength=18)

    per_class: dict[int, dict[str, float]] = {}
    for cls in range(1, 18):
        if counts[cls] == 0:
            continue
        mask = codes == cls
        values = qf[mask]
        per_class[cls] = {
            "mean": float(sums[cls] / counts[cls]),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": int(counts[cls]),
        }

    total_mean = float(np.mean(qf_valid)) if qf_valid.size else 0.0
    return {
        "per_class": per_class,
        "total_mean": total_mean,
        "total_pixels": int(valid.sum()),
        "units": "W/m²",
    }


# ── Plotly helpers ────────────────────────────────────────────────────────────

def _format_utci_hover(values: np.ndarray) -> np.ndarray:
    """Object array of hover strings (``'21.4 °C'`` or ``'N/A'``)."""
    out = np.empty(np.shape(values), dtype=object)
    finite = np.isfinite(values)
    out[finite] = [f"{v:.1f} °C" for v in np.asarray(values)[finite].ravel()]
    out[~finite] = _UTCI_NA_LABEL
    return out


def _yaxis_kwargs(ylab: str, georeferenced: bool) -> dict:
    """Y-axis layout; keeps a 1:1 aspect only for georeferenced maps."""
    kwargs: dict[str, Any] = dict(
        title=ylab, gridcolor="#e0e0e0", autorange="reversed",
    )
    if georeferenced:
        kwargs["scaleanchor"] = "x"
        kwargs["scaleratio"] = 1
    return kwargs


def _plot_utci_map(
    values: np.ndarray,
    categories: np.ndarray,
    lang: str,
    mode: str = "category",
    transform: Optional[Any] = None,
    crs: Optional[Any] = None,
    style: str = "default",
    renderer: str = "static",
    dpi: Optional[int] = None,
    isave: bool = False,
    save_extension: str = "png",
):
    """UTCI figure: discrete stress map / index map for 2-D, histogram for 1-D."""
    if np.ndim(values) == 1:
        return _plot_utci_histogram(
            values, lang, style=style, renderer=renderer, dpi=dpi,
            isave=isave, save_extension=save_extension,
        )
    if resolve_renderer(renderer) == "static":
        return _plot_utci_map_static(
            values, categories, lang, mode=mode, transform=transform,
            crs=crs, style=style, dpi=dpi, isave=isave,
            save_extension=save_extension,
        )

    h, w = np.shape(values)
    geo_kwargs: dict[str, Any] = {}
    xlab, ylab = "Column", "Row"
    if transform is not None:
        x0, ymax = transform.c, transform.f
        dx, dy = transform.a, -transform.e
        geo_kwargs = {"x0": x0, "y0": ymax + h * transform.e, "dx": dx, "dy": dy}
        geographic = True
        if crs is not None:
            try:
                geographic = bool(crs.is_geographic)
            except Exception:  # noqa: BLE001
                geographic = True
        xlab = "Longitude [°]" if geographic else "Easting [m]"
        ylab = "Latitude [°]" if geographic else "Northing [m]"

    if mode == "index":
        finite = np.isfinite(values)
        z = np.where(finite, values, np.nan)
        customdata = np.where(finite, categories, _UTCI_NA_LABEL)
        if transform is not None:
            # Heatmap y0 is the south edge, so row 0 (north) must go last.
            z, customdata = np.flipud(z), np.flipud(customdata)
        fig = go.Figure(go.Heatmap(
            z=z,
            colorscale="RdYlBu_r",
            zmid=9.0,  # 9 °C = lower edge of "no thermal stress"
            colorbar=dict(title=dict(text="UTCI [°C]", side="right")),
            customdata=customdata,
            hovertemplate=(
                "<b>UTCI: %{z:.1f} °C</b><br>"
                "%{customdata}<br>"
                "Row: %{y}<br>Col: %{x}<extra></extra>"
            ),
            **geo_kwargs,
        ))
    else:
        # Discrete category raster: z encodes the category index, so the
        # colorbar can show one tick per stress class.
        idx = np.zeros((h, w), dtype=np.float64)
        for label, i in _UTCI_CATEGORY_INDEX.items():
            idx[categories == label] = i
        z = np.where(np.isfinite(values), idx, np.nan)
        hover = _format_utci_hover(values)
        if transform is not None:
            z, hover = np.flipud(z), np.flipud(hover)
        ticks = list(range(len(_UTCI_LABELS)))
        fig = go.Figure(go.Heatmap(
            z=z,
            zmin=-0.5, zmax=len(_UTCI_LABELS) - 0.5,
            colorscale=[
                [stop, UTCI_CATEGORY_COLORS[label]]
                for i, label in enumerate(_UTCI_LABELS)
                for stop in (i / len(_UTCI_LABELS),
                             (i + 1) / len(_UTCI_LABELS))
            ],
            colorbar=dict(
                title=dict(text="Stress class", side="right"),
                tickmode="array",
                tickvals=ticks,
                ticktext=_UTCI_LABELS,
            ),
            customdata=hover,
            hovertemplate=(
                "<b>UTCI: %{customdata}</b><br>"
                "Row: %{y}<br>Col: %{x}<extra></extra>"
            ),
            **geo_kwargs,
        ))

    fig.update_layout(
        title=dict(
            text=lcz_msg("utci_map_title", lang),
            font=dict(size=18), x=0.5, xanchor="center",
        ),
        xaxis=dict(title=xlab, gridcolor="#e0e0e0"),
        yaxis=_yaxis_kwargs(ylab, transform is not None),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=120, t=70, b=60),
        width=min(1200, max(600, w * 6)),
        height=min(900, max(400, h * 6)),
    )

    if transform is not None:
        y0 = float(transform.f + h * transform.e)
        x1 = float(transform.c + w * transform.a)
        add_map_furniture(
            fig,
            bounds=(float(transform.c), y0, x1, float(transform.f)),
            crs=crs,
        )

    return finalize_export(fig, style=style, lang=lang)


def _plot_utci_histogram(
    values: np.ndarray,
    lang: str,
    style: str = "default",
    renderer: str = "static",
    dpi: Optional[int] = None,
    isave: bool = False,
    save_extension: str = "png",
):
    """Histogram of UTCI values with mean/median reference lines."""
    if resolve_renderer(renderer) == "static":
        return _plot_utci_histogram_static(
            values, lang, style=style, dpi=dpi, isave=isave,
            save_extension=save_extension,
        )
    fig = go.Figure()
    finite = values[np.isfinite(values)]
    fig.add_trace(go.Histogram(
        x=finite,
        nbinsx=50,
        marker_color="#4C72B0",
        marker_line_color="white",
        marker_line_width=0.5,
        name="UTCI",
    ))
    if finite.size:
        mean = float(np.mean(finite))
        median = float(np.median(finite))
        fig.add_vline(x=mean, line_dash="dash", line_color="#B2182B",
                      annotation_text=f"mean {mean:.1f}", annotation_position="top")
        fig.add_vline(x=median, line_dash="dot", line_color="#2166AC",
                      annotation_text=f"median {median:.1f}", annotation_position="bottom")
    fig.update_layout(
        title=dict(
            text=lcz_msg("utci_hist_title", lang),
            font=dict(size=18), x=0.5, xanchor="center",
        ),
        xaxis_title="UTCI [°C]",
        yaxis_title="Count",
        bargap=0.02,
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=70, b=55),
        width=900,
        height=500,
        showlegend=False,
    )
    return finalize_export(fig, style=style, lang=lang)


def _plot_qf_heatmap(
    qf: np.ndarray,
    classes: np.ndarray,
    lang: str,
    transform: Optional[Any] = None,
    crs: Optional[Any] = None,
    style: str = "default",
    renderer: str = "static",
    dpi: Optional[int] = None,
    isave: bool = False,
    save_extension: str = "png",
):
    """Anthropogenic heat flux figure: heatmap for 2-D, bar chart for 1-D."""
    if resolve_renderer(renderer) == "static":
        return _plot_qf_static(
            qf, classes, lang, transform=transform, crs=crs, style=style,
            dpi=dpi, isave=isave, save_extension=save_extension,
        )

    lcz_names = get_lcz_names(lang)

    if qf.ndim == 1:
        # Mean Q_F per LCZ class, computed from the class codes (a class with
        # a zero flux is a real value, not "absent" — the old check on qf > 0
        # silently blanked rural and water classes).
        classes_flat = classes.ravel()
        qf_flat = qf.ravel()
        codes = np.where((classes_flat >= 1) & (classes_flat <= 17), classes_flat, 0)
        counts = np.bincount(codes, minlength=18)
        sums = np.bincount(codes, weights=np.where(codes > 0, qf_flat, 0.0), minlength=18)
        present = [c for c in range(1, 18) if counts[c] > 0]
        means = [float(sums[c] / counts[c]) if counts[c] else 0.0 for c in present]

        fig = go.Figure(go.Bar(
            x=[f"LCZ {c}" for c in present],
            y=means,
            marker_color=[
                UTCI_CATEGORY_COLORS["Extreme heat stress"] if c <= 10 else "#1B7837"
                for c in present
            ],
            marker_line_color="black",
            marker_line_width=0.5,
            customdata=[[lcz_names[c - 1], int(counts[c])] for c in present],
            hovertemplate=(
                "<b>%{x}</b> — %{customdata[0]}<br>"
                "Q_F: %{y:.1f} W/m²<br>n=%{customdata[1]}<extra></extra>"
            ),
        ))
        fig.update_layout(
            title=dict(
                text=lcz_msg("qf_bar_title", lang),
                font=dict(size=18), x=0.5, xanchor="center",
            ),
            xaxis_title=lcz_msg("lcz_class_legend", lang),
            yaxis_title="Q_F [W/m²]",
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
            margin=dict(l=80, r=30, t=70, b=60),
            width=1000,
            height=500,
            showlegend=False,
        )
        return finalize_export(fig, style=style, lang=lang)

    h, w = qf.shape
    geo_kwargs: dict[str, Any] = {}
    xlab, ylab = "Column", "Row"
    if transform is not None:
        geo_kwargs = {
            "x0": transform.c,
            "y0": transform.f + h * transform.e,
            "dx": transform.a,
            "dy": -transform.e,
        }
        xlab, ylab = "Easting [m]", "Northing [m]"

    z = np.flipud(qf) if transform is not None else qf
    fig = go.Figure(go.Heatmap(
        z=z,
        colorscale="YlOrRd",
        colorbar=dict(title=dict(text="Q_F [W/m²]", side="right")),
        hovertemplate=(
            "<b>Q_F: %{z:.1f} W/m²</b><br>"
            "Row: %{y}<br>Col: %{x}<extra></extra>"
        ),
        **geo_kwargs,
    ))

    fig.update_layout(
        title=dict(
            text=lcz_msg("qf_map_title", lang),
            font=dict(size=18), x=0.5, xanchor="center",
        ),
        xaxis=dict(title=xlab, gridcolor="#e0e0e0"),
        yaxis=_yaxis_kwargs(ylab, transform is not None),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=90, t=70, b=60),
        width=min(1200, max(600, w * 6)),
        height=min(900, max(400, h * 6)),
    )

    if transform is not None:
        y0 = float(transform.f + h * transform.e)
        x1 = float(transform.c + w * transform.a)
        add_map_furniture(
            fig,
            bounds=(float(transform.c), y0, x1, float(transform.f)),
            crs=crs,
        )

    return finalize_export(fig, style=style, lang=lang)


# ── Native static (matplotlib/Agg) renderers ─────────────────────────────────

def _static_plane_extent(
    shape: tuple, transform: Optional[Any],
) -> tuple[float, float, float, float]:
    """Map extent ``(xmin, xmax, ymin, ymax)`` for a static axes.

    With a geotransform the extent is in map units (row 0 north); without one
    it falls back to pixel indices, mirroring the Plotly helper above.
    """
    h, w = int(shape[0]), int(shape[1])
    if transform is None:
        return (0.0, float(w), 0.0, float(h))
    x0 = float(transform.c)
    ymax = float(transform.f)
    x1 = x0 + w * float(transform.a)
    ymin = ymax + h * float(transform.e)
    return (x0, x1, ymin, ymax)


def _static_map_furniture(
    ax,
    extent: tuple,
    crs,
    *,
    add_scalebar: bool = True,
    add_north_arrow: bool = True,
) -> bool:
    """Axis labels + scale bar + north arrow for a georeferenced panel.

    Calls the individual :mod:`lcz_static` helpers rather than
    ``add_map_furniture_mpl``, whose ``add_scalebar``/``add_north_arrow``
    keyword arguments shadow the module-level functions of the same name
    (same workaround as ``lcz_plot_map._create_static_map``).
    """
    st = lcz_static
    is_geo = st.apply_map_axes(ax, extent, crs)
    if add_scalebar:
        st.add_scalebar(ax, st.metric_width(extent, crs), is_geo)
    if add_north_arrow and not is_geo:
        st.add_north_arrow(ax)
    return is_geo


def _static_pixel_axes(ax) -> None:
    """Label a non-georeferenced raster axes in pixel indices."""
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    # The figure is already sized to the raster's aspect, so equal data-unit
    # scaling keeps cells square without wasting space.
    ax.set_aspect("equal", adjustable="box")


def _utci_category_cmap():
    """Discrete colormap + norm over the UTCI stress categories."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = [UTCI_CATEGORY_COLORS[label] for label in _UTCI_LABELS]
    cmap = ListedColormap(colors, name="utci")
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1.0), cmap.N)
    return cmap, norm


def _utci_category_legend(fig, categories: np.ndarray, spec):
    """Legend of the UTCI stress categories present in ``categories``."""
    from matplotlib.patches import Patch

    present = [lbl for lbl in _UTCI_LABELS if bool(np.any(categories == lbl))]
    if not present:
        return None
    handles = [
        Patch(
            facecolor=UTCI_CATEGORY_COLORS[lbl],
            edgecolor="black",
            linewidth=0.3,
            label=lbl,
        )
        for lbl in present
    ]
    return fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(3, max(1, len(handles))),
        frameon=False,
        fontsize=max(5.5, spec.font_size - 2),
        handlelength=1.1,
        columnspacing=1.0,
        handletextpad=0.5,
    )


def _plot_utci_map_static(
    values: np.ndarray,
    categories: np.ndarray,
    lang: str,
    mode: str = "category",
    transform: Optional[Any] = None,
    crs: Optional[Any] = None,
    style: str = "default",
    dpi: Optional[int] = None,
    isave: bool = False,
    save_extension: str = "png",
):
    """Publication-quality UTCI map drawn with matplotlib (no browser needed)."""
    st = lcz_static
    h, w = np.shape(values)
    extent = _static_plane_extent((h, w), transform)
    aspect = h / max(w, 1)
    categorical = mode != "index"
    if categorical:
        n_present = sum(1 for lbl in _UTCI_LABELS if bool(np.any(categories == lbl)))
        legend_rows = max(1, int(np.ceil(n_present / 3)))
        extra_mm = 10.0 + 8.0 * legend_rows
        bottom = min(0.30, 0.12 + 0.04 * legend_rows)
    else:
        legend_rows = 0
        extra_mm = 16.0
        bottom = 0.14

    with st.style_context(style):
        fig, spec, _ = st.new_figure(
            style, aspect=aspect, extra_height_mm=extra_mm, dpi=dpi,
        )
        ax = st.add_axes(
            fig,
            bottom=bottom,
            top=st.figure_top(True),
        )

        if categorical:
            cmap, norm = _utci_category_cmap()
            idx = np.full((h, w), np.nan, dtype=np.float64)
            for label, i in _UTCI_CATEGORY_INDEX.items():
                idx[categories == label] = i
            idx = np.where(np.isfinite(values), idx, np.nan)
            ax.imshow(
                np.ma.masked_invalid(idx),
                extent=extent,
                origin="upper",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="auto",
            )
            _utci_category_legend(fig, categories, spec)
        else:
            arr = np.where(np.isfinite(values), values, np.nan)
            image = st.plot_continuous_raster(
                ax, arr, extent, colorscale="RdYlBu_r",
            )
            st.add_colorbar(fig, ax, image, units="°C", fontsize=spec.font_size)

        if transform is not None:
            _static_map_furniture(ax, extent, crs)
        else:
            _static_pixel_axes(ax)

        st.add_header(fig, lcz_msg("utci_map_title", lang), None, spec)
        st.save_static(
            fig,
            "lcz_utci_map",
            isave=isave,
            save_extension=save_extension,
            style=style,
            dpi=dpi,
            lang=lang,
        )

    return fig


def _plot_utci_histogram_static(
    values: np.ndarray,
    lang: str,
    style: str = "default",
    dpi: Optional[int] = None,
    isave: bool = False,
    save_extension: str = "png",
):
    """UTCI histogram with bars coloured by thermal-stress category."""
    st = lcz_static
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]

    n_bins = 50
    if finite.size:
        counts, edges = np.histogram(finite, bins=n_bins)
    else:
        counts, edges = np.histogram([], bins=n_bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    bin_labels = _utci_category(centers) if finite.size else np.array([], dtype=object)
    colors = [
        UTCI_CATEGORY_COLORS.get(str(lbl), "#888888") for lbl in bin_labels
    ]

    with st.style_context(style):
        fig, spec, _ = st.new_figure(
            style, aspect=0.55, extra_height_mm=26.0, dpi=dpi,
        )
        ax = st.add_axes(fig, bottom=st.figure_bottom(True), top=st.figure_top(True))

        ax.bar(
            centers, counts, width=widths, align="center",
            color=colors, edgecolor="white", linewidth=0.3,
        )
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linewidth=0.4, color="#dddddd")
        ax.xaxis.grid(False)
        ax.set_xlabel("UTCI [°C]")
        ax.set_ylabel("Count")
        ax.tick_params(labelsize=max(6, spec.font_size - 2), length=3, width=0.6)

        if finite.size:
            mean = float(np.mean(finite))
            median = float(np.median(finite))
            ax.axvline(mean, linestyle="--", linewidth=1.1, color="#B2182B")
            ax.axvline(median, linestyle=":", linewidth=1.1, color="#2166AC")
            y_top = float(counts.max()) if counts.size and counts.max() > 0 else 1.0
            ax.annotate(
                f"mean {mean:.1f}", xy=(mean, y_top), xytext=(4, -2),
                textcoords="offset points", ha="left", va="top",
                fontsize=max(5.5, spec.font_size - 2), color="#B2182B",
            )
            ax.annotate(
                f"median {median:.1f}", xy=(median, y_top), xytext=(-4, -12),
                textcoords="offset points", ha="right", va="top",
                fontsize=max(5.5, spec.font_size - 2), color="#2166AC",
            )
            _utci_category_legend(fig, _utci_category(finite), spec)

        st.add_header(fig, lcz_msg("utci_hist_title", lang), None, spec)
        st.save_static(
            fig,
            "lcz_utci_histogram",
            isave=isave,
            save_extension=save_extension,
            style=style,
            dpi=dpi,
            lang=lang,
        )

    return fig


def _plot_qf_static(
    qf: np.ndarray,
    classes: np.ndarray,
    lang: str,
    transform: Optional[Any] = None,
    crs: Optional[Any] = None,
    style: str = "default",
    dpi: Optional[int] = None,
    isave: bool = False,
    save_extension: str = "png",
):
    """Q_F figure drawn with matplotlib: cividis heatmap or per-class bar chart."""
    st = lcz_static

    if qf.ndim == 1:
        return _plot_qf_bar_static(
            qf, classes, lang, style=style, dpi=dpi, isave=isave,
            save_extension=save_extension,
        )

    h, w = np.shape(qf)
    extent = _static_plane_extent((h, w), transform)
    aspect = h / max(w, 1)

    with st.style_context(style):
        fig, spec, _ = st.new_figure(
            style, aspect=aspect, extra_height_mm=16.0, dpi=dpi,
        )
        ax = st.add_axes(fig, bottom=0.14, top=st.figure_top(True))

        image = st.plot_continuous_raster(ax, qf, extent, colorscale="cividis")
        st.add_colorbar(fig, ax, image, units="W/m²", fontsize=spec.font_size)

        if transform is not None:
            _static_map_furniture(ax, extent, crs)
        else:
            _static_pixel_axes(ax)

        st.add_header(fig, lcz_msg("qf_map_title", lang), None, spec)
        st.save_static(
            fig,
            "lcz_anthropogenic_heat_map",
            isave=isave,
            save_extension=save_extension,
            style=style,
            dpi=dpi,
            lang=lang,
        )

    return fig


def _plot_qf_bar_static(
    qf: np.ndarray,
    classes: np.ndarray,
    lang: str,
    style: str = "default",
    dpi: Optional[int] = None,
    isave: bool = False,
    save_extension: str = "png",
):
    """Mean Q_F per LCZ class as a colourblind-safe sequential bar chart."""
    st = lcz_static

    classes_flat = np.asarray(classes).ravel()
    qf_flat = np.asarray(qf).ravel()
    codes = np.where(
        (classes_flat >= 1) & (classes_flat <= 17), classes_flat, 0,
    )
    counts = np.bincount(codes, minlength=18)
    sums = np.bincount(
        codes, weights=np.where(codes > 0, qf_flat, 0.0), minlength=18,
    )
    present = [c for c in range(1, 18) if counts[c] > 0]
    means = np.array(
        [float(sums[c] / counts[c]) if counts[c] else 0.0 for c in present],
        dtype=np.float64,
    )

    with st.style_context(style):
        fig, spec, _ = st.new_figure(
            style, aspect=0.58, extra_height_mm=10.0, dpi=dpi,
        )
        ax = st.add_axes(
            fig,
            bottom=0.28 if len(present) > 8 else 0.24,
            top=st.figure_top(True),
        )

        if means.size:
            lo, hi = float(means.min()), float(means.max())
            if hi <= lo:
                hi = lo + 1.0
            bar_colors = st.mpl_cmap("cividis")(
                (means - lo) / (hi - lo) if hi > lo else np.zeros_like(means)
            )
        else:
            bar_colors = "#888888"

        ax.bar(
            np.arange(len(present)), means, color=bar_colors,
            edgecolor="black", linewidth=0.5, width=0.8,
        )
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linewidth=0.4, color="#dddddd")
        ax.xaxis.grid(False)
        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels(
            [f"LCZ {c}" for c in present],
            rotation=90 if len(present) > 8 else 0,
            fontsize=max(5.5, spec.font_size - 2),
        )
        ax.set_xlabel(lcz_msg("lcz_class_legend", lang))
        ax.set_ylabel("Q_F [W/m²]")
        ax.tick_params(
            axis="y", labelsize=max(6, spec.font_size - 2), length=3, width=0.6,
        )

        st.add_header(fig, lcz_msg("qf_bar_title", lang), None, spec)
        st.save_static(
            fig,
            "lcz_anthropogenic_heat_bar",
            isave=isave,
            save_extension=save_extension,
            style=style,
            dpi=dpi,
            lang=lang,
        )

    return fig


__all__ = [
    "lcz_utci",
    "lcz_anthropogenic_heat",
    "UTCIResult",
    "AnthropogenicHeatResult",
    "UTCI_CATEGORIES",
    "UTCI_CATEGORY_COLORS",
]
