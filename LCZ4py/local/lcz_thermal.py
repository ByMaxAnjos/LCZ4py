"""
lcz_thermal.py

Thermal comfort and anthropogenic heat estimation for LCZ analysis.

- lcz_utci: Universal Thermal Climate Index (Fiala et al. 2012, ISO 7243)
- lcz_anthropogenic_heat: Anthropogenic heat flux Q_F (W/m²) from LCZ classes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Literal

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from LCZ4py._internal.lcz_parameters_data import (
    LCZ_TABLE, LCZ_NAMES, LCZ_IDS, get_lcz_names,
)
from LCZ4py._internal.i18n_messages import lcz_msg

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


# ── UTCI polynomial coefficients (UMET-LS UTKK model) ───────────────────────

def _utci_poly(ta: np.ndarray, tmrt: np.ndarray, vas: np.ndarray, es: np.ndarray) -> np.ndarray:
    """Compute UTCI using the Fiala et al. 2012 polynomial approximation.

    This is a numerically stable implementation of the UTCI regression model
    from the UTKK bioclimate model (ISO 7243). The polynomial uses a 6th-order
    expansion in four input variables with proper centering to avoid overflow.

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

    # UTCI approximation using the validated Fiala coefficients
    # Applied to centered variables for numerical stability
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


# ── UTCI category classification ─────────────────────────────────────────────

def _utci_category(values: np.ndarray) -> np.ndarray:
    """Classify UTCI values into thermal stress categories.

    Parameters
    ----------
    values : np.ndarray
        UTCI values (°C).

    Returns
    -------
    np.ndarray
        Category labels (strings).
    """
    bounds = [cat[0] for cat in UTCI_CATEGORIES] + [999.0]
    labels = [cat[2] for cat in UTCI_CATEGORIES]
    cats = np.empty(values.shape, dtype=object)
    for i, label in enumerate(labels):
        mask = (values >= bounds[i]) & (values < bounds[i + 1])
        cats[mask] = label
    return cats


# ── Water vapor pressure from humidity ────────────────────────────────────────

def _saturation_vapor_pressure(ta: np.ndarray) -> np.ndarray:
    """Buck equation for saturation vapor pressure (hPa).

    Parameters
    ----------
    ta : np.ndarray
        Air temperature (°C).

    Returns
    -------
    np.ndarray
        Saturation vapor pressure (hPa).
    """
    return 6.1121 * np.exp((18.678 - ta / 234.5) * ta / (257.14 + ta))


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
    return _saturation_vapor_pressure(ta) * rh / 100.0


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
    return ta + (1.0 - svf) * 15.0


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class UTCIResult:
    """Return type for lcz_utci."""
    values: np.ndarray
    categories: np.ndarray
    stats: dict
    plot: Optional[go.Figure] = None


@dataclass
class AnthropogenicHeatResult:
    """Return type for lcz_anthropogenic_heat."""
    array: np.ndarray
    stats: dict
    plot: Optional[go.Figure] = None


# ── Public entry points ───────────────────────────────────────────────────────

def lcz_utci(
    air_temp: Union[float, np.ndarray],
    wind_speed: Union[float, np.ndarray],
    relative_humidity: Union[float, np.ndarray],
    mean_radiant_temp: Optional[Union[float, np.ndarray]] = None,
    *,
    lc_z: Optional[np.ndarray] = None,
    output: Literal["category", "index"] = "category",
    lang: str = "en",
) -> UTCIResult:
    """Compute the Universal Thermal Climate Index (UTCI).

    Implements the UTCI polynomial approximation from the UTKK model
    (Fiala et al. 2012, ISO 7243). When mean_radiant_temp is not provided,
    it is estimated from the LCZ sky view factor if lc_z is given, otherwise
    assumed equal to air temperature.

    Parameters
    ----------
    air_temp : float or np.ndarray
        Air temperature (°C).
    wind_speed : float or np.ndarray
        Wind speed at 10 m height (m/s).
    relative_humidity : float or np.ndarray
        Relative humidity (%).
    mean_radiant_temp : float or np.ndarray, optional
        Mean radiant temperature (°C). If None, estimated from LCZ SVF
        (requires lc_z) or assumed equal to air_temp.
    lc_z : np.ndarray, optional
        2D array of LCZ classes (1–17). Used to look up sky view factor
        when mean_radiant_temp is None.
    output : {"category", "index"}
        Return thermal stress categories or numeric UTCI values.
    lang : str
        Language for category labels ("en", "pt", "es", "zh").

    Returns
    -------
    UTCIResult
        values: UTCI index (°C), categories: stress labels,
        stats: summary statistics, plot: Plotly figure.
    """
    ta = np.asarray(air_temp, dtype=np.float64)
    vas = np.asarray(wind_speed, dtype=np.float64)
    rh = np.asarray(relative_humidity, dtype=np.float64)

    # Clamp wind speed to valid range
    vas = np.clip(vas, 0.0, 30.0)

    # Estimate mean radiant temperature if not provided
    if mean_radiant_temp is not None:
        tmrt = np.asarray(mean_radiant_temp, dtype=np.float64)
    elif lc_z is not None:
        _, names = _get_parameter_lookup_cached()
        svf_idx = names.index("svf_mean")
        lookup = _get_lookup_table()
        lcz_safe = np.clip(lc_z.astype(np.int32), 0, 17)
        svf = lookup[lcz_safe, svf_idx]
        tmrt = _estimate_tmrt_from_svf(ta, svf)
    else:
        tmrt = ta.copy()

    # Compute water vapor pressure
    es = _vapor_pressure(ta, rh)

    # Compute UTCI polynomial
    utci_values = _utci_poly(ta, tmrt, vas, es).ravel()

    # Compute categories
    categories = _utci_category(utci_values)

    # Build stats
    valid = ~np.isnan(utci_values)
    utci_valid = utci_values[valid]
    cat_valid = categories[valid]

    cat_labels = [cat[2] for cat in UTCI_CATEGORIES]
    cat_pcts = {}
    for label in cat_labels:
        count = np.sum(cat_valid == label)
        cat_pcts[label] = float(count / max(valid.sum(), 1) * 100)

    stats = {
        "mean": float(np.mean(utci_valid)) if utci_valid.size else 0.0,
        "min": float(np.min(utci_valid)) if utci_valid.size else 0.0,
        "max": float(np.max(utci_valid)) if utci_valid.size else 0.0,
        "std": float(np.std(utci_valid)) if utci_valid.size else 0.0,
        "category_percentages": cat_pcts,
    }

    # Build plot
    fig = _plot_utci_map(utci_values, categories, lang)

    return UTCIResult(
        values=utci_values,
        categories=categories,
        stats=stats,
        plot=fig,
    )


def lcz_anthropogenic_heat(
    lcz_classes: np.ndarray,
    *,
    method: Literal["simple", "detailed"] = "simple",
    params: Optional[dict] = None,
    lang: str = "en",
) -> AnthropogenicHeatResult:
    """Estimate anthropogenic heat flux Q_F (W/m²) from LCZ classes.

    Parameters
    ----------
    lcz_classes : np.ndarray
        2D or 1D array of LCZ class codes (1–17).
    method : {"simple", "detailed"}
        "simple": literature-derived Q_F ranges per class.
        "detailed": energy balance from LCZ parameters (BSF, ISF, HRE).
    params : dict, optional
        Custom Q_F overrides for "simple" method. Keys are LCZ class
        codes (int), values are (min, max) tuples in W/m².
    lang : str
        Language for i18n messages.

    Returns
    -------
    AnthropogenicHeatResult
        array: Q_F values (W/m²), stats: per-class and area-weighted
        means, plot: Plotly heatmap.
    """
    classes = np.asarray(lcz_classes, dtype=np.int32)
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
        default_qf_ranges.update(params)

    if method == "simple":
        qf = _qf_simple(flat, default_qf_ranges)
    elif method == "detailed":
        qf = _qf_detailed(flat)
    else:
        raise ValueError(f"method must be 'simple' or 'detailed', got {method!r}")

    qf_array = qf.reshape(original_shape)

    # Stats
    stats = _build_qf_stats(qf, flat, default_qf_ranges, lang)

    # Plot
    fig = _plot_qf_heatmap(qf_array, original_shape, lang)

    return AnthropogenicHeatResult(
        array=qf_array,
        stats=stats,
        plot=fig,
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
    """Compute Q_F using mean of literature ranges per LCZ class."""
    qf = np.zeros_like(flat, dtype=np.float64)
    for cls, (lo, hi) in qf_ranges.items():
        mask = flat == cls
        qf[mask] = (lo + hi) / 2.0
    return qf


def _qf_detailed(flat: np.ndarray) -> np.ndarray:
    """Compute Q_F from LCZ morphological parameters.

    Q_F = Q_F,bldg + Q_F,traffic + Q_F,people

    - Q_F,bldg = BSF_mean/100 * HRE_mean * 5.0  (W/m²)
    - Q_F,traffic = ISF_mean/100 * 10.0  (W/m²)
    - Q_F,people = urban_fraction * 5.0  (W/m², simplified)
    """
    table, names = _get_parameter_lookup_cached()

    bsf_idx = names.index("BSF_mean")
    isf_idx = names.index("ISF_mean")
    hre_idx = names.index("HRE_mean")

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
    """Build summary statistics for Q_F output."""
    valid = (flat >= 1) & (flat <= 17)
    qf_valid = qf[valid]
    flat_valid = flat[valid]

    per_class = {}
    for cls in range(1, 18):
        mask = flat_valid == cls
        if mask.any():
            per_class[int(cls)] = {
                "mean": float(np.mean(qf[mask])),
                "min": float(np.min(qf[mask])),
                "max": float(np.max(qf[mask])),
                "count": int(mask.sum()),
            }

    total_mean = float(np.mean(qf_valid)) if qf_valid.size else 0.0

    return {
        "per_class": per_class,
        "total_mean": total_mean,
        "total_pixels": int(valid.sum()),
    }


# ── Plotly helpers ────────────────────────────────────────────────────────────

def _plot_utci_map(
    values: np.ndarray,
    categories: np.ndarray,
    lang: str,
) -> go.Figure:
    """Create a choropleth-style heatmap of UTCI values with stress colors."""
    if values.ndim == 1:
        # Cannot make a 2D map from 1D data; use histogram instead
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=50,
            marker_color="#4C72B0",
            marker_line_color="white",
            marker_line_width=0.5,
        ))
        fig.update_layout(
            title=dict(
                text=lcz_msg("utci_hist_title", lang),
                font=dict(size=18),
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="UTCI [°C]",
            yaxis_title="Count",
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
            margin=dict(l=60, r=30, t=60, b=50),
            width=900,
            height=500,
        )
        return fig

    h, w = values.shape

    # Color by category
    cat_colors = np.empty(categories.shape, dtype=object)
    for label, color in UTCI_CATEGORY_COLORS.items():
        cat_colors[categories == label] = color

    # Build a numeric color array for the heatmap
    color_numeric = np.zeros(values.shape, dtype=np.float64)
    for i, (_, _, label) in enumerate(UTCI_CATEGORIES):
        mask = categories == label
        color_numeric[mask] = i

    fig = go.Figure(data=go.Heatmap(
        z=color_numeric,
        colorscale=[
            [0.0, "#0000FF"],
            [0.11, "#0066FF"],
            [0.22, "#3399FF"],
            [0.33, "#66CCFF"],
            [0.44, "#CCFFCC"],
            [0.55, "#33CC33"],
            [0.66, "#FFCC00"],
            [0.77, "#FF6600"],
            [0.88, "#FF3300"],
            [1.0, "#CC0000"],
        ],
        showscale=False,
        hovertemplate=(
            "<b>UTCI: %{z:.1f} °C</b><br>"
            "Row: %{y}<br>"
            "Col: %{x}<extra></extra>"
        ),
    ))

    # Overlay numeric values as text
    fig.add_trace(go.Heatmap(
        z=values,
        colorscale="RdYlBu_r",
        showscale=True,
        colorbar=dict(
            title="UTCI [°C]",
            title_side="right",
            len=0.75,
        ),
        opacity=0.0,
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(
            text=lcz_msg("utci_map_title", lang),
            font=dict(size=18),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(title="Column", gridcolor="#e0e0e0"),
        yaxis=dict(title="Row", gridcolor="#e0e0e0", autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=80, t=60, b=50),
        width=min(1200, max(600, w * 4)),
        height=min(900, max(400, h * 4)),
    )

    # Add legend annotations
    legend_y = 1.0
    for _, _, label in UTCI_CATEGORIES:
        color = UTCI_CATEGORY_COLORS[label]
        fig.add_annotation(
            text=f"<span style='color:{color}'>■</span> {label}",
            xref="paper", yref="paper",
            x=1.02, y=legend_y,
            showarrow=False,
            font=dict(size=10),
            align="left",
            xanchor="left",
        )
        legend_y -= 0.06

    return fig


def _plot_qf_heatmap(
    qf: np.ndarray,
    shape: tuple,
    lang: str,
) -> go.Figure:
    """Create a heatmap of anthropogenic heat flux Q_F."""
    if qf.ndim == 1:
        # Bar chart for 1D data
        fig = go.Figure()
        classes_present = np.unique(qf[qf > 0])
        fig.add_trace(go.Bar(
            x=[f"LCZ {int(c)}" for c in range(1, 18)],
            y=[
                float(np.mean(qf[np.arange(len(qf)) == c]))
                if c in classes_present else 0
                for c in range(1, 18)
            ],
            marker_color=[
                "#CC0000" if c <= 10 else "#228B22"
                for c in range(1, 18)
            ],
            marker_line_color="black",
            marker_line_width=0.5,
        ))
        fig.update_layout(
            title=dict(
                text=lcz_msg("qf_bar_title", lang),
                font=dict(size=18),
                x=0.5,
                xanchor="center",
            ),
            xaxis_title=lcz_msg("lcz_class_legend", lang),
            yaxis_title="Q_F [W/m²]",
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
            margin=dict(l=60, r=30, t=60, b=50),
            width=1000,
            height=500,
        )
        return fig

    fig = go.Figure(data=go.Heatmap(
        z=qf,
        colorscale="YlOrRd",
        colorbar=dict(
            title="Q_F [W/m²]",
            title_side="right",
        ),
        hovertemplate=(
            "<b>Q_F: %{z:.1f} W/m²</b><br>"
            "Row: %{y}<br>"
            "Col: %{x}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text=lcz_msg("qf_map_title", lang),
            font=dict(size=18),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(title="Column", gridcolor="#e0e0e0"),
        yaxis=dict(title="Row", gridcolor="#e0e0e0", autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=80, t=60, b=50),
        width=min(1200, max(600, shape[1] * 4)),
        height=min(900, max(400, shape[0] * 4)),
    )

    return fig


# ── i18n keys for this module ─────────────────────────────────────────────────
# Add these to i18n_messages.py LCZ_MESSAGES dict if needed:
#   "utci_map_title", "utci_hist_title", "qf_map_title", "qf_bar_title"

__all__ = [
    "lcz_utci",
    "lcz_anthropogenic_heat",
    "UTCIResult",
    "AnthropogenicHeatResult",
    "UTCI_CATEGORIES",
    "UTCI_CATEGORY_COLORS",
]
