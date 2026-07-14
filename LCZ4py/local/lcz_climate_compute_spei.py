"""Compute the Standardized Precipitation-Evapotranspiration Index (SPEI) at
multiple timescales.

Python port of climasus4r's ``sus_climate_compute_spei()``.

SPEI (Vicente-Serrano et al. 2010) extends SPI by using the water balance
(precipitation - potential evapotranspiration) instead of precipitation
alone, making it sensitive to warming-amplified drought.

Algorithm
---------
Per municipality and timescale s: (1) s-month rolling sum of the water
balance D = precipitation - PET, (2) transform to standard normal via the
empirical CDF (Hazen plotting position, p=(rank-0.5)/n) over the
calibration period. This mirrors the R source exactly — despite its
docstring mentioning a log-logistic fit, the actual implementation uses the
non-parametric Hazen ECDF throughout, which is valid for any shape of D and
guarantees mean~0/std~1 over the calibration period by construction.

PET, when not supplied directly, is computed with Thornthwaite (1948) from
monthly mean temperature alone (no day-length/latitude correction — reasonable
for tropical Brazil where day length is close to 12h year-round).
"""

from __future__ import annotations

import logging
import calendar
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)

_VALID_PET_METHODS = ("column", "thornthwaite")


def _transform(x_full: np.ndarray, x_calib: np.ndarray) -> np.ndarray:
    """Hazen-corrected empirical CDF -> standard normal."""
    calib = x_calib[~np.isnan(x_calib)]
    n = len(calib)
    out = np.full(len(x_full), np.nan)
    if n < 4:
        return out

    calib_sorted = np.sort(calib)
    non_na = ~np.isnan(x_full)
    xv = x_full[non_na]

    ranks = np.searchsorted(calib_sorted, xv, side="right")
    p_hazen = (ranks - 0.5) / n
    p_hazen = np.clip(p_hazen, 1e-6, 1 - 1e-6)
    out[non_na] = stats.norm.ppf(p_hazen)
    return out


def _thornthwaite_pet(df: pd.DataFrame, temp_var: str) -> pd.Series:
    pet = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby("code_muni").groups.items():
        t_mo = df.loc[idx, temp_var].to_numpy(dtype="float64")
        dates = df.loc[idx, "date"]
        month_num = dates.dt.month.to_numpy()

        t_month_means = np.array([
            np.mean(np.clip(t_mo[month_num == m], 0, None)) if (month_num == m).any() else np.nan
            for m in range(1, 13)
        ])
        i_monthly = (np.nan_to_num(t_month_means) / 5) ** 1.514
        heat_index = np.nansum(i_monthly)

        if heat_index <= 0:
            pet.loc[idx] = 0.0
            continue

        a = 6.75e-7 * heat_index**3 - 7.71e-5 * heat_index**2 + 1.792e-2 * heat_index + 0.49239
        ndm = np.array([calendar.monthrange(d.year, d.month)[1] for d in dates])
        pet_unadj = np.where((t_mo <= 0) | np.isnan(t_mo), 0.0, 16 * (10 * t_mo / heat_index) ** a)
        pet.loc[idx] = pet_unadj * (ndm / 30)
    return pet


def _compute_scale(df: pd.DataFrame, rain_var: str, pet_var: str, s: int, ref_start, ref_end, min_n: int) -> np.ndarray:
    out = np.full(len(df), np.nan)
    d_series = df[rain_var].to_numpy(dtype="float64") - df[pet_var].to_numpy(dtype="float64")
    df = df.assign(_D=d_series)

    for loc, idx in df.groupby("code_muni").groups.items():
        pos = df.index.get_indexer(idx)
        d = df.loc[idx, "_D"].to_numpy(dtype="float64")
        d_roll = pd.Series(d).rolling(window=s, min_periods=s).sum().to_numpy()

        dates = df.loc[idx, "date"]
        in_ref = pd.Series(True, index=dates.index)
        if ref_start is not None:
            in_ref &= dates >= ref_start
        if ref_end is not None:
            in_ref &= dates <= ref_end
        calib = d_roll[in_ref.to_numpy() & ~np.isnan(d_roll)]

        if len(calib) < min_n:
            continue
        out[pos] = _transform(d_roll, calib)
    return out


def lcz_climate_compute_spei(
    df: pd.DataFrame,
    rain_var: str = "rainfall_chirps_mm",
    pet_var: str = "pet_mm",
    pet_method: str = "column",
    temp_var: str = "tair_dry_bulb_c",
    scales: list[int] = (1, 3, 6, 12),
    ref_start: Optional[pd.Timestamp] = None,
    ref_end: Optional[pd.Timestamp] = None,
    min_n: int = 24,
    lang: str = "en",
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute SPEI at one or more timescales from precipitation and PET.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``code_muni``, ``date`` (monthly), ``rain_var``, and
        either ``pet_var`` (pet_method="column") or ``temp_var``
        (pet_method="thornthwaite").
    rain_var : str
        Monthly precipitation column (mm). Default "rainfall_chirps_mm".
    pet_var : str
        Monthly PET column (mm), used when pet_method="column". Default "pet_mm".
    pet_method : str
        "column" (default) or "thornthwaite" (derive PET from temp_var).
    temp_var : str
        Monthly mean temperature column (C), used for Thornthwaite PET.
        Default "tair_dry_bulb_c".
    scales : list[int]
        SPEI timescales in months. Default (1, 3, 6, 12).
    ref_start, ref_end : Timestamp, optional
        Calibration period. None (default) uses all data.
    min_n : int
        Minimum non-NA calibration values per municipality. Default 24.
    lang : str
        Message language. Default "en".
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    pd.DataFrame
        ``df`` with added ``spei_{s}mo`` column(s).
    """
    if pet_method not in _VALID_PET_METHODS:
        raise ValueError(lcz_msg("spei_invalid_pet_method", lang))
    for col in ("code_muni", "date", rain_var):
        if col not in df.columns:
            raise ValueError(lcz_msg("spei_missing_col", lang, col=col))
    if pet_method == "column" and pet_var not in df.columns:
        raise ValueError(lcz_msg("spei_pet_col_missing", lang, pet_var=pet_var))
    if pet_method == "thornthwaite" and temp_var not in df.columns:
        raise ValueError(lcz_msg("spei_temp_col_missing", lang, temp_var=temp_var))
    if not scales or any(int(s) < 1 for s in scales):
        raise ValueError(lcz_msg("spi_invalid_scales", lang))

    df = df.sort_values(["code_muni", "date"]).reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    if verbose:
        print(lcz_msg("spei_title", lang))

    effective_pet_var = pet_var
    if pet_method == "thornthwaite":
        df["_pet_thornthwaite"] = _thornthwaite_pet(df, temp_var)
        effective_pet_var = "_pet_thornthwaite"

    for s in scales:
        col_name = f"spei_{s}mo"
        if verbose:
            print(lcz_msg("spei_computing_scale", lang, s=s, col=col_name))
        df[col_name] = _compute_scale(df, rain_var, effective_pet_var, int(s), ref_start, ref_end, min_n)

    if pet_method == "thornthwaite":
        df = df.drop(columns="_pet_thornthwaite")

    df.attrs["lcz_meta"] = {
        "stage": "climate", "type": "spei", "scales": list(scales), "pet_method": pet_method,
        "history": [f"lcz_climate_compute_spei(): scales={'+'.join(str(s)+'mo' for s in scales)}, pet={pet_method}"],
    }

    if verbose:
        first_col = f"spei_{scales[0]}mo"
        print(lcz_msg("spei_done", lang, n_rows=len(df), n_na=df[first_col].isna().sum(), col1=first_col))

    return df
