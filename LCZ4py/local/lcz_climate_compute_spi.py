"""Compute the Standardized Precipitation Index (SPI) at multiple timescales.

Python port of climasus4r's ``sus_climate_compute_spi()``.

SPI (McKee et al. 1993) is a dimensionless precipitation-anomaly index:
negative = drought (<=-1 moderate, <=-1.5 severe, <=-2 extreme), positive =
wet. Runs downstream of monthly precipitation output such as
``lcz_grid_chirps(resolution="monthly")``.

Algorithm
---------
Per municipality and timescale s: (1) s-month rolling sum of precipitation,
(2) fit a zero-inflated gamma to the calibration period via method of
moments (shape = mean^2/var, rate = mean/var on positive values, mixed with
the observed zero proportion p0), (3) transform to standard normal via
``qnorm(p0 + (1-p0) * pgamma(x))``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from LCZ4py._internal.i18n_messages import lcz_msg

logger = logging.getLogger(__name__)


def _transform(x_full: np.ndarray, x_calib: np.ndarray) -> np.ndarray:
    calib = x_calib[~np.isnan(x_calib)]
    nz = calib[calib > 0]
    p0 = np.mean(calib == 0) if len(calib) else np.nan

    out = np.full(len(x_full), np.nan)
    if len(nz) < 4:
        return out

    mu, s2 = nz.mean(), nz.var(ddof=1)
    if not np.isfinite(mu) or not np.isfinite(s2) or s2 <= 0:
        return out

    shape = mu**2 / s2
    rate = mu / s2

    non_na = ~np.isnan(x_full)
    if not non_na.any():
        return out

    xv = x_full[non_na]
    cdf = np.where(xv == 0, p0, p0 + (1 - p0) * stats.gamma.cdf(xv, a=shape, scale=1 / rate))
    eps = np.finfo(float).eps
    cdf = np.clip(cdf, eps, 1 - eps)
    out[non_na] = stats.norm.ppf(cdf)
    return out


def _compute_scale(df: pd.DataFrame, var: str, s: int, ref_start, ref_end, min_n: int) -> np.ndarray:
    out = np.full(len(df), np.nan)
    for loc, idx in df.groupby("code_muni").groups.items():
        pos = df.index.get_indexer(idx)
        x = df.loc[idx, var].to_numpy(dtype="float64")
        rain_roll = pd.Series(x).rolling(window=s, min_periods=s).sum().to_numpy()

        dates = df.loc[idx, "date"]
        in_ref = pd.Series(True, index=dates.index)
        if ref_start is not None:
            in_ref &= dates >= ref_start
        if ref_end is not None:
            in_ref &= dates <= ref_end
        calib = rain_roll[in_ref.to_numpy() & ~np.isnan(rain_roll)]

        if len(calib) < min_n:
            continue
        out[pos] = _transform(rain_roll, calib)
    return out


def lcz_climate_compute_spi(
    df: pd.DataFrame,
    var: str = "rainfall_chirps_mm",
    scales: list[int] = (1, 3, 6, 12),
    ref_start: Optional[pd.Timestamp] = None,
    ref_end: Optional[pd.Timestamp] = None,
    min_n: int = 24,
    lang: str = "en",
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute SPI at one or more timescales from monthly precipitation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``code_muni``, ``date`` (monthly), and ``var``. Typically
        the output of ``lcz_grid_chirps(resolution="monthly")``.
    var : str
        Monthly precipitation column name. Default "rainfall_chirps_mm".
    scales : list[int]
        SPI timescales in months. Default (1, 3, 6, 12).
    ref_start, ref_end : Timestamp, optional
        Calibration period for the gamma fit. None (default) uses all data.
        WMO recommends >= 30 years.
    min_n : int
        Minimum non-NA calibration values per municipality. Default 24.
    lang : str
        Message language. Default "en".
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    pd.DataFrame
        ``df`` with added ``spi_{s}mo`` column(s), one per requested scale.

    Notes
    -----
    Classification: >=2.0 extremely wet, 1.5-1.99 very wet, 1.0-1.49
    moderately wet, -0.99..0.99 near normal, -1..-1.49 moderately dry (D1),
    -1.5..-1.99 severely dry (D2), <=-2.0 extremely dry (D3-D4).
    """
    if var not in df.columns:
        raise ValueError(lcz_msg("spi_var_not_found", lang, var=var, cols=", ".join(df.columns)))
    for col in ("code_muni", "date"):
        if col not in df.columns:
            raise ValueError(lcz_msg("spi_missing_col", lang, col=col))
    if not scales or any(int(s) < 1 for s in scales):
        raise ValueError(lcz_msg("spi_invalid_scales", lang))
    if ref_start is not None and ref_end is not None and ref_start >= ref_end:
        raise ValueError(lcz_msg("spi_invalid_ref_period", lang))

    df = df.sort_values(["code_muni", "date"]).reset_index(drop=True)

    if verbose:
        print(lcz_msg("spi_title", lang))

    for s in scales:
        col_name = f"spi_{s}mo"
        if verbose:
            print(lcz_msg("spi_computing_scale", lang, s=s, col=col_name))
        df[col_name] = _compute_scale(df, var, int(s), ref_start, ref_end, min_n)

    df.attrs["lcz_meta"] = {
        "stage": "climate", "type": "spi", "scales": list(scales), "var": var,
        "history": [f"lcz_climate_compute_spi(): scales={'+'.join(str(s)+'mo' for s in scales)}, var={var}"],
    }

    if verbose:
        first_col = f"spi_{scales[0]}mo"
        n_na = df[first_col].isna().sum()
        print(lcz_msg("spi_done", lang, n_rows=len(df), n_na=n_na, col1=first_col))

    return df
