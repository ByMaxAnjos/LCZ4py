"""
LCZ time-series utility functions — shared backend for lcz_ts, lcz_anomaly,
lcz_uhi_intensity, lcz_interp_map, and lcz_interp_eval.
"""

from __future__ import annotations
import logging
import os
import re
from typing import Any, Optional, Sequence, Union

import numpy as np
import polars as pl
import rasterio

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from LCZ4py._internal.lcz_parameters_data import LCZ_COLORS, LCZ_COLORBLIND

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"

# Pre-compiled regex for flexible column name matching
_LAT_RE = re.compile(r"^(lat|latitude)$", re.IGNORECASE)
_LON_RE = re.compile(r"^(lon|long|longitude)$", re.IGNORECASE)
_DATE_RE = re.compile(r"^(date|time|timestamp|datetime)$", re.IGNORECASE)


# ── Raster loader ─────────────────────────────────────────────────────────────

def load_lcz_raster(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
) -> rasterio.io.DatasetReader:
    """Open an LCZ raster from a path, or return an already-open dataset."""
    if isinstance(x, (str, os.PathLike)):
        return rasterio.open(str(x))
    return x


# ── DataFrame normalisation ───────────────────────────────────────────────────

def normalise_input_df(df: Any, var: str, station_id: str) -> pl.DataFrame:
    """Rename and coerce lat/lon/date/var/station columns to canonical names.

    Parameters
    ----------
    df : pd.DataFrame, pl.DataFrame, or dict-like
        Input observation table.
    var : str
        Column name for the meteorological variable.
    station_id : str
        Column name for the station identifier.

    Returns
    -------
    pl.DataFrame
        With columns: latitude, longitude, date, var_interp, station.
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df) if hasattr(df, "to_pandas") else pl.DataFrame(df)

    cols_lower = {c.lower(): c for c in df.columns}
    lat_col = next((cols_lower[k] for k in cols_lower if _LAT_RE.match(k)), None)
    lon_col = next((cols_lower[k] for k in cols_lower if _LON_RE.match(k)), None)
    date_col = next((cols_lower[k] for k in cols_lower if _DATE_RE.match(k)), None)

    if not all([lat_col, lon_col, date_col]):
        raise ValueError("Missing lat/lon/date columns — check column names.")

    mapping = {
        lat_col: "latitude",
        lon_col: "longitude",
        date_col: "date",
        var: "var_interp",
        station_id: "station",
    }
    out = df.rename({k: v for k, v in mapping.items() if k in df.columns})
    # date may already be Datetime from pl.from_pandas — only parse if string
    date_is_str = out["date"].dtype == pl.Utf8
    return out.with_columns([
        pl.col("latitude").cast(pl.Float64),
        pl.col("longitude").cast(pl.Float64),
        pl.col("var_interp").cast(pl.Float64),
        pl.col("date").str.to_datetime(strict=False) if date_is_str
        else pl.col("date").cast(pl.Datetime("ns")),
        pl.col("station").cast(pl.String),
    ])


def normalise_missing(df: pl.DataFrame, cols: tuple = ("var_interp",)) -> pl.DataFrame:
    """Replace common missing-value sentinels with null."""
    str_nans = ["nan", "na", "n/a", "null", "missing", "", "-9999", "-99", "inf", "-inf", "."]
    exprs = []
    for c in cols:
        if c not in df.columns:
            continue
        expr = pl.col(c)
        if df[c].dtype == pl.Utf8:
            expr = pl.when(expr.str.to_lowercase().is_in(str_nans)).then(None).otherwise(expr)
        expr = (
            pl.when(expr.is_in([-9999.0, -99.0, 9999.0, 999.0, np.inf, -np.inf]))
            .then(None)
            .otherwise(expr.cast(pl.Float64))
        )
        exprs.append(expr.alias(c))
    return df.with_columns(exprs) if exprs else df


# ── Time averaging ────────────────────────────────────────────────────────────

def time_average(
    df: pl.DataFrame,
    avg_time: str = "hour",
    type_cols: Sequence[str] = ("station",),
    value_col: str = "var_interp",
) -> pl.DataFrame:
    """Polars group_by_dynamic time averaging — faster than pandas resample."""
    polars_freq = {
        "hour": "1h", "day": "1d", "week": "1w",
        "month": "1mo", "year": "1y",
    }.get(avg_time, "1h")
    present = [c for c in type_cols if c in df.columns]
    return (
        df.sort("date")
          .group_by_dynamic("date", every=polars_freq, closed="left", group_by=present)
          .agg(pl.col(value_col).mean())
    )


def select_by_date(
    df: pl.DataFrame,
    year=None, month=None, day=None, hour=None,
    start=None, end=None,
) -> pl.DataFrame:
    """Filter a Polars DataFrame by date/time components or a date range."""
    ldf = df.lazy()
    d = pl.col("date")
    if year is not None:
        ldf = ldf.filter(d.dt.year().is_in(
            [int(y) for y in (year if hasattr(year, "__iter__") else [year])]
        ))
    if month is not None:
        _m = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        ms = [
            _m[str(m).lower()[:3]] if str(m).isalpha() else int(m)
            for m in (month if hasattr(month, "__iter__") else [month])
        ]
        ldf = ldf.filter(d.dt.month().is_in(ms))
    if day is not None:
        ldf = ldf.filter(d.dt.day().is_in(
            [int(d_) for d_ in (day if hasattr(day, "__iter__") else [day])]
        ))
    if hour is not None:
        ldf = ldf.filter(d.dt.hour().is_in(
            [int(h) for h in (hour if hasattr(hour, "__iter__") else [hour])]
        ))
    if start is not None:
        from datetime import datetime
        start_dt = datetime.fromisoformat(str(start)[:19]) if isinstance(start, str) else start
        ldf = ldf.filter(d >= pl.lit(start_dt))
    if end is not None:
        from datetime import datetime
        end_dt = datetime.fromisoformat(str(end)[:19]) if isinstance(end, str) else end
        ldf = ldf.filter(d <= pl.lit(end_dt))
    return ldf.collect()


# ── Homogeneity kernel (Numba if available, else scipy fallback) ──────────────

if HAS_NUMBA:
    @njit(fastmath=True)
    def _calc_homogeneity_numba(arr: np.ndarray, kernel_size: int) -> np.ndarray:
        h, w = arr.shape
        pad = kernel_size // 2
        out = np.empty((h, w), dtype=np.float32)
        for i in prange(h):
            for j in range(w):
                center = arr[i, j]
                if center <= 0:
                    out[i, j] = np.nan
                    continue
                match = 0
                total = 0
                for di in range(-pad, pad + 1):
                    for dj in range(-pad, pad + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if arr[ni, nj] == center:
                                match += 1
                            total += 1
                out[i, j] = match / total if total > 0 else np.nan
        return out
else:
    def _calc_homogeneity_numba(arr: np.ndarray, kernel_size: int) -> np.ndarray:
        from scipy.ndimage import generic_filter

        def _match_ratio(values: np.ndarray) -> float:
            center = values[len(values) // 2]
            return float("nan") if center <= 0 else float(np.mean(values == center))

        return generic_filter(
            arr.astype(np.float32), _match_ratio,
            size=kernel_size, mode="constant", cval=0,
        )


# ── Point extraction ──────────────────────────────────────────────────────────

def extract_lcz_at_points(
    ds: rasterio.io.DatasetReader,
    lons: np.ndarray,
    lats: np.ndarray,
    method: str = "simple",
    kernel_size: int = 5,
    threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract LCZ class values at point locations.

    Parameters
    ----------
    ds : rasterio dataset
        Opened LCZ raster (EPSG:4326).
    lons, lats : np.ndarray
        Point longitudes and latitudes.
    method : {"simple", "bilinear", "two.step"}
        ``"simple"`` — nearest-neighbour (default);
        ``"bilinear"`` — bilinear interpolation;
        ``"two.step"`` — nearest-neighbour with homogeneity filter.
    kernel_size : int
        Kernel radius for two-step homogeneity check.
    threshold : float
        Minimum homogeneity fraction to keep a point (two-step only).

    Returns
    -------
    lcz : np.ndarray[int32]
    mask : np.ndarray[bool]  — True = point is valid
    """
    arr = ds.read(1)
    transform = ds.transform
    nodata = ds.nodata if ds.nodata is not None else 0

    cols, rows = ~transform * (lons, lats)
    cols = np.clip(np.rint(cols).astype(np.int32), 0, arr.shape[1] - 1)
    rows = np.clip(np.rint(rows).astype(np.int32), 0, arr.shape[0] - 1)

    if method == "bilinear":
        cols_f, rows_f = ~transform * (lons, lats)
        c0 = np.clip(np.floor(cols_f).astype(np.int32), 0, arr.shape[1] - 2)
        r0 = np.clip(np.floor(rows_f).astype(np.int32), 0, arr.shape[0] - 2)
        dx = (cols_f - c0).astype(np.float32)
        dy = (rows_f - r0).astype(np.float32)
        v00 = arr[r0, c0].astype(np.float32)
        v01 = arr[r0, c0 + 1].astype(np.float32)
        v10 = arr[r0 + 1, c0].astype(np.float32)
        v11 = arr[r0 + 1, c0 + 1].astype(np.float32)
        lcz = np.rint(
            v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy)
            + v10 * (1 - dx) * dy + v11 * dx * dy
        ).astype(np.int32)
        return np.where(lcz == nodata, 0, lcz), np.ones(len(lcz), dtype=bool)

    lcz = arr[rows, cols]
    lcz = np.where(lcz == nodata, 0, lcz).astype(np.int32)

    if method in ("two.step", "two_step"):
        hom = _calc_homogeneity_numba(arr.astype(np.int32), kernel_size)
        station_hom = hom[rows, cols]
        keep = np.isfinite(station_hom) & (station_hom >= threshold)
        return lcz, keep

    return lcz, np.ones(len(lcz), dtype=bool)


# ── Spatial helpers ───────────────────────────────────────────────────────────

def utm_epsg_for(ds: rasterio.io.DatasetReader) -> str:
    """Return the UTM EPSG code covering the centre of a rasterio dataset."""
    bounds = ds.bounds
    lon = (bounds.left + bounds.right) / 2
    lat = (bounds.bottom + bounds.top) / 2
    zone = int(np.floor((lon + 180) / 6) + 1)
    return f"EPSG:{326 if lat >= 0 else 327}{zone:02d}"


# ── Color palette ─────────────────────────────────────────────────────────────

def lcz_palette(inclusive: bool = False) -> dict[str, str]:
    """Return a dict mapping LCZ class strings ('1'–'17') to hex colors."""
    colors = LCZ_COLORBLIND if inclusive else LCZ_COLORS
    return {str(i + 1): c for i, c in enumerate(colors)}


# ── Temporal grouping (by=) ───────────────────────────────────────────────────

_BY_VALUES = (
    "year", "season", "seasonyear", "month", "monthyear",
    "weekday", "weekend", "site", "daylight", "dst",
)

_BY_ORDER: dict[str, list[str]] = {
    "season":  ["DJF", "MAM", "JJA", "SON"],
    "month":   ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"],
    "weekend": ["Weekday", "Weekend"],
    "daylight": ["Daytime", "Nighttime"],
    "dst":     ["DST", "Standard"],
}

_MONTH_IDX = {m: i for i, m in enumerate(_BY_ORDER["month"])}
_SEASON_IDX = {s: i for i, s in enumerate(_BY_ORDER["season"])}


def _solar_elevation(
    lat_deg: float, lon_deg: float,
    utc_hours: np.ndarray, doy: np.ndarray,
) -> np.ndarray:
    """Approximate solar elevation (°) using the NOAA/Spencer formula.

    Timestamps are assumed to be in UTC. See
    https://gml.noaa.gov/grad/solcalc/ for the reference implementation.
    """
    lat = np.deg2rad(lat_deg)
    B = 2 * np.pi * (doy - 1) / 365
    # Spencer (1971): the polynomial already yields declination in RADIANS
    dec = (
        0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
        - 0.006758 * np.cos(2 * B) + 0.000907 * np.sin(2 * B)
        - 0.002697 * np.cos(3 * B) + 0.001480 * np.sin(3 * B)
    )
    eot = 229.18 * (
        0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B)
        - 0.014615 * np.cos(2 * B) - 0.040890 * np.sin(2 * B)
    )
    tst = utc_hours + lon_deg / 15 + eot / 60   # true solar time (h)
    ha = np.deg2rad(15 * (tst - 12))             # hour angle
    sin_alt = (np.sin(lat) * np.sin(dec)
               + np.cos(lat) * np.cos(dec) * np.cos(ha))
    return np.rad2deg(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))


def add_by_column(
    df: pl.DataFrame,
    by: str,
    lat: float = 0.0,
    lon: float = 0.0,
) -> pl.DataFrame:
    """Add a ``'_by'`` grouping column for temporal faceting.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain a ``'date'`` column of type ``Datetime``.
    by : str
        Split key — one of ``"year"``, ``"season"``, ``"seasonyear"``,
        ``"month"``, ``"monthyear"``, ``"weekday"``, ``"weekend"``,
        ``"site"``, ``"daylight"``, ``"dst"``.
    lat, lon : float
        Mean station latitude/longitude (degrees).  Used only for
        ``"daylight"`` (solar elevation) and ``"dst"`` (hemisphere).
    """
    if by not in _BY_VALUES:
        raise ValueError(f"by= must be one of {_BY_VALUES!r}, got {by!r}")

    d = pl.col("date")
    season_expr = (
        pl.when(d.dt.month().is_in([12, 1, 2])).then(pl.lit("DJF"))
          .when(d.dt.month().is_in([3, 4, 5])).then(pl.lit("MAM"))
          .when(d.dt.month().is_in([6, 7, 8])).then(pl.lit("JJA"))
          .otherwise(pl.lit("SON"))
    )

    if by == "year":
        return df.with_columns(d.dt.year().cast(pl.String).alias("_by"))
    if by == "season":
        return df.with_columns(season_expr.alias("_by"))
    if by == "seasonyear":
        return df.with_columns(
            (season_expr + pl.lit(" ") + d.dt.year().cast(pl.String)).alias("_by")
        )
    if by == "month":
        return df.with_columns(d.dt.strftime("%b").alias("_by"))
    if by == "monthyear":
        return df.with_columns(d.dt.strftime("%b %Y").alias("_by"))
    if by == "weekday":
        return df.with_columns(d.dt.strftime("%A").alias("_by"))
    if by == "weekend":
        return df.with_columns(
            pl.when(d.dt.weekday().is_in([6, 7]))
              .then(pl.lit("Weekend")).otherwise(pl.lit("Weekday"))
              .alias("_by")
        )
    if by == "site":
        return df.with_columns(pl.col("station").alias("_by"))
    if by == "daylight":
        doy = df["date"].dt.ordinal_day().to_numpy().astype(float)
        hrs = (df["date"].dt.hour() + df["date"].dt.minute() / 60).to_numpy().astype(float)
        elev = _solar_elevation(lat, lon, hrs, doy)
        return df.with_columns(
            pl.Series("_by", np.where(elev > 0, "Daytime", "Nighttime"))
        )
    # by == "dst"
    # Approximate: April–October = DST in Northern Hemisphere; invert for South.
    # Note: timestamps are assumed UTC per NOAA convention.
    dst_months = list(range(4, 11)) if lat >= 0 else [1, 2, 10, 11, 12]
    return df.with_columns(
        pl.when(d.dt.month().is_in(dst_months))
          .then(pl.lit("DST")).otherwise(pl.lit("Standard"))
          .alias("_by")
    )


def by_sorted_groups(series: pl.Series, by: str) -> list[str]:
    """Return unique ``'_by'`` values in natural display order."""
    unique = set(series.drop_nulls().to_list())
    if by in _BY_ORDER:
        return [g for g in _BY_ORDER[by] if g in unique]
    if by == "year":
        return sorted(unique, key=int)
    if by == "seasonyear":
        def _syk(s: str):
            p = s.split()
            return (int(p[1]), _SEASON_IDX.get(p[0], 99))
        return sorted(unique, key=_syk)
    if by == "monthyear":
        def _myk(s: str):
            p = s.split()
            return (int(p[1]), _MONTH_IDX.get(p[0], 99))
        return sorted(unique, key=_myk)
    return sorted(unique)


# ── Time-series line helpers ──────────────────────────────────────────────────

def _break_gaps_numpy(
    dates_np: np.ndarray,
    values_np: np.ndarray,
    threshold_h: float,
) -> tuple[list, list]:
    """Insert ``(None, None)`` pairs at large time gaps so Plotly breaks the line.

    Parameters
    ----------
    dates_np : numpy datetime64 array, already sorted ascending.
    values_np : float array, same length.
    threshold_h : gap size in hours above which a break is inserted.
    """
    if len(dates_np) <= 1:
        return list(dates_np), list(values_np)
    gaps_h = np.diff(dates_np.astype("datetime64[s]")).astype(np.float64) / 3600
    break_idx = np.where(gaps_h > threshold_h)[0]
    if len(break_idx) == 0:
        return list(dates_np), list(values_np)
    out_d: list = []
    out_v: list = []
    prev = 0
    for idx in break_idx:
        out_d.extend(dates_np[prev : idx + 1].tolist())
        out_v.extend(values_np[prev : idx + 1].tolist())
        out_d.append(None)
        out_v.append(None)
        prev = idx + 1
    out_d.extend(dates_np[prev:].tolist())
    out_v.extend(values_np[prev:].tolist())
    return out_d, out_v


def _daylight_intervals_batch(
    lat_deg: float,
    lon_deg: float,
    unique_dates: list,
) -> list[list[tuple[float, float]]]:
    """Return UTC daytime intervals for each date in *unique_dates*.

    Valid for all latitudes and longitudes globally.  Timestamps are assumed
    to be UTC (NOAA convention).

    Each date returns a list of ``(start_utc_h, end_utc_h)`` tuples:

    - Normal day (sun below at UTC midnight): ``[(rise_h, set_h)]``
    - Split day (sun above at UTC midnight, typical for UTC+east zones):
      ``[(0.0, set_h), (rise_h, 24.0)]``
    - Polar day: ``[(0.0, 24.0)]``
    - Polar night: ``[]``
    - Solstice boundary (single crossing): ``[(0.0, set_h)]`` or
      ``[(rise_h, 24.0)]``
    """
    if not unique_dates:
        return []
    hrs = np.linspace(0, 24, 289)   # 5-minute resolution
    n = len(unique_dates)
    doys = np.array([d.timetuple().tm_yday for d in unique_dates], dtype=float)
    hrs_2d = np.broadcast_to(hrs[:, None], (289, n)).copy()
    doy_2d = np.broadcast_to(doys[None, :], (289, n)).copy()
    elev_2d = _solar_elevation(lat_deg, lon_deg, hrs_2d, doy_2d)  # (289, n)
    above_2d = elev_2d > 0

    results = []
    for j in range(n):
        col = above_2d[:, j]
        crossings = np.where(np.diff(col))[0]

        if len(crossings) == 0:
            results.append([(0.0, 24.0)] if col[0] else [])
        elif len(crossings) == 1:
            idx = crossings[0]
            if col[0]:   # above at midnight → sets once → [(0, set)]
                results.append([(0.0, float(hrs[idx]))])
            else:        # below at midnight → rises once → [(rise, 24)]
                results.append([(float(hrs[idx]), 24.0)])
        elif len(crossings) == 2:
            c0, c1 = int(crossings[0]), int(crossings[1])
            if col[0]:
                # above at midnight → sets at c0, rises at c1 → split day
                results.append([(0.0, float(hrs[c0])), (float(hrs[c1]), 24.0)])
            else:
                # below at midnight → rises at c0, sets at c1 → normal day
                results.append([(float(hrs[c0]), float(hrs[c1]))])
        else:
            # 3+ crossings (rare, near polar boundary): collect all intervals
            intervals: list[tuple[float, float]] = []
            idx = 0
            if col[0]:
                intervals.append((0.0, float(hrs[crossings[0]])))
                idx = 1
            while idx + 1 < len(crossings):
                intervals.append((float(hrs[crossings[idx]]),
                                   float(hrs[crossings[idx + 1]])))
                idx += 2
            if idx < len(crossings) and col[-1]:
                intervals.append((float(hrs[crossings[idx]]), 24.0))
            results.append(intervals)
    return results


__all__ = [
    "load_lcz_raster",
    "normalise_input_df",
    "normalise_missing",
    "time_average",
    "select_by_date",
    "extract_lcz_at_points",
    "utm_epsg_for",
    "lcz_palette",
    "OUTPUT_DIR",
    # temporal grouping
    "_BY_VALUES",
    "add_by_column",
    "by_sorted_groups",
    # time-series line helpers
    "_break_gaps_numpy",
    "_daylight_intervals_batch",
]
