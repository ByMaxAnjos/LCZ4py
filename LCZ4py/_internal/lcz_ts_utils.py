"""
LCZ time-series utility functions — shared backend for lcz_ts, lcz_anomaly,
lcz_uhi_intensity, lcz_interp_map, and lcz_interp_eval.

Everything here is defensive by design: inputs are validated with explicit
errors (never a bare ``ColumnNotFoundError``), timezone-aware timestamps are
normalised to naive UTC, sentinel values become nulls, and raster lookups
reproject points when the LCZ raster is not in EPSG:4326.
"""

from __future__ import annotations
import logging
import os
import re
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import polars as pl
import rasterio

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover - depends on the local environment
    HAS_NUMBA = False

from LCZ4py._internal.lcz_parameters_data import LCZ_COLORS, LCZ_COLORBLIND

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"

# Pre-compiled regex for flexible column name matching
_LAT_RE = re.compile(r"^(lat|latitude)$", re.IGNORECASE)
_LON_RE = re.compile(r"^(lon|long|longitude)$", re.IGNORECASE)
_DATE_RE = re.compile(r"^(date|time|timestamp|datetime)$", re.IGNORECASE)

# Canonical column names produced by normalise_input_df()
CANON_COLS: tuple[str, ...] = (
    "latitude", "longitude", "date", "var_interp", "station",
)

# Numeric sentinels treated as missing by normalise_missing()
_NUMERIC_SENTINELS: tuple[float, ...] = (
    -9999.0, -999.0, -99.0, 9999.0, 999.0, np.inf, -np.inf,
)
_STRING_SENTINELS: tuple[str, ...] = (
    "nan", "na", "n/a", "null", "none", "missing", "", "-9999", "-9999.0",
    "-999", "-99", "9999", "999", "inf", "-inf", ".", "..", "nodata",
)

# Extra text formats tried when the default ISO parser leaves holes.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
    "%Y%m%d%H%M", "%Y%m%d",
)

# Bounded default for thread pools (see resolve_workers)
_MAX_POOL = 8


# ── Small validation helpers ─────────────────────────────────────────────────

def resolve_workers(
    n_jobs: int = -1,
    max_workers: Optional[int] = None,
    cap: int = _MAX_POOL,
) -> int:
    """Resolve a worker count, bounded by ``min(cap, cpu_count)``.

    Parameters
    ----------
    n_jobs : int
        ``-1`` or ``0`` = all available CPUs (bounded by *cap*); any positive
        value is used as-is, still bounded by ``cap``.
    max_workers : int, optional
        Explicit override (wins over *n_jobs*), clamped to ``[1, cap]``.
    cap : int
        Upper bound, so a 64-core box does not spawn 64 processes for a
        handful of tasks.

    Returns
    -------
    int
        At least 1.
    """
    cpu = os.cpu_count() or 4
    if max_workers is not None:
        try:
            requested = int(max_workers)
        except (TypeError, ValueError):
            raise ValueError(f"max_workers must be an integer, got {max_workers!r}") from None
        if requested < 1:
            raise ValueError(f"max_workers must be >= 1, got {requested}")
        return max(1, min(requested, cap))

    try:
        requested = int(n_jobs)
    except (TypeError, ValueError):
        raise ValueError(f"n_jobs must be an integer, got {n_jobs!r}") from None
    if requested in (-1, 0):
        return max(1, min(cpu, cap))
    if requested < 1:
        raise ValueError(f"n_jobs must be -1 or >= 1, got {requested}")
    return max(1, min(requested, cap))


def require_nonempty(df: pl.DataFrame, what: str = "data") -> pl.DataFrame:
    """Raise a descriptive ``ValueError`` when *df* has no rows."""
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"{what}: expected a polars DataFrame, got {type(df).__name__}")
    if df.is_empty():
        raise ValueError(
            f"No {what} left after cleaning/filtering — check the date range, "
            "the station coordinates and the variable column (all-null values?)."
        )
    return df


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    """Return the first exact match, else the first case-insensitive match."""
    cols = list(columns)
    for cand in candidates:
        if cand in cols:
            return cand
    lowered = {c.lower(): c for c in cols}
    for cand in candidates:
        hit = lowered.get(str(cand).lower())
        if hit is not None:
            return hit
    return None


# ── Raster loader ─────────────────────────────────────────────────────────────

def load_lcz_raster(
    x: Union[str, os.PathLike, rasterio.io.DatasetReader],
) -> rasterio.io.DatasetReader:
    """Open an LCZ raster from a path, or return an already-open dataset.

    Raises
    ------
    TypeError
        If *x* is neither a path nor an open rasterio dataset.
    FileNotFoundError
        If the path does not exist (instead of rasterio's opaque error).
    """
    if isinstance(x, (str, os.PathLike)):
        path = str(x)
        if not os.path.exists(path):
            raise FileNotFoundError(f"LCZ raster not found: {path}")
        ds = rasterio.open(path)
    elif isinstance(x, rasterio.io.DatasetReader):
        ds = x
    else:
        raise TypeError(
            "x must be a path to an LCZ GeoTIFF or an open rasterio dataset, "
            f"got {type(x).__name__}"
        )

    if ds.count < 1:
        raise ValueError(f"LCZ raster has no bands: {getattr(ds, 'name', x)!r}")
    if ds.crs is None:
        logger.warning(
            "LCZ raster has no CRS — station coordinates are assumed to be in "
            "the raster's native coordinate system (expected EPSG:4326)."
        )
    return ds


# ── Datetime coercion ─────────────────────────────────────────────────────────

def _coerce_datetime_series(s: pl.Series) -> pl.Series:
    """Coerce a Series to naive ``Datetime('ns')``.

    Accepts ``Datetime`` (tz-aware values are converted to UTC and the zone
    dropped), ``Date``, ISO-ish strings, and epoch numbers (s or ms).
    """
    dtype = s.dtype

    if dtype == pl.Datetime:
        if getattr(dtype, "time_zone", None) is not None:
            s = s.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        return s.cast(pl.Datetime("ns"))

    if dtype == pl.Date:
        return s.cast(pl.Datetime("ns"))

    if dtype in (pl.Utf8, pl.Categorical, pl.Enum):
        text = s.cast(pl.String)
        frame = text.alias("_d").to_frame()
        exprs = [pl.col("_d").str.to_datetime(strict=False)]
        exprs += [
            pl.col("_d").str.to_datetime(format=fmt, strict=False)
            for fmt in _DATE_FORMATS
        ]
        parsed = frame.select(pl.coalesce(exprs).alias("_d"))["_d"]
        if getattr(parsed.dtype, "time_zone", None) is not None:
            parsed = parsed.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        lost = parsed.null_count() - s.null_count()
        if lost > 0:
            logger.warning(
                "%d timestamp(s) could not be parsed and were dropped; "
                "use ISO-8601 (YYYY-MM-DD HH:MM:SS) to avoid ambiguity.",
                lost,
            )
        return parsed.cast(pl.Datetime("ns"))

    if dtype.is_numeric():
        numeric = s.cast(pl.Float64)
        peak = float(numeric.abs().max() or 0.0)
        time_unit = "ms" if peak > 1e11 else "s"
        return pl.from_epoch(
            numeric.cast(pl.Int64), time_unit=time_unit
        ).cast(pl.Datetime("ns"))

    raise TypeError(
        "Unsupported date column dtype "
        f"{dtype!r}: pass a datetime/date column or ISO-8601 strings."
    )


def _coerce_naive_utc(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
    """Return *df* with *col* as naive UTC ``Datetime('ns')``."""
    if col not in df.columns:
        raise ValueError(f"Column {col!r} not found; available: {df.columns}")
    series = df[col]
    if series.dtype == pl.Datetime and getattr(series.dtype, "time_zone", None) is None:
        return df
    return df.with_columns(_coerce_datetime_series(series).alias(col))


# ── DataFrame normalisation ───────────────────────────────────────────────────

def _as_polars(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if hasattr(df, "to_pandas"):
        return pl.from_pandas(df)
    try:
        return pl.DataFrame(df)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(
            f"Cannot convert {type(df).__name__} to a polars DataFrame: {exc}"
        ) from exc


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
        With columns ``latitude, longitude, date, var_interp, station``.
        Rows with unusable coordinates or timestamps are dropped (and logged).

    Raises
    ------
    ValueError
        If any required column is missing, or the input is empty.
    """
    if df is None:
        raise ValueError("data_frame is required (got None).")

    out = _as_polars(df)
    if out.is_empty():
        raise ValueError("data_frame is empty — nothing to analyse.")

    lat_col = next((c for c in out.columns if _LAT_RE.match(c)), None)
    lon_col = next((c for c in out.columns if _LON_RE.match(c)), None)
    date_col = next((c for c in out.columns if _DATE_RE.match(c)), None)
    var_col = _find_column(out.columns, [var] if var else [])
    station_col = _find_column(out.columns, [station_id] if station_id else [])

    missing: list[str] = []
    if lat_col is None:
        missing.append("latitude (one of: lat, latitude)")
    if lon_col is None:
        missing.append("longitude (one of: lon, long, longitude)")
    if date_col is None:
        missing.append("date (one of: date, time, timestamp, datetime)")
    if var_col is None:
        missing.append(f"variable {var!r}" + (" (var= is empty)" if not var else ""))
    if station_col is None:
        missing.append(
            f"station id {station_id!r}"
            + (" (station_id= is empty)" if not station_id else "")
        )
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + "; ".join(missing)
            + f". Available columns: {out.columns}"
        )

    mapping = {
        lat_col: "latitude",
        lon_col: "longitude",
        date_col: "date",
        var_col: "var_interp",
        station_col: "station",
    }
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(
            "Column roles overlap (e.g. var= or station_id= points at the "
            f"lat/lon/date column): {mapping}"
        )

    out = out.rename(mapping)

    n_before = out.height
    out = out.with_columns([
        pl.col("latitude").cast(pl.Float64, strict=False),
        pl.col("longitude").cast(pl.Float64, strict=False),
        pl.col("var_interp").cast(pl.Float64, strict=False),
        pl.col("station").cast(pl.String),
        _coerce_datetime_series(out["date"]).alias("date"),
    ])

    # Timestamps are mandatory: polars' group_by_dynamic rejects nulls.
    n_null_date = out["date"].null_count()
    if n_null_date:
        logger.warning("Dropping %d row(s) with a missing/unparseable timestamp.", n_null_date)
        out = out.filter(pl.col("date").is_not_null())

    nan_coords = pl.col("latitude").is_null() | pl.col("longitude").is_null()
    bad_coords = (
        ~pl.col("latitude").is_between(-90.0, 90.0)
        | ~pl.col("longitude").is_between(-180.0, 180.0)
    )
    n_bad = out.filter(nan_coords | bad_coords).height
    if n_bad:
        logger.warning(
            "Dropping %d row(s) with missing or out-of-range coordinates "
            "(latitude must be in [-90, 90], longitude in [-180, 180]).",
            n_bad,
        )
        out = out.filter(~(nan_coords | bad_coords))

    if out.height != n_before:
        logger.info("normalise_input_df: kept %d of %d rows.", out.height, n_before)

    if out.is_empty():
        raise ValueError(
            "No usable rows after cleaning: every record had a missing/invalid "
            "timestamp or coordinate."
        )
    return out


def normalise_missing(
    df: pl.DataFrame,
    cols: tuple = ("var_interp",),
    sentinels: Optional[Sequence[float]] = None,
) -> pl.DataFrame:
    """Replace common missing-value sentinels with null.

    Handles text columns (``"nan"``, ``"n/a"``, ``"-9999"`` …) as well as
    numeric sentinels and non-finite values, so a station column stored as
    strings cannot leak ``-9999`` into the analysis as a real temperature.

    Parameters
    ----------
    df : pl.DataFrame
    cols : tuple of str
        Columns to clean (missing columns are skipped).
    sentinels : sequence of float, optional
        Extra numeric sentinels to treat as null (added to the defaults).
    """
    extra = tuple(float(v) for v in (sentinels or ()))
    numeric = _NUMERIC_SENTINELS + extra
    exprs = []
    for c in cols:
        if c not in df.columns:
            continue
        series = df[c]
        if series.dtype in (pl.Utf8, pl.Categorical, pl.Enum):
            expr = (
                pl.when(
                    pl.col(c).cast(pl.String).str.strip_chars().str.to_lowercase()
                    .is_in(list(_STRING_SENTINELS))
                )
                .then(None)
                .otherwise(pl.col(c))
                .cast(pl.Float64, strict=False)
            )
        else:
            expr = pl.col(c).cast(pl.Float64, strict=False)
        expr = (
            pl.when(expr.is_in(list(numeric)) | ~expr.is_finite())
            .then(None)
            .otherwise(expr)
            .alias(c)
        )
        exprs.append(expr)
    return df.with_columns(exprs) if exprs else df


# ── Time averaging ────────────────────────────────────────────────────────────

_TIME_FREQ: dict[str, str] = {
    "hour": "1h", "day": "1d", "week": "1w", "month": "1mo", "year": "1y",
}


def time_average(
    df: pl.DataFrame,
    avg_time: str = "hour",
    type_cols: Sequence[str] = ("station",),
    value_col: str = "var_interp",
) -> pl.DataFrame:
    """Polars group_by_dynamic time averaging — faster than pandas resample."""
    if avg_time not in _TIME_FREQ:
        raise ValueError(
            f"avg_time must be one of {tuple(_TIME_FREQ)}, got {avg_time!r}"
        )
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col!r} not found; available: {df.columns}")
    present = [c for c in type_cols if c in df.columns]
    df = _coerce_naive_utc(df)
    return (
        df.sort("date")
          .group_by_dynamic("date", every=_TIME_FREQ[avg_time], closed="left",
                            group_by=present or None)
          .agg(pl.col(value_col).mean())
    )


def _to_datetime_literal(value: Any) -> Any:
    """Convert a user-facing date bound into a naive ``datetime``."""
    import datetime as _dt

    if isinstance(value, _dt.datetime):
        if value.tzinfo is not None:
            return value.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day)
    if isinstance(value, np.datetime64):
        as_dt = value.astype("datetime64[s]").astype(_dt.datetime)
        return as_dt
    text = str(value)
    try:
        parsed = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed_series = _coerce_datetime_series(pl.Series("_d", [text]))
        if parsed_series.null_count():
            raise ValueError(
                f"Cannot parse date bound {value!r}; use ISO-8601 "
                "(e.g. '2024-07-01' or '2024-07-01 13:00')."
            ) from None
        return parsed_series[0].to_pydatetime()
    if parsed.tzinfo is not None:
        return parsed.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return parsed


def _as_list(value: Any) -> list:
    if isinstance(value, (str, bytes)) or np.isscalar(value):
        return [value]
    if isinstance(value, pl.Series):
        return value.to_list()
    return list(value)


def select_by_date(
    df: pl.DataFrame,
    year=None, month=None, day=None, hour=None,
    start=None, end=None,
) -> pl.DataFrame:
    """Filter a Polars DataFrame by date/time components or a date range.

    Raises
    ------
    ValueError
        If a component is outside its valid range, or the start bound is
        after the end bound.
    """
    if "date" not in df.columns:
        raise ValueError(f"Column 'date' not found; available: {df.columns}")
    df = _coerce_naive_utc(df)
    d = pl.col("date")
    filters: list[pl.Expr] = []

    if year is not None:
        years = [int(y) for y in _as_list(year)]
        filters.append(d.dt.year().is_in(years))

    if month is not None:
        month_alias = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        months = []
        for m in _as_list(month):
            if isinstance(m, str) and m.isalpha():
                key = m.lower()[:3]
                if key not in month_alias:
                    raise ValueError(
                        f"Unknown month name {m!r}; use 1-12 or "
                        "'jan'…'dec'."
                    )
                months.append(month_alias[key])
            else:
                months.append(int(m))
        bad = [m for m in months if not 1 <= m <= 12]
        if bad:
            raise ValueError(f"month must be in 1-12, got {bad}")
        filters.append(d.dt.month().is_in(months))

    if day is not None:
        days = [int(v) for v in _as_list(day)]
        bad = [v for v in days if not 1 <= v <= 31]
        if bad:
            raise ValueError(f"day must be in 1-31, got {bad}")
        filters.append(d.dt.day().is_in(days))

    if hour is not None:
        hours = [int(h) for h in _as_list(hour)]
        bad = [h for h in hours if not 0 <= h <= 23]
        if bad:
            raise ValueError(f"hour must be in 0-23, got {bad}")
        filters.append(d.dt.hour().is_in(hours))

    start_dt = _to_datetime_literal(start) if start is not None else None
    end_dt = _to_datetime_literal(end) if end is not None else None
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        raise ValueError(f"start ({start_dt}) is after end ({end_dt})")
    if start_dt is not None:
        filters.append(d >= pl.lit(start_dt, dtype=pl.Datetime("ns")))
    if end_dt is not None:
        filters.append(d <= pl.lit(end_dt, dtype=pl.Datetime("ns")))

    if not filters:
        return df
    out = df.filter(pl.all_horizontal(filters))
    if out.is_empty():
        logger.warning("select_by_date: the date filters matched no rows.")
    return out


# ── Homogeneity kernel (Numba if available, else vectorised NumPy) ────────────

def _calc_homogeneity_numpy(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    """Fraction of in-window pixels sharing the centre class, vectorised.

    Matches :func:`_calc_homogeneity_numba` numerically (in-bounds
    neighbourhood only, ``nan`` where the centre is nodata) but runs at NumPy
    speed, replacing the previous per-window Python callback.
    """
    shape = arr.shape
    pad = int(kernel_size) // 2
    if pad < 1:
        raise ValueError(f"kernel_size must be >= 3, got {kernel_size}")
    a = np.asarray(arr, dtype=np.float64)
    h, w = shape
    padded = np.pad(a, pad, mode="constant", constant_values=np.nan)
    centre = a
    matches = np.zeros(shape, dtype=np.float32)
    for di in range(kernel_size):
        for dj in range(kernel_size):
            window = padded[di:di + h, dj:dj + w]
            # NaN never equals NaN, so padding never counts as a match.
            with np.errstate(invalid="ignore"):
                matches += (window == centre)

    rows = np.arange(h)
    cols = np.arange(w)
    row_count = np.minimum(rows + pad, h - 1) - np.maximum(rows - pad, 0) + 1
    col_count = np.minimum(cols + pad, w - 1) - np.maximum(cols - pad, 0) + 1
    total = np.outer(row_count, col_count).astype(np.float32)

    out = np.full(shape, np.nan, dtype=np.float32)
    valid = np.isfinite(centre) & (centre > 0)
    out[valid] = matches[valid] / total[valid]
    return out


if HAS_NUMBA:  # pragma: no cover - exercised only with numba installed
    @njit(fastmath=True, parallel=True)
    def _calc_homogeneity_numba(arr: np.ndarray, kernel_size: int) -> np.ndarray:
        h, w = arr.shape
        pad = kernel_size // 2
        out = np.empty((h, w), dtype=np.float32)
        for i in prange(h):
            for j in range(w):
                center = arr[i, j]
                if not np.isfinite(center) or center <= 0:
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
        return _calc_homogeneity_numpy(arr, kernel_size)


# ── Point extraction ──────────────────────────────────────────────────────────

def _points_in_raster_crs(
    ds: rasterio.io.DatasetReader,
    lons: np.ndarray,
    lats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat (EPSG:4326) onto the raster's CRS when it differs."""
    crs = ds.crs
    if crs is None:
        return lons, lats
    try:
        is_wgs84 = crs.to_epsg() == 4326
    except Exception:  # noqa: BLE001 - exotic CRS objects
        is_wgs84 = False
    if is_wgs84:
        return lons, lats

    try:
        from pyproj import Transformer
    except ImportError:  # pragma: no cover - pyproj ships with rasterio
        logger.warning(
            "pyproj unavailable: station lon/lat are used directly against a "
            "raster in %s — LCZ lookups may be wrong.",
            crs,
        )
        return lons, lats

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


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
        Opened LCZ raster. Points are given in EPSG:4326; they are reprojected
        internally when the raster uses another CRS.
    lons, lats : np.ndarray
        Point longitudes and latitudes (degrees).
    method : {"simple", "bilinear", "two.step"}
        ``"simple"`` — nearest-neighbour (default);
        ``"bilinear"`` — bilinear interpolation;
        ``"two.step"`` — nearest-neighbour with homogeneity filter.
    kernel_size : int
        Odd kernel size for the two-step homogeneity check.
    threshold : float
        Minimum homogeneity fraction to keep a point (two-step only).

    Returns
    -------
    lcz : np.ndarray[int32]
    mask : np.ndarray[bool]
        True where the point falls inside the raster and hit a valid LCZ class
        (1-17). Points outside the extent or on nodata are False.

    Raises
    ------
    ValueError
        If the input coordinate arrays are not 1-D, differ in length, contain
        non-finite values, or ``method``/``kernel_size`` are invalid.
    """
    if method not in ("simple", "bilinear", "two.step", "two_step"):
        raise ValueError(
            "method must be 'simple', 'bilinear' or 'two.step', "
            f"got {method!r}"
        )

    lons = np.asarray(lons, dtype=np.float64).ravel()
    lats = np.asarray(lats, dtype=np.float64).ravel()
    if lons.size != lats.size:
        raise ValueError(
            f"lons and lats must have the same length ({lons.size} != {lats.size})"
        )
    if not (np.isfinite(lons).all() and np.isfinite(lats).all()):
        raise ValueError("Station coordinates contain NaN/inf; clean them first.")

    arr = ds.read(1)
    transform = ds.transform
    nodata = ds.nodata
    h, w = arr.shape
    integral = np.issubdtype(arr.dtype, np.integer)

    if lons.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=bool)

    xs, ys = _points_in_raster_crs(ds, lons, lats)
    cols_f, rows_f = ~transform * (xs, ys)

    finite = np.isfinite(cols_f) & np.isfinite(rows_f)
    inside = (
        finite
        & (cols_f >= -0.5) & (cols_f <= w - 0.5)
        & (rows_f >= -0.5) & (rows_f <= h - 0.5)
    )
    cols = np.clip(np.rint(np.where(finite, cols_f, 0)).astype(np.int32), 0, w - 1)
    rows = np.clip(np.rint(np.where(finite, rows_f, 0)).astype(np.int32), 0, h - 1)

    if method == "bilinear":
        c0 = np.clip(np.floor(np.where(finite, cols_f, 0)).astype(np.int32), 0, max(w - 2, 0))
        r0 = np.clip(np.floor(np.where(finite, rows_f, 0)).astype(np.int32), 0, max(h - 2, 0))
        c1 = np.minimum(c0 + 1, w - 1)
        r1 = np.minimum(r0 + 1, h - 1)
        dx = (cols_f - c0).astype(np.float32)
        dy = (rows_f - r0).astype(np.float32)
        v00 = arr[r0, c0].astype(np.float32)
        v01 = arr[r0, c1].astype(np.float32)
        v10 = arr[r1, c0].astype(np.float32)
        v11 = arr[r1, c1].astype(np.float32)
        with np.errstate(invalid="ignore"):
            blended = (
                v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy)
                + v10 * (1 - dx) * dy + v11 * dx * dy
            )
        lcz = np.rint(np.nan_to_num(blended, nan=0.0)).astype(np.int32)
        valid = inside & (lcz >= 1) & (lcz <= 17)
        return np.where(valid, lcz, 0).astype(np.int32), valid

    raw = arr[rows, cols]
    if integral:
        invalid = ~np.isfinite(raw.astype(np.float64))
        if nodata is not None and np.isfinite(float(nodata)):
            invalid |= raw == raw.dtype.type(nodata)
    else:
        invalid = ~np.isfinite(raw)
        if nodata is not None and np.isfinite(float(nodata)):
            invalid |= np.isclose(raw, float(nodata), equal_nan=False)

    lcz = np.where(invalid, 0, raw).astype(np.int32)
    valid = inside & ~invalid & (lcz >= 1) & (lcz <= 17)

    if method in ("two.step", "two_step"):
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be an odd integer >= 3, got {kernel_size}"
            )
        hom = _calc_homogeneity_numba(
            arr.astype(np.float32) if not integral else arr.astype(np.int32),
            kernel_size,
        )
        station_hom = hom[rows, cols]
        keep = valid & np.isfinite(station_hom) & (station_hom >= threshold)
        return lcz, keep

    return lcz, valid


def assign_station_lcz(
    df: pl.DataFrame,
    ds: rasterio.io.DatasetReader,
    *,
    require: bool = True,
    what: str = "stations",
) -> pl.DataFrame:
    """Attach the raster's LCZ class to every row (via its station location).

    One raster lookup per unique (lat, lon) pair, then a single vectorised
    join — no per-row Python loop.

    Returns
    -------
    pl.DataFrame
        Rows restricted to stations whose LCZ class is 1-17, with an ``lcz``
        ``Int32`` column added.

    Raises
    ------
    ValueError
        If no station maps to a valid LCZ class and ``require`` is True.
    """
    required = {"latitude", "longitude"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"df must contain {sorted(required)}; got {df.columns}"
        )
    if df.is_empty():
        raise ValueError("No rows to assign LCZ classes to.")

    if "lcz" in df.columns:
        logger.info("Replacing the existing 'lcz' column with raster classes.")
        df = df.drop("lcz")

    stations = df.unique(subset=["latitude", "longitude"], keep="first")
    lcz_vals, mask = extract_lcz_at_points(
        ds, stations["longitude"].to_numpy(), stations["latitude"].to_numpy()
    )
    stations = stations.with_columns(
        pl.Series(name="lcz", values=np.where(mask, lcz_vals, 0)).cast(pl.Int32)
    )

    out = df.join(
        stations.select(["latitude", "longitude", "lcz"]),
        on=["latitude", "longitude"],
        how="left",
    )
    out = out.filter(pl.col("lcz").is_between(1, 17))

    if out.is_empty() and require:
        found = sorted(set(stations["lcz"].to_list()))
        raise ValueError(
            f"No {what} could be matched to an LCZ class 1-17 in the raster. "
            f"Checked {stations.height} location(s); classes found: {found}. "
            "Check that the LCZ raster covers the station coordinates and uses "
            "classes 1-17 (0 = nodata)."
        )
    return out


# ── Spatial helpers ───────────────────────────────────────────────────────────

def utm_epsg_for(ds: rasterio.io.DatasetReader) -> str:
    """Return the UTM EPSG code covering the centre of a rasterio dataset.

    The zone is derived from the raster's WGS84 bounding box and clamped to
    the valid 1-60 range (a box touching 180° used to yield zone 61).
    """
    bounds = ds.bounds
    lon = (bounds.left + bounds.right) / 2
    lat = (bounds.bottom + bounds.top) / 2
    zone = int(np.floor((lon + 180) / 6) + 1)
    zone = min(max(zone, 1), 60)
    return f"EPSG:{326 if lat >= 0 else 327}{zone:02d}"


# ── Color palette ─────────────────────────────────────────────────────────────

def lcz_palette(inclusive: bool = False) -> dict[str, str]:
    """Return a dict mapping LCZ class strings ('1'-'17') to hex colors.

    Colors always come from :mod:`LCZ4py._internal.lcz_parameters_data`; when
    fewer than 17 colors are registered the palette is padded with neutral
    greys instead of raising.
    """
    colors = list(LCZ_COLORBLIND if inclusive else LCZ_COLORS)
    if len(colors) < 17:
        logger.warning(
            "LCZ palette has %d colors; padding to 17 with greys.", len(colors)
        )
        colors += [f"#{v:02x}{v:02x}{v:02x}" for v in range(120, 120 + (17 - len(colors)))]
    return {str(i + 1): c for i, c in enumerate(colors[:17])}


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
        Must contain a ``'date'`` column (``Datetime``; tz-aware input is
        converted to naive UTC).
    by : str
        Split key — one of ``"year"``, ``"season"``, ``"seasonyear"``,
        ``"month"``, ``"monthyear"``, ``"weekday"``, ``"weekend"``,
        ``"site"``, ``"daylight"``, ``"dst"``.
    lat, lon : float
        Mean station latitude/longitude (degrees).  Used only for
        ``"daylight"`` (solar elevation) and ``"dst"`` (hemisphere).

    Raises
    ------
    ValueError
        On an unknown ``by`` value, a missing ``date``/``station`` column, or
        non-finite coordinates.
    """
    if by not in _BY_VALUES:
        raise ValueError(f"by= must be one of {_BY_VALUES!r}, got {by!r}")
    if "date" not in df.columns:
        raise ValueError(f"Column 'date' not found; available: {df.columns}")
    if by == "site" and "station" not in df.columns:
        raise ValueError("by='site' requires a 'station' column.")

    df = _coerce_naive_utc(df)
    lat = float(lat) if lat is not None else 0.0
    lon = float(lon) if lon is not None else 0.0
    if not np.isfinite(lat) or not np.isfinite(lon):
        logger.warning("add_by_column: non-finite lat/lon; using (0, 0).")
        lat, lon = 0.0, 0.0
    lat = float(np.clip(lat, -90.0, 90.0))

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
    unique = {str(v) for v in series.drop_nulls().unique().to_list()}
    if by in _BY_ORDER:
        ordered = [g for g in _BY_ORDER[by] if g in unique]
        extras = sorted(unique - set(ordered))
        return ordered + extras
    if by == "year":
        return sorted(unique, key=_sort_key_int)
    if by == "seasonyear":
        def _syk(s: str):
            p = s.split()
            return (_sort_key_int(p[1]) if len(p) > 1 else 9999,
                    _SEASON_IDX.get(p[0], 99))
        return sorted(unique, key=_syk)
    if by == "monthyear":
        def _myk(s: str):
            p = s.split()
            return (_sort_key_int(p[1]) if len(p) > 1 else 9999,
                    _MONTH_IDX.get(p[0], 99))
        return sorted(unique, key=_myk)
    return sorted(unique)


def _sort_key_int(value: Any) -> int:
    """Numeric sort key that tolerates malformed labels instead of raising."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 9999


# ── Time-series line helpers ──────────────────────────────────────────────────

def gap_threshold_hours(dates_np: np.ndarray, floor_h: float = 1.5) -> float:
    """Line-break threshold: 2× the median sampling step, at least *floor_h*."""
    if len(dates_np) <= 2:
        return max(floor_h, 1.5)
    diffs_h = np.diff(np.asarray(dates_np).astype("datetime64[s]")).astype(np.float64) / 3600
    pos = diffs_h[diffs_h > 0]
    step = float(np.median(pos)) if len(pos) else 1.0
    return max(2.0 * step, floor_h)


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
        out_d.extend(dates_np[prev: idx + 1].tolist())
        out_v.extend(values_np[prev: idx + 1].tolist())
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
    "assign_station_lcz",
    "utm_epsg_for",
    "lcz_palette",
    "OUTPUT_DIR",
    "CANON_COLS",
    "require_nonempty",
    "resolve_workers",
    "gap_threshold_hours",
    # temporal grouping
    "_BY_VALUES",
    "add_by_column",
    "by_sorted_groups",
    # time-series line helpers
    "_break_gaps_numpy",
    "_daylight_intervals_batch",
]
