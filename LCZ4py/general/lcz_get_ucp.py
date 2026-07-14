"""
====================================================================
PARALLEL GEOSPATIAL VARIABLE PROCESSOR WITH CACHE
====================================================================

Downloads, processes, and caches a comprehensive suite of geospatial 
variables for urban characterization using parallel computing.
Integrates multiple global datasets to support Local Climate Zone (LCZ) 
classification and urban climate analysis.
====================================================================
"""

import os
import hashlib
import logging
import warnings
import tempfile
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial, wraps
from datetime import datetime
from contextlib import contextmanager
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import rasterio
from rasterio import warp, features
from rasterio.crs import CRS
from shapely.geometry import box
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from diskcache import FanoutCache

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# GDAL reads these from the process environment as a global fallback, so
# setting them once here covers every /vsicurl/ open (variables + GHSL tiles)
# without wrapping each rioxarray.open_rasterio call in a context manager.
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.ovr,.zip")
# Older GDAL builds send a bare "GDAL/x.y.z" User-Agent, which Zenodo's edge
# now blocks with a 403 that GDAL surfaces as an opaque open failure.
os.environ.setdefault("GDAL_HTTP_USERAGENT", "Mozilla/5.0 (compatible; lcz4r-python)")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "LCZ4r_output"


# =============================================================================
# DECORATORS AND UTILITIES
# =============================================================================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry a function on failure with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
            raise last_exception
        return wrapper
    return decorator


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {elapsed:.2f}s")
        return result
    return wrapper


def safe_operation(default_return=None):
    """Decorator to catch all exceptions and return default value."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator


@contextmanager
def timer_context(name: str):
    """Context manager for timing operations."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} completed in {elapsed:.2f}s")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VariableDefinition:
    """Definition for a geospatial variable to process."""
    url: str
    name: str
    description: str
    na_to_zero: bool = False
    extract_method: str = "bilinear"
    resample_method: str = "bilinear"
    category: str = "other"
    
    def process(self, raster: xr.DataArray, study_area: gpd.GeoDataFrame) -> xr.DataArray:
        """Default processing: crop and mask to study area."""
        # Ensure study area is in same CRS as raster
        study_area_crs = study_area.to_crs(raster.rio.crs)
        
        # Clip to study area
        clipped = raster.rio.clip(
            study_area_crs.geometry.values,
            study_area_crs.crs,
            drop=True,
            all_touched=True
        )
        
        # Handle NA values
        if self.na_to_zero:
            clipped = clipped.fillna(0)
        
        return clipped.rename(self.name)


@dataclass
class GHSLMapping:
    """Mapping configuration for GHSL data types."""
    base_url: str
    filename_pattern: str
    var_name: str
    description: str
    units: str


@dataclass
class ProcessingResult:
    """Result of processing a single variable."""
    df: Optional[pd.DataFrame] = None
    raster: Optional[xr.DataArray] = None
    name: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass 
class ProcessingSummary:
    """Summary of processing results."""
    total_variables: int = 0
    successful: int = 0
    failed: int = 0
    total_time: float = 0.0
    failed_variables: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful / self.total_variables * 100 if self.total_variables > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_variables': self.total_variables,
            'successful': self.successful,
            'failed': self.failed,
            'total_time': self.total_time,
            'success_rate': self.success_rate,
            'failed_variables': self.failed_variables
        }


# =============================================================================
# ROBUST HTTP SESSION
# =============================================================================

class RobustSession:
    """
    HTTP session with comprehensive retry logic, timeout handling,
    and connection pooling for reliable downloads.
    """
    
    DEFAULT_TIMEOUT = 120  # seconds
    MAX_RETRIES = 5
    BACKOFF_FACTOR = 0.5
    POOL_CONNECTIONS = 20
    POOL_MAXSIZE = 100
    
    def __init__(
        self,
        max_retries: int = None,
        backoff_factor: float = None,
        timeout: int = None,
        pool_connections: int = None,
        pool_maxsize: int = None
    ):
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries or self.MAX_RETRIES,
            backoff_factor=backoff_factor or self.BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504, 408],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_connections or self.POOL_CONNECTIONS,
            pool_maxsize=pool_maxsize or self.POOL_MAXSIZE
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set user agent
        self.session.headers.update({
            "User-Agent": "UrbanParameterProcessor/2.0 (Python; Geospatial Research)",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate"
        })
    
    def head(self, url: str, **kwargs) -> requests.Response:
        """Make a HEAD request with timeout."""
        kwargs.setdefault('timeout', self.timeout)
        return self.session.head(url, **kwargs)
    
    def get(self, url: str, stream: bool = False, **kwargs) -> requests.Response:
        """Make a GET request with timeout."""
        kwargs.setdefault('timeout', self.timeout)
        response = self.session.get(url, stream=stream, **kwargs)
        response.raise_for_status()
        return response
    
    def download_file(
        self,
        url: str,
        local_path: Union[str, Path],
        chunk_size: int = 8192,
        progress_callback: Optional[Callable] = None
    ) -> Path:
        """
        Download a file to local path with progress tracking.
        
        Args:
            url: URL to download from
            local_path: Path to save file
            chunk_size: Download chunk size in bytes
            progress_callback: Optional callback(bytes_downloaded, total_size)
            
        Returns:
            Path to downloaded file
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        response = self.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        bytes_downloaded = 0
        
        # Use temporary file for atomic write
        temp_path = local_path.with_suffix(local_path.suffix + '.tmp')
        
        try:
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(bytes_downloaded, total_size)
            
            # Atomic rename
            temp_path.rename(local_path)
            return local_path
            
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    
    def check_url_available(self, url: str) -> bool:
        """Check if URL is accessible without downloading."""
        try:
            response = self.head(url, allow_redirects=True)
            return response.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# GEOGRAPHIC CACHE
# =============================================================================

class GeoCache:
    """
    Disk-based cache for geospatial data with integrity checking,
    automatic cleanup, and size management.
    """
    
    def __init__(
        self,
        cache_dir: str = "lcz4r_cache",
        max_size_mb: int = 10240,  # 10 GB default
        shard_count: int = 8
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Use FanoutCache for better performance with many files
        self.index = FanoutCache(
            str(self.cache_dir / ".cache_index"),
            size_limit=max_size_mb * 1024 * 1024,
            shards=shard_count
        )
        
        # Create subdirectories
        (self.cache_dir / "rasters").mkdir(exist_ok=True)
        (self.cache_dir / "vectors").mkdir(exist_ok=True)
        (self.cache_dir / "ghsl").mkdir(exist_ok=True)
        (self.cache_dir / "temp").mkdir(exist_ok=True)
    
    def _generate_key(self, *parts: str) -> str:
        """Generate a cache key from parts."""
        content = "_".join(str(p) for p in parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _verify_raster(self, path: Path) -> bool:
        """Verify a raster file is not corrupt."""
        try:
            with rioxarray.open_rasterio(path, chunks=None) as src:
                return src.size > 0 and src.rio.crs is not None
        except Exception:
            return False
    
    def _verify_vector(self, path: Path) -> bool:
        """Verify a vector file is not corrupt."""
        try:
            gpd.read_file(path)
            return True
        except Exception:
            return False
    
    def get_raster_path(self, var_name: str, extent_id: str) -> Optional[Path]:
        """Get cached raster path if exists and valid."""
        cache_file = self.cache_dir / "rasters" / f"{var_name}_{extent_id}.tif"
        
        if cache_file.exists() and self._verify_raster(cache_file):
            return cache_file
        
        # Clean up invalid cache
        cache_file.unlink(missing_ok=True)
        return None
    
    def save_raster(
        self,
        raster: xr.DataArray,
        var_name: str,
        extent_id: str,
        compress: str = "LZW",
        predictor: int = 2
    ) -> Path:
        """Save raster to cache with compression."""
        cache_file = self.cache_dir / "rasters" / f"{var_name}_{extent_id}.tif"
        
        # Remove band dimension if single
        if 'band' in raster.dims and raster.band.size == 1:
            raster = raster.squeeze('band', drop=True)
        
        raster.rio.to_raster(
            cache_file,
            driver="GTiff",
            compress=compress,
            predictor=predictor if compress == "LZW" else 1,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            windowed=True
        )
        
        # Update index
        key = self._generate_key(var_name, extent_id)
        self.index[key] = {
            'path': str(cache_file),
            'created': datetime.now().isoformat(),
            'size': cache_file.stat().st_size
        }
        
        return cache_file
    
    def get_vector_path(self, name: str) -> Optional[Path]:
        """Get cached vector path if exists and valid."""
        cache_file = self.cache_dir / "vectors" / f"{name}.gpkg"
        
        if cache_file.exists() and self._verify_vector(cache_file):
            return cache_file
        
        cache_file.unlink(missing_ok=True)
        return None
    
    def save_vector(self, gdf: gpd.GeoDataFrame, name: str) -> Path:
        """Save vector to cache."""
        cache_file = self.cache_dir / "vectors" / f"{name}.gpkg"
        gdf.to_file(cache_file, driver="GPKG")
        return cache_file
    
    def get_ghsl_tile_path(self, ghsl_type: str, tile_id: str) -> Optional[Path]:
        """Get cached GHSL tile path if exists and valid."""
        cache_file = self.cache_dir / "ghsl" / f"{ghsl_type}_{tile_id}.tif"
        
        if cache_file.exists() and self._verify_raster(cache_file):
            return cache_file
        
        cache_file.unlink(missing_ok=True)
        return None
    
    def save_ghsl_tile(
        self,
        raster: xr.DataArray,
        ghsl_type: str,
        tile_id: str
    ) -> Path:
        """Save GHSL tile to cache."""
        cache_file = self.cache_dir / "ghsl" / f"{ghsl_type}_{tile_id}.tif"
        
        if 'band' in raster.dims and raster.band.size == 1:
            raster = raster.squeeze('band', drop=True)
        
        raster.rio.to_raster(
            cache_file,
            driver="GTiff",
            compress="LZW",
            tiled=True,
            BIGTIFF="IF_SAFER",
            blockxsize=256,
            blockysize=256
        )
        
        return cache_file
    
    def get_temp_path(self, prefix: str, suffix: str = ".tmp") -> Path:
        """Get a temporary path in cache directory."""
        return self.cache_dir / "temp" / f"{prefix}_{os.urandom(4).hex()}{suffix}"
    
    def cleanup_temp(self):
        """Clean up temporary files."""
        temp_dir = self.cache_dir / "temp"
        for f in temp_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'raster_count': len(list((self.cache_dir / "rasters").glob("*.tif"))),
            'vector_count': len(list((self.cache_dir / "vectors").glob("*.gpkg"))),
            'ghsl_count': len(list((self.cache_dir / "ghsl").glob("*.tif"))),
            'total_size_mb': sum(
                f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file()
            ) / (1024 * 1024),
            'index_size': len(self.index)
        }
        return stats
    
    def clear(self, keep_index: bool = False):
        """Clear all cached files."""
        for subdir in ["rasters", "vectors", "ghsl", "temp"]:
            for f in (self.cache_dir / subdir).glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
        
        if not keep_index:
            self.index.clear()
    
    def close(self):
        """Close the cache."""
        self.index.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_temp()
        self.close()


# =============================================================================
# MAIN PROCESSOR CLASS
# =============================================================================

class UrbanParameterProcessor:
    """
    Parallel Geospatial Variable Processor with Cache.
    
    Downloads, processes, and caches a comprehensive suite of geospatial 
    variables for urban characterization using parallel computing.
    
    Features:
    - Fault-tolerant: continues processing if individual variables fail
    - Cached: avoids re-downloading/processing existing data
    - Parallel: uses thread/process pools for concurrent processing
    - Robust: comprehensive error handling and validation
    - Cross-platform: works on Windows, Linux, and macOS
    
    Attributes
    ----------
    lcz_map : xr.DataArray
        Loaded LCZ classification raster
    stations : gpd.GeoDataFrame
        Station locations
    cache : GeoCache
        Cache manager
    session : RobustSession
        HTTP session for downloads
    n_workers : int
        Number of parallel workers
    """
    
    # GHSL type mappings
    GHSL_MAPPINGS: Dict[str, GHSLMapping] = {
        "built_surface": GHSLMapping(
            base_url="https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2030_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_S_E2030_GLOBE_R2023A_4326_3ss_V1_0_",
            filename_pattern="GHS_BUILT_S_E2030_GLOBE_R2023A_4326_3ss_V1_0_",
            var_name="built_sur",
            description="Built-up surface fraction",
            units="m²/m²"
        ),
        "built_height": GHSLMapping(
            base_url="https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_H_GLOBE_R2023A/GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_4326_3ss_V1_0_",
            filename_pattern="GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_4326_3ss_V1_0_",
            var_name="built_hei",
            description="Building height (above ground)",
            units="m"
        ),
        "built_volume": GHSLMapping(
            base_url="https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_V_GLOBE_R2023A/GHS_BUILT_V_E2030_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_V_E2030_GLOBE_R2023A_4326_3ss_V1_0_",
            filename_pattern="GHS_BUILT_V_E2030_GLOBE_R2023A_4326_3ss_V1_0_",
            var_name="built_vol",
            description="Built-up volume",
            units="m³"
        ),
        "pop": GHSLMapping(
            base_url="https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2030_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E2030_GLOBE_R2023A_4326_3ss_V1_0_",
            filename_pattern="GHS_POP_E2030_GLOBE_R2023A_4326_3ss_V1_0_",
            var_name="pop",
            description="Population density",
            units="persons/km²"
        )
    }
    
    def __init__(
        self,
        lcz_map: Any,
        stations: Optional[Any] = None,
        cache_dir: str = "lcz4r_cache",
        n_workers: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize the processor.

        Parameters
        ----------
        lcz_map : str, Path, or xarray.DataArray
            LCZ classification raster
        stations : GeoDataFrame, str, Path, or DataFrame, optional
            Station locations. If None, station-value extraction is skipped
            and the processed parameters are returned as a raster stack
            cropped to the ``lcz_map`` extent instead.
        cache_dir : str
            Directory for caching
        n_workers : int, optional
            Number of parallel workers
        verbose : bool
            Print progress messages
        """
        # Setup workers
        cpu_count = os.cpu_count() or 4
        if n_workers is None:
            # Conservative default: remote servers/network throttle beyond
            # a handful of concurrent downloads regardless of local CPU count.
            n_workers = min(max(1, cpu_count - 1), 4)
        self.n_workers = min(n_workers, cpu_count)
        self.verbose = verbose

        # Initialize cache and session
        self.cache = GeoCache(cache_dir)
        self.session = RobustSession()
        # Single pool reused by both variable and GHSL-tile processing so
        # concurrency never exceeds n_workers across the whole run.
        self.executor = ThreadPoolExecutor(max_workers=self.n_workers)
        
        # Load and validate LCZ map
        self.lcz_map = self._load_lcz_map(lcz_map)
        self.target_crs = self.lcz_map.rio.crs
        self.target_transform = self.lcz_map.rio.transform()
        self.target_shape = self.lcz_map.shape
        # lcz_shp_wgs84 is only the rectangular bbox of lcz_map (cheap vsicurl
        # crop window) — it does not carry lcz_map's actual (often irregular)
        # valid-pixel footprint. Every resampled variable is re-masked against
        # this in _resample_to_target so outputs match lcz_map exactly, not
        # just its bounding box.
        self._lcz_valid_mask = self.lcz_map.notnull()
        
        # Setup study area
        self.lcz_shp_wgs84 = self._extract_study_area_wgs84()
        self.extent_id = self._generate_extent_id()
        
        # Validate and setup stations
        self.stations = self._validate_stations(stations)
        
        if self.verbose:
            logger.info("=" * 60)
            logger.info("Urban Parameter Processor Initialized")
            logger.info("=" * 60)
            logger.info(f"Study Area ID: {self.extent_id}")
            logger.info(f"Target CRS: {self.target_crs}")
            logger.info(f"Target Shape: {self.target_shape}")
            logger.info(f"Workers: {self.n_workers}")
            logger.info(f"Cache: {self.cache.cache_dir}")
            logger.info("=" * 60)
    
    def _load_lcz_map(self, lcz_map: Any) -> xr.DataArray:
        """Load and validate LCZ map."""
        if isinstance(lcz_map, (str, Path)):
            if not Path(lcz_map).exists():
                raise FileNotFoundError(f"LCZ map file not found: {lcz_map}")
            # Load into memory and close the file handle immediately — otherwise
            # the underlying GDAL dataset stays open for the processor's lifetime
            # and can raise unraisable errors during interpreter shutdown.
            with rioxarray.open_rasterio(lcz_map, masked=True) as src:
                da = src.load()
        elif isinstance(lcz_map, xr.DataArray):
            da = lcz_map.copy()
        elif isinstance(lcz_map, xr.Dataset):
            da = lcz_map[list(lcz_map.data_vars)[0]].copy()
        else:
            raise ValueError(
                "lcz_map must be a file path, xarray DataArray, or Dataset"
            )
        
        # Handle multi-band rasters
        if 'band' in da.dims:
            if da.band.size > 1:
                warnings.warn("Multiple bands detected. Using first band.")
            da = da.isel(band=0, drop=True)
        
        # Validate
        if da.rio.crs is None:
            raise ValueError("LCZ map must have a valid CRS defined")
        
        return da
    
    def _extract_study_area_wgs84(self) -> gpd.GeoDataFrame:
        """Extract study area boundary in WGS84."""
        bounds = self.lcz_map.rio.bounds()
        bbox = box(*bounds)
        
        gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[bbox],
            crs=self.target_crs
        )
        
        return gdf.to_crs(epsg=4326)
    
    def _generate_extent_id(self) -> str:
        """Generate unique ID for the study area extent."""
        bounds = self.lcz_map.rio.bounds()
        rounded = [round(b, 3) for b in bounds]
        return "_".join(map(str, rounded))
    
    def _validate_stations(self, stations: Any) -> Optional[gpd.GeoDataFrame]:
        """Validate and standardize station data."""
        if stations is None:
            return None
        if isinstance(stations, gpd.GeoDataFrame):
            gdf = stations.copy()
        elif isinstance(stations, (str, Path)):
            gdf = gpd.read_file(stations)
        elif isinstance(stations, pd.DataFrame):
            # Try to detect coordinate columns
            lat_col = None
            lon_col = None
            
            for col in stations.columns:
                col_lower = col.lower().strip()
                if col_lower in ['lat', 'latitude', 'y', 'northing']:
                    lat_col = col
                elif col_lower in ['lon', 'long', 'longitude', 'x', 'easting']:
                    lon_col = col
            
            if lat_col and lon_col:
                gdf = gpd.GeoDataFrame(
                    stations,
                    geometry=gpd.points_from_xy(
                        stations[lon_col], 
                        stations[lat_col]
                    ),
                    crs="EPSG:4326"
                )
            else:
                raise ValueError(
                    "Could not detect latitude/longitude columns. "
                    "Expected columns like: lat/latitude/y and lon/longitude/x"
                )
        else:
            raise ValueError(
                "stations must be a GeoDataFrame, file path, or DataFrame with lat/lon"
            )
        
        # Ensure CRS
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
            logger.warning("Stations CRS not set, assuming EPSG:4326")
        
        # Validate geometry
        if not all(gdf.geometry.is_valid):
            logger.warning("Some station geometries are invalid, attempting to fix")
            gdf.geometry = gdf.geometry.buffer(0)
        
        return gdf
    
    def _download_raster_vsicurl(
        self,
        url: str,
        study_area: gpd.GeoDataFrame
    ) -> Optional[xr.DataArray]:
        """Download raster using GDAL's vsicurl (fastest method)."""
        try:
            vsi_url = f"/vsicurl/{url}"

            with rioxarray.open_rasterio(
                vsi_url,
                masked=True,
                chunks='auto'
            ) as src:
                # Clip to study area. from_disk=True is required for global
                # rasters (e.g. GLC_FCS30D at 30m is ~527k x 1.3M px): without
                # it, dask's clip() materializes far more than the requested
                # window and OOM-kills the process; from_disk routes through
                # rasterio.mask.mask for a true windowed read (verified: same
                # global raster goes from OOM to ~50MB peak with this on).
                study_area_crs = study_area.to_crs(src.rio.crs)
                clipped = src.rio.clip(
                    study_area_crs.geometry.values,
                    study_area_crs.crs,
                    drop=True,
                    all_touched=True,
                    from_disk=True
                )
                return clipped
                
        except Exception as e:
            logger.debug(f"vsicurl failed: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def _download_and_process_raster(
        self,
        var_def: VariableDefinition,
        study_area: gpd.GeoDataFrame
    ) -> Optional[xr.DataArray]:
        """
        Download and process a raster variable via GDAL's /vsicurl/ streaming.

        No full-file fallback: /vsicurl/ only fetches the bytes it needs
        (metadata + the clipped window), so downloading the whole remote
        file on failure would be strictly heavier for no benefit. A failure
        here is retried by @retry_on_failure and otherwise surfaces as a
        normal per-variable failure.
        """
        # Check cache first
        cached_path = self.cache.get_raster_path(var_def.name, self.extent_id)
        if cached_path:
            if self.verbose:
                logger.info(f"  ✓ Using cached: {cached_path.name}")
            with rioxarray.open_rasterio(cached_path, masked=True) as src:
                return src.load()

        if self.verbose:
            logger.info(f"  ↓ Downloading: {var_def.name}")

        result = self._download_raster_vsicurl(var_def.url, study_area)

        if result is None:
            raise RuntimeError(f"vsicurl failed for {var_def.name}")

        # Apply variable-specific processing
        result = var_def.process(result, study_area)
        
        # Save to cache
        try:
            self.cache.save_raster(result, var_def.name, self.extent_id)
            if self.verbose:
                logger.info(f"  ✓ Cached: {var_def.name}")
        except Exception as e:
            logger.warning(f"  Failed to cache {var_def.name}: {e}")
        
        return result
    
    def _extract_values_at_stations(
        self,
        raster: xr.DataArray,
        method: str = "bilinear"
    ) -> np.ndarray:
        """Extract raster values at station locations via xarray's vectorised
        pointwise indexing (correct, real API — no manual affine math needed).
        """
        # Reproject stations to raster CRS
        stations_crs = self.stations.to_crs(raster.rio.crs)
        x_coords = xr.DataArray([p.x for p in stations_crs.geometry], dims="points")
        y_coords = xr.DataArray([p.y for p in stations_crs.geometry], dims="points")

        # Remove band dimension if present
        if 'band' in raster.dims and raster.band.size == 1:
            raster = raster.squeeze('band', drop=True)

        if method == "nearest":
            sampled = raster.sel(x=x_coords, y=y_coords, method="nearest")
        else:  # bilinear
            sampled = raster.interp(x=x_coords, y=y_coords, method="linear")

        return sampled.values.astype(float)
    
    def _resample_to_target(
        self,
        raster: xr.DataArray,
        method: str = "bilinear"
    ) -> xr.DataArray:
        """Resample raster to target grid."""
        try:
            # Remove band dimension if present
            if 'band' in raster.dims and raster.band.size == 1:
                raster = raster.squeeze('band', drop=True)
            
            # Reproject to match target
            resampled = raster.rio.reproject_match(
                self.lcz_map,
                resampling=getattr(warp.Resampling, method, warp.Resampling.bilinear)
            )

            # reproject_match only aligns the grid to lcz_map's bbox — it does
            # not apply lcz_map's own nodata mask, so re-mask here.
            resampled = resampled.where(self._lcz_valid_mask)

            return resampled
            
        except Exception as e:
            logger.warning(f"Resampling failed: {e}")
            return raster
    
    def _process_single_variable(
        self,
        var_name: str,
        var_def: VariableDefinition
    ) -> ProcessingResult:
        """Process a single variable with full error handling."""
        start_time = time.time()
        
        try:
            # Download and process
            raster = self._download_and_process_raster(var_def, self.lcz_shp_wgs84)
            
            if raster is None:
                return ProcessingResult(
                    name=var_name,
                    success=False,
                    error="Download/processing failed",
                    processing_time=time.time() - start_time
                )
            
            # Extract values at stations (skipped when no stations given)
            if self.stations is not None:
                extracted = self._extract_values_at_stations(
                    raster,
                    method=var_def.extract_method
                )
                id_col = self.stations.columns[0]
                df = pd.DataFrame({
                    id_col: self.stations[id_col].values,
                    var_name: extracted
                })
            else:
                df = None

            # Resample to target grid
            resampled = self._resample_to_target(raster, var_def.resample_method)
            
            return ProcessingResult(
                df=df,
                raster=resampled,
                name=var_name,
                success=True,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ProcessingResult(
                name=var_name,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _get_ghsl_tiles(self) -> List[str]:
        """Get GHSL tile IDs covering the study area."""
        # Check cache
        cached_path = self.cache.get_vector_path("ghs_titles")
        
        if cached_path:
            if self.verbose:
                logger.info("Using cached GHSL tile index")
            ghs_titles = gpd.read_file(cached_path)
        else:
            if self.verbose:
                logger.info("Downloading GHSL tile index...")
            
            url = (
                "https://github.com/ByMaxAnjos/MachineLearning_for_geospatial_analysis/"
                "raw/refs/heads/main/LCZ4r_db/GHSL_tile_id.gpkg"
            )
            
            temp_path = self.cache.get_temp_path("ghs_titles", ".gpkg")
            
            try:
                self.session.download_file(url, temp_path)
                ghs_titles = gpd.read_file(temp_path)
                self.cache.save_vector(ghs_titles, "ghs_titles")
            finally:
                temp_path.unlink(missing_ok=True)
        
        # Find intersecting tiles
        ghs_titles_wgs84 = ghs_titles.to_crs(epsg=4326)
        
        try:
            # Use spatial join for better performance
            joined = gpd.sjoin(
                ghs_titles_wgs84,
                self.lcz_shp_wgs84,
                how='inner',
                predicate='intersects'
            )
            if 'tile_id' in joined.columns:
                return joined['tile_id'].unique().tolist()
        except Exception:
            pass
        
        # Fallback: iterate and check
        tile_ids = []
        for idx, row in ghs_titles_wgs84.iterrows():
            if row.geometry.intersects(self.lcz_shp_wgs84.geometry.iloc[0]):
                if 'tile_id' in row.index:
                    tile_ids.append(row['tile_id'])
        
        return tile_ids
    
    def _process_ghsl_tile(
        self,
        tile_id: str,
        ghsl_type: str,
        mapping: GHSLMapping
    ) -> Optional[xr.DataArray]:
        """Process a single GHSL tile with fault tolerance."""
        # Check cache
        cached_path = self.cache.get_ghsl_tile_path(ghsl_type, tile_id)
        if cached_path:
            if self.verbose:
                logger.info(f"    ✓ Cached tile: {tile_id}")
            with rioxarray.open_rasterio(cached_path, masked=True) as src:
                return src.load()
        
        # Build URLs
        zip_url = f"{mapping.base_url}{tile_id}.zip"
        tif_filename = f"{mapping.filename_pattern}{tile_id}.tif"
        
        # Try vsicurl first
        try:
            vsi_path = f"/vsizip//vsicurl/{zip_url}/{tif_filename}"
            
            with rioxarray.open_rasterio(vsi_path, masked=True) as src:
                study_area_crs = self.lcz_shp_wgs84.to_crs(src.rio.crs)
                clipped = src.rio.clip(
                    study_area_crs.geometry.values,
                    study_area_crs.crs,
                    drop=True,
                    all_touched=True
                )
                
                self.cache.save_ghsl_tile(clipped, ghsl_type, tile_id)
                
                if self.verbose:
                    logger.info(f"    ✓ Processed tile: {tile_id}")
                
                return clipped
                
        except Exception as e:
            logger.debug(f"vsicurl failed for tile {tile_id}: {e}")
            return self._process_ghsl_tile_fallback(
                tile_id, ghsl_type, mapping, zip_url, tif_filename
            )
    
    def _process_ghsl_tile_fallback(
        self,
        tile_id: str,
        ghsl_type: str,
        mapping: GHSLMapping,
        zip_url: str,
        tif_filename: str
    ) -> Optional[xr.DataArray]:
        """Fallback method to download and process GHSL tile."""
        temp_zip = self.cache.get_temp_path(f"ghsl_{tile_id}", ".zip")
        
        try:
            # Download ZIP
            self.session.download_file(zip_url, temp_zip)
            
            # Extract and read
            with zipfile.ZipFile(temp_zip, 'r') as zf:
                # Find the TIF file (might have slightly different name)
                tif_names = [n for n in zf.namelist() if n.endswith('.tif')]
                
                if not tif_names:
                    raise ValueError("No TIF file found in ZIP")
                
                tif_to_use = tif_names[0]  # Use first TIF found
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    zf.extract(tif_to_use, temp_dir)
                    tif_path = Path(temp_dir) / tif_to_use
                    
                    with rioxarray.open_rasterio(tif_path, masked=True) as src:
                        study_area_crs = self.lcz_shp_wgs84.to_crs(src.rio.crs)
                        clipped = src.rio.clip(
                            study_area_crs.geometry.values,
                            study_area_crs.crs,
                            drop=True,
                            all_touched=True
                        )
                        
                        self.cache.save_ghsl_tile(clipped, ghsl_type, tile_id)
                        
                        if self.verbose:
                            logger.info(f"    ✓ Processed tile (fallback): {tile_id}")
                        
                        return clipped
                        
        except Exception as e:
            logger.warning(f"    ✗ Fallback failed for tile {tile_id}: {e}")
            return None
            
        finally:
            temp_zip.unlink(missing_ok=True)
    
    def _mosaic_rasters(
        self,
        rasters: List[xr.DataArray],
        var_name: str
    ) -> Optional[xr.DataArray]:
        """Mosaic multiple rasters into one."""
        if not rasters:
            return None
        
        if len(rasters) == 1:
            result = rasters[0]
            if 'band' in result.dims and result.band.size == 1:
                result = result.squeeze('band', drop=True)
            return result.rename(var_name)
        
        try:
            # Merge using xarray
            merged = xr.merge([r.rename(f"temp_{i}") for i, r in enumerate(rasters)])
            
            # Get the first variable
            result = merged[list(merged.data_vars)[0]]
            
            # Fill NaN with values from other rasters
            for i, r in enumerate(rasters[1:], 1):
                temp_name = f"temp_{i}"
                if temp_name in merged:
                    result = result.fillna(merged[temp_name])
            
            return result.rename(var_name)
            
        except Exception as e:
            logger.warning(f"Mosaicking failed: {e}")
            # Return first raster as fallback
            result = rasters[0]
            if 'band' in result.dims and result.band.size == 1:
                result = result.squeeze('band', drop=True)
            return result.rename(var_name)
    
    def _process_ghsl_type(
        self,
        ghsl_type: str,
        tile_ids: List[str]
    ) -> Optional[ProcessingResult]:
        """Process all tiles for a GHSL type."""
        if ghsl_type not in self.GHSL_MAPPINGS:
            logger.error(f"Invalid GHSL type: {ghsl_type}")
            return None
        
        mapping = self.GHSL_MAPPINGS[ghsl_type]
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"  Processing GHSL: {ghsl_type} ({len(tile_ids)} tiles)")
        
        # Process tiles in parallel, reusing the shared pool
        tiles_data = []

        futures = {
            self.executor.submit(
                self._process_ghsl_tile,
                tile_id,
                ghsl_type,
                mapping
            ): tile_id
            for tile_id in tile_ids
        }

        for future in as_completed(futures):
            tile_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    tiles_data.append(result)
            except Exception as e:
                logger.warning(f"Tile {tile_id} failed: {e}")
        
        if not tiles_data:
            logger.warning(f"No tiles processed for {ghsl_type}")
            return ProcessingResult(
                name=ghsl_type,
                success=False,
                error="No tiles processed",
                processing_time=time.time() - start_time
            )
        
        # Mosaic tiles
        if self.verbose:
            logger.info(f"    Mosaicking {len(tiles_data)} tiles...")
        
        mosaic = self._mosaic_rasters(tiles_data, mapping.var_name)
        
        if mosaic is None:
            return ProcessingResult(
                name=ghsl_type,
                success=False,
                error="Mosaicking failed",
                processing_time=time.time() - start_time
            )
        
        # Reproject and resample
        mosaic = self._resample_to_target(mosaic)
        
        # Extract at stations
        extracted = self._extract_values_at_stations(mosaic)
        
        id_col = self.stations.columns[0]
        df = pd.DataFrame({
            id_col: self.stations[id_col].values,
            mapping.var_name: extracted
        })
        
        return ProcessingResult(
            df=df,
            raster=mosaic,
            name=ghsl_type,
            success=True,
            processing_time=time.time() - start_time
        )
    
    def _build_variable_definitions(
        self,
        variables: Optional[List[str]] = None,
        process_wumpod: bool = True,
        process_vegetation: bool = True,
        process_directional: bool = True
    ) -> Dict[str, VariableDefinition]:
        """Build variable definitions dictionary."""
        definitions = {}
        
        if process_wumpod:
            definitions.update({
                "elevation": VariableDefinition(
                    url="https://zenodo.org/records/16941635/files/global_evevation.tif?download=1",
                    name="elevation",
                    description="Elevation (m)",
                    na_to_zero=False,
                    category="wumpod"
                ),
                "frc_esa": VariableDefinition(
                    url="https://zenodo.org/records/10039127/files/frc_esa.tif?download=1",
                    name="frc_esa",
                    description="Building fraction (ESA)",
                    na_to_zero=True,
                    category="wumpod"
                ),
                "hgt": VariableDefinition(
                    url="https://zenodo.org/records/10039127/files/hgt.tif?download=1",
                    name="hgt",
                    description="Building height (m)",
                    na_to_zero=True,
                    category="wumpod"
                ),
                "lb": VariableDefinition(
                    url="https://zenodo.org/records/10039127/files/lb.tif?download=1",
                    name="lb",
                    description="Building length (m)",
                    na_to_zero=True,
                    category="wumpod"
                ),
                "lc": VariableDefinition(
                    url="https://zenodo.org/records/10039127/files/lc.tif?download=1",
                    name="lc",
                    description="Building coverage fraction",
                    na_to_zero=True,
                    category="wumpod"
                ),
                "lf": VariableDefinition(
                    url="https://zenodo.org/records/10039127/files/lf.tif?download=1",
                    name="lf",
                    description="Land fraction",
                    na_to_zero=True,
                    category="wumpod"
                ),
                "lp": VariableDefinition(
                    url="https://zenodo.org/records/10039127/files/lp.tif?download=1",
                    name="lp",
                    description="Building perimeter (m)",
                    na_to_zero=True,
                    category="wumpod"
                ),
                "urban_frc": VariableDefinition(
                    url="https://zenodo.org/records/7298393/files/urban_fraction_300m.tif?download=1",
                    name="urban_frc",
                    description="Urban fraction (300m)",
                    na_to_zero=True,
                    category="wumpod"
                ),
                "cglc": VariableDefinition(
                    url="https://zenodo.org/records/7670653/files/CGLC_MODIS_LCZ.tif?download=1",
                    name="cglc",
                    description="CGLC MODIS LCZ classification",
                    na_to_zero=True,
                    category="wumpod"
                )
            })
        
        if process_vegetation:
            definitions.update({
                "tree": VariableDefinition(
                    url="https://zenodo.org/records/14439377/files/tree.cover_glc.fcd30d_p_30m_20220101_20221231_go_epsg4326_v20241210.tif?download=1",
                    name="tree",
                    description="Tree cover (%) from GLC_FCS30D",
                    na_to_zero=False,
                    category="vegetation"
                ),
                "urban": VariableDefinition(
                    url="https://zenodo.org/records/14439377/files/urban.cover_glc.fcd30d_p_30m_20220101_20221231_go_epsg4326_v20241210.tif?download=1",
                    name="urban",
                    description="Impervious surfaces (%) from GLC_FCS30D",
                    na_to_zero=True,
                    category="vegetation"
                )
            })
        
        if process_directional:
            directions = ["0", "45", "90", "135"]
            roughness_descriptions = {
                "zdm": "Zero-plane displacement",
                "zdr": "Roughness length",
                "zom": "Momentum roughness",
                "zor": "Thermal roughness"
            }
            
            for direction in directions:
                # Land fraction directional
                definitions[f"lf_{direction}"] = VariableDefinition(
                    url=f"https://zenodo.org/records/10039127/files/lf_{direction}.tif?download=1",
                    name=f"lf_{direction}",
                    description=f"Land fraction ({direction}°)",
                    na_to_zero=True,
                    category="directional"
                )
                
                # Height indices (excluding 0 and 135)
                if direction not in ["0", "135"]:
                    definitions[f"hi_{direction}"] = VariableDefinition(
                        url=f"https://zenodo.org/records/10039127/files/hi_{direction}.tif?download=1",
                        name=f"hi_{direction}",
                        description=f"Height index ({direction}°)",
                        na_to_zero=True,
                        category="directional"
                    )
                
                # Roughness parameters
                for prefix, desc in roughness_descriptions.items():
                    definitions[f"{prefix}_{direction}"] = VariableDefinition(
                        url=f"https://zenodo.org/records/10039127/files/{prefix}_{direction}.tif?download=1",
                        name=f"{prefix}_{direction}",
                        description=f"{desc} ({direction}°)",
                        na_to_zero=True,
                        category="directional"
                    )
        
        # Filter if specific variables requested
        if variables is not None:
            definitions = {
                k: v for k, v in definitions.items()
                if k in variables
            }
            
            # Warn about missing variables
            missing = set(variables) - set(definitions.keys())
            if missing:
                logger.warning(f"Variables not found: {missing}")
        
        return definitions
    
    def _save_stack(self, combined_rasters: xr.Dataset) -> str:
        """Write the processed parameters as a multi-band GeoTIFF stack.

        Used when no stations are given, so the raster stack (already
        cropped/aligned to the lcz_map grid via _resample_to_target) is the
        deliverable, following the same GeoTIFF conventions as
        lcz_get_parameters.write_stack_concurrent.
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "lcz_ucp_stack.tif")

        band_names = list(combined_rasters.data_vars)
        stacked = combined_rasters.to_array(dim="band")
        stacked.rio.write_crs(self.target_crs, inplace=True)
        stacked.rio.write_nodata(np.nan, inplace=True)
        stacked.rio.to_raster(out_path, compress="lzw", predictor=3, dtype="float32")

        with rasterio.open(out_path, "r+") as dst:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
                dst.update_tags(i, NAME=name)

        return out_path

    def process(
        self,
        variables: Optional[List[str]] = None,
        ghsl_tiles: Optional[List[str]] = None,
        process_ghsl: bool = True,
        process_wumpod: bool = True,
        process_vegetation: bool = True,
        process_directional: bool = True,
        use_threads: bool = True,
        fail_fast: bool = False
    ) -> Dict[str, Any]:
        """
        Process all variables in parallel.
        
        Parameters
        ----------
        variables : list of str, optional
            Specific variables to process (None = all)
        ghsl_tiles : list of str, optional
            Specific GHSL tile IDs (None = auto-detect)
        process_ghsl : bool
            Whether to process GHSL data
        process_wumpod : bool
            Whether to process WUMPOD data
        process_vegetation : bool
            Whether to process vegetation data
        process_directional : bool
            Whether to process directional parameters
        use_threads : bool
            Use threads (True) or processes (False). Downloads are I/O-bound,
            so threads are sufficient and avoid pickling ``self`` (which holds
            a ``requests.Session`` and a ``diskcache.FanoutCache`` — neither
            survives being sent to a worker process).
        fail_fast : bool
            If True, stop on first error; if False, continue processing

        Returns
        -------
        dict
            Dictionary containing:
            - 'df_vars': DataFrame with extracted values at stations
            - 'combined_rasters': xarray Dataset with all processed rasters
            - 'variable_list': List of successfully processed variables
            - 'failed_variables': List of (variable, error) tuples
            - 'summary': ProcessingSummary object
        """
        if not use_threads:
            raise ValueError(
                "use_threads=False (ProcessPoolExecutor) is not supported: "
                "this processor holds a requests.Session and a diskcache.FanoutCache "
                "on self, neither of which can be pickled to a worker process. "
                "Downloads are I/O-bound, so use_threads=True (the default) already "
                "parallelizes effectively."
            )

        start_time = time.time()
        results = []
        failed = []
        
        # Build variable definitions
        var_definitions = self._build_variable_definitions(
            variables=variables,
            process_wumpod=process_wumpod,
            process_vegetation=process_vegetation,
            process_directional=process_directional
        )
        
        if not var_definitions and not process_ghsl:
            raise ValueError("No variables to process. Check your filters.")
        
        # Process main variables in parallel
        if var_definitions:
            if self.verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {len(var_definitions)} variables")
                logger.info(f"{'='*60}")
            
            futures = {
                self.executor.submit(
                    self._process_single_variable,
                    name,
                    var_def
                ): (name, var_def)
                for name, var_def in var_definitions.items()
            }

            with tqdm(
                total=len(futures),
                desc="Variables",
                disable=not self.verbose,
                unit="var"
            ) as pbar:
                for future in as_completed(futures):
                    var_name, _ = futures[future]

                    try:
                        result = future.result(timeout=600)  # 10 min timeout

                        if result.success:
                            results.append(result)
                            pbar.set_postfix_str(f"✓ {var_name}")
                        else:
                            failed.append((var_name, result.error))
                            pbar.set_postfix_str(f"✗ {var_name}")

                            if fail_fast:
                                raise RuntimeError(f"Failed: {var_name}")

                    except Exception as e:
                        failed.append((var_name, str(e)))
                        pbar.set_postfix_str(f"✗ {var_name}")

                        if fail_fast:
                            raise
                    finally:
                        pbar.update(1)
        
        # Process GHSL data
        if process_ghsl:
            if self.verbose:
                logger.info(f"\n{'='*60}")
                logger.info("Processing GHSL Data")
                logger.info(f"{'='*60}")
            
            if ghsl_tiles is None:
                ghsl_tiles = self._get_ghsl_tiles()
                if self.verbose:
                    logger.info(f"Detected {len(ghsl_tiles)} GHSL tiles")
            
            if ghsl_tiles:
                for ghsl_type in self.GHSL_MAPPINGS.keys():
                    try:
                        result = self._process_ghsl_type(ghsl_type, ghsl_tiles)
                        if result:
                            if result.success:
                                results.append(result)
                            else:
                                failed.append((ghsl_type, result.error))
                                
                                if fail_fast:
                                    raise RuntimeError(f"GHSL failed: {ghsl_type}")
                    except Exception as e:
                        failed.append((ghsl_type, str(e)))
                        
                        if fail_fast:
                            raise
        
        # Build summary
        total_time = time.time() - start_time
        summary = ProcessingSummary(
            total_variables=len(var_definitions) + (len(self.GHSL_MAPPINGS) if process_ghsl else 0),
            successful=len(results),
            failed=len(failed),
            total_time=total_time,
            failed_variables=failed
        )
        
        # Combine results
        if not results:
            raise RuntimeError(
                "No variables were successfully processed. "
                f"Failed: {[f[0] for f in failed]}"
            )
        
        # Combine DataFrames
        dfs = [r.df for r in results if r.df is not None]
        if dfs:
            id_col = dfs[0].columns[0]
            df_vars = dfs[0].copy()
            for df in dfs[1:]:
                df_vars = df_vars.merge(df, on=id_col, how='left')
        elif self.stations is not None:
            df_vars = self.stations.copy()
        else:
            df_vars = None

        # Combine rasters
        raster_dict = {}
        for r in results:
            if r.raster is not None:
                name = r.raster.name if r.raster.name else r.name
                raster_dict[name] = r.raster

        combined_rasters = xr.Dataset(raster_dict) if raster_dict else None

        # No stations -> the raster stack is the deliverable, save it to disk
        stack_path = None
        if self.stations is None and combined_rasters is not None:
            stack_path = self._save_stack(combined_rasters)

        # Build output
        output = {
            'df_vars': df_vars,
            'combined_rasters': combined_rasters,
            'stack_path': stack_path,
            'variable_list': [r.name for r in results if r.success],
            'failed_variables': failed,
            'summary': summary
        }
        
        if self.verbose:
            logger.info(f"\n{'='*60}")
            logger.info("Processing Complete")
            logger.info(f"{'='*60}")
            logger.info(f"✓ Successful: {summary.successful}")
            logger.info(f"✗ Failed: {summary.failed}")
            logger.info(f"📊 Success Rate: {summary.success_rate:.1f}%")
            logger.info(f"⏱️  Total Time: {summary.total_time:.1f}s")
            
            if failed:
                logger.warning(f"Failed variables: {[f[0] for f in failed]}")
            
            if combined_rasters:
                logger.info(f"Raster layers: {len(combined_rasters.data_vars)}")
            if df_vars is not None:
                logger.info(f"DataFrame: {len(df_vars)} rows × {len(df_vars.columns)} cols")
            if stack_path:
                logger.info(f"Stack saved: {stack_path}")
            logger.info(f"{'='*60}")
        
        return output
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_cache_stats()
    
    def close(self):
        """Clean up resources."""
        try:
            self.executor.shutdown(wait=True)
        except Exception:
            pass
        try:
            self.cache.close()
        except Exception:
            pass
        try:
            self.session.close()
        except Exception:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def lcz_get_ucp(
    lcz_map: Any,
    stations: Optional[Any] = None,
    cache_dir: str = "lcz4r_cache",
    n_workers: Optional[int] = None,
    verbose: bool = True,
    variables: Optional[List[str]] = None,
    ghsl_tiles: Optional[List[str]] = None,
    process_ghsl: bool = True,
    process_wumpod: bool = True,
    process_vegetation: bool = True,
    process_directional: bool = True,
    use_threads: bool = True,
    fail_fast: bool = False
) -> Dict[str, Any]:
    """
    Download and Process Urban Physical Parameters in Parallel.
    
    This is the main convenience function that wraps the UrbanParameterProcessor
    class for simple one-shot usage.
    
    Parameters
    ----------
    lcz_map : str, Path, or xarray.DataArray
        LCZ classification raster defining the study area
    stations : GeoDataFrame, str, Path, or DataFrame, optional
        Station locations with coordinates. If None (default), station-value
        extraction is skipped and the processed parameters are instead saved
        as a raster stack cropped to the ``lcz_map`` extent
        (``LCZ4r_output/lcz_ucp_stack.tif``).
    cache_dir : str, optional
        Directory for caching. Default: "lcz4r_cache"
    n_workers : int, optional
        Number of parallel workers. Default: CPU count - 1
    verbose : bool, optional
        Print progress messages. Default: True
    variables : list of str, optional
        Specific variables to process. Default: all
    ghsl_tiles : list of str, optional
        Specific GHSL tile IDs. Default: auto-detect
    process_ghsl : bool, optional
        Process GHSL data. Default: True
    process_wumpod : bool, optional
        Process WUMPOD data. Default: True
    process_vegetation : bool, optional
        Process vegetation data. Default: True
    process_directional : bool, optional
        Process directional parameters. Default: True
    use_threads : bool, optional
        Use threads (True) or processes. Default: True
    fail_fast : bool, optional
        Stop on first error. Default: False (continue on errors)
    
    Returns
    -------
    dict
        Results dictionary with:
        - 'df_vars': DataFrame with extracted values (None if stations is None)
        - 'combined_rasters': xarray Dataset with rasters
        - 'stack_path': path to the saved GeoTIFF stack (set only when
          stations is None; otherwise None)
        - 'variable_list': List of successful variables
        - 'failed_variables': List of (var, error) tuples
        - 'summary': ProcessingSummary

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>>
    >>> # Create station data
    >>> stations = gpd.GeoDataFrame(
    ...     {'station': ['S1', 'S2', 'S3']},
    ...     geometry=[
    ...         Point(-46.6, -23.5),
    ...         Point(-46.65, -23.55),
    ...         Point(-46.55, -23.45)
    ...     ],
    ...     crs='EPSG:4326'
    ... )
    >>>
    >>> # Process all variables
    >>> result = lcz_get_ucp(
    ...     lcz_map='lcz_sao_paulo.tif',
    ...     stations=stations,
    ...     n_workers=4
    ... )
    >>>
    >>> # Access results
    >>> print(result['df_vars'].head())
    >>> print(result['summary'].to_dict())
    >>>
    >>> # No stations: get a raster stack instead
    >>> result = lcz_get_ucp(lcz_map='lcz_sao_paulo.tif', n_workers=4)
    >>> print(result['stack_path'])
    
    References
    ----------
    - Patel, P., & Roth, M. (2024). WUMPOD. Zenodo.
    - Patel, P., & Roth, M. (2022). Global Urban Fraction. Zenodo.
    - Zhang, X., et al. (2021). GLC_FCS30D. Earth Syst. Sci. Data.
    - Tolan, J., et al. (2024). Global Canopy Height. Zenodo.
    - Melchiorri, M., et al. (2025). GHSL Data Package 2025. JRC.
    """
    with UrbanParameterProcessor(
        lcz_map=lcz_map,
        stations=stations,
        cache_dir=cache_dir,
        n_workers=n_workers,
        verbose=verbose
    ) as processor:
        return processor.process(
            variables=variables,
            ghsl_tiles=ghsl_tiles,
            process_ghsl=process_ghsl,
            process_wumpod=process_wumpod,
            process_vegetation=process_vegetation,
            process_directional=process_directional,
            use_threads=use_threads,
            fail_fast=fail_fast
        )


# =============================================================================
# MAIN ENTRY POINT FOR CLI USAGE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and process urban physical parameters in parallel"
    )
    parser.add_argument(
        "lcz_map",
        help="Path to LCZ classification raster"
    )
    parser.add_argument(
        "stations",
        help="Path to stations file (GeoPackage, Shapefile, or CSV with lat/lon)"
    )
    parser.add_argument(
        "--cache-dir", "-c",
        default="lcz4r_cache",
        help="Cache directory"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--variables", "-v",
        nargs="+",
        default=None,
        help="Specific variables to process"
    )
    parser.add_argument(
        "--no-ghsl",
        action="store_true",
        help="Skip GHSL processing"
    )
    parser.add_argument(
        "--no-wumpod",
        action="store_true",
        help="Skip WUMPOD processing"
    )
    parser.add_argument(
        "--no-vegetation",
        action="store_true",
        help="Skip vegetation processing"
    )
    parser.add_argument(
        "--no-directional",
        action="store_true",
        help="Skip directional parameters"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path for results (CSV)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error"
    )
    
    args = parser.parse_args()
    
    # Run processing
    result = lcz_get_ucp(
        lcz_map=args.lcz_map,
        stations=args.stations,
        cache_dir=args.cache_dir,
        n_workers=args.workers,
        verbose=not args.quiet,
        variables=args.variables,
        process_ghsl=not args.no_ghsl,
        process_wumpod=not args.no_wumpod,
        process_vegetation=not args.no_vegetation,
        process_directional=not args.no_directional,
        fail_fast=args.fail_fast
    )
    
    # Save to file if requested
    if args.output:
        result['df_vars'].to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")