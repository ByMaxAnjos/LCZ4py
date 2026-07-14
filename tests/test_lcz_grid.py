"""Test suite for the raster lcz_grid_* gridded environmental data
downloaders (crop+reproject onto an existing LCZ classification GeoTIFF's
grid, the same pattern as lcz_get_lst) and the plot_lcz_relationship /
plot_grid_only visualization functions.

Fast unit tests use fabricated cache files (no internet). Network tests hit
real Zenodo endpoints and are opt-in:

    pytest test_lcz_grid.py -m "not network"   # fast, offline
    pytest test_lcz_grid.py -m network          # integration tests
"""
from __future__ import annotations

import gzip
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def muni_gdf():
    return gpd.GeoDataFrame(
        {"code_muni": ["5103403", "5002704"]},
        geometry=[box(-56, -15, -55, -14), box(-54, -20, -53, -19)],
        crs=4326,
    )


@pytest.fixture
def lcz_map_path(tmp_path):
    """A small fabricated LCZ classification GeoTIFF (classes 1-17), inside
    the bbox every fabricated raw raster below is built to cover."""
    rng = np.random.default_rng(0)
    h, w = 20, 20
    classes = rng.integers(1, 18, size=(h, w)).astype("uint8")
    transform = from_origin(-56.2, -15.0, 0.01, 0.01)  # lon -56.2..-56.0, lat -15.2..-15.0
    path = str(tmp_path / "lcz_map.tif")
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                        dtype="uint8", crs="EPSG:4326", transform=transform) as dst:
        dst.write(classes, 1)
    return path


# ── lcz_grid_chirps (offline, fabricated raster cache) ───────────────────────

def test_chirps_raster_aggregation(tmp_path, lcz_map_path):
    from LCZ4py.general.lcz_grid_chirps import lcz_grid_chirps

    cache_dir = str(tmp_path / "chirps")
    os.makedirs(os.path.join(cache_dir, "monthly"), exist_ok=True)
    transform = from_origin(-60, -10, 0.05, 0.05)  # covers the LCZ map's small bbox
    data = np.random.uniform(0, 300, size=(200, 200))
    plain = os.path.join(cache_dir, "monthly", "chirps-v2.0.2020.01.tif")
    with rasterio.open(plain, "w", driver="GTiff", height=200, width=200, count=1,
                        dtype="float64", crs="EPSG:4326", transform=transform) as dst:
        dst.write(data, 1)
    with open(plain, "rb") as f_in, gzip.open(plain + ".gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(plain)

    out = lcz_grid_chirps(lcz_map_path, resolution="monthly", years=[2020], months=[1],
                           cache=True, cache_dir=cache_dir, verbose=False)
    assert out.array.shape == (1, 20, 20)
    assert out.bands == ["2020-01-01"]
    assert out.variables == ["rainfall_chirps_mm"]
    assert np.isfinite(out.array).any()


def test_chirps_invalid_resolution_raises(lcz_map_path):
    from LCZ4py.general.lcz_grid_chirps import lcz_grid_chirps

    with pytest.raises(ValueError):
        lcz_grid_chirps(lcz_map_path, resolution="weekly", verbose=False)


# ── lcz_grid_pdsi (offline, fabricated NetCDF cache) ─────────────────────────

def test_pdsi_raster_aggregation(tmp_path, lcz_map_path):
    xr = pytest.importorskip("xarray")
    from LCZ4py.general.lcz_grid_pdsi import lcz_grid_pdsi

    cache_dir = str(tmp_path / "pdsi")
    os.makedirs(os.path.join(cache_dir, "terraclimate"), exist_ok=True)
    times = pd.date_range("2020-01-01", periods=3, freq="MS")
    lon = np.linspace(-80, 20, 60)
    lat = np.linspace(10, -40, 30)
    data = np.random.uniform(-4, 4, size=(3, 30, 60))
    ds = xr.Dataset({"PDSI": (("time", "lat", "lon"), data)}, coords={"time": times, "lat": lat, "lon": lon})
    ds.to_netcdf(os.path.join(cache_dir, "terraclimate", "TerraClimate_PDSI_2020.nc"))

    out = lcz_grid_pdsi(lcz_map_path, years=[2020], months=[1, 2, 3], source="terraclimate",
                         cache=True, cache_dir=cache_dir, verbose=False)
    assert out.array.shape == (3, 20, 20)
    assert out.bands == ["2020-01-01", "2020-02-01", "2020-03-01"]


# ── lcz_grid_era5 (offline, fabricated NetCDF cache) ─────────────────────────

def test_era5_raster_aggregation(tmp_path, lcz_map_path):
    xr = pytest.importorskip("xarray")
    from LCZ4py.general.lcz_grid_era5 import lcz_grid_era5, _nc_filename

    cache_dir = str(tmp_path / "era5")
    year_dir = os.path.join(cache_dir, "2020")
    os.makedirs(year_dir, exist_ok=True)
    times = pd.date_range("2020-01-01", periods=31, freq="D")
    lon = np.linspace(-80, 20, 60)
    lat = np.linspace(10, -40, 30)
    data = np.random.uniform(270, 305, size=(31, 30, 60))
    ds = xr.Dataset({"t2m": (("time", "lat", "lon"), data)}, coords={"time": times, "lat": lat, "lon": lon})
    fn = _nc_filename("2m_temperature", 2020, 1, "mean")
    ds.to_netcdf(os.path.join(year_dir, fn))

    out = lcz_grid_era5(lcz_map_path, years=[2020], months=[1], vars=["t2m"],
                         cache=True, cache_dir=cache_dir, verbose=False)
    assert out.array.shape == (31, 20, 20)


def test_era5_out_of_latam_coverage_raises(tmp_path):
    from LCZ4py.general.lcz_grid_era5 import lcz_grid_era5

    rng = np.random.default_rng(1)
    transform = from_origin(13.0, 53.0, 0.01, 0.01)  # Berlin: outside the LatAm mirror's coverage
    path = str(tmp_path / "berlin_lcz.tif")
    with rasterio.open(path, "w", driver="GTiff", height=20, width=20, count=1,
                        dtype="uint8", crs="EPSG:4326", transform=transform) as dst:
        dst.write(rng.integers(1, 18, size=(20, 20)).astype("uint8"), 1)

    with pytest.raises(ValueError):
        lcz_grid_era5(path, years=[2020], months=[1], vars=["t2m"], verbose=False)


# ── lcz_grid_era5_global (offline, fabricated NetCDF + monkeypatched CDS) ────

def test_era5_global_no_auth_raises(lcz_map_path, monkeypatch):
    from LCZ4py.general.lcz_grid_era5_global import lcz_grid_era5_global

    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    with pytest.raises(ValueError):
        lcz_grid_era5_global(lcz_map_path, years=[2020], months=[1], verbose=False)


def test_era5_global_invalid_vars_raises(lcz_map_path, monkeypatch):
    from LCZ4py.general.lcz_grid_era5_global import lcz_grid_era5_global

    monkeypatch.setenv("CDSAPI_KEY", "fake")
    with pytest.raises(ValueError):
        lcz_grid_era5_global(lcz_map_path, years=[2020], vars=["bogus"], verbose=False)


def test_era5_global_raster_aggregation(tmp_path, lcz_map_path, monkeypatch):
    xr = pytest.importorskip("xarray")
    import importlib; mod = importlib.import_module("LCZ4py.general.lcz_grid_era5_global")

    monkeypatch.setenv("CDSAPI_KEY", "fake")

    def fake_retrieve(request, target_path, cds_key, cds_url):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        lat = np.linspace(-14.5, -15.5, 20)
        lon = np.linspace(-56.5, -55.5, 20)
        data = np.random.uniform(290, 300, size=(1, 20, 20))
        ds = xr.Dataset(
            {"t2m": (("time", "latitude", "longitude"), data)},
            coords={"time": [0], "latitude": lat, "longitude": lon},
        )
        ds.to_netcdf(target_path)

    monkeypatch.setattr(mod, "_cds_retrieve", fake_retrieve)

    out = mod.lcz_grid_era5_global(lcz_map_path, years=[2020], months=[1], vars=["t2m"],
                                    cache=True, cache_dir=str(tmp_path / "era5_global"), verbose=False)
    assert out.array.shape == (1, 20, 20)
    assert out.bands == ["t2m_2020-01-01"]
    assert np.isfinite(out.array).any()


def test_era5_global_works_outside_latam(tmp_path, monkeypatch):
    xr = pytest.importorskip("xarray")
    import importlib; mod = importlib.import_module("LCZ4py.general.lcz_grid_era5_global")

    monkeypatch.setenv("CDSAPI_KEY", "fake")

    rng = np.random.default_rng(1)
    transform = from_origin(13.0, 53.0, 0.01, 0.01)  # Berlin
    lcz_path = str(tmp_path / "berlin_lcz.tif")
    with rasterio.open(lcz_path, "w", driver="GTiff", height=20, width=20, count=1,
                        dtype="uint8", crs="EPSG:4326", transform=transform) as dst:
        dst.write(rng.integers(1, 18, size=(20, 20)).astype("uint8"), 1)

    def fake_retrieve(request, target_path, cds_key, cds_url):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        lat = np.linspace(53.0, 52.5, 20)
        lon = np.linspace(12.5, 13.5, 20)
        data = np.random.uniform(285, 295, size=(1, 20, 20))
        ds = xr.Dataset(
            {"t2m": (("time", "latitude", "longitude"), data)},
            coords={"time": [0], "latitude": lat, "longitude": lon},
        )
        ds.to_netcdf(target_path)

    monkeypatch.setattr(mod, "_cds_retrieve", fake_retrieve)

    out = mod.lcz_grid_era5_global(lcz_path, years=[2023], months=[6], vars=["t2m"],
                                    cache=True, cache_dir=str(tmp_path / "era5_global_berlin"), verbose=False)
    assert np.isfinite(out.array).any()
    assert out.bands == ["t2m_2023-06-01"]


# ── lcz_grid_pollution_merra2 (offline, fabricated NetCDF, fake creds) ───────

def test_merra2_pm25_formula(tmp_path, lcz_map_path, monkeypatch):
    xr = pytest.importorskip("xarray")
    from LCZ4py.general.lcz_grid_pollution_merra2 import lcz_grid_pollution_merra2, _file_info

    monkeypatch.setenv("EARTHDATA_USER", "u")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "p")

    cache_dir = str(tmp_path / "merra2")
    os.makedirs(cache_dir, exist_ok=True)
    fn, _ = _file_info(2020, 1)
    lon = np.linspace(-80, 20, 100)
    lat = np.linspace(10, -40, 50)

    def rvar(scale):
        return np.random.uniform(scale * 0.5, scale, size=(50, 100))  # keep strictly positive

    ds = xr.Dataset({
        "DUSMASS25": (("lat", "lon"), rvar(1e-9)), "SSSMASS25": (("lat", "lon"), rvar(1e-9)),
        "BCSMASS": (("lat", "lon"), rvar(1e-10)), "OCSMASS": (("lat", "lon"), rvar(1e-9)),
        "SO4SMASS": (("lat", "lon"), rvar(1e-9)),
    }, coords={"lat": lat, "lon": lon})
    ds.to_netcdf(os.path.join(cache_dir, fn))

    out = lcz_grid_pollution_merra2(lcz_map_path, pollutants=["pm25"], years=[2020], months=[1],
                                     cache=True, cache_dir=cache_dir, verbose=False)
    assert out.array.shape == (1, 20, 20)
    assert out.bands == ["pm25_2020-01-01"]
    assert np.nanmin(out.array) > 0


def test_merra2_no_auth_raises(lcz_map_path, monkeypatch):
    from LCZ4py.general.lcz_grid_pollution_merra2 import lcz_grid_pollution_merra2

    monkeypatch.delenv("EARTHDATA_USER", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    with pytest.raises(ValueError):
        lcz_grid_pollution_merra2(lcz_map_path, pollutants=["pm25"], years=[2020], verbose=False)


# ── lcz_grid_pollution_ghap (offline, CF-NetCDF fixture matching the real file) ──

def test_ghap_raster_aggregation(tmp_path, lcz_map_path):
    xr = pytest.importorskip("xarray")
    from LCZ4py.general.lcz_grid_pollution_ghap import lcz_grid_pollution_ghap, _file_info

    # Mirrors the real GHAP_*.nc structure: proper lat (descending,
    # north-first)/lon coords, confirmed by downloading and inspecting the
    # actual Zenodo file (doi:10.5281/zenodo.10208188).
    lat = np.linspace(10.0, -40.0, 100)  # descending: north-first
    lon = np.linspace(-80.0, 20.0, 200)
    lat_grid = np.tile(lat.reshape(-1, 1), (1, 200)) + 30  # keep values positive
    ds = xr.Dataset({"PM2.5": (("lat", "lon"), lat_grid)}, coords={"lat": lat, "lon": lon})

    cache_dir = str(tmp_path / "ghap")
    filename, _ = _file_info("pm25", "annual", 2020)
    os.makedirs(os.path.join(cache_dir, "pm25"), exist_ok=True)
    ds.to_netcdf(os.path.join(cache_dir, "pm25", filename))

    out = lcz_grid_pollution_ghap(lcz_map_path, pollutants=["pm25"], resolution="annual", years=[2020],
                                   cache=True, cache_dir=cache_dir, verbose=False)
    assert out.array.shape == (1, 20, 20)
    assert out.bands == ["pm25_2020-01-01"]
    assert np.isfinite(out.array).any()


def test_ghap_invalid_pollutant_raises(lcz_map_path):
    from LCZ4py.general.lcz_grid_pollution_ghap import lcz_grid_pollution_ghap

    with pytest.raises(ValueError):
        lcz_grid_pollution_ghap(lcz_map_path, pollutants=["bogus"], verbose=False)


# ── plot_lcz_relationship (offline) ───────────────────────────────────────────

def test_plot_lcz_relationship_box(tmp_path):
    from LCZ4py.general.plot_lcz_relationship import plot_lcz_relationship

    transform = from_origin(-46.7, -23.4, 0.001, 0.001)
    h, w = 30, 30
    rng = np.random.default_rng(0)
    lcz = rng.choice(np.arange(1, 18), size=(h, w)).astype("float64")
    lcz_path = str(tmp_path / "lcz.tif")
    with rasterio.open(lcz_path, "w", driver="GTiff", height=h, width=w, count=1,
                        dtype="float64", crs="EPSG:4326", transform=transform) as dst:
        dst.write(lcz, 1)

    temp = 35 - lcz * 0.8
    temp_path = str(tmp_path / "temp.tif")
    with rasterio.open(temp_path, "w", driver="GTiff", height=h, width=w, count=1,
                        dtype="float64", crs="EPSG:4326", transform=transform) as dst:
        dst.write(temp, 1)

    fig = plot_lcz_relationship(lcz_path, temp_path, variable_name="Temp", plot_type="heatmap")
    z = fig.data[0].z[0]
    assert list(z) == sorted(z, reverse=True)  # higher LCZ id -> lower temp, by construction


def test_plot_lcz_relationship_bad_type_raises(tmp_path):
    from LCZ4py.general.plot_lcz_relationship import plot_lcz_relationship

    transform = from_origin(-46.7, -23.4, 0.001, 0.001)
    arr = np.ones((5, 5))
    p1, p2 = str(tmp_path / "a.tif"), str(tmp_path / "b.tif")
    for p in (p1, p2):
        with rasterio.open(p, "w", driver="GTiff", height=5, width=5, count=1,
                            dtype="float64", crs="EPSG:4326", transform=transform) as dst:
            dst.write(arr, 1)
    with pytest.raises(ValueError):
        plot_lcz_relationship(p1, p2, plot_type="bogus")


# ── plot_grid_only (offline) ──────────────────────────────────────────────────

def test_plot_grid_only_no_basemap(muni_gdf):
    import matplotlib
    matplotlib.use("Agg")
    from LCZ4py.general.plot_grid_only import plot_grid_only

    fig = plot_grid_only(muni_gdf, add_basemap=False)
    assert fig is not None


def test_plot_grid_only_bad_column_raises(muni_gdf):
    import matplotlib
    matplotlib.use("Agg")
    from LCZ4py.general.plot_grid_only import plot_grid_only

    with pytest.raises(ValueError):
        plot_grid_only(muni_gdf, color_by="bogus", add_basemap=False)


# ── Network-dependent integration tests (opt-in) ─────────────────────────────

@pytest.mark.network
def test_ghap_real_o3_file(tmp_path, lcz_map_path):
    from LCZ4py.general.lcz_grid_pollution_ghap import lcz_grid_pollution_ghap

    out = lcz_grid_pollution_ghap(lcz_map_path, pollutants=["o3"], resolution="annual", years=[2010],
                                   cache=True, cache_dir=str(tmp_path))
    assert 0 < np.nanmean(out.array) < 200


# ── lcz_cal_indexes (offline) ─────────────────────────────────────────────────

def test_cal_indexes_summary(lcz_map_path):
    from LCZ4py.general.lcz_grid_chirps import lcz_grid_chirps
    from LCZ4py.general.lcz_cal_indexes import lcz_cal_indexes

    cache_dir = os.path.join(os.path.dirname(lcz_map_path), "chirps")
    os.makedirs(os.path.join(cache_dir, "annual"), exist_ok=True)
    transform = from_origin(-60, -10, 0.05, 0.05)
    data = np.random.uniform(800, 1400, size=(200, 200))
    with rasterio.open(os.path.join(cache_dir, "annual", "chirps-v2.0.2020.tif"), "w", driver="GTiff",
                        height=200, width=200, count=1, dtype="float64", crs="EPSG:4326", transform=transform) as dst:
        dst.write(data, 1)

    grid_result = lcz_grid_chirps(lcz_map_path, resolution="annual", years=[2020],
                                   cache=True, cache_dir=cache_dir, verbose=False)
    out = lcz_cal_indexes(lcz_map_path, grid_result, variable_name="rainfall_mm")
    assert set(out.df.columns) >= {"lcz", "lcz_name", "color", "n_pixels", "mean", "std", "min", "max", "median"}
    assert (out.df["lcz"].to_numpy() >= 1).all() and (out.df["lcz"].to_numpy() <= 17).all()
    assert out.fig is not None

    df_only = lcz_cal_indexes(lcz_map_path, grid_result, iplot=False)
    assert not hasattr(df_only, "fig")


def test_cal_indexes_no_overlap_raises(lcz_map_path):
    from LCZ4py._internal._lcz_grid_raster_base import LCZGridResult
    from LCZ4py.general.lcz_cal_indexes import lcz_cal_indexes

    all_nan = LCZGridResult(array=np.full((1, 20, 20), np.nan), bands=["2020-01-01"], variables=["x"])
    with pytest.raises(ValueError):
        lcz_cal_indexes(lcz_map_path, all_nan)


# ── lcz_climate_compute_spi / spei ────────────────────────────────────────────

@pytest.fixture
def monthly_rainfall_df():
    rng = np.random.default_rng(42)
    n_months = 30 * 12
    dates = pd.date_range("1990-01-01", periods=n_months, freq="MS")
    rain = rng.gamma(shape=2.0, scale=40, size=n_months)
    rain[rng.random(n_months) < 0.1] = 0
    return pd.DataFrame({"code_muni": ["5103403"] * n_months, "date": dates, "rainfall_chirps_mm": rain})


def test_spi_is_standard_normal_over_reference_period(monthly_rainfall_df):
    from LCZ4py.local.lcz_climate_compute_spi import lcz_climate_compute_spi

    out = lcz_climate_compute_spi(monthly_rainfall_df, scales=[1, 3, 12], verbose=False)
    for col in ("spi_1mo", "spi_3mo", "spi_12mo"):
        vals = out[col].dropna()
        assert abs(vals.mean()) < 0.1
        assert abs(vals.std() - 1) < 0.1


def test_spi_missing_column_raises():
    from LCZ4py.local.lcz_climate_compute_spi import lcz_climate_compute_spi

    df = pd.DataFrame({"code_muni": ["a"], "date": pd.to_datetime(["2020-01-01"])})
    with pytest.raises(ValueError):
        lcz_climate_compute_spi(df, verbose=False)


def test_spei_column_pet_is_standard_normal(monthly_rainfall_df):
    from LCZ4py.local.lcz_climate_compute_spei import lcz_climate_compute_spei

    rng = np.random.default_rng(7)
    df = monthly_rainfall_df.copy()
    df["pet_mm"] = rng.gamma(shape=5.0, scale=15, size=len(df))

    out = lcz_climate_compute_spei(df, scales=[1, 3, 12], pet_method="column", verbose=False)
    for col in ("spei_1mo", "spei_3mo", "spei_12mo"):
        vals = out[col].dropna()
        assert abs(vals.mean()) < 0.02
        assert abs(vals.std() - 1) < 0.02  # Hazen ECDF: close to exact by construction


def test_spei_thornthwaite_pet_is_standard_normal(monthly_rainfall_df):
    from LCZ4py.local.lcz_climate_compute_spei import lcz_climate_compute_spei

    n = len(monthly_rainfall_df)
    rng = np.random.default_rng(3)
    df = monthly_rainfall_df.copy()
    df["tair_dry_bulb_c"] = 25 + 5 * np.sin(np.arange(n) / 12 * 2 * np.pi) + rng.normal(0, 1, n)

    out = lcz_climate_compute_spei(df, scales=[3], pet_method="thornthwaite", verbose=False)
    vals = out["spei_3mo"].dropna()
    assert abs(vals.mean()) < 0.02
    assert abs(vals.std() - 1) < 0.02


def test_spei_invalid_pet_method_raises():
    from LCZ4py.local.lcz_climate_compute_spei import lcz_climate_compute_spei

    df = pd.DataFrame({"code_muni": ["a"], "date": pd.to_datetime(["2020-01-01"]), "rainfall_chirps_mm": [10.0]})
    with pytest.raises(ValueError):
        lcz_climate_compute_spei(df, pet_method="bogus", verbose=False)
