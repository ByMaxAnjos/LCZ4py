"""Test suite for lcz4r_python.

Fast unit tests use a synthetic GeoTIFF fixture (no internet).
Network tests hit Nominatim + Zenodo and are opt-in:

    pytest test_lcz4r.py -m "not network"   # fast, offline
    pytest test_lcz4r.py -m network          # integration tests
    pytest test_lcz4r.py::test_name          # single test
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import geopandas as gpd
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box
import polars as pl
import plotly.graph_objects as go


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def lcz_tif(tmp_path):
    """10×10 GeoTIFF with LCZ classes 1–17, EPSG:4326."""
    path = tmp_path / "lcz.tif"
    data = np.tile(np.arange(1, 18, dtype=np.uint8), 6)[:100].reshape(10, 10)
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, 10, 10)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10,
        count=1, dtype="uint8", crs="EPSG:4326",
        transform=transform, nodata=0,
    ) as dst:
        dst.write(data, 1)
    return str(path)


@pytest.fixture
def param_tif(tmp_path):
    """3-band float GeoTIFF simulating a parameter stack (for lcz_plot_parameters)."""
    path = tmp_path / "params.tif"
    data = np.ones((3, 10, 10), dtype=np.float32) * 0.5
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, 10, 10)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10,
        count=3, dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=0,
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "SVFmean")
        dst.set_band_description(2, "AHmean")
        dst.set_band_description(3, "BSFmean")
    return str(path)


# ── lcz_get_map (unit) ────────────────────────────────────────────────────────

def test_lcz_get_map_no_args():
    from LCZ4py.general.lcz_get_map import lcz_get_map
    with pytest.raises(ValueError):
        lcz_get_map()


def test_lcz_clear_cache():
    from LCZ4py.general.lcz_get_map import lcz_clear_cache
    n = lcz_clear_cache()
    assert isinstance(n, int) and n >= 0


# ── lcz_get_map (network) ─────────────────────────────────────────────────────

@pytest.mark.network
def test_lcz_get_map_city():
    from LCZ4py.general.lcz_get_map import lcz_get_map
    path = lcz_get_map(city="Berlin", cache=False)
    assert os.path.exists(path) and path.endswith(".tif")


@pytest.mark.network
def test_lcz_get_map_roi():
    from LCZ4py.general.lcz_get_map import lcz_get_map
    roi = gpd.GeoDataFrame(geometry=[box(13.3, 52.4, 13.5, 52.6)], crs="EPSG:4326")
    path = lcz_get_map(roi=roi, cache=False)
    assert os.path.exists(path) and path.endswith(".tif")


@pytest.mark.network
def test_lcz_get_map_invalid_city():
    from LCZ4py.general.lcz_get_map import lcz_get_map
    with pytest.raises(ValueError):
        lcz_get_map(city="InvalidCityXYZ999")


@pytest.mark.network
def test_lcz_get_map_cache(tmp_path):
    from LCZ4py.general.lcz_get_map import lcz_get_map
    p1 = lcz_get_map(city="Vienna", cache=True, cache_dir=str(tmp_path))
    p2 = lcz_get_map(city="Vienna", cache=True, cache_dir=str(tmp_path))
    assert p1 == p2  # second call is a cache hit


@pytest.mark.network
def test_lcz_get_map_euro():
    from LCZ4py.general.lcz_get_map_euro import lcz_get_map_euro
    path = lcz_get_map_euro(city="Paris", cache=False)
    assert os.path.exists(path)


@pytest.mark.network
def test_lcz_get_map_usa():
    from LCZ4py.general.lcz_get_map_usa import lcz_get_map_usa
    path = lcz_get_map_usa(city="Houston", cache=False)
    assert os.path.exists(path)


@pytest.mark.network
def test_lcz_get_map_generator():
    from LCZ4py.general.lcz_get_map_generator import lcz_get_map_generator
    path = lcz_get_map_generator()
    assert os.path.exists(path)


@pytest.mark.network
def test_stream_cog_window():
    from LCZ4py._internal._lcz_downloader_base import stream_cog_window
    from LCZ4py.general.lcz_get_map import GLOBAL_URL
    arr, profile = stream_cog_window(GLOBAL_URL, (-43.2, -22.9, -43.1, -22.8))
    assert isinstance(arr, np.ndarray) and arr.size > 0
    assert "transform" in profile


# ── lcz_parameters_data ───────────────────────────────────────────────────────

def test_lcz_names():
    from LCZ4py._internal.lcz_parameters_data import LCZ_NAMES, LCZ_COLORS, LCZ_COLORBLIND, LCZ_IDS
    assert len(LCZ_NAMES) == 17
    assert all(c.startswith("#") and len(c) == 7 for c in LCZ_COLORS)
    assert all(c.startswith("#") and len(c) == 7 for c in LCZ_COLORBLIND)
    assert list(LCZ_IDS) == list(range(1, 18))


# ── i18n_messages ─────────────────────────────────────────────────────────────

def test_lcz_msg_en():
    from LCZ4py._internal.i18n_messages import lcz_msg
    msg = lcz_msg("no_map_input", "en")
    assert isinstance(msg, str) and len(msg) > 0


def test_lcz_msg_placeholder():
    from LCZ4py._internal.i18n_messages import lcz_msg
    msg = lcz_msg("city_not_found", "pt", city="Berlin")
    assert "Berlin" in msg


def test_lcz_msg_unknown_lang_falls_back():
    from LCZ4py._internal.i18n_messages import lcz_msg
    msg = lcz_msg("no_map_input", "xx")  # falls back to "en" per docstring
    assert isinstance(msg, str) and len(msg) > 0


# ── adaptive_crop_mask ────────────────────────────────────────────────────────

def test_study_area_km2():
    from LCZ4py._internal.adaptive_crop_mask import study_area_km2
    poly = gpd.GeoDataFrame(geometry=[box(-43.2, -22.9, -43.1, -22.8)], crs="EPSG:4326")
    assert study_area_km2(poly) > 0.0


def test_make_tile_grid():
    from LCZ4py._internal.adaptive_crop_mask import make_tile_grid
    grid = make_tile_grid((-43.2, -22.9, -43.1, -22.8), n_side=3)
    assert isinstance(grid, gpd.GeoDataFrame)
    assert len(grid) == 9
    assert all(grid.geometry.is_valid)


# ── lcz_get_parameters ────────────────────────────────────────────────────────

def test_map_class_to_params_vectorized():
    from LCZ4py.general.lcz_get_parameters import map_class_to_params_vectorized
    arr = np.array([[1, 5, 10], [2, 7, 15]], dtype=np.uint8)
    stack, names = map_class_to_params_vectorized(arr)
    assert stack.shape == (len(names), 2, 3)
    assert stack.dtype == np.float32
    assert len(names) == 35


def test_lcz_get_parameters_stack(lcz_tif):
    from LCZ4py.general.lcz_get_parameters import lcz_get_parameters, LCZStackResult
    result = lcz_get_parameters(lcz_tif, istack=True)
    assert isinstance(result, LCZStackResult)
    assert result.array is not None
    assert result.array.shape[0] == 35


def test_lcz_get_parameters_iselect(lcz_tif):
    from LCZ4py.general.lcz_get_parameters import lcz_get_parameters
    sel = lcz_get_parameters(lcz_tif, iselect=["svf_mean"])
    assert isinstance(sel, np.ndarray)
    assert sel.shape[0] == 1


def test_lcz_get_parameters_dict(lcz_tif):
    from LCZ4py.general.lcz_get_parameters import lcz_get_parameters
    result = lcz_get_parameters(lcz_tif, istack=False)
    assert isinstance(result, dict)
    assert "svf_mean" in result


# ── lcz_cal_area ──────────────────────────────────────────────────────────────

def test_lcz_cal_area_df_only(lcz_tif):
    from LCZ4py.general.lcz_cal_area import lcz_cal_area
    df = lcz_cal_area(lcz_tif, iplot=False)
    assert isinstance(df, pl.DataFrame)
    assert {"lcz", "area_km2", "area_perc"}.issubset(df.columns)
    assert df["area_perc"].sum() == pytest.approx(100.0, abs=0.1)


def test_lcz_cal_area_bar(lcz_tif):
    from LCZ4py.general.lcz_cal_area import lcz_cal_area, LCZAreaResult
    result = lcz_cal_area(lcz_tif, plot_type="bar")
    assert isinstance(result, LCZAreaResult)
    assert isinstance(result.fig, go.Figure)


@pytest.mark.parametrize("plot_type", ["pie", "donut", "sunburst", "treemap"])
def test_lcz_cal_area_plot_types(lcz_tif, plot_type):
    from LCZ4py.general.lcz_cal_area import lcz_cal_area, LCZAreaResult
    result = lcz_cal_area(lcz_tif, plot_type=plot_type)
    assert isinstance(result.fig, go.Figure)


def test_lcz_cal_area_invalid_plot_type(lcz_tif):
    from LCZ4py.general.lcz_cal_area import lcz_cal_area
    with pytest.raises(ValueError):
        lcz_cal_area(lcz_tif, plot_type="invalid")


def test_lcz_cal_area_inclusive(lcz_tif):
    from LCZ4py.general.lcz_cal_area import lcz_cal_area, LCZAreaResult
    result = lcz_cal_area(lcz_tif, inclusive=True)
    assert isinstance(result, LCZAreaResult)


# ── lcz_plot_map ──────────────────────────────────────────────────────────────

def test_lcz_plot_map_default(lcz_tif):
    from LCZ4py.general.lcz_plot_map import lcz_plot_map, LCZPlotResult
    result = lcz_plot_map(lcz_tif)
    assert isinstance(result, LCZPlotResult)
    assert isinstance(result.fig, go.Figure)


def test_lcz_plot_map_no_webgl(lcz_tif):
    from LCZ4py.general.lcz_plot_map import lcz_plot_map, LCZPlotResult
    result = lcz_plot_map(lcz_tif, use_webgl=False)
    assert isinstance(result, LCZPlotResult)


def test_raster_to_geoarrow(lcz_tif):
    from LCZ4py.general.lcz_plot_map import raster_to_geoarrow
    import pyarrow as pa
    with rasterio.open(lcz_tif) as src:
        arr, transform, crs = src.read(1), src.transform, str(src.crs)
    result = raster_to_geoarrow(arr, transform, crs)
    assert result is None or isinstance(result, pa.Table)


# ── lcz_plot_parameters ───────────────────────────────────────────────────────

def test_lcz_plot_parameters_single(param_tif):
    from LCZ4py.general.lcz_plot_parameters import lcz_plot_parameters
    fig = lcz_plot_parameters(param_tif, iselect=["SVFmean"])
    assert isinstance(fig, go.Figure)


def test_lcz_plot_parameters_multi(param_tif):
    from LCZ4py.general.lcz_plot_parameters import lcz_plot_parameters
    figs = lcz_plot_parameters(param_tif, iselect=["SVFmean", "AHmean"])
    assert isinstance(figs, list) and len(figs) == 2
    assert all(isinstance(f, go.Figure) for f in figs)


def test_lcz_plot_parameters_all_params(param_tif):
    from LCZ4py.general.lcz_plot_parameters import lcz_plot_parameters
    # 3-band file → all_params skips band 1, reads bands 2–3 → 2 figures
    figs = lcz_plot_parameters(param_tif, all_params=True)
    assert isinstance(figs, list) and len(figs) == 2


def test_lcz_plot_parameters_iselect_resolves_correct_band(param_tif):
    """iselect must match the requested band by description, not by position."""
    import rasterio
    from LCZ4py.general.lcz_plot_parameters import lcz_plot_parameters

    with rasterio.open(param_tif) as src:
        band3 = src.read(3)

    fig = lcz_plot_parameters(param_tif, iselect="BSFmean")
    z = np.nan_to_num(np.asarray(fig.data[0].z))
    expected = np.nan_to_num(np.flipud(np.where(band3 == 0, np.nan, band3)))
    assert np.allclose(z, expected)


def test_lcz_plot_parameters_has_class_band_false(tmp_path):
    """A stack with no leading class band (e.g. lcz_get_indices output) must
    not have its first band silently skipped by all_params."""
    import rasterio
    from rasterio.transform import from_bounds
    from LCZ4py.general.lcz_plot_parameters import lcz_plot_parameters

    path = tmp_path / "indices.tif"
    data = np.random.uniform(-0.5, 0.5, (2, 10, 10)).astype(np.float32)
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, 10, 10)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10,
        count=2, dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "NDVI")
        dst.set_band_description(2, "NDBI")

    figs = lcz_plot_parameters(str(path), all_params=True, has_class_band=False)
    assert isinstance(figs, list) and len(figs) == 2


def test_lcz_plot_parameters_chart_type_delegates_to_lcz_cal_indices(
    lcz_class_tif_matching, pc_band_stack_tif,
):
    """chart_type='scatter'/'radar'/'correlation' should produce the same
    figures as calling lcz_cal_indices directly — this is a thin delegating
    entry point, not a second implementation."""
    from LCZ4py.general.lcz_get_indices import lcz_get_indices
    from LCZ4py.general.lcz_cal_indices import lcz_cal_indices
    from LCZ4py.general.lcz_plot_parameters import lcz_plot_parameters

    stack_path, _ = pc_band_stack_tif
    idx = lcz_get_indices(stack_path, indices=["NDVI", "NDBI", "MNDWI"], isave=True, verbose=False)

    fig_scatter = lcz_plot_parameters(
        idx.path, lcz_map=lcz_class_tif_matching, chart_type="scatter",
        index_x="NDVI", index_y="NDBI",
    )
    expected_scatter = lcz_cal_indices(
        lcz_class_tif_matching, idx.path, plot_type="scatter", index_x="NDVI", index_y="NDBI",
    ).fig
    assert len(fig_scatter.data) == len(expected_scatter.data)

    fig_radar = lcz_plot_parameters(idx.path, lcz_map=lcz_class_tif_matching, chart_type="radar")
    assert len(fig_radar.data) > 0

    fig_corr = lcz_plot_parameters(
        idx.path, lcz_map=lcz_class_tif_matching, chart_type="correlation", iselect=["NDVI", "NDBI"],
    )
    assert np.array(fig_corr.data[0].z).shape == (2, 2)


def test_lcz_plot_parameters_chart_type_requires_lcz_map(pc_band_stack_tif):
    from LCZ4py.general.lcz_get_indices import lcz_get_indices
    from LCZ4py.general.lcz_plot_parameters import lcz_plot_parameters

    stack_path, _ = pc_band_stack_tif
    idx = lcz_get_indices(stack_path, indices=["NDVI", "NDBI"], isave=True, verbose=False)
    with pytest.raises(ValueError):
        lcz_plot_parameters(idx.path, chart_type="scatter", index_x="NDVI", index_y="NDBI")


def test_lcz_plot_parameters_chart_type_requires_path_input(lcz_class_tif_matching):
    from LCZ4py.general.lcz_plot_parameters import lcz_plot_parameters

    with pytest.raises(ValueError):
        lcz_plot_parameters(
            np.zeros((5, 5), dtype=np.float32), lcz_map=lcz_class_tif_matching, chart_type="radar",
        )


def test_read_with_warped_vrt(lcz_tif):
    from LCZ4py.general.lcz_plot_parameters import read_with_warped_vrt
    arr, profile = read_with_warped_vrt(lcz_tif)
    assert isinstance(arr, np.ndarray)
    assert "transform" in profile


# ── lcz_get_indices / lcz_cal_indices ──────────────────────────────────────────

@pytest.fixture
def pc_band_stack_tif(tmp_path):
    """5-band Sentinel-2-style band stack (B04,B03,B02,B08,B11), EPSG:4326."""
    path = tmp_path / "pc_stack.tif"
    rng = np.random.default_rng(0)
    h, w = 12, 12
    red = rng.uniform(0.05, 0.2, (h, w)).astype(np.float32)
    green = rng.uniform(0.05, 0.2, (h, w)).astype(np.float32)
    blue = rng.uniform(0.02, 0.1, (h, w)).astype(np.float32)
    nir = rng.uniform(0.2, 0.5, (h, w)).astype(np.float32)
    swir1 = rng.uniform(0.1, 0.3, (h, w)).astype(np.float32)
    stack = np.stack([red, green, blue, nir, swir1])
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=5, dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(stack)
        dst.update_tags(collection="sentinel-2-l2a")
        for i, name in enumerate(["B04", "B03", "B02", "B08", "B11"], start=1):
            dst.set_band_description(i, name)
    return str(path), dict(red=red, green=green, blue=blue, nir=nir, swir1=swir1)


@pytest.fixture
def lcz_class_tif_matching(tmp_path, pc_band_stack_tif):
    """LCZ class map on the same grid as pc_band_stack_tif."""
    path = tmp_path / "lcz_match.tif"
    h, w = 12, 12
    rng = np.random.default_rng(1)
    classes = rng.choice([1, 2, 11, 14, 17], size=(h, w)).astype(np.uint8)
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="uint8", crs="EPSG:4326",
        transform=transform, nodata=0,
    ) as dst:
        dst.write(classes, 1)
    return str(path)


def test_lcz_get_indices_default_selection(pc_band_stack_tif):
    from LCZ4py.general.lcz_get_indices import lcz_get_indices, INDEX_FORMULAS

    path, bands = pc_band_stack_tif
    result = lcz_get_indices(path, verbose=False)

    # Every formula computable from red/green/blue/nir/swir1 alone (this
    # fixture has no swir2 or thermal) should be auto-selected.
    available_roles = {"red", "green", "blue", "nir", "swir1"}
    expected_names = {
        name for name, (reqs, _fn) in INDEX_FORMULAS.items()
        if set(reqs) <= available_roles
    }
    assert set(result.indices) == expected_names
    assert result.array.shape == (len(expected_names), 12, 12)
    assert len(expected_names) > 8  # sanity check the expanded registry loaded

    ndvi = result.array[result.indices.index("NDVI")]
    expected = (bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"])
    assert np.allclose(ndvi, expected, atol=1e-5)


def test_lcz_get_indices_missing_bands_raises(tmp_path):
    from LCZ4py.general.lcz_get_indices import lcz_get_indices

    # Only red + nir present — NDBI needs swir1, which is missing here.
    path = tmp_path / "minimal_stack.tif"
    data = np.random.uniform(0.05, 0.3, (2, 8, 8)).astype(np.float32)
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, 8, 8)
    with rasterio.open(
        path, "w", driver="GTiff", height=8, width=8,
        count=2, dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "B04")
        dst.set_band_description(2, "B08")

    with pytest.raises(ValueError):
        lcz_get_indices(str(path), indices=["NDBI"], verbose=False)


def test_lcz_get_indices_unknown_index_raises(pc_band_stack_tif):
    from LCZ4py.general.lcz_get_indices import lcz_get_indices

    path, _ = pc_band_stack_tif
    with pytest.raises(ValueError):
        lcz_get_indices(path, indices=["NOT_A_REAL_INDEX"], verbose=False)


def test_index_formulas_registry_covers_all_categories():
    """Guards against silently losing coverage across the ~75-formula source
    registry (vegetation/water/urban/soil/thermal) in a future refactor."""
    from LCZ4py.general.lcz_get_indices import INDEX_FORMULAS

    assert len(INDEX_FORMULAS) >= 50
    for name in ("NDVI", "EVI", "MSAVI", "OSAVI"):  # vegetation
        assert name in INDEX_FORMULAS
    for name in ("NDWI", "MNDWI", "AWEI_NSH", "NDTI"):  # water
        assert name in INDEX_FORMULAS
    for name in ("NDBI", "IBI", "UI", "ISA"):  # urban
        assert name in INDEX_FORMULAS
    for name in ("BSI", "SI", "RI", "IRON_OXIDE"):  # soil
        assert name in INDEX_FORMULAS
    for name in ("LST_K", "LST_C"):  # thermal
        assert name in INDEX_FORMULAS


def test_lcz_get_indices_thermal_only_for_landsat(tmp_path):
    """LST_K/LST_C need the 'thermal' role, which only Landsat's lwir11
    asset maps to — a Sentinel-2 stack (no thermal band) must not offer it,
    while a Landsat stack with lwir11 must compute it correctly in Kelvin."""
    from LCZ4py.general.lcz_get_indices import lcz_get_indices

    h, w = 8, 8
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, w, h)

    # Sentinel-2: no thermal band available at all.
    s2_path = tmp_path / "s2_stack.tif"
    s2_data = np.random.uniform(1000, 3000, (4, h, w)).astype(np.float32)
    with rasterio.open(
        s2_path, "w", driver="GTiff", height=h, width=w, count=4,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(s2_data)
        dst.update_tags(collection="sentinel-2-l2a")
        for i, name in enumerate(["B04", "B03", "B02", "B08"], start=1):
            dst.set_band_description(i, name)
    s2_result = lcz_get_indices(s2_path, verbose=False)
    assert "LST_K" not in s2_result.indices
    assert "LST_C" not in s2_result.indices

    # Landsat: lwir11 present, official DN -> Kelvin scale (0.00341802, 149.0).
    ls_path = tmp_path / "landsat_stack.tif"
    thermal_dn = np.full((h, w), 40000.0, dtype=np.float32)  # -> ~285.7 K
    red_dn = np.full((h, w), 9000.0, dtype=np.float32)
    with rasterio.open(
        ls_path, "w", driver="GTiff", height=h, width=w, count=2,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(np.stack([red_dn, thermal_dn]))
        dst.update_tags(collection="landsat-c2-l2")
        dst.set_band_description(1, "red")
        dst.set_band_description(2, "lwir11")
    ls_result = lcz_get_indices(ls_path, indices=["LST_K", "LST_C"], verbose=False)

    expected_k = 40000.0 * 0.00341802 + 149.0
    assert np.allclose(ls_result.array[ls_result.indices.index("LST_K")], expected_k, atol=1e-3)
    assert np.allclose(
        ls_result.array[ls_result.indices.index("LST_C")], expected_k - 273.15, atol=1e-3,
    )


def test_lcz_get_indices_negative_reflectance_masked(tmp_path):
    """A pixel with atmospheric-correction-artifact negative reflectance in
    one band must become NaN in that band's ratio indices, not an extreme
    blown-up value (verified against real Rio de Janeiro Landsat data,
    where this exact failure mode inflated RI's max from 5656 to 362187)."""
    from LCZ4py.general.lcz_get_indices import lcz_get_indices

    h, w = 4, 4
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, w, h)
    # DN chosen so landsat scale (0.0000275, -0.2) makes blue reflectance
    # slightly negative at pixel (0, 0) only.
    red_dn = np.full((h, w), 9000.0, dtype=np.float32)
    blue_dn = np.full((h, w), 9000.0, dtype=np.float32)
    blue_dn[0, 0] = 100.0  # -> reflectance = 100*0.0000275 - 0.2 < 0

    path = tmp_path / "neg_refl.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=2,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(np.stack([red_dn, blue_dn]))
        dst.update_tags(collection="landsat-c2-l2")
        dst.set_band_description(1, "red")
        dst.set_band_description(2, "blue")

    result = lcz_get_indices(path, indices=["IRON_OXIDE"], verbose=False)
    arr = result.array[0]
    assert np.isnan(arr[0, 0])  # masked, not a huge finite ratio
    assert np.all(np.isfinite(arr[1:, 1:]))  # unaffected pixels stay normal


def test_lcz_cal_indices_stats_and_charts(lcz_class_tif_matching, pc_band_stack_tif):
    from LCZ4py.general.lcz_get_indices import lcz_get_indices
    from LCZ4py.general.lcz_cal_indices import lcz_cal_indices

    stack_path, _ = pc_band_stack_tif
    idx = lcz_get_indices(stack_path, indices=["NDVI", "NDBI"], verbose=False)

    stats = lcz_cal_indices(lcz_class_tif_matching, idx, plot_type="both")
    assert isinstance(stats.df, pl.DataFrame)
    assert set(stats.df["parameter"].unique().to_list()) == {"NDVI", "NDBI"}
    for col in ("mean", "median", "std", "ci_lower", "ci_upper", "lcz_name"):
        assert col in stats.df.columns

    assert isinstance(stats.fig, dict) and set(stats.fig) == {"box", "bar"}
    assert len(stats.fig["bar"].data) == 2  # one trace per parameter subplot


def test_lcz_cal_indices_shape_mismatch_raises(pc_band_stack_tif, tmp_path):
    from LCZ4py.general.lcz_get_indices import lcz_get_indices
    from LCZ4py.general.lcz_cal_indices import lcz_cal_indices

    stack_path, _ = pc_band_stack_tif
    idx = lcz_get_indices(stack_path, indices=["NDVI"], verbose=False)

    mismatched = tmp_path / "small_lcz.tif"
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, 5, 5)
    with rasterio.open(
        mismatched, "w", driver="GTiff", height=5, width=5,
        count=1, dtype="uint8", crs="EPSG:4326",
        transform=transform, nodata=0,
    ) as dst:
        dst.write(np.ones((5, 5), dtype=np.uint8), 1)

    with pytest.raises(ValueError):
        lcz_cal_indices(str(mismatched), idx)


def test_lcz_cal_indices_type_column(lcz_class_tif_matching, pc_band_stack_tif):
    from LCZ4py.general.lcz_get_indices import lcz_get_indices
    from LCZ4py.general.lcz_cal_indices import lcz_cal_indices

    stack_path, _ = pc_band_stack_tif
    idx = lcz_get_indices(stack_path, indices=["NDVI"], verbose=False)
    df = lcz_cal_indices(lcz_class_tif_matching, idx, iplot=False)

    assert "type" in df.columns
    types = dict(zip(df["lcz"].to_list(), df["type"].to_list()))
    for lcz, t in types.items():
        assert t == ("Built-up" if lcz <= 10 else "Natural")


def test_lcz_cal_indices_cohens_d_known_separation(tmp_path):
    """Two groups with a known mean/std separation should yield a Cohen's d
    close to the analytically expected value, not a fabricated one (unlike
    lcz_multi_app's source app, whose Cohen's d is np.random.uniform(-2, 2))."""
    from LCZ4py.general.lcz_cal_indices import lcz_cal_indices

    h, w = 20, 20
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, w, h)
    classes = np.empty((h, w), dtype=np.uint8)
    classes[: h // 2, :] = 3    # urban (LCZ 1-10)
    classes[h // 2:, :] = 14    # natural (LCZ 11-17)

    lcz_path = tmp_path / "lcz_cohens.tif"
    with rasterio.open(
        lcz_path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0,
    ) as dst:
        dst.write(classes, 1)

    rng = np.random.default_rng(0)
    values = np.empty((h, w), dtype=np.float32)
    values[: h // 2, :] = rng.normal(5.0, 1.0, (h // 2, w))
    values[h // 2:, :] = rng.normal(0.0, 1.0, (h - h // 2, w))

    idx_path = tmp_path / "idx_cohens.tif"
    with rasterio.open(
        idx_path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(values, 1)
        dst.set_band_description(1, "SYNTHETIC")

    stats = lcz_cal_indices(str(lcz_path), str(idx_path), plot_type="effect_size")
    row = stats.cohens_d.filter(pl.col("parameter") == "SYNTHETIC").to_dicts()[0]
    assert row["magnitude"] == "Large"
    assert 3.5 < row["cohens_d"] < 6.5  # true separation mean_diff=5, std=1 -> d~5
    assert len(stats.fig.data) == 1


def test_lcz_cal_indices_new_plot_types_smoke(lcz_class_tif_matching, pc_band_stack_tif):
    """One smoke test per new plot_type — real Rio-scale validation for
    correctness already ran manually; this just guards against regressions."""
    from LCZ4py.general.lcz_get_indices import lcz_get_indices
    from LCZ4py.general.lcz_cal_indices import lcz_cal_indices

    stack_path, _ = pc_band_stack_tif
    idx = lcz_get_indices(stack_path, indices=["NDVI", "NDBI", "MNDWI"], verbose=False)

    scatter = lcz_cal_indices(
        lcz_class_tif_matching, idx, plot_type="scatter",
        index_x="NDVI", index_y="NDBI", size_by="MNDWI",
    )
    assert len(scatter.fig.data) > 0

    radar = lcz_cal_indices(lcz_class_tif_matching, idx, plot_type="radar")
    assert len(radar.fig.data) > 0

    corr = lcz_cal_indices(lcz_class_tif_matching, idx, plot_type="correlation")
    assert np.array(corr.fig.data[0].z).shape == (3, 3)

    combined = lcz_cal_indices(
        lcz_class_tif_matching, idx, plot_type="all", index_x="NDVI", index_y="NDBI",
    )
    assert set(combined.fig) == {"box", "bar", "radar", "correlation", "effect_size", "scatter"}
    assert combined.cohens_d is not None


def test_lcz_cal_indices_scatter_requires_index_x_y(lcz_class_tif_matching, pc_band_stack_tif):
    from LCZ4py.general.lcz_get_indices import lcz_get_indices
    from LCZ4py.general.lcz_cal_indices import lcz_cal_indices

    stack_path, _ = pc_band_stack_tif
    idx = lcz_get_indices(stack_path, indices=["NDVI"], verbose=False)
    with pytest.raises(ValueError):
        lcz_cal_indices(lcz_class_tif_matching, idx, plot_type="scatter")


# ── lcz_uhi_surface ──────────────────────────────────────────────────────────

@pytest.fixture
def lst_stack_tif(tmp_path):
    """6-day LST stack (Kelvin, date-string band descriptions), on the same
    12x12 grid as pc_band_stack_tif/lcz_class_tif_matching — mirrors
    lcz_get_lst's _write_lst_stack output convention."""
    path = tmp_path / "lst_stack.tif"
    h, w = 12, 12
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]
    rng = np.random.default_rng(3)
    stack = rng.uniform(295, 305, (len(dates), h, w)).astype(np.float32)
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=len(dates), dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(stack)
        dst.update_tags(units="K", variable="LST")
        for i, d in enumerate(dates, start=1):
            dst.set_band_description(i, d)
    return str(path), dates, stack


def test_lcz_uhi_surface_urban_rural_known_separation(tmp_path):
    """Urban pixels set 5K warmer than rural on every date should give a
    SUHI intensity close to 5, not a fabricated/placeholder number."""
    import rasterio
    from rasterio.transform import from_bounds
    from LCZ4py.local.lcz_uhi_surface import lcz_uhi_surface

    h, w = 10, 10
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, w, h)
    classes = np.empty((h, w), dtype=np.uint8)
    classes[: h // 2, :] = 3    # urban
    classes[h // 2:, :] = 14    # rural reference (also RURAL_CLASSES)

    lcz_path = tmp_path / "lcz.tif"
    with rasterio.open(
        lcz_path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0,
    ) as dst:
        dst.write(classes, 1)

    dates = ["2024-06-01", "2024-06-02"]
    lst = np.empty((2, h, w), dtype=np.float32)
    for i in range(2):
        lst[i, : h // 2, :] = 305.0
        lst[i, h // 2:, :] = 300.0

    lst_path = tmp_path / "lst.tif"
    with rasterio.open(
        lst_path, "w", driver="GTiff", height=h, width=w, count=2,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(lst)
        dst.update_tags(units="K", variable="LST")
        for i, d in enumerate(dates, start=1):
            dst.set_band_description(i, d)

    result = lcz_uhi_surface(str(lcz_path), str(lst_path), method="urban_rural")
    assert np.allclose(result.df["suhi"].to_numpy(), 5.0, atol=1e-4)


def test_lcz_uhi_surface_lcz_method_reference_is_zero(lcz_class_tif_matching, lst_stack_tif):
    from LCZ4py.local.lcz_uhi_surface import lcz_uhi_surface, REFERENCE_LCZ

    lst_path, _, _ = lst_stack_tif
    result = lcz_uhi_surface(lcz_class_tif_matching, lst_path, method="lcz", iplot=False)
    ref_rows = result.filter(pl.col("lcz") == REFERENCE_LCZ)
    assert np.allclose(ref_rows["delta_t"].to_numpy(), 0.0, atol=1e-4)


def test_lcz_uhi_surface_utfvi_kelvin_conversion(lcz_class_tif_matching, lst_stack_tif):
    """Celsius input must be converted to Kelvin before the UTFVI ratio —
    otherwise the ratio (and its ecological classification) would be wrong
    by roughly a factor of 273/T (see module docstring)."""
    from LCZ4py.local.lcz_uhi_surface import lcz_uhi_surface

    lst_path, _, stack_k = lst_stack_tif
    result_k = lcz_uhi_surface(lcz_class_tif_matching, lst_path, method="utfvi")

    # Same physical scene, but the file says Celsius -> must be converted
    # back to Kelvin internally, giving the identical UTFVI array.
    import rasterio
    lst_c_path = lst_path.replace("lst_stack.tif", "lst_stack_c.tif")
    with rasterio.open(lst_path) as src:
        profile = src.profile.copy()
        descriptions = list(src.descriptions)
    with rasterio.open(lst_c_path, "w", **profile) as dst:
        dst.write(stack_k - 273.15)
        dst.update_tags(units="C", variable="LST")
        for i, d in enumerate(descriptions, start=1):
            dst.set_band_description(i, d)

    result_c = lcz_uhi_surface(lcz_class_tif_matching, lst_c_path, method="utfvi")
    assert np.allclose(result_k.array, result_c.array, equal_nan=True, atol=1e-4)


def test_lcz_uhi_surface_hotspot_detects_known_cluster(tmp_path):
    """A deliberately hot 3x3 block in an otherwise flat raster must be
    picked up as a significant Gi* hot spot."""
    import rasterio
    from rasterio.transform import from_bounds
    from LCZ4py.local.lcz_uhi_surface import lcz_uhi_surface

    h, w = 20, 20
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, w, h)
    classes = np.full((h, w), 3, dtype=np.uint8)

    lcz_path = tmp_path / "lcz_hot.tif"
    with rasterio.open(
        lcz_path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0,
    ) as dst:
        dst.write(classes, 1)

    rng = np.random.default_rng(4)
    lst = rng.normal(300.0, 0.5, (1, h, w)).astype(np.float32)
    lst[0, 8:11, 8:11] = 320.0  # a sharp, spatially contiguous hot block

    lst_path = tmp_path / "lst_hot.tif"
    with rasterio.open(
        lst_path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(lst)
        dst.update_tags(units="K", variable="LST")
        dst.set_band_description(1, "2024-06-01")

    result = lcz_uhi_surface(str(lcz_path), str(lst_path), method="hotspot", hotspot_window=3)
    assert result.array[9, 9] > 1.96  # center of the hot block: significant hot spot
    assert "Hot spot" in result.df["category"].to_list()


def test_lcz_uhi_surface_persistence_bounds(lcz_class_tif_matching, lst_stack_tif):
    from LCZ4py.local.lcz_uhi_surface import lcz_uhi_surface

    lst_path, _, _ = lst_stack_tif
    result = lcz_uhi_surface(lcz_class_tif_matching, lst_path, method="persistence")
    valid = ~np.isnan(result.array)
    assert (result.array[valid] >= 0).all() and (result.array[valid] <= 100).all()
    assert set(result.fig.keys()) == {"map", "bar"}


def test_lcz_uhi_surface_all_methods_and_iplot_false(lcz_class_tif_matching, lst_stack_tif):
    from LCZ4py.local.lcz_uhi_surface import lcz_uhi_surface

    lst_path, _, _ = lst_stack_tif
    all_results = lcz_uhi_surface(lcz_class_tif_matching, lst_path, method="all")
    assert set(all_results) == {
        "urban_rural", "percentile", "lcz", "utfvi", "hotspot", "transect", "persistence",
    }

    all_dfs = lcz_uhi_surface(lcz_class_tif_matching, lst_path, method="all", iplot=False)
    assert all(isinstance(df, pl.DataFrame) for df in all_dfs.values())

    single_df = lcz_uhi_surface(lcz_class_tif_matching, lst_path, method="urban_rural", iplot=False)
    assert isinstance(single_df, pl.DataFrame)


def test_lcz_uhi_surface_unknown_method_raises(lcz_class_tif_matching, lst_stack_tif):
    from LCZ4py.local.lcz_uhi_surface import lcz_uhi_surface

    lst_path, _, _ = lst_stack_tif
    with pytest.raises(ValueError):
        lcz_uhi_surface(lcz_class_tif_matching, lst_path, method="bogus")


def test_lcz_uhi_surface_shape_mismatch_raises(tmp_path, lst_stack_tif):
    from LCZ4py.local.lcz_uhi_surface import lcz_uhi_surface

    lst_path, _, _ = lst_stack_tif
    mismatched = tmp_path / "small_lcz.tif"
    transform = from_bounds(-43.2, -22.9, -43.1, -22.8, 5, 5)
    with rasterio.open(
        mismatched, "w", driver="GTiff", height=5, width=5,
        count=1, dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0,
    ) as dst:
        dst.write(np.ones((5, 5), dtype=np.uint8), 1)

    with pytest.raises(ValueError):
        lcz_uhi_surface(str(mismatched), lst_path, method="urban_rural")
