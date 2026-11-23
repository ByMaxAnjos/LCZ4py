"""Offline smoke tests for the core, network-free entry points.

Everything here works on a tiny synthetic LCZ raster created in ``tmp_path``;
no HTTP, no Planetary Computer, no CDS.  Optional accelerators (duckdb,
geoarrow, datashader, exactextract) are explicitly disabled so the tests behave
the same on a minimal install as on a fully loaded one.

Covered:
    * ``lcz_get_lst._load_target_grid`` / ``_daterange`` / ``_cache_key``
    * ``lcz_cal_area`` (both dataframe-only and interactive chart paths)
    * ``lcz_get_parameters`` / ``map_class_to_params_vectorized``
    * ``lcz_plot_map`` (raster input) and ``lcz_plot_parameters`` (array input)

Run with::

    python -m pytest tests/test_offline_smoke.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

SIZE = 64
N_CLASSES = 17
PIXEL_SIZE = 30.0  # metres, EPSG:3857
# Arbitrary Berlin-ish origin in Web Mercator so the WGS84 bbox is realistic.
ORIGIN_X = 1_489_000.0
ORIGIN_Y = 6_894_000.0


@pytest.fixture(scope="module")
def synthetic_lcz(tmp_path_factory: pytest.TempPathFactory) -> str:
    """64x64 uint8 LCZ class raster in EPSG:3857 containing classes 1..17."""
    directory = tmp_path_factory.mktemp("lcz_smoke")
    path = directory / "synthetic_lcz.tif"

    # Cycle 1..17 so every class appears at least once, then a ragged tail so
    # the class histogram is not perfectly uniform.
    flat = (np.arange(SIZE * SIZE, dtype=np.int32) % N_CLASSES) + 1
    flat[-SIZE:] = 0  # one nodata row, to exercise the nodata path too
    classes = flat.reshape(SIZE, SIZE).astype("uint8")

    profile = {
        "driver": "GTiff",
        "height": SIZE,
        "width": SIZE,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": from_origin(ORIGIN_X, ORIGIN_Y, PIXEL_SIZE, PIXEL_SIZE),
        "nodata": 0,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(classes, 1)

    return str(path)


@pytest.fixture(scope="module")
def synthetic_classes(synthetic_lcz: str) -> np.ndarray:
    with rasterio.open(synthetic_lcz) as src:
        return src.read(1)


# ---------------------------------------------------------------------------
# Fixture sanity
# ---------------------------------------------------------------------------

def test_synthetic_raster_is_valid(synthetic_lcz: str, synthetic_classes: np.ndarray) -> None:
    with rasterio.open(synthetic_lcz) as src:
        assert (src.height, src.width) == (SIZE, SIZE)
        assert src.crs.to_epsg() == 3857
        assert src.nodata == 0

    unique = set(np.unique(synthetic_classes).tolist())
    assert set(range(1, N_CLASSES + 1)).issubset(unique)
    assert 0 in unique, "nodata row missing from fixture"


# ---------------------------------------------------------------------------
# lcz_get_lst helpers (pure, no I/O beyond the local raster)
# ---------------------------------------------------------------------------

def test_load_target_grid(synthetic_lcz: str) -> None:
    from LCZ4py.general.lcz_get_lst import _load_target_grid

    grid = _load_target_grid(synthetic_lcz)

    assert grid.shape == (SIZE, SIZE)
    assert grid.crs.to_epsg() == 3857
    assert grid.valid_mask.shape == (SIZE, SIZE)
    assert grid.valid_mask.dtype == bool
    # Only the nodata row is invalid.
    assert grid.valid_mask.sum() == SIZE * (SIZE - 1)

    left, bottom, right, top = grid.bbox_wgs84
    assert left < right and bottom < top
    assert -180.0 <= left <= 180.0 and -180.0 <= right <= 180.0
    assert -90.0 <= bottom <= 90.0 and -90.0 <= top <= 90.0
    # Berlin-ish in Web Mercator.
    assert 12.0 < left < 14.0
    assert 51.5 < bottom < 53.5


def test_daterange() -> None:
    from LCZ4py.general.lcz_get_lst import _daterange

    days = _daterange("2024-03-01", "2024-03-04")
    assert len(days) == 4
    assert [d.isoformat() for d in days] == [
        "2024-03-01",
        "2024-03-02",
        "2024-03-03",
        "2024-03-04",
    ]
    assert _daterange("2024-03-01", "2024-03-01") == days[:1]

    with pytest.raises(ValueError):
        _daterange("2024-03-04", "2024-03-01")


def test_cache_key(synthetic_lcz: str) -> None:
    from LCZ4py.general.lcz_get_lst import _cache_key, _load_target_grid

    grid = _load_target_grid(synthetic_lcz)
    key = _cache_key("sentinel3", "slstr", grid, "2024-03-01", "2024-03-04", 13.0)

    assert isinstance(key, str) and len(key) == 16
    assert all(c in "0123456789abcdef" for c in key)

    # Deterministic for identical inputs...
    same = _cache_key("sentinel3", "slstr", grid, "2024-03-01", "2024-03-04", 13.0)
    assert key == same

    # ...and sensitive to every part of the key.
    assert key != _cache_key("goes", "slstr", grid, "2024-03-01", "2024-03-04", 13.0)
    assert key != _cache_key("sentinel3", "slstr", grid, "2024-03-02", "2024-03-04", 13.0)
    assert key != _cache_key("sentinel3", "slstr", grid, "2024-03-01", "2024-03-04", 14.0)


# ---------------------------------------------------------------------------
# lcz_cal_area
# ---------------------------------------------------------------------------

def test_cal_area_dataframe_only(synthetic_lcz: str) -> None:
    from LCZ4py.general.lcz_cal_area import lcz_cal_area

    result = lcz_cal_area(
        synthetic_lcz,
        iplot=False,
        use_duckdb=False,
        use_geoarrow=False,
    )

    assert hasattr(result, "columns"), "expected a polars DataFrame with iplot=False"
    assert {"lcz", "count", "area_km2", "area_perc"} .issubset(set(result.columns))
    assert result.height == N_CLASSES
    assert result["lcz"].to_list() == list(range(1, N_CLASSES + 1))

    valid_pixels = int((np.asarray(result["count"].to_list())).sum())
    assert valid_pixels == SIZE * (SIZE - 1)  # every non-nodata pixel counted

    area_km2 = float(np.asarray(result["area_km2"].to_list()).sum())
    assert area_km2 > 0.0

    perc = float(np.asarray(result["area_perc"].to_list()).sum())
    assert perc == pytest.approx(100.0, abs=0.5)


def test_cal_area_returns_figure(synthetic_lcz: str) -> None:
    go = pytest.importorskip("plotly.graph_objects")

    from LCZ4py.general.lcz_cal_area import LCZAreaResult, lcz_cal_area

    result = lcz_cal_area(
        synthetic_lcz,
        plot_type="bar",
        iplot=True,
        use_duckdb=False,
        use_geoarrow=False,
    )

    assert isinstance(result, LCZAreaResult)
    assert result.df.height == N_CLASSES
    assert result.renderer == "static"
    import matplotlib.figure
    assert isinstance(result.fig, matplotlib.figure.Figure)
    assert len(result.fig.axes) > 0


# ---------------------------------------------------------------------------
# lcz_get_parameters
# ---------------------------------------------------------------------------

def test_get_parameters_stack(synthetic_lcz: str, synthetic_classes: np.ndarray) -> None:
    from LCZ4py.general.lcz_get_parameters import (
        LCZStackResult,
        lcz_get_parameters,
        parameter_names,
    )

    result = lcz_get_parameters(
        synthetic_lcz,
        use_duckdb=False,
        use_geoarrow=False,
    )

    assert isinstance(result, LCZStackResult)
    assert result.array is not None, "istack=True must populate .array"
    assert result.array.ndim == 3
    assert result.array.shape[1:] == (SIZE, SIZE)

    names = parameter_names()
    assert result.array.shape[0] == len(names)
    assert names[0] == "lcz"
    assert len(names) >= 34

    # Class band round-trips exactly.
    np.testing.assert_array_equal(
        result.array[0].astype("uint8"), synthetic_classes
    )


def test_map_class_to_params_vectorized(synthetic_classes: np.ndarray) -> None:
    from LCZ4py.general.lcz_get_parameters import (
        map_class_to_params_vectorized,
        parameter_names,
    )

    stack, names = map_class_to_params_vectorized(synthetic_classes)
    assert stack.shape == (len(names), SIZE, SIZE)
    assert names == parameter_names()
    assert stack.dtype == np.float32

    # Nodata pixels must stay zero across every parameter band.
    nodata_rows = stack[:, -1, :]
    assert np.all(nodata_rows == 0.0)


# ---------------------------------------------------------------------------
# Plotting constructors
# ---------------------------------------------------------------------------

def test_plot_map_from_raster(synthetic_lcz: str) -> None:
    go = pytest.importorskip("plotly.graph_objects")
    pytest.importorskip("plotly")

    from LCZ4py.general.lcz_plot_map import LCZPlotResult, lcz_plot_map

    result = lcz_plot_map(
        synthetic_lcz,
        renderer="plotly",
        use_duckdb=False,
        use_geoarrow=False,
        add_scalebar=False,
        add_north_arrow=False,
    )

    assert isinstance(result, LCZPlotResult)
    assert result.renderer == "plotly"
    assert isinstance(result.fig, go.Figure)
    assert len(result.fig.data) > 0


def test_plot_map_static_is_default_and_browser_free(
    synthetic_lcz: str, tmp_path, monkeypatch
) -> None:
    import importlib

    import matplotlib.figure

    plot_module = importlib.import_module("LCZ4py.general.lcz_plot_map")

    # Prove the static path never asks Kaleido/Chromium for a screenshot.
    def _boom(*args, **kwargs):
        raise AssertionError("static renderer must not call save_plotly_figure")

    monkeypatch.setattr(plot_module, "save_plotly_figure", _boom)
    monkeypatch.chdir(tmp_path)

    result = plot_module.lcz_plot_map(synthetic_lcz, isave=True, style="nature")

    assert result.renderer == "static"
    assert isinstance(result.fig, matplotlib.figure.Figure)
    out = tmp_path / "LCZ4r_output" / "lcz_plot_map.png"
    assert out.exists() and out.stat().st_size > 1000


def test_plot_parameters_from_array() -> None:
    go = pytest.importorskip("plotly.graph_objects")

    from LCZ4py.general.lcz_plot_parameters import lcz_plot_parameters

    rows, cols = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    param = (rows + cols).astype("float32")

    fig = lcz_plot_parameters(
        param,
        has_class_band=False,
        renderer="plotly",
        use_datashader=False,
        add_scalebar=False,
        add_north_arrow=False,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_meta_tree_quadkeys() -> None:
    from LCZ4py.general.lcz_get_meta_tree import _quadkeys

    # Tile 1202120103 covers lon 13.008-13.359, lat 52.052-52.268 (from tiles.geojson).
    assert _quadkeys(13.1, 52.1, 13.2, 52.2) == ["1202120103"]
    assert len(_quadkeys(13.0, 52.0, 13.4, 52.3)) == 9  # 3x3 tiles
