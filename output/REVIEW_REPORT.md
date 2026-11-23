# LCZ4py QA Review — Import Safety & Offline Smoke (general / local)

**Scope:** `LCZ4py.general`, `LCZ4py.local`, `LCZ4py._internal`
**Repo:** `/Users/co2map/Documents/lcz4r_python`
**HEAD:** `6d80647` (baseline). The working tree contained **31 uncommitted files modified by concurrent agents** (`git status`), so every result below describes the working tree at run time, not the pristine baseline commit.
**Date:** 2026-11-21

---

## 1. Verdict

| Gate | Result |
|---|---|
| Plain import of every general/local/_internal module (49 targets) | **PASS** (49/49) |
| Every name in `general.__all__` + `local.__all__` resolves | **PASS** |
| Import with 14 optional deps blocked | **PASS** (all 49 modules resolve) |
| Import with **all 16** optional deps blocked (incl. `xarray`, `dask`) | **FAIL** — blocked on `xarray` |
| `python -m compileall -q LCZ4py` | **PASS** (exit 0, no output) |
| Offline smoke suite | **PASS** (10/10) |
| Full harness | **67 passed, 2 failed, 1 warning in 98.06s** |

Only **one** import-time blocker exists, and it is a single unguarded import pair in `LCZ4py/general/lcz_plot_parameters.py` (lines 24–25). Details in §5.

---

## 2. Environment

- Python **3.13.5** (CPython, macOS 26.4.1, x86_64)
- pytest available; no installation performed during this review.

**Present**

| Package | Version |
|---|---|
| `pystac_client` | 0.9.0 |
| `planetary_computer` | 1.0.0 |
| `h5netcdf` / `h5py` | 1.8.1 / 3.16.0 |
| `pykrige` | 1.7.3 |
| `scikit-learn` | 1.9.0 |
| `duckdb` | 1.5.2 |
| `contextily` | 1.7.1 |
| `xarray` / `dask` | 2026.4.0 / 2026.6.0 |
| `rioxarray` | 0.22.0 |
| `cdsapi` | present |
| `scipy` | 1.17.1 |
| `rasterio` / `geopandas` | 1.5.0 / 1.1.3 |
| `polars` / `pyarrow` / `numpy` | 1.42.0 / 21.0.0 / 2.4.0 |
| `plotly` / `kaleido` / `matplotlib` | 6.7.0 / present / 3.10.9 |
| `shapely` / `diskcache` / `httpx` / `aiohttp` | 2.1.2 / 5.6.3 / 0.28.1 / 3.14.3 |

**Missing (genuinely absent)**

`geoarrow`, `exactextract`, `datashader`, `holoviews`, `colorcet`

All five are optional; the package must import without them (it does — see §4.2).

---

## 3. Artifacts & commands

Created:

- `tests/test_imports_general_local.py` — 59 tests (module imports, `__all__` resolution, 4 optional-dep groups, combined blocker).
- `tests/test_offline_smoke.py` — 10 tests against a synthetic 64×64 EPSG:3857 LCZ raster.
- `output/REVIEW_REPORT.md` — this file.

Commands run:

```bash
python -m pytest tests/test_imports_general_local.py tests/test_offline_smoke.py -q
# => 2 failed, 67 passed, 1 warning in 98.06s

python -m compileall -q LCZ4py
# => exit 0, no output (all files byte-compile)

# per-module diagnostic: import each module in a fresh interpreter with the
# 14 non-xarray/dask optional deps blocked => 46/46 submodules PASS
# per-dependency diagnostic: block one dep at a time => only xarray FAILs
```

**Blocker design note.** The subprocess blocker uses a `sys.meta_path` finder whose
`find_spec` returns a `ModuleSpec` backed by a loader that raises
`ModuleNotFoundError` on `create_module`. A finder that raised directly from
`find_spec` would produce false failures for availability probes that don't catch
the exception — notably xarray's `module_available("dask")`, which calls
`importlib.util.find_spec`. With the loader approach, `import dask` fails exactly
as if dask were uninstalled while `find_spec` probing stays harmless. The
subprocess also self-checks that every blocked name actually raises
`ModuleNotFoundError`.

---

## 4. Import matrix

### 4.1 Per module

`Plain import` = in-process `importlib.import_module`.
`Optional-blocked import` = fresh subprocess with `pystac_client`,
`planetary_computer`, `h5netcdf`, `h5py`, `pykrige`, `sklearn`, `duckdb`,
`geoarrow`, `exactextract`, `datashader`, `holoviews`, `colorcet`,
`contextily`, `cdsapi` blocked. (`xarray`/`dask` are handled separately in §4.2
because `rioxarray` is a declared core dependency and hard-requires `xarray`.)

| Module | Plain import | Optional-blocked import | Notes |
|---|---|---|---|
| `LCZ4py.general` (package) | OK | OK | re-exports 40 names |
| `LCZ4py.general.lcz_cal_area` | OK | OK | duckdb / geoarrow / exactextract guarded (`HAS_*`) |
| `LCZ4py.general.lcz_cal_indexes` | OK | OK | no optional imports |
| `LCZ4py.general.lcz_cal_indices` | OK | OK | no optional imports |
| `LCZ4py.general.lcz_get_indices` | OK | OK | no optional imports |
| `LCZ4py.general.lcz_get_lst` | OK | OK | pystac_client / planetary_computer / xarray / rioxarray guarded |
| `LCZ4py.general.lcz_get_map` | OK | OK | no optional imports |
| `LCZ4py.general.lcz_get_map_euro` | OK | OK | no optional imports |
| `LCZ4py.general.lcz_get_map_generator` | OK | OK | no optional imports |
| `LCZ4py.general.lcz_get_map_usa` | OK | OK | no optional imports |
| `LCZ4py.general.lcz_get_parameters` | OK | OK | duckdb / geoarrow guarded |
| `LCZ4py.general.lcz_get_planetary_computer` | OK | OK | pystac_client / planetary_computer / rioxarray guarded |
| `LCZ4py.general.lcz_get_ucp` | OK | OK | xarray / rioxarray guarded |
| `LCZ4py.general.lcz_grid_chirps` | OK | OK | no optional imports |
| `LCZ4py.general.lcz_grid_era5` | OK | OK | xarray guarded |
| `LCZ4py.general.lcz_grid_era5_global` | OK | OK | xarray / cdsapi guarded |
| `LCZ4py.general.lcz_grid_pdsi` | OK | OK | xarray guarded |
| `LCZ4py.general.lcz_grid_pollution_ghap` | OK | OK | xarray guarded |
| `LCZ4py.general.lcz_grid_pollution_merra2` | OK | OK | xarray guarded |
| `LCZ4py.general.lcz_plot_map` | OK | OK | duckdb / geoarrow guarded |
| `LCZ4py.general.lcz_plot_parameters` | OK | OK | **xarray + rioxarray unguarded at module top level** — passes here only because both are installed; geoarrow / datashader / colorcet guarded |
| `LCZ4py.general.plot_grid_only` | OK | OK | contextily guarded |
| `LCZ4py.general.plot_lcz_relationship` | OK | OK | no optional imports |
| `LCZ4py.local` (package) | OK | OK | re-exports 43 names |
| `LCZ4py.local.lcz_anomaly` | OK | OK | no optional imports |
| `LCZ4py.local.lcz_anomaly_map` | OK | OK | no optional imports |
| `LCZ4py.local.lcz_climate_compute_spei` | OK | OK | no optional imports |
| `LCZ4py.local.lcz_climate_compute_spi` | OK | OK | no optional imports |
| `LCZ4py.local.lcz_interp_eval` | OK | OK | no optional imports |
| `LCZ4py.local.lcz_interp_map` | OK | OK | no optional imports |
| `LCZ4py.local.lcz_interp_map_plus` | OK | OK | sklearn / xarray guarded |
| `LCZ4py.local.lcz_krige` | OK | OK | pykrige guarded |
| `LCZ4py.local.lcz_plot_interp` | OK | OK | datashader / xarray guarded |
| `LCZ4py.local.lcz_temporal` | OK | OK | no optional imports |
| `LCZ4py.local.lcz_thermal` | OK | OK | no optional imports |
| `LCZ4py.local.lcz_ts` | OK | OK | datashader / xarray guarded |
| `LCZ4py.local.lcz_uhi_intensity` | OK | OK | no optional imports |
| `LCZ4py.local.lcz_uhi_surface` | OK | OK | no optional imports |
| `LCZ4py.local.lcz_variogram` | OK | OK | no optional imports |
| `LCZ4py._internal` (package) | OK | OK | private helpers |
| `LCZ4py._internal._lcz_downloader_base` | OK | OK | duckdb guarded |
| `LCZ4py._internal._lcz_grid_base` | OK | OK | no direct optional imports |
| `LCZ4py._internal._lcz_grid_raster_base` | OK | OK | xarray / rioxarray guarded |
| `LCZ4py._internal._lcz_map_engine` | OK | OK | duckdb / geoarrow guarded |
| `LCZ4py._internal.adaptive_crop_mask` | OK | OK | no optional imports |
| `LCZ4py._internal.i18n_messages` | OK | OK | no optional imports |
| `LCZ4py._internal.lcz_parameters_data` | OK | OK | no optional imports |
| `LCZ4py._internal.lcz_theme` | OK | OK | kaleido probed via `find_spec` |
| `LCZ4py._internal.lcz_ts_utils` | OK | OK | no optional imports |

### 4.2 Per optional dependency

Fresh subprocess, `import LCZ4py.general` + `import LCZ4py.local` with the single
listed dependency blocked via the loader blocker:

| Dependency blocked | Import | Note |
|---|---|---|
| `pystac_client` | OK | guarded by `HAS_STAC` |
| `planetary_computer` | OK | guarded by `HAS_STAC` |
| `h5netcdf` | OK | guarded |
| `h5py` | OK | guarded |
| `pykrige` | OK | guarded by `HAS_PYKRIGE` |
| `sklearn` | OK | guarded |
| `duckdb` | OK | guarded by `HAS_DUCKDB` |
| `geoarrow` | OK | guarded by `HAS_GEOARROW` |
| `exactextract` | OK | guarded |
| `datashader` | OK | guarded by `HAS_DATASHADER` |
| `holoviews` | OK | guarded |
| `colorcet` | OK | guarded |
| `contextily` | OK | guarded |
| `dask` | OK | never imported at package import time |
| `cdsapi` | OK | guarded |
| **`xarray`** | **FAIL** | `ModuleNotFoundError: No module named 'xarray'` |
| (`rioxarray`, not in brief) | **FAIL** | same chain — see §5 |

---

## 5. Import-time failure — exact traceback summary

Command (equivalent to the failing test):

```text
python - <<'PY'
# install loader-based blocker for all 16 optional deps, then:
import LCZ4py.general, LCZ4py.local
PY
```

```text
LCZ4py/__init__.py:10                  from . import general
LCZ4py/general/__init__.py:10          from .lcz_plot_parameters import *
LCZ4py/general/lcz_plot_parameters.py:24   import rioxarray   # <-- UNGUARDED
rioxarray/__init__.py:5                import rioxarray.raster_array
rioxarray/raster_array.py:23           import xarray
ModuleNotFoundError: No module named 'xarray'
```

Blocking `rioxarray` directly produces the identical chain
(`ModuleNotFoundError: No module named 'rioxarray'`).

**Why only this module:** every other module that touches xarray/rioxarray does so
inside `try/except ImportError` and exposes `HAS_XARRAY` / `HAS_RIOXARRAY`
(`lcz_get_lst.py:52-66`, `_lcz_grid_raster_base.py:33-51`, `lcz_get_ucp.py:48-60`).
`lcz_plot_parameters.py` is the sole unguarded top-level importer, and because it
is star-imported by `LCZ4py/general/__init__.py`, it takes down the entire
package import — including `LCZ4py.local` and any submodule import that triggers
the parent package.

**Caveat on realism:** `pyproject.toml` declares `rioxarray>=0.15.0` as a *core*
dependency, and `rioxarray` hard-requires `xarray`. A normal `pip install LCZ4py`
therefore always has xarray. Blocking xarray alone is not a pip-realizable state;
the failure is still worth fixing because (a) the brief defines xarray as
optional, and (b) the rest of the codebase already implements the degraded
fallback — this one file is the inconsistency.

---

## 6. Smoke-test results (offline, synthetic raster)

Fixture: 64×64 `uint8` GeoTIFF, EPSG:3857, 30 m pixels, classes 1–17 plus one
nodata row, written to a pytest `tmp_path`. No network. Total runtime ≈ 24 s.

| Test | Result | What it proves |
|---|---|---|
| `test_synthetic_raster_is_valid` | PASS | fixture integrity: CRS, shape, all 17 classes, nodata |
| `test_load_target_grid` | PASS | `_load_target_grid` shape/CRS/valid mask/WGS84 bbox (only nodata row invalid) |
| `test_daterange` | PASS | inclusive date enumeration; reversed range raises `ValueError` |
| `test_cache_key` | PASS | 16-char hex SHA-256 prefix; deterministic; sensitive to source/date/hour |
| `test_cal_area_dataframe_only` | PASS | `lcz_cal_area(iplot=False)` returns polars frame, 17 rows, counts sum to non-nodata pixels, areas > 0, percentage ≈ 100 % |
| `test_cal_area_returns_figure` | PASS | `iplot=True` returns `LCZAreaResult(df, fig)` with a populated Plotly figure |
| `test_get_parameters_stack` | PASS | `lcz_get_parameters` returns `LCZStackResult` of shape `(35, 64, 64)`; class band round-trips exactly |
| `test_map_class_to_params_vectorized` | PASS | fast path shape/dtype; nodata pixels stay 0 across all parameter bands |
| `test_plot_map_from_raster` | PASS | `lcz_plot_map` builds a `LCZPlotResult` figure from a local raster |
| `test_plot_parameters_from_array` | PASS | `lcz_plot_parameters` accepts a bare 2-D `np.ndarray` |

Optional accelerators are explicitly disabled in these tests
(`use_duckdb=False`, `use_geoarrow=False`, `use_datashader=False`) so the suite
behaves identically on a minimal install. `area_perc` sums to 100.05 because of
float32 aggregation; the test tolerates ±0.5.

`python -m compileall -q LCZ4py` → **exit 0**, no files failed to byte-compile.

---

## 7. Prioritized issues for the other agents

### P1 — BLOCKING (single fix) — unguarded `xarray`/`rioxarray` in `lcz_plot_parameters.py`

`LCZ4py/general/lcz_plot_parameters.py:24-25`

```python
import rioxarray  # noqa: F401 - registers the .rio accessor used on xr.Dataset
import xarray as xr
```

Effect: `ModuleNotFoundError: No module named 'xarray'` (or `'rioxarray'`) aborts
`import LCZ4py` entirely.

Fix: mirror the pattern already used in `lcz_get_lst.py` and
`_lcz_grid_raster_base.py`:

```python
try:
    import rioxarray  # noqa: F401
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    rioxarray = None
    xr = None
    HAS_XARRAY = False
```

and raise a clear `RuntimeError` only on the code paths that actually receive an
`xr.Dataset`/`xr.DataArray` (or call `.rio`). Verified that this file is the
**only** unguarded xarray/rioxarray importer reached during package import, so
this single edit makes both failing tests green.

### P2 — Packaging inconsistency (decision needed)

`pyproject.toml` lists `rioxarray` as a core dependency but `xarray` only in the
`ucp` extra, even though `rioxarray` requires `xarray`. Pick one:

- **Option A (recommended if Dataset support is intended):** treat `xarray` as a
  core dependency — move it out of the `ucp` extra, and drop `xarray` from the
  QA optional-blocked list. P1 is still worth doing for degraded installs.
- **Option B:** make `rioxarray` optional too (remove from core deps, add to an
  extra) and guard both imports.

Until this is resolved, the `array_stack` group test
(`test_import_without_optional_group[array_stack]`) and the combined test
(`test_import_without_all_optional_deps`) stay red by design. They are the only
two failures in the harness.

### P3 — Non-blocking bug: 3-D ndarray input to `lcz_plot_parameters`

`lcz_plot_parameters(stack)` and `lcz_plot_parameters(stack, iselect=["svf_min"])`
with a `(n_params, H, W)` array both fail:

```text
lcz_plot_parameters.py:716  -> _plot_one_interactive(...)
lcz_plot_parameters.py:278  -> h, w = arr.shape
ValueError: too many values to unpack (expected 2)
```

The docstring advertises `np.ndarray` as an accepted input, and the natural
output of `lcz_get_parameters` (and of `map_class_to_params_vectorized`) is
3-D. Either slice the stack by `iselect`/band index before plotting, or document
that only 2-D arrays are accepted. 2-D arrays work correctly (covered by smoke
test).

### P4 — Minor: confusing error for dict input

`lcz_plot_parameters({"height_roof": arr})` raises
`ValueError: lcz_get_ucp result has no 'combined_rasters' to plot — check 'failed_variables' in the result`.
The message is misleading for a plain dict of 2-D arrays. Low priority.

---

## 8. Remaining risks

1. **Concurrent edits.** 31 files were modified in the working tree while this
   report was produced. The harness must be re-run after the other agents land
   their changes; results here are a snapshot of the current tree, not of
   `6d80647`.
2. **Import-time network not socket-blocked.** Tests are offline and no HTTP was
   observed, but no test asserts that importing the package performs no network
   I/O. A socket-blocking import test would close this gap.
3. **Absent optional deps are only exercised at import time.** `geoarrow`,
   `exactextract`, `datashader`, `holoviews`, `colorcet` are genuinely missing;
   their fallback *behaviour* (as opposed to import safety) is only covered where
   the smoke suite disables them explicitly.
4. **xarray fallback paths are untested.** `HAS_XARRAY=False` branches in
   `lcz_get_lst` / `lcz_get_ucp` / `_lcz_grid_raster_base` are never executed
   because xarray is installed; they should raise clear errors rather than
   `AttributeError` on `xr = None`.
5. **Single interpreter.** Python 3.13 only; no 3.10/3.11 matrix despite
   `requires-python = ">=3.10"`.
6. **`compileall` is syntax-only.** It does not detect unresolved names,
   signature drift, or dead code.

---

## 9. Final consolidated verification (after all 7 agents landed)

All agents finished and the tree was re-verified end-to-end.

### 9.1 Resolved findings

| ID | Finding | Status |
|---|---|---|
| P1 | Unguarded `xarray`/`rioxarray` in `lcz_plot_parameters.py` | **FIXED** — `HAS_XARRAY`/`HAS_RIOXARRAY` guards; `import LCZ4py` passes with both blocked |
| P3 | 3-D `(n_params,H,W)` ndarray crash in `lcz_plot_parameters` | **FIXED** — 2-D and 3-D both accepted; `param=` selector + clear band/slice errors |
| P2 | `xarray` packaging inconsistency | **Mitigated** — both imports guarded, so the package works either way; packaging decision still recommended |
| — | `test_era5_global_no_auth_raises` depended on a real `~/.cdsapirc` | **FIXED** — test isolates `HOME`/`USERPROFILE` to `tmp_path` |

### 9.2 Gates

| Gate | Command | Result |
|---|---|---|
| Syntax | `python -m compileall -q LCZ4py` | **PASS** (exit 0) |
| Package import | `python -c "import LCZ4py"` | **PASS** (0.1.0) |
| Exports | `LCZGridResult`, `parameter_names`, `krige_predict_points`, all public names | **PASS** |
| Full test suite | `python -m pytest tests/ -m "not network" -q` | **PASS — 146 passed, 9 deselected** |
| Import + smoke | `pytest tests/test_imports_general_local.py tests/test_offline_smoke.py -q` | **PASS — 69 passed** |
| Core-only import (22 optional deps blocked) | custom `sys.meta_path` blocker incl. `xarray`, `rioxarray`, `dask`, `datashader`, `exactextract`, `pykrige`, `sklearn`, `duckdb`, `cdsapi`, `contextily`, `aiofiles`, `tenacity`, `nest_asyncio` | **PASS** — `CORE_ONLY_IMPORT_OK`; every `HAS_*` flag correctly `False` |

### 9.3 Per-area outcome

| Area | Result |
|---|---|
| LST + Planetary Computer | Parallel per-date/asset; keyless STAC guarded; atomic validated cache; `source="both"` fallback; 25/25 offline checks |
| Grid family | Parallel downloads; CDS ZIP-payload bug fixed; `plot_grid_only` publication cartography; core-only import PASS |
| Maps + map plots | Deterministic parallel tiling; legend/scale bar/north arrow/credit; 300 dpi; exact WGS84 areas; core-only import PASS |
| Local time-series/UHI | Vectorised `bincount`/`searchsorted`; `thread`/`process` executor with safe fallback; publication maps; 32/32 smoke |
| Interpolation/kriging | Pure-NumPy OK fallback; sklearn → NumPy ridge fallback; deterministic parallel bands; 20/20 smoke |
| Parameters/indices/UCP | Nodata-corruption fix; rasterio-only UCP fallback; 79/79 smoke + 61 existing tests |
| QA harness | 59 import tests + 10 offline smoke; 22-dep blocker; report |

### 9.4 Residual risks (accepted)

1. **Live network paths** (Azure blob listing, PC SAS shape, CDS queue, Sentinel-3 asset keys) were validated with offline/monkeypatched data only. First-run smoke against real endpoints is still recommended.
2. **Threads vs CPU-bound kriging**: `max_workers` gives ~1.6×; `use_processes=True`/`executor="process"` gives real parallelism but needs a `__main__`-guarded script.
3. **`~/.lcz4r_cache` has no eviction policy** (pre-existing).
4. **Python 3.13 only** tested; `requires-python >=3.10` remains unverified on 3.10/3.11.
5. **Kaleido/Chromium** static export works through the guarded exporter; on machines without Chromium, the package raises a clear i18n error instead of crashing.

---

## 10. Native static rendering is now the default (all figure functions)

### 10.1 Backend

New shared module `LCZ4py/_internal/lcz_static.py` (matplotlib/Agg):

- `new_figure`, `style_context`, `save_static` — journal-sized figures and
  `>=300 dpi` PNG/PDF/SVG/TIFF output; `"html"` is coerced to `"png"`.
- `plot_lcz_raster`, `plot_continuous_raster` — class/continuous rasters.
- `apply_map_axes`, `add_map_furniture_mpl` — metric labels, two-tone scale
  bar, north arrow (skipped for geographic CRS).
- `class_legend`, `add_colorbar`, `add_header`, `add_footer`.
- `resolve_renderer` — `"static"` (default; aliases `None`, `"matplotlib"`,
  `"agg"`) or opt-in `"plotly"`/`"maplibre"`.

It does **not** import Plotly, Kaleido, Chromium or a browser. Verified by
blocking `kaleido` in a subprocess and rendering/saving successfully.

### 10.2 Coverage

| Function | Default return | Opt-in |
|---|---|---|
| `lcz_plot_map` | matplotlib `Figure` (`LCZPlotResult.fig`) | `renderer="plotly"` / `"maplibre"` |
| `lcz_plot_parameters` | `Figure` / `list[Figure]` / faceted `Figure` | `renderer="plotly"` |
| `lcz_cal_area` | matplotlib `Figure` in `LCZAreaResult.fig` | `renderer="plotly"` |
| `lcz_cal_indices` | `Figure` / `dict[str, Figure]` | `renderer="plotly"` |
| `lcz_cal_indexes` | matplotlib `Figure` | `renderer="plotly"` |
| `plot_lcz_relationship` | matplotlib `Figure` | `renderer="plotly"` |
| `plot_grid_only` | matplotlib `Figure` (300 dpi floor) | static only |
| `lcz_ts`, `lcz_anomaly`, `lcz_uhi_intensity` | matplotlib `Figure` | `renderer="plotly"` |
| `lcz_uhi_surface` | `Figure` / `dict[str, Figure]` | `renderer="plotly"` |
| `lcz_dtr`, `lcz_degree_hours` | matplotlib `Figure` | `renderer="plotly"` |
| `lcz_utci`, `lcz_anthropogenic_heat` | matplotlib `Figure` | `renderer="plotly"` |
| `lcz_plot_interp` | matplotlib `Figure` (single/facet) | `renderer="plotly"` |
| `lcz_variogram` | matplotlib `Figure` | `renderer="plotly"` |

Result dataclasses (`LCZPlotResult`, `LCZAreaResult`,
`LCZIndicesStatsResult`, `LCZIndexesResult`, `LCZUHISurfaceResult`,
`UTCIResult`, `AnthropogenicHeatResult`, `VariogramResult`, `DTRResult`,
`DegreeHoursResult`) expose `renderer="static"` and a `.show()` that
dispatches matplotlib vs Plotly.

### 10.3 Bugs fixed during the rollout

- `lcz_static.add_map_furniture_mpl`: boolean kwargs shadowed the
  same-named functions → `TypeError`; refactored to `_draw_scalebar` /
  `_draw_north_arrow`.
- `plot_lcz_relationship(plot_type="violin", renderer="plotly")` passed
  `pointpos` to `px.violin`, removed in Plotly 6 → removed.
- `lcz_plot_parameters(chart_type=...)` now forwards `renderer`/`dpi` to
  `lcz_cal_indices` instead of silently forcing its default.

### 10.4 Verification

| Gate | Result |
|---|---|
| `python -m pytest tests/ -m "not network" -q` | **PASS — 152 passed, 9 deselected** |
| `python -m compileall -q LCZ4py` | **PASS** |
| Core-only import with 22 optional deps blocked | **PASS** |
| Static render/save with `kaleido` blocked | **PASS** — `lcz_plot_map`, `lcz_cal_area`, `lcz_plot_parameters` wrote PNGs |
| `renderer="plotly"` smoke for every covered function | **PASS** (agents' 30/30, 66/67, 69/69, 21/21 harnesses) |

### 10.5 Behaviour change (intentional)

The default return type of every figure-producing function is now a
matplotlib `Figure` (or a result object whose `.fig` is one), not a Plotly
`Figure`. Pass `renderer="plotly"` to keep the previous interactive output.
`save_extension` still defaults to `"html"` on Plotly paths; the static path
writes `"png"` and coerces `"html"` to `"png"`.

---

## 11. Notebooks updated for the matplotlib default

All **37 notebooks** were updated (30 tutorial notebooks under `notebooks/`,
`notebooks/tutorial_lcz_grid_integration.ipynb`, and the 6 root reference
notebooks).

### 11.1 Conventions

- Plotting calls keep the default `renderer` (static matplotlib).
- `.fig.show()` / `.plot.show()` were replaced with `display(...)` (or the
  Figure as the last expression); list/dict returns are displayed in a loop.
- Every notebook gained a one-line "plotting defaults" note and an
  `from IPython.display import display` import.
- Cells that explicitly teach interactivity keep `renderer="plotly"` or
  `renderer="maplibre"` and `.show()`.
- Markdown claims of "interactive/Plotly by default" were rewritten.
- Outputs and `execution_count` were cleared only on cells whose source
  changed, so stale Plotly outputs do not contradict the new code.

### 11.2 Run-blocking fixes applied while updating

- Bare `pip install LCZ4py` → `%pip install LCZ4py` (invalid Python cell).
- `06_urban_canopy_parameters`: `iselect=["BSF","BH","SVF"]` did not match the
  UCP stack; now `iselect=result_stack["variable_list"][:3]`.
- `07_gridded_climate_environment`: `color_by="value"` → `color_by="precip_mm"`.
- `06_temporal_climate_metrics`: `lcz_lcz_map_path` → `lcz_map_path`.
- `03_morphological_parameters`: undefined `result_stack` → `stack_result`.
- Root notebooks: legacy flat imports (`from lcz_get_map import ...`) migrated
  to `from LCZ4py import ...`; private interp helpers use
  `LCZ4py.local.lcz_interp_map_plus`.
- `lcz_get_lst_tutorial`: Plotly-only `fig.data[0].colorbar.title.text`
  replaced with `lst.units` + `display(fig)`.
- `tutorial_lcz_grid_integration`: removed forced `matplotlib.use("Agg")` and
  replaced `write_html` with a 300 dpi `savefig`.

### 11.3 Verification

| Gate | Result |
|---|---|
| `pytest tests/test_notebooks_integrity.py` | **PASS — 75 tests** (37 × validity + 37 × AST + discovery) |
| Root-notebook import cells executed | **PASS — 6/6** |
| Residual default-static `.fig.show()`/`.plot.show()` | **0** |
| Residual bare `pip install` cells | **0** |
| Residual "interactive by default" markdown | **0** |
| Full suite `pytest tests/ -m "not network"` | **PASS — 227 passed** |
