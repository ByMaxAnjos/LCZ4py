# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python port of the R package `lcz4r` — tools for downloading, processing, and visualizing **Local Climate Zone (LCZ)** maps (Stewart & Oke 2012). No package structure; it's a flat collection of modules imported directly.

## Running / testing

No build step, no test runner. Execute the usage example directly:

```bash
python test.py
```

To exercise a single function interactively:

```bash
python -c "from lcz_get_map import lcz_get_map; print(lcz_get_map('Berlin'))"
```

Install deps:

```bash
pip install -r requirements.txt
```

## Architecture

### Module roles

| Module | Role |
|---|---|
| `_lcz_downloader_base.py` | Shared async HTTP/COG helpers: geocoding, cache read/write, raster streaming, crop+mask. All `lcz_get_map_*` modules import from here. Also exposes `lcz_get_map_fast` (sync wrapper over async COG downloader). |
| `lcz_get_map.py` | Global LCZ map (Zenodo, Demuzere 2022). Two-tier cache: GeoJSON boundary + clipped GeoTIFF. Delegates heavy crop/mask to `adaptive_crop_mask.py`. |
| `lcz_get_map_euro.py` | European map variant (different Zenodo URL, same base helpers). |
| `lcz_get_map_usa.py` | CONUS map variant. |
| `adaptive_crop_mask.py` | Three-tier crop strategy by area: standard (<500k km²), simplified boundary (500k–5M km²), tiled parallel mosaic (>5M km²). |
| `lcz_cal_area.py` | Area calculation per LCZ class. Returns `LCZAreaResult(df, fig, geoarrow_table)`. Polars for aggregation, DuckDB optional fast path, Plotly for charts (bar/pie/donut/sunburst/treemap). |
| `lcz_get_parameters.py` | Extracts 34 morphological/thermal parameters per pixel via NumPy fancy indexing into a precomputed `(18, n_params)` lookup table. Returns `LCZStackResult(path, array, gdf, geoarrow_table)`. |
| `lcz_plot_map.py` | Interactive Plotly (WebGL) map rendering with DuckDB/GeoArrow support. |
| `lcz_plot_parameters.py` | Per-parameter raster visualization with optional Datashader for large rasters. |
| `lcz_parameters_data.py` | Single source of truth: `LCZ_NAMES`, `LCZ_COLORS`, `LCZ_COLORBLIND`, `LCZ_IDS`, and the `LCZ_TABLE` dataclass with all 34 parameters for classes 1–17. |
| `i18n_messages.py` | `lcz_msg(key, lang, **kwargs)` — multilingual messages (en/pt/es/zh). Add new keys to `LCZ_MESSAGES` dict. |

### Key conventions

**Optional dependencies** — `duckdb`, `geoarrow`, `exactextract` all use `try/import` guards and `HAS_*` flags. Functions degrade gracefully without them. Never assume they're present.

**Outputs** always land in `LCZ4r_output/` (local, created on demand). Disk cache lives at `~/.lcz4r_cache/`.

**Cache keys** are SHA1/SHA256 hashes of `city_slug|bbox` so the same city+region always hits the same file. Two-level: geocoded boundary (GeoJSON) + clipped raster (GeoTIFF).

**COG streaming** — remote rasters are never fully downloaded. `_lcz_downloader_base.stream_cog_window` opens via `/vsicurl/{url}` and reads only the intersecting window. `WarpedVRT` handles CRS mismatches in-memory.

**LCZ class numbering** — classes 1–10 are urban, 11–17 (A–G in R notation) are natural/water. Index 0 = nodata. The lookup table in `lcz_get_parameters.py` is indexed `[0..17]` with row 0 zeroed out.

**i18n** — pass `lang=` ("en"/"pt"/"es"/"zh") through to `lcz_msg`. Default is `"en"`.
