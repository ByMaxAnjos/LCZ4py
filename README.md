# LCZ4py

Python port of [LCZ4r](https://github.com/ByMaxAnjos/LCZ4r) — tools for downloading, processing, and
visualizing Local Climate Zone (LCZ) maps (Stewart & Oke 2012) and applying them to urban climate
analysis (air temperature, land surface temperature, UHI, thermal comfort).

Reference: Anjos, M. et al. (2025). *LCZ4r: An open-source R package for calculating local climate
zone products*. Scientific Reports. https://www.nature.com/articles/s41598-025-92000-0

## Install

```bash
pip install -e .            # core
pip install -e ".[all]"     # + optional extras (viz, geoarrow, duckdb, zonal, interp, ucp, lst, era5-global)
```

## Usage

```python
from LCZ4py import lcz_get_map, lcz_plot_map, lcz_get_parameters

map_path = lcz_get_map(city="Berlin", isave_map=True)
```

## Structure

Functions are grouped following the LCZ4r paper's categorisation:

- `LCZ4py.general` — LCZ map/raster acquisition, morphological parameters, gridded environmental
  data (LST, CHIRPS, ERA5, PDSI, pollution), area/index statistics.
- `LCZ4py.local` — station/point-based air-temperature time series, anomalies, UHI (air + surface),
  spatial interpolation, variograms, thermal comfort, drought indices.
- `LCZ4py._internal` — shared helpers (downloader/cache/i18n/grid base). Not public API.

## Testing

```bash
pytest tests/
python test.py   # end-to-end usage example (downloads a real LCZ map)
```
