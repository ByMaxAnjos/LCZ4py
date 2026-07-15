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

## Tutorials

Two Colab notebook series — `general/` (raster-wide LCZ map operations) and `local/` (station/point-based
analysis) — each available in English and Portuguese. Click a badge to open and run it directly in Colab.

### General

| # | Topic | EN | PT |
|---|---|---|---|
| 01 | Map acquisition (global, Europe, USA, LCZ Generator) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/01_map_acquisition.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/01_map_acquisition.pt.ipynb) |
| 02 | Visualization & area statistics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/02_visualization_area_stats.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/02_visualization_area_stats.pt.ipynb) |
| 03 | Morphological parameters (34 LCZ parameters) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/03_morphological_parameters.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/03_morphological_parameters.pt.ipynb) |
| 04 | Remote sensing: LST & Planetary Computer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/04_remote_sensing_lst_pc.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/04_remote_sensing_lst_pc.pt.ipynb) |
| 05 | Spectral indices (NDVI, NDBI, ...) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/05_spectral_indices.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/05_spectral_indices.pt.ipynb) |
| 06 | Urban Canopy Parameters (GHSL, WUDAPT, vegetation) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/06_urban_canopy_parameters.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/06_urban_canopy_parameters.pt.ipynb) |
| 07 | Gridded climate & environmental data (CHIRPS, ERA5, PDSI, pollution) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/07_gridded_climate_environment.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/general/07_gridded_climate_environment.pt.ipynb) |

### Local

| # | Topic | EN | PT |
|---|---|---|---|
| 01 | LCZ-stratified time series | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/01_lcz_time_series.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/01_lcz_time_series.pt.ipynb) |
| 02 | Temperature anomalies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/02_temperature_anomalies.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/02_temperature_anomalies.pt.ipynb) |
| 03 | Urban Heat Island (air + surface) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/03_urban_heat_island.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/03_urban_heat_island.pt.ipynb) |
| 04 | Spatial interpolation & geostatistics (kriging, variograms) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/04_spatial_interpolation_geostats.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/04_spatial_interpolation_geostats.pt.ipynb) |
| 05 | ML-based interpolation with Urban Canopy Parameters | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/05_ml_interpolation_ucp.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/05_ml_interpolation_ucp.pt.ipynb) |
| 06 | Temporal climate metrics (DTR, degree hours) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/06_temporal_climate_metrics.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/06_temporal_climate_metrics.pt.ipynb) |
| 07 | Thermal comfort (UTCI) & anthropogenic heat | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/07_thermal_comfort_anthropogenic_heat.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/07_thermal_comfort_anthropogenic_heat.pt.ipynb) |
| 08 | Drought indices (SPI, SPEI) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/08_drought_indices_spi_spei.en.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ByMaxAnjos/LCZ4py/blob/main/notebooks/local/08_drought_indices_spi_spei.pt.ipynb) |

The repo root also has earlier working notebooks (`test_general.ipynb`, `test_local.ipynb`,
`interp_map.ipynb`, `lcz_grid_tutorial.ipynb`, `lcz_get_planetary_computer_tutorial.ipynb`,
`lcz_get_lst_tutorial.ipynb`) kept for reference; the `notebooks/` series above is the maintained,
didactic tutorial set going forward.

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
