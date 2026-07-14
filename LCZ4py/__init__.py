"""LCZ4py — Python port of the LCZ4r R package (Local Climate Zone tools).

Functions are grouped following the LCZ4r paper's categorisation:
- LCZ4py.general — LCZ map/raster acquisition, morphological parameters, gridded
  environmental data, area/index statistics.
- LCZ4py.local — station/point-based air-temperature time series, anomalies,
  UHI, interpolation, and thermal comfort indices.
"""

from . import general
from . import local

from .general import *  # noqa: F401,F403
from .local import *  # noqa: F401,F403

__all__ = list(general.__all__) + list(local.__all__)

__version__ = "0.1.0"
