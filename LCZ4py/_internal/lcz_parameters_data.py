"""
LCZ parameters reference data — Python equivalent of the ``lcz.df`` table
embedded inside ``lcz_get_parameters.R``.

Single source of truth for LCZ class codes, names, colors and the 34
morphological / thermal parameters from Stewart & Oke (2012).

Best-practice notes
-------------------
* All numeric vectors are stored as ``numpy`` arrays so downstream code can
  use vectorised reductions instead of Python ``for`` loops.
* The two colorblind palettes are kept side-by-side; pick via ``inclusive``.
* ``LCZ_COLUMNS`` is the canonical column order for stack / shapefile
  outputs and matches the original R code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

# ── LCZ class metadata ─────────────────────────────────────────────────────────

LCZ_NAMES: Final[list[str]] = [
    "Compact highrise", "Compact midrise", "Compact lowrise", "Open highrise",
    "Open midrise", "Open lowrise", "Lightweight low-rise", "Large lowrise",
    "Sparsely built", "Heavy Industry", "Dense trees", "Scattered trees",
    "Bush, scrub", "Low plants", "Bare rock or paved", "Bare soil or sand",
    "Water",
]

LCZ_NAMES_I18N: Final[dict[str, list[str]]] = {
    "en": LCZ_NAMES,
    "pt": [
        "Compacto de alta densidade", "Compacto de média densidade",
        "Compacto de baixa densidade", "Aberto de alta densidade",
        "Aberto de média densidade", "Aberto de baixa densidade",
        "Construção leve de baixo gabarito", "Grande estrutura de baixo gabarito",
        "Edificação esparsa", "Indústria pesada",
        "Árvores densas", "Árvores esparsas",
        "Arbustos e matagal", "Vegetação baixa",
        "Rocha exposta ou pavimento", "Solo nu ou areia", "Água",
    ],
    "es": [
        "Compacto de gran altura", "Compacto de altura media",
        "Compacto de baja altura", "Abierto de gran altura",
        "Abierto de altura media", "Abierto de baja altura",
        "Construcción ligera de baja altura", "Gran edificación de baja altura",
        "Edificación dispersa", "Industria pesada",
        "Árboles densos", "Árboles dispersos",
        "Arbustos y matorral", "Plantas bajas",
        "Roca desnuda o pavimento", "Suelo desnudo o arena", "Agua",
    ],
    "zh": [
        "紧凑高层", "紧凑中层", "紧凑低层",
        "开放高层", "开放中层", "开放低层",
        "轻型低层", "大型低层", "稀疏建筑",
        "重工业", "密林", "疏林",
        "灌木丛", "低矮植被", "裸岩或铺装",
        "裸土或沙地", "水体",
    ],
}


def get_lcz_names(lang: str = "en") -> list[str]:
    """Return the 17 LCZ class names in the requested language (falls back to 'en')."""
    return LCZ_NAMES_I18N.get(lang, LCZ_NAMES_I18N["en"])

LCZ_COLORS: Final[list[str]] = [
    "#910613", "#D9081C", "#FF0A22", "#C54F1E", "#FF6628", "#FF985E",
    "#FDED3F", "#BBBBBB", "#FFCBAB", "#565656", "#006A18", "#00A926",
    "#628432", "#B5DA7F", "#000000", "#FCF7B1", "#656BFA",
]

LCZ_COLORBLIND: Final[list[str]] = [
    "#E16A86", "#D8755E", "#C98027", "#B48C00",
    "#989600", "#739F00", "#36A631", "#00AA63",
    "#00AD89", "#00ACAA", "#00A7C5", "#009EDA",
    "#6290E5", "#9E7FE5", "#C36FDA", "#D965C6",
    "#E264A9",
]

LCZ_CODES: Final[list[str]] = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "A", "B", "C", "D", "E", "F", "G",
]

LCZ_IDS: Final[np.ndarray] = np.arange(1, 18, dtype=np.int16)


# ── Lookup tables (the Stewart & Oke 2012 ranges) ─────────────────────────────

# Helper builders that mirror the R ``c(rep(0.9, 4))`` style.
def _rep(value: float, n: int) -> list[float]:
    return [float(value)] * n


def _tail(values: list[float]) -> list[float]:
    return list(values)


# ── Parameter ranges ──────────────────────────────────────────────────────────
# Order matches LCZ_IDS (17 classes).

_PARAM_RANGES: dict[str, list[float]] = {
    "SVFmin": [0.2, 0.3, 0.2, 0.5, 0.5, 0.6, 0.2, 0.75, 0.85, 0.6,
               0.35, 0.5, 0.7] + _rep(0.9, 4),
    "SVFmax": [0.4, 0.6, 0.6, 0.7, 0.8, 0.9, 0.5, 0.75, 0.85, 0.9,
               0.35, 0.8, 0.9] + _rep(0.9, 4),
    "ARmin":  [3, 0.75, 0.75, 0.75, 0.3, 0.3, 1, 0.1, 0.1, 0.2,
               1.5, 0.25, 0.25] + _rep(0.1, 4),
    "ARmax":  [3, 2, 1.5, 1.25, 0.75, 0.75, 2, 0.3, 0.25, 0.5,
               1.5, 0.75, 1.0] + _rep(0.1, 4),
    "BSFmin": [40, 40, 40] + _rep(20, 3) + [60, 30, 10, 20] + _rep(9, 7),
    "BSFmax": [60, 70, 70] + _rep(40, 3) + [90, 50, 20, 30] + _rep(9, 7),
    "ISFmin": [40, 40, 40] + _rep(20, 3) + [60, 30, 10, 20] + _rep(0, 7),
    "ISFmax": [60, 70, 70] + _rep(40, 3) + [90, 50, 20, 30] + _rep(10, 7),
    "PSFmax": [10, 20, 30, 40, 40, 60, 30, 20, 80, 50] + _rep(100, 4) + [10, 100, 100],
    "PSFmin": [0, 0, 0, 30, 20, 30, 0, 0, 60, 40] + _rep(90, 4) + [0, 90, 90],
    "TSFmin": _rep(0, 10) + [90, 90] + _rep(0, 5),
    "TSFmax": _rep(0, 10) + [100, 100] + _rep(0, 5),
    "HREmin": [26, 10, 3, 26, 10, 3, 2, 3, 3, 5, 3, 3, 2.9, 0.9, 0.24, 0.23, 0],
    "HREmax": [26, 25, 10, 26, 25, 10, 4, 10, 10, 15, 30, 15, 2.9, 0.9, 0.24, 0.23, 0],
    "TRCmin": [8, 6, 6, 7, 5, 5, 4, 5, 5, 5, 8, 5, 4, 3, 1, 1, 1],
    "TRCmax": [8, 7, 6, 8, 6, 6, 5, 5, 6, 6, 8, 6, 5, 4, 2, 2, 1],
    "SADmin": [1.500, 1.500, 1.200, 1.400, 1.400, 1.200, 800, 1.200, 1.000, 1.000,
               0, 1.000, 700, 1.200, 1.200, 600, 1.500],
    "SADmax": [1.800, 2.000, 1.800, 1.800, 2.000, 1.800, 1.500, 1.800, 1.800, 2.500,
               0, 1.800, 1.500, 1.600, 2.500, 1.400, 1.500],
    "SALmin": _rep(0.10, 3) + _rep(0.12, 3) + _rep(0.15, 2) + _rep(0.12, 2)
              + [0.10] + _rep(0.15, 4) + [0.20, 0.02],
    "SALmax": _rep(0.20, 3) + _rep(0.25, 3) + [0.35, 0.25, 0.25, 0.20, 0.20,
                                                0.25, 0.30, 0.25, 0.30, 0.35, 0.10],
    "AHmin":  [50, 74, 74, 49, 24, 24, 34, 49, 9, 310] + _rep(0, 7),
    "AHmax":  [300, 74, 74, 49, 24, 24, 34, 49, 9, 310] + _rep(0, 7),
}


# ── z0 (roughness length) by code ─────────────────────────────────────────────

_Z0_BY_CODE: dict[str, float] = {
    "G": 0.0002, "E": 0.0005, "F": 0.0005, "D": 0.03,
    "7": 0.10, "C": 0.10,
    "8": 0.25, "B": 0.25,
    "2": 0.5, "3": 0.5, "5": 0.5, "6": 0.5, "9": 0.5, "10": 0.5,
    "4": 1.0,
    "1": 2.0, "A": 2.0,
}


def z0_for_code(code: str) -> float:
    """Roughness-length class for a Stewart & Oke LCZ code."""
    return _Z0_BY_CODE.get(code, float("nan"))


# ── Parameter metadata for plotting ───────────────────────────────────────────

PARAM_NAMES: Final[dict[str, str]] = {
    "SVFmin": "Minimum Sky View Factor",
    "SVFmax": "Maximum Sky View Factor",
    "SVFmean": "Mean Sky View Factor",
    "ARmin": "Minimum Aspect Ratio",
    "ARmax": "Maximum Aspect Ratio",
    "ARmean": "Mean Aspect Ratio",
    "BSFmin": "Minimum Building Surface Fraction",
    "BSFmax": "Maximum Building Surface Fraction",
    "BSFmean": "Mean Building Surface Fraction",
    "ISFmin": "Minimum Impervious Surface Fraction",
    "ISFmax": "Maximum Impervious Surface Fraction",
    "ISFmean": "Mean Impervious Surface Fraction",
    "PSFmin": "Minimum Pervious Surface Fraction",
    "PSFmax": "Maximum Pervious Surface Fraction",
    "PSFmean": "Mean Pervious Surface Fraction",
    "TSFmin": "Minimum Tree Surface Fraction",
    "TSFmax": "Maximum Tree Surface Fraction",
    "TSFmean": "Mean Tree Surface Fraction",
    "HREmin": "Minimum Height Roughness Elements",
    "HREmax": "Maximum Height Roughness Elements",
    "HREmean": "Mean Height Roughness Elements",
    "TRCmin": "Minimum Terrain Roughness Class",
    "TRCmax": "Maximum Terrain Roughness Class",
    "TRCmean": "Mean Terrain Roughness Class",
    "SADmin": "Minimum Surface Admittance",
    "SADmax": "Maximum Surface Admittance",
    "SADmean": "Mean Surface Admittance",
    "SALmin": "Minimum Surface Albedo",
    "SALmax": "Maximum Surface Albedo",
    "SALmean": "Mean Surface Albedo",
    "AHmin": "Minimum Anthropogenic Heat Output",
    "AHmax": "Maximum Anthropogenic Heat Output",
    "AHmean": "Mean Anthropogenic Heat Output",
    "z0": "Roughness Length",

    # Spectral indices (lcz_get_indices) — vegetation
    "NDVI": "Normalized Difference Vegetation Index",
    "EVI": "Enhanced Vegetation Index",
    "SAVI": "Soil-Adjusted Vegetation Index",
    "MSAVI": "Modified Soil-Adjusted Vegetation Index",
    "GNDVI": "Green Normalized Difference Vegetation Index",
    "GRNDVI": "Green-Red Normalized Difference Vegetation Index",
    "ARVI": "Atmospherically Resistant Vegetation Index",
    "VARI": "Visible Atmospherically Resistant Index",
    "CIG": "Chlorophyll Index Green",
    "CVI": "Chlorophyll Vegetation Index",
    "TVI": "Triangular Vegetation Index",
    "OSAVI": "Optimized Soil-Adjusted Vegetation Index",
    "RDVI": "Renormalized Difference Vegetation Index",
    "DVI": "Difference Vegetation Index",
    "IPVI": "Infrared Percentage Vegetation Index",
    "WDVI": "Weighted Difference Vegetation Index",
    "TNDVI": "Transformed Normalized Difference Vegetation Index",
    "UVI": "Urban Vegetation Index",

    # Spectral indices — water / moisture
    "NDWI": "Normalized Difference Water Index",
    "MNDWI": "Modified Normalized Difference Water Index",
    "NDMI": "Normalized Difference Moisture Index",
    "MSI": "Moisture Stress Index",
    "LSWI": "Land Surface Water Index",
    "AWEI_NSH": "Automated Water Extraction Index (non-shadow)",
    "AWEI_SH": "Automated Water Extraction Index (shadow)",
    "WI": "Water Index",
    "NDPI": "Normalized Difference Pond Index",
    "MNDWI2": "Modified Normalized Difference Water Index (SWIR2)",
    "SWI": "Surface Water Index",
    "NDTI": "Normalized Difference Turbidity Index",

    # Spectral indices — urban / built-up
    "NDBI": "Normalized Difference Built-up Index",
    "IBI": "Index-based Built-up Index",
    "EBBI": "Enhanced Built-up and Bareness Index",
    "UI": "Urban Index",
    "BAEI": "Built-up Area Extraction Index",
    "NBAI": "New Built-up Area Index",
    "BUI": "Built-up Index",
    "VIBI": "Visible and Infrared Built-up Index",
    "BRBA": "Band Ratio for Built-up Area",
    "NBLI": "New Built-up Land Index",
    "UTFVI": "Urban Thermal Field Variance Index",
    "ISA": "Impervious Surface Area",

    # Spectral indices — soil
    "BSI": "Bare Soil Index",
    "SI": "Salinity Index",
    "BI": "Brightness Index",
    "CI": "Clay Index",
    "RI": "Redness Index",
    "SBI": "Soil Brightness Index",
    "CARI": "Carbonate Index",
    "GSI": "Gypsum Soil Index",
    "SOIL_MOISTURE": "Soil Moisture Index",
    "IRON_OXIDE": "Iron Oxide Index",
    "FERROUS_IRON": "Ferrous Iron Index",
    "SOIL_COMPOSITION": "Soil Composition Index",

    # Spectral indices — thermal (Landsat only)
    "LST_K": "Land Surface Temperature",
    "LST_C": "Land Surface Temperature",
}

PARAM_UNITS: Final[dict[str, str]] = {
    "SVFmin": "[0 - 1]", "SVFmax": "[0 - 1]", "SVFmean": "[0 - 1]",
    "ARmin":  "[0 - 3]", "ARmax":  "[0 - 3]", "ARmean":  "[0 - 3]",
    "BSFmin": "[%]",     "BSFmax": "[%]",     "BSFmean": "[%]",
    "ISFmin": "[%]",     "ISFmax": "[%]",     "ISFmean": "[%]",
    "PSFmin": "[%]",     "PSFmax": "[%]",     "PSFmean": "[%]",
    "TSFmin": "[%]",     "TSFmax": "[%]",     "TSFmean": "[%]",
    "HREmin": "[m]",     "HREmax": "[m]",     "HREmean": "[m]",
    "TRCmin": "[m]",     "TRCmax": "[m]",     "TRCmean": "[m]",
    "SADmin": "[J m-2 s1/2 K-1]", "SADmax": "[J m-2 s1/2 K-1]",
    "SADmean": "[J m-2 s1/2 K-1]",
    "SALmin": "[0 - 0.5]", "SALmax": "[0 - 0.5]", "SALmean": "[0 - 0.5]",
    "AHmin":  "[W m-2]",   "AHmax":  "[W m-2]",   "AHmean":  "[W m-2]",
    "z0":     "[m]",

    "NDVI": "[-1 - 1]", "EVI": "[-1 - 1]", "SAVI": "[-1 - 1]",
    "MSAVI": "[-1 - 1]", "GNDVI": "[-1 - 1]", "GRNDVI": "[-1 - 1]",
    "ARVI": "[-1 - 1]", "VARI": "[ratio]", "CIG": "[ratio]", "CVI": "[ratio]",
    "TVI": "[index]", "OSAVI": "[-1 - 1]", "RDVI": "[ratio]", "DVI": "[reflectance diff]",
    "IPVI": "[0 - 1]", "WDVI": "[reflectance diff]", "TNDVI": "[index]", "UVI": "[ratio]",

    "NDWI": "[-1 - 1]", "MNDWI": "[-1 - 1]", "NDMI": "[-1 - 1]",
    "MSI": "[ratio]", "LSWI": "[-1 - 1]", "AWEI_NSH": "[index]", "AWEI_SH": "[index]",
    "WI": "[ratio]", "NDPI": "[-1 - 1]", "MNDWI2": "[-1 - 1]", "SWI": "[reflectance]",
    "NDTI": "[-1 - 1]",

    "NDBI": "[-1 - 1]", "IBI": "[-1 - 1]", "EBBI": "[ratio]", "UI": "[-1 - 1]",
    "BAEI": "[ratio]", "NBAI": "[-1 - 1]", "BUI": "[-1 - 1]", "VIBI": "[ratio]",
    "BRBA": "[ratio]", "NBLI": "[ratio]", "UTFVI": "[ratio]", "ISA": "[0 - 1]",

    "BSI": "[-1 - 1]", "SI": "[reflectance]", "BI": "[reflectance]", "CI": "[ratio]",
    "RI": "[ratio]", "SBI": "[reflectance]", "CARI": "[index]", "GSI": "[ratio]",
    "SOIL_MOISTURE": "[-1 - 1]", "IRON_OXIDE": "[ratio]", "FERROUS_IRON": "[ratio]",
    "SOIL_COMPOSITION": "[ratio]",

    "LST_K": "[K]", "LST_C": "[°C]",
}

# Color scheme (MetBrewer palettes)
PARAM_PALETTE_DEFAULT: Final[dict[str, str]] = {
    "SVFmin": "Archambault", "SVFmax": "Archambault", "SVFmean": "Archambault",
    "ARmin":  "Greek",       "ARmax":  "Greek",       "ARmean":  "Greek",
    "BSFmin": "VanGogh1",    "BSFmax": "VanGogh1",    "BSFmean": "VanGogh1",
    "ISFmin": "VanGogh2",    "ISFmax": "VanGogh2",    "ISFmean": "VanGogh2",
    "PSFmin": "VanGogh3",    "PSFmax": "VanGogh3",    "PSFmean": "VanGogh3",
    "TSFmin": "Hokusai3",    "TSFmax": "Hokusai3",    "TSFmean": "Hokusai3",
    "HREmin": "Hokusai2",    "HREmax": "Hokusai2",    "HREmean": "Hokusai2",
    "TRCmin": "Pissaro",     "TRCmax": "Pissaro",     "TRCmean": "Pissaro",
    "SADmin": "Tam",         "SADmax": "Tam",         "SADmean": "Tam",
    "SALmin": "Renoir",      "SALmax": "Renoir",      "SALmean": "Renoir",
    "AHmin":  "Manet",       "AHmax":  "Manet",       "AHmean":  "Manet",
    "z0":     "Troy",

    # Vegetation -> green-leaning, water -> blue-leaning, urban -> warm,
    # soil -> earthy, thermal -> hot.
    "NDVI": "Hokusai2", "EVI": "Hokusai2", "SAVI": "Hokusai2", "MSAVI": "Hokusai2",
    "GNDVI": "Hokusai2", "GRNDVI": "Hokusai2", "ARVI": "Hokusai2", "VARI": "Hokusai2",
    "CIG": "Hokusai2", "CVI": "Hokusai2", "TVI": "Hokusai2", "OSAVI": "Hokusai2",
    "RDVI": "Hokusai2", "DVI": "Hokusai2", "IPVI": "Hokusai2", "WDVI": "Hokusai2",
    "TNDVI": "Hokusai2", "UVI": "Hokusai2",

    "NDWI": "VanGogh1", "MNDWI": "VanGogh1", "NDMI": "VanGogh2", "MSI": "VanGogh1",
    "LSWI": "VanGogh2", "AWEI_NSH": "VanGogh1", "AWEI_SH": "VanGogh1", "WI": "VanGogh1",
    "NDPI": "VanGogh1", "MNDWI2": "VanGogh1", "SWI": "VanGogh2", "NDTI": "VanGogh1",

    "NDBI": "Greek", "IBI": "Greek", "EBBI": "Greek", "UI": "Greek", "BAEI": "Greek",
    "NBAI": "Greek", "BUI": "Greek", "VIBI": "Greek", "BRBA": "Greek", "NBLI": "Greek",
    "UTFVI": "Greek", "ISA": "Greek",

    "BSI": "Cassatt1", "SI": "Cassatt1", "BI": "Cassatt1", "CI": "Cassatt1",
    "RI": "Cassatt1", "SBI": "Cassatt1", "CARI": "Cassatt1", "GSI": "Cassatt1",
    "SOIL_MOISTURE": "Cassatt1", "IRON_OXIDE": "Cassatt1", "FERROUS_IRON": "Cassatt1",
    "SOIL_COMPOSITION": "Cassatt1",

    "LST_K": "Hokusai3", "LST_C": "Hokusai3",
}

PARAM_PALETTE_INCLUSIVE: Final[dict[str, str]] = {
    "SVFmin": "Archambault", "SVFmax": "Archambault", "SVFmean": "Archambault",
    "ARmin":  "Ingres",      "ARmax":  "Ingres",      "ARmean":  "Ingres",
    "BSFmin": "Cassatt1",    "BSFmax": "Cassatt1",    "BSFmean": "Cassatt1",
    "ISFmin": "Cassatt2",    "ISFmax": "Cassatt2",    "ISFmean": "Cassatt2",
    "PSFmin": "VanGogh3",    "PSFmax": "VanGogh3",    "PSFmean": "VanGogh3",
    "TSFmin": "Hokusai3",    "TSFmax": "Hokusai3",    "TSFmean": "Hokusai3",
    "HREmin": "Hokusai2",    "HREmax": "Hokusai2",    "HREmean": "Hokusai2",
    "TRCmin": "Pissaro",     "TRCmax": "Pissaro",     "TRCmean": "Pissaro",
    "SADmin": "Tam",         "SADmax": "Tam",         "SADmean": "Tam",
    "SALmin": "Renoir",      "SALmax": "Renoir",      "SALmean": "Renoir",
    "AHmin":  "Demuth",      "AHmax":  "Demuth",      "AHmean":  "Demuth",
    "z0":     "Troy",

    "NDVI": "Cassatt2", "EVI": "Cassatt2", "SAVI": "Cassatt2", "MSAVI": "Cassatt2",
    "GNDVI": "Cassatt2", "GRNDVI": "Cassatt2", "ARVI": "Cassatt2", "VARI": "Cassatt2",
    "CIG": "Cassatt2", "CVI": "Cassatt2", "TVI": "Cassatt2", "OSAVI": "Cassatt2",
    "RDVI": "Cassatt2", "DVI": "Cassatt2", "IPVI": "Cassatt2", "WDVI": "Cassatt2",
    "TNDVI": "Cassatt2", "UVI": "Cassatt2",

    "NDWI": "Ingres", "MNDWI": "Ingres", "NDMI": "Tam", "MSI": "Ingres",
    "LSWI": "Tam", "AWEI_NSH": "Ingres", "AWEI_SH": "Ingres", "WI": "Ingres",
    "NDPI": "Ingres", "MNDWI2": "Ingres", "SWI": "Tam", "NDTI": "Ingres",

    "NDBI": "Renoir", "IBI": "Renoir", "EBBI": "Renoir", "UI": "Renoir",
    "BAEI": "Renoir", "NBAI": "Renoir", "BUI": "Renoir", "VIBI": "Renoir",
    "BRBA": "Renoir", "NBLI": "Renoir", "UTFVI": "Renoir", "ISA": "Renoir",

    "BSI": "Demuth", "SI": "Demuth", "BI": "Demuth", "CI": "Demuth",
    "RI": "Demuth", "SBI": "Demuth", "CARI": "Demuth", "GSI": "Demuth",
    "SOIL_MOISTURE": "Demuth", "IRON_OXIDE": "Demuth", "FERROUS_IRON": "Demuth",
    "SOIL_COMPOSITION": "Demuth",

    "LST_K": "Manet", "LST_C": "Manet",
}


# ── Output column order (matches R `string_list`) ─────────────────────────────

LCZ_COLUMNS: Final[list[str]] = [
    "lcz_class",
    "svf_min", "svf_max", "AR_min", "AR_max",
    "BSF_min", "BSF_max", "ISF_min", "ISF_max",
    "PSF_max", "PSF_min", "TSF_min", "TSF_max",
    "HRE_min", "HRE_max", "TRC_min", "TRC_max",
    "SAD_min", "SAD_max", "SAL_min", "SAL_max",
    "AH_min", "AH_max", "z0",
    "svf_mean", "aspect_mean", "BSF_mean", "ISF_mean",
    "PSF_mean", "TSF_mean", "HRE_mean", "TRC_mean",
    "SAD_mean", "SAL_mean", "AH_mean",
    "geometry",
]


# ── Build the full table as a numpy structured view ───────────────────────────

@dataclass(frozen=True)
class LCZTable:
    """Immutable view of the LCZ parameter table.

    Use :attr:`arrays` for vectorised math and :attr:`df` for a Pandas /
    GeoPandas dataframe.
    """
    ids: np.ndarray            # (17,) int16
    codes: np.ndarray           # (17,) <U2
    names: np.ndarray          # (17,) <U24
    svf_min: np.ndarray
    svf_max: np.ndarray
    ar_min: np.ndarray
    ar_max: np.ndarray
    bsf_min: np.ndarray
    bsf_max: np.ndarray
    isf_min: np.ndarray
    isf_max: np.ndarray
    psf_max: np.ndarray
    psf_min: np.ndarray
    tsf_min: np.ndarray
    tsf_max: np.ndarray
    hre_min: np.ndarray
    hre_max: np.ndarray
    trc_min: np.ndarray
    trc_max: np.ndarray
    sad_min: np.ndarray
    sad_max: np.ndarray
    sal_min: np.ndarray
    sal_max: np.ndarray
    ah_min: np.ndarray
    ah_max: np.ndarray
    z0: np.ndarray

    def means(self) -> dict[str, np.ndarray]:
        """Vectorised mean for every parameter that has min/max pairs."""
        return {
            "SVFmean": (self.svf_min + self.svf_max) / 2,
            "ARmean":  (self.ar_min  + self.ar_max)  / 2,
            "BSFmean": (self.bsf_min + self.bsf_max) / 2,
            "ISFmean": (self.isf_min + self.isf_max) / 2,
            "PSFmean": (self.psf_min + self.psf_max) / 2,
            "TSFmean": (self.tsf_min + self.tsf_max) / 2,
            "HREmean": (self.hre_min + self.hre_max) / 2,
            "TRCmean": (self.trc_min + self.trc_max) / 2,
            "SADmean": (self.sad_min + self.sad_max) / 2,
            "SALmean": (self.sal_min + self.sal_max) / 2,
            "AHmean":  (self.ah_min  + self.ah_max)  / 2,
        }


def build_lcz_table() -> LCZTable:
    """Construct the :class:`LCZTable` from :data:`_PARAM_RANGES`."""
    return LCZTable(
        ids=np.asarray(LCZ_IDS, dtype=np.int16),
        codes=np.asarray(LCZ_CODES),
        names=np.asarray(LCZ_NAMES),
        svf_min=np.asarray(_PARAM_RANGES["SVFmin"]),
        svf_max=np.asarray(_PARAM_RANGES["SVFmax"]),
        ar_min=np.asarray(_PARAM_RANGES["ARmin"]),
        ar_max=np.asarray(_PARAM_RANGES["ARmax"]),
        bsf_min=np.asarray(_PARAM_RANGES["BSFmin"]),
        bsf_max=np.asarray(_PARAM_RANGES["BSFmax"]),
        isf_min=np.asarray(_PARAM_RANGES["ISFmin"]),
        isf_max=np.asarray(_PARAM_RANGES["ISFmax"]),
        psf_max=np.asarray(_PARAM_RANGES["PSFmax"]),
        psf_min=np.asarray(_PARAM_RANGES["PSFmin"]),
        tsf_min=np.asarray(_PARAM_RANGES["TSFmin"]),
        tsf_max=np.asarray(_PARAM_RANGES["TSFmax"]),
        hre_min=np.asarray(_PARAM_RANGES["HREmin"]),
        hre_max=np.asarray(_PARAM_RANGES["HREmax"]),
        trc_min=np.asarray(_PARAM_RANGES["TRCmin"]),
        trc_max=np.asarray(_PARAM_RANGES["TRCmax"]),
        sad_min=np.asarray(_PARAM_RANGES["SADmin"]),
        sad_max=np.asarray(_PARAM_RANGES["SADmax"]),
        sal_min=np.asarray(_PARAM_RANGES["SALmin"]),
        sal_max=np.asarray(_PARAM_RANGES["SALmax"]),
        ah_min=np.asarray(_PARAM_RANGES["AHmin"]),
        ah_max=np.asarray(_PARAM_RANGES["AHmax"]),
        z0=np.asarray([z0_for_code(c) for c in LCZ_CODES]),
    )


LCZ_TABLE: Final[LCZTable] = build_lcz_table()


__all__ = [
    "LCZ_NAMES", "LCZ_NAMES_I18N", "get_lcz_names",
    "LCZ_COLORS", "LCZ_COLORBLIND", "LCZ_CODES", "LCZ_IDS",
    "LCZ_TABLE", "LCZTable",
    "PARAM_NAMES", "PARAM_UNITS",
    "PARAM_PALETTE_DEFAULT", "PARAM_PALETTE_INCLUSIVE",
    "LCZ_COLUMNS",
    "build_lcz_table", "z0_for_code",
]