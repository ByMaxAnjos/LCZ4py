"""
lcz_get_indices.py

Spectral vegetation, water, urban/built-up, soil, and thermal indices
computed from a Sentinel-2/Landsat band stack produced by
``lcz_get_planetary_computer``, already cropped to an LCZ map's footprint.

Ported from the ~75-formula registry in lcz_multi_app/lcz_parameters.py,
scoped down to what's a genuine per-pixel spectral formula computable from
already-atmospherically-corrected surface reflectance (+ Landsat's surface
temperature band). Deliberately NOT ported:

- PCA components / Tasseled Cap: need whole-image statistics or Landsat-8-
  OLI-specific coefficients not validated for Sentinel-2 — a data-driven
  decomposition, not a fixed spectral formula.
- Per-band saturation/completeness/shadow-mask "quality" flags: need raw,
  un-rescaled DN and knowledge of the original acquisition, which no longer
  exists once bands are converted to reflectance (see _REFLECTANCE_SCALE).
- BUILDING_HEIGHT_PROXY / URBAN_COMPACTNESS / ROAD_INDEX / BT /
  THERMAL_ANOMALY / THERMAL_HETEROGENEITY / HEAT_STORAGE_PROXY: arbitrary
  reflectance thresholds or focal-window heuristics in the source app, not
  established remote-sensing indices with a stable definition.
- NDRE, GVI: NDRE is coded byte-for-byte identical to NDVI in the source
  app (not a real red-edge index without a red-edge band); GVI just returns
  the raw green band, not a derived index. Both omitted as non-indices.

A few formulas ARE genuinely identical to another index already in this
registry (also true in the source app, not a porting error here) — see the
"# alias of ..." comments below.

Band-role resolution is sensor-agnostic: Sentinel-2 (B02/B03/B04/B08/B11/B12)
and Landsat (blue/green/red/nir08/swir16/swir22/lwir11) asset names both map
onto the same canonical roles (blue/green/red/nir/swir1/swir2/thermal), so
any index whose required bands are present in the input stack can be
computed regardless of which sensor's shortcut ("sentinel-2-l2a" or
"landsat") was used. ``thermal`` only exists for Landsat (Sentinel-2 has no
thermal band), and unlike the source app's simplified single-channel
brightness-temperature reimplementation, LST here reads Landsat Collection 2
Level-2's already atmospherically-corrected Surface Temperature band
directly — more accurate, and there's no raw-DN thermal-constant calibration
step to get wrong.

A caveat worth knowing before trusting a mean/std from ``lcz_cal_indices``:
normalized-difference indices (NDVI-style, denominator is a *sum* of two
bands) are numerically stable, but pure single-band *ratio* indices — MSI,
WI, VIBI, BRBA, IRON_OXIDE, FERROUS_IRON, CI, RI, GSI, NBLI, UTFVI,
SOIL_COMPOSITION — divide by one lone band (or, for RI, a cubed band), so
any pixel where that band is small (water, deep shadow) produces an
extreme value. This is inherent to how these formulas are defined in the
remote-sensing literature (and in the source app) — verified on a real
Rio de Janeiro Landsat scene: RI ranged 10..362187 while NDVI stayed a
sane -0.29..0.87 on the same scene. Prefer the median/IQR columns over the
mean for these specific indices, or pass ``max_cloud_cover``/a tighter
date range to reduce noisy pixels.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import rasterio

from LCZ4py._internal.i18n_messages import lcz_msg
from LCZ4py.general.lcz_get_planetary_computer import LCZPCResult

logger = logging.getLogger(__name__)
OUTPUT_DIR = "LCZ4r_output"

# ── Band-role aliases across sensors ──────────────────────────────────────────
_BAND_ALIASES: dict[str, tuple[str, ...]] = {
    "blue": ("B02", "blue"),
    "green": ("B03", "green"),
    "red": ("B04", "red"),
    "nir": ("B08", "nir08", "nir"),
    "swir1": ("B11", "swir16", "swir1"),
    "swir2": ("B12", "swir22", "swir2"),
    "thermal": ("lwir11", "thermal"),  # Landsat surface temperature; no Sentinel-2 equivalent
}

# DN -> true [0, 1] surface reflectance, per Planetary Computer collection id.
# Pure normalized-difference ratios (NDVI, NDWI, NDBI, BSI, ...) would be
# scale-invariant on their own — but Landsat's offset is non-zero, which
# breaks that invariance for ratios too, and EVI/SAVI/BAEI/IBI bake in
# additive constants (e.g. EVI's "+1", SAVI's "L=0.5") calibrated for true
# reflectance — computing them on raw DN (Sentinel-2: 0-10000, Landsat:
# ~7000-65000) makes EVI's denominator swing through zero and blow up.
# Verified against a real Sentinel-2 scene over Rio de Janeiro: EVI ranged
# -1576..1956 on raw DN vs. a sane -1..1-ish range once scaled.
#
# The thermal band uses a completely different scale (-> Kelvin directly,
# per the official Landsat Collection 2 Level-2 Science Product Guide), so
# scaling is applied per band-role, not uniformly per collection.
_REFLECTANCE_SCALE: dict[str, tuple[float, float]] = {
    "sentinel-2-l2a": (0.0001, 0.0),
    "landsat-c2-l2": (0.0000275, -0.2),
}
_THERMAL_SCALE: dict[str, tuple[float, float]] = {
    "landsat-c2-l2": (0.00341802, 149.0),  # DN -> Kelvin
}


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise division that returns NaN instead of raising/inf on 0/0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return np.where(np.isfinite(result), result, np.nan)


def _safe_sqrt(x: np.ndarray) -> np.ndarray:
    """sqrt() that returns NaN instead of a runtime warning on negative input."""
    return np.sqrt(np.clip(x, 0, None))


# ── Index formulas: name -> (required band roles, formula(bands_dict)) ───────
# Grouped by category (comment headers) to mirror lcz_multi_app/lcz_parameters.py.
INDEX_FORMULAS: dict[str, tuple[tuple[str, ...], Callable[[dict[str, np.ndarray]], np.ndarray]]] = {

    # ── Vegetation ────────────────────────────────────────────────────────────
    "NDVI": (("nir", "red"), lambda b: _safe_divide(b["nir"] - b["red"], b["nir"] + b["red"])),
    "EVI": (
        ("nir", "red", "blue"),
        lambda b: 2.5 * _safe_divide(b["nir"] - b["red"], b["nir"] + 6 * b["red"] - 7.5 * b["blue"] + 1),
    ),
    "SAVI": (("nir", "red"), lambda b: 1.5 * _safe_divide(b["nir"] - b["red"], b["nir"] + b["red"] + 0.5)),
    "MSAVI": (
        ("nir", "red"),
        lambda b: 0.5 * (2 * b["nir"] + 1 - _safe_sqrt((2 * b["nir"] + 1) ** 2 - 8 * (b["nir"] - b["red"]))),
    ),
    "GNDVI": (("nir", "green"), lambda b: _safe_divide(b["nir"] - b["green"], b["nir"] + b["green"])),
    "GRNDVI": (
        ("nir", "green", "red"),
        lambda b: _safe_divide(b["nir"] - (b["green"] + b["red"]), b["nir"] + (b["green"] + b["red"])),
    ),
    "ARVI": (
        ("nir", "red", "blue"),
        lambda b: _safe_divide(
            b["nir"] - (b["red"] - 2 * (b["blue"] - b["red"])),
            b["nir"] + (b["red"] - 2 * (b["blue"] - b["red"])),
        ),
    ),
    "VARI": (
        ("green", "red", "blue"),
        lambda b: _safe_divide(b["green"] - b["red"], b["green"] + b["red"] - b["blue"]),
    ),
    "CIG": (("nir", "green"), lambda b: _safe_divide(b["nir"], b["green"]) - 1),
    "CVI": (("nir", "red", "green"), lambda b: b["nir"] * _safe_divide(b["red"], b["green"] ** 2)),
    "TVI": (
        ("nir", "red", "green"),
        lambda b: 0.5 * (120 * (b["nir"] - b["green"]) - 200 * (b["red"] - b["green"])),
    ),
    "OSAVI": (("nir", "red"), lambda b: _safe_divide(b["nir"] - b["red"], b["nir"] + b["red"] + 0.16)),
    "RDVI": (("nir", "red"), lambda b: _safe_divide(b["nir"] - b["red"], _safe_sqrt(b["nir"] + b["red"]))),
    "DVI": (("nir", "red"), lambda b: b["nir"] - b["red"]),
    "IPVI": (("nir", "red"), lambda b: _safe_divide(b["nir"], b["nir"] + b["red"])),
    "WDVI": (("nir", "red"), lambda b: b["nir"] - b["red"]),  # alias of DVI (soil-line slope=1.0, per source)
    "TNDVI": (
        ("nir", "red"),
        lambda b: _safe_sqrt(_safe_divide(b["nir"] - b["red"], b["nir"] + b["red"]) + 0.5),
    ),
    "UVI": (
        ("green", "red", "nir"),
        lambda b: _safe_divide(b["green"] * b["nir"], b["red"] + b["green"] + b["nir"]),
    ),

    # ── Water / moisture ──────────────────────────────────────────────────────
    "NDWI": (("green", "nir"), lambda b: _safe_divide(b["green"] - b["nir"], b["green"] + b["nir"])),
    "MNDWI": (("green", "swir1"), lambda b: _safe_divide(b["green"] - b["swir1"], b["green"] + b["swir1"])),
    "NDMI": (("nir", "swir1"), lambda b: _safe_divide(b["nir"] - b["swir1"], b["nir"] + b["swir1"])),
    "MSI": (("nir", "swir1"), lambda b: _safe_divide(b["swir1"], b["nir"])),
    "LSWI": (("nir", "swir1"), lambda b: _safe_divide(b["nir"] - b["swir1"], b["nir"] + b["swir1"])),  # alias of NDMI
    "AWEI_NSH": (
        ("green", "nir", "swir1", "swir2"),
        lambda b: 4 * (b["green"] - b["swir1"]) - (0.25 * b["nir"] + 2.75 * b["swir2"]),
    ),
    "AWEI_SH": (
        ("blue", "green", "nir", "swir1", "swir2"),
        lambda b: b["blue"] + 2.5 * b["green"] - 1.5 * (b["nir"] + b["swir1"]) - 0.25 * b["swir2"],
    ),
    "WI": (("green", "nir"), lambda b: _safe_divide(b["green"], b["nir"])),
    "NDPI": (("swir1", "green"), lambda b: _safe_divide(b["swir1"] - b["green"], b["swir1"] + b["green"])),
    "MNDWI2": (("green", "swir2"), lambda b: _safe_divide(b["green"] - b["swir2"], b["green"] + b["swir2"])),
    "SWI": (("nir", "swir1"), lambda b: (b["nir"] + b["swir1"]) / 2),
    "NDTI": (("red", "green"), lambda b: _safe_divide(b["red"] - b["green"], b["red"] + b["green"])),

    # ── Urban / built-up ──────────────────────────────────────────────────────
    "NDBI": (("swir1", "nir"), lambda b: _safe_divide(b["swir1"] - b["nir"], b["swir1"] + b["nir"])),
    "IBI": (
        ("green", "red", "nir", "swir1"),
        lambda b: _safe_divide(
            2 * _safe_divide(b["swir1"], b["swir1"] + b["nir"])
            - (_safe_divide(b["nir"], b["nir"] + b["red"]) + _safe_divide(b["green"], b["green"] + b["swir1"])),
            2 * _safe_divide(b["swir1"], b["swir1"] + b["nir"])
            + (_safe_divide(b["nir"], b["nir"] + b["red"]) + _safe_divide(b["green"], b["green"] + b["swir1"])),
        ),
    ),
    "EBBI": (
        ("swir1", "nir", "thermal"),
        lambda b: _safe_divide(b["swir1"] - b["nir"], 10 * _safe_sqrt(b["swir1"] + b["thermal"])),
    ),
    "UI": (("swir2", "nir"), lambda b: _safe_divide(b["swir2"] - b["nir"], b["swir2"] + b["nir"])),
    "BAEI": (("red", "green", "swir1"), lambda b: _safe_divide(b["red"] + 0.3, b["green"] + b["swir1"])),
    "NBAI": (
        ("swir1", "nir", "red"),
        lambda b: _safe_divide(b["swir1"] - _safe_divide(b["nir"], b["red"]), b["swir1"] + _safe_divide(b["nir"], b["red"])),
    ),
    "BUI": (("red", "swir1"), lambda b: _safe_divide(b["red"] - b["swir1"], b["red"] + b["swir1"])),
    "VIBI": (("red", "nir"), lambda b: _safe_divide(b["red"], b["nir"])),
    "BRBA": (("swir1", "nir"), lambda b: _safe_divide(b["swir1"], b["nir"])),
    "NBLI": (("red", "swir1", "nir"), lambda b: _safe_divide(b["red"] * b["swir1"], b["nir"])),
    "UTFVI": (
        ("thermal", "nir", "red"),
        lambda b: _safe_divide(b["thermal"], _safe_divide(b["nir"] - b["red"], b["nir"] + b["red"])),
    ),
    "ISA": (
        ("red", "nir", "swir1"),
        lambda b: np.clip((
            _safe_divide(b["swir1"] - b["nir"], b["swir1"] + b["nir"])
            - _safe_divide(b["nir"] - b["red"], b["nir"] + b["red"]) + 1
        ) / 2, 0, 1),
    ),

    # ── Soil ──────────────────────────────────────────────────────────────────
    "BSI": (
        ("swir1", "red", "nir", "blue"),
        lambda b: _safe_divide(
            (b["swir1"] + b["red"]) - (b["nir"] + b["blue"]),
            (b["swir1"] + b["red"]) + (b["nir"] + b["blue"]),
        ),
    ),
    "SI": (("red", "swir1"), lambda b: _safe_sqrt(b["red"] * b["swir1"])),
    "BI": (("red", "nir"), lambda b: _safe_sqrt((b["red"] ** 2 + b["nir"] ** 2) / 2)),
    "CI": (("red", "blue"), lambda b: _safe_divide(b["red"] - b["blue"], b["red"])),
    "RI": (("red", "blue", "green"), lambda b: _safe_divide(b["red"] ** 2, b["blue"] * b["green"] ** 3)),
    "SBI": (
        ("blue", "green", "red", "nir"),
        lambda b: 0.3 * b["blue"] + 0.3 * b["green"] + 0.3 * b["red"] + 0.1 * b["nir"],
    ),
    "CARI": (
        ("blue", "green", "red"),
        lambda b: np.abs((b["green"] - b["red"]) - 0.2 * (b["green"] - b["blue"])) * _safe_divide(b["red"], b["green"]),
    ),
    "GSI": (("red", "nir", "swir1"), lambda b: _safe_divide(b["red"] + b["swir1"], b["nir"])),
    "SOIL_MOISTURE": (("nir", "swir1"), lambda b: _safe_divide(b["nir"] - b["swir1"], b["nir"] + b["swir1"])),  # alias of NDMI
    "IRON_OXIDE": (("red", "blue"), lambda b: _safe_divide(b["red"], b["blue"])),
    "FERROUS_IRON": (("swir1", "nir"), lambda b: _safe_divide(b["swir1"], b["nir"])),  # alias of MSI
    "SOIL_COMPOSITION": (("swir1", "red", "nir"), lambda b: _safe_divide(b["swir1"] * b["red"], b["nir"])),

    # ── Thermal (Landsat only — Sentinel-2 has no thermal band) ─────────────
    "LST_K": (("thermal",), lambda b: b["thermal"]),
    "LST_C": (("thermal",), lambda b: b["thermal"] - 273.15),
}


@dataclass
class LCZIndicesResult:
    """Return type for lcz_get_indices."""
    path: Optional[str] = None
    array: Optional[np.ndarray] = None
    indices: Optional[list[str]] = None
    collection: Optional[str] = None
    gdf: Optional[object] = None
    geoarrow_table: Optional[object] = None


def _resolve_band_roles(band_names: list[str]) -> dict[str, int]:
    """Map canonical roles (red, nir, ...) to 0-based positions in ``band_names``."""
    roles: dict[str, int] = {}
    for role, aliases in _BAND_ALIASES.items():
        for i, name in enumerate(band_names):
            if name in aliases:
                roles[role] = i
                break
    return roles


def _load_stack(x: Union[str, Path, LCZPCResult]) -> tuple[np.ndarray, list[str], dict, str]:
    """Return (array, band_names, profile, collection) read from disk.

    Always re-reads from the on-disk GeoTIFF (rather than trusting an
    in-memory ``LCZPCResult.array``) so crs/transform/nodata are guaranteed
    consistent with the written file.
    """
    if isinstance(x, LCZPCResult):
        if not x.path or not os.path.exists(x.path):
            raise ValueError(
                "LCZPCResult has no on-disk raster to read (path is None or "
                "missing) — call lcz_get_planetary_computer with cache=True "
                "(the default) or isave=True, or pass a raster path directly."
            )
        x = x.path

    with rasterio.open(str(x)) as src:
        array = src.read()
        band_names = [src.descriptions[i] or f"band_{i + 1}" for i in range(src.count)]
        profile = src.profile.copy()
        collection = src.tags().get("collection", "")

    return array, band_names, profile, collection


def _write_indices_stack(
    path: str, stack: np.ndarray, names: list[str], profile: dict, collection: str,
) -> None:
    out_profile = profile.copy()
    out_profile.update(
        count=stack.shape[0], dtype="float32", nodata=np.nan,
        compress="lzw", predictor=3, BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(stack)
        dst.update_tags(collection=collection or "", indices=",".join(names))
        for i, name in enumerate(names, start=1):
            dst.set_band_description(i, name)


# ── Public API ────────────────────────────────────────────────────────────────

def lcz_get_indices(
    x: Union[str, Path, LCZPCResult],
    indices: Optional[list[str]] = None,
    isave: bool = False,
    lang: str = "en",
    verbose: bool = True,
) -> LCZIndicesResult:
    """Compute spectral indices from a Sentinel-2/Landsat band stack.

    Reads the multi-band GeoTIFF produced by ``lcz_get_planetary_computer``
    (or its ``LCZPCResult``), resolves band names to canonical roles
    (red/green/blue/nir/swir1/swir2) regardless of sensor naming, and
    computes every requested index whose required bands are present.

    Parameters
    ----------
    x : str, Path, or LCZPCResult
        Path to the GeoTIFF written by ``lcz_get_planetary_computer``, or
        its returned ``LCZPCResult`` (must have a valid ``.path``, i.e.
        ``cache=True`` (default) or ``isave=True`` was used).
    indices : list of str, optional
        Index names to compute — see ``INDEX_FORMULAS`` for the full list
        (vegetation: NDVI/EVI/SAVI/MSAVI/GNDVI/GRNDVI/ARVI/VARI/CIG/CVI/TVI/
        OSAVI/RDVI/DVI/IPVI/WDVI/TNDVI/UVI; water: NDWI/MNDWI/NDMI/MSI/LSWI/
        AWEI_NSH/AWEI_SH/WI/NDPI/MNDWI2/SWI/NDTI; urban: NDBI/IBI/EBBI/UI/
        BAEI/NBAI/BUI/VIBI/BRBA/NBLI/UTFVI/ISA; soil: BSI/SI/BI/CI/RI/SBI/
        CARI/GSI/SOIL_MOISTURE/IRON_OXIDE/FERROUS_IRON/SOIL_COMPOSITION;
        thermal (Landsat only): LST_K/LST_C). Defaults to every index
        computable from the bands present in ``x``.
    isave : bool
        Also write the stack to ``LCZ4r_output/lcz_indices_<collection>.tif``.
    lang : str
        Message language ("en"/"pt"/"es"/"zh").
    verbose : bool
        Log each computed index.

    Returns
    -------
    LCZIndicesResult
        ``array`` is (n_indices, H, W) float32, NaN wherever the source
        stack was NaN (already limited to the LCZ map's footprint);
        ``indices[i]`` names the index for band ``i``, in the same
        pixel grid (crs/transform/shape) as the input stack.

    Examples
    --------
    >>> from LCZ4py.general.lcz_get_planetary_computer import lcz_get_planetary_computer
    >>> s2 = lcz_get_planetary_computer(
    ...     "lcz_map.tif", collection="sentinel-2-l2a",
    ...     start_date="2024-06-01", end_date="2024-08-31",
    ...     assets=["B04", "B03", "B02", "B08", "B11"],  # +B11 unlocks NDBI/MNDWI
    ... )
    >>> idx = lcz_get_indices(s2, indices=["NDVI", "NDBI", "MNDWI"])
    >>> idx.indices
    ['NDVI', 'NDBI', 'MNDWI']
    """
    array, band_names, profile, collection = _load_stack(x)
    roles = _resolve_band_roles(band_names)

    available = {
        name for name, (reqs, _fn) in INDEX_FORMULAS.items()
        if all(r in roles for r in reqs)
    }
    if not available:
        raise ValueError(lcz_msg("indices_no_bands", lang, bands=", ".join(band_names)))

    if indices is None:
        requested = sorted(available, key=list(INDEX_FORMULAS).index)
    else:
        requested = list(indices)
        unknown = [n for n in requested if n not in INDEX_FORMULAS]
        if unknown:
            raise ValueError(lcz_msg(
                "indices_unknown", lang,
                unknown=", ".join(unknown), known=", ".join(INDEX_FORMULAS),
            ))
        unavailable = [n for n in requested if n not in available]
        if unavailable:
            missing_bands = sorted({
                r for n in unavailable for r in INDEX_FORMULAS[n][0] if r not in roles
            })
            raise ValueError(lcz_msg(
                "indices_missing_bands", lang,
                indices=", ".join(unavailable), bands=", ".join(missing_bands),
            ))

    scale, offset = _REFLECTANCE_SCALE.get(collection, (1.0, 0.0))
    tscale, toffset = _THERMAL_SCALE.get(collection, (1.0, 0.0))
    bands = {}
    for role, i in roles.items():
        if role == "thermal":
            bands[role] = array[i].astype(np.float32) * tscale + toffset
        else:
            rescaled = array[i].astype(np.float32) * scale + offset
            # Surface reflectance is physically >= 0; atmospheric correction
            # occasionally pushes a few pixels (deep water/shadow) slightly
            # negative — treat that as missing data rather than a real value.
            # (Tried flooring to a small positive epsilon instead of NaN-ing:
            # made things worse — on real Rio Landsat data it inflated RI's
            # max from 5656 to 362187, because formulas like RI divide by a
            # cubed band and are already extremely sensitive to any small
            # positive denominator, not specifically to negative ones. That
            # sensitivity is inherent to how RI/IRON_OXIDE/UTFVI/etc. are
            # defined in the literature — see the module docstring.)
            bands[role] = np.where(rescaled < 0, np.nan, rescaled)

    out_bands = []
    for name in requested:
        _, fn = INDEX_FORMULAS[name]
        out_bands.append(fn(bands).astype(np.float32))
        if verbose:
            logger.info(lcz_msg("indices_computed", lang, name=name))

    stack = np.stack(out_bands).astype(np.float32)

    result_path = None
    if isave:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        safe_collection = (collection or "indices").replace("/", "_")
        result_path = os.path.join(OUTPUT_DIR, f"lcz_indices_{safe_collection}.tif")
        _write_indices_stack(result_path, stack, requested, profile, collection)
        if verbose:
            logger.info("Saved: %s", os.path.abspath(result_path))

    return LCZIndicesResult(
        path=result_path, array=stack, indices=requested, collection=collection,
    )


__all__ = ["lcz_get_indices", "LCZIndicesResult", "INDEX_FORMULAS"]
