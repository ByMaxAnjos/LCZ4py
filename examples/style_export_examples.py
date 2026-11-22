"""Create local examples for verifying LCZ4py publication styles.

This script avoids external downloads by generating small synthetic rasters
and then calling the plotting APIs with the different style presets.
Outputs are written to LCZ4r_output/.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from LCZ4py.general.lcz_cal_area import lcz_cal_area
from LCZ4py.general.lcz_plot_map import lcz_plot_map
OUT_DIR = Path("LCZ4r_output")


def write_synthetic_lcz(path: Path) -> None:
    arr = np.array(
        [
            [1, 1, 2, 2, 6, 6],
            [1, 3, 3, 2, 6, 7],
            [4, 4, 3, 8, 7, 7],
            [5, 4, 9, 8, 8, 10],
            [11, 11, 9, 12, 13, 13],
            [11, 14, 14, 12, 15, 17],
        ],
        dtype=np.uint8,
    )
    transform = from_origin(0, 600, 100, 100)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": arr.dtype,
        "crs": "EPSG:3857",
        "transform": transform,
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, "LCZ")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        lcz_path = tmpdir / "synthetic_lcz.tif"
        write_synthetic_lcz(lcz_path)

        # 1) Default vs publication presets on the same LCZ map
        lcz_plot_map(
            lcz_path,
            title="Synthetic LCZ map",
            style="default",
            isave=True,
            save_extension="png",
            add_scalebar=True,
            add_north_arrow=True,
        )
        lcz_plot_map(
            lcz_path,
            title="Synthetic LCZ map",
            style="science",
            isave=True,
            save_extension="pdf",
            add_scalebar=True,
            add_north_arrow=True,
        )
        lcz_plot_map(
            lcz_path,
            title="Synthetic LCZ map",
            style="nature",
            isave=True,
            save_extension="png",
            add_scalebar=True,
            add_north_arrow=True,
        )

        # 2) Area chart presets
        lcz_cal_area(
            lcz_path,
            plot_type="bar",
            title="Synthetic LCZ area",
            style="generic_bw",
            isave=True,
            save_extension="png",
        )


if __name__ == "__main__":
    main()
