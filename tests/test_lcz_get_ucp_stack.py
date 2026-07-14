"""Self-check for UrbanParameterProcessor._save_stack (no network calls)."""

import numpy as np
import rasterio
import xarray as xr

from LCZ4py.general.lcz_get_ucp import UrbanParameterProcessor


def _fake_processor():
    processor = UrbanParameterProcessor.__new__(UrbanParameterProcessor)
    processor.target_crs = "EPSG:4326"
    return processor


def demo():
    ds = xr.Dataset(
        {
            "var_a": (("y", "x"), np.ones((4, 5), dtype="float32")),
            "var_b": (("y", "x"), np.zeros((4, 5), dtype="float32")),
        },
        coords={"y": np.arange(4), "x": np.arange(5)},
    )
    ds.rio.write_crs("EPSG:4326", inplace=True)
    ds.rio.write_transform(rasterio.transform.from_bounds(0, 0, 5, 4, 5, 4), inplace=True)

    processor = _fake_processor()
    out_path = processor._save_stack(ds)

    with rasterio.open(out_path) as src:
        assert src.count == 2, f"expected 2 bands, got {src.count}"
        assert src.descriptions == ("var_a", "var_b"), src.descriptions

    print("OK:", out_path)


if __name__ == "__main__":
    demo()
