import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from app.ml.yieldsat import audit_yieldsat_quality, convert_yieldsat_to_csv


def official_format_fixture(path):
    # The first two positions are masked, as in the user's initial inspection.
    values = np.ones((16, 4, 4), dtype=np.float32) * 1000
    values[:, :2] = np.nan
    ds = xr.Dataset(
        data_vars={
            "sample": (("index", "time_step", "band"), values),
            "target": ("index", np.linspace(3, 6, 16)),
            "crop": ("index", [0] * 8 + [1] * 8),
            "year": ("index", [0] * 8 + [1] * 8),
            "country": ("index", [0] * 16),
            "farm_identifier": ("index", [0] * 8 + [1] * 8),
            "field_shared_name": ("index", np.repeat(np.arange(4), 4)),
            "times": (("index", "time_step"), np.tile(np.array(
                ["NaT", "NaT", "2020-05-01", "2020-06-01"], dtype="datetime64[ns]"), (16, 1))),
        },
        coords={"index": np.arange(16), "time_step": np.arange(4),
                "band": ["B02", "B03", "B04", "B08"]},
    )
    ds["crop"].attrs.update({"0": "wheat", "1": "rapeseed"})
    ds["year"].attrs.update({"0": "2020", "1": "2021"})
    ds["country"].attrs["0"] = "Germany"
    ds["farm_identifier"].attrs.update({"0": "farm_A", "1": "farm_B"})
    ds["field_shared_name"].attrs.update({str(i): f"field_{i}" for i in range(4)})
    ds.to_netcdf(path, engine="h5netcdf")
    return path


def test_audit_detects_valid_later_steps_and_decodes_official_attributes(tmp_path):
    source = official_format_fixture(tmp_path / "Germany.nc")
    report = audit_yieldsat_quality(source, probe_rows=16, metadata_batch_size=3)
    assert report["metadata_rows_scanned"] == 16
    assert report["crop_pixel_counts"] == {"wheat": 8, "rapeseed": 8}
    assert report["year_pixel_counts"] == {"2020": 8, "2021": 8}
    assert report["country_pixel_counts"] == {"Germany": 16}
    assert report["farm_groups"] == 2
    assert report["field_record_groups"] == 4
    assert report["probe"]["finite_fraction_by_time_step"] == [0., 0., 1., 1.]
    assert report["probe"]["rows_with_any_complete_optical_observation"] == 16
    assert report["target"]["declared_unit"] is None
    assert any("unit" in item for item in report["warnings"])
    json.dumps(report, allow_nan=False)
    output = tmp_path / "converted.csv"
    convert_yieldsat_to_csv(source, output, max_samples=16, target_unit="t/ha", batch_size=3)
    frame = pd.read_csv(output)
    assert frame.crop.tolist() == ["wheat"] * 8 + ["rapeseed"] * 8
    assert set(frame.campaign) == {2020, 2021}
    assert frame.parcel_id.nunique() == 4
    assert set(frame.country) == {"Germany"}


def test_audit_bounds_working_memory_and_counts_invalid_targets(tmp_path):
    source = official_format_fixture(tmp_path / "Germany.nc")
    with pytest.raises(ValueError, match="reduce probe_rows"):
        audit_yieldsat_quality(source, max_probe_bytes=10)
    with pytest.raises(ValueError, match="metadata_batch_size"):
        audit_yieldsat_quality(source, metadata_batch_size=65537)
    with xr.open_dataset(source, engine="h5netcdf") as dataset:
        modified = dataset.load()
    modified["target"].values[:3] = [np.nan, -1., np.inf]
    modified.to_netcdf(source, engine="h5netcdf", mode="w")
    report = audit_yieldsat_quality(source)
    assert report["target"]["invalid_count"] == 2
    assert report["target"]["negative_count"] == 1
