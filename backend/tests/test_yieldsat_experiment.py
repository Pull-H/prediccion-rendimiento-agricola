import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from app.ml.pipeline import inspect_csv
from app.ml.yieldsat_experiment import build_field_experiment


def fixture(path):
    bands = ["B02", "B03", "B04", "B08", "clay_0-5", "phh2o_0-5", "soc_0-5", "temp_mean"]
    sample = np.ones((8, 6, len(bands)), dtype=np.float32)
    for step, value in enumerate([10, 20, 30, 40, 50, 999999]):
        sample[:, step, :4] = value
    sample[:2, 1, :4] = np.nan
    sample[:, :, 4:7] = [120, 65, 240]
    sample[:, :, 7] = 14591
    start = np.datetime64("2018-01-01", "ns")
    times = np.tile(start + np.array([10, 70, 130, 190, 240, 241]).astype("timedelta64[D]"), (8, 1))
    dataset = xr.Dataset(
        data_vars={
            "sample": (("index", "time_step", "band"), sample),
            "times": (("index", "time_step"), times),
            "target": ("index", [2., 4., 4., 6., 6., 8., 8., 10.]),
            "crop": ("index", [0]*8), "year": ("index", [0]*8),
            "country": ("index", [0]*8), "farm_identifier": ("index", [0]*8),
            "field_shared_name": ("index", np.repeat(np.arange(4), 2)),
            "seeding_date": ("index", [0]*8), "harvesting_date": ("index", [0]*8),
            "seeding_date_type": ("index", [0]*6+[1]*2),
        }, coords={"index": np.arange(8), "time_step": np.arange(6), "band": bands},
    )
    for name, mapping in {
        "crop": {"0": "wheat"}, "year": {"0": "2018"}, "country": {"0": "Germany"},
        "farm_identifier": {"0": "farm1"}, "field_shared_name": {str(i):f"field{i}" for i in range(4)},
        "seeding_date": {"0": "2018-01-01"}, "harvesting_date": {"0": "2018-11-01"},
        "seeding_date_type": {"0": "provided_by_farmer", "1": "290 days before harvest"},
    }.items():
        dataset[name].attrs.update(mapping)
    dataset.to_netcdf(path, engine="h5netcdf")
    return path


def test_complete_fields_causal_cutoff_and_missing_observation_channels(tmp_path):
    source = fixture(tmp_path / "Germany.nc")
    output = tmp_path / "fields.csv"
    manifest = build_field_experiment(source, output, crop="wheat", target_unit="t/ha",
                                      target_unit_evidence="official tutorial", batch_size=3)
    frame = pd.read_csv(output)
    assert manifest["candidate_field_records"] == 4
    assert manifest["output_rows"] == 3
    assert frame.yield_t_ha.tolist() == [3., 5., 7.]
    assert all(item["pixels"] == 2 for item in manifest["field_provenance"])
    assert frame["B02_source_t3"].tolist() == [45., 45., 45.]  # includes day 240, excludes 241
    assert frame["B02_source_t1"].tolist() == [0., 20., 20.]
    assert frame["B02_observations_t1"].tolist() == [0., 1., 1.]
    assert not any("temp_mean" in name for name in frame)
    assert not any("harvest" in name or "sowing" in name for name in frame)
    assert manifest["excluded_fields"][0]["field"] == "field3"
    assert "farmer-reported" in manifest["excluded_fields"][0]["reasons"][0]
    assert manifest["source_target_declared_unit"] is None
    assert inspect_csv(output)["spatial_unit"] == "parcel"
    json.dumps(manifest, allow_nan=False)


def test_cutoff_must_precede_harvest_and_unknown_units_are_not_guessed(tmp_path):
    source = fixture(tmp_path / "Germany.nc")
    output = tmp_path / "fields.csv"
    with pytest.raises(ValueError, match="target_unit_evidence"):
        build_field_experiment(source, output, target_unit="t/ha", target_unit_evidence="")
    report = build_field_experiment(source, output, target_unit="t/ha", target_unit_evidence="tutorial",
                                    horizon_days=400, batch_size=4)
    assert report["output_rows"] == 0
    assert not output.exists()
    assert output.with_suffix(".provenance.json").is_file()
    assert all(any("harvested" in why for why in row["reasons"]) for row in report["excluded_fields"])
