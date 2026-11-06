import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from app.ml import convert_yieldsat_to_csv, discover_yieldsat_files, inspect_yieldsat_file


def _fixture(path, *, unit="kg/ha"):
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = np.arange(6 * 2 * 2, dtype=float).reshape(6, 2, 2) / 100
    dataset = xr.Dataset(
        data_vars={
            "sample": (("index", "time_step", "band"), sample),
            "target": ("index", [3000., 3200., 4100., 3900., 5100., 4800.]),
            "crop": ("index", [1, 1, 1, 1, 2, 2]),
            "year": ("index", [2023, 2023, 2024, 2024, 2023, 2023]),
            "field_shared_name": ("index", ["f1", "f1", "f2", "f2", "f3", "f3"]),
            "farm_identifier": ("index", ["farm1", "farm1", "farm2", "farm2", "farm3", "farm3"]),
            "country": ("index", [1, 1, 2, 2, 3, 3]),
            "times": ("time_step", [10, 20]),
        },
        coords={"index": np.arange(6), "time_step": [0, 1], "band": ["NDVI", "rain mm"]},
        attrs={"synthetic_demo": "true",
               "country_code_to_name": json.dumps({"1": "Argentina", "2": "Brazil", "3": "Germany"})},
    )
    if unit is not None:
        dataset["target"].attrs["units"] = unit
    dataset["crop"].attrs["code_to_name"] = json.dumps({"1": "corn", "2": "wheat"})
    dataset.to_netcdf(path, engine="h5netcdf")
    return path


def test_discover_inspect_and_bounded_conversion(tmp_path):
    source = _fixture(tmp_path / "merged" / "synthetic.nc")
    assert discover_yieldsat_files(tmp_path) == [str(source.resolve())]
    report = inspect_yieldsat_file(tmp_path)
    assert report["valid"] is True
    assert report["dimensions"] == {"index": 6, "time_step": 2, "band": 2}
    assert any("country" in warning for warning in report["warnings"])
    output = tmp_path / "converted.csv"
    manifest = convert_yieldsat_to_csv(tmp_path, output, max_samples=4, batch_size=2)
    frame = pd.read_csv(output)
    assert len(frame) == 4
    assert frame["yield_t_ha"].iloc[0] == 3.0
    assert frame["region"].tolist() == frame["country"].tolist()
    assert frame["crop"].tolist() == ["corn"] * 4
    assert frame["parcel_id"].iloc[0] == frame["parcel_id"].iloc[1]
    assert "NDVI_t0" in frame and "rain_mm_t1" in frame
    assert frame["synthetic_demo"].all()
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["selection"] == {"start": 0, "count": 4, "method": "contiguous source index range"}
    assert json.loads(output.with_suffix(".provenance.json").read_text())["region_granularity"] == "country"


def test_explicit_units_and_output_limit(tmp_path):
    source = _fixture(tmp_path / "synthetic.nc", unit=None)
    output = tmp_path / "converted.csv"
    with pytest.raises(ValueError, match="Target units are absent"):
        convert_yieldsat_to_csv(source, output, max_samples=3)
    assert not output.exists()
    with pytest.raises(ValueError, match="Estimated CSV size"):
        convert_yieldsat_to_csv(source, output, max_samples=6, target_unit="kg/ha", max_output_bytes=100)
    result = convert_yieldsat_to_csv(source, output, max_samples=3, target_unit="kg/ha")
    assert result["source_target_unit"] == "kg/ha"
    assert len(pd.read_csv(output)) == 3


def test_cli_inspect_and_required_export_limit(tmp_path):
    source = _fixture(tmp_path / "synthetic.nc")
    result = subprocess.run([sys.executable, "-m", "app.ml", "inspect", "--source", str(source)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["valid"] is True
    result = subprocess.run([sys.executable, "-m", "app.ml", "convert", "--source", str(source),
                             "--output", str(tmp_path / "out.csv")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "--max-samples" in result.stderr


def test_official_merge_filename_discovery(tmp_path):
    source = _fixture(tmp_path / "preprocessed-24-ts" / "Germany" /
                      "merge_s2-soil-dem-weather-coords.nc")
    assert discover_yieldsat_files(tmp_path, variant="merged") == [str(source.resolve())]
