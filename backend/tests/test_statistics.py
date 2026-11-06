from app.ml.statistics import _holm_adjust
from app.ml.statistics import compare_models
from app.ml.pipeline import _read_dataset, _sha256
import json
import numpy as np
import pandas as pd
import pytest


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    assert _holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_comparison_reports_methods_and_rejects_mismatched_artifacts(tmp_path, monkeypatch):
    source = tmp_path / "fields.csv"
    pd.DataFrame([{"parcel_id": f"P{year}-{i}", "crop": "wheat", "region": "farm1",
                   "campaign": year, "spatial_unit": "parcel", "yield_t_ha": 5 + i / 10,
                   "ndvi_t0": .3, "ndvi_t1": .5}
                  for year in range(2016, 2023) for i in range(6)]).to_csv(source, index=False)
    _, features = _read_dataset(source)
    names = ["RandomForest", "XGBoost", "LSTM"]
    artifacts = {name: tmp_path / f"{name}.joblib" for name in names}
    for name, artifact in artifacts.items():
        artifact.with_suffix(".json").write_text(json.dumps({
            "source_sha256": _sha256(source), "validation": "temporal", "seed": 42,
            "model_type": name, "numeric_feature_names": features, "epochs": 30 if name == "LSTM" else None,
            "device": "cpu"}), encoding="utf-8")
    def estimates(artifact, observations):
        offset = (names.index(artifact.stem) + 1) / 10
        return observations.yield_t_ha.to_numpy() + offset
    monkeypatch.setattr("app.ml.statistics._batch_predict", estimates)
    result = compare_models(source, artifacts, "temporal", 42, bootstrap_repetitions=100)
    assert result["test_observations"] == result["block_count"] == 12
    assert result["independence_verified"] is False
    assert result["test_region_counts"] == {"farm1": 12}
    assert result["friedman"]["degrees_of_freedom"] == 2
    assert result["training_details"]["LSTM"]["epochs"] == 30
    assert "not an individual" in result["methods"]["bootstrap_interval_scope"]
    assert len(result["wilcoxon_holm"]) == 3
    for row in result["wilcoxon_holm"]:
        assert np.isfinite(row["statistic"])
        assert 0 <= row["p_value"] <= row["holm_p_value"] <= 1
    assert result["models"]["RandomForest"]["test"]["mae"] == pytest.approx(.1)
    assert result["models"]["RandomForest"]["mae_95pct_cluster_bootstrap_ci"] == pytest.approx([.1, .1])
    path = artifacts["LSTM"].with_suffix(".json")
    metadata = json.loads(path.read_text()); metadata["source_sha256"] = "changed"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not trained on this dataset"):
        compare_models(source, artifacts, "temporal", 42, bootstrap_repetitions=100)
