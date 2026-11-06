import math

import pandas as pd
import pytest

from app.ml import categorical_domain_warnings, create_demo_csv, inspect_csv, predict, train


def test_demo_random_forest_round_trip_and_group_split(tmp_path):
    source = create_demo_csv(tmp_path / "demo.csv")
    summary = inspect_csv(source)
    assert summary["synthetic_demo"] is True
    assert "crop" in summary["feature_names"]
    result = train(source, "RandomForest", tmp_path / "artifacts", seed=9)
    assert result["synthetic_demo"] is True
    assert result["split_info"]["group_overlap"] is False
    assert result["split_info"]["train_rows"] + result["split_info"]["validation_rows"] + result["split_info"]["test_rows"] == summary["rows"]
    assert result["metrics"]["test"]["n"] > 1
    observation = pd.read_csv(source).iloc[0].to_dict()
    assert math.isfinite(predict(result["artifact_path"], observation))
    assert not categorical_domain_warnings(result["artifact_path"], observation)
    changed = {**observation, "region": "unseen_region"}
    assert any("region=unseen_region" in warning for warning in categorical_domain_warnings(result["artifact_path"], changed))


def test_invalid_csv_and_incomplete_temporal_sequence(tmp_path):
    source = create_demo_csv(tmp_path / "demo.csv")
    frame = pd.read_csv(source)
    frame["country"] = "synthetic_land"
    frame.drop(columns="ndvi_t2").to_csv(source, index=False)
    assert "country" not in inspect_csv(source)["feature_names"]
    with pytest.raises(ValueError, match="missing time steps"):
        train(source, "LSTM", tmp_path / "artifacts", epochs=1)
    frame.drop(columns="yield_t_ha").to_csv(source, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        inspect_csv(source)


def test_temporal_validation_keeps_newest_years_for_test(tmp_path):
    source = create_demo_csv(tmp_path / "temporal.csv")
    frame = pd.read_csv(source)
    frame["campaign"] = [str(2010 + index % 10) for index in range(len(frame))]
    frame.to_csv(source, index=False)
    result = train(source, "RandomForest", tmp_path / "models", validation="temporal", seed=7)
    split = result["split_info"]
    assert split["strategy"] == "chronological_campaign_holdout"
    assert max(split["train_years"]) < min(split["validation_years"])
    assert max(split["validation_years"]) < min(split["test_years"])
    assert split["group_overlap"] is False
