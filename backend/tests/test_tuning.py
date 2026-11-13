import numpy as np
import pandas as pd
import pytest

from app.ml import create_demo_csv, predict
from app.ml.tuning import _folds, optimize, validate_parameters
from app.ml.kaggle import prepare


def test_parameters_reject_unsafe_or_invalid_options():
    for parameters in ({"n_jobs": 100}, {"max_depth": -1}, {"max_depth": 2.5},
                       {"max_depth": True}, {"max_features": float("nan")}):
        with pytest.raises(ValueError):
            validate_parameters("RandomForest", parameters)


def test_temporal_folds_preserve_whole_years():
    frame = pd.DataFrame({"campaign": np.repeat(np.arange(2000, 2015), 3)})
    for a, b in _folds(frame, "temporal", 5):
        assert frame.iloc[a].campaign.max() < frame.iloc[b].campaign.min()


def test_nested_temporal_folds_have_enough_history():
    frame = pd.DataFrame({"campaign": np.repeat(np.arange(1990, 2008), 3)})
    for a, b in _folds(frame, "temporal", 5, min_train_years=4):
        assert frame.iloc[a].campaign.max() < frame.iloc[b].campaign.min()
        assert len(_folds(frame.iloc[a], "temporal", 3)) == 3


def test_kaggle_adapter_collapses_repeated_labels_and_checks_conflicts(tmp_path):
    import zipfile
    data = pd.DataFrame({"Area": ["A", "A", "B", "C"], "Item": ["Wheat"] * 4,
                         "Year": [2000] * 4, "hg/ha_yield": [10000] * 4,
                         "average_rain_fall_mm_per_year": [400] * 4,
                         "pesticides_tonnes": [10] * 4, "avg_temp": [10, 20, 15, 18]})
    source = tmp_path / "source.zip"
    def write_archive():
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("yield_df.csv", data.to_csv(index=False))
    write_archive()
    output = tmp_path / "converted.csv"
    result = prepare(source, output)
    converted = pd.read_csv(output)
    assert result["collapsed_rows"] == 1
    assert converted.yield_t_ha.eq(1).all()
    assert converted.iloc[0].temperature_c == 15
    assert converted.spatial_unit.eq("administrative_region").all()
    data.loc[1, "hg/ha_yield"] = 20000
    write_archive()
    with pytest.raises(ValueError, match="Conflicting"):
        prepare(source, tmp_path / "invalid.csv")


def test_nested_search_seals_test_and_produces_compatible_artifact(tmp_path):
    source = create_demo_csv(tmp_path / "demo.csv")
    result = optimize(source, tmp_path / "search", trials=1, outer_folds=2, inner_folds=2)
    test = set(result["test_indices"])
    frame = pd.read_csv(source)
    for fold in result["outer_folds"]:
        a, b = set(fold["train_indices"]), set(fold["validation_indices"])
        assert not a & b and not test & (a | b)
        assert not set(frame.iloc[list(a)].parcel_id) & set(frame.iloc[list(b)].parcel_id)
    assert result["synthetic_demo"] is True
    assert np.isfinite(predict(result["artifact_path"], frame.iloc[0].to_dict()))
    assert result["baseline_test_metrics"]["n"] == len(test)


def test_insufficient_groups_rejected_before_search(tmp_path):
    source = create_demo_csv(tmp_path / "demo.csv")
    with pytest.raises(ValueError, match="independent"):
        optimize(source, tmp_path / "search", validation="region", trials=1)
    assert not (tmp_path / "search").exists()
