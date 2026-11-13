import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app.ml import create_demo_csv, predict, train
from app.ml.analysis import context, evaluate, explain, infer_many, permutation
from app.ml.tuning import optimize, TrainingCancelled
from app.ml.figures import figure_bytes
from app.research_reports import render_report


@pytest.fixture
def fitted(tmp_path):
    source = create_demo_csv(tmp_path / "demo.csv")
    result = train(source, "RandomForest", tmp_path / "model")
    return source, result


def test_evaluation_and_calibration_are_disjoint(fitted):
    source, result = fitted
    _, _, frame, training, calibration, test = context(result["artifact_path"], source)
    assert not set(training) & set(calibration) and not set(test) & set(calibration)
    report = evaluate(result["artifact_path"], source, repetitions=100)
    assert report["metrics"] == pytest.approx(result["metrics"]["test"])
    assert report["interval"]["available"]
    assert 0 <= report["interval"]["test_coverage"] <= 1
    assert report["confidence"]["blocks"] == frame.iloc[test].parcel_id.nunique()
    assert len(report["points"]) == len(test)
    # Changing test outcomes cannot change the calibration radius.
    from app.ml.analysis import interval_info
    predictor, metadata, frame, _, calibration, test = context(result["artifact_path"], source)
    before = interval_info(predictor, metadata, frame, calibration)
    frame.loc[test, "yield_t_ha"] += 100
    assert interval_info(predictor, metadata, frame, calibration)["radius"] == pytest.approx(before["radius"])


def test_local_shapley_additivity_and_batch_domain(fitted):
    source, result = fitted
    row = pd.read_csv(source).iloc[0][result["feature_names"]].to_dict()
    attribution = explain(result["artifact_path"], source, row, permutations=2, backgrounds=2)
    assert abs(attribution["additivity_error"]) < 1e-6
    assert attribution["prediction"] == pytest.approx(predict(result["artifact_path"], row))
    changed = {**row, "soil_ph": 999, "region": "unseen"}
    batch = infer_many(result["artifact_path"], source, [row, changed])
    assert len(batch["predictions"]) == 2
    assert batch["predictions"][0]["interval"] is not None
    assert any("soil_ph" in w for w in batch["predictions"][1]["warnings"])
    assert any("region" in w for w in batch["predictions"][1]["warnings"])
    with pytest.raises(ValueError, match="finite"):
        infer_many(result["artifact_path"], source, [{**row, "soil_ph": float("inf")}])
    assert permutation(result["artifact_path"], source, repeats=1, max_rows=20)["n"] == 20


def test_exports_are_readable_and_figures_have_600_dpi(fitted):
    source, result = fitted
    report = evaluate(result["artifact_path"], source, repetitions=100)
    report.update({"importance": permutation(result["artifact_path"], source, repeats=1, max_rows=20),
                   "model": {"id": 1, "type": "RandomForest"},
                   "dataset": {"name": "=FORMULA()", "source": "synthetic", "license": "test", "synthetic": True}})
    image = Image.open(BytesIO(figure_bytes(report)))
    assert image.width == 4200 and image.height == 2700
    assert image.info["dpi"][0] == pytest.approx(600, abs=1)
    assert json.loads(render_report(report, "json"))["metrics"] == report["metrics"]
    assert render_report(report, "pdf").startswith(b"%PDF")
    from docx import Document
    assert Document(BytesIO(render_report(report, "docx"))).inline_shapes
    from openpyxl import load_workbook
    book = load_workbook(BytesIO(render_report(report, "xlsx")))
    assert len(book["Predicciones prueba"]["A"]) == len(report["points"]) + 1
    assert all(cell.data_type != "f" for sheet in book for row in sheet for cell in row)


def test_cancel_resume_preserves_trials_and_rejects_configuration_change(tmp_path):
    source = create_demo_csv(tmp_path / "demo.csv")
    output = tmp_path / "search"
    state = {"cancel": False}
    def progress(_):
        state["cancel"] = True
    with pytest.raises(TrainingCancelled):
        optimize(source, output, trials=2, outer_folds=2, inner_folds=2,
                 progress=progress, cancelled=lambda: state["cancel"])
    with pytest.raises(ValueError, match="configuration"):
        optimize(source, output, trials=3, outer_folds=2, inner_folds=2, resume=True)
    result = optimize(source, output, trials=2, outer_folds=2, inner_folds=2, resume=True)
    assert result["calibration_reserved"]
    _, _, _, train_idx, calibration, test = context(result["artifact_path"], source)
    assert len(calibration) > 0
    for fold in result["outer_folds"]:
        assert not set(calibration) & set(fold["train_indices"] + fold["validation_indices"])
        assert not set(test) & set(fold["train_indices"] + fold["validation_indices"])
    # A completed resume returns the exact report without running or re-evaluating.
    again = optimize(source, output, trials=2, outer_folds=2, inner_folds=2, resume=True)
    assert again == result


@pytest.mark.parametrize("kind", ["LSTM", "CNN-LSTM", "CNN-Transformer"])
def test_temporal_models_support_configured_training_and_explanation(tmp_path, kind):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    source = create_demo_csv(tmp_path / "demo.csv")
    result = train(source, kind, tmp_path / kind, epochs=1,
                   hyperparameters={"hidden": 16, "batch_size": 32, "learning_rate": .001})
    row = pd.read_csv(source).iloc[0][result["feature_names"]].to_dict()
    attribution = explain(result["artifact_path"], source, row, permutations=1, backgrounds=1)
    assert abs(attribution["additivity_error"]) < 1e-5
    assert np.isfinite(attribution["prediction"])
    assert evaluate(result["artifact_path"], source, repetitions=100)["interval"]["available"]


@pytest.mark.parametrize("kind", ["LSTM", "CNN-LSTM", "CNN-Transformer"])
def test_nested_search_for_temporal_models(tmp_path, kind):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    source = create_demo_csv(tmp_path / "demo.csv")
    result = optimize(source, tmp_path / "search", model_type=kind, trials=1,
                      outer_folds=2, inner_folds=2,
                      search_ranges={"epochs": [1, 1], "hidden": [16, 16], "batch_size": [32, 32]})
    assert result["calibration_reserved"]
    row = pd.read_csv(source).iloc[0][result["feature_names"]].to_dict()
    assert np.isfinite(predict(result["artifact_path"], row))
