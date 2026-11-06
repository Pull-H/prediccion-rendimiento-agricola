"""Extract frozen model evidence inside the backend container; never retrain.

Copy this file into /tmp and execute with --output /app/data/manuscript_evidence.json.
Permutation analysis is post hoc on validation only and does not select models.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Dataset, Experiment, ModelVersion
from app.ml.pipeline import CATEGORIES, DEEP_MODELS, _deep_arrays, _metrics, _read_dataset, _sha256, _split
from app.ml.statistics import compare_models


def predictor(artifact):
    payload = joblib.load(artifact)
    columns = payload["features"] + payload["categories"]
    if payload["model_type"] in DEEP_MODELS:
        import torch
        from app.ml.deep import TemporalRegressor
        network = TemporalRegressor(payload["model_type"], **payload["model_config"])
        network.load_state_dict(payload["state_dict"])
        network.eval()
        def estimate(frame):
            matrix = payload["preprocessor"].transform(frame[columns])
            seq, static = _deep_arrays(matrix, payload["features"], payload["sequence_schema"])
            with torch.no_grad():
                return network(torch.from_numpy(seq), torch.from_numpy(static)).numpy().astype(float)
    else:
        def estimate(frame):
            matrix = payload["preprocessor"].transform(frame[columns])
            return np.asarray(payload["model"].predict(matrix), dtype=float)
    return payload, estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with SessionLocal() as db:
        dataset = db.get(Dataset, 9)
        assert dataset is not None and not dataset.synthetic and dataset.source.startswith("YieldSAT")
        frame, numeric = _read_dataset(dataset.file_path)
        train, validation, test, split = _split(frame, "temporal", 42)
        assert [len(train), len(validation), len(test)] == [132, 15, 37]
        expected_sha = "fa9496d9598896042152d3bd669edd1290384630f0e880d79fdde483aee0e2e1"
        assert _sha256(dataset.file_path) == expected_sha
        names = ("RandomForest", "XGBoost", "LSTM", "CNN-LSTM", "CNN-Transformer")
        models = {name: db.get(ModelVersion, identifier) for name, identifier in zip(names, range(14, 19))}
        assert all(model and model.model_type == name for name, model in models.items())
        output = {"dataset_id": dataset.id, "dataset_sha256": expected_sha, "split": split,
                  "model_ids": {name: model.id for name, model in models.items()},
                  "model_details": {}, "predictions": {}, "metrics": {}, "strata": {}, "permutation": {}}
        groups = {f"Window_{i+1}": [name for name in numeric if name.endswith(f"_t{i}")] for i in range(4)}
        groups["Soil"] = [name for name in numeric if "_t" not in name and not name.startswith("dem_")]
        groups["DEM"] = [name for name in numeric if name.startswith("dem_")]
        groups["Context"] = list(CATEGORIES)
        assert set(sum(groups.values(), [])) == set(numeric + list(CATEGORIES))
        assert sum(map(len, groups.values())) == len(numeric) + len(CATEGORIES)
        role = {int(i): label for label, indices in (("train", train), ("validation", validation), ("test", test)) for i in indices}
        output["records"] = [{"row": int(i), "campaign": row.campaign, "region": row.region,
                              "role": role[int(i)], "yield_t_ha": float(row.yield_t_ha)} for i, row in frame.iterrows()]
        train_mean = float(frame.iloc[train].yield_t_ha.mean())
        mean_predictions = np.full(len(frame), train_mean)
        output["predictions"]["TrainingMean"] = mean_predictions.tolist()
        output["metrics"]["TrainingMean"] = {label: _metrics(frame.iloc[ix].yield_t_ha, mean_predictions[ix])
            for label, ix in (("train", train), ("validation", validation), ("test", test))}
        output["baseline_value"] = train_mean
        for name, model in models.items():
            metadata = json.loads(Path(model.artifact_path).with_suffix(".json").read_text())
            assert metadata["source_sha256"] == expected_sha and metadata["seed"] == 42
            experiment = db.get(Experiment, model.experiment_id)
            assert experiment.dataset_id == dataset.id and experiment.validation == "temporal"
            payload, estimate = predictor(model.artifact_path)
            predictions = estimate(frame)
            output["predictions"][name] = predictions.tolist()
            output["metrics"][name] = {label: _metrics(frame.iloc[ix].yield_t_ha, predictions[ix])
                for label, ix in (("train", train), ("validation", validation), ("test", test))}
            assert abs(output["metrics"][name]["test"]["mae"] - model.metrics["test"]["mae"]) < 1e-6
            details = {key: metadata.get(key) for key in ("epochs", "device", "training_seconds", "seed", "validation")}
            details.update({"model_id": model.id, "experiment_id": experiment.id,
                            "artifact_sha256": _sha256(model.artifact_path),
                            "model_config": payload.get("model_config"),
                            "sequence_schema": payload.get("sequence_schema")})
            if name not in DEEP_MODELS:
                keys = ("n_estimators", "max_depth", "min_samples_leaf", "learning_rate", "subsample", "colsample_bytree", "random_state", "n_jobs", "tree_method", "objective")
                params = payload["model"].get_params()
                details["parameters"] = {key: params[key] for key in keys if key in params}
            output["model_details"][name] = details
            holdout = frame.iloc[validation].copy()
            reference_mae = float(np.mean(np.abs(holdout.yield_t_ha.to_numpy() - estimate(holdout))))
            permutations = {}
            # Same sequence of permutations for every model and group; no fitting or tuning.
            for group, columns in groups.items():
                rng = np.random.default_rng(20261106)
                changes = []
                for _ in range(30):
                    altered = holdout.copy()
                    altered.loc[:, columns] = holdout.iloc[rng.permutation(len(holdout))][columns].to_numpy()
                    changes.append(float(np.mean(np.abs(holdout.yield_t_ha.to_numpy() - estimate(altered))) - reference_mae))
                permutations[group] = {"columns": columns, "mean_delta_mae": float(np.mean(changes)),
                                       "sd_delta_mae": float(np.std(changes, ddof=1)), "replicates": changes}
            output["permutation"][name] = {"evaluation_role": "validation", "n": len(validation),
                "repetitions": 30, "seed": 20261106, "baseline_mae": reference_mae, "groups": permutations}
            for axis in ("campaign", "region"):
                output["strata"].setdefault(axis, {})[name] = {}
                for category in frame.iloc[test][axis].unique():
                    indices = np.array([i for i in test if frame.iloc[i][axis] == category])
                    output["strata"][axis][name][str(category)] = _metrics(frame.iloc[indices].yield_t_ha, predictions[indices])
            print(f"Evidence and grouped permutation completed: {name}", flush=True)
        output["statistics"] = compare_models(dataset.file_path, {name: model.artifact_path for name, model in models.items()}, "temporal", 42)
        output["versions"] = {package: importlib.metadata.version(package) for package in
            ("numpy", "pandas", "scikit-learn", "xgboost-cpu", "torch", "scipy", "fastapi", "sqlalchemy")}
        output["python"] = platform.python_version()
        output["analysis_note"] = "Frozen models; no retraining. Post hoc grouped permutation on 2020 validation only; descriptive, non-causal, not SHAP. No held-out test tuning."
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
