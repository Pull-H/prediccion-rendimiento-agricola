"""Verify a trusted study bundle and recompute frozen predictions and statistics.

No web account or database is required. Only load joblib files from a trusted
source: hashes check integrity, not the safety of untrusted serialized code.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.pipeline import _metrics, _read_dataset, _split, train
from app.ml.statistics import _batch_predict, compare_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/reproducibility/replay_results.json"))
    parser.add_argument("--retrain", action="store_true", help="Run five NEW models; never overwrite frozen artifacts")
    parser.add_argument("--verify-permutation", action="store_true", help="Recompute 30 validation permutations for all seven groups and five models")
    args = parser.parse_args()
    root = args.bundle.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, declared in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or invalid bundle path: {name}")
        if path.stat().st_size != declared["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != declared["sha256"]:
            raise ValueError(f"Integrity mismatch: {name}")
    csv = root / "yieldsat_Germany_wheat_240d.csv"
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == manifest["dataset_sha256"]
    frame, _ = _read_dataset(csv)
    evidence = json.loads((root / "evidencia_modelos.json").read_text(encoding="utf-8"))
    indices = _split(frame, "temporal", 42)[:3]
    split_rows = pd.read_csv(root / "splits.csv")
    assert split_rows["row"].tolist() == list(range(len(frame)))
    assert split_rows.parcel_id.tolist() == frame.parcel_id.tolist()
    assert split_rows.campaign.astype(str).tolist() == frame.campaign.tolist()
    for role, selected in zip(("train", "validation", "test"), indices):
        assert sorted(split_rows.loc[split_rows.role == role, "row"].tolist()) == sorted(map(int, selected))
    artifacts = {entry["model_type"]: root / entry["file"] for entry in manifest["models"]}
    metrics = {}
    for name, path in artifacts.items():
        predictions = _batch_predict(path, frame)
        assert np.allclose(predictions, evidence["predictions"][name], rtol=0, atol=1e-5), name
        metrics[name] = {role: _metrics(frame.iloc[ix].yield_t_ha, predictions[ix])
                         for role, ix in zip(("train", "validation", "test"), indices)}
        for role in metrics[name]:
            for key in ("mae", "rmse", "r2"):
                assert abs(metrics[name][role][key] - evidence["metrics"][name][role][key]) < 1e-5, (name, role, key)
    statistics = compare_models(csv, artifacts, "temporal", seed=42, bootstrap_repetitions=2000)
    assert np.isclose(statistics["friedman"]["p_value"], evidence["statistics"]["friedman"]["p_value"], atol=1e-10)
    def check_numeric(actual, expected):
        if isinstance(expected, dict):
            for key in expected:
                if key not in {"artifact_path", "metadata_path"}:
                    check_numeric(actual[key], expected[key])
        elif isinstance(expected, list):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                check_numeric(a, b)
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            assert np.isclose(actual, expected, rtol=0, atol=1e-5), (actual, expected)
    for key in ("friedman", "wilcoxon_holm", "models"):
        check_numeric(statistics[key], evidence["statistics"][key])
    result = {"files_verified": len(manifest["files"]), "rows": len(frame), "models_verified": len(artifacts),
              "prediction_tolerance": 1e-5, "metrics": metrics, "statistics": statistics}
    if args.verify_permutation:
        holdout = frame.iloc[indices[1]].copy()
        verified = 0
        for name, path in artifacts.items():
            original = np.mean(np.abs(holdout.yield_t_ha.to_numpy() - _batch_predict(path, holdout)))
            for group in evidence["permutation"][name]["groups"].values():
                rng = np.random.default_rng(20261106)
                values = []
                for _ in range(30):
                    altered = holdout.copy()
                    columns = group["columns"]
                    altered.loc[:, columns] = holdout.iloc[rng.permutation(len(holdout))][columns].to_numpy()
                    values.append(float(np.mean(np.abs(holdout.yield_t_ha.to_numpy() - _batch_predict(path, altered))) - original))
                assert np.allclose(values, group["replicates"], rtol=0, atol=1e-5), name
                verified += 1
            print(f"Permutation verified: {name}", flush=True)
        result["permutation_groups_verified"] = verified
    if args.retrain:
        output = args.output.resolve().parent / "new_training"
        if output.exists():
            raise FileExistsError("Use a new output location for each retraining")
        output.mkdir(parents=True)
        result["new_training"] = {name: train(csv, name, output, seed=42, validation="temporal", epochs=30)
                                  for name in artifacts}
        result["retraining_note"] = "New runs, not a replacement for frozen results; cross-platform numeric differences may occur."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"verified_models": len(artifacts), "rows": len(frame), "output": str(args.output)}))


if __name__ == "__main__":
    main()
