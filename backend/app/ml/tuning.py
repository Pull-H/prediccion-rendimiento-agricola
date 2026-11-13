"""Nested Optuna search with fold-local preprocessing and a sealed test set.

Run: python -m app.ml.tuning --source data.csv --output artifacts/search
Temporal networks require real sequence columns and the optional PyTorch dependency.
"""
from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .pipeline import (CATEGORIES, VALIDATION_GROUPS, _canonical_model, _metrics,
                       _preprocessor, _read_dataset, _sha256, _split, _is_demo,
                       DEEP_MODELS, _sequence_schema)


RANGES = {
    "RandomForest": {
        "n_estimators": (100, 500), "max_depth": (2, 20),
        "min_samples_leaf": (2, 16), "min_samples_split": (2, 20),
        "max_features": (0.4, 1.0),
    },
    "XGBoost": {
        "n_estimators": (100, 600), "max_depth": (2, 6),
        "learning_rate": (0.01, 0.15), "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0), "min_child_weight": (1, 20),
        "reg_alpha": (0.0001, 10.0), "reg_lambda": (0.1, 30.0),
    },
}
for _kind in DEEP_MODELS:
    RANGES[_kind] = {"hidden": (16, 64), "learning_rate": (0.0001, 0.01),
                     "weight_decay": (0.000001, 0.01), "batch_size": (16, 128), "epochs": (1, 50)}


class TrainingCancelled(Exception):
    """Cooperative cancellation; completed folds and trials remain resumable."""


def package_versions(kind):
    result = {}
    for name in ("numpy", "pandas", "scikit-learn", "optuna", "xgboost", *(["torch"] if kind in DEEP_MODELS else [])):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            # The Linux deployment uses the CPU-only distribution of XGBoost.
            try:
                result[name] = version("xgboost-cpu") if name == "xgboost" else None
            except PackageNotFoundError:
                result[name] = None
    return result


def validate_parameters(kind, parameters):
    allowed = RANGES.get(kind, {})
    for name, value in parameters.items():
        if name not in allowed:
            raise ValueError(f"Unsupported hyperparameter for {kind}: {name}")
        low, high = allowed[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        if isinstance(low, int) and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if kind in DEEP_MODELS and name == "hidden" and value % 16:
            raise ValueError("hidden must be a multiple of 16")
    return dict(parameters)


def validated_ranges(kind, requested=None):
    ranges = dict(RANGES[kind])
    for key, bounds in (requested or {}).items():
        if key not in ranges or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Invalid search range: {key}")
        validate_parameters(kind, {key: bounds[0]})
        validate_parameters(kind, {key: bounds[1]})
        if bounds[0] > bounds[1]:
            raise ValueError(f"Range minimum exceeds maximum: {key}")
        ranges[key] = tuple(bounds)
    return ranges


def _folds(frame, validation, count, min_train_years=1):
    column = "campaign" if validation == "temporal" else VALIDATION_GROUPS[validation]
    groups = frame[column].to_numpy()
    if validation == "temporal":
        years = np.sort(np.unique(groups.astype(int)))
        if len(years) < count + min_train_years:
            raise ValueError(f"Temporal CV needs at least {count + min_train_years} campaign years")
        return [(np.flatnonzero(np.isin(groups.astype(int), years[a])),
                 np.flatnonzero(np.isin(groups.astype(int), years[b])))
                for a, b in TimeSeriesSplit(n_splits=count,
                    test_size=max(1, (len(years) - min_train_years) // count)).split(years)]
    if len(np.unique(groups)) < count:
        raise ValueError(f"CV needs at least {count} independent {column} groups")
    return list(GroupKFold(n_splits=count).split(frame, groups=groups))


def _estimator(kind, features, parameters, seed, cancelled=None):
    if kind == "RandomForest":
        model = RandomForestRegressor(**parameters, random_state=seed, n_jobs=2)
    elif kind == "XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(**parameters, random_state=seed, n_jobs=2,
                             objective="reg:squarederror", tree_method="hist")
    else:
        from .temporal_estimator import TemporalEstimator
        model = TemporalEstimator(kind, features, seed=seed, cancelled=cancelled, **parameters)
    return Pipeline([("preprocess", _preprocessor(features, kind in DEEP_MODELS)), ("model", model)])


def optimize(source, output, model_type="RandomForest", validation="parcel", seed=42,
             trials=20, outer_folds=5, inner_folds=3, progress=None,
             cancelled=None, resume=False, search_ranges=None):
    import optuna

    started = time.monotonic()
    kind = _canonical_model(model_type)
    if kind not in RANGES:
        raise ValueError("Optuna search currently supports RandomForest and XGBoost")
    if not 1 <= trials <= 200 or not 2 <= outer_folds <= 10 or not 2 <= inner_folds <= 10:
        raise ValueError("Use 1–200 trials and 2–10 folds")
    source_hash = _sha256(source)
    frame, features = _read_dataset(source)
    if kind in DEEP_MODELS:
        _sequence_schema(features)
    train_idx, val_idx, test_idx, split_info = _split(frame, validation, seed)
    # Validation is now a dedicated calibration set and is never used for selection.
    development_idx = np.sort(train_idx)
    development = frame.iloc[development_idx].reset_index(drop=True)
    outer = _folds(development, validation, outer_folds, min_train_years=inner_folds + 1)
    # Reject an infeasible design before starting any expensive training.
    for a, _ in outer:
        _folds(development.iloc[a], validation, inner_folds)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    database = root / "trials.sqlite3"
    ranges = validated_ranges(kind, search_ranges)
    manifest = root / "search-control.json"
    configuration = {"source_sha256": source_hash, "model": kind, "seed": seed,
                     "validation": validation, "trials": trials, "outer_folds": outer_folds,
                     "inner_folds": inner_folds, "ranges": {k: list(v) for k, v in ranges.items()},
                     "protocol": "nested_cv_reserved_calibration_v2"}
    if resume:
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous["configuration"] != configuration:
            raise ValueError("Cannot resume: dataset or search configuration changed")
        run_name = previous["run_id"]
    else:
        if manifest.exists():
            raise ValueError("Search already exists; use resume or a new output directory")
        run_name = uuid4().hex
        manifest.write_text(json.dumps({"configuration": configuration, "run_id": run_name}), encoding="utf-8")
    finished = root / f"{run_name}.json"
    if finished.exists():
        return json.loads(finished.read_text(encoding="utf-8"))
    columns = features + list(CATEGORIES)
    results = []

    def check_cancel():
        if cancelled and cancelled():
            raise TrainingCancelled("Training cancelled by user")

    def search(data, name):
        check_cancel()
        folds = _folds(data, validation, inner_folds)
        sampler_path = root / f"sampler-{name}.joblib"
        sampler = joblib.load(sampler_path) if resume and sampler_path.exists() else optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(
            direction="minimize", sampler=sampler,
            storage="sqlite:///" + database.as_posix().replace("{", "{{").replace("}", "}}"),
            study_name=run_name + name, load_if_exists=resume)

        def objective(trial):
            check_cancel()
            parameters = {}
            for key, (low, high) in ranges.items():
                if isinstance(low, int):
                    parameters[key] = trial.suggest_int(key, low, high, step=16 if key == "hidden" else 1)
                else:
                    parameters[key] = trial.suggest_float(
                        key, low, high, log=key in {"learning_rate", "reg_alpha", "reg_lambda", "weight_decay"})
            losses = []
            for a, b in folds:
                check_cancel()
                estimator = _estimator(kind, features, parameters, seed, cancelled)
                estimator.fit(data.iloc[a][columns], data.iloc[a].yield_t_ha)
                losses.append(_metrics(data.iloc[b].yield_t_ha.to_numpy(),
                                       estimator.predict(data.iloc[b][columns]))["rmse"])
            value = float(np.mean(losses))
            if progress:
                study_index = outer_folds if name == "final" else int(name.split("-")[-1])
                progress({"study": name, "trial": trial.number, "rmse": value,
                          "percent": min(94, 5 + int(89 * (study_index + min(trial.number + 1, trials) / trials) / (outer_folds + 1)))})
            return value

        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        study.optimize(objective, n_trials=max(0, trials - completed), n_jobs=1,
                       callbacks=[lambda current, _: joblib.dump(current.sampler, sampler_path)])
        return study.best_params, study.best_value

    for number, (a, b) in enumerate(outer):
        check_cancel()
        checkpoint = root / f"fold-{number}.json"
        if resume and checkpoint.exists():
            results.append(json.loads(checkpoint.read_text(encoding="utf-8")))
            continue
        data = development.iloc[a].reset_index(drop=True)
        parameters, inner_rmse = search(data, f"outer-{number}")
        estimator = _estimator(kind, features, parameters, seed, cancelled)
        estimator.fit(data[columns], data.yield_t_ha)
        results.append({
            "fold": number, "parameters": parameters, "inner_rmse": inner_rmse,
            "train_indices": development_idx[a].tolist(),
            "validation_indices": development_idx[b].tolist(),
            "metrics": _metrics(development.iloc[b].yield_t_ha.to_numpy(),
                                estimator.predict(development.iloc[b][columns])),
        })
        checkpoint.write_text(json.dumps(results[-1]), encoding="utf-8")
    # Tune again on all development data; never pick parameters using outer/test scores.
    parameters, final_cv = search(development, "final")
    estimator = _estimator(kind, features, parameters, seed, cancelled)
    estimator.fit(development[columns], development.yield_t_ha)
    check_cancel()
    test = frame.iloc[test_idx]
    predictions = estimator.predict(test[columns])
    if _sha256(source) != source_hash:
        raise ValueError("Source changed during search; results were not saved")
    report = {
        "run_id": run_name, "source_sha256": source_hash,
        "source_path": str(Path(source).resolve()),
        "feature_names": columns, "numeric_feature_names": features,
        "categorical_features": list(CATEGORIES), "epochs": None,
        "transformed_feature_names": list(estimator.named_steps["preprocess"].get_feature_names_out()),
        "model_type": kind, "validation": validation, "seed": seed,
        "synthetic_demo": _is_demo(frame), "search_ranges": ranges,
        "trials_per_study": trials, "outer_folds": results, "inner_folds": inner_folds,
        "best_parameters": parameters, "final_inner_rmse": final_cv,
        "outer_rmse_mean": float(np.mean([r["metrics"]["rmse"] for r in results])),
        "outer_rmse_std": float(np.std([r["metrics"]["rmse"] for r in results], ddof=1)),
        "test_indices": test_idx.tolist(), "test_metrics": _metrics(test.yield_t_ha.to_numpy(), predictions),
        "baseline_test_metrics": _metrics(test.yield_t_ha.to_numpy(),
            np.full(len(test), development.yield_t_ha.mean())),
        "calibration_reserved": True,
        "split_info": {**split_info, "train_rows": len(development),
                       "strategy": "nested_cv_reserved_calibration_v2",
                       "initial_holdout_split": split_info,
                       "train_groups": int(development[split_info["group_column"]].nunique()),
                       "calibration_rows": len(val_idx)},
        "warning": "Exploratory; folds are dependent and do not establish validity in Peru.",
    }
    model_path = root / f"{run_name}.joblib"
    # Match the prediction payload used by the application.
    payload = {"model_type": kind, "features": features, "categories": list(CATEGORIES),
                 "preprocessor": estimator.named_steps["preprocess"],
                 "sequence_schema": None}
    if kind in DEEP_MODELS:
        payload.update(estimator.named_steps["model"].payload())
    else:
        payload["model"] = estimator.named_steps["model"]
    joblib.dump(payload, model_path)
    report["artifact_path"] = str(model_path)
    report["training_seconds"] = round(time.monotonic() - started, 3)
    report["metrics"] = {"test": report["test_metrics"], "baseline_test": report["baseline_test_metrics"],
                         "nested_cv_rmse": report["outer_rmse_mean"]}
    report["versions"] = package_versions(kind)
    (root / f"{run_name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return json.loads((root / f"{run_name}.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=list(RANGES), default="RandomForest")
    parser.add_argument("--validation", choices=[*VALIDATION_GROUPS, "temporal"], default="parcel")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume the identical persisted search")
    args = parser.parse_args()
    print(json.dumps(optimize(args.source, args.output, args.model, args.validation, args.seed,
                             args.trials, args.outer_folds, args.inner_folds, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
