"""Exploratory paired comparison on recorded spatial identifiers."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from .pipeline import CATEGORIES, DEEP_MODELS, _deep_arrays, _metrics, _read_dataset, _sha256, _split


def _batch_predict(artifact_path: str | Path, observations) -> np.ndarray:
    payload = joblib.load(artifact_path)
    required = payload["features"] + payload["categories"]
    if set(required) != set(observations.columns) - {"parcel_id", "yield_t_ha", "country", "spatial_unit", "synthetic_demo"}:
        raise ValueError("Model feature schema does not match the evaluation dataset")
    matrix = payload["preprocessor"].transform(observations[required])
    if payload["model_type"] not in DEEP_MODELS:
        return np.asarray(payload["model"].predict(matrix), dtype=float).reshape(-1)

    import torch
    from .deep import TemporalRegressor

    model = TemporalRegressor(payload["model_type"], **payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    sequence, static = _deep_arrays(matrix, payload["features"], payload["sequence_schema"])
    with torch.no_grad():
        return model(torch.from_numpy(sequence), torch.from_numpy(static)).numpy().astype(float).reshape(-1)


def _holm_adjust(raw_p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values in their original order."""
    order = sorted(range(len(raw_p)), key=lambda i: raw_p[i])
    adjusted = [0.0] * len(raw_p)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(raw_p) - rank) * raw_p[index]))
        adjusted[index] = previous
    return adjusted


def compare_models(
    csv_path: str | Path,
    model_artifacts: dict[str, str | Path],
    validation: str,
    seed: int = 42,
    bootstrap_repetitions: int = 2000,
) -> dict:
    """Compare matched test predictions using spatial IDs as resampling blocks.

    Friedman and Wilcoxon operate on mean absolute error per spatial block.
    Bootstrap resamples whole blocks and estimates observation-weighted MAE CIs.
    """
    if len(model_artifacts) < 3:
        raise ValueError("At least three matched models are required for Friedman")
    if bootstrap_repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")
    frame, numeric_features = _read_dataset(csv_path)
    _, _, test_idx, split_info = _split(frame, validation, seed)
    test = frame.iloc[test_idx].copy()
    groups = sorted(test["parcel_id"].unique().tolist())
    if len(groups) < 3:
        raise ValueError("At least three independent spatial blocks are required")
    source_hash = _sha256(csv_path)
    model_names = list(model_artifacts)
    y_true = test["yield_t_ha"].to_numpy(dtype=float)
    predictions = []
    training_details = {}
    for name, artifact_path in model_artifacts.items():
        metadata_path = Path(artifact_path).with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata["source_sha256"] != source_hash or metadata["validation"] != validation
                or metadata["seed"] != seed or metadata["model_type"] != name
                or metadata["numeric_feature_names"] != numeric_features):
            raise ValueError(f"Model {name} was not trained on this dataset and split")
        predictions.append(_batch_predict(artifact_path, test))
        training_details[name] = {"epochs": metadata.get("epochs"), "device": metadata.get("device")}
    estimates = np.column_stack(predictions)
    if estimates.shape != (len(test), len(model_names)) or not np.isfinite(estimates).all():
        raise ValueError("Model predictions are incomplete or non-finite")
    absolute_errors = np.abs(estimates - y_true[:, None])
    block_ids = test["parcel_id"].to_numpy()
    block_counts = np.array([np.sum(block_ids == group) for group in groups], dtype=int)
    block_abs_sum = np.stack([absolute_errors[block_ids == group].sum(axis=0) for group in groups])
    block_mae = block_abs_sum / block_counts[:, None]

    statistic, p_value = friedmanchisquare(*[block_mae[:, i] for i in range(len(model_names))])
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        raise ValueError("Friedman test is undefined for these block losses")
    pairwise = []
    for first, second in combinations(range(len(model_names)), 2):
        difference = block_mae[:, first] - block_mae[:, second]
        if np.allclose(difference, 0):
            raw = 1.0
            pair_statistic = 0.0
        else:
            paired = wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto")
            raw, pair_statistic = float(paired.pvalue), float(paired.statistic)
        pairwise.append({"first": model_names[first], "second": model_names[second],
                         "statistic": pair_statistic,
                         "median_block_mae_difference": float(np.median(difference)), "p_value": raw})
    corrected = _holm_adjust([row["p_value"] for row in pairwise])
    for row, value in zip(pairwise, corrected):
        row["holm_p_value"] = value
        row["significant_0_05"] = bool(value < 0.05)

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), size=(bootstrap_repetitions, len(groups)))
    sampled_sum = block_abs_sum[sampled].sum(axis=1)
    sampled_count = block_counts[sampled].sum(axis=1)
    boot_mae = sampled_sum / sampled_count[:, None]
    intervals = np.quantile(boot_mae, [0.025, 0.975], axis=0)
    metrics = {}
    for i, name in enumerate(model_names):
        metrics[name] = {
            "test": _metrics(y_true, estimates[:, i]),
            "mae_95pct_cluster_bootstrap_ci": [float(intervals[0, i]), float(intervals[1, i])],
        }
    return {
        "validation": validation, "seed": seed, "dataset_sha256": source_hash,
        "split_info": split_info, "test_observations": len(test),
        "block_column": "parcel_id", "spatial_unit": str(test["spatial_unit"].iloc[0]) if "spatial_unit" in test else "undeclared",
        "independent_blocks": len(groups), "bootstrap_repetitions": bootstrap_repetitions,
        "block_count": len(groups), "alpha": 0.05,
        "training_details": training_details,
        "methods": {
            "loss": "Mean absolute error per parcel_id block, in t/ha",
            "friedman": "Tie-corrected statistic; asymptotic chi-square p-value",
            "wilcoxon": "Paired two-sided signed-rank; zero_method=wilcox; method=auto",
            "multiple_comparisons": "Holm correction across all model pairs",
            "bootstrap": "Percentile 95% interval of observation-weighted MAE; whole parcel_id resampling",
            "bootstrap_interval_scope": "Uncertainty in aggregate MAE, not an individual yield prediction interval",
        },
        "independence_verified": False,
        "test_region_counts": {str(name): int(count) for name, count in test["region"].value_counts().items()},
        "models": metrics,
        "friedman": {"statistic": float(statistic), "p_value": float(p_value),
                     "degrees_of_freedom": len(model_names) - 1,
                     "kendall_w": float(statistic / (len(groups) * (len(model_names) - 1)))},
        "wilcoxon_holm": pairwise,
        "interpretation": "Exploratory paired comparison within this dataset; a single split and spatial dependence limit inference. Lower MAE is better. No evidence of transfer to Peru.",
    }
