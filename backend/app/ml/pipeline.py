"""CSV validation, leakage-safe evaluation, model training and inference.

Only columns supplied by the CSV are used. Deep models require features named
``<variable>_t0``, ``<variable>_t1``, etc. The same parcel, campaign, or region
never appears in more than one split when that validation mode is selected.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



REQUIRED = ("parcel_id", "crop", "region", "campaign", "yield_t_ha")
CATEGORIES = ("crop", "region", "campaign")
EXCLUDED = set(REQUIRED) | {"synthetic_demo", "country", "spatial_unit"}
VALIDATION_GROUPS = {"parcel": "parcel_id", "campaign": "campaign", "region": "region"}
DEEP_MODELS = {"LSTM", "CNN-LSTM", "CNN-Transformer"}
TEMPORAL_PATTERN = re.compile(r"^(.+)_t(\d+)$")


def _read_dataset(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CSV file not found: {source}")
    if source.suffix.lower() != ".csv":
        raise ValueError("The input must be a CSV file")
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if len(header) != len(set(header)):
            raise ValueError("CSV contains duplicate column names")
        frame = pd.read_csv(source, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"Cannot read CSV: {exc}") from exc
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if len(frame) < 3:
        raise ValueError("At least three records are required")
    for name in ("parcel_id", *CATEGORIES):
        if frame[name].isna().any() or (frame[name].astype(str).str.strip() == "").any():
            raise ValueError(f"Column {name} has missing or blank values")
        frame[name] = frame[name].astype(str)
    target = pd.to_numeric(frame["yield_t_ha"], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy()).all():
        raise ValueError("yield_t_ha must contain finite numeric values")
    if (target < 0).any():
        raise ValueError("yield_t_ha cannot contain negative values")
    frame["yield_t_ha"] = target.astype(float)
    spatial_unit = None
    if "spatial_unit" in frame:
        units = frame["spatial_unit"].dropna().astype(str).str.strip().unique().tolist()
        if len(units) != 1 or units[0] not in {"parcel", "administrative_region", "pixel"}:
            raise ValueError("spatial_unit must have one consistent value: parcel, administrative_region or pixel")
        spatial_unit = units[0]
    features = [name for name in frame.columns if name not in EXCLUDED]
    if not features:
        raise ValueError("At least one numeric feature column is required")
    for name in features:
        raw = frame[name]
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & numeric.isna()
        if invalid.any() or np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"Feature {name} must be numeric or missing")
        if numeric.isna().all():
            raise ValueError(f"Feature {name} contains no observed values")
        frame[name] = numeric.astype(float)
    return frame, features


def inspect_csv(path: str | Path) -> dict[str, Any]:
    """Validate a user CSV and summarize its usable data without training."""
    frame, features = _read_dataset(path)
    return {
        "path": str(Path(path).resolve()),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "feature_names": features + list(CATEGORIES),
        "numeric_feature_names": features,
        "categorical_features": list(CATEGORIES),
        "missing_by_feature": {key: int(frame[key].isna().sum()) for key in features},
        "unique_parcels": int(frame.parcel_id.nunique()),
        "unique_campaigns": int(frame.campaign.nunique()),
        "unique_regions": int(frame.region.nunique()),
        "synthetic_demo": _is_demo(frame),
        "spatial_unit": str(frame["spatial_unit"].iloc[0]) if "spatial_unit" in frame else None,
    }


def _is_demo(frame: pd.DataFrame) -> bool:
    if "synthetic_demo" not in frame:
        return False
    return frame["synthetic_demo"].astype(str).str.lower().isin(("true", "1", "yes")).any().item()


def _canonical_model(value: str) -> str:
    names = {
        "randomforest": "RandomForest", "random_forest": "RandomForest",
        "random forest": "RandomForest", "rf": "RandomForest",
        "xgboost": "XGBoost", "xgb": "XGBoost",
        "lstm": "LSTM", "cnn-lstm": "CNN-LSTM", "cnn_lstm": "CNN-LSTM",
        "cnn-transformer": "CNN-Transformer", "cnn_transformer": "CNN-Transformer",
    }
    canonical = names.get(value.strip().lower())
    if canonical is None:
        raise ValueError(f"Unknown model {value!r}. Choose RandomForest, XGBoost, LSTM, CNN-LSTM or CNN-Transformer")
    return canonical


def _split(frame: pd.DataFrame, validation: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if validation == "temporal":
        campaigns = frame["campaign"].astype(str)
        if not campaigns.str.fullmatch(r"(?:19|20)\d{2}").all():
            raise ValueError("temporal validation requires four-digit campaign years")
        years = sorted(int(year) for year in campaigns.astype(int).unique())
        if len(years) < 5:
            raise ValueError("temporal validation requires at least five distinct campaign years")
        test_count = max(1, int(np.ceil(len(years) * 0.2)))
        validation_count = max(1, int(np.ceil((len(years) - test_count) * 0.2)))
        train_years = years[:-(test_count + validation_count)]
        validation_years = years[-(test_count + validation_count):-test_count]
        test_years = years[-test_count:]
        values = campaigns.astype(int).to_numpy()
        train_idx = np.flatnonzero(np.isin(values, train_years))
        val_idx = np.flatnonzero(np.isin(values, validation_years))
        test_idx = np.flatnonzero(np.isin(values, test_years))
        split = {
            "strategy": "chronological_campaign_holdout", "group_column": "campaign",
            "train_years": train_years, "validation_years": validation_years,
            "test_years": test_years, "train_rows": len(train_idx),
            "validation_rows": len(val_idx), "test_rows": len(test_idx),
            "train_groups": len(train_years), "validation_groups": len(validation_years),
            "test_groups": len(test_years), "group_overlap": False,
        }
        return train_idx, val_idx, test_idx, split
    if validation not in VALIDATION_GROUPS:
        raise ValueError("validation must be 'parcel', 'campaign', 'region' or 'temporal'")
    group_column = VALIDATION_GROUPS[validation]
    groups = frame[group_column].to_numpy()
    if len(np.unique(groups)) < 3:
        raise ValueError(f"At least three distinct {group_column} values are required for grouped train/validation/test splits")
    positions = np.arange(len(frame))
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val, test = next(first.split(positions, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_rel, val_rel = next(second.split(train_val, groups=groups[train_val]))
    train_idx, val_idx = train_val[train_rel], train_val[val_rel]
    split = {
        "strategy": f"grouped_by_{validation}", "group_column": group_column,
        "train_rows": len(train_idx), "validation_rows": len(val_idx), "test_rows": len(test),
        "train_groups": len(set(groups[train_idx])),
        "validation_groups": len(set(groups[val_idx])),
        "test_groups": len(set(groups[test])),
        "group_overlap": False,
    }
    return train_idx, val_idx, test, split


def _preprocessor(features: list[str], deep: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if deep:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORIES)),
    ], sparse_threshold=0)


def _sequence_schema(features: list[str]) -> dict:
    variables: dict[str, set[int]] = {}
    static = []
    for name in features:
        match = TEMPORAL_PATTERN.fullmatch(name)
        if match:
            variables.setdefault(match.group(1), set()).add(int(match.group(2)))
        else:
            static.append(name)
    if not variables:
        raise ValueError("Temporal models require at least one <variable>_t0, <variable>_t1 sequence")
    steps = sorted(set().union(*variables.values()))
    if len(steps) < 2 or steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError("Temporal models require at least two consecutive time steps")
    for variable, present in variables.items():
        if sorted(present) != steps:
            raise ValueError(f"Temporal feature {variable} is missing time steps; all variables need {steps}")
    return {"variables": sorted(variables), "timesteps": steps, "static_features": static}


def _deep_arrays(matrix: np.ndarray, features: list[str], schema: dict) -> tuple[np.ndarray, np.ndarray]:
    lookup = {name: idx for idx, name in enumerate(features)}
    indices = [[lookup[f"{variable}_t{step}"] for variable in schema["variables"]]
               for step in schema["timesteps"]]
    sequence = matrix[:, np.asarray(indices)].astype(np.float32)
    static_indices = [lookup[name] for name in schema["static_features"]]
    static_indices.extend(range(len(features), matrix.shape[1]))
    static = matrix[:, static_indices].astype(np.float32)
    return sequence, static


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    valid_mape = np.all(np.abs(y_true) > 1e-8)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if valid_mape else None,
        "n": int(len(y_true)),
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _notify(progress: Callable | None, stage: str, percent: int) -> None:
    if progress:
        progress({"stage": stage, "percent": percent})


def train(path: str | Path, model_type: str, artifact_dir: str | Path, seed: int = 42,
          validation: str = "parcel", epochs: int = 5,
          progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Train and evaluate with disjoint group splits; save a reusable artifact."""
    started = time.monotonic()
    kind = _canonical_model(model_type)
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    _notify(progress, "validating", 5)
    frame, features = _read_dataset(path)
    train_idx, val_idx, test_idx, split_info = _split(frame, validation, seed)
    schema = _sequence_schema(features) if kind in DEEP_MODELS else None
    if kind == "CNN-Transformer" and len(schema["variables"]) < 2:
        raise ValueError("CNN-Transformer multimodal requires at least two temporal variables")
    preprocessor = _preprocessor(features, kind in DEEP_MODELS)
    input_cols = features + list(CATEGORIES)
    x_train = preprocessor.fit_transform(frame.iloc[train_idx][input_cols])
    x_val = preprocessor.transform(frame.iloc[val_idx][input_cols])
    x_test = preprocessor.transform(frame.iloc[test_idx][input_cols])
    y = frame["yield_t_ha"].to_numpy(dtype=float)
    _notify(progress, "training", 25)
    payload: dict[str, Any] = {
        "model_type": kind, "features": features, "categories": list(CATEGORIES),
        "preprocessor": preprocessor, "sequence_schema": schema,
    }
    if kind == "RandomForest":
        model = RandomForestRegressor(n_estimators=160, min_samples_leaf=2,
                                      random_state=seed, n_jobs=-1)
        model.fit(x_train, y[train_idx])
        predict_matrix = model.predict
        payload["model"] = model
        device = "cpu"
    elif kind == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("XGBoost is unavailable; install backend/ml-requirements.txt") from exc
        model = XGBRegressor(n_estimators=180, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
                             random_state=seed, n_jobs=2, tree_method="hist")
        model.fit(x_train, y[train_idx], eval_set=[(x_val, y[val_idx])], verbose=False)
        predict_matrix = model.predict
        payload["model"] = model
        device = "cpu"
    else:
        import torch
        from .deep import TemporalRegressor

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seq_train, static_train = _deep_arrays(x_train, features, schema)
        model = TemporalRegressor(kind, seq_train.shape[2], static_train.shape[1],
                                  seq_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        loss_fn = torch.nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(seq_train),
                                                  torch.from_numpy(static_train),
                                                  torch.from_numpy(y[train_idx].astype(np.float32)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(64, len(dataset)),
                                             shuffle=True, generator=torch.Generator().manual_seed(seed))
        for epoch in range(epochs):
            model.train()
            for seq_batch, static_batch, target_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                estimate = model(seq_batch.to(device), static_batch.to(device))
                loss = loss_fn(estimate, target_batch.to(device))
                loss.backward()
                optimizer.step()
            _notify(progress, "training", 25 + round(55 * (epoch + 1) / epochs))

        def predict_matrix(matrix: np.ndarray) -> np.ndarray:
            seq, static = _deep_arrays(matrix, features, schema)
            model.eval()
            with torch.no_grad():
                return model(torch.from_numpy(seq).to(device),
                             torch.from_numpy(static).to(device)).cpu().numpy()

        payload["model_config"] = {
            "sequence_channels": seq_train.shape[2], "static_features": static_train.shape[1],
            "timesteps": seq_train.shape[1],
        }
    _notify(progress, "evaluating", 85)
    metrics = {
        "train": _metrics(y[train_idx], predict_matrix(x_train)),
        "validation": _metrics(y[val_idx], predict_matrix(x_val)),
        "test": _metrics(y[test_idx], predict_matrix(x_test)),
    }
    if kind in DEEP_MODELS:
        payload["state_dict"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    artifact_path = artifact_root / f"{run_id}.joblib"
    tmp_path = artifact_root / f"{run_id}.tmp"
    joblib.dump(payload, tmp_path)
    os.replace(tmp_path, artifact_path)
    metadata = {
        "run_id": run_id, "model_type": kind, "artifact_path": str(artifact_path.resolve()),
        "feature_names": features + list(CATEGORIES),
        "numeric_feature_names": features, "categorical_features": list(CATEGORIES),
        "transformed_feature_names": list(preprocessor.get_feature_names_out()),
        "metrics": metrics, "split_info": split_info, "seed": seed,
        "validation": validation, "epochs": epochs if kind in DEEP_MODELS else None,
        "device": device, "training_seconds": round(time.monotonic() - started, 3),
        "source_path": str(Path(path).resolve()), "source_sha256": _sha256(path),
        "synthetic_demo": _is_demo(frame),
        "scientific_status": "synthetic demonstration only" if _is_demo(frame) else "evaluated on supplied data",
    }
    metadata_path = artifact_root / f"{run_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    _notify(progress, "complete", 100)
    return metadata


def predict(artifact_path: str | Path, features: dict[str, Any]) -> float:
    """Predict t/ha for one observation using its saved training preprocessors."""
    payload = joblib.load(artifact_path)
    required = payload["features"] + payload["categories"]
    missing = [name for name in required if name not in features]
    if missing:
        raise ValueError(f"Missing prediction features: {', '.join(missing)}")
    observation = pd.DataFrame([{name: features[name] for name in required}])
    for name in payload["features"]:
        observation[name] = pd.to_numeric(observation[name], errors="coerce")
    if observation[payload["features"]].isna().any().any():
        raise ValueError("Prediction numeric features must be finite numbers")
    if not np.isfinite(observation[payload["features"]].to_numpy(dtype=float)).all():
        raise ValueError("Prediction numeric features must be finite numbers")
    for name in payload["categories"]:
        observation[name] = observation[name].astype(str)
    matrix = payload["preprocessor"].transform(observation[required])
    if payload["model_type"] in DEEP_MODELS:
        import torch
        from .deep import TemporalRegressor

        model = TemporalRegressor(payload["model_type"], **payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        seq, static = _deep_arrays(matrix, payload["features"], payload["sequence_schema"])
        with torch.no_grad():
            estimate = model(torch.from_numpy(seq), torch.from_numpy(static)).item()
    else:
        estimate = float(payload["model"].predict(matrix)[0])
    return float(estimate)


def categorical_domain_warnings(artifact_path: str | Path, features: dict[str, Any]) -> list[str]:
    """Identify categories the fitted encoder never saw during training."""
    payload = joblib.load(artifact_path)
    encoder = payload["preprocessor"].named_transformers_["cat"]
    warnings = []
    for name, known in zip(payload["categories"], encoder.categories_):
        if name in features and str(features[name]) not in set(map(str, known)):
            warnings.append(f"{name}={features[name]} no estuvo presente en el entrenamiento")
    return warnings
