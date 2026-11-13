"""Regression diagnostics and model-agnostic explanations in original variables."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .pipeline import (CATEGORIES, DEEP_MODELS, _deep_arrays, _metrics,
                       _read_dataset, _sha256, _split)


class Predictor:
    def __init__(self, artifact):
        self.payload = joblib.load(artifact)
        self.columns = self.payload["features"] + self.payload["categories"]
        self.model = self.payload.get("model")
        if self.payload["model_type"] in DEEP_MODELS:
            from .deep import TemporalRegressor
            self.model = TemporalRegressor(self.payload["model_type"], **self.payload["model_config"])
            self.model.load_state_dict(self.payload["state_dict"])
            self.model.eval()

    def __call__(self, frame):
        matrix = self.payload["preprocessor"].transform(frame[self.columns])
        if self.payload["model_type"] not in DEEP_MODELS:
            result = self.model.predict(matrix)
        else:
            import torch
            sequence, static = _deep_arrays(matrix, self.payload["features"], self.payload["sequence_schema"])
            with torch.no_grad():
                result = self.model(torch.from_numpy(sequence), torch.from_numpy(static)).numpy()
        result = np.asarray(result, dtype=float).reshape(-1)
        if not np.isfinite(result).all():
            raise ValueError("The model produced non-finite predictions")
        return result


def context(artifact, source):
    metadata = json.loads(Path(artifact).with_suffix(".json").read_text(encoding="utf-8"), parse_constant=lambda _: None)
    if _sha256(source) != metadata["source_sha256"]:
        raise ValueError("Dataset changed since training")
    frame, _ = _read_dataset(source)
    train, validation, test, _ = _split(frame, metadata["validation"], metadata["seed"])
    strategy = metadata.get("split_info", {}).get("strategy")
    if strategy == "nested_cv_then_development_refit":
        train, validation = np.sort(np.concatenate([train, validation])), np.array([], dtype=int)
    return Predictor(artifact), metadata, frame, train, validation, test


def interval_info(predictor, metadata, frame, calibration, alpha=0.1):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if len(calibration) < 10:
        return {"available": False, "reason": "No hay al menos 10 observaciones de calibración independiente"}
    # A legacy sidecar does not prove its validation was kept out of model selection.
    if not metadata.get("calibration_reserved"):
        return {"available": False, "reason": "Modelo anterior sin calibración reservada; requiere reentrenamiento"}
    residuals = np.abs(frame.iloc[calibration].yield_t_ha.to_numpy() - predictor(frame.iloc[calibration]))
    rank = int(np.ceil((len(residuals) + 1) * (1 - alpha)))
    if rank > len(residuals):
        return {"available": False, "reason": "Calibración insuficiente para la cobertura solicitada"}
    radius = float(np.sort(residuals)[rank - 1])
    return {"available": True, "method": "split_conformal_absolute_residuals", "radius": radius,
            "nominal_coverage": 1 - alpha, "calibration_n": len(residuals),
            "warning": "La cobertura nominal supone intercambiabilidad; dependencia o cambios temporales/espaciales pueden invalidarla"}


def assess_coverage(interval, residuals):
    if interval["available"]:
        interval["test_coverage"] = float(np.mean(np.abs(residuals) <= interval["radius"]))
        interval["mean_width"] = 2 * interval["radius"]
        interval["empirical_target_met"] = interval["test_coverage"] >= interval["nominal_coverage"]
        interval["status"] = "experimental"
        if not interval["empirical_target_met"]:
            interval["warning"] += "; cobertura de prueba inferior al objetivo nominal, sin cobertura garantizada"
    return interval


def evaluate(artifact, source, repetitions=500):
    predictor, metadata, frame, train, calibration, test = context(artifact, source)
    data = frame.iloc[test]
    observed = data.yield_t_ha.to_numpy()
    predicted = predictor(data)
    residuals = observed - predicted
    metrics = _metrics(observed, predicted)
    baseline = np.full(len(data), frame.iloc[train].yield_t_ha.mean())
    interval = interval_info(predictor, metadata, frame, calibration)
    assess_coverage(interval, residuals)
    # Bootstrap whole spatial units, not individual correlated rows.
    groups = data.parcel_id.to_numpy()
    unique = np.unique(groups)
    rng = np.random.default_rng(metadata["seed"])
    confidence = None
    if len(unique) >= 3:
        counts = np.array([np.sum(groups == group) for group in unique])
        absolute = np.array([np.abs(residuals[groups == group]).sum() for group in unique])
        squared = np.array([(residuals[groups == group] ** 2).sum() for group in unique])
        samples = rng.integers(0, len(unique), size=(repetitions, len(unique)))
        denominator = counts[samples].sum(axis=1)
        confidence = {"method": "spatial_block_percentile_bootstrap", "repetitions": repetitions,
                      "blocks": len(unique), "level": 0.95,
                      "mae": np.quantile(absolute[samples].sum(axis=1) / denominator, [.025, .975]).tolist(),
                      "rmse": np.quantile(np.sqrt(squared[samples].sum(axis=1) / denominator), [.025, .975]).tolist()}
    by_group = []
    for key in ("crop", "region", "campaign"):
        for label in sorted(data[key].unique()):
            mask = (data[key] == label).to_numpy()
            by_group.append({"dimension": key, "group": str(label), **_metrics(observed[mask], predicted[mask])})
    # Return all test points for exports; endpoint consumers may render a subset.
    points = [{"record_id": int(idx), "observed": float(y), "predicted": float(p),
               "residual": float(y - p), "group": str(group)}
              for idx, y, p, group in zip(test, observed, predicted, groups)]
    return {"synthetic": bool(metadata.get("synthetic_demo")),
            "metrics": metrics, "baseline": _metrics(observed, baseline), "confidence": confidence,
            "interval": interval, "groups": by_group, "points": points,
            "interpretation": f"MAE={metrics['mae']:.3f} t/ha; residual = observado menos predicho. "
                              "El intervalo de confianza de la métrica no es un intervalo de predicción.",
            "warnings": (["DEMOSTRACIÓN SINTÉTICA: sin validez científica"] if metadata.get("synthetic_demo") else []) +
                        ["Comparación retrospectiva, sin garantía de transferencia a Perú",
                         "Bloques espaciales cercanos pueden ser dependientes"],
            "protocol": {k: metadata.get(k) for k in ("run_id", "source_sha256", "seed", "validation",
                         "split_info", "best_parameters", "hyperparameters", "versions")}}


def permutation(artifact, source, repeats=3, max_rows=256):
    predictor, metadata, frame, _, _, test = context(artifact, source)
    rng = np.random.default_rng(metadata["seed"])
    selected = rng.choice(test, size=min(max_rows, len(test)), replace=False)
    data = frame.iloc[selected].copy()
    y = data.yield_t_ha.to_numpy()
    baseline = _metrics(y, predictor(data))["rmse"]
    values = []
    for column in predictor.columns:
        losses = []
        for _ in range(repeats):
            shuffled = data.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            losses.append(_metrics(y, predictor(shuffled))["rmse"] - baseline)
        values.append({"feature": column, "importance": float(np.mean(losses)),
                       "std": float(np.std(losses)), "repeats": repeats})
    return {"method": "permutation_delta_rmse", "causal": False, "n": len(data),
            "values": sorted(values, key=lambda x: x["importance"], reverse=True),
            "interpretation": "Aumento del RMSE al barajar una variable; mayor valor indica más dependencia predictiva. "
                              "Variables correlacionadas pueden compartir importancia. No usar para reajustar sobre prueba."}


def explain(artifact, source, features, permutations=8, backgrounds=4):
    predictor, metadata, frame, train, _, _ = context(artifact, source)
    if set(features) != set(predictor.columns):
        raise ValueError("Variables must match the trained model exactly")
    numeric = predictor.payload["features"]
    observation = pd.DataFrame([features])[predictor.columns]
    for column in numeric:
        observation[column] = pd.to_numeric(observation[column], errors="raise")
    if not np.isfinite(observation[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Numeric features must be finite")
    for column in CATEGORIES:
        if pd.isna(observation.iloc[0][column]) or not str(observation.iloc[0][column]).strip():
            raise ValueError("Categorical features cannot be blank")
        observation[column] = observation[column].astype(str)
    rng = np.random.default_rng(metadata["seed"])
    reference = frame.iloc[rng.choice(train, size=min(backgrounds, len(train)), replace=False)][predictor.columns].copy()
    # Impute the reference with training medians so every coalition is evaluable.
    reference[numeric] = reference[numeric].fillna(frame.iloc[train][numeric].median()).fillna(0)
    contributions = np.zeros(len(predictor.columns))
    paths, orders = [], []
    for _, background in reference.iterrows():
        for _ in range(permutations):
            order = rng.permutation(len(predictor.columns))
            current = background.to_dict()
            path = [dict(current)]
            for index in order:
                name = predictor.columns[index]
                current[name] = observation.iloc[0][name]
                path.append(dict(current))
            paths.extend(path)
            orders.append(order)
    estimates = predictor(pd.DataFrame(paths)).reshape(len(orders), len(predictor.columns) + 1)
    for values, order in zip(estimates, orders):
        contributions[order] += np.diff(values)
    contributions /= len(orders)
    base = float(estimates[:, 0].mean())
    prediction = float(predictor(observation)[0])
    return {"method": "permutation_shapley", "approximate": True, "causal": False,
            "prediction": prediction, "base_value": base, "background_n": len(reference),
            "permutations": permutations, "additivity_error": float(prediction - base - contributions.sum()),
            "values": [{"feature": name, "contribution": float(value)}
                       for name, value in zip(predictor.columns, contributions)],
            "interpretation": "Atribuciones Shapley aproximadas respecto de registros de entrenamiento. "
                              "Base + contribuciones = predicción en t/ha. No son efectos causales; "
                              "combinaciones de variables pueden ser poco plausibles si están correlacionadas."}


def infer_many(artifact, source, records):
    predictor, metadata, frame, train, calibration, test = context(artifact, source)
    if not records or len(records) > 500:
        raise ValueError("Send between 1 and 500 records")
    if any(set(record) != set(predictor.columns) for record in records):
        raise ValueError("Variables must match the trained model exactly in every record")
    data = pd.DataFrame(records)[predictor.columns]
    numeric = predictor.payload["features"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Numeric features must be finite")
    if data[list(CATEGORIES)].isna().any().any() or data[list(CATEGORIES)].astype(str).apply(lambda c: c.str.strip().eq("")).any().any():
        raise ValueError("Categorical features cannot be blank")
    data[list(CATEGORIES)] = data[list(CATEGORIES)].astype(str)
    estimates = predictor(data)
    interval = interval_info(predictor, metadata, frame, calibration)
    if interval["available"]:
        assess_coverage(interval, frame.iloc[test].yield_t_ha.to_numpy() - predictor(frame.iloc[test]))
    low, high = frame.iloc[train][numeric].min(), frame.iloc[train][numeric].max()
    domain = {key: set(frame.iloc[train][key].astype(str)) for key in CATEGORIES}
    results = []
    for index, (_, row) in enumerate(data.iterrows()):
        warnings = [f"{name}: fuera del rango de entrenamiento" for name in numeric
                    if row[name] < low[name] or row[name] > high[name]]
        warnings += [f"{name}: categoría no vista en entrenamiento" for name in CATEGORIES if str(row[name]) not in domain[name]]
        if interval["available"] and not interval.get("empirical_target_met", True):
            warnings.append("Intervalo experimental: la cobertura de prueba fue inferior al objetivo nominal")
        if estimates[index] < 0:
            warnings.append("Rendimiento negativo: predicción físicamente inválida; no se truncó silenciosamente")
        results.append({"index": index, "yield_t_ha": float(estimates[index]), "warnings": warnings,
                        "interval": [float(estimates[index] - interval["radius"]),
                                     float(estimates[index] + interval["radius"])] if interval["available"] else None})
    return {"predictions": results, "interval_protocol": interval,
            "warning": "Predicción experimental; verificar cultivo, geografía y momento de disponibilidad de las variables"}
