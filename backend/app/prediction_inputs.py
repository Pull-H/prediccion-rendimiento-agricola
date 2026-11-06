"""Load exact historical input records, excluding observed yield from predictors."""

import json
from pathlib import Path

from fastapi import HTTPException

from .ml.pipeline import CATEGORIES, _read_dataset, _sha256, _split
from .models import Dataset, Experiment


def input_context(db, model):
    experiment = db.get(Experiment, model.experiment_id)
    dataset = db.get(Dataset, experiment.dataset_id) if experiment else None
    if not dataset:
        raise HTTPException(404, "Dataset del modelo no encontrado")
    try:
        metadata = json.loads(Path(model.artifact_path).with_suffix(".json").read_text(encoding="utf-8"))
        if _sha256(dataset.file_path) != metadata["source_sha256"]:
            raise ValueError("El CSV cambió desde el entrenamiento")
        frame, numeric = _read_dataset(dataset.file_path)
        if set(model.feature_names) != set(numeric + list(CATEGORIES)):
            raise ValueError("Las variables del dataset no coinciden con el modelo")
        train, validation, test, _ = _split(frame, experiment.validation, experiment.seed)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, f"No se pueden cargar las variables guardadas: {exc}") from exc
    roles = {int(index): role for role, indices in
             (("train", train), ("validation", validation), ("test", test)) for index in indices}
    return dataset, frame, roles


def input_record(db, model, record_id):
    dataset, frame, roles = input_context(db, model)
    if not 0 <= record_id < len(frame):
        raise HTTPException(404, "Registro de entrada no encontrado")
    row = frame.iloc[record_id]
    if row[model.feature_names].isna().any():
        raise HTTPException(409, "El registro tiene variables faltantes; seleccione otro registro completo")
    features = {name: str(row[name]) if name in CATEGORIES else float(row[name])
                for name in model.feature_names}
    return dataset, row, features, roles[record_id]
