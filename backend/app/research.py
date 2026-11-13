"""Authenticated research workspace: exploration, diagnostics, explanations and exports."""
import json
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_db
from .models import Dataset, Experiment, ModelVersion, User
from .security import current_user, require_roles
from .ml.analysis import evaluate, explain, infer_many, permutation
from .ml.figures import figure_bytes
from .ml.pipeline import _read_dataset, inspect_csv

router = APIRouter(prefix="/api/research", tags=["research"])


@router.get("/runtime")
def runtime(user: User = Depends(current_user)):
    import importlib.util
    from importlib.metadata import version
    import platform
    return {"python": platform.python_version(), "scikit_learn": version("scikit-learn"),
            "optuna": version("optuna"), "matplotlib": version("matplotlib"),
            "temporal_models_available": importlib.util.find_spec("torch") is not None,
            "protocol": "grouped_or_temporal_nested_cv_with_reserved_calibration",
            "target_unit": "t/ha"}


def visible(db, entity, identity):
    from .main import research_condition
    row = db.scalar(select(entity).where(entity.id == identity, research_condition(entity)))
    if not row:
        raise HTTPException(404, "Registro no encontrado")
    return row


def model_source(db, model_id):
    model = visible(db, ModelVersion, model_id)
    experiment = visible(db, Experiment, model.experiment_id)
    dataset = visible(db, Dataset, experiment.dataset_id)
    return model, dataset


def checked(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


class MetadataBody(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    source: str = Field(min_length=1, max_length=50)
    license: str = Field(max_length=255)
    provenance: str = Field(max_length=4000)


@router.patch("/datasets/{dataset_id}/metadata")
def metadata(dataset_id: int, body: MetadataBody, user: User = Depends(require_roles("admin", "researcher")), db: Session = Depends(get_db)):
    from .main import event
    row = visible(db, Dataset, dataset_id)
    if not body.name.strip() or not body.source.strip():
        raise HTTPException(422, "Nombre y fuente no pueden estar vacíos")
    before = {key: getattr(row, key) for key in type(body).model_fields}
    for key, value in body.model_dump().items():
        setattr(row, key, value.strip())
    db.commit()
    event(db, "dataset.metadata", user.id, {"dataset_id": row.id, "before": before, "after": body.model_dump()})
    return {"id": row.id, **body.model_dump()}


@router.get("/datasets/{dataset_id}/profile")
def profile(dataset_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    row = visible(db, Dataset, dataset_id)
    def build():
        frame, features = _read_dataset(row.file_path)
        result = inspect_csv(row.file_path)
        histograms = {}
        for column in ["yield_t_ha", *features]:
            counts, edges = np.histogram(frame[column].dropna().to_numpy(), bins=12)
            histograms[column] = [{"x": float((edges[i] + edges[i + 1]) / 2), "count": int(value)}
                                  for i, value in enumerate(counts)]
        correlations = frame[["yield_t_ha", *features]].corr()["yield_t_ha"].drop("yield_t_ha")
        result.update({"histograms": histograms,
                       "correlations": [{"feature": key, "correlation": float(value) if pd.notna(value) else None}
                                        for key, value in correlations.items()],
                       "groups": {key: frame[key].value_counts().to_dict() for key in ("crop", "region", "campaign")},
                       "preview": json.loads(frame.head(30).to_json(orient="records")),
                       "interpretation": "Frecuencias, faltantes y correlaciones describen esta muestra. "
                                         "Una correlación no implica causalidad ni utilidad fuera de esta población."})
        return result
    return checked(build)


@router.get("/models/{model_id}/evaluation")
def evaluation(model_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    model, dataset = model_source(db, model_id)
    return checked(evaluate, model.artifact_path, dataset.file_path)


@router.get("/models/{model_id}/permutation")
def importance(model_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    model, dataset = model_source(db, model_id)
    return checked(permutation, model.artifact_path, dataset.file_path)


class ExplainBody(BaseModel):
    features: dict[str, float | str] | None = None
    record_id: int | None = Field(default=None, ge=0)


@router.post("/models/{model_id}/explain")
def explanation(model_id: int, body: ExplainBody, user: User = Depends(current_user), db: Session = Depends(get_db)):
    model, dataset = model_source(db, model_id)
    if (body.features is None) == (body.record_id is None):
        raise HTTPException(422, "Envíe variables o un registro, exclusivamente")
    features = body.features
    if body.record_id is not None:
        from .prediction_inputs import input_record
        _, _, features, _ = input_record(db, model, body.record_id)
    return checked(explain, model.artifact_path, dataset.file_path, features)


class BatchBody(BaseModel):
    records: list[dict[str, float | str]] = Field(min_length=1, max_length=500)


@router.post("/models/{model_id}/batch")
def batch(model_id: int, body: BatchBody, user: User = Depends(require_roles("admin", "researcher", "analyst")), db: Session = Depends(get_db)):
    from .main import event
    model, dataset = model_source(db, model_id)
    result = checked(infer_many, model.artifact_path, dataset.file_path, body.records)
    event(db, "prediction.batch", user.id, {"model_id": model.id, "rows": len(body.records)})
    result["synthetic"] = model.synthetic
    return result


@router.get("/models/{model_id}/template.csv")
def template(model_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    model, _ = model_source(db, model_id)
    return Response(pd.DataFrame(columns=model.feature_names).to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="model-{model_id}-template.csv"'})


@router.get("/models/{model_id}/figure/{kind}.png")
def figure(model_id: int, kind: Literal["residuals", "observed"], user: User = Depends(current_user), db: Session = Depends(get_db)):
    model, dataset = model_source(db, model_id)
    result = checked(evaluate, model.artifact_path, dataset.file_path)
    return Response(figure_bytes(result, kind), media_type="image/png",
                    headers={"Content-Disposition": f'attachment; filename="model-{model_id}-{kind}-600dpi.png"'})


@router.get("/models/{model_id}/report.{ext}")
def report(model_id: int, ext: Literal["json", "pdf", "docx", "xlsx"], user: User = Depends(current_user), db: Session = Depends(get_db)):
    from .research_reports import render_report
    from .main import event
    model, dataset = model_source(db, model_id)
    result = checked(evaluate, model.artifact_path, dataset.file_path)
    result["importance"] = checked(permutation, model.artifact_path, dataset.file_path)
    result["dataset"] = {"id": dataset.id, "name": dataset.name, "source": dataset.source,
                         "license": dataset.license, "provenance": dataset.provenance, "synthetic": dataset.synthetic}
    result["model"] = {"id": model.id, "type": model.model_type}
    mime = {"json": "application/json", "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    payload = render_report(result, ext)
    event(db, "research.report", user.id, {"model_id": model.id, "format": ext})
    return Response(payload, media_type=mime[ext],
                    headers={"Content-Disposition": f'attachment; filename="model-{model_id}-research.{ext}"'})


class CompareBody(BaseModel):
    model_ids: list[int] = Field(min_length=2, max_length=5)


@router.post("/compare")
def compare(body: CompareBody, user: User = Depends(current_user), db: Session = Depends(get_db)):
    from scipy.stats import friedmanchisquare, wilcoxon
    from .ml.statistics import _holm_adjust
    if len(set(body.model_ids)) != len(body.model_ids):
        raise HTTPException(422, "Seleccione modelos distintos")
    evaluations, models = [], []
    for identity in body.model_ids:
        model, dataset = model_source(db, identity)
        evaluations.append(checked(evaluate, model.artifact_path, dataset.file_path))
        models.append(model)
    first = evaluations[0]
    signature = lambda r: (r["protocol"]["source_sha256"], r["protocol"]["seed"], r["protocol"]["validation"],
                           [p["record_id"] for p in r["points"]])
    if any(signature(r) != signature(first) for r in evaluations[1:]):
        raise HTTPException(422, "Los modelos deben compartir dataset, semilla y prueba")
    strategies = [r["protocol"]["split_info"]["strategy"] for r in evaluations]
    if len(set(strategies)) != 1:
        raise HTTPException(422, "Los modelos deben compartir protocolo de entrenamiento")
    groups = sorted({p["group"] for p in first["points"]})
    if len(groups) < 3:
        raise HTTPException(422, "Se requieren al menos tres bloques espaciales")
    losses = np.array([[np.mean([abs(p["residual"]) for p in r["points"] if p["group"] == group])
                        for r in evaluations] for group in groups])
    pairwise = []
    for a in range(len(models)):
        for b in range(a + 1, len(models)):
            difference = losses[:, a] - losses[:, b]
            p = 1.0 if np.allclose(difference, 0) else float(wilcoxon(difference).pvalue)
            pairwise.append({"model_a": models[a].id, "model_b": models[b].id, "p": p,
                             "mean_mae_difference": float(difference.mean())})
    for row, adjusted in zip(pairwise, _holm_adjust([p["p"] for p in pairwise])):
        row["p_holm"] = adjusted
    friedman = None
    if len(models) >= 3 and not np.allclose(losses, losses[:, :1]):
        statistic, p = friedmanchisquare(*losses.T)
        if np.isfinite(statistic) and np.isfinite(p):
            friedman = {"statistic": float(statistic), "p": float(p)}
    return {"blocks": len(groups), "pairwise": pairwise, "friedman": friedman,
            "interpretation": "Wilcoxon sobre MAE por bloque espacial, ajuste de Holm entre parejas. "
                              "Friedman requiere al menos tres modelos. Diferencia negativa favorece A; "
                              "dependencia entre regiones limita la interpretación."}


@router.get("/search-space")
def search_space(user: User = Depends(current_user)):
    from .ml.tuning import RANGES
    return RANGES


@router.get("/experiments/{experiment_id}/trials")
def trials(experiment_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    import optuna
    from .main import ARTIFACT_DIR
    visible(db, Experiment, experiment_id)
    directory = ARTIFACT_DIR / f"search-{experiment_id}"
    database = directory / "trials.sqlite3"
    if not database.exists():
        return {"studies": []}
    storage = "sqlite:///" + database.resolve().as_posix().replace("{", "{{").replace("}", "}}")
    studies = []
    for summary in optuna.get_all_study_summaries(storage=storage):
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        values = [{"number": t.number, "state": t.state.name, "value": t.value,
                   "params": t.params, "seconds": t.duration.total_seconds() if t.duration else None}
                  for t in study.trials]
        importance = {}
        if sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) >= 5:
            try:
                importance = optuna.importance.get_param_importances(study)
            except ValueError:
                pass
        studies.append({"name": study.study_name, "trials": values, "importance": importance})
    return {"studies": studies, "interpretation": "Menor RMSE interno es mejor. "
            "La importancia de hiperparámetros requiere ensayos suficientes y no es importancia de variables agrícolas."}


@router.post("/experiments/{experiment_id}/resume", status_code=202)
def resume_experiment(experiment_id: int, tasks: BackgroundTasks, user: User = Depends(require_roles("admin", "researcher")), db: Session = Depends(get_db)):
    from sqlalchemy import func
    from .main import ARTIFACT_DIR, run_training, research_condition, event
    row = visible(db, Experiment, experiment_id)
    if row.status not in {"cancelled", "failed"}:
        raise HTTPException(409, "Solo puede reanudar búsquedas canceladas o fallidas")
    request = (row.split_info or {}).get("request", {})
    if not request.get("tuning_trials") or not (ARTIFACT_DIR / f"search-{row.id}" / "search-control.json").exists():
        raise HTTPException(409, "No hay búsqueda persistida que reanudar")
    active = db.scalar(select(func.count()).select_from(Experiment).where(
        research_condition(Experiment), Experiment.status.in_(["queued", "running", "cancel_requested"])))
    if active >= 2:
        raise HTTPException(429, "Límite de dos entrenamientos simultáneos")
    row.status = "queued"; row.error = None; row.completed_at = None; db.commit()
    event(db, "experiment.resumed", user.id, {"experiment_id": row.id})
    tasks.add_task(run_training, row.id, request.get("epochs", 5), request["tuning_trials"],
                   request.get("outer_folds", 5), request.get("inner_folds", 3),
                   request.get("search_ranges"), request.get("hyperparameters"), True)
    return {"id": row.id, "status": row.status}
