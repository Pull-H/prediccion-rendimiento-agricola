import csv
import os
import secrets
import time
from collections import deque
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from threading import Lock
from typing import Literal

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from .db import Base, SessionLocal, engine, get_db
from .models import Audit, Dataset, Experiment, GeneratedReport, ModelVersion, Parcel, Prediction, SessionToken, User, now
from .reports import report_bytes
from .security import create_token, current_user, ensure_secret, hasher, require_roles, user_json, ROLES
from .publication import router as publication_router

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "./artifacts"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"
DEMO_DATA_ENABLED = os.getenv("ENABLE_DEMO_DATA", "false").lower() == "true"
_login_attempts: dict[tuple[str, str], deque[float]] = {}
_login_lock = Lock()
app = FastAPI(title="Plataforma de predicción agrícola", version="0.1.0")
app.include_router(publication_router)
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST", "PATCH"], allow_headers=["Content-Type", "X-CSRF-Token"])

@app.on_event("startup")
def startup():
    ensure_secret()
    # Alembic is the deployment path; create_all supports a lightweight local development run.
    if os.getenv("AUTO_CREATE_DB", "false").lower() == "true":
        Base.metadata.create_all(engine)

def event(db: Session, action: str, user_id: int | None = None, detail: dict | None = None):
    db.add(Audit(user_id=user_id, action=action, detail=detail or {}))
    db.commit()

def as_json(row, fields):
    return {name: getattr(row, name) for name in fields}


def research_condition(entity):
    """Keep synthetic fixtures out of the research application; retain their history."""
    if DEMO_DATA_ENABLED:
        return True
    dataset_ids = select(Dataset.id).where(Dataset.synthetic.is_(False))
    experiment_ids = select(Experiment.id).where(Experiment.dataset_id.in_(dataset_ids))
    model_ids = select(ModelVersion.id).where(ModelVersion.experiment_id.in_(experiment_ids))
    return {
        Dataset: Dataset.synthetic.is_(False),
        Parcel: Parcel.dataset_id.in_(dataset_ids),
        Experiment: Experiment.dataset_id.in_(dataset_ids),
        ModelVersion: ModelVersion.experiment_id.in_(experiment_ids),
        Prediction: Prediction.model_id.in_(model_ids),
    }[entity]

USER_FIELDS = ("id", "email", "name", "role", "active", "created_at")
DATASET_FIELDS = ("id", "name", "source", "synthetic", "license", "provenance", "row_count", "variables", "quality", "created_at")
PARCEL_FIELDS = ("id", "dataset_id", "external_id", "crop", "region", "country", "campaign", "latitude", "longitude", "observed_yield")
EXPERIMENT_FIELDS = ("id", "dataset_id", "model_type", "status", "seed", "validation", "progress", "logs", "metrics", "split_info", "error", "created_at", "completed_at")
MODEL_FIELDS = ("id", "experiment_id", "model_type", "feature_names", "metrics", "synthetic", "approved", "created_at")
PREDICTION_FIELDS = ("id", "model_id", "user_id", "parcel_id", "input_features", "yield_t_ha", "warnings", "synthetic", "created_at")

class LoginBody(BaseModel):
    email: str
    password: str

class UserBody(BaseModel):
    email: str
    name: str
    password: str = Field(min_length=12)
    role: Literal["admin", "researcher", "analyst", "viewer"]

class ExperimentBody(BaseModel):
    dataset_id: int
    model_type: Literal["RandomForest", "XGBoost", "LSTM", "CNN-LSTM", "CNN-Transformer"]
    seed: int = 42
    validation: Literal["parcel", "campaign", "region", "temporal"] = "parcel"
    epochs: int = Field(default=5, ge=1, le=100)

class PredictionBody(BaseModel):
    model_id: int
    features: dict[str, float | str] | None = None
    parcel_id: int | None = None
    record_id: int | None = Field(default=None, ge=0)

class NavigationBody(BaseModel):
    module: str = Field(pattern=r"^[a-z-]{2,40}$")
    direction: Literal["enter", "leave"]

class AssistantBody(BaseModel):
    question: str = Field(min_length=3, max_length=500)

@app.get("/health")
def health(db: Session = Depends(get_db)):
    db.scalar(select(func.count()).select_from(User))
    return {"status": "ok", "database": "ok"}

@app.post("/api/auth/login")
def login(body: LoginBody, request: Request, response: Response, db: Session = Depends(get_db)):
    email = body.email.lower().strip()
    key = (email, request.client.host if request.client else "unknown")
    stamp = time.monotonic()
    with _login_lock:
        recent = deque(x for x in _login_attempts.get(key, ()) if stamp - x < 900)
        _login_attempts[key] = recent
        if len(recent) >= 5:
            raise HTTPException(429, "Demasiados intentos; vuelva a intentarlo más tarde")
    user = db.scalar(select(User).where(User.email == email))
    if not user or not user.active or not hasher.verify(body.password, user.password_hash):
        with _login_lock:
            _login_attempts[key].append(stamp)
        event(db, "auth.failed", detail={"email": email[:255]})
        raise HTTPException(401, "Credenciales inválidas")
    with _login_lock:
        _login_attempts.pop(key, None)
    csrf = secrets.token_urlsafe(32)
    token, jti, expiry = create_token(user)
    db.add(SessionToken(user_id=user.id, jti=jti, expires_at=expiry)); db.commit()
    response.set_cookie("session", token, max_age=28800, httponly=True, secure=COOKIE_SECURE, samesite="lax")
    response.set_cookie("csrf_token", csrf, max_age=28800, httponly=False, secure=COOKIE_SECURE, samesite="lax")
    event(db, "auth.login", user.id)
    return {"user": user_json(user)}

@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return user_json(user)

@app.post("/api/auth/logout")
def logout(request: Request, response: Response, user: User = Depends(current_user), db: Session = Depends(get_db)):
    session = db.scalar(select(SessionToken).where(SessionToken.jti == request.state.jti))
    session.revoked = True
    event(db, "auth.logout", user.id)
    response.delete_cookie("session")
    response.delete_cookie("csrf_token")
    return {"ok": True}

@app.get("/api/users")
def users(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    return [as_json(x, USER_FIELDS) for x in db.scalars(select(User).order_by(User.id)).all()]

@app.post("/api/users", status_code=201)
def create_user(body: UserBody, user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    email = body.email.lower().strip()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Correo ya registrado")
    row = User(email=email, name=body.name.strip(), role=body.role, password_hash=hasher.hash(body.password))
    db.add(row); db.commit(); db.refresh(row)
    event(db, "user.create", user.id, {"user_id": row.id, "role": row.role})
    return as_json(row, USER_FIELDS)

@app.patch("/api/users/{user_id}/role")
def set_role(user_id: int, role: Literal["admin", "researcher", "analyst", "viewer"], user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    row = db.get(User, user_id)
    if not row:
        raise HTTPException(404, "Usuario no encontrado")
    if row.id == user.id and role != "admin":
        raise HTTPException(400, "No puede retirarse su propio permiso de administrador")
    row.role = role; db.commit(); event(db, "user.role", user.id, {"user_id": user_id, "role": role})
    return as_json(row, USER_FIELDS)

@app.get("/api/dashboard")
def dashboard(user: User = Depends(current_user), db: Session = Depends(get_db)):
    status_counts = dict(db.execute(select(Experiment.status, func.count()).where(research_condition(Experiment)).group_by(Experiment.status)).all())
    models = db.scalars(select(ModelVersion).where(research_condition(ModelVersion)).order_by(desc(ModelVersion.id)).limit(5)).all()
    predictions = db.scalars(select(Prediction).where(research_condition(Prediction)).order_by(desc(Prediction.id)).limit(5)).all()
    return {
        "datasets": db.scalar(select(func.count()).select_from(Dataset).where(research_condition(Dataset))),
        "parcels": db.scalar(select(func.count()).select_from(Parcel).where(research_condition(Parcel))),
        "campaigns": db.scalar(select(func.count(func.distinct(Parcel.campaign))).where(research_condition(Parcel))),
        "experiments": status_counts,
        "models": db.scalar(select(func.count()).select_from(ModelVersion).where(research_condition(ModelVersion))),
        "approved_models": db.scalar(select(func.count()).select_from(ModelVersion).where(research_condition(ModelVersion)).where(ModelVersion.approved.is_(True))),
        "recent_models": [as_json(x, MODEL_FIELDS) for x in models],
        "recent_predictions": [as_json(x, PREDICTION_FIELDS) for x in predictions],
        "synthetic_data_present": bool(db.scalar(select(func.count()).select_from(Dataset).where(research_condition(Dataset)).where(Dataset.synthetic.is_(True)))),
    }

@app.get("/api/datasets")
def datasets(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [as_json(x, DATASET_FIELDS) for x in db.scalars(select(Dataset).where(research_condition(Dataset)).order_by(desc(Dataset.id))).all()]

def register_csv(db: Session, path: Path, name: str, source: str, synthetic: bool,
                 license_name: str | None = None, provenance_note: str | None = None):
    from .ml import inspect_csv
    info = inspect_csv(path)
    if not DEMO_DATA_ENABLED and (synthetic or info.get("synthetic_demo")):
        raise ValueError("Los datos de prueba no se admiten en el modo de investigacion")
    frame = pd.read_csv(path)
    row = Dataset(name=name, source=source, synthetic=synthetic or bool(info.get("synthetic_demo")),
                  license=license_name, provenance=provenance_note or str(path), file_path=str(path),
                  row_count=len(frame), variables=list(frame.columns), quality=info)
    db.add(row); db.flush()
    for (parcel_id, campaign), group in frame.groupby(["parcel_id", "campaign"], dropna=False):
        first = group.iloc[0]
        def optional_number(key):
            value = first.get(key)
            return None if value is None or pd.isna(value) else float(value)
        db.add(Parcel(dataset_id=row.id, external_id=str(parcel_id), campaign=str(campaign), crop=str(first["crop"]), region=str(first["region"]), country=str(first.get("country", "")) or None, latitude=optional_number("latitude"), longitude=optional_number("longitude"), observed_yield=float(group["yield_t_ha"].mean())))
    db.commit(); db.refresh(row)
    return row

@app.post("/api/datasets/import", status_code=201)
async def import_dataset(file: UploadFile = File(...),
                         source_name: str = Form(default="user_upload", max_length=80),
                         license_name: str | None = Form(default=None, max_length=255),
                         provenance_note: str | None = Form(default=None, max_length=1000),
                         user: User = Depends(require_roles("admin", "researcher")),
                         db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Se requiere un archivo CSV")
    source_name = source_name.strip()
    if not source_name:
        raise HTTPException(422, "El nombre de la fuente no puede estar vacío")
    license_name = (license_name or "").strip() or None
    provenance_note = (provenance_note or "").strip() or None
    data = await file.read(20_000_001)
    if len(data) > 20_000_000:
        raise HTTPException(413, "Máximo 20 MB")
    try:
        data.decode("utf-8-sig")
    except UnicodeError:
        raise HTTPException(400, "CSV debe estar codificado en UTF-8")
    path = DATA_DIR / f"import-{secrets.token_hex(12)}.csv"
    path.write_bytes(data)
    try:
        row = register_csv(db, path, file.filename, source_name, False, license_name, provenance_note)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, f"CSV inválido: {exc}")
    event(db, "dataset.import", user.id, {"dataset_id": row.id, "rows": row.row_count, "source": row.source})
    return as_json(row, DATASET_FIELDS)

@app.post("/api/datasets/demo", status_code=201)
def create_demo(user: User = Depends(require_roles("admin", "researcher")), db: Session = Depends(get_db)):
    if not DEMO_DATA_ENABLED:
        raise HTTPException(404, "La generacion de datos de prueba esta deshabilitada")
    from .ml import create_demo_csv
    path = DATA_DIR / f"demo-{secrets.token_hex(8)}.csv"
    create_demo_csv(path)
    row = register_csv(db, path, "Demostración sintética", "synthetic_demo", True, "Datos sintéticos generados localmente")
    event(db, "dataset.demo", user.id, {"dataset_id": row.id})
    return as_json(row, DATASET_FIELDS)

@app.get("/api/parcels")
def parcels(dataset_id: int | None = None, crop: str | None = None, region: str | None = None, user: User = Depends(current_user), db: Session = Depends(get_db)):
    stmt = select(Parcel).where(research_condition(Parcel))
    if dataset_id: stmt = stmt.where(Parcel.dataset_id == dataset_id)
    if crop: stmt = stmt.where(Parcel.crop == crop)
    if region: stmt = stmt.where(Parcel.region == region)
    return [as_json(x, PARCEL_FIELDS) for x in db.scalars(stmt.order_by(Parcel.id).limit(2000)).all()]

@app.get("/api/experiments")
def experiments(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [as_json(x, EXPERIMENT_FIELDS) for x in db.scalars(select(Experiment).where(research_condition(Experiment)).order_by(desc(Experiment.id))).all()]

@app.get("/api/experiments/{experiment_id}")
def experiment(experiment_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    row = db.scalar(select(Experiment).where(research_condition(Experiment), Experiment.id == experiment_id))
    if not row: raise HTTPException(404, "Experimento no encontrado")
    return as_json(row, EXPERIMENT_FIELDS)

def run_training(experiment_id: int, epochs: int):
    from .ml import train
    with SessionLocal() as db:
        row = db.scalar(select(Experiment).where(research_condition(Experiment), Experiment.id == experiment_id))
        if not row or row.status == "cancelled": return
        row.status = "running"; row.progress = 1; db.commit()
        dataset = db.scalar(select(Dataset).where(research_condition(Dataset), Dataset.id == row.dataset_id))
        def progress(update):
            with SessionLocal() as inner:
                task = inner.get(Experiment, experiment_id)
                if task:
                    task.progress = int(update.get("percent", task.progress))
                    task.logs = (task.logs + "\n" + str(update.get("stage", "processing")))[-10000:]
                    inner.commit()
        try:
            result = train(dataset.file_path, row.model_type, str(ARTIFACT_DIR), seed=row.seed, validation=row.validation, epochs=epochs, progress=progress)
            row.metrics = result["metrics"]
            row.split_info = result.get("split_info", {})
            row.status = "completed"; row.progress = 100; row.completed_at = now()
            db.add(ModelVersion(experiment_id=row.id, model_type=row.model_type, artifact_path=result["artifact_path"], feature_names=result["feature_names"], metrics=result["metrics"], synthetic=dataset.synthetic))
            db.commit(); event(db, "experiment.completed", detail={"experiment_id": row.id})
        except Exception as exc:
            row.status = "failed"; row.error = str(exc)[:1000]; row.completed_at = now(); db.commit()
            event(db, "experiment.failed", detail={"experiment_id": row.id, "error": row.error})

@app.post("/api/experiments", status_code=202)
def start_experiment(body: ExperimentBody, tasks: BackgroundTasks, user: User = Depends(require_roles("admin", "researcher")), db: Session = Depends(get_db)):
    if not db.scalar(select(Dataset).where(research_condition(Dataset), Dataset.id == body.dataset_id)): raise HTTPException(404, "Dataset no encontrado")
    active = db.scalar(select(func.count()).select_from(Experiment).where(research_condition(Experiment)).where(Experiment.status.in_(["queued", "running"])))
    if active >= 2: raise HTTPException(429, "Límite de dos entrenamientos simultáneos")
    row = Experiment(dataset_id=body.dataset_id, model_type=body.model_type, seed=body.seed, validation=body.validation)
    db.add(row); db.commit(); db.refresh(row)
    event(db, "experiment.started", user.id, {"experiment_id": row.id})
    tasks.add_task(run_training, row.id, body.epochs)
    return as_json(row, EXPERIMENT_FIELDS)

@app.post("/api/experiments/{experiment_id}/cancel")
def cancel_experiment(experiment_id: int, user: User = Depends(require_roles("admin", "researcher")), db: Session = Depends(get_db)):
    row = db.scalar(select(Experiment).where(research_condition(Experiment), Experiment.id == experiment_id))
    if not row: raise HTTPException(404, "Experimento no encontrado")
    if row.status != "queued": raise HTTPException(409, "Solo se puede cancelar una tarea en cola")
    row.status = "cancelled"; db.commit(); event(db, "experiment.cancelled", user.id, {"experiment_id": row.id})
    return as_json(row, EXPERIMENT_FIELDS)

@app.get("/api/models")
def models(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [as_json(x, MODEL_FIELDS) for x in db.scalars(select(ModelVersion).where(research_condition(ModelVersion)).order_by(desc(ModelVersion.id))).all()]

@app.get("/api/statistics/compare")
def compare_model_statistics(dataset_id: int, validation: Literal["parcel", "campaign", "region", "temporal"],
                             seed: int = 42, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Friedman, paired Wilcoxon-Holm and spatial-block bootstrap on one held-out split."""
    from .ml.statistics import compare_models

    dataset = db.scalar(select(Dataset).where(research_condition(Dataset), Dataset.id == dataset_id))
    if not dataset:
        raise HTTPException(404, "Dataset no encontrado")
    candidates = db.execute(
        select(ModelVersion, Experiment).join(Experiment, ModelVersion.experiment_id == Experiment.id)
        .where(Experiment.dataset_id == dataset_id, Experiment.validation == validation,
               Experiment.seed == seed, Experiment.status == "completed")
        .order_by(desc(ModelVersion.id))
    ).all()
    artifacts = {}
    versions = {}
    for model, experiment in candidates:
        if model.model_type not in artifacts:
            artifacts[model.model_type] = model.artifact_path
            versions[model.model_type] = {"model_id": model.id, "experiment_id": experiment.id,
                                          "created_at": model.created_at.isoformat()}
    required = ("RandomForest", "XGBoost", "LSTM", "CNN-LSTM", "CNN-Transformer")
    missing = [name for name in required if name not in artifacts]
    if missing:
        raise HTTPException(409, {"message": "Faltan modelos comparables", "missing_models": missing})
    try:
        result = compare_models(dataset.file_path, {name: artifacts[name] for name in required}, validation, seed)
        result.update({"dataset_id": dataset.id, "dataset_name": dataset.name, "source": dataset.source,
                       "model_versions": {name: versions[name] for name in required}})
        return result
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(422, f"No se pudo comparar los modelos: {exc}") from exc

@app.get("/api/models/{model_id}/importance")
def model_importance(model_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    import joblib
    model = db.scalar(select(ModelVersion).where(research_condition(ModelVersion), ModelVersion.id == model_id))
    if not model: raise HTTPException(404, "Modelo no encontrado")
    if model.model_type not in {"RandomForest", "XGBoost"}:
        raise HTTPException(501, "La importancia global para este modelo profundo aún no está implementada")
    artifact = joblib.load(model.artifact_path)
    features = artifact["preprocessor"].get_feature_names_out()
    importances = artifact["model"].feature_importances_
    values = sorted(({"feature": str(name), "importance": float(value)} for name, value in zip(features, importances)), key=lambda item: item["importance"], reverse=True)
    return {"model_id": model_id, "method": "model_feature_importances", "synthetic": model.synthetic, "causal": False, "values": values[:30]}

@app.post("/api/models/{model_id}/approve")
def approve_model(model_id: int, user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    row = db.scalar(select(ModelVersion).where(research_condition(ModelVersion), ModelVersion.id == model_id))
    if not row: raise HTTPException(404, "Modelo no encontrado")
    if row.synthetic: raise HTTPException(409, "Un modelo de demostración no se puede aprobar para uso científico")
    row.approved = True; db.commit(); event(db, "model.approve", user.id, {"model_id": model_id})
    return as_json(row, MODEL_FIELDS)

@app.get("/api/predictions")
def predictions(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [as_json(x, PREDICTION_FIELDS) for x in db.scalars(select(Prediction).where(research_condition(Prediction)).order_by(desc(Prediction.id)).limit(100)).all()]

@app.get("/api/models/{model_id}/prediction-inputs")
def prediction_inputs(model_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    from .prediction_inputs import input_context
    model = db.scalar(select(ModelVersion).where(research_condition(ModelVersion), ModelVersion.id == model_id))
    if not model:
        raise HTTPException(404, "Modelo no encontrado")
    dataset, frame, roles = input_context(db, model)
    return {"model_id": model.id, "dataset_id": dataset.id, "dataset_name": dataset.name,
            "source": dataset.source, "synthetic": dataset.synthetic,
            "feature_count": len(model.feature_names),
            "records": [{"record_id": int(index), "external_id": row["parcel_id"],
                         "campaign": row["campaign"], "crop": row["crop"], "region": row["region"],
                         "evaluation_role": roles[int(index)]} for index, row in frame.iterrows()]}


@app.get("/api/models/{model_id}/prediction-inputs/{record_id}")
def prediction_input(model_id: int, record_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    from .prediction_inputs import input_record
    model = db.scalar(select(ModelVersion).where(research_condition(ModelVersion), ModelVersion.id == model_id))
    if not model:
        raise HTTPException(404, "Modelo no encontrado")
    dataset, row, features, role = input_record(db, model, record_id)
    return {"model_id": model.id, "dataset_id": dataset.id, "record_id": record_id,
            "external_id": row["parcel_id"], "campaign": row["campaign"],
            "evaluation_role": role, "features": features}


@app.post("/api/predictions", status_code=201)
def make_prediction(body: PredictionBody, user: User = Depends(require_roles("admin", "researcher", "analyst")), db: Session = Depends(get_db)):
    from .ml import categorical_domain_warnings, predict
    model = db.scalar(select(ModelVersion).where(research_condition(ModelVersion), ModelVersion.id == body.model_id))
    if not model: raise HTTPException(404, "Modelo no encontrado")
    historical_role = None
    if body.record_id is not None:
        from .prediction_inputs import input_record
        if body.features is not None or body.parcel_id is not None:
            raise HTTPException(422, "Seleccione un registro automático o envíe variables manuales, no ambos")
        source_dataset, source_row, body.features, historical_role = input_record(db, model, body.record_id)
        parcel = db.scalar(select(Parcel).where(research_condition(Parcel)).where(
            Parcel.dataset_id == source_dataset.id, Parcel.external_id == source_row["parcel_id"],
            Parcel.campaign == source_row["campaign"]))
        if not parcel:
            raise HTTPException(409, "La parcela del registro no está disponible")
        body.parcel_id = parcel.id
    if body.features is None:
        raise HTTPException(422, "Seleccione un registro o proporcione las variables del modelo")
    if set(body.features) != set(model.feature_names):
        raise HTTPException(422, {"required_features": model.feature_names, "message": "Las variables deben coincidir exactamente con el entrenamiento"})
    if body.parcel_id and not db.scalar(select(Parcel).where(research_condition(Parcel), Parcel.id == body.parcel_id)): raise HTTPException(404, "Parcela no encontrada")
    value = float(predict(model.artifact_path, body.features))
    warnings = ["Resultado de demostración sintética; sin validez científica"] if model.synthetic else []
    warnings.extend(categorical_domain_warnings(model.artifact_path, body.features))
    if historical_role:
        labels = {"train": "entrenamiento", "validation": "validación", "test": "prueba"}
        warnings.append(f"Predicción sobre un registro histórico de {labels[historical_role]}; no es un pronóstico de una campaña nueva")
    experiment = db.scalar(select(Experiment).where(research_condition(Experiment), Experiment.id == model.experiment_id))
    dataset = db.scalar(select(Dataset).where(research_condition(Dataset), Dataset.id == experiment.dataset_id)) if experiment else None
    if dataset and dataset.source.startswith("YieldSAT wheat Germany 240d satellite+soil"):
        warnings.append("Piloto de trigo de Alemania a 240 días de siembra, con satélite y suelo; sin clima integrado ni validación en Perú")
    if dataset and dataset.quality.get("spatial_unit") == "administrative_region":
        warnings.append("Modelo evaluado a escala regional; no validado para parcelas o subparcelas")
        if "CY-Bench" in dataset.source or "CY-Bench" in (dataset.provenance or ""):
            warnings.append("Datos de maíz de España; la transferencia a Perú no está validada")
    if dataset and "latitude" in body.features and "longitude" in body.features:
        min_lat, max_lat, min_lon, max_lon = db.execute(
            select(func.min(Parcel.latitude), func.max(Parcel.latitude),
                   func.min(Parcel.longitude), func.max(Parcel.longitude))
            .where(Parcel.dataset_id == dataset.id)
        ).one()
        if all(bound is not None for bound in (min_lat, max_lat, min_lon, max_lon)):
            latitude, longitude = float(body.features["latitude"]), float(body.features["longitude"])
            if not (min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon):
                warnings.append("Coordenadas fuera del rango geográfico observado en el dataset")
    row = Prediction(model_id=model.id, user_id=user.id, parcel_id=body.parcel_id, input_features=body.features, yield_t_ha=value, warnings=warnings, synthetic=model.synthetic)
    db.add(row); db.commit(); db.refresh(row)
    event(db, "prediction.create", user.id, {"prediction_id": row.id, "model_id": model.id})
    return as_json(row, PREDICTION_FIELDS)

@app.get("/api/audit")
def audit(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rows = db.scalars(select(Audit).order_by(desc(Audit.id)).limit(200)).all()
    return [as_json(x, ("id", "user_id", "action", "detail", "created_at")) for x in rows]

@app.post("/api/audit/navigation")
def navigation(body: NavigationBody, user: User = Depends(current_user), db: Session = Depends(get_db)):
    event(db, f"navigation.{body.direction}", user.id, {"module": body.module})
    return {"ok": True}

@app.post("/api/assistant/ask")
def ask_assistant(body: AssistantBody, user: User = Depends(current_user), db: Session = Depends(get_db)):
    question = body.question.lower()
    if any(word in question for word in ("modelo", "métrica", "mae", "rmse", "rendimiento")):
        models = db.scalars(select(ModelVersion).where(research_condition(ModelVersion)).order_by(desc(ModelVersion.id)).limit(5)).all()
        if not models:
            return {"answer": "No hay modelos entrenados registrados. No puedo informar métricas todavía.", "sources": [], "mode": "consulta estructurada"}
        lines = []
        for model in models:
            scope = "demostración sintética" if model.synthetic else "datos importados"
            lines.append(f"Modelo {model.id} ({model.model_type}, {scope}): {model.metrics.get('test', {})}")
        return {"answer": "\n".join(lines), "sources": [f"model_versions:{x.id}" for x in models], "mode": "consulta estructurada"}
    if any(word in question for word in ("dataset", "datos", "fuente", "variable")):
        datasets = db.scalars(select(Dataset).where(research_condition(Dataset)).order_by(desc(Dataset.id)).limit(5)).all()
        if not datasets:
            return {"answer": "No hay datasets registrados.", "sources": [], "mode": "consulta estructurada"}
        return {"answer": "\n".join(f"Dataset {x.id}: {x.name}; {x.row_count} registros; {'sintético' if x.synthetic else 'importado'}; variables: {', '.join(x.variables)}" for x in datasets), "sources": [f"datasets:{x.id}" for x in datasets], "mode": "consulta estructurada"}
    return {"answer": "Solo puedo consultar datasets y resultados registrados. Reformula la pregunta para referirse a un dataset, variable, modelo o métrica.", "sources": [], "mode": "consulta estructurada"}

@app.get("/api/reports/{kind}.{ext}")
def report(kind: Literal["operational", "managerial", "scientific"], ext: Literal["pdf", "docx", "xlsx"], user: User = Depends(current_user), db: Session = Depends(get_db)):
    dataset_rows = db.scalars(select(Dataset).where(research_condition(Dataset)).order_by(Dataset.id)).all()
    experiment_rows = db.scalars(select(Experiment).where(research_condition(Experiment)).order_by(Experiment.id)).all()
    dataset_count = len(dataset_rows)
    synthetic_count = sum(x.synthetic for x in dataset_rows)
    model_rows = db.scalars(select(ModelVersion).where(research_condition(ModelVersion)).order_by(ModelVersion.id)).all()
    dataset_label = f"Datasets: {dataset_count}"
    if DEMO_DATA_ENABLED:
        dataset_label += f" (sintéticos: {synthetic_count})"
    lines = [dataset_label, f"Parcelas: {db.scalar(select(func.count()).select_from(Parcel).where(research_condition(Parcel)))}", f"Experimentos: {db.scalar(select(func.count()).select_from(Experiment).where(research_condition(Experiment)))}", f"Predicciones: {db.scalar(select(func.count()).select_from(Prediction).where(research_condition(Prediction)))}"]
    for model in model_rows:
        label = "DEMOSTRACIÓN SINTÉTICA" if model.synthetic else "DATOS IMPORTADOS"
        lines.append(f"Modelo {model.id}: {model.model_type} | {label} | {model.metrics}")
    if kind == "operational":
        lines.extend(f"Tarea {x.id}: {x.model_type}; estado={x.status}; error={x.error or 'ninguno'}" for x in experiment_rows)
        lines.append(f"Eventos de auditoría: {db.scalar(select(func.count()).select_from(Audit))}")
    if kind == "scientific":
        for dataset in dataset_rows:
            lines.append(f"Dataset {dataset.id}: {dataset.name}; origen={dataset.source}; licencia={dataset.license or 'no declarada'}; procedencia={dataset.provenance or 'no declarada'}; sintético={dataset.synthetic}; filas={dataset.row_count}; variables={', '.join(dataset.variables)}")
        for experiment in experiment_rows:
            lines.append(f"Experimento {experiment.id}: {experiment.model_type}; dataset={experiment.dataset_id}; semilla={experiment.seed}; validación={experiment.validation}; partición={experiment.split_info or {}}; métricas={experiment.metrics or {}}")
        if DEMO_DATA_ENABLED:
            lines.append("Los resultados sintéticos no constituyen evidencia científica.")
        lines.extend(["La transferencia a Perú requiere validación local.", "Importancia predictiva no implica causalidad."])
    data = report_bytes(kind, ext, lines)
    db.add(GeneratedReport(user_id=user.id, kind=kind, format=ext)); db.commit()
    event(db, "report.download", user.id, {"kind": kind, "format": ext})
    mime = {"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}[ext]
    return StreamingResponse(BytesIO(data), media_type=mime, headers={"Content-Disposition": f'attachment; filename="report-{kind}.{ext}"'})
