import os
from math import isclose
import tempfile
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

root = Path(tempfile.mkdtemp(prefix="agri-api-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(root / 'test.db').as_posix()}"
os.environ["DATA_DIR"] = str(root / "data")
os.environ["ARTIFACT_DIR"] = str(root / "artifacts")
os.environ["JWT_SECRET"] = "test-only-secret-with-more-than-32-characters"
os.environ["ENABLE_DEMO_DATA"] = "true"

from fastapi.testclient import TestClient
from app.db import Base, SessionLocal, engine
from app.main import app
from app.models import User
from app.security import hasher

Base.metadata.create_all(engine)
with SessionLocal() as db:
    db.add(User(email="admin@example.test", name="Admin", role="admin", password_hash=hasher.hash("strong-password-123")))
    db.add(User(email="viewer@example.test", name="Viewer", role="viewer", password_hash=hasher.hash("strong-password-123")))
    db.commit()

def login(client, email="admin@example.test"):
    response = client.post("/api/auth/login", json={"email": email, "password": "strong-password-123"})
    assert response.status_code == 200
    return {"X-CSRF-Token": client.cookies.get("csrf_token")}

def test_auth_rbac_and_audit():
    with TestClient(app) as client:
        assert client.get("/api/dashboard").status_code == 401
        assert client.post("/api/auth/login", json={"email": "admin@example.test", "password": "wrong"}).status_code == 401
        headers = login(client, "viewer@example.test")
        assert client.post("/api/datasets/demo", headers=headers).status_code == 403
        assert client.get("/api/audit").status_code == 403
        old_token = client.cookies.get("session")
        assert client.post("/api/auth/logout", headers=headers).status_code == 200
        client.cookies.set("session", old_token)
        assert client.get("/api/auth/me").status_code == 401
        client.cookies.delete("session")
        headers = login(client)
        assert client.post("/api/audit/navigation", json={"module": "dashboard", "direction": "enter"}, headers=headers).status_code == 200
        events = client.get("/api/audit").json()
        assert any(x["action"] == "auth.failed" for x in events)
        assert any(x["action"] == "navigation.enter" for x in events)

def test_demo_train_predict_and_reports():
    with TestClient(app) as client:
        headers = login(client)
        dataset_response = client.post("/api/datasets/demo", headers=headers)
        assert dataset_response.status_code == 201, dataset_response.text
        dataset = dataset_response.json()
        assert dataset["synthetic"] is True
        assert dataset["row_count"] > 0
        assert client.get("/api/parcels").json()
        run = client.post("/api/experiments", headers=headers, json={"dataset_id": dataset["id"], "model_type": "RandomForest", "seed": 42})
        assert run.status_code == 202, run.text
        experiment = client.get(f"/api/experiments/{run.json()['id']}").json()
        assert experiment["status"] == "completed", experiment.get("error")
        model = client.get("/api/models").json()[0]
        comparison = client.get(f"/api/statistics/compare?dataset_id={dataset['id']}&validation=parcel&seed=42")
        assert comparison.status_code == 409
        assert "XGBoost" in comparison.json()["detail"]["missing_models"]
        importance = client.get(f"/api/models/{model['id']}/importance")
        assert importance.status_code == 200
        assert importance.json()["values"]
        assert importance.json()["causal"] is False
        row = __import__("pandas").read_csv(__import__("app.main", fromlist=["DATA_DIR"]).DATA_DIR / next(x.name for x in __import__("app.main", fromlist=["DATA_DIR"]).DATA_DIR.iterdir() if x.name.startswith("demo-"))).iloc[0]
        features = {name: (str(row[name]) if name in {"crop", "region", "campaign"} else float(row[name])) for name in model["feature_names"]}
        predicted = client.post("/api/predictions", headers=headers, json={"model_id": model["id"], "features": features})
        assert predicted.status_code == 201, predicted.text
        assert predicted.json()["synthetic"] is True
        assert predicted.json()["warnings"]
        options = client.get(f"/api/models/{model['id']}/prediction-inputs")
        assert options.status_code == 200, options.text
        assert options.json()["dataset_id"] == dataset["id"]
        assert len(options.json()["records"]) == dataset["row_count"]
        assert {item["evaluation_role"] for item in options.json()["records"]} == {"train", "validation", "test"}
        assert "yield_t_ha" not in options.text and "observed_yield" not in options.text
        detail = client.get(f"/api/models/{model['id']}/prediction-inputs/0")
        assert detail.status_code == 200, detail.text
        assert detail.json()["features"] == features
        automatic = client.post("/api/predictions", headers=headers,
                                json={"model_id": model["id"], "record_id": 0})
        assert automatic.status_code == 201, automatic.text
        assert isclose(automatic.json()["yield_t_ha"], predicted.json()["yield_t_ha"], abs_tol=1e-10)
        parcel = next(p for p in client.get("/api/parcels").json() if p["id"] == automatic.json()["parcel_id"])
        assert parcel["dataset_id"] == dataset["id"] and parcel["external_id"] == str(row.parcel_id)
        assert any("histórico" in warning for warning in automatic.json()["warnings"])
        assert client.get(f"/api/models/{model['id']}/prediction-inputs/999999").status_code == 404
        assert client.post("/api/predictions", headers=headers, json={"model_id": model["id"]}).status_code == 422
        assert client.post("/api/predictions", headers=headers, json={
            "model_id": model["id"], "record_id": 0, "features": features}).status_code == 422
        from app.models import Dataset
        with SessionLocal() as db:
            source_path = Path(db.get(Dataset, dataset["id"]).file_path)
        original = source_path.read_bytes()
        try:
            source_path.write_bytes(original + b"\n")
            assert client.get(f"/api/models/{model['id']}/prediction-inputs").status_code == 409
            assert client.post("/api/predictions", headers=headers, json={
                "model_id": model["id"], "record_id": 0}).status_code == 409
        finally:
            source_path.write_bytes(original)
        for ext, signature in (("pdf", b"%PDF"), ("docx", b"PK"), ("xlsx", b"PK")):
            response = client.get(f"/api/reports/scientific.{ext}")
            assert response.status_code == 200
            assert response.content.startswith(signature)

def test_bootstrap_handles_windows_line_ending(monkeypatch):
    from app import bootstrap
    monkeypatch.setattr("sys.argv", ["bootstrap", "--email", "new-admin@example.test"])
    monkeypatch.setattr(bootstrap.getpass, "getpass", lambda prompt: "strong-password-456\r")
    bootstrap.main()
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == "new-admin@example.test").one()
        assert hasher.verify("strong-password-456", user.password_hash)

def test_login_rate_limit():
    with TestClient(app) as client:
        for _ in range(5):
            response = client.post("/api/auth/login", json={"email": "unknown@example.test", "password": "wrong"})
            assert response.status_code == 401
        response = client.post("/api/auth/login", json={"email": "unknown@example.test", "password": "wrong"})
        assert response.status_code == 429


def test_import_csv_and_reject_invalid_schema():
    with TestClient(app) as client:
        headers = login(client)
        csv_bytes = (
            "parcel_id,crop,region,campaign,yield_t_ha,ndvi_t0,ndvi_t1\n"
            "P001,maize,costa,2025,5.2,0.31,0.68\n"
            "P002,maize,costa,2025,5.5,0.35,0.71\n"
            "P003,maize,sierra,2024,4.3,0.28,0.59\n"
        ).encode("utf-8")
        imported = client.post(
            "/api/datasets/import", headers=headers,
            files={"file": ("parcelas.csv", csv_bytes, "text/csv")},
            data={"source_name": "Censo local", "license_name": "Uso institucional", "provenance_note": "Registros de campo declarados por el usuario"},
        )
        assert imported.status_code == 201, imported.text
        assert imported.json()["row_count"] == 3
        assert imported.json()["synthetic"] is False
        assert imported.json()["source"] == "Censo local"
        assert imported.json()["license"] == "Uso institucional"
        assert imported.json()["provenance"] == "Registros de campo declarados por el usuario"
        assert len([x for x in client.get("/api/parcels").json() if x["dataset_id"] == imported.json()["id"]]) == 3
        scientific = client.get("/api/reports/scientific.docx")
        with ZipFile(BytesIO(scientific.content)) as document:
            body = document.read("word/document.xml").decode("utf-8")
        assert "Censo local" in body and "Uso institucional" in body
        assert "Registros de campo declarados por el usuario" in body

        invalid = client.post(
            "/api/datasets/import", headers=headers,
            files={"file": ("invalid.csv", b"parcel_id,yield_t_ha\nP1,5\n", "text/csv")},
        )
        assert invalid.status_code == 422


def test_research_mode_excludes_synthetic_history(monkeypatch):
    import app.main as main_module
    from app.models import Dataset
    with TestClient(app) as client:
        headers = login(client)
        demo = client.post("/api/datasets/demo", headers=headers).json()
        run = client.post("/api/experiments", headers=headers, json={
            "dataset_id": demo["id"], "model_type": "RandomForest", "seed": 42})
        assert run.status_code == 202, run.text
        model = next(m for m in client.get("/api/models").json() if m["experiment_id"] == run.json()["id"])
        predicted = client.post("/api/predictions", headers=headers, json={"model_id": model["id"], "record_id": 0})
        assert predicted.status_code == 201, predicted.text
        imported = client.post("/api/datasets/import", headers=headers,
            files={"file": ("research-scope-fixture.csv", (
                "parcel_id,crop,region,campaign,yield_t_ha,ndvi_t0\n"
                "R1,wheat,A,2020,5,0.3\nR2,wheat,A,2021,6,0.4\nR3,wheat,B,2022,7,0.5\n"
            ).encode(), "text/csv")})
        assert imported.status_code == 201, imported.text
        monkeypatch.setattr(main_module, "DEMO_DATA_ENABLED", False)
        rows = client.get("/api/datasets").json()
        assert any(d["id"] == imported.json()["id"] for d in rows)
        assert rows and all(d["synthetic"] is False for d in rows)
        assert not any(e["dataset_id"] == demo["id"] for e in client.get("/api/experiments").json())
        assert all(m["synthetic"] is False for m in client.get("/api/models").json())
        assert all(p["synthetic"] is False for p in client.get("/api/predictions").json())
        assert client.get(f"/api/parcels?dataset_id={demo['id']}").json() == []
        dashboard = client.get("/api/dashboard").json()
        assert dashboard["datasets"] == len(rows)
        assert dashboard["models"] == len(client.get("/api/models").json())
        assert dashboard["synthetic_data_present"] is False
        assert client.post("/api/datasets/demo", headers=headers).status_code == 404
        assert client.post("/api/experiments", headers=headers, json={
            "dataset_id": demo["id"], "model_type": "RandomForest"}).status_code == 404
        assert client.get(f"/api/experiments/{run.json()['id']}").status_code == 404
        assert client.get(f"/api/models/{model['id']}/prediction-inputs").status_code == 404
        assert client.post("/api/predictions", headers=headers, json={
            "model_id": model["id"], "record_id": 0}).status_code == 404
        scientific = client.get("/api/reports/scientific.docx")
        with ZipFile(BytesIO(scientific.content)) as document:
            body = document.read("word/document.xml").decode("utf-8")
        assert "research-scope-fixture.csv" in body and "Demostración sintética" not in body
        with SessionLocal() as db:
            assert db.get(Dataset, demo["id"]) is not None  # History was not deleted.


def test_publication_authentication_declared_files_and_integrity(tmp_path, monkeypatch):
    import hashlib
    import json
    import app.publication as publication
    root = tmp_path / "publication"
    root.mkdir()
    image = root / "figure.png"
    content = b"\x89PNG\r\n\x1a\nresearch-fixture"
    image.write_bytes(content)
    catalogue = {"figures": [{"id": i} for i in range(1, 7)],
                 "tables": [{"id": i} for i in range(1, 7)],
                 "files": {"figure.png": {"path": "figure.png", "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest()}}}
    manifest = root / "catalogo_publicacion.json"
    manifest.write_text(json.dumps(catalogue))
    monkeypatch.setattr(publication, "PUBLICATION_DIR", root)
    with TestClient(app) as client:
        assert client.get("/api/publication").status_code == 401
        login(client, "viewer@example.test")
        response = client.get("/api/publication")
        assert response.status_code == 200
        assert "path" not in response.json()["files"]["figure.png"]
        response = client.get("/api/publication/files/figure.png?download=true")
        assert response.status_code == 200 and response.content == content
        assert "attachment" in response.headers["content-disposition"]
        assert client.get("/api/publication/files/.env").status_code == 404
        image.write_bytes(content + b"changed")
        assert client.get("/api/publication/files/figure.png").status_code == 409
        catalogue["files"]["figure.png"]["path"] = "../private.png"
        (tmp_path / "private.png").write_bytes(content)
        manifest.write_text(json.dumps(catalogue))
        assert client.get("/api/publication/files/figure.png").status_code == 404
        manifest.unlink()
        assert client.get("/api/publication").status_code == 503
