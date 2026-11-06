"""Authenticated Docker smoke test. Run with email as argv and password on stdin."""
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import pandas as pd
from app.ml import create_demo_csv

base = "http://localhost:8000"
email = sys.argv[1] if len(sys.argv) > 1 else "integration@example.test"
password = sys.stdin.readline().rstrip("\r\n")
assert password, "Password required on stdin"

with httpx.Client(base_url=base, timeout=120) as client:
    assert client.get("/health").json()["status"] == "ok"
    login = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    headers = {"X-CSRF-Token": client.cookies["csrf_token"]}
    csv_fixture = (
        "parcel_id,crop,region,campaign,yield_t_ha,ndvi_t0,ndvi_t1,synthetic_demo\n"
        "D1,maize,costa,2024,4.1,0.3,0.6,true\n"
        "D2,maize,costa,2024,4.6,0.4,0.7,true\n"
        "D3,wheat,sierra,2025,3.2,0.2,0.5,true\n"
    ).encode("utf-8")
    imported = client.post(
        "/api/datasets/import", headers=headers,
        files={"file": ("integration-fixture.csv", csv_fixture, "text/csv")},
        data={"source_name": "integration_fixture", "license_name": "synthetic test data",
              "provenance_note": "Docker smoke fixture"},
    )
    assert imported.status_code == 201, imported.text
    assert imported.json()["synthetic"] is True
    assert imported.json()["provenance"] == "Docker smoke fixture"
    dataset = client.post("/api/datasets/demo", headers=headers)
    assert dataset.status_code == 201, dataset.text
    dataset_id = dataset.json()["id"]
    run = client.post("/api/experiments", headers=headers, json={"dataset_id": dataset_id, "model_type": "XGBoost", "seed": 42})
    assert run.status_code == 202, run.text
    run_id = run.json()["id"]
    for _ in range(90):
        status = client.get(f"/api/experiments/{run_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(1)
    assert status["status"] == "completed", status.get("error")
    model = next(x for x in client.get("/api/models").json() if x["experiment_id"] == run_id)
    importance = client.get(f"/api/models/{model['id']}/importance")
    assert importance.status_code == 200, importance.text
    assert importance.json()["values"] and importance.json()["causal"] is False
    with TemporaryDirectory() as temp:
        sample = pd.read_csv(create_demo_csv(Path(temp) / "demo.csv")).iloc[0]
        features = {name: (str(sample[name]) if name in {"crop", "region", "campaign"} else float(sample[name])) for name in model["feature_names"]}
    predicted = client.post("/api/predictions", headers=headers, json={"model_id": model["id"], "features": features})
    assert predicted.status_code == 201, predicted.text
    assert predicted.json()["synthetic"] is True
    assert predicted.json()["warnings"]
    for extension, signature in (("pdf", b"%PDF"), ("docx", b"PK"), ("xlsx", b"PK")):
        report = client.get(f"/api/reports/scientific.{extension}")
        assert report.status_code == 200 and report.content.startswith(signature)
    assert client.get("/api/audit").status_code == 200
    with httpx.Client(base_url="http://localhost:5173", timeout=20) as frontend:
        assert frontend.get("/").status_code == 200
        assert frontend.get("/api/dashboard").status_code == 401
    token = client.cookies["session"]
    assert client.post("/api/auth/logout", headers=headers).status_code == 200
    client.cookies.set("session", token)
    assert client.get("/api/auth/me").status_code == 401
    print(f"Docker smoke passed: CSV import, demo dataset={dataset_id}, experiment={run_id}, model={model['id']}, importance, XGBoost prediction and 3 reports")
