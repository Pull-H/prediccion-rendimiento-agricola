"""Check automatic historical inputs against manual inference for current models.

Run from the repository root with Docker Compose running. Does not train models.
"""

import json
import math
import secrets
import subprocess
from pathlib import Path

import httpx


email = f"inputs-{secrets.token_hex(8)}@example.test"
password = secrets.token_urlsafe(24)
compose = ["docker", "compose", "exec", "-T", "backend", "python", "-c"]
create = """
import sys
from app.db import SessionLocal
from app.models import User
from app.security import hasher
email, password = [sys.stdin.readline().strip() for _ in range(2)]
with SessionLocal() as db:
    db.add(User(email=email, name='Automatic inputs integration', role='admin', password_hash=hasher.hash(password)))
    db.commit()
"""
disable = """
import sys
from sqlalchemy import select
from app.db import SessionLocal
from app.models import User
email = sys.stdin.readline().strip()
with SessionLocal() as db:
    user = db.scalar(select(User).where(User.email == email))
    if user:
        user.active = False
        db.commit()
"""
subprocess.run(compose + [create], input=f"{email}\n{password}\n", text=True, check=True, timeout=60)
try:
    results = []
    with httpx.Client(base_url="http://localhost:8000", timeout=60) as client:
        login = client.post("/api/auth/login", json={"email": email, "password": password})
        assert login.status_code == 200, login.text
        headers = {"X-CSRF-Token": client.cookies["csrf_token"]}
        response = client.get("/api/models")
        response.raise_for_status()
        latest = {}
        for model in sorted(response.json(), key=lambda item: item["id"], reverse=True):
            latest.setdefault(model["model_type"], model)
        assert len(latest) == 5
        for model in latest.values():
            path = f"/api/models/{model['id']}/prediction-inputs"
            response = client.get(path)
            assert response.status_code == 200, response.text
            inputs = response.json()
            assert "yield_t_ha" not in response.text and "observed_yield" not in response.text
            record = next(row for row in inputs["records"] if row["evaluation_role"] == "test")
            response = client.get(f"{path}/{record['record_id']}")
            assert response.status_code == 200, response.text
            detail = response.json()
            assert set(detail["features"]) == set(model["feature_names"])
            automatic = client.post("/api/predictions", headers=headers,
                json={"model_id": model["id"], "record_id": record["record_id"]})
            assert automatic.status_code == 201, automatic.text
            manual = client.post("/api/predictions", headers=headers,
                json={"model_id": model["id"], "features": detail["features"]})
            assert manual.status_code == 201, manual.text
            assert math.isclose(automatic.json()["yield_t_ha"], manual.json()["yield_t_ha"], abs_tol=1e-6)
            assert automatic.json()["parcel_id"] is not None
            assert automatic.json()["synthetic"] is inputs["synthetic"]
            assert any("hist" in warning for warning in automatic.json()["warnings"])
            results.append({"model_id": model["id"], "model_type": model["model_type"],
                "dataset_id": inputs["dataset_id"], "available_records": len(inputs["records"]),
                "loaded_variables": len(detail["features"]), "record_id": record["record_id"],
                "prediction_id": automatic.json()["id"], "manual_equivalent": True})
            print(f"{model['model_type']} #{model['id']}: {len(detail['features'])} variables loaded, prediction matched", flush=True)
        assert client.post("/api/auth/logout", headers=headers).status_code == 200
    Path("data/automatic_inputs_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
finally:
    subprocess.run(compose + [disable], input=email + "\n", text=True, check=True, timeout=60)
