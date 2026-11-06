"""Read live statistics for the three real research cohorts; no model training."""

import json
import secrets
import subprocess
from pathlib import Path

import httpx


email = f"statistics-{secrets.token_hex(8)}@example.test"
password = secrets.token_urlsafe(24)
compose = ["docker", "compose", "exec", "-T", "backend", "python", "-c"]
create = """
import sys
from app.db import SessionLocal
from app.models import User
from app.security import hasher
email, password = [sys.stdin.readline().strip() for _ in range(2)]
with SessionLocal() as db:
    db.add(User(email=email, name='Statistics verification', role='viewer', password_hash=hasher.hash(password)))
    db.commit()
"""
disable = """
import sys
from sqlalchemy import select
from app.db import SessionLocal
from app.models import User
with SessionLocal() as db:
    user = db.scalar(select(User).where(User.email == sys.stdin.readline().strip()))
    if user:
        user.active = False
        db.commit()
"""
subprocess.run(compose + [create], input=f"{email}\n{password}\n", text=True, check=True, timeout=60)
try:
    results = []
    with httpx.Client(base_url="http://localhost:8000", timeout=120) as client:
        assert client.get("/api/statistics/compare", params={
            "dataset_id": 9, "validation": "temporal", "seed": 42}).status_code == 401
        login = client.post("/api/auth/login", json={"email": email, "password": password})
        assert login.status_code == 200, login.text
        for dataset_id, validation in ((9, "temporal"), (7, "temporal"), (7, "parcel")):
            response = client.get("/api/statistics/compare", params={
                "dataset_id": dataset_id, "validation": validation, "seed": 42})
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["dataset_id"] == dataset_id
            assert len(result["models"]) == len(result["model_versions"]) == 5
            assert len(result["wilcoxon_holm"]) == 10
            assert result["bootstrap_repetitions"] == 2000
            assert result["friedman"]["degrees_of_freedom"] == 4
            assert result["independence_verified"] is False
            assert sum(result["test_region_counts"].values()) == result["test_observations"]
            for comparison in result["wilcoxon_holm"]:
                assert 0 <= comparison["p_value"] <= comparison["holm_p_value"] <= 1
                assert comparison["statistic"] >= 0
            if dataset_id == 9:
                assert result["test_observations"] == result["block_count"] == 37
                assert result["test_region_counts"] == {"Germany/farm2": 36, "Germany/farm6": 1}
                assert all(run["epochs"] == 30 for name, run in result["training_details"].items()
                           if name in {"LSTM", "CNN-LSTM", "CNN-Transformer"})
            results.append(result)
            print(f"Dataset {dataset_id}, {validation}: five models, ten comparisons, traceability and intervals verified", flush=True)
        assert client.get("http://localhost:5173/estadisticas").status_code == 200
        assert client.post("/api/auth/logout", headers={"X-CSRF-Token": client.cookies["csrf_token"]}).status_code == 200
    Path("data/statistics_module_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
finally:
    subprocess.run(compose + [disable], input=email + "\n", text=True, check=True, timeout=60)
