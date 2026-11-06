"""Import the authorized small YieldSAT export and evaluate through the live API.

Run from the repository root with Docker Compose running:
    python backend/tests/run_yieldsat_e2e.py data/yieldsat_wheat_240d/yieldsat_Germany_wheat_240d.csv
Uses a temporary administrator, deactivated even when the workflow fails.
No source NetCDF or SharePoint download is performed.
"""

import argparse
import hashlib
import json
import secrets
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--reuse-dataset-id", type=int)
    args = parser.parse_args()
    source = args.source
    manifest = json.loads(source.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == manifest["output_sha256"], "CSV hash mismatch"
    assert manifest["crop"] == "wheat" and manifest["horizon_days_after_sowing"] == 240
    frame = pd.read_csv(source)
    assert len(frame) == manifest["output_rows"]
    assert not frame.isna().any().any() and np.isfinite(frame.select_dtypes("number")).all().all()
    assert not frame.duplicated(["parcel_id", "campaign"]).any()
    assert set(frame.parcel_id) == {item["parcel_id"] for item in manifest["field_provenance"]}
    for item in manifest["field_provenance"]:
        sow, cutoff, harvest = map(pd.Timestamp, (item["sowing_date"], item["cutoff"], item["harvesting_date"]))
        assert (cutoff - sow).days == 240 and cutoff < harvest
        assert item["sowing_date_origins"] == ["provided_by_farmer"]

    email = f"yieldsat-{secrets.token_hex(8)}@example.test"
    password = secrets.token_urlsafe(24)
    compose = ["docker", "compose", "exec", "-T", "backend", "python", "-c"]
    create = """
import sys
from app.db import SessionLocal
from app.models import User
from app.security import hasher
email, password = [sys.stdin.readline().strip() for _ in range(2)]
with SessionLocal() as db:
    db.add(User(email=email, name='YieldSAT integration', role='admin', password_hash=hasher.hash(password)))
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
    output = source.parent / "live_temporal_results.json"
    try:
        with httpx.Client(base_url="http://localhost:8000", timeout=120) as client:
            login = client.post("/api/auth/login", json={"email": email, "password": password})
            assert login.status_code == 200, login.text
            headers = {"X-CSRF-Token": client.cookies["csrf_token"]}
            if args.reuse_dataset_id:
                datasets = client.get("/api/datasets")
                datasets.raise_for_status()
                dataset = next(x for x in datasets.json() if x["id"] == args.reuse_dataset_id)
            else:
                with source.open("rb") as stream:
                    response = client.post("/api/datasets/import", headers=headers,
                        files={"file": (source.name, stream, "text/csv")}, data={
                            "source_name": "YieldSAT wheat Germany 240d satellite+soil",
                            "license_name": "Acceso autorizado; licencia no aportada",
                            "provenance_note": (
                                f"CSV SHA256 {digest}; NetCDF SHA256 {manifest['source_sha256']}; "
                                "184 field records; farmer-reported sowing only; cutoff sowing+240d; "
                                "4 records excluded. Satellite+soil pilot; weather and spatial patches pending. "
                                "Target t/ha according to official tutorial; source has no unit attribute. "
                                "See yieldsat_Germany_wheat_240d.provenance.json. No Peru validation."),
                        })
                assert response.status_code == 201, response.text
                dataset = response.json()
            assert dataset["row_count"] == len(frame) and dataset["synthetic"] is False
            assert dataset["quality"]["spatial_unit"] == "parcel"
            subprocess.run(["docker", "compose", "cp", str(source.with_suffix(".provenance.json")),
                            f"backend:/app/data/dataset_{dataset['id']}.provenance.json"],
                           check=True, timeout=60)
            results = {"dataset_id": dataset["id"], "source_sha256": digest, "seed": 42,
                       "validation": "temporal", "deep_epochs": args.epochs,
                       "executed_at_utc": datetime.now(timezone.utc).isoformat(), "models": {}}

            def save():
                output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

            save()
            # Use a held-out observation to verify serving, not to select parameters.
            first_row = frame[frame.campaign == 2021].iloc[0]
            for kind in ("RandomForest", "XGBoost", "LSTM", "CNN-LSTM", "CNN-Transformer"):
                response = client.post("/api/experiments", headers=headers, json={
                    "dataset_id": dataset["id"], "model_type": kind, "seed": 42,
                    "validation": "temporal", "epochs": args.epochs})
                assert response.status_code == 202, response.text
                experiment_id = response.json()["id"]
                print(f"Started {kind}, experiment {experiment_id}", flush=True)
                for _ in range(900):
                    state_response = client.get(f"/api/experiments/{experiment_id}")
                    state_response.raise_for_status()
                    state = state_response.json()
                    if state["status"] in {"completed", "failed"}:
                        break
                    time.sleep(2)
                assert state["status"] == "completed", f"{kind}: {state.get('error')}"
                assert [state["split_info"][key] for key in ("train_rows", "validation_rows", "test_rows")] == [132, 15, 37]
                response = client.get("/api/models")
                response.raise_for_status()
                model = next(x for x in response.json() if x["experiment_id"] == experiment_id)
                features = {name: str(first_row[name]) if name in {"crop", "region", "campaign"}
                            else float(first_row[name]) for name in model["feature_names"]}
                prediction = client.post("/api/predictions", headers=headers,
                                         json={"model_id": model["id"], "features": features})
                assert prediction.status_code == 201, prediction.text
                assert prediction.json()["synthetic"] is False
                assert any("240" in warning for warning in prediction.json()["warnings"])
                results["models"][kind] = {"experiment_id": experiment_id, "model_id": model["id"],
                    "metrics": state["metrics"], "split_info": state["split_info"],
                    "prediction_id": prediction.json()["id"]}
                save()
                print(f"{kind}: RMSE={state['metrics']['test']['rmse']:.4f}, R2={state['metrics']['test']['r2']:.4f}", flush=True)

            statistics = client.get("/api/statistics/compare", params={
                "dataset_id": dataset["id"], "validation": "temporal", "seed": 42})
            assert statistics.status_code == 200, statistics.text
            results["statistics"] = statistics.json()
            assert results["statistics"]["independence_verified"] is False
            assert results["statistics"]["test_region_counts"] == {"Germany/farm2": 36, "Germany/farm6": 1}
            # Training-mean baseline uses no held-out labels to fit its estimate.
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            train_mean = float(frame.loc[frame.campaign <= 2019, "yield_t_ha"].mean())
            target = frame.loc[frame.campaign >= 2021, "yield_t_ha"].to_numpy()
            baseline = np.full(len(target), train_mean)
            results["training_mean_baseline"] = {"value_t_ha": train_mean,
                "mae": float(mean_absolute_error(target, baseline)),
                "rmse": float(np.sqrt(mean_squared_error(target, baseline))),
                "r2": float(r2_score(target, baseline))}
            report = client.get("/api/reports/scientific.docx")
            assert report.status_code == 200 and report.content.startswith(b"PK"), report.text[:200]
            (source.parent / "scientific_report.docx").write_bytes(report.content)
            save()
            assert client.post("/api/auth/logout", headers=headers).status_code == 200
            print(f"Passed: dataset {dataset['id']}, five models/predictions, statistical tests and report. {output}", flush=True)
    finally:
        subprocess.run(compose + [disable], input=email + "\n", text=True, check=True, timeout=60)


if __name__ == "__main__":
    main()
