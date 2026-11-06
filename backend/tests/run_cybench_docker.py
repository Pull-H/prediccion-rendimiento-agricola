"""Register and evaluate the official real CY-Bench sample through the live API.

Usage: python backend/tests/run_cybench_docker.py EMAIL CSV [--validation temporal|parcel] [--reuse-dataset-id ID]
The password is read from stdin. This script intentionally never creates users.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("email")
parser.add_argument("source", type=Path)
parser.add_argument("--validation", choices=("temporal", "parcel"), default="temporal")
parser.add_argument("--reuse-dataset-id", type=int)
args = parser.parse_args()
email = args.email
source = args.source
password = sys.stdin.readline().rstrip("\r\n")
if not password or not source.is_file():
    raise SystemExit("Provide an existing account password on stdin and a converted CSV path")

with httpx.Client(base_url="http://localhost:8000", timeout=120) as client:
    login = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    headers = {"X-CSRF-Token": client.cookies["csrf_token"]}
    provenance = json.loads(source.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    note = (f"Official WUR-AI sample commit {provenance['source_revision']}; "
            f"converted SHA-256 {provenance['output_sha256']}; regional units; "
            "mid-season features; see local provenance manifest")
    if args.reuse_dataset_id:
        response = client.get("/api/datasets")
        assert response.status_code == 200, response.text
        dataset = next((item for item in response.json() if item["id"] == args.reuse_dataset_id), None)
        assert dataset is not None, f"Dataset {args.reuse_dataset_id} not found"
    else:
        with source.open("rb") as stream:
            imported = client.post(
                "/api/datasets/import", headers=headers,
                files={"file": (source.name, stream, "text/csv")},
                data={"source_name": "CY-Bench maize ES (WUR-AI official sample)",
                      "license_name": "CY-Bench EUPL-1.2; JRC labels CC BY 4.0",
                      "provenance_note": note},
            )
        assert imported.status_code == 201, imported.text
        dataset = imported.json()
    assert dataset["row_count"] == provenance["output_rows"]
    assert dataset["synthetic"] is False
    assert dataset["quality"]["spatial_unit"] == "administrative_region"
    results = {"dataset_id": dataset["id"], "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "source_revision": provenance["source_revision"],
               "spatial_unit": "NUTS administrative region", "seed": 42,
               "validation": args.validation, "executed_at_utc": datetime.now(timezone.utc).isoformat(),
               "models": {}}
    first_row = pd.read_csv(source).iloc[0]
    for kind in ("RandomForest", "XGBoost", "LSTM", "CNN-LSTM", "CNN-Transformer"):
        launch = client.post(
            "/api/experiments", headers=headers,
            json={"dataset_id": dataset["id"], "model_type": kind, "seed": 42,
                  "validation": args.validation, "epochs": 5},
        )
        assert launch.status_code == 202, launch.text
        experiment_id = launch.json()["id"]
        for _ in range(180):
            state = client.get(f"/api/experiments/{experiment_id}").json()
            if state["status"] in {"completed", "failed"}:
                break
            time.sleep(2)
        assert state["status"] == "completed", f"{kind}: {state.get('error')}"
        model = next(x for x in client.get("/api/models").json() if x["experiment_id"] == experiment_id)
        features = {name: (str(first_row[name]) if name in {"crop", "region", "campaign"}
                           else float(first_row[name])) for name in model["feature_names"]}
        prediction = client.post("/api/predictions", headers=headers,
                                 json={"model_id": model["id"], "features": features})
        assert prediction.status_code == 201, f"{kind}: {prediction.text}"
        assert prediction.json()["synthetic"] is False
        results["models"][kind] = {"experiment_id": experiment_id, "model_id": model["id"],
                                   "metrics": state["metrics"], "split_info": state["split_info"],
                                   "prediction_id": prediction.json()["id"]}
        print(f"{kind}: model={model['id']}, test_rmse={state['metrics']['test']['rmse']:.4f}, "
              f"test_r2={state['metrics']['test']['r2']:.4f}", flush=True)
    report = client.get("/api/reports/scientific.docx")
    assert report.status_code == 200 and report.content.startswith(b"PK")
    assert client.post("/api/auth/logout", headers=headers).status_code == 200
    output = Path(f"data/cybench_{args.validation}_results.json")
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Live CY-Bench API workflow passed: dataset={dataset['id']}, 5 models, 5 predictions, scientific report")
