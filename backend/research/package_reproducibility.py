"""Build an access-controlled, complete study bundle from existing Docker artifacts.

No retraining, database dump, credentials or original NetCDF download. This script
does not upload its output. Redistribution requires the provider's permission.
Run from the repository root with the original study installation running.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/yieldsat_wheat_240d"
EXPECTED_SHA = "fa9496d9598896042152d3bd669edd1290384630f0e880d79fdde483aee0e2e1"


def docker(*args):
    return subprocess.check_output(["docker", "compose", *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/reproducibility/yieldsat-wheat-v1.0.0")
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(ROOT / "data"):
        raise ValueError("Store restricted bundles inside the ignored data directory")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing bundle: {target}")
    csv = SOURCE / "yieldsat_Germany_wheat_240d.csv"
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == EXPECTED_SHA
    evidence = ROOT / "docs/manuscrito/evidencia_modelos.json"
    captured = json.loads(evidence.read_text(encoding="utf-8"))
    assert captured["dataset_sha256"] == EXPECTED_SHA
    query = """
import json
from app.db import SessionLocal
from app.models import Dataset, ModelVersion
from app.ml.pipeline import _sha256
with SessionLocal() as db:
    d=db.get(Dataset,9)
    assert _sha256(d.file_path)=='fa9496d9598896042152d3bd669edd1290384630f0e880d79fdde483aee0e2e1'
    print(json.dumps([{'id':i,'model_type':db.get(ModelVersion,i).model_type,
                      'artifact_path':db.get(ModelVersion,i).artifact_path}
                     for i in range(14,19)]))
"""
    models = json.loads(docker("exec", "-T", "backend", "python", "-c", query))
    environment = json.loads(docker("exec", "-T", "backend", "python", "-m", "pip", "list", "--format=json"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    target.mkdir(parents=True)
    (target / "models").mkdir()
    shutil.copy2(csv, target / csv.name)
    for name in ("yieldsat_Germany_wheat_240d.provenance.json", "experiment_plan.json"):
        shutil.copy2(SOURCE / name, target / name)
    shutil.copy2(evidence, target / evidence.name)
    (target / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    locks = [f"{p['name']}=={p['version']}" for p in environment if p["name"].lower() not in {"pip", "setuptools", "wheel"}]
    (target / "requirements-lock.txt").write_text("--extra-index-url https://download.pytorch.org/whl/cpu\n" + "\n".join(locks) + "\n", encoding="utf-8")
    import csv as csv_module
    with csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv_module.DictReader(stream))
    with (target / "splits.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv_module.writer(stream)
        writer.writerow(["row", "parcel_id", "campaign", "role"])
        for record in captured["records"]:
            index = record["row"]
            writer.writerow([index, rows[index]["parcel_id"], rows[index]["campaign"], record["role"]])
    for entry in models:
        artifact = Path(entry.pop("artifact_path"))
        filename = f"model_{entry['id']}.joblib"
        docker("cp", f"backend:{artifact.as_posix()}", str(target / "models" / filename))
        docker("cp", f"backend:{artifact.with_suffix('.json').as_posix()}", str(target / "models" / f"model_{entry['id']}.json"))
        entry["file"] = f"models/{filename}"
        assert hashlib.sha256((target / entry["file"]).read_bytes()).hexdigest() == captured["model_details"][entry["model_type"]]["artifact_sha256"]
    (target / "ACCESS.txt").write_text(
        "RESTRICTED LOCAL RESEARCH BUNDLE. NOT AN OPEN DATA DEPOSIT.\n"
        "Contains YieldSAT-derived records and fitted weights. Do not redistribute\n"
        "without reviewing the original access terms and obtaining permission when required.\n"
        "No database users, passwords, cookies or original download links are included.\n"
        "Study code: https://github.com/Pull-H/prediccion-rendimiento-agricola\n"
        "Verification: python backend/research/replay_study.py --bundle <this-directory>\n",
        encoding="utf-8")
    inventory = {p.relative_to(target).as_posix(): {"bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(target.rglob("*")) if p.is_file()}
    manifest = {"study": "yieldsat-wheat-240d", "version": "1.0.0", "source_commit": commit,
                "data_access": "restricted_pending_redistribution_terms", "dataset_sha256": EXPECTED_SHA,
                "models": models, "files": inventory}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = target.parent / (target.name + ".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(target))
    print(json.dumps({"bundle": str(archive), "files": len(inventory) + 1,
                      "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
