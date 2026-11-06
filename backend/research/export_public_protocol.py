"""Export aggregate protocol metadata; never publish observation-level evidence."""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    evidence = json.loads((ROOT / "docs/manuscrito/evidencia_modelos.json").read_text(encoding="utf-8"))
    bundle = ROOT / "data/reproducibility/yieldsat-wheat-v1.0.0"
    output = ROOT / "reproducibility"
    output.mkdir(exist_ok=True)
    groups = {name: {"evaluation_role": record["evaluation_role"], "n": record["n"],
                    "repetitions": record["repetitions"], "seed": record["seed"],
                    "groups": {key: {field: group[field] for field in ("columns", "mean_delta_mae", "sd_delta_mae")}
                               for key, group in record["groups"].items()}}
              for name, record in evidence["permutation"].items()}
    study = {"version": "1.0.0", "dataset": "YieldSAT Germany wheat 240-day field aggregates",
             "dataset_sha256": evidence["dataset_sha256"], "rows": 184,
             "data_access": "provider_authorization_required; local CSV and weights not publicly deposited",
             "provider_url": "https://yieldsat.github.io/", "split": evidence["split"],
             "seed": 42, "python": evidence["python"], "versions": evidence["versions"],
             "model_details": evidence["model_details"], "metrics": evidence["metrics"],
             "baseline_value": evidence["baseline_value"], "statistics": evidence["statistics"],
             "permutation_summary": groups, "analysis_note": evidence["analysis_note"]}
    (output / "study.json").write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "campaign", "role"])
        for row in evidence["records"]:
            writer.writerow([row["row"], row["campaign"], row["role"]])
    shutil.copy2(bundle / "requirements-lock.txt", output / "requirements-lock.txt")
    print("Exported aggregate study.json, row-index splits and observed environment lock")


if __name__ == "__main__":
    main()
