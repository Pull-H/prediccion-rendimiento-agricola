"""Repeat the five-model protocol on an authorized CSV without a database.

Writes new artifacts only. Changing the dataset, seed or epoch count produces
a new experiment and must not be reported as the original frozen study.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.pipeline import train
from app.ml.statistics import compare_models

EXPECTED_SHA = "fa9496d9598896042152d3bd669edd1290384630f0e880d79fdde483aee0e2e1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--new-study", action="store_true", help="Allow a CSV different from the original study")
    args = parser.parse_args()
    digest = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA and not args.new_study:
        raise ValueError("CSV differs from the published study. Use --new-study explicitly for a new dataset.")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; frozen artifacts must not be overwritten")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    args.output.mkdir(parents=True)
    results = {}
    for name in ("RandomForest", "XGBoost", "LSTM", "CNN-LSTM", "CNN-Transformer"):
        results[name] = train(args.csv, name, args.output / "models", seed=args.seed, validation="temporal", epochs=args.epochs)
        print(f"Training completed: {name}", flush=True)
    statistics = compare_models(args.csv, {name: result["artifact_path"] for name, result in results.items()}, "temporal", seed=args.seed)
    payload = {"dataset_sha256": digest, "seed": args.seed, "epochs": args.epochs,
               "same_protocol": digest == EXPECTED_SHA and args.seed == 42 and args.epochs == 30,
               "note": "New training; numerical agreement across hardware/software is not guaranteed.",
               "models": results, "statistics": statistics}
    (args.output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results saved: {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
