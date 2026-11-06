"""Local YieldSAT CLI: python -m app.ml discover|inspect|convert.

Examples (from backend/, after installing yieldsat-requirements.txt):

  python -m app.ml discover --source D:/YieldSAT --variant merged
  python -m app.ml inspect --source D:/YieldSAT/merged/data.nc
  python -m app.ml convert --source D:/YieldSAT/merged/data.nc \
      --output D:/exports/yieldsat.csv --max-samples 1000

The local CLI bypasses the web upload limit, but conversion always requires an
explicit row count and enforces an output byte cap.
"""

import argparse
import json

from .yieldsat import (DEFAULT_MAX_OUTPUT_BYTES, convert_yieldsat_to_csv,
                       audit_yieldsat_quality, discover_yieldsat_files, inspect_yieldsat_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or convert local YieldSAT ML-ready NetCDF files")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("discover", "inspect", "audit", "convert"):
        command = commands.add_parser(name)
        command.add_argument("--source", required=True, help="Local NetCDF file or dataset directory")
        command.add_argument("--variant", choices=("merged", "all"), default="merged")
        if name == "audit":
            command.add_argument("--probe-rows", type=int, default=256)
        if name == "convert":
            command.add_argument("--output", required=True, help="Destination CSV path")
            command.add_argument("--max-samples", type=int, required=True,
                                 help="Explicit count of consecutive source samples to export")
            command.add_argument("--start", type=int, default=0)
            command.add_argument("--batch-size", type=int, default=256)
            command.add_argument("--target-unit", choices=("t/ha", "kg/ha"))
            command.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES)
    fields = commands.add_parser("prepare-fields", help="Export a field-level satellite+soil pilot")
    fields.add_argument("--source", required=True)
    fields.add_argument("--output", required=True)
    fields.add_argument("--crop", default="wheat")
    fields.add_argument("--target-unit", required=True, choices=("t/ha", "kg/ha"))
    fields.add_argument("--target-unit-evidence", required=True)
    fields.add_argument("--horizon-days", type=int, default=240)
    fields.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.command == "discover":
        result = {"files": discover_yieldsat_files(args.source, args.variant)}
    elif args.command == "inspect":
        result = inspect_yieldsat_file(args.source, args.variant)
    elif args.command == "audit":
        result = audit_yieldsat_quality(args.source, probe_rows=args.probe_rows)
    elif args.command == "prepare-fields":
        from .yieldsat_experiment import build_field_experiment
        result = build_field_experiment(args.source, args.output, crop=args.crop,
                                        target_unit=args.target_unit,
                                        target_unit_evidence=args.target_unit_evidence,
                                        horizon_days=args.horizon_days, batch_size=args.batch_size)
    else:
        result = convert_yieldsat_to_csv(
            args.source, args.output, max_samples=args.max_samples, start=args.start,
            batch_size=args.batch_size, target_unit=args.target_unit, variant=args.variant,
            max_output_bytes=args.max_output_bytes,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
