"""Auditable adapter for patelris/crop-yield-prediction-dataset.

Annual country observations are administrative regions, never parcels.
Repeated country/crop/year records in the merged CSV have varying temperatures;
collapse to one record using mean temperature only after checking other values.
This is a retrospective annual benchmark, not an in-season forecast dataset.
"""
import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

from .pipeline import _sha256, inspect_csv


def prepare(source, output):
    with zipfile.ZipFile(source) as archive:
        data = pd.read_csv(archive.open("yield_df.csv"))
    keys = ["Area", "Item", "Year"]
    values = ["hg/ha_yield", "average_rain_fall_mm_per_year", "pesticides_tonnes"]
    groups = data.groupby(keys, sort=True)
    if (groups[values].nunique(dropna=False) > 1).any().any():
        raise ValueError("Conflicting target/rainfall/pesticides within country/crop/year")
    clean = groups[values + ["avg_temp"]].mean().reset_index()
    frame = pd.DataFrame({
        "parcel_id": "country:" + clean.Area, "crop": clean.Item,
        "region": clean.Area, "country": clean.Area, "campaign": clean.Year,
        "spatial_unit": "administrative_region", "yield_t_ha": clean["hg/ha_yield"] / 10000,
        "rain_mm_year": clean.average_rain_fall_mm_per_year,
        "pesticides_tonnes_country": clean.pesticides_tonnes, "temperature_c": clean.avg_temp,
    })
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    provenance = {
        "source": "https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset",
        "source_sha256": _sha256(source), "output_sha256": _sha256(target),
        "original_rows": len(data), "output_rows": len(frame),
        "collapsed_rows": len(data) - len(frame),
        "aggregation": "One country/crop/year; mean avg_temp; other values verified constant",
        "target_conversion": "hg/ha divided by 10000 gives t/ha",
        "limitations": ["Country aggregates, not parcels", "Annual predictors: retrospective evaluation",
                        "Temperature merge provenance requires further review", "License requires source review"],
        "quality": inspect_csv(target),
    }
    target.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2, ensure_ascii=False))
