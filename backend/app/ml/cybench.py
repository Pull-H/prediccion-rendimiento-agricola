"""Reproducible adapter for WUR-AI's public CY-Bench maize Spain sample.

One row represents an administrative region and harvest year. Inputs stop at
the middle of the documented growing season. Harvest area and production are
never used as predictors because they can reveal the target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
import pandas as pd


REVISION = "b55e6918de1e1fa1b998966f6014b578cd36b83b"
BASE_URL = f"https://raw.githubusercontent.com/WUR-AI/sample_data/{REVISION}/maize/ES"
SOURCE_FILES = {
    "yield_maize_ES.csv": 46631,
    "location_maize_ES.csv": 1568,
    "soil_maize_ES.csv": 2729,
    "crop_calendar_maize_ES.csv": 2511,
    "ndvi_maize_ES.csv": 2108102,
    "fpar_maize_ES.csv": 1641604,
    "meteo_maize_ES.csv": 33213420,
    "soil_moisture_maize_ES.csv": 22709903,
}
SERIES = {
    "meteo": ("tmin", "tmax", "prec", "rad", "tavg", "et0", "vpd", "cwb"),
    "soil_moisture": ("ssm", "rsm"),
    "ndvi": ("ndvi",),
    "fpar": ("fpar",),
}
STEPS = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_sample(directory: str | Path) -> dict:
    """Download the pinned, public sample with per-file byte bounds."""
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, expected_size in SOURCE_FILES.items():
        target = root / name
        if not target.is_file() or target.stat().st_size != expected_size:
            temporary = root / f".{name}.{uuid4().hex}.tmp"
            try:
                request = Request(f"{BASE_URL}/{name}", headers={"User-Agent": "agri-yield-research/1.0"})
                with urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
                    read = 0
                    while block := response.read(1024 * 1024):
                        read += len(block)
                        if read > expected_size:
                            raise ValueError(f"Unexpected size for {name}")
                        stream.write(block)
                if read != expected_size:
                    raise ValueError(f"Incomplete download for {name}: {read}/{expected_size} bytes")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        files[name] = {"bytes": target.stat().st_size, "sha256": _sha256(target), "url": f"{BASE_URL}/{name}"}
    manifest = {
        "source": "WUR-AI CY-Bench official sample, maize Spain", "revision": REVISION,
        "source_repository": "https://github.com/WUR-AI/sample_data",
        "dataset_paper": "https://doi.org/10.5194/essd-18-3997-2026",
        "license": "CY-Bench EUPL-1.2; underlying JRC crop statistics CC BY 4.0; attribution required",
        "files": files,
    }
    (root / "download.provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _read(root: Path, prefix: str, required: tuple[str, ...]) -> pd.DataFrame:
    path = root / f"{prefix}_maize_ES.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CY-Bench sample file: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    if "crop_name" in frame and not frame["crop_name"].eq("maize").all():
        raise ValueError(f"{path.name} contains crops other than maize")
    return frame


def _series_features(root: Path, labels: pd.DataFrame, prefix: str,
                     variables: tuple[str, ...]) -> pd.DataFrame:
    frame = _read(root, prefix, ("adm_id", "date", *variables))
    dates = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d", errors="raise")
    frame = frame.assign(harvest_year=dates.dt.year, day_of_year=dates.dt.dayofyear)
    windows = labels[["adm_id", "harvest_year", "sos", "forecast_doy"]]
    frame = frame.merge(windows, on=["adm_id", "harvest_year"], how="inner", validate="many_to_one")
    frame = frame[(frame.day_of_year >= frame.sos) & (frame.day_of_year < frame.forecast_doy)].copy()
    if frame.empty:
        raise ValueError(f"No in-season observations before forecast cutoff for {prefix}")
    span = frame.forecast_doy - frame.sos
    frame["time_bin"] = np.minimum(STEPS - 1, np.floor(STEPS * (frame.day_of_year - frame.sos) / span).astype(int))
    for variable in variables:
        frame[variable] = pd.to_numeric(frame[variable], errors="coerce")
    if prefix == "ndvi":
        frame.loc[~frame.ndvi.between(50, 250), "ndvi"] = np.nan
        frame["ndvi"] = (frame["ndvi"] - 50) / 200
    if prefix == "fpar":
        frame.loc[~frame.fpar.between(0, 100), "fpar"] = np.nan
        frame["fpar"] = frame["fpar"] / 100
    sums = {"prec", "rad", "et0"}
    aggregated = frame.groupby(["adm_id", "harvest_year", "time_bin"], observed=True).agg(
        {name: ("sum" if name in sums else "mean") for name in variables}
    )
    # Keep missing groups missing; pandas' default sum would silently make them zero.
    if sums.intersection(variables):
        for name in sums.intersection(variables):
            aggregated[name] = frame.groupby(["adm_id", "harvest_year", "time_bin"], observed=True)[name].sum(min_count=1)
    wide = aggregated.unstack("time_bin")
    wide = wide.reindex(columns=pd.MultiIndex.from_product([variables, range(STEPS)]))
    wide.columns = [f"{name}_t{step}" for name, step in wide.columns]
    return wide.reset_index()


def convert_sample(directory: str | Path, output_csv: str | Path) -> dict:
    """Join observed regional yields to weather, soil and vegetation signals."""
    root = Path(directory).expanduser()
    target = Path(output_csv).expanduser()
    if target.suffix.lower() != ".csv":
        raise ValueError("Output must be a CSV")
    labels = _read(root, "yield", ("adm_id", "harvest_year", "yield", "harvest_area", "production"))
    if labels.duplicated(["adm_id", "harvest_year"]).any():
        raise ValueError("Duplicate region-year yield labels")
    labels = labels.rename(columns={"yield": "yield_t_ha"})
    numeric = ("harvest_year", "yield_t_ha", "harvest_area", "production")
    for column in numeric:
        labels[column] = pd.to_numeric(labels[column], errors="raise")
    consistency = (labels["production"] / labels["harvest_area"] - labels["yield_t_ha"]).abs()
    bad = (labels.harvest_area > 0) & (labels.yield_t_ha > 0) & (consistency / labels.yield_t_ha > 0.05)
    if bad.any():
        raise ValueError(f"{bad.sum()} source yields differ from production/harvest_area by more than 5%")
    source_labels = len(labels)
    labels = labels[(labels.yield_t_ha > 0) & (labels.harvest_area > 0)].copy()
    calendar = _read(root, "crop_calendar", ("adm_id", "sos", "eos"))
    if calendar.adm_id.duplicated().any():
        raise ValueError("Duplicate crop calendar regions")
    calendar["sos"] = pd.to_numeric(calendar.sos, errors="raise")
    calendar["eos"] = pd.to_numeric(calendar.eos, errors="raise")
    calendar["forecast_doy"] = calendar.sos + (calendar.eos - calendar.sos) / 2
    if not ((calendar.sos >= 1) & (calendar.forecast_doy > calendar.sos) &
            (calendar.eos <= 366)).all():
        raise ValueError("Crop calendar must fit a same-year growing season")
    labels = labels.merge(calendar[["adm_id", "sos", "eos", "forecast_doy"]], on="adm_id", how="inner", validate="many_to_one")
    labels = labels[labels.harvest_year >= 2003].copy()  # first year with all four signal families
    static = _read(root, "soil", ("adm_id", "awc", "bulk_density", "drainage_class"))
    locations = _read(root, "location", ("adm_id", "latitude", "longitude"))
    if static.adm_id.duplicated().any() or locations.adm_id.duplicated().any():
        raise ValueError("Duplicate static region identifiers")
    labels = labels.merge(static[["adm_id", "awc", "bulk_density", "drainage_class"]], on="adm_id", how="inner", validate="many_to_one")
    labels = labels.merge(locations[["adm_id", "latitude", "longitude"]], on="adm_id", how="inner", validate="many_to_one")
    coverage = {}
    for prefix, variables in SERIES.items():
        before = len(labels)
        labels = labels.merge(_series_features(root, labels, prefix, variables), on=["adm_id", "harvest_year"], how="left", validate="one_to_one")
        feature_columns = [f"{name}_t{step}" for name in variables for step in range(STEPS)]
        coverage[prefix] = {"rows_with_observation": int(labels[feature_columns].notna().any(axis=1).sum()),
                            "total_rows": before}
    # Keep rows with every input family observed at least once before forecast time.
    usable = np.ones(len(labels), dtype=bool)
    for variables in SERIES.values():
        names = [f"{name}_t{step}" for name in variables for step in range(STEPS)]
        usable &= labels[names].notna().any(axis=1).to_numpy()
    dropped_for_coverage = int((~usable).sum())
    labels = labels.loc[usable].copy()
    if len(labels) < 30 or labels.adm_id.nunique() < 5 or labels.harvest_year.nunique() < 5:
        raise ValueError("Too few region-year observations after modality alignment")
    output = pd.DataFrame({
        "parcel_id": labels.adm_id.astype(str),  # historical API field; IDs are administrative regions
        "spatial_unit": "administrative_region",
        "crop": "maize", "region": labels.adm_id.astype(str).str[:3],
        "campaign": labels.harvest_year.astype(int).astype(str),
        "yield_t_ha": labels.yield_t_ha.astype(float), "country": "ES",
        "latitude": labels.latitude, "longitude": labels.longitude,
        "awc": labels.awc, "bulk_density": labels.bulk_density,
        "drainage_class": labels.drainage_class,
    })
    for variables in SERIES.values():
        for name in variables:
            for step in range(STEPS):
                column = f"{name}_t{step}"
                output[column] = labels[column]
    if not np.isfinite(output.yield_t_ha).all():
        raise ValueError("Yield contains non-finite values")
    output = output.sort_values(["campaign", "parcel_id"]).reset_index(drop=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    output.to_csv(temporary, index=False)
    os.replace(temporary, target)
    files = {name: {"sha256": _sha256(root / name), "bytes": (root / name).stat().st_size}
             for name in SOURCE_FILES}
    manifest = {
        "source": "CY-Bench official WUR-AI sample, maize Spain", "source_revision": REVISION,
        "source_url": "https://github.com/WUR-AI/sample_data",
        "paper": "https://doi.org/10.5194/essd-18-3997-2026",
        "license": "CY-Bench EUPL-1.2; JRC yield labels CC BY 4.0; attribution required",
        "spatial_unit": "NUTS administrative region, not field or parcel",
        "target_unit": "t/ha", "synthetic": False,
        "forecast_cutoff": "mid-season: sos + (eos - sos)/2, same harvest year",
        "time_bins": STEPS, "bin_statistic": "sum for prec/rad/et0, mean for all other signals",
        "ndvi_transform": "(source_value - 50) / 200; values outside 50..250 missing",
        "fpar_transform": "source_percent / 100; values outside 0..100 missing",
        "source_yield_rows": source_labels, "output_rows": len(output),
        "output_regions": int(output.parcel_id.nunique()),
        "output_years": sorted(int(x) for x in labels.harvest_year.unique()),
        "dropped_no_calendar_or_early_year_or_zero_target": source_labels - len(usable),
        "dropped_missing_modality": dropped_for_coverage,
        "coverage_before_filter": coverage,
        "excluded_target_derived_columns": ["harvest_area", "production"],
        "limitations": ["Administrative-region yield cannot validate field-level or subfield-level prediction",
                        "Spain data cannot validate predictions in Peru",
                        "This sample is not a substitute for full CY-Bench or YieldSAT"],
        "source_files": files, "output_csv": str(target.resolve()),
        "output_sha256": _sha256(target),
    }
    target.with_suffix(".provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="CY-Bench public maize Spain sample adapter")
    sub = parser.add_subparsers(dest="action", required=True)
    download = sub.add_parser("download", help="Download pinned official sample files")
    download.add_argument("--directory", required=True)
    convert = sub.add_parser("convert", help="Build pre-harvest regional training CSV")
    convert.add_argument("--directory", required=True)
    convert.add_argument("--output", required=True)
    args = parser.parse_args()
    result = download_sample(args.directory) if args.action == "download" else convert_sample(args.directory, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
