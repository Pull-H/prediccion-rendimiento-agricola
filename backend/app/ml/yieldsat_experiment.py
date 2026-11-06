"""Bounded field-level pilot export with an explicit pre-harvest cutoff.

This first profile uses source spectral bands and static soil/topography only.
Meteorology with unverified aggregation, slope, TWI and coordinates are excluded.
It does not reconstruct spatial patches or claim full compliance with tema (2).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .yieldsat import _metadata_values, _unit_multiplier, _xarray


def build_field_experiment(source: str | Path, output_csv: str | Path, *,
                           crop: str = "wheat", target_unit: str,
                           target_unit_evidence: str, horizon_days: int = 240,
                           stages: int = 4, batch_size: int = 512,
                           progress=None) -> dict:
    """Aggregate all pixels of eligible field records, without sampling labels.

    Only farmer-reported sowing dates are eligible. Harvest-derived sowing
    dates are excluded because they would make the cutoff depend on the future.
    Each spectral mean has an observation-count channel: zero fills denote
    missing observations, not measured zero reflectance. No future filling.
    """
    path, output = Path(source), Path(output_csv)
    if not path.is_file():
        raise FileNotFoundError("The NetCDF is not available in this runtime")
    if output.suffix.lower() != ".csv" or output.resolve() == path.resolve():
        raise ValueError("Select a CSV destination different from the NetCDF source")
    if not crop.strip() or not target_unit_evidence.strip():
        raise ValueError("crop and target_unit_evidence are required")
    if type(horizon_days) is not int or not 1 <= horizon_days <= 730:
        raise ValueError("horizon_days must be between 1 and 730")
    if type(stages) is not int or not 2 <= stages <= 12:
        raise ValueError("stages must be between 2 and 12")
    if type(batch_size) is not int or not 1 <= batch_size <= 2048:
        raise ValueError("batch_size must be between 1 and 2048")
    xr = _xarray()
    accumulators = {}
    selected_pixels = 0
    static_candidates = ("clay_0-5", "sand_0-5", "silt_0-5", "phh2o_0-5",
                         "soc_0-5", "nitrogen_0-5", "dem")
    with xr.open_dataset(path, chunks=None, cache=False) as dataset:
        required = ("sample", "target", "times", "seeding_date", "harvesting_date",
                    "seeding_date_type", "crop", "year", "country", "farm_identifier", "field_shared_name")
        missing = [name for name in required if name not in dataset]
        if missing:
            raise ValueError("Missing required variables: " + ", ".join(missing))
        if set(dataset["sample"].dims) != {"index", "time_step", "band"}:
            raise ValueError("Unexpected sample dimensions")
        if set(dataset["times"].dims) != {"index", "time_step"}:
            raise ValueError("Expected per-pixel acquisition timestamps")
        if not np.issubdtype(dataset["times"].dtype, np.datetime64):
            raise ValueError("Acquisition timestamps were not decoded as datetime64")
        multiplier, source_unit = _unit_multiplier(dataset["target"].attrs.get("units"), target_unit)
        bands = [str(value) for value in dataset["band"].values]
        optical = [name for name in bands if re.fullmatch(r"B(?:\d{2}|8A)", name)]
        if not {"B02", "B03", "B04", "B08"}.issubset(optical):
            raise ValueError("Required optical channels B02/B03/B04/B08 are absent")
        static = [name for name in static_candidates if name in bands]
        if not {"clay_0-5", "phh2o_0-5", "soc_0-5"}.issubset(static):
            raise ValueError("Required soil channels are absent")
        chosen = optical + static
        positions = [bands.index(name) for name in chosen]
        n, t, b = int(dataset.sizes["index"]), int(dataset.sizes["time_step"]), len(optical)
        if batch_size * t * len(chosen) * 24 > 64 * 1024**2:
            raise ValueError("Reduce batch_size to keep working arrays bounded")
        metadata_names = ("crop", "year", "country", "farm_identifier", "field_shared_name",
                          "seeding_date", "harvesting_date", "seeding_date_type")
        for first in range(0, n, batch_size):
            end = min(n, first + batch_size)
            crops = _metadata_values(dataset, "crop", first, end)
            keep = np.array([value.casefold() == crop.casefold() for value in crops])
            if not keep.any():
                continue
            metadata = {name: (_metadata_values(dataset, name, first, end) if name != "crop" else crops)
                        for name in metadata_names}
            selected_pixels += int(keep.sum())
            values = np.asarray(dataset["sample"].isel(index=slice(first, end), band=positions).transpose(
                "index", "time_step", "band").values, dtype=float)
            dates = dataset["times"].isel(index=slice(first, end)).transpose("index", "time_step").values
            sow = pd.to_datetime(metadata["seeding_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
            harvest = pd.to_datetime(metadata["harvesting_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
            ages = (dates - sow[:, None]) / np.timedelta64(1, "D")
            allowed = np.isfinite(ages) & (ages >= 0) & (ages <= horizon_days)
            stage_index = np.minimum((np.where(allowed, ages, 0) * stages / horizon_days).astype(int), stages-1)
            labels = np.asarray(dataset["target"].isel(index=slice(first, end)).values, dtype=float) * multiplier
            keys = {}
            for i in np.flatnonzero(keep):
                key = tuple(metadata[name][i] for name in ("country", "farm_identifier", "field_shared_name", "crop", "year"))
                keys.setdefault(key, []).append(int(i))
            for key, indices in keys.items():
                if key not in accumulators:
                    if len(accumulators) >= 5000:
                        raise ValueError("More than 5000 field records; use a different aggregation strategy")
                    accumulators[key] = {
                        "pixels": 0, "target_sum": 0., "bad_target": False,
                        "dates": set(), "origins": set(), "date_problem": False,
                        "sum": np.zeros((stages, b)), "count": np.zeros((stages, b), dtype=np.int64),
                        "static_sum": np.zeros(len(static)), "static_count": np.zeros(len(static), dtype=np.int64),
                    }
                state = accumulators[key]
                ix = np.asarray(indices)
                state["pixels"] += len(ix)
                valid_labels = np.isfinite(labels[ix]) & (labels[ix] >= 0)
                state["bad_target"] |= not bool(valid_labels.all())
                state["target_sum"] += float(labels[ix][valid_labels].sum())
                for i in ix:
                    state["dates"].add((metadata["seeding_date"][i], metadata["harvesting_date"][i]))
                    state["origins"].add(metadata["seeding_date_type"][i])
                state["date_problem"] |= bool((np.isnat(sow[ix]) | np.isnat(harvest[ix])
                                                | ((harvest[ix] - sow[ix]) <= np.timedelta64(horizon_days, "D"))).any())
                spectral = values[ix, :, :b]
                for stage in range(stages):
                    valid = np.isfinite(spectral) & (allowed[ix] & (stage_index[ix] == stage))[:, :, None]
                    state["sum"][stage] += np.where(valid, spectral, 0.).sum(axis=(0, 1))
                    state["count"][stage] += valid.sum(axis=(0, 1))
                static_values = values[ix, :, b:]
                valid_static = np.isfinite(static_values)
                state["static_sum"] += np.where(valid_static, static_values, 0.).sum(axis=(0, 1))
                state["static_count"] += valid_static.sum(axis=(0, 1))
            if progress:
                progress({"source_rows_processed": end, "source_rows": n, "field_records": len(accumulators)})

        rows, exclusions, field_provenance = [], [], []
        reported_origins = {"provided_by_farmer", "farmer_reported", "reported_by_farmer"}
        for key, state in sorted(accumulators.items()):
            reasons = []
            if state["bad_target"]:
                reasons.append("invalid target in field record")
            if state["date_problem"] or len(state["dates"]) != 1:
                reasons.append("missing/inconsistent dates or crop harvested on/before cutoff")
            if not state["origins"] or any(origin.casefold().strip() not in reported_origins for origin in state["origins"]):
                reasons.append("sowing date not explicitly farmer-reported; may depend on future harvest")
            if not state["count"].any():
                reasons.append("no valid optical observation within cutoff")
            if not re.fullmatch(r"(?:19|20)\d{2}", key[4]):
                reasons.append("undecoded or invalid campaign year")
            elif len(state["dates"]) == 1:
                harvest_text = next(iter(state["dates"]))[1]
                if harvest_text[:4] != key[4]:
                    reasons.append("harvest year differs from campaign label")
            if reasons:
                exclusions.append({"field": key[2], "year": key[4], "sowing_date_origins": sorted(state["origins"]), "reasons": reasons})
                continue
            row = {"parcel_id": json.dumps(list(key[:3]), separators=(",", ":")),
                   "spatial_unit": "parcel", "crop": key[3], "region": f"{key[0]}/{key[1]}",
                   "campaign": key[4], "country": key[0], "yield_t_ha": state["target_sum"]/state["pixels"]}
            means = np.divide(state["sum"], state["count"], out=np.zeros_like(state["sum"]), where=state["count"] > 0)
            for stage in range(stages):
                for i, name in enumerate(optical):
                    row[f"{name}_source_t{stage}"] = float(means[stage, i])
                    row[f"{name}_observations_t{stage}"] = float(state["count"][stage, i]/state["pixels"])
            for i, name in enumerate(static):
                prefix = re.sub(r"[^A-Za-z0-9_]+", "_", name) + "_source"
                count = int(state["static_count"][i])
                row[prefix] = float(state["static_sum"][i]/count) if count else 0.
                row[prefix + "_observed_fraction"] = count/(state["pixels"]*t)
            rows.append(row)
            seeding, harvesting = next(iter(state["dates"]))
            field_provenance.append({"parcel_id": row["parcel_id"], "pixels": state["pixels"],
                                     "sowing_date": seeding, "harvesting_date": harvesting,
                                     "sowing_date_origins": sorted(state["origins"]),
                                     "cutoff": str((np.datetime64(seeding, "D") + np.timedelta64(horizon_days, "D")))})
        manifest = {
            "profile": "YieldSAT field-level satellite+soil pilot", "crop": crop,
            "source_file": path.name, "source_size_bytes": path.stat().st_size,
            "candidate_field_records": len(accumulators), "selected_crop_pixels": selected_pixels,
            "output_rows": len(rows), "excluded_fields": exclusions, "field_provenance": field_provenance,
            "horizon_days_after_sowing": horizon_days, "stages": stages,
            "sowing_policy": "farmer-reported only; harvest-derived and unknown origins excluded",
            "aggregation": "all source pixels per eligible field record; means over available pixel-time observations in each day window",
            "missing_values": "zero sentinel paired with per-band observation counts; no interpolation or future filling",
            "source_target_declared_unit": dataset["target"].attrs.get("units"),
            "source_target_unit_used": source_unit, "target_unit_evidence": target_unit_evidence,
            "output_target_unit": "t/ha", "spatial_unit": "parcel",
            "feature_units": "source values unchanged; no physical unit or radiometric conversion inferred",
            "weather_status": "excluded pending units/aggregation verification",
            "excluded_source_channels": [name for name in bands if name not in chosen],
            "limitations": ["Initial satellite+soil profile; climate fusion, calibrated indices and spatial CNN patches remain pending",
                            "Field records may not identify the same physical parcel across years",
                            "Retrospective eligibility excludes fields already harvested by the cutoff; report resulting selection",
                            "No models have been trained by this exporter"],
        }
        receipt_path = path.with_suffix(path.suffix + ".receipt.json")
        manifest["source_sha256"] = None
        manifest["source_hash_verification"] = "not computed by exporter"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("bytes") == path.stat().st_size
                    and re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256", "")))):
                manifest["source_sha256"] = receipt["sha256"]
                manifest["source_hash_verification"] = "from extraction receipt; not recomputed by exporter"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if len(frame):
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        if not np.isfinite(numeric).all():
            raise ValueError("Non-finite exported features")
        csv_bytes = frame.to_csv(index=False).encode("utf-8")
        if len(csv_bytes) > 20_000_000:
            raise ValueError("CSV exceeds the application upload limit")
        temp = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
        try:
            temp.write_bytes(csv_bytes)
            os.replace(temp, output)
        finally:
            if temp.exists():
                temp.unlink()
        manifest["output_csv"] = output.name
        manifest["output_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
        manifest["output_bytes"] = len(csv_bytes)
        manifest["field_counts_by_year"] = {str(k): int(v) for k, v in frame.groupby("campaign").size().items()}
    else:
        # Never leave a stale successful export next to a blocked manifest.
        if output.exists():
            raise ValueError("No eligible records; choose a new output path to avoid mixing with an older CSV")
        manifest["status"] = "blocked: no eligible field records; inspect exclusions before changing the protocol"
    sidecar = output.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return manifest
