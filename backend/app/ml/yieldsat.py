"""Local, bounded adapter for YieldSAT's preprocessed (ML-ready) NetCDF files.

The adapter never downloads data. It reads ``sample(index,time_step,band)`` in
bounded slices and keeps a manifest next to each converted CSV. A CSV contains
pixel/sample rows, not field aggregates; all rows from the same field receive
the same parcel_id so the modelling split can keep them together.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


NETCDF_SUFFIXES = {".nc", ".nc4", ".netcdf"}
REQUIRED_VARIABLES = ("sample", "target", "band", "crop", "year",
                      "field_shared_name", "farm_identifier", "country")
DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024 * 1024


def _xarray():
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("YieldSAT NetCDF support requires backend/yieldsat-requirements.txt") from exc
    return xr


def discover_yieldsat_files(root: str | Path, variant: str = "merged") -> list[str]:
    """Find local NetCDF files; ``merged`` accepts merged/ and merge_*.nc."""
    source = Path(root).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"YieldSAT path not found: {source}")
    if source.is_file():
        if source.suffix.lower() not in NETCDF_SUFFIXES:
            raise ValueError("YieldSAT source must be a NetCDF file")
        return [str(source.resolve())]
    if variant not in {"merged", "all"}:
        raise ValueError("variant must be 'merged' or 'all'")
    candidates = sorted((item for item in source.rglob("*")
                         if item.is_file() and item.suffix.lower() in NETCDF_SUFFIXES),
                        key=lambda item: str(item).lower())
    if variant == "merged":
        candidates = [item for item in candidates
                      if any(part.lower().startswith(("merged", "merge_", "merge-"))
                             for part in item.relative_to(source).parts)]
    return [str(item.resolve()) for item in candidates]


def _source_path(source: str | Path, variant: str) -> Path:
    paths = discover_yieldsat_files(source, variant)
    if not paths:
        raise FileNotFoundError(f"No {variant} NetCDF file found under {source}")
    if len(paths) > 1:
        raise ValueError(f"Found {len(paths)} NetCDF files; select one explicit file path")
    return Path(paths[0])


def _string(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return str(value).strip()


def _code_map(array: Any, dataset_attrs: dict | None = None, variable_name: str | None = None) -> dict[str, str]:
    sources = [array.attrs]
    if dataset_attrs and variable_name:
        sources.append({key: dataset_attrs.get(f"{variable_name}_{key}")
                        for key in ("code_to_name", "mapping", "categories", "classes")})
        sources.append({"mapping": dataset_attrs.get(variable_name)})
    for attrs in sources:
        for key in ("code_to_name", "mapping", "categories", "classes"):
            value = attrs.get(key)
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    continue
            if isinstance(value, dict):
                return {_string(code): _string(name) for code, name in value.items()}
        # The official release stores categories directly as numeric attribute
        # names (e.g. crop.attrs['0'] = 'wheat'), rather than a nested mapping.
        direct = {_string(code): _string(name) for code, name in attrs.items()
                  if re.fullmatch(r"-?\d+", _string(code)) and _string(name)}
        if direct:
            return direct
    return {}


def _label(value: Any, mapping: dict[str, str]) -> str:
    raw = _string(value)
    return mapping.get(raw, raw)


def _band_columns(dataset: Any) -> tuple[list[str], list[dict[str, str]]]:
    raw_bands = dataset["band"].values  # only the small coordinate is read
    names = []
    descriptions = []
    mapping = _code_map(dataset["band"], dataset.attrs, "band")
    for value in raw_bands:
        label = _label(value, mapping)
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_")
        if not cleaned:
            raise ValueError("A band has an empty label")
        if cleaned[0].isdigit():
            cleaned = f"band_{cleaned}"
        names.append(cleaned)
        descriptions.append({"source_band": _string(value), "feature_prefix": cleaned})
    if len(names) != len(set(names)):
        raise ValueError("Band labels are duplicated after CSV-safe normalization")
    return names, descriptions


def _inspect_open(dataset: Any, path: Path) -> dict[str, Any]:
    errors = []
    warnings = []
    missing = [name for name in REQUIRED_VARIABLES if name not in dataset]
    if missing:
        errors.append(f"Missing variables/coordinates: {', '.join(missing)}")
    dimensions = {key: int(value) for key, value in dataset.sizes.items()}
    if "sample" in dataset:
        sample = dataset["sample"]
        if set(sample.dims) != {"index", "time_step", "band"} or len(sample.dims) != 3:
            errors.append("sample must have dimensions index,time_step,band")
        if not np.issubdtype(sample.dtype, np.number):
            errors.append("sample must be numeric")
    if "target" in dataset:
        target = dataset["target"]
        if target.dims != ("index",):
            errors.append("target must have dimension index")
        if not np.issubdtype(target.dtype, np.number):
            errors.append("target must be numeric")
        if not target.attrs.get("units"):
            warnings.append("target units missing; conversion requires target_unit")
    for name in ("crop", "year", "field_shared_name", "farm_identifier", "country"):
        if name in dataset and dataset[name].dims not in (("index",), ()):
            errors.append(f"{name} must be scalar or indexed by index")
    if "region" in dataset and dataset["region"].dims not in (("index",), ()):
        errors.append("region must be scalar or indexed by index")
    if "band" in dataset and dataset["band"].dims != ("band",):
        errors.append("band coordinate must have dimension band")
    if "times" not in dataset:
        warnings.append("times metadata absent; time-step dates cannot be traced")
    elif not set(dataset["times"].dims).issubset({"index", "time_step"}):
        errors.append("times dimensions must use index and/or time_step")
    for name in ("index", "time_step", "band"):
        if dimensions.get(name, 0) < 1:
            errors.append(f"dimension {name} must be nonempty")
    band_descriptions = []
    if "band" in dataset and dimensions.get("band", 0) > 0:
        try:
            _, band_descriptions = _band_columns(dataset)
        except ValueError as exc:
            errors.append(str(exc))
    if "region" not in dataset:
        warnings.append("region metadata absent; country will be copied into region with country-level granularity")
    if "country" in dataset and np.issubdtype(dataset["country"].dtype, np.number):
        if not _code_map(dataset["country"], dataset.attrs, "country"):
            warnings.append("country values are numeric codes without a name mapping; region will retain the source codes")
    return {
        "source_path": str(path.resolve()), "file_size_bytes": path.stat().st_size,
        "dimensions": dimensions,
        "variables": list(dataset.variables), "band_mapping": band_descriptions,
        "target_units": dataset["target"].attrs.get("units") if "target" in dataset else None,
        "valid": not errors, "errors": errors, "warnings": warnings,
        "synthetic_demo": _string(dataset.attrs.get("synthetic_demo", "false")).lower() in {"true", "1", "yes"},
    }


def inspect_yieldsat_file(source: str | Path, variant: str = "merged") -> dict[str, Any]:
    """Inspect dimensions and metadata without materializing the sample cube."""
    path = _source_path(source, variant)
    xr = _xarray()
    with xr.open_dataset(path, chunks=None, cache=False, decode_times=False) as dataset:
        return _inspect_open(dataset, path)


def audit_yieldsat_quality(source: str | Path, *, probe_rows: int = 256,
                          metadata_batch_size: int = 8192,
                          max_probe_bytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    """Scan 1D labels/groups and probe all time steps in bounded index windows.

    The feature probe is a diagnostic sample, not a complete quality audit or
    a representative experimental subset. No features or large files are saved.
    """
    from collections import Counter

    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError("NetCDF not found; reuse the existing Colab runtime or prepare it again")
    for name, value in (("probe_rows", probe_rows), ("metadata_batch_size", metadata_batch_size),
                        ("max_probe_bytes", max_probe_bytes)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if metadata_batch_size > 65536:
        raise ValueError("metadata_batch_size must not exceed 65536")
    xr = _xarray()
    with xr.open_dataset(path, chunks=None, cache=False) as dataset:
        required = ("sample", "target", "band", "crop", "year", "country",
                    "farm_identifier", "field_shared_name")
        missing = [name for name in required if name not in dataset]
        if missing:
            raise ValueError(f"Missing variables: {', '.join(missing)}")
        sample = dataset["sample"]
        if set(sample.dims) != {"index", "time_step", "band"}:
            raise ValueError("sample must have dimensions index,time_step,band")
        if dataset["target"].dims != ("index",):
            raise ValueError("target must have dimension index")
        n, steps, channels = (int(dataset.sizes[key]) for key in ("index", "time_step", "band"))
        if min(n, steps, channels) < 1:
            raise ValueError("Dataset dimensions must be nonempty")
        rows = min(probe_rows, n)
        # Reserve both the float64 probe and an additional working array.
        estimated = rows * steps * channels * 16
        if estimated > max_probe_bytes:
            raise ValueError(f"Probe working arrays need about {estimated} bytes; reduce probe_rows")
        categorical_names = ("crop", "year", "country", "farm_identifier", "field_shared_name")
        mappings = {name: _code_map(dataset[name], dataset.attrs, name) for name in categorical_names}
        counts = {name: Counter() for name in ("crop", "year", "country")}
        farms, groups = set(), Counter()
        finite_targets = invalid_targets = negative_targets = 0
        target_min, target_max = float("inf"), float("-inf")
        for first in range(0, n, metadata_batch_size):
            end = min(n, first + metadata_batch_size)
            block = {}
            for name in categorical_names:
                var = dataset[name]
                if var.dims not in (("index",), ()):
                    raise ValueError(f"{name} must be scalar or indexed by index")
                values = [var.values.item()] * (end-first) if var.dims == () else var.isel(index=slice(first, end)).values
                block[name] = [_label(value, mappings[name]) for value in values]
                if any(not label for label in block[name]):
                    raise ValueError(f"{name} contains missing group labels")
            for name in counts:
                counts[name].update(block[name])
            farms.update(zip(block["country"], block["farm_identifier"]))
            groups.update(zip(block["country"], block["farm_identifier"], block["field_shared_name"],
                              block["crop"], block["year"]))
            if len(groups) > 20000 or len(farms) > 20000:
                raise ValueError("More than 20000 metadata groups; a different audit strategy is needed")
            values = np.asarray(dataset["target"].isel(index=slice(first, end)).values, dtype=float)
            valid = np.isfinite(values)
            finite_targets += int(valid.sum())
            invalid_targets += int((~valid).sum())
            negative_targets += int((values[valid] < 0).sum())
            if valid.any():
                target_min = min(target_min, float(values[valid].min()))
                target_max = max(target_max, float(values[valid].max()))

        windows = min(8, rows)
        width = max(1, rows // windows)
        starts = np.unique(np.linspace(0, max(0, n-width), windows, dtype=int))
        ranges, probed = [], 0
        finite_counts = np.zeros((steps, channels), dtype=np.int64)
        band_min = np.full(channels, np.inf)
        band_max = np.full(channels, -np.inf)
        band_names = [_string(value) for value in dataset["band"].values]
        optical = [i for i, name in enumerate(band_names) if re.fullmatch(r"B(?:\d{2}|8A)", name)]
        optical_counts = np.zeros(steps, dtype=np.int64)
        optical_rows_with_any = 0
        for start in starts:
            first = int(start)
            end = min(n, first + width, first + rows-probed)
            if end <= first:
                continue
            values = np.asarray(sample.isel(index=slice(first, end)).transpose(
                "index", "time_step", "band").values, dtype=float)
            finite = np.isfinite(values)
            finite_counts += finite.sum(axis=0)
            band_min = np.minimum(band_min, np.where(finite, values, np.inf).min(axis=(0, 1)))
            band_max = np.maximum(band_max, np.where(finite, values, -np.inf).max(axis=(0, 1)))
            if optical:
                optical_available = finite[:, :, optical].all(axis=2)
                optical_counts += optical_available.sum(axis=0)
                optical_rows_with_any += int(optical_available.any(axis=1).sum())
            ranges.append({"start": first, "stop": end})
            probed += end-first

        attrs_names = ("target", "sample", "times", "seeding_date", "harvesting_date")
        attributes = {name: {str(key): _string(value) for key, value in dataset[name].attrs.items()}
                      for name in attrs_names if name in dataset}
        if "times" in dataset:
            attributes["times"]["decoded_dtype"] = str(dataset["times"].dtype)
            for key in ("units", "calendar"):
                if key in dataset["times"].encoding:
                    attributes["times"]["encoding_" + key] = str(dataset["times"].encoding[key])
        units = _string(dataset["target"].attrs.get("units")) or None
        warnings = ["Feature coverage is based on bounded diagnostic windows, not the complete cube",
                    "Full-season inputs require a prediction cutoff before any forecast experiment",
                    "Rows are pixels; groups are field records and are not automatically distinct physical parcels"]
        if not units:
            warnings.append("Target unit is not declared in this file; document and supply an explicit unit before conversion")
        unknown_codes = [name for name in categorical_names
                         if np.issubdtype(dataset[name].dtype, np.number) and not mappings[name]]
        if unknown_codes:
            warnings.append("Numeric category mappings missing for: " + ", ".join(unknown_codes))
        if optical and optical_rows_with_any == 0:
            warnings.append("No complete optical observation found in the diagnostic windows")
        return {
            "audit_version": 1, "source_file": path.name, "source_file_bytes": path.stat().st_size,
            "dimensions": {"index": n, "time_step": steps, "band": channels},
            "metadata_rows_scanned": n, "crop_pixel_counts": dict(counts["crop"]),
            "year_pixel_counts": dict(counts["year"]), "country_pixel_counts": dict(counts["country"]),
            "farm_groups": len(farms), "field_record_groups": len(groups),
            "fields": [{"country": key[0], "farm": key[1], "field": key[2],
                        "crop": key[3], "year": key[4], "pixels": count}
                       for key, count in sorted(groups.items())],
            "categorical_mappings": mappings, "attributes": attributes,
            "target": {"declared_unit": units, "finite_count": finite_targets,
                       "invalid_count": invalid_targets, "negative_count": negative_targets,
                       "min": target_min if finite_targets else None,
                       "max": target_max if finite_targets else None},
            "probe": {"rows": probed, "ranges": ranges, "all_time_steps": True,
                      "all_bands": True, "working_arrays_upper_estimate_bytes": estimated,
                      "finite_fraction_by_time_step": (finite_counts.sum(axis=1)/(probed*channels)).tolist(),
                      "optical_band_names": [band_names[i] for i in optical],
                      "complete_optical_fraction_by_time_step": (optical_counts/probed).tolist() if optical else None,
                      "rows_with_any_complete_optical_observation": optical_rows_with_any if optical else None,
                      "bands": [{"name": name, "finite_fraction": float(finite_counts[:, i].sum()/(probed*steps)),
                                 "min": float(band_min[i]) if np.isfinite(band_min[i]) else None,
                                 "max": float(band_max[i]) if np.isfinite(band_max[i]) else None}
                                for i, name in enumerate(band_names)]},
            "warnings": warnings, "status": "diagnostic audit only; no training, unit conversion or feature scaling performed",
        }


def _unit_multiplier(source_unit: Any, explicit_unit: str | None) -> tuple[float, str]:
    def canonical(value: Any) -> str | None:
        if value is None:
            return None
        name = re.sub(r"\s+", "", _string(value).lower())
        if name in {"t/ha", "tonne/ha", "tonnes/ha", "metrictons/ha", "mg/ha",
                    "mgha-1", "mgha^-1", "tha-1", "tha^-1"}:
            return "t/ha"
        if name in {"kg/ha", "kgha-1", "kgha^-1"}:
            return "kg/ha"
        raise ValueError(f"Unsupported target unit {value!r}; use t/ha or kg/ha")

    source = canonical(source_unit)
    explicit = canonical(explicit_unit)
    if source and explicit and source != explicit:
        raise ValueError(f"target_unit {explicit} conflicts with source units {source}")
    unit = source or explicit
    if unit is None:
        raise ValueError("Target units are absent; pass target_unit='t/ha' or 'kg/ha' explicitly")
    return (0.001 if unit == "kg/ha" else 1.0), unit


def _metadata_values(dataset: Any, name: str, start: int, stop: int) -> list[str]:
    variable = dataset[name]
    mapping = _code_map(variable, dataset.attrs, name)
    if variable.dims == ():
        values = [variable.values.item()] * (stop - start)
    else:
        values = variable.isel(index=slice(start, stop)).values
    labels = [_label(value, mapping) for value in values]
    if any(not label for label in labels):
        raise ValueError(f"{name} has a missing value in rows {start}:{stop}")
    return labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_yieldsat_to_csv(source: str | Path, output_csv: str | Path, *, max_samples: int,
                            start: int = 0, batch_size: int = 256,
                            target_unit: str | None = None, variant: str = "merged",
                            max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Convert an explicit contiguous index range to the internal training CSV.

    ``max_samples`` is required: there is no implicit full import or hidden
    downsampling. The output byte limit is checked before and during writing.
    """
    path = _source_path(source, variant)
    target = Path(output_csv).expanduser()
    if target.suffix.lower() != ".csv":
        raise ValueError("output_csv must end in .csv")
    if target.resolve() == path.resolve():
        raise ValueError("Source and output paths must differ")
    for name, value in (("max_samples", max_samples), ("batch_size", batch_size),
                        ("max_output_bytes", max_output_bytes)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(start) is not int or start < 0:
        raise ValueError("start must be a nonnegative integer")
    xr = _xarray()
    with xr.open_dataset(path, chunks=None, cache=False, decode_times=False) as dataset:
        inspection = _inspect_open(dataset, path)
        if not inspection["valid"]:
            raise ValueError("Invalid YieldSAT file: " + "; ".join(inspection["errors"]))
        total = inspection["dimensions"]["index"]
        if start >= total or max_samples > total - start:
            raise ValueError(f"Requested [{start}, {start + max_samples}) exceeds {total} samples")
        multiplier, source_unit = _unit_multiplier(dataset["target"].attrs.get("units"), target_unit)
        band_names, band_mapping = _band_columns(dataset)
        timesteps = inspection["dimensions"]["time_step"]
        temporal_columns = [f"{name}_t{step}" for step in range(timesteps) for name in band_names]
        reserved = {"parcel_id", "crop", "region", "campaign", "yield_t_ha", "country", "synthetic_demo"}
        if reserved.intersection(temporal_columns):
            raise ValueError("A band label collides with an internal column name")
        columns = ["parcel_id", "crop", "region", "campaign", "yield_t_ha", "country",
                   *temporal_columns, "synthetic_demo"]
        estimated_bytes = max_samples * (len(columns) * 24 + 64) + len(",".join(columns)) + 1
        if estimated_bytes > max_output_bytes:
            raise ValueError(f"Estimated CSV size {estimated_bytes} bytes exceeds max_output_bytes={max_output_bytes}; set an explicit smaller max_samples or larger byte limit")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        region_source = "region" if "region" in dataset else "country"
        try:
            with temp.open("w", encoding="utf-8", newline="") as stream:
                for first in range(start, start + max_samples, batch_size):
                    end = min(first + batch_size, start + max_samples)
                    sample = np.asarray(dataset["sample"].isel(index=slice(first, end)).transpose(
                        "index", "time_step", "band").values, dtype=float)
                    if np.isinf(sample).any():
                        raise ValueError(f"sample contains infinite features in rows {first}:{end}")
                    labels = np.asarray(dataset["target"].isel(index=slice(first, end)).values,
                                        dtype=float) * multiplier
                    if not np.isfinite(labels).all() or (labels < 0).any():
                        raise ValueError(f"target contains missing, infinite or negative yields in rows {first}:{end}")
                    crops = _metadata_values(dataset, "crop", first, end)
                    countries = _metadata_values(dataset, "country", first, end)
                    regions = _metadata_values(dataset, region_source, first, end)
                    campaigns = _metadata_values(dataset, "year", first, end)
                    fields = _metadata_values(dataset, "field_shared_name", first, end)
                    farms = _metadata_values(dataset, "farm_identifier", first, end)
                    parcels = [json.dumps([country, farm, field], ensure_ascii=False,
                                          separators=(",", ":"))
                               for country, farm, field in zip(countries, farms, fields)]
                    frame = pd.DataFrame(sample.reshape(end - first, -1), columns=temporal_columns)
                    frame.insert(0, "country", countries)
                    frame.insert(0, "yield_t_ha", labels)
                    frame.insert(0, "campaign", campaigns)
                    frame.insert(0, "region", regions)
                    frame.insert(0, "crop", crops)
                    frame.insert(0, "parcel_id", parcels)
                    frame["synthetic_demo"] = inspection["synthetic_demo"]
                    frame.to_csv(stream, index=False, header=(first == start), columns=columns)
                    stream.flush()
                    if temp.stat().st_size > max_output_bytes:
                        raise ValueError(f"CSV exceeded max_output_bytes={max_output_bytes} during conversion")
            manifest = {
                "adapter": "YieldSAT ML-ready NetCDF", "source_path": str(path.resolve()),
                "source_sha256": _sha256(path), "source_size_bytes": path.stat().st_size,
                "source_license": _string(dataset.attrs.get("license")) or None,
                "source_version": _string(dataset.attrs.get("version")) or None,
                "source_title": _string(dataset.attrs.get("title")) or None,
                "output_csv": str(target.resolve()), "output_rows": max_samples,
                "source_rows": total, "selection": {"start": start, "count": max_samples,
                                                    "method": "contiguous source index range"},
                "sample_shape": list(dataset["sample"].shape),
                "source_target_unit": source_unit, "output_target_unit": "t/ha",
                "band_mapping": band_mapping,
                "time_step_mapping": [{"source_position": step, "output_suffix": f"t{step}"}
                                      for step in range(timesteps)],
                "times_variable_shape": list(dataset["times"].shape) if "times" in dataset else None,
                "region_source": region_source,
                "region_granularity": "region" if region_source == "region" else "country",
                "parcel_id_source": ["country", "farm_identifier", "field_shared_name"],
                "synthetic_demo": inspection["synthetic_demo"],
                "validation_issues": inspection["warnings"],
                "limitations": (["Contiguous subsets are development samples, not representative validation samples"]
                                if max_samples < total else [])
                               + ["Per-sample acquisition dates in times are not exported to tabular CSV; source time-step order is preserved"],
            }
            sidecar = target.with_suffix(".provenance.json")
            sidecar_tmp = sidecar.with_name(f".{sidecar.name}.{uuid4().hex}.tmp")
            sidecar_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, target)
            os.replace(sidecar_tmp, sidecar)
            return manifest
        finally:
            if temp.exists():
                temp.unlink()
