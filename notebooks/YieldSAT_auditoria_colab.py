"""Paste this file into one new cell of the existing YieldSAT Colab notebook.
No ZIP is downloaded. Only an audit JSON is exported to the browser.
Generated from backend/app/ml/yieldsat.py; do not edit helpers independently.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
import numpy as np

def _xarray():
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("YieldSAT NetCDF support requires backend/yieldsat-requirements.txt") from exc
    return xr

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

# Run only inside the existing managed Colab runtime.
try:
    from google.colab import files
except ImportError:
    raise RuntimeError("Ejecuta esta celda en Colab, no en tu equipo local.") from None
if not Path('/content').is_dir():
    raise RuntimeError("No se encontro el disco remoto de Colab.")
source = Path(globals().get('netcdf_path', '/content/yieldsat_work/Germany.nc')).resolve()
if Path('/content') not in source.parents:
    raise RuntimeError("El NetCDF debe estar en el disco remoto /content.")
if not source.is_file():
    raise RuntimeError("No esta el NetCDF: usa la misma sesion de Colab que lo extrajo. Si termino, deberas prepararlo nuevamente en Colab.")
print('Auditando metadatos y 256 pixeles distribuidos; no se descarga ningun ZIP.')
quality_report = audit_yieldsat_quality(source, probe_rows=256)
quality_path = source.parent/'inspection_quality.json'
quality_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({
    'crop_pixel_counts': quality_report['crop_pixel_counts'],
    'year_pixel_counts': quality_report['year_pixel_counts'],
    'field_record_groups': quality_report['field_record_groups'],
    'target': quality_report['target'],
    'finite_fraction_by_time_step': quality_report['probe']['finite_fraction_by_time_step'],
    'rows_with_any_complete_optical_observation': quality_report['probe']['rows_with_any_complete_optical_observation'],
    'warnings': quality_report['warnings'],
}, indent=2, ensure_ascii=False))
print('Se descarga solo el informe pequeno:', quality_path.name)
files.download(str(quality_path))
