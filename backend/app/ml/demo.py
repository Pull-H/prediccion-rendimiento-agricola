"""Clearly labelled synthetic data for exercising the application only."""

from pathlib import Path

import numpy as np
import pandas as pd


def create_demo_csv(path: str | Path, seed: int = 42) -> str:
    """Write a synthetic CSV and return its path; it is not experimental evidence."""
    rng = np.random.default_rng(seed)
    rows = []
    crops = ("maize", "wheat", "rice")
    regions = ("demo_north", "demo_central", "demo_south")
    for parcel in range(90):
        crop = crops[parcel % len(crops)]
        region = regions[(parcel // len(crops)) % len(regions)]
        soil_ph = float(rng.normal(6.5, 0.5))
        soil_carbon = float(rng.uniform(0.8, 3.2))
        elevation_m = float(rng.uniform(100, 1600))
        for campaign in ("demo_2023", "demo_2024"):
            row = {
                "parcel_id": f"synthetic_{parcel:03d}",
                "crop": crop,
                "region": region,
                "campaign": campaign,
                "soil_ph": soil_ph,
                "soil_carbon_pct": soil_carbon,
                "elevation_m": elevation_m,
            }
            ndvis = []
            rains = []
            for timestep in range(4):
                ndvi = float(np.clip(0.2 + timestep * 0.13 + rng.normal(0, 0.07), 0, 1))
                rain = float(max(0, rng.normal(75 + 10 * timestep, 15)))
                temp = float(rng.normal(22, 3))
                row[f"ndvi_t{timestep}"] = ndvi
                row[f"rain_mm_t{timestep}"] = rain
                row[f"temperature_c_t{timestep}"] = temp
                ndvis.append(ndvi)
                rains.append(rain)
            row["yield_t_ha"] = float(max(0.1, 1.5 + 4 * np.mean(ndvis)
                + 0.005 * sum(rains) + 0.25 * soil_carbon + rng.normal(0, 0.45)))
            row["synthetic_demo"] = True
            rows.append(row)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    return str(target.resolve())
