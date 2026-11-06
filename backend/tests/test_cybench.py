import pandas as pd

from app.ml import inspect_csv
from app.ml.cybench import convert_sample


def test_cybench_adapter_uses_only_pre_forecast_observations(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    regions = [f"ES1{index:02d}" for index in range(5)]
    labels, calendar, locations, soil = [], [], [], []
    weather, moisture, ndvi, fpar = [], [], [], []
    for region_index, region in enumerate(regions):
        calendar.append({"crop_name": "maize", "adm_id": region, "sos": 100, "eos": 300})
        locations.append({"crop_name": "maize", "adm_id": region, "latitude": 40 + region_index, "longitude": -4})
        soil.append({"crop_name": "maize", "adm_id": region, "awc": 10, "bulk_density": 1.5, "drainage_class": 4})
        for year in range(2003, 2009):
            observed = 5 + region_index / 10 + (year - 2003) / 10
            labels.append({"crop_name": "maize", "country_code": "ES", "adm_id": region,
                           "harvest_year": year, "yield": observed, "harvest_area": 100,
                           "production": observed * 100})
            for month, day in ((4, 20), (5, 10), (6, 1), (7, 1), (12, 1)):
                date = year * 10000 + month * 100 + day
                late = month == 12
                weather.append({"crop_name": "maize", "adm_id": region, "date": date,
                                "tmin": 10, "tmax": 25, "prec": 3, "rad": 100,
                                "tavg": 17, "et0": 2, "vpd": 1, "cwb": 1})
                moisture.append({"crop_name": "maize", "adm_id": region, "date": date,
                                 "ssm": 5, "rsm": 200})
                ndvi.append({"crop_name": "maize", "adm_id": region, "date": date,
                             "ndvi": 250 if late else 150})
                fpar.append({"crop_name": "maize", "adm_id": region, "date": date,
                             "fpar": 90 if late else 40})
    for prefix, records in (("yield", labels), ("crop_calendar", calendar),
                            ("location", locations), ("soil", soil),
                            ("meteo", weather), ("soil_moisture", moisture),
                            ("ndvi", ndvi), ("fpar", fpar)):
        pd.DataFrame(records).to_csv(root / f"{prefix}_maize_ES.csv", index=False)
    output = tmp_path / "training.csv"
    manifest = convert_sample(root, output)
    frame = pd.read_csv(output)
    assert len(frame) == 30
    assert manifest["output_rows"] == 30
    assert manifest["spatial_unit"].startswith("NUTS administrative region")
    assert "production" not in frame and "harvest_area" not in frame
    assert frame["ndvi_t0"].eq(0.5).all() and frame["fpar_t3"].eq(0.4).all()
    assert set(frame["country"]) == {"ES"}
    assert frame["spatial_unit"].eq("administrative_region").all()
    summary = inspect_csv(output)
    assert summary["unique_campaigns"] == 6
    assert summary["spatial_unit"] == "administrative_region"
    assert "spatial_unit" not in summary["feature_names"]
