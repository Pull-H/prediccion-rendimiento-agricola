# Caso ilustrativo de trigo (Crop Yield Prediction Dataset)

Resultados agregados del artículo AgroVisión, depositados con la versión 1.1.0
(DOI [10.5281/zenodo.24352582](https://doi.org/10.5281/zenodo.24352582)). No contienen
datos de la fuente ni modelos entrenados; el CSV derivado debe regenerarse a partir
del archivo original.

| Archivo | Contenido |
| --- | --- |
| `trigo_repro_*_busqueda.json` | Búsqueda anidada temporal (Random Forest y XGBoost) |
| `trigo_repro_robustez.json` | Persistencia, MASE/WAPE, contrastes pareados, países no vistos, recalibración conformal y partición 70/15/15 |
| `trigo_lag_*` | Análisis exploratorio post hoc con el rendimiento de la campaña anterior |
| `figura_*.png` | Figuras 2 y 3 del artículo (600 ppp) |

Fuente: <https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset>
(archivo ZIP con SHA-256 `cd9360e68cb884a6d98a884c78ded0d98f34f949d2fd14c1c1f5aa2cbddceee0`).
CSV de trigo derivado: SHA-256 `dbc0964b57d995f25d321145f7f77704db31cc8edd0c44fa10dd0d7b6707fa66`.

Entorno: Python 3.14.5, NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, XGBoost 3.4.1,
Optuna 5.0.0. Semilla 42.

```powershell
$env:PYTHONPATH = 'backend'
python -m app.ml.kaggle --source crop-yield-prediction-dataset.zip --output data/kaggle/yield-all.csv
# Filtrar crop == "Wheat" y guardar como data/kaggle/wheat-yield.csv
python -m app.ml.tuning --source data/kaggle/wheat-yield.csv --output artifacts/rf --model RandomForest --validation temporal --trials 12
python -m app.ml.tuning --source data/kaggle/wheat-yield.csv --output artifacts/xgb --model XGBoost --validation temporal --trials 12
python -m app.ml.robustness --source data/kaggle/wheat-yield.csv --runs artifacts/rf/<run>.json artifacts/xgb/<run>.json --output artifacts/robustez.json --figures artifacts/figuras
```

La variante con rezago añade la columna `yield_lag1_t_ha` (rendimiento de la campaña
anterior disponible del mismo país) y repite los mismos comandos. Se diseñó después de
conocer el periodo de prueba y no constituye una evaluación confirmatoria.
