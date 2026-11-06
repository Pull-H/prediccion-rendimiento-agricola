# CY-Bench: entrenamiento con observaciones reales

## Fuente y alcance

La aplicación usa la [muestra oficial de CY-Bench publicada por WUR-AI](https://github.com/WUR-AI/sample_data), fijada en el commit `b55e6918de1e1fa1b998966f6014b578cd36b83b`. CY-Bench integra estadísticas de rendimiento con clima, suelo e indicadores satelitales; su [artículo científico](https://doi.org/10.5194/essd-18-3997-2026) y [código oficial](https://github.com/WUR-AI/AgML-CY-Bench) documentan la construcción. El artículo declara la licencia EUPL-1.2 para CY-Bench; las [estadísticas agrícolas europeas de JRC](https://github.com/WUR-AI/AgML-CY-Bench/blob/main/data_preparation/crop_statistics_EU/README.md) se distribuyen bajo CC BY 4.0. La atribución de las fuentes originales debe conservarse.

La muestra elegida contiene **maíz de España a escala de región administrativa NUTS**. No contiene observaciones de parcela ni imágenes Sentinel-2 sin agregar. Los indicadores NDVI proceden de MODIS/VIIRS y el fPAR de JRC. En el [diccionario NDVI oficial](https://github.com/WUR-AI/AgML-CY-Bench/blob/main/data_preparation/global_ndvi_MOD09CMG/README.md) los valores 50–250 se convierten con `(x-50)/200`; fPAR se expresa como porcentaje en su [diccionario oficial](https://github.com/WUR-AI/AgML-CY-Bench/blob/main/data_preparation/global_fpar_500m/README.md). La muestra sirve para comprobar predicción regional con observaciones reales, pero no valida predicción a nivel de parcela/subparcela ni transferencia a Perú.

## Adquisición y transformación reproducibles

Desde la raíz del repositorio:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
.venv\Scripts\python -m app.ml.cybench download --directory data/cybench_sample/maize/ES
.venv\Scripts\python -m app.ml.cybench convert --directory data/cybench_sample/maize/ES --output data/cybench_maize_es.csv
```

`download` recupera ocho CSV públicos desde el commit fijado, comprueba el tamaño esperado y guarda SHA-256 por archivo. `convert` produce un CSV y `data/cybench_maize_es.provenance.json` con hashes, selección, transformaciones y límites. `data/` está excluido de Git.

Cada fila de salida representa una **región y año de cosecha**. El objetivo es rendimiento observado en t/ha. La transformación une calendario de cultivo, centroide, suelo, meteorología, humedad del suelo, NDVI y fPAR por `adm_id` y año. Divide el intervalo desde inicio de temporada (`sos`) hasta mitad de temporada (`sos + (eos-sos)/2`) en cuatro pasos temporales. Suma precipitación, radiación y evapotranspiración por paso y promedia las demás variables. Descarta observaciones posteriores al punto de predicción, rendimientos nulos y registros sin una modalidad requerida. Excluye `production` y `harvest_area` de las variables predictoras porque determinan directamente el rendimiento. El campo `spatial_unit=administrative_region` se conserva como metadato y no entra al modelo.

La ejecución local del 4 de noviembre de 2026 produjo **864 filas, 50 regiones, 18 campañas (2003–2020) y 53 características numéricas**. De 1.146 etiquetas originales, 282 quedaron fuera por fecha, calendario o rendimiento inválido; ninguna de las 864 filas alineadas perdió una modalidad completa. Un único valor temporal de NDVI quedó ausente y el preprocesador del modelo lo imputó usando únicamente entrenamiento. Estos conteos corresponden a la revisión fijada de la muestra, no al CY-Bench completo.

El archivo de salida mide aproximadamente 0,73 MB y puede importarse por la interfaz en **Datasets**. Declare la fuente `CY-Bench maize ES (WUR-AI official sample)`, la licencia y el commit/hash del manifiesto como procedencia. La interfaz conserva estos datos como declaraciones del importador. Los scripts `backend/tests/run_cybench_docker.py` y `backend/tests/run_cybench_e2e.py` reproducen importación, cinco entrenamientos y cinco inferencias a través de la API; el segundo crea y desactiva automáticamente una cuenta temporal.

## Evaluación ejecutada en Docker

Los cinco modelos usaron semilla 42, el mismo CSV y partición temporal: entrenamiento 2003–2013 (**532 filas**), validación 2014–2016 (**148**) y prueba independiente 2017–2020 (**184**). Las redes se entrenaron durante cinco épocas en CPU. Métricas del conjunto de prueba, redondeadas; valores exactos en `data/cybench_temporal_results.json` del workspace:

| Modelo | MAE (t/ha) | RMSE (t/ha) | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 1,626 | 2,098 | 0,673 | 17,54 % |
| XGBoost | 1,617 | 2,056 | 0,685 | 17,62 % |
| LSTM | 2,025 | 2,709 | 0,454 | 28,13 % |
| CNN-LSTM | 3,225 | 3,615 | 0,027 | 39,81 % |
| CNN-Transformer | 3,191 | 3,541 | 0,067 | 44,92 % |

La prueba temporal Docker registró el dataset `#7`, los modelos `#4–#8`, una predicción por modelo y un informe científico. La partición temporal evalúa años posteriores en las mismas regiones.

Una segunda prueba en Docker separó por identificador de región NUTS: **30 regiones** para entrenamiento (520 observaciones), **10** para validación (173) y **10** para prueba (171), sin regiones compartidas entre particiones. Usó semilla 42, el mismo CSV y cinco épocas para redes. Valores exactos en `data/cybench_parcel_results.json`:

| Modelo | MAE (t/ha) | RMSE (t/ha) | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 1,363 | 1,826 | 0,436 | 13,20 % |
| XGBoost | 1,428 | 1,894 | 0,393 | 13,88 % |
| LSTM | 1,846 | 2,836 | -0,360 | 26,21 % |
| CNN-LSTM | 2,471 | 2,760 | -0,288 | 27,60 % |
| CNN-Transformer | 2,965 | 3,266 | -0,804 | 29,46 % |

Esta segunda prueba registró los modelos `#9–#13`, cinco predicciones y un informe. La opción de API se llama `validation=parcel`, pero en este CSV `parcel_id` contiene el identificador de **región**; por tanto, la separación es espacial por región, no por parcela. Las métricas de las dos tablas usan particiones de prueba distintas y no se deben comparar como si fueran el mismo conjunto. El resultado espacial es una sola partición aleatoria; no estima la variación entre particiones ni demuestra transferencia a otro país.

Estas cifras son **resultados de estas ejecuciones**, no resultados publicados por los autores de CY-Bench. No se optimizaron hiperparámetros; cinco épocas no aseguran convergencia de las redes. Tampoco se usan imágenes espaciales crudas: la CNN opera sobre secuencias de indicadores agregados. La muestra completa, una evaluación de campo y la validación peruana siguen pendientes. Las predicciones del sistema advierten expresamente sobre la escala regional, la ausencia de validación en Perú, categorías desconocidas y coordenadas fuera del rango observado.

Posteriormente se añadieron [Friedman, Wilcoxon-Holm e intervalos bootstrap por región](STATISTICS.md) recalculando las predicciones de los modelos en estos dos conjuntos de prueba. Los intervalos describen incertidumbre de la muestra de prueba y no sustituyen la validación externa.
