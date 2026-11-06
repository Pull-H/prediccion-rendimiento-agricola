# Primer perfil de trigo de YieldSAT

El usuario eligió continuar con trigo de Alemania tras recibir `inspection_quality.json`. Este documento fija el perfil antes de conocer resultados de modelos. Es una primera prueba de satélite y suelo; la integración meteorológica y las CNN espaciales siguen pendientes.

## Evidencia de la auditoría recibida

La auditoría recorrió las 609.645 etiquetas y sus identificadores. Encontró 299 registros de parcela de seis granjas: 188 de trigo (306.843 píxeles) y 111 de colza (302.802 píxeles). Todos los objetivos son finitos y no negativos. La inspección de características, en cambio, corresponde solo a 256 píxeles distribuidos en ocho ventanas: todos tienen al menos una observación óptica completa, pero hay numerosos pasos sin imagen. `twi` no tiene valores finitos en esas ventanas. No se ha revisado cada valor del cubo.

Los canales de temperatura alcanzan valores cercanos a 14.600 y no declaran unidades ni agregación en los atributos inspeccionados. No deben describirse ni corregirse automáticamente como °C o K. El primer perfil excluye meteorología, `twi`, pendiente y coordenadas de representación hasta verificar sus convenciones. Las bandas de satélite y las propiedades de suelo se conservan en unidades fuente; no se calcula EVI ni se atribuyen unidades físicas nuevas.

| Campaña | Registros de trigo candidatos |
| --- | ---: |
| 2016 | 27 |
| 2017 | 29 |
| 2018 | 30 |
| 2019 | 49 |
| 2020 | 16 |
| 2021 | 20 |
| 2022 | 17 |

Antes de filtrar por elegibilidad, una separación temporal sería 2016–2019 para entrenamiento (135 registros), 2020 para validación (16) y 2021–2022 para prueba (37). **Estos conteos no son todavía el tamaño del experimento:** faltan exclusiones por fechas, origen de siembra y observaciones previas al corte. La granja 2 concentra 124 de los 188 registros de trigo, por lo que una evaluación por granjas requiere informar ese desequilibrio.

## Perfil de exportación

`backend/app/ml/yieldsat_experiment.py` agrega todos los píxeles de cada registro elegible, en lotes. El objetivo es su rendimiento medio por píxel. La unidad t/ha se proporciona explícitamente con atribución al [tutorial oficial](https://yieldsat.github.io/notebooks/yieldsat-ml-tutorial.html); el hecho de que falte en el archivo original se conserva en la procedencia.

- Se aceptan solo fechas de siembra explícitamente declaradas por el agricultor. El [notebook oficial](https://yieldsat.github.io/notebooks/yieldsat-overview-notebook.html) también documenta siembras calculadas como 290 o 333 días antes de cosecha: esas fechas dependen del futuro y se excluyen del perfil anticipado.
- Se usan imágenes con fecha conocida entre la siembra y el día 240, inclusive, y se exige que la cosecha ocurra después del corte. Las cuatro ventanas de 60 días son intervalos cronológicos, no etapas biológicas confirmadas.
- Cada media espectral lleva un canal de número de observaciones por píxel. Una ventana vacía se representa con cero y contador cero, sin inventar reflectancias, interpolar ni utilizar observaciones futuras.
- Se conservan textura, pH, carbono orgánico y nitrógeno superficiales, además del DEM, en unidades fuente y con indicador de disponibilidad.
- Fechas de siembra/cosecha, códigos de píxel, estadísticas globales del dataset y valores `yield_ground_truth` de atributos no se usan como predictores. Los identificadores se conservan para agrupar la evaluación.

Las exclusiones quedan en el manifiesto. El filtrado por cosecha posterior al corte define una cohorte retrospectiva de ciclos suficientemente largos; debe declararse esta selección. Los registros de parcela no acreditan por sí solos identidad física longitudinal ni independencia espacial. El primer CSV pierde la variación dentro de la parcela, por lo que no sustituye una reconstrucción de parches para CNN espaciales.

## Ejecutar en el mismo Colab

Copie el contenido de `notebooks/YieldSAT_exportar_trigo_colab.py` en una nueva celda del cuaderno donde se mantiene `Germany.nc` y ejecute esa celda. No repita las celdas de descarga. Si Colab eliminó el archivo, el script se detiene en lugar de descargarlo automáticamente.

El script genera y descarga un paquete pequeño `yieldsat_wheat_240d_export.zip` con:

1. `yieldsat_Germany_wheat_240d.csv`, si existen registros elegibles.
2. Su manifiesto `.provenance.json`, con agregación, unidad explícita, fechas de corte, píxeles utilizados y exclusiones.
3. `experiment_plan.json`, con conteos finales y la propuesta de separación temporal.

Si no hay registros elegibles, el paquete contiene el diagnóstico y no un CSV inventado. No se relaja el origen de la siembra de forma automática. Si faltan grupos suficientes, el plan tampoco afirma estar listo para entrenamiento. Este script **exporta datos; todavía no entrena modelos**.

## Paquete real recibido el 6 de noviembre de 2026

Se recibió `yieldsat_wheat_240d_export.zip` (98.493 bytes). Se verificó el SHA-256 del CSV contra el manifiesto: `fa9496d9598896042152d3bd669edd1290384630f0e880d79fdde483aee0e2e1`. El CSV ocupa 308.551 bytes y contiene **184 registros**, 110 variables numéricas y tres categorías de modelo. No presenta faltantes, infinitos ni duplicados de parcela/campaña. Las fechas del manifiesto cumplen siembra + 240 días y corte anterior a cosecha. No se descargó el NetCDF en este equipo.

Se excluyeron los cuatro registros de farm5 porque su siembra se calculó como 290 días antes de cosecha. Los 184 restantes declaran siembra proporcionada por el agricultor. El manifiesto procede de la ejecución remota: verificar el hash del CSV no equivale a auditar de nuevo el NetCDF original.

| Uso | Campañas | Registros |
| --- | --- | ---: |
| Entrenamiento | 2016–2019 | 132 |
| Validación | 2020 | 15 |
| Prueba | 2021–2022 | 37 |

Quedan cinco granjas: farm1 (44), farm2 (124), farm3 (5), farm4 (5) y farm6 (6). **La prueba contiene 36 registros de farm2 y uno de farm6.** Los 37 identificadores no prueban independencia física: los contrastes y el bootstrap por identificador de parcela son exploratorios y pueden subestimar la incertidumbre por dependencia dentro de granja. Solo dos granjas en prueba tampoco permiten una inferencia fiable por granjas. Se necesita otra evaluación con granjas reservadas y mayor cobertura para estudiar transferencia espacial.

Las ventanas ópticas sin observaciones se mantienen con cero y su contador cero: para B02 son 23, 75, 14 y 2 registros, respectivamente. No se interpretan como reflectancia cero observada.

El flujo reproducible de importación, entrenamiento de cinco modelos, inferencias, estadística y reporte es:

```powershell
.\.venv\Scripts\python.exe backend/tests/run_yieldsat_e2e.py data/yieldsat_wheat_240d/yieldsat_Germany_wheat_240d.csv --epochs 30
```

Requiere Compose activo con PyTorch. Crea una cuenta temporal y la desactiva al terminar; los datasets y experimentos permanecen para revisión. Usa semilla 42 y 30 épocas fijas para las tres redes, sin selección de hiperparámetros por el conjunto de prueba. Incluye una referencia constante igual a la media del entrenamiento. Guarda los resultados en `data/yieldsat_wheat_240d/live_temporal_results.json` y el reporte DOCX en el mismo directorio.

Cualquier métrica describe este perfil de satélite y suelo. Siguen pendientes clima con agregación verificada, índices calibrados, parches espaciales y validación en Perú; este piloto no completa todos los requisitos de `tema (2)`.

## Ejecución real en la aplicación

El flujo anterior terminó correctamente en Docker: **dataset #9**, experimentos y modelos **#14–#18**, cinco inferencias no sintéticas, comparación estadística y un DOCX científico de 41.655 bytes. La cuenta temporal se desactivó al terminar. El informe DOCX reúne los registros de la aplicación; el JSON del piloto identifica los resultados de este dataset. El manifiesto también se conserva en el volumen de datos como `dataset_9.provenance.json`.

Resultados en los mismos 37 registros de prueba, semilla 42; las tres redes usaron 30 épocas fijas:

| Modelo | ID | MAE (t/ha) | RMSE (t/ha) | R² |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 14 | 1,201 | 1,647 | −0,787 |
| XGBoost | 15 | 1,335 | 1,789 | −1,108 |
| LSTM | 16 | 1,098 | 1,437 | −0,359 |
| CNN-LSTM | 17 | 0,998 | 1,380 | −0,254 |
| CNN-Transformer | 18 | 1,210 | 1,590 | −0,666 |
| Media del entrenamiento (9,977 t/ha) | — | 1,500 | 1,902 | −1,383 |

CNN-LSTM obtuvo el menor error en esta ejecución. Los cinco modelos mejoran el MAE/RMSE de la referencia que predice la media del entrenamiento, pero **todos tienen R² negativo**: su error cuadrático supera el de una constante igual a la media del conjunto de prueba, calculada retrospectivamente. Esa media de prueba no se usa como predictor disponible en producción. Los resultados no respaldan todavía una buena capacidad predictiva ni una superioridad general de una arquitectura. No se ajustaron parámetros después de observar la prueba.

Friedman por identificador de parcela: χ²=10,3568, p=0,03483, W de Kendall=0,06998. De las diez comparaciones Wilcoxon con Holm, solo CNN-LSTM frente a CNN-Transformer resultó p ajustada <0,05 (0,02026). El IC 95 % exploratorio del MAE de CNN-LSTM, con 2.000 remuestreos de identificadores, fue [0,713; 1,325] t/ha. **Estos valores no corrigen la dependencia entre parcelas de la misma granja** y no constituyen evidencia confirmatoria. La API expone `independence_verified: false` y la distribución de registros por región/granja; la interfaz muestra esa limitación.

En la aplicación, abra Pruebas estadísticas, seleccione el dataset YieldSAT y evaluación Temporal. El endpoint es `/api/statistics/compare?dataset_id=9&validation=temporal&seed=42`. Los modelos se pueden consultar en Registro de modelos e Inferencias con el esquema exacto de sus variables.
