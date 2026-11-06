# Pruebas estadísticas de los modelos

## Ubicación en el documento de investigación

[`tema (2).docx`](<../tema (2).docx>) menciona la evaluación estadística en **F9. Evaluación estadística** y detalla las pruebas en **6.5. Pruebas estadísticas**: Friedman para varios modelos sobre bloques independientes, Wilcoxon pareado, corrección de Holm y bootstrap para intervalos de confianza. La última frase de 6.5 advierte que Friedman no debe aplicarse a cinco métricas agregadas. También enumera McNemar y DeLong, que corresponden a clasificación/ROC, y Diebold-Mariano, que solo procede con un diseño de pronóstico temporal apropiado.

## Implementación

El módulo independiente **Pruebas estadísticas** (`/estadisticas`) y `GET /api/statistics/compare?dataset_id=9&validation=temporal&seed=42` muestran el análisis. La selección inicial prioriza YieldSAT. Se requiere que los cinco modelos estén entrenados con el mismo CSV, semilla y partición; el servicio comprueba los hashes y el esquema guardados con cada artefacto. El análisis se recalcula sobre las mismas observaciones de prueba de cada modelo.

La página explica los métodos y sus supuestos, muestra los tamaños de entrenamiento/validación/prueba, el hash del CSV, los identificadores de modelo/experimento y las épocas realmente usadas. Presenta Friedman, grados de libertad y W de Kendall, las diez parejas Wilcoxon con sus estadísticos, p originales y ajustadas por Holm, y métricas con intervalos bootstrap del MAE. Permite recalcular y descargar el resultado completo en JSON. Los protocolos disponibles se derivan de experimentos reales completos; no se inventan resultados para combinaciones sin los cinco modelos.

Cada bloque es un `parcel_id`. En CY-Bench ese campo contiene un **código de región NUTS**, no una parcela. Dentro de cada bloque se calcula MAE para los cinco modelos. Friedman usa esas pérdidas por bloque, Wilcoxon compara cada pareja de modelos sobre los mismos bloques, y Holm ajusta las diez comparaciones. Un bootstrap de 2.000 remuestreos de regiones completas proporciona intervalos percentiles del 95 % para el MAE global, ponderado por observación. El código está en `backend/app/ml/statistics.py`; las respuestas completas de esta ejecución se guardaron en `data/cybench_statistics_temporal.json` y `data/cybench_statistics_parcel.json` del workspace (archivos excluidos de Git).

| Prueba de CY-Bench | Regiones de prueba | Friedman χ² | p | W de Kendall | Random Forest MAE, IC 95 % | XGBoost MAE, IC 95 % |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Años 2017–2020 | 50 | 74,352 | 2,73 × 10⁻¹⁵ | 0,372 | 1,626 [1,329; 1,917] | 1,617 [1,337; 1,881] |
| Regiones reservadas | 10 | 18,880 | 0,000830 | 0,472 | 1,363 [1,024; 1,800] | 1,428 [1,013; 2,057] |

Las unidades de MAE e intervalos son **t/ha**. Tras Holm, la comparación **Random Forest frente a XGBoost** no resultó significativa en la partición temporal (`p=0,789`) ni en la espacial (`p=1,000`). En la temporal sí hubo diferencias ajustadas entre Random Forest, XGBoost y LSTM frente a CNN-LSTM y CNN-Transformer; en la espacial hubo tres parejas ajustadas con `p<0,05`: Random Forest frente a CNN-LSTM y CNN-Transformer, y XGBoost frente a CNN-Transformer. Los valores exactos de las diez parejas están en los JSON.

Estos análisis son **exploratorios**: cada tipo de validación usa una sola partición, los bloques cercanos pueden estar correlacionados, y las redes se entrenaron solo cinco épocas sin ajuste de hiperparámetros. Un `p` pequeño describe diferencias de error en esta muestra y configuración; no prueba superioridad universal, causalidad ni validez en Perú. McNemar y DeLong no se ejecutan porque el objetivo es rendimiento continuo y no existe una etiqueta de clasificación/curva ROC. Diebold-Mariano requiere un protocolo temporal de pronóstico y dependencia serial definido previamente; aquí no se ha aplicado.

## YieldSAT: piloto temporal real

El dataset #9 incorpora 184 registros de trigo de Alemania, con 37 en prueba (2021–2022). Se ejecutaron Friedman, diez Wilcoxon con Holm y bootstrap de 2.000 remuestreos usando los cinco modelos #14–#18, semilla 42 y 30 épocas para las redes. [Métricas, valores p y límites](YIELDSAT_WHEAT_PILOT.md#ejecución-real-en-la-aplicación).

La prueba contiene 36 registros de farm2 y uno de farm6. Los bloques son identificadores del archivo, cuya independencia física no se ha demostrado. El bootstrap por identificador no contempla la dependencia dentro de granja: sus intervalos y los valores p son exploratorios. La respuesta API incluye `independence_verified: false` y `test_region_counts`; la interfaz muestra esta limitación. Los resultados completos están en `data/yieldsat_wheat_240d/live_temporal_results.json`. Para consultarlos en la plataforma, seleccione YieldSAT en Pruebas estadísticas o use `/api/statistics/compare?dataset_id=9&validation=temporal&seed=42`.

## Supuestos y referencias metodológicas

- **Friedman:** rangos pareados, bloques independientes y estadístico corregido por empates. Se conserva el valor p de la aproximación χ²; la documentación de SciPy advierte sobre su fiabilidad con pocos modelos. Este piloto compara cinco y no presenta una p exacta o por permutaciones. [SciPy: friedmanchisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html).
- **Wilcoxon:** contraste bilateral de diferencias pareadas; su hipótesis nula implica simetría alrededor de cero. Se usa `zero_method="wilcox"`, `method="auto"`. La independencia entre bloques y la simetría no se declaran demostradas. [SciPy: wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
- **Holm:** ajuste de la familia completa de diez comparaciones, con α=0,05. No modifica los errores ni constituye una prueba de precisión absoluta. [R stats: p.adjust](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).
- **Bootstrap percentil:** implementación local que remuestrea bloques enteros y usa percentiles 2,5/97,5 del MAE ponderado por observación. No utiliza el método BCa predeterminado de SciPy y no calcula intervalos de rendimiento individual. [SciPy: descripción del método percentil](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

El campo histórico `independent_blocks` se conserva por compatibilidad, pero solo cuenta identificadores; el módulo usa el nuevo `block_count` y advierte que la independencia no está verificada. La respuesta incluye `alpha`, `methods`, `training_details`, `model_versions`, fuente y dataset. El campo `statistic` de cada Wilcoxon es el estadístico bilateral devuelto por SciPy; cuando todas las diferencias son numéricamente nulas se informa 0 con p=1.
