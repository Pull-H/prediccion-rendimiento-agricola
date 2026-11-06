# Inspección recibida de YieldSAT — 6 de noviembre de 2026

El archivo `inspection.json` fue generado por el usuario en Colab sobre el NetCDF de Alemania. Confirma 609.645 muestras por píxel, 24 pasos temporales, 120 canales y las variables de cultivo, año, granja, parcela, coordenadas de imagen y fechas necesarias para continuar la integración. Las muestras por píxel no equivalen a parcelas independientes.

## Hallazgos y límites

| Evidencia recibida | Interpretación y acción |
| --- | --- |
| Doce bandas Sentinel-2: B01–B09, B11, B12 y B8A | Están los canales necesarios para calcular índices. La escala radiométrica y los posibles desplazamientos deben comprobarse antes de calcular EVI. |
| `temp_mean`, `temp_max`, `temp_min`, `total_prec` | Hay variables meteorológicas; sus unidades y agregaciones deben documentarse. |
| `clay_*`, `sand_*`, `silt_*`, `phh2o_*`, `soc_*` y otros canales de suelo | El esquema incluye textura, pH y carbono orgánico por profundidad. Carbono orgánico no equivale directamente a materia orgánica. |
| `target_attributes: {}` | El archivo inspeccionado no declara la unidad del objetivo. El [tutorial oficial](https://yieldsat.github.io/notebooks/yieldsat-ml-tutorial.html) documenta t/ha, pero esta atribución externa debe quedar explícita antes de convertir; el adaptador exige una unidad declarada por el archivo o proporcionada por el operador. |
| `probe_shape: [16, 2, 8]`, `probe_missing_fraction: 1.0` | Solo se inspeccionaron los primeros 16 píxeles, dos pasos y ocho bandas. El resultado no describe la calidad del cubo completo. La [documentación oficial](https://yieldsat.github.io/notebooks/yieldsat-ml-tutorial.html) indica enmascaramiento fuera de la temporada; esa es una explicación posible que todavía debe verificarse. |
| `target_probe_finite: true` | Los primeros 16 objetivos son finitos; no implica que todos los objetivos sean válidos. |
| Cultivos, años y parcelas presentes, sin categorías en el informe inicial | Deben recuperarse sus mapas de códigos. La distribución oficial guarda mapas directamente como atributos numéricos; se añadió este formato al adaptador. |

Este informe no contiene conteos reales de parcelas/cultivos/campañas, distribución completa de etiquetas ni cobertura temporal suficiente. No se debe entrenar aún con una conversión contigua del principio del archivo, ni interpretar los valores ausentes como un dataset inutilizable.

## Continuar en la sesión existente de Colab

Abra `notebooks/YieldSAT_auditoria_colab.py`, copie su contenido completo y péguelo en **una nueva celda al final del mismo cuaderno de Colab** que extrajo el NetCDF. Ejecute únicamente esa celda. El script usa `netcdf_path` si existe y, en caso contrario, busca `/content/yieldsat_work/Germany.nc`.

No solicita el enlace de SharePoint ni descarga ZIP alguno. Recorre las etiquetas y los identificadores en lotes, y revisa todos los canales y todos los pasos temporales de hasta 256 píxeles distribuidos en ocho ventanas del archivo. No carga el cubo completo. Al terminar descarga solo `inspection_quality.json`, que incluye:

- Conteos de píxeles por cultivo, año y país; grupos por granja y registro de parcela.
- Mapas de códigos y atributos de fechas, unidades y escala.
- Revisión de objetivos finitos, negativos y su rango sobre todas las etiquetas.
- Cobertura de datos por paso temporal y por canal, con rangos en la muestra diagnóstica.

Si Colab ya eliminó el entorno, el script se detendrá: habrá que preparar nuevamente los archivos **en Colab**. El notebook de preparación del repositorio también incorpora esta auditoría para futuras ejecuciones.

La misma auditoría está disponible para archivos locales autorizados mediante `python -m app.ml audit --source RUTA.nc --probe-rows 256`. Esto no significa que se haya descargado el dataset a este equipo.

## Antes de preparar el experimento

La auditoría permitirá elegir un cultivo real y los grupos de evaluación, decidir cómo tratar pasos enmascarados y documentar unidades y escala. Luego se implementará una selección de parcelas completas, un horizonte temporal sin datos futuros y una exportación compatible con el sistema. La versión CSV actual conserva todas las bandas y posiciones temporales: puede producir columnas totalmente vacías y miles de variables, por lo que no es todavía el perfil experimental apropiado para este archivo.

La auditoría no valida independencia geográfica, no reconstruye imágenes para las CNN, no calcula índices sin verificar escalas y no entrena modelos. Esas tareas siguen pendientes. Alemania sirve para comprobar el flujo; si se mantiene maíz como cultivo objetivo, se deberá inspeccionar un país que lo contenga antes de seleccionar la cohorte científica.

## Auditoría ampliada recibida y cultivo elegido

Se recibió `inspection_quality.json`: 188 registros de trigo y 111 de colza, seis granjas, campañas 2016–2022 y 609.645 etiquetas finitas y no negativas. Los 256 píxeles de la muestra tienen alguna observación óptica válida; el bloque inicial vacío no implicaba ausencia de imágenes durante toda la temporada. También se detectaron rangos meteorológicos sin interpretación física confirmada y `twi` vacío en la muestra.

El usuario eligió trigo de Alemania. Se preparó un [perfil de exportación por parcela con corte temporal, controles de fechas y procedencia](YIELDSAT_WHEAT_PILOT.md). Se mantiene la distinción entre la auditoría real recibida, las pruebas del código con fixtures y un entrenamiento real, todavía pendiente.
