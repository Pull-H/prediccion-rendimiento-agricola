# Fuentes de datos y diccionario de variables

## Estado de las fuentes

**Fuente ejecutable actual:** la [muestra oficial CY-Bench de WUR-AI](https://github.com/WUR-AI/sample_data) permite descargar datos reales de maíz español sin una solicitud de acceso. Se descargaron ocho CSV, se alinearon 864 observaciones regionales de 2003–2020 y se entrenaron los cinco modelos. El [procedimiento, atribución, métricas y límites](CYBENCH.md) documentan esta elección. La licencia declarada para CY-Bench es EUPL-1.2 y las estadísticas de rendimiento JRC son CC BY 4.0; la atribución sigue siendo necesaria.

**Fuente inicial para escala de parcela/subparcela:** [YieldSAT](https://yieldsat.github.io/) enlaza un [formulario de solicitud de acceso](https://forms.office.com/e/VzgL85A3x8); el código de este repositorio no descarga ni redistribuye YieldSAT automáticamente. Al 4 de noviembre de 2026, la página, los [notebooks oficiales](https://yieldsat.github.io/notebooks/yieldsat-overview-notebook.html) y el [repositorio público](https://github.com/yieldsat/yieldsat.github.io) consultados no mostraban una licencia explícita del **dataset**. Solicite acceso y conserve los términos recibidos antes de importar, compartir o publicar sus archivos. La ausencia de un archivo `LICENSE` en el repositorio web no implica que el dataset sea de dominio público.

**Actualización del 6 de noviembre:** el usuario obtuvo un enlace autorizado de YieldSAT y ejecutó la preparación en Colab. Se recibió `inspection.json` de Alemania: 609.645 píxeles, 24 pasos y 120 canales. Esto confirma el esquema, pero no un entrenamiento ni la calidad del archivo completo. La [evaluación de esa inspección y la auditoría siguiente](YIELDSAT_INSPECTION.md) documentan los faltantes de la muestra inicial y la ausencia de unidades en el objetivo. El enlace privado no se incorpora a los notebooks ni a los documentos.

También se recibió la auditoría ampliada: 188 registros de trigo y 111 de colza en seis granjas y siete campañas. El usuario eligió trigo y entregó `yieldsat_wheat_240d_export.zip`. El CSV real verificado contiene **184 registros elegibles**, después de excluir cuatro con siembra deducida de la cosecha. El [perfil de trigo](YIELDSAT_WHEAT_PILOT.md) documenta el SHA-256, las variables, las exclusiones y la evaluación temporal de 132/15/37 registros. Incluye satélite y suelo; clima e imágenes espaciales siguen pendientes. La prueba está muy concentrada en una granja y no demuestra transferencia espacial.

YieldSAT reúne 2.173 parcelas de Argentina, Brasil, Uruguay y Alemania, de los cultivos maíz, colza, soja y trigo, en campañas de 2016 a 2024. El sitio informa más de 12,2 millones de muestras de rendimiento a 10 m y más de 113.000 imágenes Sentinel-2. **No incluye Perú**; toda transferencia a Perú requiere calibración y evaluación local antes de presentarse como validada. [Fuente: proyecto YieldSAT](https://yieldsat.github.io/).

## Estructura publicada por los autores

El [notebook de exploración](https://yieldsat.github.io/notebooks/yieldsat-overview-notebook.html) describe dos distribuciones:

| Distribución | Organización | Uso |
| --- | --- | --- |
| `preprocessed-24-ts/` | Un NetCDF por país, `merge_s2-soil-dem-weather-coords.nc` | Entrenamiento por muestra/píxel con 24 pasos de tiempo. |
| `original-preprocessed/` | País → parcela → archivos por modalidad | Procesamiento propio de imágenes, máscaras y rasters. |

La [guía de ML](https://yieldsat.github.io/notebooks/yieldsat-ml-tutorial.html) usa una ruta con subcarpeta `merged/`, mientras el notebook de exploración muestra el archivo directamente dentro del país. El importador debe localizar ambas variantes y validar la estructura efectiva; no debe asumir una de ellas sin comprobar los archivos autorizados.

El ejemplo alemán del notebook muestra dimensiones `index=609645`, `time_step=24`, `band=120`, con `sample(index,time_step,band)` y `target(index)` en `float32`. Solo `sample` sin comprimir ocupa aproximadamente 7,02 GB (6,54 GiB) en memoria (`609645×24×120×4` bytes); el procesamiento debe abrir el NetCDF de forma diferida y trabajar por lotes. Estas dimensiones son del ejemplo alemán, no una garantía para otros países o versiones. [Fuente: notebook oficial](https://yieldsat.github.io/notebooks/yieldsat-overview-notebook.html).

### Diccionario mínimo observado en el formato preprocesado

| Campo | Dimensión | Interpretación y tratamiento |
| --- | --- | --- |
| `index` | muestra | Identificador de la muestra/píxel. |
| `time_step` | 24 pasos en el ejemplo | Orden de la secuencia estandarizada. Las fechas reales están en `times`. |
| `band` | característica | Nombres de bandas/variables, inspeccionables en cada archivo. |
| `sample` | `index,time_step,band` | Entradas multimodales ya alineadas. |
| `target` | `index` | Rendimiento observado, en t/ha según el [tutorial](https://yieldsat.github.io/notebooks/yieldsat-ml-tutorial.html). |
| `times` | `index,time_step` | Fechas de adquisición o referencia de cada paso. |
| `country`, `crop`, `year` | `index` | País, cultivo y campaña; los códigos se decodifican desde atributos del propio NetCDF. |
| `farm_identifier`, `field_shared_name` | `index` | Grupos de granja y parcela esenciales para particiones sin fuga. |
| `row`, `col` | `index` | Posición para reconstrucción del mapa de rendimiento. |
| `seeding_date`, `harvesting_date`, `seeding_date_type` | `index` | Información fenológica y procedencia de la fecha de siembra. |
| `stats-mean`, `stats-min`, `stats-max`, `stats-std` | `band` | Resúmenes incluidos en el archivo; **no** utilizarlos para normalizar antes de separar entrenamiento y prueba, pues podrían incorporar información de prueba. |

La lista exacta de canales debe venir de `ds.band.values`; el ejemplo incluye reflectancias Sentinel-2, propiedades de suelo por profundidad, topografía, meteorología y coordenadas. En el producto por parcela, los autores muestran `s2_images/`, `scl_masks/`, `weather/`, `soil/`, `dem/`, `yield_masks/` y un archivo de metadatos. El esquema efectivo se valida en cada importación. [Fuente: notebook oficial](https://yieldsat.github.io/notebooks/yieldsat-overview-notebook.html).

## Procedimiento de adquisición y control

1. Solicitar acceso mediante el [formulario oficial](https://forms.office.com/e/VzgL85A3x8), revisar la autorización y guardar su texto fuera del repositorio.
2. Descargar por el mecanismo facilitado por los titulares. Registrar origen, fecha, versión, país, tamaño y SHA-256 de cada archivo.
3. Montar el directorio autorizado en una ruta local configurada por el operador. No copiar los datos a Git ni a un volumen público.
4. Validar existencia y lectura de NetCDF, coordenadas y variables requeridas, finitud del objetivo, unidad, rango razonable, atributos de códigos, duplicados, fechas y agrupación por parcela.
5. Registrar errores, recuentos, transformaciones y procedencia; conservar los originales de forma inmutable y generar derivados versionados.

La variante `original-preprocessed/` permite estudiar las máscaras de nubes SCL y el raster de rendimiento. La ruta `preprocessed-24-ts/` facilita el primer entrenamiento, pero su fusión y remuestreo ya están hechos por los autores: no puede describirse como procesamiento propio. [Fuente: YieldSAT](https://yieldsat.github.io/).

### Adaptador local NetCDF → CSV

`backend/app/ml/yieldsat.py` y la CLI `python -m app.ml` inspeccionan y convierten **archivos NetCDF locales del formato preprocesado**. El adaptador reconoce tanto la subcarpeta `merged/` como el archivo `merge_*.nc` ubicado directamente bajo el país. No descarga YieldSAT, no procesa la distribución flexible por rasters y no registra el CSV convertido automáticamente en la base de datos. La verificación usó un fixture NetCDF sintético, no archivos distribuidos por YieldSAT.

Desde PowerShell en la raíz, después de obtener acceso legítimo y colocar los archivos fuera de Git:

```powershell
python -m pip install -r backend/yieldsat-requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
python -m app.ml discover --source 'D:\YieldSAT\preprocessed-24-ts'
python -m app.ml inspect --source 'D:\YieldSAT\preprocessed-24-ts\Germany\merge_s2-soil-dem-weather-coords.nc'
python -m app.ml convert --source 'D:\YieldSAT\preprocessed-24-ts\Germany\merge_s2-soil-dem-weather-coords.nc' --output '.\data\yieldsat-dev.csv' --max-samples 100 --batch-size 32 --target-unit t/ha --max-output-bytes 20000000
```

El ejemplo de unidad `t/ha` debe confirmarse con metadatos o términos del archivo recibido; si declara `kg/ha`, el adaptador convierte a t/ha. Si la unidad falta, `--target-unit` es obligatorio; si contradice la unidad declarada, falla. `inspect` devuelve dimensiones, variables, mapeo de bandas, unidad y advertencias sin materializar el cubo completo. `convert` exige `--max-samples`, lee bloques por `index`, controla el límite de bytes antes y durante la escritura, y crea `yieldsat-dev.provenance.json` con SHA-256, selección, bandas, unidades y limitaciones. El rango es contiguo (`--start` opcional): una muestra pequeña sirve para desarrollo y puede no contener parcelas o regiones suficientes para validar modelos. Las fechas por muestra de `times` no se exportan al CSV; se conserva el orden de `time_step`.

El CSV local se puede cargar por la interfaz **solo si no supera 20 MB**; `--max-output-bytes 20000000` acota esta conversión de ejemplo. Una conversión mayor puede servir para análisis local, pero la API no tiene importación directa NetCDF ni registro automático del CSV generado. La inclusión de campos de un archivo real, sus categorías y su licencia todavía debe comprobarse con datos autorizados.

### Diccionario del CSV que admite esta aplicación

El importador web acepta CSV tabular UTF-8 de hasta 20 MB. La CLI local anterior abre NetCDF preprocesado y produce ese esquema CSV, pero no sustituye el paso de registro en la aplicación. Sus campos obligatorios son:

En el formulario de importación, `source_name`, `license_name` y `provenance_note` permiten conservar la fuente, licencia y procedencia declaradas. Son campos opcionales aportados por el usuario y **no se verifican automáticamente**. El informe científico muestra esas declaraciones junto con las variables y resultados persistidos. Antes de publicar un dataset o reporte, el operador debe contrastar licencia y procedencia con los documentos recibidos de su titular.

| Campo | Tipo | Uso |
| --- | --- | --- |
| `parcel_id` | Texto no vacío | Identificador del grupo espacial; en CY-Bench contiene un código NUTS, no una parcela. |
| `spatial_unit` | Texto opcional: `parcel`, `administrative_region` o `pixel` | Declara la unidad espacial; se valida y se excluye de las variables predictoras. |
| `crop` | Texto no vacío | Cultivo y categoría de modelo. |
| `region` | Texto no vacío | Región y categoría de modelo. |
| `campaign` | Texto no vacío | Campaña y categoría de modelo. |
| `yield_t_ha` | Número finito ≥ 0 | Objetivo observado. |
| Otra columna numérica | Número, admite faltantes parciales | Característica de entrenamiento; debe tener alguna observación. |

`country` es metadato textual opcional y `latitude`/`longitude` son coordenadas numéricas opcionales. Para modelos temporales, los nombres `<variable>_t0`, `<variable>_t1`, etc. deben formar pasos consecutivos comunes a todas las variables. El importador no verifica unidad física de cada característica adicional; el proveedor debe documentarla. La columna `synthetic_demo` identifica los registros generados localmente.

## Fuentes complementarias, solo cuando falte una modalidad

YieldSAT ya integra datos ERA5-Land, SoilGrids y SRTM. Evite obtenerlos de nuevo salvo para extender cobertura, periodo o variables y documente la diferencia de resolución y licencia.

| Fuente | Uso potencial | Acceso y términos |
| --- | --- | --- |
| [Copernicus Data Space STAC](https://documentation.dataspace.copernicus.eu/APIs/STAC.html) | Buscar Sentinel-2 L2A adicional. | La colección expone metadatos y condiciones; verificar los términos del producto concreto. |
| [ERA5-Land en CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | Meteorología de campañas nuevas. | Requiere cuenta y aceptación de términos; la ficha indica licencia CC BY. |
| [SoilGrids, ISRIC](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_02.html) | Propiedades del suelo fuera de YieldSAT. | WCS/WMS/WebDAV; [ISRIC indica CC BY 4.0](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_04.html). |
| [NASA Earthdata SRTM](https://www.earthdata.nasa.gov/data/instruments/srtm) | Elevación y derivadas. | Revisar acceso y términos del producto elegido. |

## Datos propios y demostración

Los registros peruanos deben incluir cultivo, campaña, geometría o localización con CRS, observaciones disponibles y rendimiento medido con unidad inequívoca. Una importación que carezca de variable objetivo puede servir para inferencia si el modelo admite esas características, pero no para evaluación supervisada. Etiquetar cada dataset como `real` o `synthetic` y mantener ese origen hasta predicciones, métricas e informes. Las métricas de demostración no son resultados científicos.

El documento científico principal está en [`tema (2).docx`](<../tema (2).docx>). Define la integración de Sentinel-2, clima, índices de vegetación y suelo, con predicción en t/ha, validación por parcelas y campañas y comparación de cinco modelos. Las características efectivamente disponibles en un dataset importado deben comprobarse antes de entrenar; el documento temático no sustituye los metadatos de los archivos.
