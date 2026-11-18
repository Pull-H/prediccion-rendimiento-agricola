# Predicción del rendimiento agrícola

Plataforma de investigación para importar datos agrícolas, entrenar cinco modelos,
evaluar resultados y consultar predicciones con explicaciones. Utiliza React y
TypeScript, FastAPI y PostgreSQL/PostGIS.

El repositorio distribuye el software, las migraciones, las pruebas y las herramientas
de preparación de datos. Los datasets, modelos entrenados, documentos científicos,
resultados particulares y respaldos se conservan localmente y se excluyen de Git.

## Ejecutar con Docker

Requisitos: Git, Docker y Docker Compose. Desde la raíz:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Establecer POSTGRES_PASSWORD y JWT_SECRET propios en .env.
# Configurar INSTALL_TORCH=true para habilitar las tres redes temporales.
docker compose up --build -d
docker compose ps
docker compose exec backend python -m app.bootstrap --email admin@example.org --name Administrador
```

La última orden solicita una contraseña de al menos 12 caracteres. No se incluye
una cuenta administradora ni una contraseña de producción en el código.

- Aplicación: http://localhost:5173
- Documentación interactiva de la API: http://localhost:8000/docs
- Salud: http://localhost:8000/health
- Detener conservando datos: `docker compose stop`

Una instalación nueva comienza sin usuarios, datasets ni modelos. Los volúmenes
`postgres_data`, `app_data` y `model_artifacts` conservan la información local.
Antes de trasladar una instalación, respaldar los tres volúmenes de forma coherente;
no incluir esos respaldos en Git. `docker compose down -v` elimina los volúmenes.

## Funciones principales

| Módulo | Función |
| --- | --- |
| Datasets | Importación CSV autorizada, metadatos y validación de calidad. |
| Investigación | Exploración, entrenamiento, evaluación, explicabilidad, reportes, inferencia y Optuna. |
| Laboratorio | Historial de experimentos y versiones de modelos. |
| Predicciones | Consulta individual y recuperación de variables históricas. |
| Pruebas estadísticas | Comparación pareada, corrección de Holm y bootstrap por bloques. |
| Informes | Exportación PDF, Word y Excel. |
| Consulta de resultados | Consultas estructuradas sobre registros; no utiliza un LLM. |
| Usuarios y auditoría | Roles, sesiones y trazabilidad de acciones. |

Random Forest y XGBoost funcionan sin PyTorch. LSTM, CNN-LSTM y CNN-Transformer
requieren PyTorch y secuencias temporales compatibles; sus convoluciones son 1D.
Un CSV tabular no se convierte automáticamente en imágenes satelitales.

El módulo opcional **Artículo y figuras** utiliza materiales locales en
`docs/manuscrito/`, montados por Compose como `/app/publication`. Requiere un
`catalogo_publicacion.json` válido y sus archivos asociados. Esos materiales no
se incluyen al clonar; su ausencia no impide entrenar, predecir o generar nuevos
informes en Investigación. La documentación extensa de `docs/` también es local.

## Datos y protocolo de investigación

El CSV de entrada requiere `parcel_id`, `crop`, `region`, `campaign`, `yield_t_ha`
y al menos una variable numérica. La carga web admite UTF-8 y hasta 20 MB.
Las variables temporales deben usar nombres como `ndvi_t0`, `ndvi_t1`, etc.
El objetivo se expresa en t/ha. La unidad de análisis debe declararse correctamente;
una región administrativa no es una parcela.

La separación puede ser por unidad espacial, campaña, región o tiempo. Optuna
realiza CV anidada solo sobre entrenamiento y reserva calibración y prueba.
Los informes distinguen intervalos de confianza de métricas e intervalos de
predicción, muestran la cobertura observada y advierten si no alcanza el objetivo.
Las explicaciones son predictivas y no establecen causalidad. Las consultas
históricas no constituyen pronósticos de campañas nuevas.

Los datos de YieldSAT requieren autorización. También se incluye un adaptador
para un dataset tabular público de Kaggle; su procedencia y licencia deben revisarse
antes de redistribuirlo. Ninguno se descarga automáticamente al iniciar la aplicación.
Los resultados son exploratorios y no acreditan transferencia a Perú. Los datos
sintéticos se limitan a pruebas y su creación está deshabilitada por defecto.

## Desarrollo y verificación

El entorno Docker utiliza Python 3.12. Para desarrollo local, usar una versión
compatible con las dependencias fijadas:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
# Para redes temporales, instalar también backend/ml-requirements.txt.
$env:PYTHONPATH = 'backend'
.\.venv\Scripts\python.exe -m pytest backend/tests/test_api.py backend/tests/test_ml.py backend/tests/test_tuning.py backend/tests/test_research.py backend/tests/test_statistics.py backend/tests/test_robustness.py -q
```

Las pruebas de redes se omiten si PyTorch no está instalado. Los adaptadores
NetCDF requieren `backend/yieldsat-requirements.txt`. La interfaz se compila
con `npm ci` y `npm run build` desde `frontend/`.

Para iniciar una búsqueda local con un CSV ya preparado:

```powershell
$env:PYTHONPATH = 'backend'
.\.venv\Scripts\python.exe -m app.ml.tuning --source data/mi-dataset.csv --output artifacts/mi-busqueda --model XGBoost --validation temporal --trials 12
```

La CLI escribe artefactos locales; la creación de experimentos desde la aplicación
los registra además en su base de datos.

Tras completar búsquedas temporales de Random Forest y XGBoost sobre el mismo CSV,
el módulo de robustez añade, sin reajustar con el periodo de prueba, la referencia
de persistencia (campaña anterior del mismo país), WAPE y MASE, contrastes pareados
con varianza agrupada por país y ajuste de Holm, validación con países no vistos,
recalibración conformal secuencial y una partición cronológica 70/15/15:

```powershell
$env:PYTHONPATH = 'backend'
.\.venv\Scripts\python.exe -m app.ml.robustness --source data/mi-dataset.csv --runs artifacts/rf/<run>.json artifacts/xgb/<run>.json --output artifacts/robustez.json --figures artifacts/figuras
```

## Estructura y mantenimiento

```text
backend/app/          API, modelos, entrenamiento, evaluación y reportes
backend/alembic/      Migraciones de base de datos
backend/tests/        Pruebas y recorridos de integración
backend/research/     Scripts de análisis y reproducción científica
frontend/             Interfaz y configuración de compilación
notebooks/            Herramientas de preparación autorizada de datos
reproducibility/      Instrucciones y dependencias del estudio original
workflows/            Ejemplo opcional de control de salud
```

Se conservan código, pruebas, migraciones, archivos de dependencias y sus lockfiles,
configuración de despliegue, ejemplos `.env.example`, licencia y metadatos de cita.
No se deben añadir `.env`, datos privados, pesos, bases, documentos, resultados de
estudios, cachés, dependencias instaladas o builds. `.gitignore` no excluye archivos
que ya estaban rastreados: comprobar también `git ls-files -ci --exclude-standard`.
La exclusión actual no elimina copias que ya existan en el historial de Git.

## Licencia y cita

El código propio usa [MIT](LICENSE), con el alcance indicado en
[LICENSE_SCOPE.md](LICENSE_SCOPE.md). `CITATION.cff` describe la versión 1.1.0
(DOI [10.5281/zenodo.24352582](https://doi.org/10.5281/zenodo.24352582)), que añade
el módulo de robustez y los resultados agregados del caso de trigo en
`reproducibility/kaggle-trigo/`. La versión 1.0.0
([10.5281/zenodo.24022432](https://doi.org/10.5281/zenodo.24022432)) conserva el
piloto con YieldSAT; el DOI [10.5281/zenodo.24022431](https://doi.org/10.5281/zenodo.24022431)
reúne todas las versiones. Los datos de terceros y los modelos entrenados no se
incluyen en los depósitos.
