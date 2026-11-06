# Predicción del rendimiento agrícola

Plataforma de investigación para importar datos agrícolas, comparar cinco modelos, consultar predicciones y estudiar sus resultados estadísticos. Implementada con React/TypeScript, FastAPI y PostgreSQL/PostGIS.

El estudio principal utiliza **184 registros de trigo de Alemania, de YieldSAT**, con variables satelitales y de suelo a 240 días de siembra. La integración climática no forma parte de este perfil. Los resultados son exploratorios: todos los modelos obtuvieron R² negativo en prueba y no existe validación en Perú. [Protocolo, métricas y límites](docs/YIELDSAT_WHEAT_PILOT.md).

## Ejecutar con Docker

Requisitos: Git, Docker y Docker Compose. Desde la raíz:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Editar .env: establecer POSTGRES_PASSWORD y JWT_SECRET propios.
# Configurar INSTALL_TORCH=true para utilizar los cinco modelos.
docker compose up --build -d
docker compose ps
docker compose exec backend python -m app.bootstrap --email admin@example.org --name Administrador
```

La última orden solicita una contraseña de al menos 12 caracteres. No existe una contraseña de administrador incluida en el repositorio.

- Aplicación: http://localhost:5173
- API y documentación: http://localhost:8000/docs
- Salud: http://localhost:8000/health
- Detener conservando datos: `docker compose stop`

Sin PyTorch (`INSTALL_TORCH=false`) pueden entrenarse Random Forest y XGBoost. Para LSTM, CNN-LSTM y CNN-Transformer active esa opción y reconstruya la imagen.

**Una instalación nueva comienza sin usuarios, datasets ni modelos entrenados.** Git contiene el código y los materiales del artículo; los registros y modelos de la instalación actual permanecen en los volúmenes Docker. Para trasladarlos, consulte [despliegue y respaldos](docs/DEPLOYMENT.md). YieldSAT requiere acceso autorizado y sus archivos no se redistribuyen aquí.

## Módulos

| Módulo | Función |
| --- | --- |
| Resumen | Indicadores y comparación de modelos del mismo estudio. |
| Datasets | Importación y validación de CSV autorizados. |
| Experimentos y modelos | Entrenamiento, métricas y versiones de cinco algoritmos. |
| Predicciones | Selección de modelo, campaña y parcela con carga automática de variables. |
| Pruebas estadísticas | Friedman, Wilcoxon pareado, ajuste de Holm e intervalos bootstrap. |
| Artículo y figuras | Seis figuras con interpretación y descargas a 600 dpi, y seis tablas. |
| Informes | Exportación de registros en PDF, DOCX y XLSX. |
| Consulta de resultados | Consultas estructuradas sobre registros; no utiliza un LLM. |
| Usuarios y auditoría | Roles, sesiones y trazabilidad de acciones. |

La carga automática recupera las variables del CSV original y verifica su hash. Las consultas sobre registros históricos no constituyen pronósticos de campañas nuevas. Los datos sintéticos se utilizan exclusivamente como fixtures de pruebas aisladas; su creación está deshabilitada en la aplicación habitual.

## Estructura

```text
backend/app/          API, autenticación, modelos y análisis estadístico
backend/alembic/      Migraciones de base de datos
backend/tests/        Pruebas unitarias y recorridos de integración
backend/research/     Generación y comprobación del artículo y las figuras
frontend/             Interfaz React/TypeScript y configuración Nginx
notebooks/            Preparación y exportación autorizada en Colab
workflows/            Control de salud opcional para n8n
docs/                 Arquitectura, operación y metodología
docs/manuscrito/      Artículo, tablas, figuras y catálogo de integridad
```

`tema (2).docx` y `Normas.docx` se conservan como fuentes de los requisitos científicos y editoriales. Los formatos originales de las figuras se conservan porque el módulo los sirve con comprobación de hash.

## Documentación

- [Arquitectura](docs/ARCHITECTURE.md), [API](docs/API.md) y [seguridad](docs/SECURITY.md).
- [Despliegue, volúmenes y respaldos](docs/DEPLOYMENT.md).
- [Datos y condiciones de uso](docs/DATASETS.md), [perfil YieldSAT](docs/YIELDSAT_WHEAT_PILOT.md) y [antecedente CY-Bench](docs/CYBENCH.md).
- [Metodología estadística](docs/STATISTICS.md) y [alcance del sistema](docs/SCOPE_REVIEW.md).
- [Artículo y reproducción de materiales](docs/manuscrito/README.md).
- [Pruebas](docs/TESTING.md), [estado de implementación](IMPLEMENTATION_STATUS.md) y [contenido del repositorio](docs/REPOSITORY.md).

## Desarrollo y verificación

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
$env:PYTHONPATH = 'backend'
.\.venv\Scripts\python.exe -m pytest backend/tests/test_api.py backend/tests/test_statistics.py -q
```

Para compilar la interfaz fuera de Docker, ejecute `npm ci` y `npm run build` desde `frontend/`. Las pruebas que usan NetCDF o redes neuronales necesitan las dependencias adicionales indicadas en [la guía de pruebas](docs/TESTING.md).

No publique `.env`, datos originales, artefactos entrenados ni respaldos de base de datos. Las exclusiones están declaradas en `.gitignore`.

## Reproducibilidad y reutilización

La versión `v1.0.0` está archivada en Zenodo: **[DOI: 10.5281/zenodo.24022432](https://doi.org/10.5281/zenodo.24022432)**. El depósito preserva el código y los materiales públicos de esa versión; los datos individuales y los pesos restringidos no se incluyen.

La versión `v1.0.0` reúne código, resultados agregados, particiones por índice de fila, versiones del entorno y scripts para volver a entrenar o verificar los modelos originales. Consulte [el protocolo reproducible y acceso a datos](reproducibility/README.md). Los datos fuente se solicitan a YieldSAT; el CSV derivado y los pesos originales se conservan en un paquete local pendiente de revisión de sus condiciones de redistribución. El código propio usa [MIT](LICENSE), con [alcance y exclusiones](LICENSE_SCOPE.md). Para citar el software utilice `CITATION.cff`.
