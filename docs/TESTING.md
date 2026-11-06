# Pruebas y verificaciones

Esta guía separa lo que debe comprobarse de lo que efectivamente se ejecutó. Los comandos, resultados y dependencias observados se consignan en [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md); no inferir una prueba aprobada por la presencia de código.

## Comandos de desarrollo

```powershell
# Desde la raíz, con dependencias instaladas en el entorno Python:
$env:PYTHONPATH = 'backend'
python -m pytest backend/tests/test_api.py backend/tests/test_ml.py backend/tests/test_yieldsat.py backend/tests/test_cybench.py -q --disable-warnings

# Frontend:
cd frontend
npm ci
npm run build
```

Para integración con PostgreSQL/PostGIS:

```powershell
docker compose up --build -d
docker compose ps
docker compose exec backend alembic current
curl.exe http://localhost:8000/health
```

El smoke `backend/tests/smoke_docker.py` necesita una cuenta administradora de prueba ya creada y recibe su contraseña por stdin. Desde PowerShell, con el entorno virtual local preparado y el sistema levantado, puede ejecutarse sin poner el secreto en el historial:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
.venv\Scripts\python -c "import getpass,subprocess,sys; subprocess.run([sys.executable,'backend/tests/smoke_docker.py','smoke@example.test'], input=getpass.getpass('Contraseña de prueba:')+'\n', text=True, check=True)"
```

Este smoke crea un dataset sintético, entrena XGBoost, predice, descarga tres formatos de informe y verifica auditoría y revocación de sesión. Deja registros de demostración en la base; use una instalación de prueba.

La suite combinada más reciente terminó con **14 aprobadas y 4 advertencias en 38,50 s**. Las pruebas de YieldSAT usan un NetCDF sintético; las de CY-Bench usan CSV de prueba para comprobar el corte temporal y las variables excluidas. La evaluación adicional **sí usó observaciones reales** de la muestra oficial: `backend/tests/run_cybench_docker.py` importó 864 filas, entrenó cinco modelos en Docker y ejecutó cinco inferencias con prueba temporal. `backend/tests/run_cybench_e2e.py` repitió los cinco modelos con 10 regiones NUTS completas reservadas para prueba. Se descargó un informe científico en cada ejecución y se probó la [comparación estadística](STATISTICS.md) por API para ambas particiones. [Métricas y límites](CYBENCH.md). Las pruebas no evalúan compatibilidad con un archivo YieldSAT autorizado ni validez en Perú.

## Matriz mínima de aceptación

| Área | Comprobación útil |
| --- | --- |
| Migración | Base vacía → `alembic upgrade head` → tablas esperadas; repetir sin cambios. |
| Autenticación | Login correcto/incorrecto, expiración, salida, CSRF ausente y rol insuficiente. |
| Datos | CSV válido, campos faltantes, codificación, tamaño, valores no finitos y duplicados; demo marcada `synthetic`. |
| Modelos | RF y XGBoost con particiones agrupadas disjuntas; cinco modelos con secuencias reales o fixtures adecuados; inferencia con mismo esquema guardado. |
| Métricas | MAE/RMSE/R² calculados en prueba independiente; MAPE omitido cuando corresponda. |
| Persistencia | Entrenamiento, versión y predicción sobreviven reinicio; artefacto recuperable. |
| Reportes | PDF, DOCX y XLSX descargables y legibles, contenido coherente con la base. |
| Auditoría | Eventos de login, fallo, importación, entrenamiento, predicción, navegación y descarga. |
| Frontend | Build, login, navegación, estados vacíos y errores, importación, experimento, inferencia y reportes por API. |
| Despliegue | Servicios sanos, migración, primera cuenta interactiva y flujo de extremo a extremo. |

La evaluación científica con YieldSAT exige archivos autorizados y capacidad de cómputo adecuada. Un test con datos sintéticos verifica que el software recorre el flujo; no valida capacidad de predicción agronómica ni transferencia a Perú.
