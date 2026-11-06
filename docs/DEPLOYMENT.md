# Ejecución local y despliegue

## Docker Compose

Se requiere Docker Engine con el complemento Compose y capacidad para construir imágenes Python y Node. El entorno local arranca PostgreSQL con PostGIS, FastAPI y una aplicación React compilada y servida por Nginx. **No se requiere GPU** para iniciar el sistema. El contenedor backend instala `backend/requirements.txt`, que incluye XGBoost CPU. Con `INSTALL_TORCH=false`, los modelos profundos no están disponibles dentro del contenedor (ver [estado](../IMPLEMENTATION_STATUS.md)).

En PowerShell, desde la raíz del proyecto:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Solo en una instalación nueva, edite .env y reemplace POSTGRES_PASSWORD y JWT_SECRET por valores aleatorios propios.
docker compose up --build -d
docker compose ps
docker compose logs backend --tail 80
```

Las migraciones se ejecutan con `alembic upgrade head` durante el arranque del backend. La API queda en `http://localhost:8000`, OpenAPI en `http://localhost:8000/docs`, salud en `http://localhost:8000/health` y la interfaz en `http://localhost:5173`. La base de datos no publica puerto al host. Los puertos de frontend y API se enlazan a `127.0.0.1`, por lo que el acceso remoto requiere configuración adicional de proxy.

No existe un usuario predeterminado. Cree el primero por terminal interactiva una vez que el backend esté sano:

```powershell
docker compose exec backend python -m app.bootstrap --email admin@example.org --name Administrador
```

El programa pedirá la contraseña sin mostrarla y exigirá 12 caracteres o más. Entre a la interfaz con ese correo y contraseña. La contraseña **no** debe escribirse en `.env`, el historial de comandos ni este repositorio.

Para entrenar los modelos profundos por CPU dentro de Docker, edite `.env` para establecer `INSTALL_TORCH=true` y reconstruya con `docker compose up --build -d`. El Dockerfile instala PyTorch desde el índice CPU oficial; la imagen resultante requiere más descarga, disco y memoria. En este workspace se activó y se probaron LSTM, CNN-LSTM y CNN-Transformer en Compose con la [muestra real de CY-Bench](CYBENCH.md).

Para una instalación Python local sin Compose, configure `DATABASE_URL`, `JWT_SECRET` y `AUTO_CREATE_DB=true`, instale `backend/requirements.txt` y ejecute `uvicorn app.main:app` desde `backend/`. El valor SQLite por defecto de `backend/app/db.py` es para desarrollo; el entorno Compose usa PostgreSQL. No mezcle los datos de ambos modos esperando persistencia compartida.

## Volúmenes y respaldo

`postgres_data` guarda registros; `app_data`, los CSV; `model_artifacts`, los artefactos `joblib` y metadatos JSON. Respaldar los tres de forma consistente. Un respaldo lógico de base de datos puede obtenerse así:

```powershell
docker compose exec -T db pg_dump -U agriculture -d agriculture -Fc -f /tmp/agriculture.dump
$dbContainer = docker compose ps -q db
docker cp "${dbContainer}:/tmp/agriculture.dump" .\agriculture.dump
```

Para restaurar, cree un entorno de destino aislado, restaure primero la base de datos con `pg_restore`, copie los volúmenes de datos y artefactos conservando sus rutas, y verifique que los `file_path` y `artifact_path` registrados existen. No sustituya volúmenes activos durante un entrenamiento. El comando anterior es un procedimiento documentado; consulte [pruebas ejecutadas](../IMPLEMENTATION_STATUS.md) para saber si se verificó en este entorno.

## Puesta en producción

Use HTTPS y `COOKIE_SECURE=true`; establezca `CORS_ORIGINS` al origen exacto de la interfaz. Coloque secretos en un gestor adecuado y controle acceso a los tres volúmenes. La cola `BackgroundTasks` y el almacenamiento local de modelos son apropiados para una instalación pequeña y demostración; para reinicios durante entrenamiento o varios workers se necesita una cola duradera y almacenamiento compartido. No se incluyen servicios MinIO, MLflow, n8n ni LangFlow en Compose.
## Materiales del artículo

Compose monta `./docs/manuscrito` en `/app/publication` en modo de solo lectura y configura `PUBLICATION_DIR`. Deben conservarse juntos el catálogo `catalogo_publicacion.json` y los archivos que declara; sus tamaños y hashes se verifican al servirlos. La API exige sesión para las figuras y descargas. Este directorio se distribuye junto al despliegue, mientras que los modelos y datos siguen en sus volúmenes.

El manuscrito no se recalcula al entrenar una versión nueva. Para regenerarlo se ejecutan los scripts documentados en [el directorio del artículo](manuscrito/README.md), se valida el resultado y se reemplaza el catálogo con sus archivos correspondientes. No editar una imagen aislada dejando un hash antiguo.

