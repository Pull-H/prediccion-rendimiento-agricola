# API REST

FastAPI expone el contrato interactivo en `/docs` y OpenAPI JSON en `/openapi.json`. La ruta `/health` comprueba el acceso a la base de datos. Las rutas `/api/*`, salvo el inicio de sesión, requieren cookie `session`. Para `POST` y `PATCH` autenticados, enviar la cookie `csrf_token` también en `X-CSRF-Token`. El navegador debe incluir credenciales (`credentials: "include"`). Los errores habituales son `401` sin sesión, `403` sin permiso/CSRF, `422` por validación y `429` por límite de entrenamientos.

## Sesión y roles

| Método | Ruta | Rol | Acción |
| --- | --- | --- | --- |
| `POST` | `/api/auth/login` | Público | `{ "email": "...", "password": "..." }`; establece cookies de sesión y CSRF. |
| `GET` | `/api/auth/me` | Autenticado | Usuario actual. |
| `POST` | `/api/auth/logout` | Autenticado | Cierra la sesión del navegador. |
| `GET` | `/api/users` | admin | Lista usuarios. |
| `POST` | `/api/users` | admin | Crea usuario con `email`, `name`, `password` de al menos 12 caracteres y `role`. |
| `PATCH` | `/api/users/{id}/role?role=analyst` | admin | Cambia rol. |

Los roles admitidos son `admin`, `researcher`, `analyst` y `viewer`. La autorización se aplica en FastAPI; la visibilidad de botones del frontend no concede acceso.

## Datos y análisis

| Método | Ruta | Rol | Acción |
| --- | --- | --- | --- |
| `GET` | `/api/dashboard` | Autenticado | Conteos y actividad almacenada. |
| `GET` | `/api/datasets` | Autenticado | Datasets y control de calidad. |
| `POST` | `/api/datasets/import` | admin, researcher | Multipart `file` CSV UTF-8, máximo 20 MB; metadatos opcionales `source_name`, `license_name`, `provenance_note`. |
| `POST` | `/api/datasets/demo` | admin, researcher | Crea dataset sintético reproducible, marcado como tal. |
| `GET` | `/api/parcels?dataset_id=&crop=&region=` | Autenticado | Parcelas, máximo 2.000 por respuesta. |
| `GET` | `/api/experiments` | Autenticado | Historial de entrenamientos. |
| `GET` | `/api/experiments/{id}` | Autenticado | Estado, avance, métricas o error. |
| `POST` | `/api/experiments` | admin, researcher | Solicita entrenamiento en segundo plano. |
| `POST` | `/api/experiments/{id}/cancel` | admin, researcher | Solo cancela tareas aún en cola. |
| `GET` | `/api/models` | Autenticado | Versiones de modelos registrados. |
| `GET` | `/api/models/{id}/importance` | Autenticado | Importancia global para Random Forest y XGBoost; `501` para modelos profundos. |
| `POST` | `/api/models/{id}/approve` | admin | Aprueba un modelo no sintético. |
| `GET` | `/api/predictions` | Autenticado | Últimas 100 predicciones. |
| `POST` | `/api/predictions` | admin, researcher, analyst | Infiere y almacena una predicción. |

Al importar un CSV, `source_name` identifica la fuente declarada (máximo 80 caracteres; por defecto `user_upload`), `license_name` la licencia declarada (máximo 255) y `provenance_note` una nota de procedencia (máximo 1.000). La respuesta devuelve `source`, `license` y `provenance`. Son declaraciones del usuario que la aplicación conserva, **no verificaciones automáticas** de licencia o autenticidad. Si falta la nota, `provenance` conserva la ruta interna del CSV. El informe científico incluye esos campos.

`POST /api/experiments` recibe, por ejemplo:

```json
{"dataset_id": 1, "model_type": "RandomForest", "seed": 42, "validation": "parcel", "epochs": 5}
```

`model_type` admite `RandomForest`, `XGBoost`, `LSTM`, `CNN-LSTM` y `CNN-Transformer`. Los cinco se entrenaron con la [muestra real de CY-Bench](CYBENCH.md), aunque cada nuevo experimento debe comprobarse por su estado. `validation` admite `parcel`, `campaign`, `region` y `temporal`. Esta última exige campañas como años de cuatro dígitos, asigna los años más recientes a prueba y los intermedios a validación. La implementación limita a dos experimentos `queued`/`running`. Su ejecución usa `BackgroundTasks` dentro del proceso API, por lo que no tiene la durabilidad de una cola externa.

`GET /api/statistics/compare?dataset_id=7&validation=temporal&seed=42` compara cinco modelos completos sobre la misma prueba mediante Friedman, Wilcoxon pareado con ajuste de Holm e intervalos bootstrap por bloque espacial. Requiere autenticación y devuelve `409` si falta un modelo comparable. [Método, resultados y límites](STATISTICS.md).

`POST /api/predictions` recibe `model_id`, `features` y `parcel_id` opcional. Las claves de `features` deben coincidir **exactamente** con `feature_names` del modelo; las características numéricas llevan números y `crop`, `region`, `campaign` llevan texto. Consultar `GET /api/models` antes de enviarlas. La respuesta contiene `yield_t_ha`, versión (`model_id`), entradas, fecha, advertencias y origen sintético.

La respuesta de importancia devuelve `model_id`, `method: "model_feature_importances"`, `synthetic`, `causal: false` y hasta 30 pares `{feature, importance}` en orden descendente. Usa la importancia global incluida en los estimadores de árboles; **no es SHAP ni una explicación causal**. Para redes profundas responde `501` porque ese método aún no está implementado.

## Reportes y auditoría

| Método | Ruta | Rol | Acción |
| --- | --- | --- | --- |
| `GET` | `/api/reports/{kind}.{ext}` | Autenticado | Descarga reporte real. `kind`: `operational`, `managerial`, `scientific`; `ext`: `pdf`, `docx`, `xlsx`. |
| `GET` | `/api/audit` | admin | Últimos 200 eventos. |
| `POST` | `/api/audit/navigation` | Autenticado | `{ "module": "dashboard", "direction": "enter" }` o `leave`. |
| `POST` | `/api/assistant/ask` | Autenticado | `{ "question": "métricas del modelo" }`; respuesta estructurada desde datasets/modelos registrados. |

Los informes consultan registros persistidos. El operacional incluye estados y errores de tareas y recuento de auditoría; el científico incluye origen, licencia y variables de datasets, además de semilla, partición y métricas por experimento. Aún carecen de gráficos científicos y pruebas estadísticas.

El asistente responde preguntas sobre datasets, variables, modelos y métricas usando registros de la base y devuelve `sources` con identificadores como `model_versions:1`. No consulta documentos externos ni usa LangChain/LLM; si no tiene datos, lo indica.

## Sesión reproducible con curl

Ejemplo con un administrador ya creado mediante el procedimiento seguro del [README](../README.md). Las variables deben ponerse en el shell local y nunca añadirse al repositorio.

```bash
curl -c cookies.txt -b cookies.txt -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.org","password":"CONTRASENA_LOCAL"}' \
  http://localhost:8000/api/auth/login

curl -b cookies.txt http://localhost:8000/api/dashboard
```

Para `POST` autenticados, lea el valor de `csrf_token` de `cookies.txt` y envíelo como `X-CSRF-Token`; el navegador del frontend realiza este paso. Mantenga `cookies.txt` fuera de Git.
## Artículo y figuras

- `GET /api/publication`: catálogo autenticado de un manuscrito congelado, seis figuras, seis tablas y archivos disponibles. Incluye explicaciones, limitaciones, resolución y hashes; omite rutas internas.
- `GET /api/publication/files/{filename}?download=true`: entrega un archivo declarado, comprobando tamaño y SHA-256; sin `download=true` se permite vista en línea. No admite navegación por directorios. Requiere sesión, también para imágenes y documentos.
- Si los materiales faltan o el catálogo no es válido devuelve 503; un archivo ajeno al catálogo o fuera de la raíz devuelve 404; una alteración respecto del hash devuelve 409. El módulo no modifica modelos ni genera datos nuevos.

