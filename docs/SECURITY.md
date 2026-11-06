# Seguridad y operación

## Controles presentes

- Las contraseñas se almacenan con `pwdlib` y Argon2. El administrador se crea por CLI con contraseña solicitada de manera interactiva y longitud mínima de 12 caracteres; no hay contraseña predeterminada en el repositorio.
- La sesión usa una cookie `session` HttpOnly con JWT HS256 y vencimiento de ocho horas. Cada JWT lleva un `jti` registrado en `session_tokens`; al cerrar sesión se revoca y las peticiones posteriores se rechazan. `JWT_SECRET` debe tener al menos 32 caracteres; el backend rechaza su arranque si es más corto.
- Las operaciones `POST` y `PATCH` de una sesión autenticada requieren que el valor de la cookie `csrf_token` coincida con la cabecera `X-CSRF-Token`. Las cookies usan `SameSite=Lax`. `COOKIE_SECURE=true` debe activarse detrás de HTTPS.
- FastAPI comprueba los roles `admin`, `researcher`, `analyst` y `viewer`. El administrador gestiona usuarios y auditoría; investigador y administrador importan datos e inician entrenamientos; analista además puede inferir. Las rutas de lectura requieren sesión.
- Los CSV se limitan a 20 MB, extensión `.csv`, UTF-8 y validación de columnas/contenido. Las consultas de datos usan SQLAlchemy; los nombres de archivo recibidos no se usan como rutas de escritura. La API limita a dos experimentos en cola o ejecución.
- El login limita cinco fallos por combinación de correo e IP cliente en una ventana de 15 minutos; el sexto intento recibe HTTP 429.
- Se registran acceso, fallos de autenticación, cambios de rol, importaciones, entrenamientos, inferencias, navegación y descargas de informes. El endpoint de auditoría se limita a administradores.

## Configuración

Guarde `POSTGRES_PASSWORD` y `JWT_SECRET` solo en `.env` local o un gestor de secretos; excluya `.env`, CSV autorizados, archivos `joblib`, cookies y respaldos de Git. En Compose, PostgreSQL no publica un puerto al host y los puertos de aplicación se enlazan a `127.0.0.1`. Para acceso remoto, usar un proxy HTTPS confiable y activar `COOKIE_SECURE=true`. Limitar `CORS_ORIGINS` a los orígenes del despliegue; no usar `*` con cookies.

## Límites conocidos y endurecimiento pendiente

Hay revocación por `jti` al cerrar sesión, pero todavía no hay una pantalla de administración de sesiones, rotación de claves ni limpieza de sesiones expiradas. El límite de intentos de login vive en la memoria de cada proceso: se reinicia al reiniciar y no se comparte entre workers; un despliegue distribuido requiere un almacén común. Tampoco se observó recuperación de contraseña ni política de retención/anonimización de auditoría. El correo de un intento fallido se registra en auditoría y requiere una política de protección de datos personales.

Los artefactos `joblib` se cargan para inferencia; este formato puede ejecutar código al deserializar. Mantener el volumen de artefactos bajo control del proceso confiable, no aceptar artefactos subidos por usuarios y revisar cualquier proceso futuro de importación de modelos. Los CSV importados están acotados por tamaño, pero el entrenamiento con datos arbitrarios aún puede consumir mucha memoria o CPU. `BackgroundTasks` no ofrece aislamiento de recursos por trabajo ni persistencia tras reinicio. La API de datasets devuelve `provenance`, que actualmente contiene una ruta local; conviene sustituirla por un identificador público de procedencia antes de exponer la API fuera del entorno local. La aprobación de modelos no constituye una revisión estadística automática: el administrador debe revisar origen y evaluación.

No se debe exponer este despliegue de desarrollo a Internet sin completar los controles pendientes y pruebas de seguridad. Los datos reales de parcelas pueden revelar localizaciones e información comercial sensible; revisar permisos de acceso, finalidad y retención de acuerdo con los términos de cada fuente.
