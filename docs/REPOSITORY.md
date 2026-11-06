# Contenido y mantenimiento del repositorio

## Archivos conservados

El repositorio contiene el código ejecutable, configuración reproducible, migraciones, pruebas, documentación técnica y científica, scripts de preparación autorizada y los materiales que utiliza el módulo Artículo y figuras. Los documentos originales `tema (2).docx` y `Normas.docx` permiten contrastar los requisitos con la implementación.

Los PNG, TIFF, PDF y vistas previas del artículo tienen funciones distintas: visualización ligera, originales editoriales y descarga vectorial. No son copias prescindibles: el catálogo verifica sus hashes y la interfaz los enlaza. Los scripts de investigación y las pruebas se conservan para mantener y verificar el sistema.

## Limpieza realizada

- Se retiraron cachés Python y pytest, la compilación local de Vite y metadatos de compilación regenerables.
- Se retiraron las dos bases SQLite temporales de verificación de migraciones.
- Los adjuntos de inspección y el ZIP autorizado se trasladaron de la raíz a `data/source_receipts/`, excluido de Git; se conservan como evidencia privada.
- Se actualizó el README para describir el piloto YieldSAT actual y retirar el recorrido obsoleto de demostración.

Los elementos retirados se guardaron en un respaldo local fuera del repositorio, `../respaldo-limpieza-agricola-20261106/`, para permitir recuperar cualquier archivo.

Las dependencias instaladas (`.venv` y `frontend/node_modules`) permanecen disponibles localmente, excluidas de Git. Los volúmenes Docker y los modelos entrenados permanecen intactos.

## Información local excluida

`.env`, `data/`, `artifacts/`, respaldos de base de datos y `docs/manuscrito/evidencia_modelos.json` no se suben. Este último contiene observaciones y predicciones individuales derivadas de datos autorizados; los documentos y gráficos agregados ya generados permiten usar el módulo del artículo sin ese archivo.

El acceso autorizado a YieldSAT no se interpreta como permiso de redistribución. El repositorio se crea privado; antes de hacerlo público deben revisarse los términos de difusión de los derivados y los documentos de autoría.

## Qué ocurre al clonar

Docker puede construir e iniciar la aplicación y servir las seis figuras y tablas. La base de datos estará vacía: se crea una cuenta administradora y se importan datos autorizados o se restauran los tres volúmenes de una instalación existente según [DEPLOYMENT.md](DEPLOYMENT.md). Los identificadores de modelos del artículo pertenecen a la instalación del estudio y no aparecen automáticamente en una base nueva.

Para regenerar y verificar íntegramente el artículo se necesitan el CSV autorizado, los modelos congelados y la evidencia individual local. Los comandos están en [la guía del manuscrito](manuscrito/README.md). No se deben sustituir los resultados por datos inventados para rellenar una instalación nueva.
