# Artículo, seis tablas y seis figuras científicas

## Entrega

- [Artículo editable en Word](articulo_trigo_yieldsat.docx).
- [Artículo en PDF](articulo_trigo_yieldsat.pdf).
- [Artículo fuente en Markdown](articulo_trigo_yieldsat.md).
- [Seis tablas en Excel](tablas_articulo.xlsx) y [CSV individuales](tablas/).
- [Figuras originales a 600 dpi y tablas, ZIP](figuras_y_tablas_600dpi.zip).
- [Control de las directivas de Normas.docx](CONTROL_NORMAS.md).
- [Registro de búsqueda y verificación bibliográfica](REGISTRO_BUSQUEDA.md).

El título ajustado es **Evaluación temporal del rendimiento de trigo con satélite y suelo: piloto reproducible en Alemania**. Se excluye clima del título porque el perfil real YieldSAT todavía no lo integra. La afiliación proporcionada es **Universidad Nacional de Trujillo**. El idioma principal es español, con título y resumen en inglés; el formato de citas es autor-año provisional, al no haberse elegido revista.

El manuscrito contiene seis tablas nativas de Word y seis figuras embebidas. Cada figura se entrega en PNG y TIFF a 600 dpi, PDF vectorial y vista previa a 150 dpi. La aplicación usa la vista ligera y ofrece los originales de 600 dpi para descarga. No se deben utilizar las vistas previas como originales editoriales.

## Módulo de la aplicación

En el menú, abra **Artículo y figuras** (`http://localhost:5173/articulo`). Incluye las seis figuras, sus leyendas, explicaciones, límites de interpretación, las seis tablas. El bloque de presentación y descarga del manuscrito se retiró de la interfaz; los documentos se conservan en esta carpeta. El acceso exige sesión. El backend sirve exclusivamente archivos declarados en `catalogo_publicacion.json`, comprueba su integridad y no permite navegar por rutas arbitrarias. Compose monta esta carpeta en modo de solo lectura para la aplicación.

El artículo es una fotografía del dataset #9 y modelos #14–#18. Nuevos entrenamientos no cambian automáticamente sus gráficos ni conclusiones. Para modificar la evidencia, se debe generar una nueva versión y revisar el manuscrito.

## Evidencia y análisis añadido

`evidencia_modelos.json` es un archivo local excluido de Git que reúne predicciones de modelos congelados, particiones, versiones, hashes, métricas y sensibilidad agrupada. La interpretación por permutación usa solo los 15 registros de validación de 2020, siete grupos, 30 permutaciones y semilla 20261106, sin reentrenar o ajustar modelos después de examinar la prueba. No es SHAP ni causalidad. Las diferencias y desviaciones estándar se presentan completas en la tabla 6.

Las métricas negativas, la concentración de 36/37 registros de prueba en una granja, la diferencia entre ganador observado en prueba y selección por validación, y la falta de clima verificado se mantienen explícitas. Se entrega un borrador científico para revisión; no se garantiza aceptación o elegibilidad editorial Q1.

## Reproducir

Desde la raíz del repositorio:

```powershell
docker compose cp backend/research/export_manuscript_evidence.py backend:/tmp/export_manuscript_evidence.py
docker compose exec -T backend python /tmp/export_manuscript_evidence.py --output /app/data/manuscript_evidence.json
docker compose cp backend:/app/data/manuscript_evidence.json docs/manuscrito/evidencia_modelos.json
.\.venv\Scripts\python.exe backend/research/verify_references.py
.\.venv\Scripts\python.exe backend/research/build_publication.py
.\.venv\Scripts\python.exe backend/research/render_manuscript.py
```

Las dos últimas órdenes usan matplotlib, Pillow, NumPy, pandas, python-docx, openpyxl y reportlab en el entorno de investigación. Los modelos requeridos deben existir en el volumen y corresponder a los identificadores y hashes del estudio. Los scripts se detienen si esas condiciones no se cumplen. La descarga bibliográfica consulta Crossref y los registros de fuentes documentados; no descarga YieldSAT.

## Datos que deben completar los autores antes de enviar

Nombres, orden de autoría, ORCID, correo y autor de correspondencia, facultad/departamento, declaraciones de financiación y conflictos y requisitos éticos institucionales. Confirmar términos para difusión de derivados del dataset y seleccionar revista para adaptar extensión, estilo de referencias y políticas de asistencia con IA. No se inventaron estos datos ni se enviaron materiales a ninguna revista.

La cohorte es pequeña y no respalda validación espacial o peruana. Si se pretende una contribución de generalización o fusión climática, se requiere ampliar el experimento antes de formular esas afirmaciones; editar el título no resuelve esa limitación.
