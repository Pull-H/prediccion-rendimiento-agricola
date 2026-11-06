# Reproducir y ampliar el estudio

Versión del protocolo: **1.0.0**. Estudio: trigo alemán de YieldSAT, satélite y suelo a 240 días; 184 registros, semilla 42 y cinco modelos. [Código y resultados de la versión](https://github.com/Pull-H/prediccion-rendimiento-agricola/releases/tag/v1.0.0).

## Qué está disponible públicamente

- Código completo de la aplicación, las transformaciones y los análisis.
- `study.json`: hash del CSV, configuración real, versiones y resultados agregados.
- `splits.csv`: índice de fila, campaña y partición; no contiene rendimientos ni identificadores fuente. El índice solo identifica una fila cuando se usa el CSV con el hash declarado.
- `requirements-lock.txt`: versiones de los paquetes del contenedor del estudio.
- Seis tablas y seis figuras, originales a 600 dpi, en `docs/manuscrito/`.
- Scripts para volver a entrenar, verificar modelos congelados y regenerar materiales.

Esta entrega permite inspeccionar el código y los resultados agregados. Ejecutar los análisis sobre observaciones originales requiere obtener acceso a los datos.

## Obtener el mismo conjunto de datos

1. Solicitar YieldSAT desde [su sitio oficial](https://yieldsat.github.io/) y aceptar las condiciones del proveedor. La ficha del conjunto declara acceso bajo una licencia específica y uso académico no comercial: [apartados B.5–B.7](https://arxiv.org/html/2604.00940v1).
2. Usar el NetCDF preprocesado de Alemania. No hace falta descargar el original en el equipo local: los scripts `notebooks/` permiten trabajar en Colab.
3. Ejecutar `notebooks/YieldSAT_exportar_trigo_colab.py` sobre el archivo autorizado. Se excluyen siembras inferidas desde cosecha y se agregan las observaciones hasta siembra + 240 días. El manifiesto registra exclusiones y procedencia.
4. Comprobar que el SHA-256 de `yieldsat_Germany_wheat_240d.csv` coincide con:

```text
fa9496d9598896042152d3bd669edd1290384630f0e880d79fdde483aee0e2e1
```

En PowerShell: `Get-FileHash <ruta-del-CSV> -Algorithm SHA256`. Una revisión nueva de YieldSAT puede producir otro hash; en tal caso se trata como una cohorte nueva, no como reproducción exacta del CSV original.

## Volver a entrenar sin usuarios ni base de datos

Se recomienda Python 3.12 y el entorno Linux CPU del estudio. El archivo de versiones refleja el entorno observado, no garantiza disponibilidad futura de todos los binarios ni identidad numérica entre plataformas.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r reproducibility/requirements-lock.txt
.\.venv\Scripts\python.exe backend/research/train_study.py --csv data/yieldsat_wheat_240d/yieldsat_Germany_wheat_240d.csv --output data/reproduction_run_01
```

Se entrenan los cinco modelos con semilla 42, corte temporal 132/15/37 y 30 épocas para las redes. Se generan artefactos nuevos y comparaciones estadísticas; el código rechaza sobrescribir una carpeta existente. Para extender el trabajo con otra cohorte se usa `--new-study`; cambiar datos, semilla o épocas produce un experimento nuevo. La prueba original ya fue examinada y no debe reutilizarse como evaluación ciega después de ajustar decisiones con sus resultados.

## Verificar los pesos originales

Se preparó un paquete local completo `data/reproducibility/yieldsat-wheat-v1.0.0.zip` con el CSV exacto, procedencia, particiones con identificadores, cinco modelos y sus preprocesadores, predicciones individuales, análisis de permutación y entorno. **Ese paquete no está depositado públicamente:** su redistribución y la de los pesos entrenados quedan pendientes de revisar las condiciones recibidas del proveedor. No se ofrece aquí acceso automático ni entrega a terceros sin comprobar esas condiciones.

Cuando se disponga legítimamente del paquete, extraerlo y ejecutar desde la raíz del código:

```powershell
.\.venv\Scripts\python.exe backend/research/replay_study.py --bundle data/reproducibility/yieldsat-wheat-v1.0.0 --verify-permutation
```

El verificador comprueba los hashes, las asignaciones de partición, las predicciones de los cinco modelos, MAE/RMSE/R², Friedman, Wilcoxon-Holm, intervalos bootstrap y las 30 permutaciones de cada grupo. Solo deben cargarse artefactos `joblib` de una fuente confiable. Este proceso no reentrena ni requiere la base original.

Para reconstruir las figuras, colocar el CSV y su `.provenance.json` en `data/yieldsat_wheat_240d/` y copiar la evidencia del paquete a `docs/manuscrito/evidencia_modelos.json`. Instalar `backend/research/requirements.txt` y seguir [la guía del manuscrito](../docs/manuscrito/README.md). La fase de exportación desde la base original no es necesaria si ya se dispone de esa evidencia.

## Preservación y cita

La etiqueta `v1.0.0` identifica el código publicado. Esta versión se archivó en Zenodo con **[DOI: 10.5281/zenodo.24022432](https://doi.org/10.5281/zenodo.24022432)**. `CITATION.cff` permite citarla como software. El ZIP depositado corresponde a la instantánea original de la versión; sus declaraciones sobre la ausencia de DOI preceden al registro, y esta documentación se actualizó después de publicarlo. El depósito incluye código y materiales públicos, sin datos individuales ni pesos restringidos. Si se autoriza su redistribución, deberá crearse una nueva versión del depósito y actualizar esta declaración.

`validation.json` registra las comprobaciones ejecutadas: verificación de los cinco modelos congelados y sus análisis, y un reentrenamiento separado de los cinco modelos con las mismas versiones y CPU, sin alterar los artefactos originales. El nuevo entrenamiento reprodujo las métricas de prueba dentro de la precisión numérica indicada; esto no garantiza igualdad en otros entornos.

La licencia MIT de este repositorio se aplica al código propio, no concede derechos sobre YieldSAT ni reemplaza las condiciones de los datos, las dependencias o los materiales de terceros.
