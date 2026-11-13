# Herramientas para reproducir el estudio original

Este directorio conserva las instrucciones y las dependencias fijadas del
estudio v1.0.0. Los resultados particulares, manifiestos, particiones y registros
del depósito se conservan localmente y no forman parte del repositorio de software
actual. Para obtener el protocolo publicado, consultar la versión original
identificada por `CITATION.cff` y verificar su contenido y condiciones de acceso.

Las herramientas de preparación están en `notebooks/` y `backend/app/ml/`.
Los scripts `backend/research/train_study.py` y `replay_study.py` permiten
entrenar y comprobar un paquete de estudio obtenido legítimamente. El CSV fuente,
los modelos y sus preprocesadores deben adquirirse con autorización. No se
incluyen datos de YieldSAT ni pesos entrenados.

La licencia MIT del código no autoriza redistribuir datos o derivados de terceros.
El módulo general Investigación permite crear protocolos nuevos con datos propios;
no hay que rellenar los materiales ausentes con resultados inventados. La prueba
del estudio original ya fue examinada y no debe presentarse como una evaluación
ciega nueva después de ajustar decisiones con sus resultados.

`requirements-lock.txt` describe el entorno del estudio original y puede diferir
del entorno actual. Revisar su compatibilidad antes de usarlo. Los resultados de
nuevos experimentos deben llevar su propia configuración, semilla, hashes y versiones.
