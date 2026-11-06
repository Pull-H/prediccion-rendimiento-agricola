# Revisión del alcance funcional

Revisión del 6 de noviembre de 2026 contra `tema (2).docx`, especialmente AP-02, F9 y 6.5, y contra el piloto real elegido por el usuario.

## Módulos que corresponden a la investigación

| Módulo | Función y alcance efectivo |
| --- | --- |
| Resumen | Datos y actividad registrados. El gráfico compara las últimas versiones del mismo dataset, evaluación temporal y semilla 42; da prioridad a YieldSAT. No mezcla experimentos de fuentes diferentes. |
| Datasets | Importación, calidad y procedencia. YieldSAT es el piloto de trigo actual; CY-Bench conserva el antecedente real regional español. Los sintéticos no forman parte del modo normal. |
| Unidades y mapa | Muestra registros y coordenadas existentes. No inventa coordenadas, límites de parcelas ni mapas de rendimiento. |
| Laboratorio | Cinco algoritmos del tema y sus experimentos reales. El registro principal muestra la última versión de cada algoritmo; las versiones anteriores permanecen en el historial. |
| Pruebas estadísticas | Página independiente `/estadisticas`: metodología, resultados reales, versiones, muestra, p originales y ajustadas, intervalos y limitaciones. El protocolo solo puede seleccionarse si existen los cinco modelos completos con el mismo dataset, partición y semilla. |
| Predicciones | Carga automática de datos importados por campaña/parcela y registro de estimaciones. Identifica consultas históricas; no promete pronósticos de campañas nuevas sin observaciones compatibles. |
| Informes | PDF/DOCX/XLSX sobre los datos reales visibles. Los resultados estadísticos completos se descargan en JSON desde el nuevo módulo; los reportes científicos generales todavía no incorporan sus tablas automáticamente. |
| Consulta de resultados | Consulta estructurada de la base. Se renombró la navegación para no presentarla como agente LLM, RAG o recomendador agronómico. |
| Usuarios y auditoría | Acceso por roles y trazabilidad, contemplados en la arquitectura del tema. |

## Limpieza realizada

- Se eliminó el generador de demostraciones de la interfaz y se deshabilitó su API en el modo normal. Los registros sintéticos y sus derivados no aparecen en listas, indicadores, consultas ni reportes.
- Se retiró el aviso genérico de demostraciones del resumen y el CSS sin uso del generador.
- Se corrigió la mezcla de fuentes/protocolos del gráfico inicial y la selección predeterminada de CY-Bench en la comparación estadística.
- Se sustituyó el indicador estático «Sistema en línea» por «Entorno de investigación»: ese texto no era una comprobación de salud.
- El acceso y el resumen describen investigación y experimentos, sin afirmar que el perfil YieldSAT ya integra clima o está listo para decisiones operativas.
- Se separan las métricas descriptivas de los contrastes y de la estimación de incertidumbre. No se añaden pruebas de clasificación al objetivo continuo.
- La administración muestra cuentas activas por defecto; las cuentas inactivas de las verificaciones técnicas solo aparecen al solicitar el historial.

## Requisitos científicos todavía pendientes

La revisión de pertinencia no completa los requisitos aún ausentes: clima de agregación verificada en el piloto YieldSAT, índices calibrados NDVI/EVI/NDWI, etapas fenológicas comprobadas, CNN con parches espaciales, SHAP/interpretabilidad temporal, optimización y validación espacial/externa suficientes. Las cuatro ventanas temporales actuales no acreditan etapas fenológicas.

La meteorología y los índices de CY-Bench no acreditan esas modalidades en YieldSAT. La importancia global de árboles no se presenta como SHAP. La consulta estructurada no se presenta como LangChain/LangFlow. El workflow de salud preparado no equivale a seis flujos de automatización ejecutados.

## Interpretación estadística

Friedman se calcula sobre pérdidas por bloque y usa la aproximación χ² de SciPy; con cinco modelos se explicita la cautela sobre esa aproximación. Wilcoxon usa diferencias pareadas bilaterales y presupone simetría. Holm ajusta la familia de diez comparaciones. El bootstrap local remuestrea identificadores y calcula percentiles del MAE; no proporciona intervalos individuales de rendimiento. Las referencias oficiales se incluyen en el módulo y en [STATISTICS.md](STATISTICS.md).

El piloto tiene 37 registros de prueba, 36 de una misma granja: no se declara verificada la independencia de esos bloques. No se interpreta una p menor de 0,05 como prueba de buena precisión, causalidad ni superioridad universal. McNemar y DeLong no corresponden al objetivo continuo; no se implementa Diebold-Mariano sin un protocolo temporal adecuado.
