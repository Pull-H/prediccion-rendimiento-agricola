# Registro de revisión narrativa y verificación bibliográfica

Fecha de consulta: 6 de noviembre de 2026. No se tuvo una sesión de búsqueda en Scopus o Web of Science. No se afirma revisión sistemática, exhaustividad ni flujo PRISMA. La revisión se construyó mediante búsqueda web de fuentes primarias, lectura de apartados pertinentes y seguimiento de referencias, con verificación de DOI en Crossref o registros editoriales.

## Consultas efectivamente realizadas

1. `YieldSAT A Global Satellite Dataset for Crop Yield Prediction 2604.00940 authors`
2. `CY Bench crop yield benchmark 2025 dataset paper DOI`
3. `crop yield prediction deep learning satellite weather soil temporal spatial validation remote sensing primary paper`
4. `Roberts 2017 cross validation strategies data temporal spatial hierarchical phylogenetic structure DOI`
5. `site.dl.acm.org 10.1145/2939672.2939785 XGBoost`
6. `site.direct.mit.edu Long Short-Term Memory 1997 1735 1780`
7. `site.proceedings.neurips.cc Attention Is All You Need 2017 Vaswani`
8. `site.pmc.ncbi.nlm.nih.gov Altmann permutation importance corrected feature importance measure 2010`
9. `site.pmc.ncbi.nlm.nih.gov Kapoor Narayanan Leakage reproducibility crisis machine learning science 100804`
10. `site.projecteuclid.org Efron 1979 Bootstrap Methods Another Look Jackknife`
11. `site.jstor.org Holm 1979 A Simple Sequentially Rejective Multiple Test Procedure`
12. `"Individual Comparisons by Ranking Methods" "80-83" Wilcoxon 1945`

Se usaron las palabras de las consultas tal como fueron formuladas; la primera búsqueda podía contener un título aproximado, que se sustituyó en la referencia por el título real del registro. No se inventaron filtros de base, números de artículos recuperados o una fecha límite formal no aplicada.

## Criterios de selección

Pertinencia directa para datos agrícolas multimodales, separación temporal/espacial, las familias de modelos realmente implementadas, contrastes utilizados o interpretabilidad por perturbación. Se priorizaron trabajos originales y descripciones primarias de datasets. Se distinguieron artículos de revista, actas, preprints y documentación de software; no se afirmó que todas las referencias fueran Q1.

Se consultaron contenido metodológico y resultados pertinentes de YieldSAT (arXiv 2604.00940v1), Pathak et al. (arXiv 2308.08948v1 y DOI de IGARSS), Khaki et al. (Frontiers), CY-Bench (editorial ESSD), validación estructurada de Roberts et al., el estudio de fugas de Kapoor y Narayanan y las fuentes de métodos. Los resúmenes y metadatos editoriales se utilizaron para identificar publicaciones, sin convertirlos en una revisión exhaustiva.

## Verificación

`referencias_verificadas.json` conserva título, autores, fecha, revista/actas, volumen, número, páginas o número de artículo y DOI cuando se recuperó. Doce DOI se verificaron contra `api.crossref.org/works/{doi}`. Las páginas 80–83 de Wilcoxon se contrastaron con el escaneo original alojado por la Universidad de Granada, pues Crossref solo devolvía la primera página. Los registros oficiales de arXiv y NeurIPS y el artículo original de Holm respaldan las entradas sin DOI editorial declarado. Dos entradas documentales corresponden a SciPy y al tutorial oficial de YieldSAT.

Las primeras consultas paralelas con agente personalizado recibieron HTTP 429; se completó la recuperación secuencial y se guardaron los registros devueltos. No se rellenaron DOI ausentes con identificadores generados. El artículo arXiv de YieldSAT se cita como preprint: la mención de un evento en el tutorial no se tomó como prueba de publicación editorial final.

## Matriz y límites

La matriz complementaria `matriz_literatura.csv` contrasta problema, unidad, método, validación y límites de los antecedentes agrícolas pertinentes. Las métricas de diferentes cultivos, modalidades o unidades no se combinan ni se usan como ranking externo del presente piloto. Las referencias se emplean para las afirmaciones de contexto/método que respaldan, mientras que los resultados nuevos proceden de los artefactos del proyecto.

Antes del envío, los autores pueden ampliar esta búsqueda con sus accesos institucionales a Scopus/WoS, conservar estrategias exportables y adaptar el estado del arte a la revista elegida. Esta recomendación no se presenta como una búsqueda ya ejecutada.
