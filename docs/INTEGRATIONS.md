# Automatización y asistente

## n8n

El archivo [`workflows/WF-06-health.json`](../workflows/WF-06-health.json) es un flujo importable de control de salud. Cada 15 minutos consulta `GET /health` y falla si HTTP devuelve error o si `status`/`database` no son `ok`. No requiere credenciales porque ese endpoint es público y solo informa salud. Al importarlo, configure la URL del nodo **Consultar salud** según dónde corra n8n: `http://localhost:8000/health` si está en el host, o `http://backend:8000/health` si comparte la red de Compose. Pruebe una ejecución manual y publique/active el flujo para que funcione la programación. [Documentación oficial de Schedule Trigger](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.scheduletrigger/).

No se incluyó n8n en `compose.yaml`; el JSON está preparado para importar en una instancia propia. No se configuró un destino de notificaciones. Los fallos quedan en el historial de ejecuciones de n8n; para alertar por correo o webhook se necesita un destino y credenciales proporcionados por el operador.

Los workflows WF-01 a WF-05, incluidos la descarga de reportes y las predicciones programadas, siguen pendientes. La API actual autentica mediante cookie de sesión emitida en login. Un flujo automático necesitaría almacenar una cuenta de servicio segura, gestionar sesión y proteger los datos de ejecución que podrían contener cookies. No se suministran tokens ni contraseñas ficticias, ni se registra un flujo de reporte que no haya sido probado. La [documentación de HTTP Request](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.httprequest/) describe la obtención de cabeceras y archivos cuando se implemente un mecanismo de autenticación para automatización.

## LangChain y LangFlow

Hay una alternativa local en `POST /api/assistant/ask`: consulta estructurada de los últimos datasets y modelos de la base, con identificadores de origen. No usa un LLM, LangChain ni RAG. No hay servidor LangFlow en esta versión. Para incorporarlos, primero se necesita un conjunto autorizado de documentos, índice versionado, permisos de lectura por rol y una fuente verificable para cada respuesta. FastAPI debería entregar al asistente resultados persistidos mediante una API limitada; las métricas deben proceder de la base, nunca de generación libre. LangFlow tendría un papel de diseño y evaluación visual de esos flujos y no debería crear un segundo motor de predicción.
