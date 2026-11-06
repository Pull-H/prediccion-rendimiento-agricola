FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
ARG INSTALL_TORCH=false
RUN if [ "$INSTALL_TORCH" = "true" ]; then pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2,<3'; fi
COPY backend/app /app/backend/app
COPY backend/alembic /app/backend/alembic
COPY alembic.ini /app/alembic.ini
RUN mkdir -p /app/data /app/artifacts
ENV PYTHONPATH=/app/backend
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
