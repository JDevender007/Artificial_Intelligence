# Serving image: the exported model + FastAPI. Build after `dvc repro` (or `python -m credops.pipeline train`).
FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models PREDICTION_LOG=/app/logs/predictions.jsonl PORT=8000 WEB_CONCURRENCY=2
# lightgbm needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps .
COPY models/model.joblib models/metadata.json models/reference_scores.npy ./models/
RUN useradd --create-home --uid 1000 app && mkdir -p /app/logs && chown -R app /app/logs
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn credops.api:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY}"]
