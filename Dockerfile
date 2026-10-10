# MAI-13 backend + dashboard image (CPU-only).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3" \
    && pip install -r requirements.txt

COPY alembic.ini pyproject.toml ./
COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts
COPY experiments/configs ./experiments/configs

# Run as an unprivileged user; data/artifacts/models are mounted volumes.
RUN useradd --create-home --uid 1000 mai13 \
    && mkdir -p /app/data /app/artifacts /app/datasets /app/reports /app/mlruns /models \
    && chown -R mai13:mai13 /app /models
USER mai13

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
