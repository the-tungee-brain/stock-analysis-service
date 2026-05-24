#!/bin/sh
set -e

WORKERS="${GUNICORN_WORKERS:-1}"
TIMEOUT="${GUNICORN_TIMEOUT:-120}"

exec gunicorn app.main:app \
  --workers="${WORKERS}" \
  --worker-class=uvicorn.workers.UvicornWorker \
  --bind=0.0.0.0:8000 \
  --timeout="${TIMEOUT}" \
  --graceful-timeout=30 \
  --max-requests=300 \
  --max-requests-jitter=50 \
  --keep-alive=5
