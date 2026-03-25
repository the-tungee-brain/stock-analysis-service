FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["gunicorn", "app.main:app", \
    "--workers=2", \
    "--worker-class=uvicorn.workers.UvicornWorker", \
    "--bind=0.0.0.0:8000", \
    "--timeout=300", \
    "--keep-alive=5"]
