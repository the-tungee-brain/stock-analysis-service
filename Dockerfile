FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY analysis ./analysis
COPY data ./data
COPY features ./features
COPY models ./models
COPY backtest ./backtest
COPY api ./api
COPY docker-entrypoint.sh .
RUN mkdir -p artifacts data/raw data/features
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

CMD ["./docker-entrypoint.sh"]
