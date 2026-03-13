#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="tungxd96/sas-server:local"
CONTAINER_NAME="sas-server-dev"
ENV_FILE=".env"

# Ensure .env exists
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE not found in current directory"
  exit 1
fi

echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

# Stop & remove existing container if present
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping existing container: $CONTAINER_NAME"
  docker stop "$CONTAINER_NAME" || true
  echo "Removing existing container: $CONTAINER_NAME"
  docker rm "$CONTAINER_NAME" || true
fi

echo "Starting container: $CONTAINER_NAME"
docker run -d \
  --name "$CONTAINER_NAME" \
  --env-file "$ENV_FILE" \
  -p 8000:8000 \
  --restart unless-stopped \
  "$IMAGE_NAME"

echo "Container started. Logs:"
docker logs -f "$CONTAINER_NAME"
