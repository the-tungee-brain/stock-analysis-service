#!/usr/bin/env bash
# Run ON the OCI VM (ubuntu@...) after git pull or copy this repo to the VM.
# Patches sas-server in-place and finishes ranking bootstrap (daily + portfolio).
set -euo pipefail

CONTAINER="${SAS_CONTAINER:-sas-server}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "Container $CONTAINER is not running" >&2
  exit 1
fi

docker cp "$REPO_ROOT/features/patterns.py" "$CONTAINER:/app/features/patterns.py"
docker cp "$REPO_ROOT/ranking_pipeline/datetime_utils.py" "$CONTAINER:/app/ranking_pipeline/datetime_utils.py"
docker cp "$REPO_ROOT/ranking_pipeline/features/parquet_store.py" \
  "$CONTAINER:/app/ranking_pipeline/features/parquet_store.py"
docker cp "$REPO_ROOT/ranking_pipeline/pipeline/features_batch.py" \
  "$CONTAINER:/app/ranking_pipeline/pipeline/features_batch.py"
docker cp "$REPO_ROOT/ranking_pipeline/pipeline/rank.py" \
  "$CONTAINER:/app/ranking_pipeline/pipeline/rank.py"
docker cp "$REPO_ROOT/ranking_pipeline/regime/detector.py" \
  "$CONTAINER:/app/ranking_pipeline/regime/detector.py"
docker cp "$REPO_ROOT/ranking_pipeline/portfolio/constructor.py" \
  "$CONTAINER:/app/ranking_pipeline/portfolio/constructor.py"

echo "=== hotfix applied; running daily + portfolio (features already built) ==="
docker exec -w /app "$CONTAINER" python scripts/run_ranking_daily.py
docker exec -w /app "$CONTAINER" python scripts/run_portfolio_with_risk.py
touch "${RANKING_BOOTSTRAP_MARKER:-/home/ubuntu/sas-ranking-persist/.pipeline_bootstrapped}"
echo "=== done — marker set ==="
