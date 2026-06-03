#!/usr/bin/env bash
# Run ranking pipeline inside sas-server on the OCI VM (called from GitHub Actions SSH).
set -euo pipefail

CONTAINER="${SAS_CONTAINER:-sas-server}"
MARKER="${RANKING_BOOTSTRAP_MARKER:-/home/ubuntu/sas-ranking-persist/.pipeline_bootstrapped}"
LOG_DIR="${RANKING_LOG_DIR:-/home/ubuntu/logs}"

mkdir -p "$(dirname "$MARKER")" "$LOG_DIR"

exec_in() {
  docker exec -w /app "$CONTAINER" "$@"
}

imports_ok() {
  local err
  err="$(exec_in python -c "import ranking_pipeline, data.download; print('ok')" 2>&1)" || {
    echo "Import check failed inside $CONTAINER:" >&2
    echo "$err" >&2
    exec_in sh -c 'echo "PYTHONPATH=${PYTHONPATH:-<unset>}"; ls -d /app/ranking_pipeline /app/data/download.py 2>&1' >&2 || true
    return 1
  }
  return 0
}

run_bootstrap() {
  echo "=== ranking bootstrap (universe + SPY + daily + portfolio) ==="
  exec_in python scripts/run_ranking_universe_weekly.py
  exec_in python scripts/download_symbols.py --symbols SPY
  exec_in python scripts/run_ranking_daily.py
  exec_in python scripts/run_portfolio_with_risk.py
  touch "$MARKER"
  echo "=== bootstrap complete ==="
}

run_daily() {
  echo "=== ranking daily + portfolio ==="
  exec_in python scripts/run_ranking_daily.py
  exec_in python scripts/run_portfolio_with_risk.py
  echo "=== daily complete ==="
}

run_weekly() {
  echo "=== universe weekly refresh ==="
  exec_in python scripts/run_ranking_universe_weekly.py
  echo "=== weekly complete ==="
}

main() {
  local mode="${1:-daily}"
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "Container $CONTAINER is not running" >&2
    exit 1
  fi
  if ! imports_ok; then
    echo "Fix: ensure Deploy SAS Server succeeded (log: pipeline imports ok), then re-run: $0 $mode" >&2
    exit 1
  fi
  case "$mode" in
    bootstrap) run_bootstrap ;;
    daily) run_daily ;;
    weekly) run_weekly ;;
    *)
      echo "Usage: $0 {bootstrap|daily|weekly}" >&2
      exit 1
      ;;
  esac
}

main "$@"
