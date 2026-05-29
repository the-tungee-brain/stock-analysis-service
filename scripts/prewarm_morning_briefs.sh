#!/usr/bin/env bash
set -euo pipefail

: "${API_BASE_URL:?Set API_BASE_URL, e.g. https://thetungeebrain.duckdns.org/api/v1}"
: "${CRON_SECRET:?Set CRON_SECRET to match the server env var}"

curl -sS -X POST \
  "${API_BASE_URL}/internal/prewarm-morning-briefs" \
  -H "X-Cron-Secret: ${CRON_SECRET}" \
  -H "Content-Type: application/json"

echo
