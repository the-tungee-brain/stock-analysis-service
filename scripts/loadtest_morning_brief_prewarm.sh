#!/usr/bin/env bash
# Times morning-brief prewarm; same checks as morning-brief.yml prewarm job.
set -euo pipefail

: "${API_BASE_URL:?Set API_BASE_URL, e.g. https://your-host/api/v1}"
: "${CRON_SECRET:?Set CRON_SECRET}"

MAX_SECONDS="${PREWARM_MAX_SECONDS:-600}"
URL="${API_BASE_URL%/}/internal/prewarm-morning-briefs"

echo "POST ${URL} (max ${MAX_SECONDS}s)"
start_epoch=$(date +%s)

response="$(curl -sS --max-time "${MAX_SECONDS}" -w "\n%{http_code}" -X POST \
  "${URL}" \
  -H "X-Cron-Secret: ${CRON_SECRET}" \
  -H "Content-Type: application/json")"

end_epoch=$(date +%s)
duration=$((end_epoch - start_epoch))

http_code="${response##*$'\n'}"
body="${response%$'\n'*}"

echo "HTTP ${http_code}"
echo "${body}"
echo "duration_seconds=${duration}"

if [ "${http_code}" -ge 400 ]; then
  exit 1
fi

failed="$(echo "${body}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('failed', 0))")"
if [ "${failed}" != "0" ]; then
  echo "Morning brief pre-warm reported failures."
  exit 1
fi

echo "loadtest prewarm OK"
