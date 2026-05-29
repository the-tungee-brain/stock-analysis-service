import http from "k6/http";
import { check } from "k6";

/**
 * End-to-end timing for POST /internal/prewarm-morning-briefs (all Schwab users).
 * One VU — the server parallelizes internally via MORNING_BRIEF_PREWARM_WORKERS.
 */
const baseUrl = (__ENV.API_BASE_URL || "").replace(/\/$/, "");
const cronSecret = __ENV.CRON_SECRET || "";
const maxDurationSeconds = Number(__ENV.PREWARM_MAX_SECONDS || "600");

export const options = {
  scenarios: {
    prewarm: {
      executor: "shared-iterations",
      vus: 1,
      iterations: 1,
      maxDuration: `${maxDurationSeconds}s`,
    },
  },
  thresholds: {
    http_req_failed: ["rate==0"],
    http_req_duration: [`p(95)<${maxDurationSeconds * 1000}`],
    checks: ["rate==1"],
  },
};

export function setup() {
  if (!baseUrl || !cronSecret) {
    throw new Error("Set API_BASE_URL and CRON_SECRET");
  }
}

export default function () {
  const res = http.post(`${baseUrl}/internal/prewarm-morning-briefs`, null, {
    headers: {
      "X-Cron-Secret": cronSecret,
      "Content-Type": "application/json",
    },
    timeout: `${maxDurationSeconds}s`,
    tags: { name: "prewarm-morning-briefs" },
  });

  check(res, {
    "status is 200": (r) => r.status === 200,
  });

  let body;
  try {
    body = res.json();
  } catch {
    body = null;
  }

  check(body, {
    "response is JSON object": (b) => b !== null && typeof b === "object",
    "no failed users": (b) => b && b.failed === 0,
  });

  if (body) {
    console.log(
      `prewarm attempted=${body.attempted} warmed=${body.warmed} skipped=${body.skipped} failed=${body.failed} duration_ms=${res.timings.duration}`,
    );
    if (body.errors && body.errors.length > 0) {
      console.log(`errors sample: ${JSON.stringify(body.errors.slice(0, 5))}`);
    }
  }
}
