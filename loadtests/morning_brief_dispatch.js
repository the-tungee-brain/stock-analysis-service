import http from "k6/http";
import { check } from "k6";

/**
 * POST /internal/dispatch-morning-briefs — sends real emails. Staging only.
 */
const baseUrl = (__ENV.API_BASE_URL || "").replace(/\/$/, "");
const cronSecret = __ENV.CRON_SECRET || "";
const maxDurationSeconds = Number(__ENV.DISPATCH_MAX_SECONDS || "300");
const force = __ENV.DISPATCH_FORCE === "true";

export const options = {
  scenarios: {
    dispatch: {
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
  if (__ENV.ALLOW_DISPATCH !== "true") {
    throw new Error(
      "Refusing dispatch load test without ALLOW_DISPATCH=true (sends email)",
    );
  }
}

export default function () {
  const url = `${baseUrl}/internal/dispatch-morning-briefs${force ? "?force=true" : ""}`;
  const res = http.post(url, null, {
    headers: {
      "X-Cron-Secret": cronSecret,
      "Content-Type": "application/json",
    },
    timeout: `${maxDurationSeconds}s`,
    tags: { name: "dispatch-morning-briefs" },
  });

  check(res, {
    "status is 200": (r) => r.status === 200,
  });

  const body = res.json();
  check(body, {
    "no failed users": (b) => b.failed === 0,
  });

  console.log(
    `dispatch attempted=${body.attempted} sent=${body.sent} skipped=${body.skipped} failed=${body.failed} duration_ms=${res.timings.duration}`,
  );
}
