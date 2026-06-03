# Frontend Integration Guide (Web + iOS)

Clients consume **only** precomputed v1 APIs. No database or feature-level access.

## Base URL

```
https://<your-api-host>/api/v1
```

All responses include `"api_version": "v1"`.

Authentication: same Bearer token as other Tomcrest protected routes.

---

## Endpoints

| Endpoint | Poll interval | Purpose |
|----------|---------------|---------|
| `GET /rankings/top?limit=20` | 60–120s | Top movers table |
| `GET /portfolio/latest` | 60s | Allocation + risk dashboard |
| `GET /health` | 30–60s | Status badge, regime, pipeline freshness |

---

## Web (React / Next.js)

### Fetch example

```typescript
const API = process.env.NEXT_PUBLIC_API_URL;

async function fetchRankings(token: string) {
  const res = await fetch(`${API}/api/v1/rankings/top?limit=20`, {
    headers: { Authorization: `Bearer ${token}` },
    next: { revalidate: 60 },
  });
  if (!res.ok) throw new Error("rankings failed");
  return res.json();
}

async function fetchPortfolio(token: string) {
  const res = await fetch(`${API}/api/v1/portfolio/latest`, {
    headers: { Authorization: `Bearer ${token}` },
    next: { revalidate: 60 },
  });
  if (!res.ok) throw new Error("portfolio failed");
  return res.json();
}

async function fetchHealth(token: string) {
  const res = await fetch(`${API}/api/v1/health`, {
    headers: { Authorization: `Bearer ${token}` },
    next: { revalidate: 30 },
  });
  return res.json();
}
```

### UI mapping

| UI block | API fields |
|----------|------------|
| Top movers table | `items[].symbol`, `final_score`, `ml_probability`, `expected_excess_return` |
| Regime badge | `rankings.regime_id` or `health.regime_id` |
| Allocation pie/bar | `portfolio.holdings[].symbol`, `weight` |
| Risk dashboard | `metrics.beta_vs_spy`, `metrics.volatility`, `metrics.correlation_risk_score` |
| Sector chart | `metrics.sector_breakdown` or `risk_layer.sector_breakdown` |
| System status | `health.system_status` |

### Polling hook (client-side)

```typescript
useEffect(() => {
  const id = setInterval(() => {
    fetchPortfolio(token).then(setPortfolio).catch(console.error);
  }, 60_000);
  return () => clearInterval(id);
}, [token]);
```

---

## iOS (SwiftUI)

### Models

```swift
struct RankingsTopResponse: Codable {
    let apiVersion: String
    let timestamp: String
    let runId: String
    let asOfDate: String
    let regimeId: String?
    let items: [RankingItem]

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case timestamp, runId = "run_id", asOfDate = "as_of_date"
        case regimeId = "regime_id", items
    }
}

struct RankingItem: Codable, Identifiable {
    var id: String { symbol }
    let symbol: String
    let rank: Int
    let finalScore: Double
    let mlProbability: Double?
    let expectedExcessReturn: Double?

    enum CodingKeys: String, CodingKey {
        case symbol, rank
        case finalScore = "final_score"
        case mlProbability = "ml_probability"
        case expectedExcessReturn = "expected_excess_return"
    }
}

struct PortfolioLatestResponse: Codable {
    let apiVersion: String
    let timestamp: String
    let portfolioId: String
    let holdings: [PortfolioHolding]
    let metrics: PortfolioMetrics
    let riskLayer: RiskLayer?

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version", timestamp
        case portfolioId = "portfolio_id", holdings, metrics
        case riskLayer = "risk_layer"
    }
}

struct PortfolioHolding: Codable, Identifiable {
    var id: String { symbol }
    let symbol: String
    let weight: Double
    let scoreContribution: Double

    enum CodingKeys: String, CodingKey {
        case symbol, weight
        case scoreContribution = "score_contribution"
    }
}

struct PortfolioMetrics: Codable {
    let expectedReturn5d: Double
    let expectedExcess5d: Double
    let volatility: Double?
    let betaVsSpy: Double?
    let correlationRiskScore: Double?
    let sectorBreakdown: [String: Double]
    let turnoverEstimate: Double?

    enum CodingKeys: String, CodingKey {
        case expectedReturn5d = "expected_return_5d"
        case expectedExcess5d = "expected_excess_5d"
        case volatility
        case betaVsSpy = "beta_vs_spy"
        case correlationRiskScore = "correlation_risk_score"
        case sectorBreakdown = "sector_breakdown"
        case turnoverEstimate = "turnover_estimate"
    }
}
```

### Polling + cache

```swift
@MainActor
final class RankingPortfolioStore: ObservableObject {
    @Published private(set) var rankings: RankingsTopResponse?
    @Published private(set) var portfolio: PortfolioLatestResponse?

    private let cache = UserDefaults.standard

    func refresh(token: String) async {
        do {
            let r = try await APIClient.fetchRankings(token: token)
            rankings = r
            cache.set(try? JSONEncoder().encode(r), forKey: "rankings_v1")
        } catch {
            if rankings == nil, let data = cache.data(forKey: "rankings_v1") {
                rankings = try? JSONDecoder().decode(RankingsTopResponse.self, from: data)
            }
        }
    }
}
```

Poll every 60s in `.task` with `Task.sleep(nanoseconds: 60_000_000_000)`.

---

## Versioning

- Contract version is `api_version: "v1"` on every payload.
- Breaking changes require `v2` paths or explicit client opt-in.
- Clients should ignore unknown JSON fields.
