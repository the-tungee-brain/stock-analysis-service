from datetime import date
from unittest.mock import MagicMock

from app.adapters.finnhub.finnhub_adapter import FinnhubAdapter
from app.builders.finnhub_builder import FinnhubBuilder


def test_finnhub_adapter_company_news_query_params_match_curl_format():
    adapter = FinnhubAdapter(api_key="test-key")
    captured: dict[str, object] = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = dict(kwargs.get("params") or {})
        response = MagicMock()
        response.ok = True
        response.headers = {"Content-Type": "application/json"}
        response.json.return_value = []
        return response

    adapter.finnhub_client._session.get = fake_get
    adapter.get_company_news("AMZN", _from="2026-05-18", to="2026-05-25")

    assert "/company-news" in str(captured["url"])
    params = captured["params"]
    assert params["symbol"] == "AMZN"
    assert params["from"] == "2026-05-18"
    assert params["to"] == "2026-05-25"
    assert adapter.finnhub_client.api_key == "test-key"


def test_finnhub_builder_formats_company_news_dates_as_yyyy_mm_dd():
    adapter = MagicMock()
    adapter.get_company_news.return_value = []
    builder = FinnhubBuilder(finnhub_adapter=adapter)

    builder.get_company_news(
        symbol="AMZN",
        _from=date(2026, 5, 18),
        to=date(2026, 5, 25),
    )

    adapter.get_company_news.assert_called_once_with(
        symbol="AMZN",
        _from="2026-05-18",
        to="2026-05-25",
    )
