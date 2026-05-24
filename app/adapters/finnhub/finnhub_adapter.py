import finnhub


class FinnhubAdapter:
    def __init__(self, api_key: str):
        self.finnhub_client = finnhub.Client(api_key=api_key)

    def get_company_news(self, symbol: str, _from: str, to: str):
        return self.finnhub_client.company_news(symbol=symbol, _from=_from, to=to)

    def get_company_profile(self, symbol: str):
        return self.finnhub_client.company_profile2(symbol=symbol)

    def get_quote(self, symbol: str):
        return self.finnhub_client.quote(symbol=symbol)

    def get_company_earnings(self, symbol: str, limit: int | None = None):
        return self.finnhub_client.company_earnings(symbol=symbol, limit=limit)

    def get_earnings_calendar(
        self,
        _from: str,
        to: str,
        symbol: str = "",
        international: bool = False,
    ):
        return self.finnhub_client.earnings_calendar(
            _from=_from,
            to=to,
            symbol=symbol,
            international=international,
        )

    def get_transcripts_list(self, symbol: str):
        return self.finnhub_client.transcripts_list(symbol=symbol)

    def get_transcript(self, transcript_id: str):
        return self.finnhub_client.transcripts(_id=transcript_id)

    def get_press_releases(self, symbol: str, _from: str, to: str):
        return self.finnhub_client.press_releases(
            symbol=symbol, _from=_from, to=to
        )

    def get_stock_peers(self, symbol: str) -> list[str]:
        return self.finnhub_client.stock_peers(symbol=symbol)
