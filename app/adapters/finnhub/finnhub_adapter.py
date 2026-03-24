import finnhub


class FinnhubAdapter:
    def __init__(self, api_key: str):
        self.finnhub_client = finnhub.Client(api_key=api_key)

    def get_company_news(self, symbol: str, _from: str, to: str):
        return self.finnhub_client.company_news(symbol=symbol, _from=_from, to=to)
