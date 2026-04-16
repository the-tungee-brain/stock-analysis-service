import pandas as pd
from typing import Protocol


class MarketDataAdapter(Protocol):
    def get_daily_closes_1y(self, symbol: str) -> pd.Series: ...
