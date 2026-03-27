from fastapi import APIRouter, HTTPException, Query
import yfinance as yf
import pandas as pd

router = APIRouter()


@router.get("/get-stock-data")
def get_stock_data(
    symbol: str,
    period: str = Query("1mo", description="1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"),
    interval: str = Query(
        "1d", description="1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo"
    ),
):
    ticker = yf.Ticker(symbol.upper())
    hist = ticker.history(period=period, interval=interval)

    if hist.empty:
        raise HTTPException(status_code=404, detail="No data found")

    if hist.index.name is None:
        hist.index.name = "Date"
    hist = hist.reset_index()

    if "Date" not in hist.columns:
        raise HTTPException(
            status_code=500, detail="Missing Date column in history data"
        )

    data = []
    for _, row in hist.iterrows():
        date_value = row["Date"]
        if isinstance(date_value, pd.Timestamp):
            date_str = date_value.isoformat()
        else:
            date_str = str(date_value)

        data.append(
            {
                "date": date_str,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
        )

    info = ticker.info
    return {
        "symbol": symbol.upper(),
        "name": info.get("longName", symbol.upper()),
        "currency": info.get("currency", "USD"),
        "data": data,
    }
