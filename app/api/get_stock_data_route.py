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
    """
    Fetch OHLCV data from Yahoo Finance.

    period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    """
    ticker = yf.Ticker(symbol.upper())
    hist = ticker.history(period=period, interval=interval)

    if hist.empty:
        raise HTTPException(status_code=404, detail="No data found")

    if hist.index.name is None:
        hist.index.name = "Date"

    hist = hist.reset_index()

    cols = {c.lower(): c for c in hist.columns}
    date_col = cols.get("date")
    open_col = cols.get("open")
    high_col = cols.get("high")
    low_col = cols.get("low")
    close_col = cols.get("close")
    volume_col = cols.get("volume")

    if not all([date_col, open_col, high_col, low_col, close_col, volume_col]):
        raise HTTPException(
            status_code=500,
            detail="Unexpected data format from yfinance (missing expected columns)",
        )

    data = []
    for _, row in hist.iterrows():
        date_value = row[date_col]
        if isinstance(date_value, pd.Timestamp):
            date_str = date_value.isoformat()
        else:
            date_str = str(date_value)

        data.append(
            {
                "date": date_str,
                "open": float(row[open_col]),
                "high": float(row[high_col]),
                "low": float(row[low_col]),
                "close": float(row[close_col]),
                "volume": int(row[volume_col]),
            }
        )

    info = ticker.info
    return {
        "symbol": symbol.upper(),
        "name": info.get("longName", symbol.upper()),
        "currency": info.get("currency", "USD"),
        "data": data,
    }
