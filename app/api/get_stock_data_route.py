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

    index_is_datetime = isinstance(hist.index, pd.DatetimeIndex)

    if index_is_datetime:
        date_values = hist.index
    else:
        date_values = pd.to_datetime(hist.index, errors="coerce")
        if date_values.isna().all():
            date_values = hist.index

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in hist.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing expected OHLCV columns: {', '.join(missing)}",
        )

    data = []
    for i, (_, row) in enumerate(hist.iterrows()):
        dv = date_values[i]
        if isinstance(dv, pd.Timestamp):
            date_str = dv.isoformat()
        else:
            date_str = str(dv)

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
