from fastapi import FastAPI, HTTPException
import yfinance as yf

router = FastAPI()


@router.get("/get-stock-data")
def get_stock_data(symbol: str, period: str = "1mo", interval: str = "1d"):
    """
    period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    """
    ticker = yf.Ticker(symbol.upper())
    hist = ticker.history(period=period, interval=interval)

    if hist.empty:
        raise HTTPException(status_code=404, detail="No data found")

    hist = hist.reset_index()

    data = []
    for _, row in hist.iterrows():
        data.append(
            {
                "date": row["Date"].isoformat(),
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
