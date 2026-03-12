mock_data = {
    "accountId": "123",
    "positions": [
        {
            "shortQuantity": 0,
            "longQuantity": 1,
            "averagePrice": 130.53,
            "marketValue": 37.80,
            "currentDayProfitLoss": 9273.66,
            "instrument": {
                "symbol": "MSFT",
                "assetType": "OPTION",
                "putCall": "CALL",
                "strikePrice": 435,
                "expirationDate": "2026-12-18",
            },
        },
    ],
}

"""
{
            "shortQuantity": 0,
            "longQuantity": 100,
            "averagePrice": 170,
            "marketValue": 18250,
            "currentDayProfitLoss": 100,
            "instrument": {"symbol": "AAPL", "assetType": "EQUITY"},
        },
        {
            "shortQuantity": 1,
            "longQuantity": 0,
            "averagePrice": 3.2,
            "marketValue": -150,
            "currentDayProfitLoss": 50,
            "instrument": {
                "symbol": "AAPL_041726C190",
                "assetType": "OPTION",
                "putCall": "CALL",
                "strikePrice": 190,
                "expirationDate": "2026-04-17",
            },
        },
"""
