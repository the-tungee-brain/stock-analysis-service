def detect_strategy(option, stock):
    if option.side == "sell" and option.type == "call":
        if stock and stock.shares >= option.contracts * 100:
            return "covered_call"
        return "naked_call"

    if option.side == "sell" and option.type == "put":
        return "cash_secured_put"

    if option.side == "buy" and option.type == "call":
        return "long_call"

    if option.side == "buy" and option.type == "put":
        return "long_put"

    return "unknown"
