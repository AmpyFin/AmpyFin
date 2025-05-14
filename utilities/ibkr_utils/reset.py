from ib_insync import IB, Stock, LimitOrder, AccountValue, Stock, MarketOrder, LimitOrder, Order, Trade
import yfinance as yf

# — new globals for your two new tags —
latest_cash_value = 0.0
latest_portfolio_value = 0.0
latest_available_funds = 0.0
latest_buying_power = 0.0

MAX_ORDER_VALUE = 5_000

def account_summary_handler(accountValue: AccountValue):
    """
    Only handle & print TotalCashValue, NetLiquidation,
    AvailableFunds and BuyingPower.
    """
    global latest_cash_value, latest_portfolio_value
    global latest_available_funds, latest_buying_power

    tag = accountValue.tag
    val = accountValue.value

    if tag == "TotalCashValue":
        latest_cash_value = val

    elif tag == "NetLiquidation":
        latest_portfolio_value = val

    elif tag == "AvailableFunds":
        latest_available_funds = val

    elif tag == "BuyingPower":
        latest_buying_power = val

    # all other tags are ignored

def print_account_summary():
    """
    Request only the four tags we care about, then print them.
    """
    ib.accountSummary()
    print(f"\n💰 TotalCashValue   → ${latest_cash_value}")
    print(f"📊 NetLiquidation   → ${latest_portfolio_value}")
    print(f"🪙 AvailableFunds   → ${latest_available_funds}")
    print(f"⚡ BuyingPower      → ${latest_buying_power}\n")

def reset_account(ib: IB):
    """Your existing cancel + liquidation logic…"""
    # — cancel open orders —
    open_orders = ib.openOrders()
    for order in open_orders:
        ib.cancelOrder(order)
    print(f"✅ Canceled {len(open_orders)} open orders.")

    # — liquidate positions in chunks —
    for pos in ib.positions():
        symbol = pos.contract.symbol
        size = abs(pos.position)
        action = "SELL" if pos.position > 0 else "BUY"
        contract = Stock(symbol, "SMART", "USD")

        try:
            last_price = yf.Ticker(symbol).history(period="1d")["Close"].iloc[-1]
            buffer = last_price * 0.01
            limit_price = (last_price - buffer) if action == "SELL" else (last_price + buffer)
        except Exception as e:
            print(f"⚠️ Error fetching price for {symbol}: {e}. Using fallback.")
            last_price, limit_price = 1.0, (0.01 if action=="SELL" else 9_999.99)

        max_chunk = int(MAX_ORDER_VALUE / last_price)
        remaining = size

        print(f"\n🔻 Closing {size} shares of {symbol}…")
        while remaining > 0:
            chunk = min(remaining, max_chunk)
            price = round(limit_price, 2)

            ib.reqMarketDataType(3)
            ib.reqMktData(contract, "", True)
            ib.sleep(1)

            order = LimitOrder(action, chunk, price)
            ib.placeOrder(contract, order)
            print(f"🔄 {action} {chunk} of {symbol} @ ${price}")
            ib.sleep(3)
            remaining -= chunk

        print(f"✅ All orders for {symbol} submitted.")
    print("\n✅ All liquidation orders submitted.")

def reset_account_with_summary(ib: IB):
    print("\n👀 Account summary BEFORE reset:")
    print_account_summary()

    reset_account(ib)

    print("👀 Account summary AFTER reset:")
    print_account_summary()

if __name__ == "__main__":
    ib = IB()
    ib.connect("127.0.0.1", 4002, clientId=1)

    ib.accountSummaryEvent += account_summary_handler
    reset_account_with_summary(ib)
    

    ib.disconnect()