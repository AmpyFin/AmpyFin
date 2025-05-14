from ib_insync import IB, AccountValue, util, Trade, Fill, Stock, MarketOrder
import signal

# ─── Global cache for account values ──────────────────────────────────────────
latest_cash_value = 0.0
latest_portfolio_value = 0.0

def account_summary_handler(accountValue: AccountValue):
    global latest_cash_value, latest_portfolio_value
    if accountValue.tag == "AvailableFunds":
        latest_cash_value = float(accountValue.value)
        print(f"[AccountSummary] AvailableFunds updated → ${latest_cash_value:,.2f}")
    elif accountValue.tag == "NetLiquidation":
        latest_portfolio_value = float(accountValue.value)
        print(f"[AccountSummary] NetLiquidation updated → ${latest_portfolio_value:,.2f}")

def on_exec_details(trade: Trade, fill:Fill):
    print(f"[ExecDetails] {trade.contract.symbol} {trade.order.action} {trade.order.totalQuantity} @ {fill.execution.price}")
    print(f"  {fill.execution.time} - {fill.execution.cumQty} @ {fill.execution.avgPrice}")
    print(f"  {trade.orderStatus.status} - {trade.orderStatus.remaining}")



def main():
    util.startLoop()          # optional: only if you want notebook/Qt integration
    ib = IB()
    ib.connect('127.0.0.1', 4002, clientId=4)

    # 1) wire up your handler
    ib.accountValueEvent += account_summary_handler

    # 2) explicitly subscribe to account‐value streams
    ib.reqAccountUpdates()

    ib.execDetailsEvent.clear()
    ib.execDetailsEvent += on_exec_details

    contract = Stock('KULR', "SMART", "USD")
    ib.qualifyContracts(contract)
    order    = MarketOrder('SELL', 10, outsideRth=True)
    trade = ib.placeOrder(contract, order)
    while not trade.isDone():
        ib.sleep(0.1)

    # 3) (optional) grab a one‐time snapshot of all tags immediately
    ib.accountSummary()       # legacy snapshot
    # or ib.reqAccountSummary(1, 'All', 'AvailableFunds,NetLiquidation')

    # 4) keep the script alive and processing events
    #    cleanly exit on Ctrl+C
    signal.signal(signal.SIGINT, lambda *_: ib.disconnect())
    ib.run()

if __name__ == '__main__':
    main()
