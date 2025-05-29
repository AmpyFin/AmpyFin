from ib_insync import IB, AccountValue

# ─── Global cache for account values ──────────────────────────────────────────
latest_cash_value = 0.0
latest_portfolio_value = 0.0

def account_summary_handler(accountValue: AccountValue):
    if accountValue.tag == "TotalCashValue":
        latest_cash_value = float(accountValue.value)
        print(f"[AccountSummary] TotalCashValue updated → ${latest_cash_value:,.2f}")
    elif accountValue.tag == "NetLiquidation":
        latest_portfolio_value = float(accountValue.value)
        print(f"[AccountSummary] NetLiquidation updated → ${latest_portfolio_value:,.2f}")

def main():
    # 1) establish single IB connection
    ib = IB()
    ib.connect("127.0.0.1", 4002, clientId=1)

    #  #  ── AFTER connecting, subscribe to executions ────────────────────
    # ib.execDetailsEvent += on_exec
    # ib.reqExecutions()                # ← this kicks off the stream
    # # ─────────────────────────────────────────────────────────────────


    # 2) subscribe to all account summary updates
    ib.accountSummaryEvent += account_summary_handler
   
    # 5) on shutdown
    # ib.disconnect()

if __name__ == "__main__":
    main()
