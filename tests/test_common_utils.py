import sys
import copy
import pytest
import pandas as pd
import sqlite3
from datetime import datetime

# adjust this import to wherever your function lives
from utilities.common_utils import get_ndaq_tickers, fetch_price_from_db, fetch_strategy_decisions, execute_trade

# Adjust this import path to wherever execute_trade lives
import utilities.common_utils as cu

#### get_ndaq_tickers
def test_get_ndaq_tickers_success(monkeypatch):
    # Prepare a dummy DataFrame with a "Ticker" column
    dummy_df = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "GOOG"]})
    # Monkey‑patch pandas.read_html to return a list whose 5th element is our dummy_df
    monkeypatch.setattr(pd, "read_html", lambda url: [None, None, None, None, dummy_df])

    # Call the function
    result = get_ndaq_tickers()

    # Assert we got back exactly our tickers
    assert result == ["AAPL", "MSFT", "GOOG"]


def test_get_ndaq_tickers_short_list(monkeypatch):
    # Simulate read_html returning too few tables
    monkeypatch.setattr(pd, "read_html", lambda url: [])
    # Expect an IndexError because tables[4] doesn’t exist
    with pytest.raises(IndexError):
        get_ndaq_tickers()


def test_get_ndaq_tickers_missing_column(monkeypatch):
    # Simulate a DataFrame without the "Ticker" column
    bad_df = pd.DataFrame({"Symbol": ["X", "Y", "Z"]})
    monkeypatch.setattr(pd, "read_html", lambda url: [None, None, None, None, bad_df])
    # Expect a KeyError because "Ticker" isn’t in bad_df
    with pytest.raises(KeyError):
        get_ndaq_tickers()

#### fetch_price_from_db
@pytest.fixture
def in_memory_db(monkeypatch):
    # 1) Create an in‑memory SQLite DB
    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()

    # 2) Create two ticker tables with the same schema as your real DB
    for ticker in ("AAPL", "GOOG"):
        cur.execute(f"""
            CREATE TABLE "{ticker}" (
                Date TEXT,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                "Adj Close" REAL,
                Volume REAL
            )
        """)

    # 3) Insert sample rows for AAPL
    aapl_rows = [
        ("2020-01-01", 100, 101,  99, 100.5, 100.5, 1000),
        ("2020-01-02", 101, 102, 100, 101.5, 101.5, 2000),
        ("2020-02-01", 102, 103, 101, 102.5, 102.5, 3000),
    ]
    cur.executemany('INSERT INTO "AAPL" VALUES (?,?,?,?,?,?,?)', aapl_rows)

    # 4) Insert a single sample row for GOOG
    goog_rows = [
        ("2020-01-15", 1500, 1510, 1490, 1505, 1505, 1500),
    ]
    cur.executemany('INSERT INTO "GOOG" VALUES (?,?,?,?,?,?,?)', goog_rows)

    conn.commit()

    # 5) Monkey‑patch sqlite3.connect so your function uses this in‑memory DB
    monkeypatch.setattr(sqlite3, "connect", lambda db_path: conn)

    yield conn
    conn.close()


def test_fetch_price_single_ticker(in_memory_db):
    start = pd.Timestamp("2020-01-01")
    end   = pd.Timestamp("2020-01-31")

    df = fetch_price_from_db(start, end, ["AAPL"])

    # Should only get the two Jan 2020 rows
    assert set(df["Date"]) == {"2020-01-01", "2020-01-02"}
    # Every row should carry the correct ticker
    assert all(df["Ticker"] == "AAPL")

    # Verify all expected columns are present, in order
    assert list(df.columns) == [
        "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"
    ]


def test_fetch_price_multiple_tickers(in_memory_db):
    start = pd.Timestamp("2020-01-01")
    end   = pd.Timestamp("2020-02-28")

    df = fetch_price_from_db(start, end, ["AAPL", "GOOG"])

    # We inserted 3 rows for AAPL and 1 for GOOG, but AAPL's Feb-01 row is within range:
    assert len(df) == 4
    # Both tickers should appear
    assert set(df["Ticker"]) == {"AAPL", "GOOG"}


def test_fetch_price_no_data(in_memory_db):
    # A date range outside of any inserted data
    start = pd.Timestamp("2021-01-01")
    end   = pd.Timestamp("2021-12-31")

    df = fetch_price_from_db(start, end, ["AAPL", "GOOG"])
    assert df.empty

#### fetch_strategy_decisions
@pytest.fixture
def in_memory_strategy_db(monkeypatch):
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    # Define two strategies as functions so we can test callable handling
    def strat_alpha():
        pass

    def strat_beta():
        pass

    # Create tables for two tickers with Date + two strategy columns
    for ticker in ("AAPL", "GOOG"):
        cur.execute(f"""
            CREATE TABLE "{ticker}" (
                Date TEXT,
                "{strat_alpha.__name__}" INTEGER,
                "{strat_beta.__name__}" INTEGER
            )
        """)

    # Insert sample data for AAPL
    aapl_rows = [
        ("2021-01-01",  1, -1),
        ("2021-02-01",  0,  1),
        ("2021-03-01", -1,  0),
    ]
    cur.executemany(
        f'INSERT INTO "AAPL" VALUES (?,?,?)',
        aapl_rows
    )

    # Insert sample data for GOOG
    goog_rows = [
        ("2021-01-15",  1,  1),
        ("2021-04-01",  0,  0),
    ]
    cur.executemany(
        f'INSERT INTO "GOOG" VALUES (?,?,?)',
        goog_rows
    )

    conn.commit()

    # Monkey‑patch sqlite3.connect so the function uses our in-memory DB
    monkeypatch.setattr(sqlite3, "connect", lambda db_path: conn)

    # Yield the strategy functions so tests can reference their __name__
    yield (conn, [strat_alpha, strat_beta])
    conn.close()


def test_fetch_strategy_single_ticker(in_memory_strategy_db):
    conn, strategies = in_memory_strategy_db
    strat_names = [s.__name__ for s in strategies]

    # Query only AAPL for Jan–Feb 2021
    start = pd.Timestamp("2021-01-01")
    end   = pd.Timestamp("2021-02-28")

    df = fetch_strategy_decisions(start, end, ["AAPL"], strategies)

    # Should only get the two matching rows for AAPL
    assert set(df["Date"]) == {"2021-01-01", "2021-02-01"}
    # All rows carry ticker "AAPL"
    assert all(df["Ticker"] == "AAPL")
    # Check strategy columns exist
    for name in strat_names:
        assert name in df.columns

    # Verify that the values match what we inserted
    row_map = df.set_index("Date").to_dict(orient="index")
    assert row_map["2021-01-01"][strat_names[0]] == 1
    assert row_map["2021-02-01"][strat_names[1]] == 1


def test_fetch_strategy_multiple_tickers(in_memory_strategy_db):
    conn, strategies = in_memory_strategy_db
    strat_names = [s.__name__ for s in strategies]

    # Query both AAPL and GOOG for all of 2021
    start = pd.Timestamp("2021-01-01")
    end   = pd.Timestamp("2021-12-31")

    df = fetch_strategy_decisions(start, end, ["AAPL", "GOOG"], strategies)

    # We inserted 3 rows for AAPL and 2 for GOOG
    assert len(df) == 5
    # Both tickers should appear
    assert set(df["Ticker"]) == {"AAPL", "GOOG"}
    # All strategy columns present
    for name in strat_names:
        assert name in df.columns


def test_fetch_strategy_no_data(in_memory_strategy_db):
    conn, strategies = in_memory_strategy_db

    # Use a date range outside any inserted data
    start = pd.Timestamp("2022-01-01")
    end   = pd.Timestamp("2022-12-31")

    df = fetch_strategy_decisions(start, end, ["AAPL", "GOOG"], strategies)
    assert df.empty


def test_fetch_strategy_with_string_names(monkeypatch, in_memory_strategy_db):
    """
    Verify that passing strategy names as strings works the same as callables.
    """
    conn, callables = in_memory_strategy_db
    # Convert callables to their names
    strat_names = [s.__name__ for s in callables]

    # Monkey-patch sqlite3.connect here too
    monkeypatch.setattr(sqlite3, "connect", lambda db_path: conn)

    start = pd.Timestamp("2021-01-01")
    end   = pd.Timestamp("2021-03-31")

    # Pass strings instead of functions
    df = fetch_strategy_decisions(start, end, ["AAPL"], strat_names)

    # Should fetch exactly the three AAPL rows we inserted
    assert len(df) == 3
    assert all(df["Ticker"] == "AAPL")
    # Strategy columns present
    for name in strat_names:
        assert name in df.columns


#### compute_trade_quantities
@pytest.fixture(autouse=True)
def reset_trade_limit():
    """Ensure trade_asset_limit is reset before each test."""
    # default to 100% of portfolio
    cu.trade_asset_limit = 1.0
    yield
    cu.trade_asset_limit = 1.0


def test_buy_limited_by_max_investment():
    # only 10% of total portfolio can be invested
    cu.trade_asset_limit = 0.1
    action, qty = cu.compute_trade_quantities(
        action="Buy",
        current_price=100.0,
        account_cash=1_000_000.0,      # plenty of cash
        portfolio_qty=0,
        total_portfolio_value=1_000.0, # max_investment = 100.0
    )
    assert action == "buy"
    # max_investment//price = 1, cash//price = 10000 → qty = 1
    assert qty == 1


def test_buy_limited_by_cash():
    # allow 100% of portfolio, but cash is the limiter
    cu.trade_asset_limit = 1.0
    action, qty = cu.compute_trade_quantities(
        action="Buy",
        current_price=50.0,
        account_cash=120.0,            # can only afford 2 shares
        portfolio_qty=0,
        total_portfolio_value=10_000.0,# max_investment = 10_000
    )
    assert action == "buy"
    # max_investment//price = 200, cash//price = 2 → qty = 2
    assert qty == 2


def test_sell_half_of_portfolio():
    cu.trade_asset_limit = 1.0
    action, qty = cu.compute_trade_quantities(
        action="Sell",
        current_price=100.0,
        account_cash=0.0,
        portfolio_qty=10,
        total_portfolio_value=10_000.0,
    )
    assert action == "sell"
    # half of 10 is 5 → qty = 5
    assert qty == 5


def test_sell_minimum_one_share():
    cu.trade_asset_limit = 1.0
    action, qty = cu.compute_trade_quantities(
        action="Sell",
        current_price=100.0,
        account_cash=0.0,
        portfolio_qty=1,
        total_portfolio_value=10_000.0,
    )
    assert action == "sell"
    # half of 1 is 0.5 → int(...) = 0 → max(1,0) = 1 → qty = 1
    assert qty == 1


def test_sell_no_shares_results_in_hold():
    cu.trade_asset_limit = 1.0
    action, qty = cu.compute_trade_quantities(
        action="Sell",
        current_price=100.0,
        account_cash=0.0,
        portfolio_qty=0,
        total_portfolio_value=10_000.0,
    )
    assert action == "hold"
    assert qty == 0


@pytest.mark.parametrize("act", ["Hold", "hold", "UNKNOWN"])
def test_unknown_or_hold_action(act):
    cu.trade_asset_limit = 1.0
    action, qty = cu.compute_trade_quantities(
        action=act,
        current_price=100.0,
        account_cash=1_000.0,
        portfolio_qty=10,
        total_portfolio_value=10_000.0,
    )
    assert action == "hold"
    assert qty == 0

#### execute_trade

# Dummy strategy class so strategy.__name__ works
class DummyStrategy:
    pass


@pytest.fixture(autouse=True)
def patch_limits(monkeypatch):
    # from your specs:
    # train_rank_liquidity_limit = 15000
    # train_rank_asset_limit     = 0.3
    monkeypatch.setattr(cu, "train_rank_liquidity_limit", 15000)
    monkeypatch.setattr(cu, "train_rank_asset_limit", 0.3)
    yield


@pytest.fixture
def base_simulator_and_points():
    name = DummyStrategy.__name__
    simulator = {
        name: {
            "holdings": {},
            "amount_cash": 50_000,
            "total_trades": 0,
            "successful_trades": 0,
            "neutral_trades": 0,
            "failed_trades": 0,
            "portfolio_value": 50_000,
        }
    }
    points = {name: 0}
    return simulator, points


def test_buy_success(base_simulator_and_points):
    sim, pts = base_simulator_and_points
    original_pts = pts.copy()
    strategy_name = DummyStrategy.__name__
    # qty*price = 10*100 = 1_000 → 1_000/50_000 = 0.02 < 0.3 asset limit
    # cash 50_000 > 15_000 liquidity limit
    sim_out, pts_out = execute_trade(
        decision="buy",
        qty=10,
        ticker="AAPL",
        current_price=100.0,
        strategy_name=strategy_name,
        trading_simulator=sim,
        points=pts,
        time_delta=sys.float_info.min,
        portfolio_qty=0,
        total_portfolio_value=50_000,
    )

    # amount_cash reduced by 10*100
    assert sim_out[strategy_name]["amount_cash"] == 49_000
    # holdings created with correct qty & price
    assert sim_out[strategy_name]["holdings"] == {
        "AAPL": {"quantity": 10, "price": 100.0}
    }
    # total_trades incremented
    assert sim_out[strategy_name]["total_trades"] == 1
    # points untouched
    assert pts_out == original_pts


# def test_buy_fails_when_insufficient_liquidity(base_simulator_and_points):
#     sim, pts = base_simulator_and_points
#     original_sim = copy.deepcopy(sim)
#     original_pts = pts.copy()

#     strategy_name = DummyStrategy.__name__
#     # lower cash to exactly the limit
#     sim[strategy_name]["amount_cash"] = 15_000
    
#     sim_out, pts_out = execute_trade(
#         decision="buy",
#         qty=10,
#         ticker="AAPL",
#         current_price=100.0,
#         strategy_name=strategy_name,
#         trading_simulator=sim,
#         points=pts,
#         time_delta=sys.float_info.min,
#         portfolio_qty=0,
#         total_portfolio_value=50_000,
#     )

#     # no change
#     assert sim_out == original_sim
#     assert pts_out == original_pts


def test_buy_fails_when_exceeds_asset_limit(base_simulator_and_points):
    sim, pts = base_simulator_and_points
    original_sim = copy.deepcopy(sim)
    original_pts = pts.copy()
    strategy_name = DummyStrategy.__name__
    # qty*price = 150*100 = 15_000 → 15_000/50_000 = 0.3 not < 0.3 → fail
    sim_out, pts_out = execute_trade(
        decision="buy",
        qty=150,
        ticker="AAPL",
        current_price=100.0,
        strategy_name=strategy_name,
        trading_simulator=sim,
        points=pts,
        time_delta=sys.float_info.min,
        portfolio_qty=0,
        total_portfolio_value=50_000,
    )

    assert sim_out == original_sim
    assert pts_out == original_pts


def test_sell_success_invokes_update(monkeypatch, base_simulator_and_points):
    sim, pts = base_simulator_and_points
    strategy_name = DummyStrategy.__name__

    # preload holdings so sell branch triggers
    sim[strategy_name]["holdings"] = {"AAPL": {"quantity": 20, "price": 50.0}}

    # stub out update_points_and_trades
    called = {}

    def fake_update(strategy_name, ratio, price, sim_arg, pts_arg, delta, ticker, qty):
        called["args"] = {
            "strategy_name": strategy_name,
            "ratio": ratio,
            "price": price,
            "sim": sim_arg,
            "pts": pts_arg,
            "delta": delta,
            "ticker": ticker,
            "qty": qty,
        }
        # return a clearly different simulator & points
        return {"X": 999}, {"Y": 888}

    monkeypatch.setattr(cu, "update_points_and_trades", fake_update)
    strategy_name = DummyStrategy.__name__
    sim_out, pts_out = execute_trade(
        decision="sell",
        qty=5,
        ticker="AAPL",
        current_price=100.0,
        strategy_name=strategy_name,
        trading_simulator=sim,
        points=pts,
        time_delta=sys.float_info.min,
        portfolio_qty=20,
        total_portfolio_value=50_000,
    )

    # stub was called once with correct ratio and args
    args = called["args"]
    assert args["strategy_name"] is DummyStrategy.__name__
    assert args["ratio"] == pytest.approx(100.0 / 50.0)
    assert args["price"] == 100.0
    assert args["delta"] == sys.float_info.min
    assert args["ticker"] == "AAPL"
    assert args["qty"] == 5

    # execute_trade returns the stub's simulator & points (swapped)
    assert sim_out == {"Y": 888}
    assert pts_out == {"X": 999}


def test_sell_fails_with_insufficient_holdings(monkeypatch, base_simulator_and_points):
    sim, pts = base_simulator_and_points
    name = DummyStrategy.__name__

    # holdings < qty
    sim[name]["holdings"] = {"AAPL": {"quantity": 2, "price": 50.0}}

    # make sure stub would error if called
    def should_not_be_called(*args, **kwargs):
        raise AssertionError("update_points_and_trades was called but shouldn't be")

    monkeypatch.setattr(cu, "update_points_and_trades", should_not_be_called)

    sim_before = copy.deepcopy(sim)
    pts_before = pts.copy()
    strategy_name = DummyStrategy.__name__
    sim_out, pts_out = execute_trade(
        decision="sell",
        qty=5,
        ticker="AAPL",
        current_price=100.0,
        strategy_name=strategy_name,
        trading_simulator=sim,
        points=pts,
        time_delta=sys.float_info.min,
        portfolio_qty=2,
        total_portfolio_value=50_000,
    )

    assert sim_out == sim_before
    assert pts_out == pts_before


@pytest.mark.parametrize("decision", ["hold", "unknown", "HOLD"])
def test_hold_or_unknown_decision_does_nothing(monkeypatch, base_simulator_and_points, decision):
    sim, pts = base_simulator_and_points

    # stub out update_points_and_trades to catch any unexpected calls
    monkeypatch.setattr(cu, "update_points_and_trades", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    sim_before = copy.deepcopy(sim)
    pts_before = pts.copy()
    strategy_name = DummyStrategy.__name__
    sim_out, pts_out = execute_trade(
        decision=decision,
        qty=1,
        ticker="AAPL",
        current_price=100.0,
        strategy_name=strategy_name,
        trading_simulator=sim,
        points=pts,
        time_delta=sys.float_info.min,
        portfolio_qty=0,
        total_portfolio_value=50_000,
    )

    assert sim_out == sim_before
    assert pts_out == pts_before
