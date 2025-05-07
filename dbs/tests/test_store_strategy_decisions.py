import logging.config
import os
import sqlite3
import sys

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compute_store_strategy_decisions import (
    check_ticker_tables_exist,
    compute_and_store_strategy_decisions,
)
from log_config import LOG_CONFIG

# Get the current filename without extension
module_name = os.path.splitext(os.path.basename(__file__))[0]
log_filename = f"log/{module_name}.log"
LOG_CONFIG["handlers"]["file_dynamic"]["filename"] = log_filename

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def setup_test_db(tmp_path):
    """Fixture to set up and tear down test databases."""
    TEST_DB_PATH = tmp_path / "test.db"
    TEST_STRATEGY_DECISIONS_DB_PATH = tmp_path / "strategy.db"

    # TEST_DB_PATH = os.path.join("dbs", "tests", "test_price_data.db")
    # TEST_STRATEGY_DECISIONS_DB_PATH = os.path.join(
    #     "dbs", "tests", "test_strategy_decisions_data.db"
    # )

    ticker1 = "APP"
    ticker2 = "MSFT"
    ticker3 = "GOOG"

    # create price data df
    dates = pd.date_range(end="2025-04-27", periods=5, freq="D")
    data = {
        "Ticker": ["APP"] * 5,
        "Open": [100.1, 102, 101, 103, 104],
        "High": [105.1, 106, 104, 107, 108],
        "Low": [99.1, 100, 98, 101, 102],
        "Close": [104.1, 103, 102, 106, 107],
        "Volume": [500, 600, 550, 650, 700],
    }
    price_df = pd.DataFrame(data, index=dates)
    price_df.index.name = "Date"

    # setup price db
    with sqlite3.connect(TEST_DB_PATH) as conn:
        try:
            price_df.to_sql(
                "APP",
                conn,
                if_exists="replace",
                dtype={"Date": "DATE PRIMARY KEY NOT NULL"},
            )
        except Exception as e:
            logger.error(
                f"""Error saving "APP" OHLCV price data to
                      {TEST_DB_PATH}: {e}"""
            )

    yield TEST_DB_PATH, TEST_STRATEGY_DECISIONS_DB_PATH, ticker1, ticker2, ticker3

    """
    Cleanup after tests
    tmp_path is automatically cleaned up.
    """


def test_check_ticker_tables_exist(setup_test_db):
    TEST_DB_PATH, _, ticker1, ticker2, ticker3 = setup_test_db

    tickers = [ticker1, ticker2, ticker3]
    result = check_ticker_tables_exist(TEST_DB_PATH, tickers)

    assert result[ticker1] is True
    assert result[ticker2] is False
    assert result[ticker3] is False


def test_compute_and_store_strategy_decisions_basic(setup_test_db):
    TEST_DB_PATH, TEST_STRATEGY_DECISIONS_DB_PATH, ticker1, _, _ = setup_test_db

    def mock_strategy(df):
        out = pd.DataFrame(index=df.index)
        out["mock_signal"] = 1
        return out

    ticker_list = [ticker1]
    strategies = [mock_strategy]
    compute_and_store_strategy_decisions(
        TEST_DB_PATH,
        TEST_STRATEGY_DECISIONS_DB_PATH,
        ticker_list,
        strategies,
        logger,
    )

    # Check that the strategy DB has the expected table and data
    with sqlite3.connect(TEST_STRATEGY_DECISIONS_DB_PATH) as con:
        df = pd.read_sql_query(f"SELECT * FROM '{ticker1}'", con, index_col="Date")

    assert "mock_signal" in df.columns
    assert df.iloc[0]["mock_signal"] == 1


def test_compute_and_store_strategy_decisions_missing_price_table(
    setup_test_db,
):
    TEST_DB_PATH, TEST_STRATEGY_DECISIONS_DB_PATH, _, _, ticker3 = setup_test_db

    ticker_list = [ticker3]
    strategies = [lambda df: pd.DataFrame({"signal": [1]}, index=df.index)]

    compute_and_store_strategy_decisions(
        TEST_DB_PATH,
        TEST_STRATEGY_DECISIONS_DB_PATH,
        ticker_list,
        strategies,
        logger,
    )

    with sqlite3.connect(TEST_STRATEGY_DECISIONS_DB_PATH) as con:
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()

    assert len(tables) == 0  # No tables created


def test_compute_and_store_strategy_decisions_empty_ticker_list(setup_test_db):
    TEST_DB_PATH, TEST_STRATEGY_DECISIONS_DB_PATH, _, _, _ = setup_test_db

    ticker_list = []
    strategies = [lambda df: pd.DataFrame({"signal": [1]}, index=df.index)]

    compute_and_store_strategy_decisions(
        TEST_DB_PATH,
        TEST_STRATEGY_DECISIONS_DB_PATH,
        ticker_list,
        strategies,
        logger,
    )

    with sqlite3.connect(TEST_STRATEGY_DECISIONS_DB_PATH) as con:
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()

    assert len(tables) == 0


def test_strategy_exception_handling(setup_test_db):
    TEST_DB_PATH, TEST_STRATEGY_DECISIONS_DB_PATH, ticker1, _, _ = setup_test_db

    def bad_strategy(df):
        raise ValueError("fail")

    ticker_list = [ticker1]
    strategies = [bad_strategy]

    with pytest.raises(ValueError):
        compute_and_store_strategy_decisions(
            TEST_DB_PATH,
            TEST_STRATEGY_DECISIONS_DB_PATH,
            ticker_list,
            strategies,
            logger,
        )
