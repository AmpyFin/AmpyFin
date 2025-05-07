import logging.config
import os
import sqlite3
import sys
from unittest.mock import patch

import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log_config import LOG_CONFIG
from store_price_data import download_OHLCV_from_yf, store_OHLCV_in_db

# Configure logging
module_name = os.path.splitext(os.path.basename(__file__))[0]
log_filename = f"log/{module_name}.log"
LOG_CONFIG["handlers"]["file_dynamic"]["filename"] = log_filename

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def test_data(tmp_path):
    """Fixture for setting up test data."""
    TEST_DB_PATH = tmp_path / "test.db"

    dates = pd.date_range(end="2025-04-27", periods=5, freq="D")
    data = {
        "Ticker": ["APP"] * 5,
        "Open": [100.1, 102, 101, 103, 104],
        "High": [105.1, 106, 104, 107, 108],
        "Low": [99.1, 100, 98, 101, 102],
        "Close": [104.1, 103, 102, 106, 107],
        "Volume": [500.0, 600, 550, 650, 700],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    yield df, TEST_DB_PATH
    # return df


def test_download_OHLCV_empty_input():
    """Test downloading OHLCV data with empty inputs."""
    ticker_list = []
    expected = pd.DataFrame()
    actual = download_OHLCV_from_yf(ticker_list, logger)

    assert actual is not None, "download_OHLCV_from_yf returned None"
    assert actual.empty
    assert expected.index.tolist() == actual.index.tolist()
    assert expected.columns.tolist() == actual.columns.tolist()


def test_download_OHLCV_invalid_ticker(caplog):
    """Test that invalid ticker produces error log"""
    ticker_list = ["INVALID_TICKER_123"]

    with caplog.at_level(logging.ERROR):
        actual = download_OHLCV_from_yf(ticker_list, logger)

    # Assert that an error was logged - check for actual error message
    assert (
        "YFTzMissingError" in caplog.text
    ), "Expected YFTzMissingError error in log messages"

    # Assert the returned DataFrame
    assert actual is not None
    assert actual.empty if actual is not None else True


def test_download_OHLCV_network_error():
    """Test that download_OHLCV_from_yf handles network errors gracefully."""
    with patch("store_price_data.yf.download", side_effect=Exception("Network error")):
        ticker_list = ["AAPL"]
        result = download_OHLCV_from_yf(ticker_list, logger)
        assert result is not None
        assert result.empty


def test_ticker_list_empty(test_data):
    """Test if ticker_list is empty."""
    df, _ = test_data
    ticker_list = []
    percentage_of_tickers_saved, tickers_with_no_data = store_OHLCV_in_db(
        df, ticker_list, ":memory:", logger
    )

    assert percentage_of_tickers_saved == 0
    assert tickers_with_no_data == []


def test_no_price_data_for_ticker(test_data):
    """Test if no price data for ticker list."""
    df, _ = test_data
    ticker_list = ["MSFT"]
    percentage_of_tickers_saved, tickers_with_no_data = store_OHLCV_in_db(
        df, ticker_list, ":memory:", logger
    )

    assert percentage_of_tickers_saved == 0
    assert tickers_with_no_data == ["MSFT"]


def test_some_price_data_for_ticker(test_data):
    """Test if some price data is available for a given ticker list."""
    df, _ = test_data
    ticker_list = ["APP", "MSFT"]
    percentage_of_tickers_saved, tickers_with_no_data = store_OHLCV_in_db(
        df, ticker_list, ":memory:", logger
    )

    assert percentage_of_tickers_saved == 50
    assert tickers_with_no_data == ["MSFT"]


# @pytest.mark.skip(reason="not ready")
def test_sqlite_db_structure(test_data):
    """Test ticker table exists and correct table structure."""
    df, TEST_DB_PATH = test_data
    # TEST_DB_PATH = "test_price_data.db"

    _percentage_of_tickers_saved, _tickers_with_no_data = store_OHLCV_in_db(
        df, ["APP"], TEST_DB_PATH, logger
    )

    with sqlite3.connect(TEST_DB_PATH) as con_price_data:
        ticker_price_history = pd.read_sql_query(
            "SELECT * FROM 'APP'", con_price_data, index_col="Date"
        )

    expected_price_df = df.copy()
    expected_price_df.drop(["Ticker"], axis=1, inplace=True)
    expected_price_df.index = expected_price_df.index.strftime("%Y-%m-%d")

    pdt.assert_frame_equal(ticker_price_history, expected_price_df)

    """Check table exists"""
    with sqlite3.connect(TEST_DB_PATH) as con_price_data:
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        actual_table_list = pd.read_sql_query(query, con_price_data)["name"].tolist()

    assert actual_table_list == ["APP"]

    """Check table structure"""
    with sqlite3.connect(TEST_DB_PATH) as con_price_data:
        query = "PRAGMA table_info(APP)"
        actual_PRAGMA_table = pd.read_sql_query(query, con_price_data, index_col="cid")

    # expected_pragma_table = pd.DataFrame(
    #     {
    #         "name": ["Date", "Open", "High", "Low", "Close", "Volume"],
    #         "type": ["DATE", "REAL", "REAL", "REAL", "REAL", "REAL"],
    #         "notnull": [1, 0, 0, 0, 0, 0],
    #         "dflt_value": [None, None, None, None, None, None],
    #         "pk": [1, 0, 0, 0, 0, 0],
    #     },
    #     index=[0, 1, 2, 3, 4, 5],
    # )
    cid = [0, 1, 2, 3, 4, 5]
    data = {
        "name": ["Date", "Open", "High", "Low", "Close", "Volume"],
        "type": ["DATE", "REAL", "REAL", "REAL", "REAL", "REAL"],
        "notnull": [1, 0, 0, 0, 0, 0],
        "dflt_value": [None, None, None, None, None, None],
        "pk": [1, 0, 0, 0, 0, 0],
    }
    expected_pragma_table = pd.DataFrame(data, index=cid)
    expected_pragma_table.index.name = "cid"

    pdt.assert_frame_equal(actual_PRAGMA_table, expected_pragma_table)


@pytest.mark.skip(reason="not ready")
def test_db_unique():
    """Try inserting duplicate records."""


@pytest.mark.skip(reason="not ready")
def test_get_price_data_retry_loop_retries():
    """Test get_price_data_retry_loop retries on failure."""
    with patch(
        "store_price_data.download_OHLCV_from_yf",
        side_effect=[Exception("fail"), pd.DataFrame({"AAPL": [1]})],
    ):
        pass
