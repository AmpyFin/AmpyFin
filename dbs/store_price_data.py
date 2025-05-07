import logging
import logging.config
import os
import sqlite3
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from log_config import LOG_CONFIG

from config import PRICE_DB_PATH
from dbs.helper_functions import get_ndaq_tickers, retry_with_backoff


def download_OHLCV_from_yf(ticker_list, logger):
    logger.info(f"start downloading data {len(ticker_list)=}")
    yf_period = "max"
    df = pd.DataFrame()
    try:
        df = yf.download(
            ticker_list,
            group_by="Ticker",
            period=yf_period,
            interval="1d",
            auto_adjust=True,
            repair=True,
            rounding=True,
        )
    except Exception as e:
        logger.error(f"yf error {e}")

    if df is not None and not df.empty:  # Check if df was assigned
        # stack multi-level column index
        df = (
            df.stack(level=0, future_stack=True)
            .rename_axis(["Date", "Ticker"])
            .reset_index(level=1)
        )
        print("Index type:", type(df.index))
        print("Index dtype:", df.index.dtype)

    return df


def store_OHLCV_in_db(df, ticker_list, price_data_db_name, logger):
    logger.info(f"Saving {len(ticker_list)} tickers to {price_data_db_name}")
    if not ticker_list:
        logger.warning("Ticker list is empty")
        return 0, []
    tickers_saved = []
    tickers_with_no_data = []
    percentage_of_tickers_saved = 0
    for ticker in ticker_list:
        df_single_ticker = df[["Open", "High", "Low", "Close", "Volume"]].loc[
            df["Ticker"] == ticker
        ]
        df_single_ticker = df_single_ticker.dropna()
        df_single_ticker.index = df_single_ticker.index.strftime("%Y-%m-%d")

        # store ticker in price_data.db
        if df_single_ticker.empty:
            tickers_with_no_data.append(ticker)
            logger.warning(f"no OHLCV data for {ticker}")
        else:
            with sqlite3.connect(price_data_db_name) as conn:
                try:
                    df_single_ticker.to_sql(
                        ticker,
                        conn,
                        if_exists="replace",
                        # dtype={"Date": "TEXT PRIMARY KEY NOT NULL"},
                        dtype={"Date": "DATE PRIMARY KEY NOT NULL"},
                    )
                    tickers_saved.append(ticker)
                except Exception as e:
                    logger.error(
                        f"""error saving {ticker} OHLCV price data to
                          {price_data_db_name}: {e}"""
                    )

    percentage_of_tickers_saved = round(
        (len(tickers_saved) / len(ticker_list)) * 100, 2
    )
    logger.info(
        f"{len(tickers_saved)} of {len(ticker_list)} ({percentage_of_tickers_saved} %) tickers saved to {price_data_db_name}"
    )
    if len(tickers_with_no_data) > 0:
        logger.warning(
            f"""no data for {len(tickers_with_no_data)} ticker(s):
             {tickers_with_no_data}"""
        )
    return percentage_of_tickers_saved, tickers_with_no_data


def get_price_data_retry_loop(
    PRICE_DB_PATH,
    ticker_list,
    logger,
    ticker_download_threshold=90,
    max_retries=3,
    initial_delay=30,
    backoff_factor=10,
):
    percentage_of_tickers_saved = 0
    tickers_with_no_data = []
    for attempt in range(max_retries):
        if len(ticker_list) == 0:
            logger.error(f"No tickers to download! {len(ticker_list)=}")
            break
        df = download_OHLCV_from_yf(ticker_list, logger)

        if df is not None and not df.empty:
            percentage_of_tickers_saved, tickers_with_no_data = (
                store_OHLCV_in_db(df, ticker_list, PRICE_DB_PATH, logger)
            )

        if percentage_of_tickers_saved >= ticker_download_threshold:
            logger.info(
                f"Ticker download threshold met ({percentage_of_tickers_saved}% >={ticker_download_threshold}%)"
            )
            break  # Exit loop if successful
        else:
            logger.warning(f"Attempt {attempt + 1}/{max_retries}")
            logger.warning(
                f"Ticker download threshold not met ({percentage_of_tickers_saved}% < {ticker_download_threshold}%). Retrying..."
            )
            if attempt < max_retries - 1:
                delay = initial_delay * (backoff_factor**attempt)
                logger.info(
                    f"Waiting for {delay:.2f} seconds before next retry."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Max retries ({max_retries}) reached. Ticker download threshold still not met."
                )
    return percentage_of_tickers_saved, tickers_with_no_data


if __name__ == "__main__":
    """
    Downloads historical OHLCV data for a list of tickers and stores it in a SQLite database.

    Uses yfinance to download the maximum available
    historical data ('max' period) with a '1d' interval.
    Each ticker's data is stored in a separate table named
    after the ticker symbol within the specified database file. Existing tables
    for the same ticker will be replaced.
    """
    # Get the current filename without extension
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    log_filename = f"log/{module_name}.log"
    LOG_CONFIG["handlers"]["file_dynamic"]["filename"] = log_filename

    logging.config.dictConfig(LOG_CONFIG)
    logger = logging.getLogger(__name__)
    # logger.info("Logging started")

    # ticker_list = train_tickers
    # ticker_list = ["APP"]
    ticker_list = []
    if not ticker_list:
        try:
            logger.info("getting ndaq tickers from wikipedia...")
            ticker_list = retry_with_backoff(get_ndaq_tickers, logger)
            # ticker_list = get_ndaq_tickers()
        except Exception as e:
            logger.error(f"Error getting ndaq tickers {e}")

    # %, retry if downloaded tickers pct less than this.
    ticker_download_threshold = 90

    if ticker_list:
        percentage_of_tickers_saved, tickers_with_no_data = (
            get_price_data_retry_loop(
                PRICE_DB_PATH, ticker_list, logger, ticker_download_threshold
            )
        )

        # Final check after all retries
        assert (
            percentage_of_tickers_saved >= ticker_download_threshold
        ), f"""Ticker download threshold not met.
          Only {percentage_of_tickers_saved}% of tickers were saved,
            which is less than the required {ticker_download_threshold}%.
              Tickers with no data: {tickers_with_no_data}"""
