import logging
import logging.config
import os
import sqlite3
import sys
import time

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log_config import LOG_CONFIG  # noqa: E402

PRICE_DB_PATH = os.path.join('dbs', 'databases', 'price_data.db')
STRATEGY_DECISIONS_DB_PATH = os.path.join('dbs', 'databases', 'strategy_decisions.db')
# from control import train_tickers  # noqa: E402
from strategies.categorise_talib_indicators_vect import (
    strategies,
)  # noqa: E402
from dbs.helper_functions import (  # noqa: E402
    get_ndaq_tickers,
    retry_with_backoff,
)

"""
This script computes and stores strategy decisions for a list of stock tickers.

It performs the following steps:
- Checks if historical price data exists for each ticker in the price database.
- Loads each ticker's price history from the database.
- Applies a set of strategy functions to the price data to generate
 strategy decisions.
- Combines the results from all strategies into a single DataFrame.
- Stores the combined strategy decisions in a separate SQLite database,
 with one table per ticker.

The script is designed to minimize RAM usage by processing and writing results
 on a per-ticker basis.
Logging is used throughout to track progress and errors.

Typical usage:
    Run this script as a standalone module to process all tickers in the
      configured list.

Configuration:
    - Database paths and ticker lists are imported from config files.
    - Strategies are imported from the strategies module.
"""


def check_ticker_tables_exist(db_path, ticker_list):
    """
    Checks if each ticker in ticker_list has a corresponding table
      in the SQLite database at db_path.
    Args:
    db_path (str): Path to the SQLite database containing tables.
    tickers_list (list): List of ticker symbols to process.

    Returns a dict: {ticker: True/False}
    """
    table_exists = {}
    with sqlite3.connect(db_path) as con:
        cursor = con.cursor()
        for ticker in ticker_list:
            cursor.execute(
                """SELECT name FROM sqlite_master
                WHERE type='table' AND name=?;""",
                (ticker,),
            )
            table_exists[ticker] = cursor.fetchone() is not None
    return table_exists


def compute_and_store_strategy_decisions(
    PRICE_DB_PATH, STRATEGY_DECISIONS_DB_PATH, ticker_list, strategies, logger
):
    """
    Computes and stores strategy decisions for a list of tickers.

    For each ticker in the provided list:
      - Checks if price data exists in the price database.
      - Loads the ticker's historical price data from the database.
      - Applies each strategy function to the price data, collecting their
        results.
      - Concatenates all strategy results into a single DataFrame.
      - Stores the combined strategy decisions in the strategy decisions
      database,
        replacing any existing table for that ticker.

    Args:
        PRICE_DB_PATH (str): Path to the SQLite database containing price data
        tables.
        STRATEGY_DECISIONS_DB_PATH (str): Path to the SQLite database for
        storing strategy decisions.
        tickers_list (list): List of ticker symbols to process.
        strategies (list): List of strategy functions to apply to
          each ticker's price data.

    NOTES:
        Strategy decisions are written to db on a ticker by ticker basis.
        This is a conscious design choice to reduce RAM usage if calculating
        many tickers.
        It may be quicker to to bulk write all at once, however 1GB vps may
        struggle with many tickers.
        There is a speed v RAM tradeoff.
    """
    start_time = time.time()

    # check ticker price data exists
    table_exists_dict = {}
    try:
        logger.info(f"Checking tickers exist in {PRICE_DB_PATH}")
        table_exists_dict = check_ticker_tables_exist(
            PRICE_DB_PATH, ticker_list
        )
    except Exception as e:
        logger.error(
            f"error checking if tickers exist in {PRICE_DB_PATH}. {e}"
        )

    # Compute and store strategy decisions by ticker
    for idx, ticker in enumerate(ticker_list):
        logger.info(
            f"Computing decisions: {ticker} ({idx + 1}/{len(ticker_list)})."
            f"{len(strategies)=}"
        )
        # check ticker price data exists
        if table_exists_dict[ticker] is False:
            logger.warning(
                f"""{ticker} not found in price_data.db.
                {table_exists_dict[ticker]=}, skipping..."""
            )
            continue

        strategy_results = []

        # get ticker price data from db
        with sqlite3.connect(PRICE_DB_PATH) as con_price_data:
            ticker_price_history = pd.read_sql_query(
                "SELECT * FROM '{tab}'".format(tab=ticker),
                con_price_data,
                index_col="Date",
            )

        # compute strategy decision
        for idx, strategy in enumerate(strategies):
            # strategy_name = strategy.__name__
            # logger.info(f"{strategy_name} ({idx + 1}/{len(strategies)})")
            strategy_result = strategy(ticker_price_history.copy())
            strategy_results.append(strategy_result)

        combined_strategy_results = pd.concat(strategy_results, axis=1)

        # store strategy decisions in db
        with sqlite3.connect(
            STRATEGY_DECISIONS_DB_PATH
        ) as con_strategy_decisions:
            combined_strategy_results.to_sql(
                ticker,
                con_strategy_decisions,
                if_exists="replace",
                index=True,
                dtype={"Date": "DATE PRIMARY KEY NOT NULL"},
            )
            logger.info(f"Data for {ticker} saved to database.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Execution time for main(): {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # Get the current filename without extension
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    log_filename = f"log/{module_name}.log"
    LOG_CONFIG["handlers"]["file_dynamic"]["filename"] = log_filename

    logging.config.dictConfig(LOG_CONFIG)
    logger = logging.getLogger(__name__)

    # ticker_list = train_tickers
    # ticker_list = ["xxxx", "APP"]
    ticker_list = []
    if not ticker_list:
        logger.info("getting ndaq tickers from wikipedia...")
        try:
            ticker_list = retry_with_backoff(get_ndaq_tickers, logger)
        except Exception as e:
            logger.error(f"Error getting ndaq tickers {e}")

    try:
        compute_and_store_strategy_decisions(
            PRICE_DB_PATH,
            STRATEGY_DECISIONS_DB_PATH,
            ticker_list,
            strategies,
            logger,
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
