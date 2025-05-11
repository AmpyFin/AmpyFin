import functools
from datetime import datetime, timedelta
import heapq
from multiprocessing import Pool, cpu_count
import sqlite3
from typing import Callable
import pandas as pd
import yfinance as yf
from statistics import median
from control import (
    trade_asset_limit,
    train_loss_price_change_ratio_d1,
    train_loss_price_change_ratio_d2,
    train_loss_profit_time_d1,
    train_loss_profit_time_d2,
    train_loss_profit_time_else,
    train_profit_price_change_ratio_d1,
    train_profit_price_change_ratio_d2,
    train_profit_profit_time_d1,
    train_profit_profit_time_d2,
    train_profit_profit_time_else,
    train_rank_asset_limit,
    train_rank_liquidity_limit,
    train_time_delta_balanced,
    train_time_delta_increment,
    train_time_delta_multiplicative,
)
# from helper_files.client_helper import get_ndaq_tickers, strategies
from helper_files.train_client_helper import get_historical_data
from helper_files.client_helper import strategies, get_ndaq_tickers
from utilities.session import limiter
import os

# from strategies.talib_indicators import *

def update_time_delta(time_delta, mode):
    """
    Updates time_delta based on the specified mode
    """
    if mode == "additive":
        return time_delta + train_time_delta_increment
    elif mode == "multiplicative":
        return time_delta * train_time_delta_multiplicative
    elif mode == "balanced":
        return time_delta + train_time_delta_balanced * time_delta
    return time_delta


def update_points_and_trades(
    strategy, ratio, current_price, trading_simulator, points, time_delta, ticker, qty
):
    """
    Updates points based on trade performance and manages trade statistics
    """
    if (
        current_price
        > trading_simulator[strategy.__name__]["holdings"][ticker]["price"]
    ):
        trading_simulator[strategy.__name__]["successful_trades"] += 1
        if ratio < train_profit_price_change_ratio_d1:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + time_delta * train_profit_profit_time_d1
            )
        elif ratio < train_profit_price_change_ratio_d2:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + time_delta * train_profit_profit_time_d2
            )
        else:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + time_delta * train_profit_profit_time_else
            )
    elif (
        current_price
        == trading_simulator[strategy.__name__]["holdings"][ticker]["price"]
    ):
        trading_simulator[strategy.__name__]["neutral_trades"] += 1
    else:
        trading_simulator[strategy.__name__]["failed_trades"] += 1
        if ratio > train_loss_price_change_ratio_d1:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + -time_delta * train_loss_profit_time_d1
            )
        elif ratio > train_loss_price_change_ratio_d2:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + -time_delta * train_loss_profit_time_d2
            )
        else:
            points[strategy.__name__] = (
                points.get(strategy.__name__, 0)
                + -time_delta * train_loss_profit_time_else
            )

    trading_simulator[strategy.__name__]["holdings"][ticker]["quantity"] -= qty
    if trading_simulator[strategy.__name__]["holdings"][ticker]["quantity"] == 0:
        del trading_simulator[strategy.__name__]["holdings"][ticker]
    elif trading_simulator[strategy.__name__]["holdings"][ticker]["quantity"] < 0:
        raise Exception("Quantity cannot be negative")
    trading_simulator[strategy.__name__]["total_trades"] += 1

    return points, trading_simulator


def execute_trade(
    decision,
    qty,
    ticker,
    current_price,
    strategy,
    trading_simulator,
    points,
    time_delta,
    portfolio_qty,
    total_portfolio_value,
):
    """
    Executes a trade based on the strategy decision and updates trading simulator and points
    """
    if (
        decision == "buy"
        and trading_simulator[strategy.__name__]["amount_cash"]
        > train_rank_liquidity_limit
        and qty > 0
        and ((portfolio_qty + qty) * current_price) / total_portfolio_value
        < train_rank_asset_limit
    ):
        trading_simulator[strategy.__name__]["amount_cash"] -= qty * current_price

        if ticker in trading_simulator[strategy.__name__]["holdings"]:
            trading_simulator[strategy.__name__]["holdings"][ticker]["quantity"] += qty
        else:
            trading_simulator[strategy.__name__]["holdings"][ticker] = {"quantity": qty}

        trading_simulator[strategy.__name__]["holdings"][ticker][
            "price"
        ] = current_price
        trading_simulator[strategy.__name__]["total_trades"] += 1

    elif (
        decision == "sell"
        and trading_simulator[strategy.__name__]["holdings"]
        .get(ticker, {})
        .get("quantity", 0)
        >= qty
    ):
        trading_simulator[strategy.__name__]["amount_cash"] += qty * current_price
        ratio = (
            current_price
            / trading_simulator[strategy.__name__]["holdings"][ticker]["price"]
        )

        points, trading_simulator = update_points_and_trades(
            strategy,
            ratio,
            current_price,
            trading_simulator,
            points,
            time_delta,
            ticker,
            qty,
        )

    return trading_simulator, points


def simulate_trading_day(
    current_date: pd.Timestamp,
    ticker_price_history: pd.DataFrame,
    precomputed_decisions: pd.DataFrame,
    strategies: list[Callable],
    train_tickers: list[str],
    logger,
    trading_simulator: dict,
    points: dict,
    time_delta: float,
):
    """
    Optimized version of simulate_trading_day that uses precomputed strategy decisions.
    """
    # current_date = current_date.strftime("%Y-%m-%d")
    print(f"Simulating trading for {current_date}.")

    for ticker in train_tickers:
        key = (ticker, current_date)
        if key in ticker_price_history.index:
            current_price = ticker_price_history.loc[key, 'Close']
            print(f"Current price for {ticker} on {current_date}: {current_price}")
        else:
            current_price = None
            print(f'No price for {ticker} on {current_date}. Skipping.')
            continue
        if current_price:
            print('Getting precomputed decisions...')
            # Get precomputed strategy decisions for the current date
            for strategy in strategies:
                strategy_name = strategy.__name__

                
                # Get precomputed strategy decision
                num_action = precomputed_decisions.at[key, strategy_name]
                print(f"Precomputed decision for {ticker} on {current_date}: {num_action}")
               
                if num_action == 1: 
                    action = 'Buy'
                elif num_action == -1:
                    action = 'Sell'
                else:
                    action = 'Hold'

                # Get account details for trade size calculation
                account_cash = trading_simulator[strategy_name]["amount_cash"]
                portfolio_qty = (
                    trading_simulator[strategy_name]["holdings"]
                    .get(ticker, {})
                    .get("quantity", 0)
                )
                total_portfolio_value = trading_simulator[strategy_name][
                    "portfolio_value"
                ]

                # Compute trade decision and quantity based on precomputed action
                decision, qty = compute_trade_quantities(
                    action,
                    current_price,
                    account_cash,
                    portfolio_qty,
                    total_portfolio_value,
                )
                print(f"Decision: {decision}, Quantity: {qty}")
                # Execute trade
                trading_simulator, points = execute_trade(
                    decision,
                    qty,
                    ticker,
                    current_price,
                    strategy,
                    trading_simulator,
                    points,
                    time_delta,
                    portfolio_qty,
                    total_portfolio_value,
                )

    return trading_simulator, points


def compute_trade_quantities(
    action, current_price, account_cash, portfolio_qty, total_portfolio_value
):
    """
    Computes trade decision and quantity based on the precomputed action.
    This replaces the quantity calculation part of simulate_strategy.
    """
    max_investment = total_portfolio_value * trade_asset_limit

    if action == "Buy":
        return "buy", min(
            int(max_investment // current_price), int(account_cash // current_price)
        )
    elif action == "Sell" and portfolio_qty > 0:
        return "sell", min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
    else:
        return "hold", 0


def precompute_strategy_decisions(
    strategies,
    ticker_price_history,
    train_tickers,
    ideal_period,
    start_date,
    end_date,
    logger,
):
    """
    Precomputes strategy decisions using parallel processing.
    """
    logger.info("Precomputing strategy decisions with parallel processing...")

    # Convert string dates to datetime objects if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Gather all valid trading days first
    trading_days = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Skip weekends
            trading_days.append(current_date)
        current_date += timedelta(days=1)

    # Initialize result structure
    precomputed_decisions = {
        strategy.__name__: {ticker: {} for ticker in train_tickers}
        for strategy in strategies
    }

    # Prepare parameters for parallel processing
    # We'll process by date to allow better sharing of historical data
    worker_func = functools.partial(
        _process_single_day,
        strategies=strategies,
        ticker_price_history=ticker_price_history,
        train_tickers=train_tickers,
        ideal_period=ideal_period,
    )

    # Use a process pool to parallel process dates
    num_workers = min(cpu_count(), len(trading_days))
    logger.info(f"Using {num_workers} worker processes")

    with Pool(processes=num_workers) as pool:
        results = pool.map(worker_func, trading_days)

    # Combine results from all processed days
    for day_results in results:
        if day_results:  # Skip empty results
            date_str = day_results["date"]
            for strategy_name, strategy_data in day_results["strategies"].items():
                for ticker, action in strategy_data.items():
                    precomputed_decisions[strategy_name][ticker][date_str] = action

    logger.info(
        f"Strategy decision precomputation complete. Processed {len(results)} trading days."
    )
    return precomputed_decisions


def _process_single_day(
    date, strategies, ticker_price_history, train_tickers, ideal_period
):
    """
    Process a single day for all tickers and strategies.
    This function will be executed in a separate process.
    """
    date_str = date.strftime("%Y-%m-%d")
    result = {
        "date": date_str,
        "strategies": {strategy.__name__: {} for strategy in strategies},
    }

    # Find tickers with data for this date
    available_tickers = [
        ticker
        for ticker in train_tickers
        if date_str in ticker_price_history[ticker].index
    ]

    if not available_tickers:
        return None  # No tickers have data for this date

    # Process each ticker and strategy
    for ticker in available_tickers:
        for strategy in strategies:
            strategy_name = strategy.__name__

            try:
                # Get historical data
                historical_data = get_historical_data(
                    ticker, date, ideal_period[strategy_name], ticker_price_history
                )

                if historical_data is None or historical_data.empty:
                    continue

                # Compute strategy signal
                action = strategy(ticker, historical_data)
                result["strategies"][strategy_name][ticker] = action

            except Exception:
                # Skip errors in worker process
                continue

    return result



def update_ranks(client):
    """
    based on portfolio values, rank the strategies to use for actual trading_simulator
    """

    db = client.trading_simulator
    points_collection = db.points_tally
    rank_collection = db.rank
    algo_holdings = db.algorithm_holdings
    """
   delete all documents in rank collection first
   """
    rank_collection.delete_many({})
    """
   Reason why delete rank is so that rank is intially null and
   then we can populate it in the order we wish
   now update rank based on successful_trades - failed
   """
    q = []
    for strategy_doc in algo_holdings.find({}):
        """
        based on (points_tally (less points pops first), failed-successful(more negtive pops first),
        portfolio value (less value pops first), and then strategy_name), we add to heapq.
        """
        strategy_name = strategy_doc["strategy"]
        if strategy_name == "test" or strategy_name == "test_strategy":
            continue
        if points_collection.find_one({"strategy": strategy_name})["total_points"] > 0:
            heapq.heappush(
                q,
                (
                    points_collection.find_one({"strategy": strategy_name})[
                        "total_points"
                    ]
                    * 2
                    + (strategy_doc["portfolio_value"]),
                    strategy_doc["successful_trades"] - strategy_doc["failed_trades"],
                    strategy_doc["amount_cash"],
                    strategy_doc["strategy"],
                ),
            )
        else:
            heapq.heappush(
                q,
                (
                    strategy_doc["portfolio_value"],
                    strategy_doc["successful_trades"] - strategy_doc["failed_trades"],
                    strategy_doc["amount_cash"],
                    strategy_doc["strategy"],
                ),
            )
    rank = 1
    while q:
        _, _, _, strategy_name = heapq.heappop(q)
        rank_collection.insert_one({"strategy": strategy_name, "rank": rank})
        rank += 1

    """
   Delete historical database so new one can be used tomorrow
   """
    db = client.HistoricalDatabase
    collection = db.HistoricalDatabase
    collection.delete_many({})
    print("Successfully updated ranks")
    print("Successfully deleted historical database")



def weighted_majority_decision_and_median_quantity(decisions_and_quantities):
    """
    Determines the majority decision (buy, sell, or hold) and returns the weighted median quantity for the chosen action.
    Groups 'strong buy' with 'buy' and 'strong sell' with 'sell'.
    Applies weights to quantities based on strategy coefficients.
    """
    buy_decisions = ["buy", "strong buy"]
    sell_decisions = ["sell", "strong sell"]

    weighted_buy_quantities = []
    weighted_sell_quantities = []
    buy_weight = 0
    sell_weight = 0
    hold_weight = 0

    # Process decisions with weights
    for decision, quantity, weight in decisions_and_quantities:
        if decision in buy_decisions:
            weighted_buy_quantities.extend([quantity])
            buy_weight += weight
        elif decision in sell_decisions:
            weighted_sell_quantities.extend([quantity])
            sell_weight += weight
        elif decision == "hold":
            hold_weight += weight

    # Determine the majority decision based on the highest accumulated weight
    if buy_weight > sell_weight and buy_weight > hold_weight:
        return (
            "buy",
            median(weighted_buy_quantities) if weighted_buy_quantities else 0,
            buy_weight,
            sell_weight,
            hold_weight,
        )
    elif sell_weight > buy_weight and sell_weight > hold_weight:
        return (
            "sell",
            median(weighted_sell_quantities) if weighted_sell_quantities else 0,
            buy_weight,
            sell_weight,
            hold_weight,
        )
    else:
        return "hold", 0, buy_weight, sell_weight, hold_weight





def fetch_price_from_db(start_date: pd.Timestamp, end_date: pd.Timestamp, train_tickers: list[str]) -> pd.DataFrame:

     # Get the current file's directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the database path relative to the current file
    db_path = os.path.join(current_file_dir, '../dbs/databases/price_data.db')
    conn = sqlite3.connect(db_path)
    combined_df = pd.DataFrame()

    # Convert start_date and end_date to string format
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    for ticker in train_tickers:
        query = f"""
            SELECT *
            FROM "{ticker}"
            WHERE Date BETWEEN ? AND ?
        """
        df = pd.read_sql_query(query, conn, params=(start_date_str, end_date_str))
        df['Ticker'] = ticker
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    conn.close()
    return combined_df

def fetch_strategy_decisions(
  start_date: pd.Timestamp, end_date: pd.Timestamp, train_tickers: list[str],  strategies: list) -> pd.DataFrame:
    # Get the current file's directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the database path relative to the current file
    db_path = os.path.join(current_file_dir, '../dbs/databases/strategy_decisions.db')
    conn = sqlite3.connect(db_path)
    combined_df = pd.DataFrame()

    # Convert start_date and end_date to string format
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    # Convert strategy functions to their string names if needed
    strategy_names = [s.__name__ if callable(s) else str(s) for s in strategies]

    for ticker in train_tickers:
        columns = ', '.join([f'"{s}"' for s in strategy_names])  # safely quote column names
        query = f"""
            SELECT Date, {columns}
            FROM "{ticker}"
            WHERE Date BETWEEN ? AND ?
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        df['Ticker'] = ticker
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    conn.close()
    return combined_df