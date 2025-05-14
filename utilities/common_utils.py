import heapq
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
import os

# from strategies.talib_indicators import *

def get_ndaq_tickers() -> list[str]:
    """
        Returns a list of NASDAQ-100 tickers.

        This function scrapes the list of NASDAQ-100 companies from a Wikipedia page
        and extracts the ticker symbols.

        Returns:
            list: A list of strings, where each string is a NASDAQ-100 ticker symbol.
        """
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    df = tables[4]  # NASDAQ-100 companies table
    return df["Ticker"].tolist()


def fetch_price_from_db(start_date: pd.Timestamp, end_date: pd.Timestamp, train_tickers: list[str]) -> pd.DataFrame:
    """Fetches price data from a SQLite database for specified tickers and date range.

        Args:
            start_date (pd.Timestamp): The start date for the data retrieval.
            end_date (pd.Timestamp): The end date for the data retrieval.
            train_tickers (list[str]): A list of ticker symbols to retrieve data for.

        Returns:
            pd.DataFrame: A DataFrame containing the price data for the specified tickers and date range.
                          The DataFrame includes 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', and 'Ticker' columns.
                          Returns an empty DataFrame if no data is found.
    """
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

def fetch_strategy_decisions(start_date: pd.Timestamp, end_date: pd.Timestamp, train_tickers: list[str],  strategies: list) -> pd.DataFrame:
    """Fetches strategy decisions from a SQLite database for specified tickers and date range.

        Args:
            start_date (pd.Timestamp): The start date for fetching decisions.
            end_date (pd.Timestamp): The end date for fetching decisions.
            train_tickers (list[str]): A list of ticker symbols to fetch data for.
            strategies (list): A list of strategy functions or their names.

        Returns:
            pd.DataFrame: A DataFrame containing the strategy decisions, with columns
                for 'Date', each strategy specified in `strategies`, and 'Ticker'.
                The 'Date' column contains the date of the decision, the strategy
                columns contain the decision values, and the 'Ticker' column
                identifies the ticker symbol for the decision.

        Raises:
            sqlite3.Error: If there is an error connecting to or querying the database.
    """
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
) -> tuple[dict, dict]:
    """Simulate trading for a single day using precomputed strategy decisions.
        Args:
            current_date (pd.Timestamp): The date for which to simulate trading.
            ticker_price_history (pd.DataFrame): DataFrame containing historical price data for tickers.
                The index should be a MultiIndex with levels 'ticker' and 'date'.
                Must contain a 'Close' column.
            precomputed_decisions (pd.DataFrame): DataFrame containing precomputed trading decisions for each strategy.
                The index should be a MultiIndex with levels 'ticker' and 'date'.
                Columns should be the names of the strategies.
            strategies (list[Callable]): A list of trading strategy functions.
            train_tickers (list[str]): A list of ticker symbols to trade.
            logger: Logger object for logging information.
            trading_simulator (dict): A dictionary containing the trading simulator state.
                Includes account cash, holdings, and portfolio value for each strategy.
            points (dict): A dictionary to store points/rewards earned by each strategy.
            time_delta (float): The time delta used for calculating transaction costs or slippage.
        Returns:
            tuple[dict, dict]: A tuple containing the updated trading simulator state and the updated points dictionary.
    """
    # current_date = current_date.strftime("%Y-%m-%d")
    # print(f"Simulating trading for {current_date}.")

    for ticker in train_tickers:
        key = (ticker, current_date)
        if key in ticker_price_history.index:
            current_price = ticker_price_history.loc[key, 'Close']
            # print(f"Current price for {ticker} on {current_date}: {current_price}")
        else:
            current_price = None
            print(f'No price for {ticker} on {current_date}. Skipping.')
            continue
        if current_price:
            # Get precomputed strategy decisions for the current date
            for strategy in strategies:
                strategy_name = strategy.__name__

                
                # Get precomputed strategy decision
                num_action = precomputed_decisions.at[key, strategy_name]
                # print(f"Precomputed decision for {ticker} on {current_date}: {num_action}")
               
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
                # print(f"Decision: {decision}, Quantity: {qty}")
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


def local_update_portfolio_values(
    current_date: pd.Timestamp,
    strategies: list[Callable],
    trading_simulator: dict,
    ticker_price_history: pd.DataFrame,
    logger,
) -> tuple[int, dict]:
    """Updates the portfolio values for each strategy based on current holdings and market prices.
    Args:
        current_date (pd.Timestamp): The date for which to update the portfolio values.
        strategies (list[Callable]): A list of strategy classes to update.
        trading_simulator (dict): A dictionary containing the trading simulation data,
            including cash balance and holdings for each strategy.
        ticker_price_history (pd.DataFrame): A DataFrame containing the historical
            price data for each ticker.  The index should be a MultiIndex with
            levels 'Ticker' and 'Date'.
        logger (logging.Logger): A logger object for logging information and errors.
    Returns:
        tuple[int, dict]: A tuple containing:
            - active_count (int): The number of strategies with a portfolio value
              different from the initial $50,000.
            - trading_simulator (dict): The updated trading simulator dictionary
              with updated portfolio values for each strategy.
    """
   
    logger.info(f"Updating portfolio values for {current_date.strftime('%Y-%m-%d')}.")

    active_count = 0

    for strategy in strategies:
        strategy_name = strategy.__name__
        # logger.info(f"Processing strategy: {strategy_name}.")

        # Reset portfolio value to cash balance
        trading_simulator[strategy_name]["portfolio_value"] = trading_simulator[
            strategy_name
        ]["amount_cash"]
        # logger.info(f"{strategy_name}: Starting portfolio value (cash only): {trading_simulator[strategy_name]['portfolio_value']}")

        amount = 0

        # Update portfolio value based on current holdings
        for ticker in trading_simulator[strategy_name]["holdings"]:
            qty = trading_simulator[strategy_name]["holdings"][ticker]["quantity"]
            key = (ticker, current_date)
            if key in ticker_price_history.index:
                current_price = ticker_price_history.loc[key, 'Close']
                position_value = qty * current_price
                amount += position_value
                logger.info(f"{strategy_name}: {ticker} - Qty: {qty}, Price: {current_price}, Position Value: {position_value}")
            else:
                logger.info(f'No price for {ticker} on {current_date}. Skipping.')
                continue

        cash = trading_simulator[strategy_name]["amount_cash"]
        trading_simulator[strategy_name]["portfolio_value"] = amount + cash

        logger.info(f"{strategy_name}: Final portfolio value: {trading_simulator[strategy_name]['portfolio_value']}")

        # Count active strategies (i.e., those with a portfolio value different from the initial $50,000)
        if trading_simulator[strategy_name]["portfolio_value"] != 50000:
            active_count += 1
            logger.info(f"{strategy_name}: Strategy is active.")

    # logger.info(f"Total active strategies: {active_count}")
    logger.info(f"Completed portfolio update for {current_date.strftime('%Y-%m-%d')}.")

    return active_count, trading_simulator


def update_time_delta(time_delta: float, mode: str) -> float:
    """Updates a time delta based on a specified mode.
        Args:
            time_delta (float): The current time delta value.
            mode (str): The update mode. Can be "additive", "multiplicative", or "balanced".
        Returns:
            float: The updated time delta value.
    """
    if mode == "additive":
        return time_delta + train_time_delta_increment
    elif mode == "multiplicative":
        return time_delta * train_time_delta_multiplicative
    elif mode == "balanced":
        return time_delta + train_time_delta_balanced * time_delta
    return time_delta


def weighted_majority_decision_and_median_quantity(decisions_and_quantities: list[tuple[str, float, float]]) -> tuple[str, float, float, float, float]:
    """Computes a weighted majority decision and the median quantity based on given decisions and quantities.

        Args:
            decisions_and_quantities (list of tuples): A list where each tuple contains a decision (string),
                                                        a quantity (numeric), and a weight (numeric).
                                                        Example: [("buy", 100, 0.6), ("sell", 50, 0.3), ("hold", 0, 0.1)]

        Returns:
            tuple: A tuple containing:
                - The majority decision ("buy", "sell", or "hold").
                - The median quantity corresponding to the majority decision (numeric). Returns 0 if no quantities
                  are associated with the majority decision.
                - The accumulated buy weight.
                - The accumulated sell weight.
                - The accumulated hold weight.

        Raises:
            TypeError: If the input is not a list of tuples or if the tuples do not contain the expected data types.
            ValueError: If the decision string is not one of "buy", "strong buy", "sell", "strong sell", or "hold".
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


######## LEVEL 1 DEPENDENCIES - For the functions mentioned above, their supporting functions are given below
def compute_trade_quantities(
    action: str,
    current_price: float,
    account_cash: float,
    portfolio_qty: int,
    total_portfolio_value: float,
) -> tuple[str, int]:
    """Computes trade decision and quantity based on the precomputed action.
        Args:
            action (str): A string representing the trading action to take ("Buy", "Sell", or "Hold").
            current_price (float): The current price of the asset.
            account_cash (float): The amount of cash available in the account.
            portfolio_qty (int): The current quantity of the asset held in the portfolio.
            total_portfolio_value (float): The total value of the portfolio.

        Returns:
            tuple: A tuple containing the trade decision (string) and the quantity to trade (int).
                   The trade decision can be "buy", "sell", or "hold".  The quantity is the number
                   of shares to buy or sell.

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
    
def execute_trade(
    decision: str,
    qty: int,
    ticker: str,
    current_price: float,
    strategy: object,
    trading_simulator: dict,
    points: dict,
    time_delta: float,
    portfolio_qty: int,
    total_portfolio_value: float,
) -> tuple[dict, dict]:
    """Executes a trade based on the given decision, quantity, and ticker.
        Args:
            decision (str): The decision to make, either "buy" or "sell".
            qty (int): The quantity of the asset to trade.
            ticker (str): The ticker symbol of the asset.
            current_price (float): The current price of the asset.
            strategy (object): The trading strategy being used.
            trading_simulator (dict): The trading simulator dictionary.
            points (float): The current points accumulated by the strategy.
            time_delta (timedelta): The time difference since the last trade.
            portfolio_qty (int): The current quantity of the asset in the portfolio.
            total_portfolio_value (float): The total value of the portfolio.

        Returns:
            tuple: A tuple containing the updated trading simulator and points.
        
        Raises:
            KeyError: If the ticker is not found in the holdings when selling.
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


######## LEVEL 2 DEPENDENCIES - For the functions mentioned above, their supporting functions are given below
def update_points_and_trades(strategy: object, ratio: float, current_price: float, trading_simulator: dict, points: dict, time_delta: float, ticker: str, qty: int) -> tuple[dict, dict]:
    """Updates points and trade statistics based on the trading outcome.
        Args:
            strategy (class): The trading strategy being evaluated.
            ratio (float): The price change ratio (current_price / initial_price).
            current_price (float): The current price of the asset.
            trading_simulator (dict): A dictionary containing the trading simulation data.
            points (dict): A dictionary storing the points earned by each strategy.
            time_delta (float): The time elapsed since the last trade.
            ticker (str): The ticker symbol of the asset.
            qty (int): The quantity of the asset traded.

        Returns:
            tuple: A tuple containing the updated points dictionary and the updated trading simulator dictionary.
            
        Raises:
            Exception: If the quantity of the asset becomes negative.
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










