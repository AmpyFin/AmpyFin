#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing module for the trading simulator.

This module handles the testing phase of the trading simulation, including:
- Loading trained strategy results
- Applying weighted majority voting for trading decisions
- Implementing portfolio management with stop-loss and take-profit mechanisms
- Calculating and reporting performance metrics
"""

import heapq
import json
import os
from datetime import timedelta
import pandas as pd

from pymongo import MongoClient
import wandb

# Local imports
from variables import config_dict
from control import (
    benchmark_asset,
    test_period_end,
    test_period_start,
    trade_asset_limit,
    train_start_cash,
    train_stop_loss,
    train_suggestion_heap_limit,
    train_take_profit,
    train_tickers,
    train_time_delta_mode,
    train_trade_asset_limit,
    train_trade_liquidity_limit,
)
from utilities.ranking_trading_utils import strategies
from utilities.testing_utils import (
    calculate_metrics,
    generate_tear_sheet,
)
from utilities.common_utils import (
    get_ndaq_tickers,
    compute_trade_quantities,
    simulate_trading_day,
    update_time_delta,
    weighted_majority_decision_and_median_quantity,
    fetch_price_from_db,
    fetch_strategy_decisions,
    local_update_portfolio_values,
)

from utilities.logging import setup_logging
logger = setup_logging(__name__)

def initialize_test_account() -> dict:
    """
    Initialize the test trading account with starting parameters.
    
    Returns:
        dict: The initialized trading account with default values.
    """
    return {
        "holdings": {},  # Current stock holdings
        "cash": train_start_cash,  # Starting cash amount
        "trades": [],  # History of all trades
        "total_portfolio_value": train_start_cash,  # Initial portfolio value
    }

def check_stop_loss_take_profit(account: dict, ticker: str, current_price: float) -> dict:
    """
    Check and execute stop loss and take profit orders for a given ticker.
    
    Args:
        account (dict): The trading account to update.
        ticker (str): The stock ticker to check.
        current_price (float): The current price of the stock.
        
    Returns:
        dict: The updated trading account.
    """
    if ticker in account["holdings"]:
        if account["holdings"][ticker]["quantity"] > 0:
            # Check if price has dropped below stop loss or risen above take profit level
            if (
                current_price < account["holdings"][ticker]["stop_loss"]
                or current_price > account["holdings"][ticker]["take_profit"]
            ):
                # Record the sell trade
                account["trades"].append(
                    {
                        "symbol": ticker,
                        "quantity": account["holdings"][ticker]["quantity"],
                        "price": current_price,
                        "action": "sell",
                    }
                )
                # Update cash position after sale
                account["cash"] += (
                    account["holdings"][ticker]["quantity"] * current_price
                )
                # Remove ticker from holdings
                del account["holdings"][ticker]
    return account


def execute_buy_orders(buy_heap: list, suggestion_heap: list, account: dict, ticker_price_history: pd.DataFrame, current_date: pd.Timestamp) -> dict:
    """
    Execute buy orders from the buy and suggestion heaps.
    
    Args:
        buy_heap (list): Heap of prioritized buy orders.
        suggestion_heap (list): Heap of suggested buy orders.
        account (dict): The trading account to update.
        ticker_price_history (pd.DataFrame): Historical price data for tickers.
        current_date (datetime): The current trading date.
        
    Returns:
        dict: The updated trading account.
    """
    # Continue executing orders until either no more orders or insufficient cash
    while (buy_heap or suggestion_heap) and float(account["cash"]) > train_trade_liquidity_limit:
        # Choose which heap to pop from
        if buy_heap and float(account["cash"]) > train_trade_liquidity_limit:
            heap = buy_heap
        elif suggestion_heap and float(account["cash"]) > train_trade_liquidity_limit:
            heap = suggestion_heap
        else:
            break

        # Pop the highest priority order from the heap
        _, quantity, ticker = heapq.heappop(heap)

        # Get current price for the ticker
        key = (ticker, current_date)
        if key in ticker_price_history.index:
            current_price = ticker_price_history.loc[key, 'Close']
        else:
            # Skip if no price data available
            continue

        # Record the buy trade
        account["trades"].append(
            {
                "symbol": ticker,
                "quantity": quantity,
                "price": current_price,
                "action": "buy",
                "date": current_date.strftime("%Y-%m-%d"),
            }
        )

        # Update cash and holdings
        account["cash"] -= quantity * current_price
        account["holdings"][ticker] = {
            "quantity": quantity,
            "price": current_price,
            "stop_loss": current_price * (1 - train_stop_loss),
            "take_profit": current_price * (1 + train_take_profit),
        }

    return account


def update_strategy_ranks(strategies: list, points: dict, trading_simulator: dict) -> dict:
    """
    Update strategy rankings based on performance metrics.
    
    Args:
        strategies (list): List of strategy functions.
        points (dict): Points earned by each strategy.
        trading_simulator (dict): Performance data for each strategy.
        
    Returns:
        dict: Updated rank for each strategy.
    """
    rank = {}
    q = []
    
    # Calculate scores for each strategy based on points and portfolio value
    for strategy in strategies:
        if points[strategy.__name__] > 0:
            score = (
                points[strategy.__name__] * 2
                + trading_simulator[strategy.__name__]["portfolio_value"]
            )
        else:
            score = trading_simulator[strategy.__name__]["portfolio_value"]

        # Add strategy to priority queue with score and other metrics
        heapq.heappush(
            q,
            (
                score,
                trading_simulator[strategy.__name__]["successful_trades"]
                - trading_simulator[strategy.__name__]["failed_trades"],
                trading_simulator[strategy.__name__]["amount_cash"],
                strategy.__name__,
            ),
        )

    # Assign rank to each strategy based on its position in the queue
    coeff_rank = 1
    while q:
        _, _, _, strategy_name = heapq.heappop(q)
        rank[strategy_name] = coeff_rank
        coeff_rank += 1

    return rank


def test(mongo_client: MongoClient) -> None:
    """
    Run the testing phase of the trading simulator.
    
    This function loads trained strategy results, applies a weighted majority
    voting system for trading decisions, implements portfolio management with
    stop-loss and take-profit, and calculates performance metrics.
    
    Args:
        mongo_client (MongoClient): MongoDB client connection.
        
    Returns:
        None: Results are logged to file and W&B.
    """
    logger.info("Starting testing phase...")

    # Get rank coefficients from database
    db = mongo_client.trading_simulator
    r_t_c = db.rank_to_coefficient
    rank_to_coefficient = {doc["rank"]: doc["coefficient"] for doc in r_t_c.find({})}

    # Load saved training results
    results_dir = os.path.join('../artifacts', 'results')
    
    with open(os.path.join(results_dir, f"{config_dict['experiment_name']}.json"), "r") as json_file:
        results = json.load(json_file)
        trading_simulator = results["trading_simulator"]
        points = results["points"]
        time_delta = results["time_delta"]
    logger.info("Training results loaded successfully")

    # Initialize testing variables
    strategy_to_coefficient = {}
    account = initialize_test_account()
    logger.info("Test account initialized")

    rank = update_strategy_ranks(strategies, points, trading_simulator)
    logger.info("Strategy ranks updated")
    
    # Parse date strings to datetime objects
    start_date = pd.to_datetime(test_period_start, format="%Y-%m-%d")
    end_date = pd.to_datetime(test_period_end, format="%Y-%m-%d")
    current_date = start_date
    
    # Initialize series to track account values for metrics calculation
    account_values = pd.Series(index=pd.date_range(start=start_date, end=end_date))
    
    tickers = train_tickers if train_tickers else get_ndaq_tickers()
    logger.info(f"Testing with {len(tickers)} tickers from {test_period_start} to {test_period_end}")
    
    # Fetch price data for the entire testing period
    logger.info("Fetching historical price data and strategy decisions...")

    ticker_price_history = fetch_price_from_db(
        start_date - timedelta(days=1), end_date, tickers)
    ticker_price_history['Date'] = pd.to_datetime(ticker_price_history['Date'], format="%Y-%m-%d")
    ticker_price_history.set_index(['Ticker', 'Date'], inplace=True)
    
    # Preload strategy decisions for the testing period
    precomputed_decisions = fetch_strategy_decisions( 
        start_date - timedelta(days=1),
        end_date,
        tickers,
        strategies,
    ) 
    precomputed_decisions['Date'] = pd.to_datetime(precomputed_decisions['Date'], format="%Y-%m-%d")
    precomputed_decisions.set_index(['Ticker', 'Date'], inplace=True)

    logger.info("Data preparation complete")

    # Get unique trading dates from price history
    dates = ticker_price_history.index.get_level_values(1).unique()
    dates = [date.strftime("%Y-%m-%d") for date in dates]
    dates = sorted(dates)
    logger.info(f"Found {len(dates)} trading days in the period")
    
    # Main simulation loop
    logger.info("Beginning simulation...")
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Skip non-trading days (weekends or holidays)
        if date_str not in dates:

            current_date += timedelta(days=1)
            continue

        # Update strategy coefficients based on rankings
        for strategy in strategies:
            strategy_to_coefficient[strategy.__name__] = rank_to_coefficient[
                rank[strategy.__name__]
            ]
        
        # Initialize buy heaps for this trading day
        buy_heap = []
        suggestion_heap = []
        
        # Process each ticker
        for ticker in tickers:
            # Get current price for this ticker
            key = (ticker, current_date)
            if key in ticker_price_history.index:
                current_price = ticker_price_history.loc[key, 'Close']
            else:
                continue
            
            # Check stop loss and take profit conditions
            account = check_stop_loss_take_profit(account, ticker, current_price)

            # Collect decisions and quantities from all strategies
            decisions_and_quantities = []
            portfolio_qty = account["holdings"].get(ticker, {}).get("quantity", 0)

            # Process each strategy's decision for the current ticker
            for strategy in strategies:
                strategy_name = strategy.__name__
                
                # Get precomputed strategy decision
                num_action = precomputed_decisions.at[key, strategy_name]
                
                # Convert numeric action to string action
                if num_action == 1: 
                    action = 'Buy'
                elif num_action == -1:
                    action = 'Sell'
                else:
                    action = 'Hold'
                
                account_cash = account["cash"]
                total_portfolio_value = account["total_portfolio_value"]

                # Compute trade decision and quantity
                decision, qty = compute_trade_quantities(
                    action,
                    current_price,
                    account_cash,
                    portfolio_qty,
                    total_portfolio_value,
                )

                # Add decision to the list with its weight
                weight = strategy_to_coefficient[strategy.__name__]
                decisions_and_quantities.append((decision, qty, weight))

            # Calculate weighted majority decision
            (
                decision,
                quantity,
                buy_weight,
                sell_weight,
                hold_weight,
            ) = weighted_majority_decision_and_median_quantity(
                decisions_and_quantities
            )

            # Execute trading decision based on majority vote
            if (
                decision == "buy"
                and ((portfolio_qty + quantity) * current_price) / account["total_portfolio_value"]
                <= train_trade_asset_limit
            ):
                # Add buy order to priority queue
                heapq.heappush(
                    buy_heap,
                    (
                        -(buy_weight - (sell_weight + (hold_weight * 0.5))),
                        quantity,
                        ticker,
                    ),
                )

            elif decision == "sell" and ticker in account["holdings"]:
                # Execute sell order immediately
                quantity = max(quantity, 1)
                quantity = account["holdings"][ticker]["quantity"]
                account["trades"].append(
                    {
                        "symbol": ticker,
                        "quantity": quantity,
                        "price": current_price,
                        "action": "sell",
                        "date": current_date.strftime("%Y-%m-%d"),
                    }
                )
                account["cash"] += quantity * current_price
                del account["holdings"][ticker]

            elif (
                portfolio_qty == 0.0
                and buy_weight > sell_weight
                and ((quantity * current_price) / account["total_portfolio_value"]) < trade_asset_limit
                and float(account["cash"]) >= train_trade_liquidity_limit
            ):
                # Add suggested buy to the suggestion heap
                max_investment = account["total_portfolio_value"] * train_trade_asset_limit
                buy_quantity = min(
                    int(max_investment // current_price),
                    int(account["cash"] // current_price),
                )
                if buy_weight > train_suggestion_heap_limit:
                    buy_quantity = max(2, buy_quantity)
                    buy_quantity = buy_quantity // 2
                    heapq.heappush(
                        suggestion_heap,
                        (-(buy_weight - sell_weight), buy_quantity, ticker),
                    )

        # Execute buy orders for this trading day
        account = execute_buy_orders(
            buy_heap, suggestion_heap, account, ticker_price_history, current_date
        )
        
        # Update strategy simulations for the day
        trading_simulator, points = simulate_trading_day(
            current_date,
            ticker_price_history.copy(),
            precomputed_decisions.copy(),
            strategies,
            tickers,
            trading_simulator,
            points,
            time_delta,
            logger
        )
        
        # Update portfolio values for all strategies
        active_count, trading_simulator = local_update_portfolio_values(
            current_date, strategies, trading_simulator, ticker_price_history, logger
        )

        # Update time delta according to specified mode
        time_delta = update_time_delta(time_delta, train_time_delta_mode)

        # Calculate and update total portfolio value
        total_value = account["cash"]
        for ticker in account["holdings"]:
            key = (ticker, current_date)
            if key in ticker_price_history.index:
                current_price = ticker_price_history.loc[key, 'Close']
                total_value += account["holdings"][ticker]["quantity"] * current_price
        
        account["total_portfolio_value"] = total_value
        account_values[current_date] = total_value

        # Update strategy rankings
        rank = update_strategy_ranks(strategies, points, trading_simulator)

        # Log daily results
        logger.info("-------------------------------------------------")
        logger.info(f"Account Cash: ${account['cash']: ,.2f}")
        logger.info(f"Total Portfolio Value: ${account['total_portfolio_value']: ,.2f}")
        logger.info(f"Active Count: {active_count}")
        logger.info("-------------------------------------------------")

        # Move to next day
        current_date += timedelta(days=1)

    # Calculate final metrics and generate tear sheet
    metrics = calculate_metrics(account_values)
    wandb.log(metrics)
    
    # Generate performance visualization
    generate_tear_sheet(account_values, filename=f"{benchmark_asset}_vs_strategy")
    
    # Log final results
    logger.info("Testing Completed")
    logger.info(f"Final Portfolio Value: ${account['total_portfolio_value']:,.2f}")
    logger.info(f"Final Cash Balance: ${account['cash']:,.2f}")
    logger.info(f"Number of Holdings: {len(account['holdings'])}")
    logger.info(f"Performance Metrics:")
    
    # Log key metrics
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
