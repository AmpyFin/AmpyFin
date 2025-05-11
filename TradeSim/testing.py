import heapq
import json
import os
import time
from datetime import datetime, timedelta

import certifi
import pandas as pd
from pymongo import MongoClient
from variables import config_dict

import wandb
from config import mongo_url
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

train_tickers
from helper_files.client_helper import get_ndaq_tickers, strategies
from helper_files.train_client_helper import (
    calculate_metrics,
    generate_tear_sheet,
    local_update_portfolio_values,
)
from TradeSim.utils import (
    compute_trade_quantities,
    simulate_trading_day,
    update_time_delta,
    weighted_majority_decision_and_median_quantity, 
    fetch_price_from_db,
    fetch_strategy_decisions,
)
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

ca = certifi.where()
mongo_client = MongoClient(mongo_url, tlsCAFile=ca)


def initialize_test_account():
    """
    Initializes the test trading account with starting parameters
    """
    return {
        "holdings": {},
        "cash": train_start_cash,
        "trades": [],
        "total_portfolio_value": train_start_cash,
    }


def check_stop_loss_take_profit(account, ticker, current_price):
    """
    Checks and executes stop loss and take profit orders
    """
    if ticker in account["holdings"]:
        if account["holdings"][ticker]["quantity"] > 0:
            if (
                current_price < account["holdings"][ticker]["stop_loss"]
                or current_price > account["holdings"][ticker]["take_profit"]
            ):
                account["trades"].append(
                    {
                        "symbol": ticker,
                        "quantity": account["holdings"][ticker]["quantity"],
                        "price": current_price,
                        "action": "sell",
                    }
                )
                account["cash"] += (
                    account["holdings"][ticker]["quantity"] * current_price
                )
                del account["holdings"][ticker]
    return account


def execute_buy_orders(
    buy_heap, suggestion_heap, account, ticker_price_history, current_date
):
    """
    Executes buy orders from the buy and suggestion heaps
    """
    while (buy_heap or suggestion_heap) and float(
        account["cash"]
    ) > train_trade_liquidity_limit:
        if buy_heap and float(account["cash"]) > train_trade_liquidity_limit:
            heap = buy_heap
        elif suggestion_heap and float(account["cash"]) > train_trade_liquidity_limit:
            heap = suggestion_heap
        else:
            break

        _, quantity, ticker = heapq.heappop(heap)
        # print(f"Executing BUY order for {ticker} of quantity {quantity}")

        # Use the passed dataframe ticker_price_history to get the current price
        key = (ticker, current_date)
        if key in ticker_price_history.index:
            
            current_price = ticker_price_history.loc[key, 'Close']
            # print(ticker, current_date, current_price)
        else:
            current_price = None
            print(f'No price for {ticker} on {current_date}. Skipping.')
            continue

        account["trades"].append(
            {
                "symbol": ticker,
                "quantity": quantity,
                "price": current_price,
                "action": "buy",
                "date": current_date.strftime("%Y-%m-%d"),
            }
        )

        account["cash"] -= quantity * current_price
        account["holdings"][ticker] = {
            "quantity": quantity,
            "price": current_price,
            "stop_loss": current_price * (1 - train_stop_loss),
            "take_profit": current_price * (1 + train_take_profit),
        }

    return account


def update_strategy_ranks(strategies, points, trading_simulator):
    """
    Updates strategy rankings based on performance
    """
    rank = {}
    q = []
    for strategy in strategies:
        if points[strategy.__name__] > 0:
            score = (
                points[strategy.__name__] * 2
                + trading_simulator[strategy.__name__]["portfolio_value"]
            )
        else:
            score = trading_simulator[strategy.__name__]["portfolio_value"]

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

    coeff_rank = 1
    while q:
        _, _, _, strategy_name = heapq.heappop(q)
        rank[strategy_name] = coeff_rank
        coeff_rank += 1

    return rank


def test(
    mongo_client, logger
):
    """
    Runs the testing phase of the trading simulator.
    """
    global train_tickers
    logger.info("Starting testing phase...")

    # Get rank coefficients from database
    db = mongo_client.trading_simulator
    r_t_c = db.rank_to_coefficient
    rank_to_coefficient = {doc["rank"]: doc["coefficient"] for doc in r_t_c.find({})}
    logger.info("Rank coefficients retrieved from database.")

    # Load saved results
    logger.info("Loading saved training results...")
    with open(
        os.path.join(results_dir, f"{config_dict['experiment_name']}.json"), "r"
    ) as json_file:
        results = json.load(json_file)
        trading_simulator = results["trading_simulator"]
        points = results["points"]
        time_delta = results["time_delta"]
    logger.info("Training results loaded successfully.")

    # Initialize testing variables
    strategy_to_coefficient = {}
    account = initialize_test_account()
    rank = update_strategy_ranks(strategies, points, trading_simulator)
    
    # start_date = datetime.strptime(test_period_start, "%Y-%m-%d")
    # end_date = datetime.strptime(test_period_end, "%Y-%m-%d")
    # current_date = start_date

    start_date = pd.to_datetime(test_period_start, format="%Y-%m-%d")
    end_date = pd.to_datetime(test_period_end, format="%Y-%m-%d")
    current_date = start_date
    
    account_values = pd.Series(index=pd.date_range(start=start_date, end=end_date))
    
    # Fetch price data for the specified date range
    ticker_price_history = fetch_price_from_db(
        start_date - timedelta(days=1), end_date, train_tickers)
    ticker_price_history['Date'] = pd.to_datetime(ticker_price_history['Date'], format="%Y-%m-%d")
    ticker_price_history.set_index(['Ticker', 'Date'], inplace=True)
    # print(ticker_price_history)

    # Get data from start_date to end_date for all tickers in train_tickers and also filter by strategies. Get only those strategies that are in the strategies list
    # Preload and use them
    precomputed_decisions = fetch_strategy_decisions( 
        start_date - timedelta(days=1),
        end_date,
        train_tickers,
        strategies,
    ) 
    precomputed_decisions['Date'] = pd.to_datetime(precomputed_decisions['Date'], format="%Y-%m-%d")
    precomputed_decisions.set_index(['Ticker', 'Date'], inplace=True)

    # print(ticker_price_history)
    # print(precomputed_decisions)
    logger.info(f"Testing period: {start_date} to {end_date}")
    if not train_tickers:
        train_tickers = get_ndaq_tickers()

    # dates = ticker_price_history["Date"].unique()
    dates = ticker_price_history.index.get_level_values(1).unique()
    dates = [date.strftime("%Y-%m-%d") for date in dates]
    dates = sorted(dates)
    
    while current_date <= end_date:
        # print(f"Processing date: {current_date.strftime('%Y-%m-%d')}")

        # Skip non-trading days
        if current_date.strftime("%Y-%m-%d") not in dates:
            print(
                f"Skipping {current_date.strftime('%Y-%m-%d')} (weekend or missing data)."
            )
            current_date += timedelta(days=1)
            continue

        # Update strategy coefficients
        for strategy in strategies:
            strategy_to_coefficient[strategy.__name__] = rank_to_coefficient[
                rank[strategy.__name__]
            ]
        # logger.info(f"Strategy coefficients updated: {strategy_to_coefficient}")
        
        # print(ticker_price_history)
      
        # Process trading day
        buy_heap, suggestion_heap = [], []
        # date_str = current_date.strftime("%Y-%m-%d")
        # current_date = pd.to_datetime(date_str)
        for ticker in train_tickers:
            # print(' Processing Ticker:', ticker)
            key = (ticker, current_date)
            if key in ticker_price_history.index:
                current_price = ticker_price_history.loc[key, 'Close']
                print(f"{ticker} Price: {current_price}")
            else:
                print(f'No price for {ticker} on {current_date}. Skipping.')
                continue
            
            # logger.info(f"{ticker} - Current price: {current_price}")
            # print(f"{ticker} - Current price: {current_price}")
            # Check stop loss and take profit
            account = check_stop_loss_take_profit(account, ticker, current_price)

            # Get strategy decisions
            decisions_and_quantities = []
            portfolio_qty = account["holdings"].get(ticker, {}).get("quantity", 0)

            for strategy in strategies:
                strategy_name = strategy.__name__
                
                # Get precomputed strategy decision
                num_action = precomputed_decisions.at[key, strategy_name]
                print(f"IN TESTING - {strategy_name}: {ticker} - {num_action}")
                
                if num_action == 1: 
                    action = 'Buy'
                elif num_action == -1:
                    action = 'Sell'
                else:
                    action = 'Hold'
                
                # print(ticker, strategy_name, action)
                account_cash = account["cash"]
                total_portfolio_value = account["total_portfolio_value"]

                # Compute trade decision and quantity based on precomputed action
                decision, qty = compute_trade_quantities(
                    action,
                    current_price,
                    account_cash,
                    portfolio_qty,
                    total_portfolio_value,
                )

                weight = strategy_to_coefficient[strategy.__name__]
                decisions_and_quantities.append((decision, qty, weight))

            print(decisions_and_quantities)
            # Process weighted decisions
            (
                decision,
                quantity,
                buy_weight,
                sell_weight,
                hold_weight,
            ) = weighted_majority_decision_and_median_quantity(
                decisions_and_quantities
            )

            # print(
            #     f"{ticker} - Decision: {decision}, Quantity: {quantity}, Buy Weight: {buy_weight}, Sell Weight: {sell_weight}, Hold Weight: {hold_weight}"
            # )

            # Execute trading decisions
            if (
                decision == "buy"
                and ((portfolio_qty + quantity) * current_price)
                / account["total_portfolio_value"]
                <= train_trade_asset_limit
            ):
                heapq.heappush(
                    buy_heap,
                    (
                        -(buy_weight - (sell_weight + (hold_weight * 0.5))),
                        quantity,
                        ticker,
                    ),
                )

            elif decision == "sell" and ticker in account["holdings"]:
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
                logger.info(
                    f"{ticker} - Sold {quantity} shares at ${current_price}"
                )

            elif (
                portfolio_qty == 0.0
                and buy_weight > sell_weight
                and ((quantity * current_price) / account["total_portfolio_value"])
                < trade_asset_limit
                and float(account["cash"]) >= train_trade_liquidity_limit
            ):
                max_investment = (
                    account["total_portfolio_value"] * train_trade_asset_limit
                )
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

        
        # Execute buy orders
        account = execute_buy_orders(
            buy_heap, suggestion_heap, account, ticker_price_history, current_date
        )
        # print(account)
       
        # Simulate ranking updates

        trading_simulator, points = simulate_trading_day(
            current_date,
            ticker_price_history.copy(),
            precomputed_decisions.copy(),
            strategies,
            train_tickers,
            logger,
            trading_simulator,
            points,
            time_delta
        )
        print(trading_simulator)
        # Update portfolio values
        # Rewrite the function call with correct parameters from train_client_helper.py
        active_count, trading_simulator = local_update_portfolio_values(
            current_date, strategies, trading_simulator, ticker_price_history, logger
        )

        # Update time delta
        time_delta = update_time_delta(time_delta, train_time_delta_mode)

        # Calculate and update total portfolio value
        total_value = account["cash"]
        for ticker in account["holdings"]:
            key = (ticker, current_date)
            if key in ticker_price_history.index:
                current_price = ticker_price_history.loc[key, 'Close']
            total_value += account["holdings"][ticker]["quantity"] * current_price
        account["total_portfolio_value"] = total_value

        # Update account values for metrics
        account_values[current_date] = total_value

        # Update rankings
        rank = update_strategy_ranks(strategies, points, trading_simulator)

        # Log daily results
        logger.info("-------------------------------------------------")
        logger.info(f"Account Cash: ${account['cash']: ,.2f}")
        logger.info(f"Trades: {account['trades']}")
        logger.info(f"Holdings: {account['holdings']}")
        logger.info(f"Total Portfolio Value: ${account['total_portfolio_value']: ,.2f}")
        logger.info(f"Active Count: {active_count}")
        logger.info("-------------------------------------------------")

        current_date += timedelta(days=1)
        print('**'*50)
        # time.sleep(5)



    # Calculate final metrics and generate tear sheet
    metrics = calculate_metrics(account_values)
    wandb.log(metrics)

    logger.info("Final metrics calculated.")
    logger.info(metrics)

    generate_tear_sheet(account_values, filename=f"{benchmark_asset}_vs_strategy")
    logger.info("Tear sheet generated.")

    # Print final results
    logger.info("Testing Completed.")
    logger.info("-------------------------------------------------")
    logger.info(f"Account Cash: ${account['cash']: ,.2f}")
    logger.info(f"Total Portfolio Value: ${account['total_portfolio_value']: ,.2f}")
    logger.info("-------------------------------------------------")
