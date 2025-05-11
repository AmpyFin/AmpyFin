import heapq
import json
import os
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
from variables import config_dict

import wandb

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from control import (
    train_period_end,
    train_period_start,
    train_tickers,
    train_time_delta,
    train_time_delta_mode,
)

train_tickers
from helper_files.client_helper import get_ndaq_tickers, strategies
from helper_files.train_client_helper import local_update_portfolio_values
from TradeSim.utils import simulate_trading_day, update_time_delta, fetch_price_from_db, fetch_strategy_decisions

results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


def train(
     logger
):
    """
    get from ndaq100
    """
    global train_tickers
    if not train_tickers:
        train_tickers = get_ndaq_tickers()
        logger.info(f"Fetched {len(train_tickers)} tickers.")

    logger.info(f"Ticker price history initialized for {len(train_tickers)} tickers.")
    # logger.info(f"Ideal period determined: {ideal_period}")

    trading_simulator = {
        strategy.__name__: {
            "holdings": {},
            "amount_cash": 50000,
            "total_trades": 0,
            "successful_trades": 0,
            "neutral_trades": 0,
            "failed_trades": 0,
            "portfolio_value": 50000,
        }
        for strategy in strategies
    }

    points = {strategy.__name__: 0 for strategy in strategies}
    time_delta = train_time_delta

    logger.info("Trading simulator and points initialized.")
    start_date = pd.to_datetime(train_period_start, format="%Y-%m-%d")
    end_date = pd.to_datetime(train_period_end, format="%Y-%m-%d")
    current_date = start_date
    # print(type(current_date))

    #write a query to get all price data from start_date to end_date for all tickers in train_tickers

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
    # print(type(precomputed_decisions.index.get_level_values(1).loc[0]))
    # print(precomptued_decisions['Date'].loc[0])
    # print(type(ticker_price_history.index.get_level_values(1).loc[0]))

    # print("Ticker price history index types:")
    # for level, level_name in enumerate(ticker_price_history.index.names):
    #     print(f"Level {level} ({level_name}): {type(ticker_price_history.index.get_level_values(level)[0])}")

    # print("Precomputed decisions index types:")
    # for level, level_name in enumerate(precomputed_decisions.index.names):
    #     print(f"Level {level} ({level_name}): {type(precomputed_decisions.index.get_level_values(level)[0])}")
    # print(ticker_price_history)
    # print(precomputed_decisions)

    # print(f"Training period: {start_date} to {end_date}")
    # print(current_date, end_date)
    dates = ticker_price_history.index.get_level_values(1).unique()
    dates = [date.strftime("%Y-%m-%d") for date in dates]
    dates = sorted(dates)
    # print(dates)
    while current_date <= end_date:
        print(f"Processing date: {current_date.strftime('%Y-%m-%d')}")
        
        if current_date.strftime("%Y-%m-%d") not in dates :
            print(
                f"Skipping {current_date.strftime('%Y-%m-%d')} (weekend or missing data)."
            )
            current_date += timedelta(days=1)
            continue
        
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
        # print('Before update', trading_simulator)
        active_count, trading_simulator = local_update_portfolio_values(
            current_date, strategies, trading_simulator, ticker_price_history.copy(), logger
        )
        print('After update', trading_simulator)

       
        logger.info(f"Trading simulator: {trading_simulator}")
        logger.info(f"Points: {points}")
        logger.info(f"Date: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"time_delta: {time_delta}")
        logger.info(f"Active count: {active_count}")
        logger.info("-------------------------------------------------")

        # Update time delta
        time_delta = update_time_delta(time_delta, train_time_delta_mode)
        logger.info(f"Updated time delta: {time_delta}")

        # Move to next day
        current_date += timedelta(days=1)
        print('**'*50)
        # time.sleep(5)

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info(f"Created results directory: {results_dir}")

    results = {
        "trading_simulator": trading_simulator,
        "points": points,
        "date": current_date.strftime("%Y-%m-%d"),
        "time_delta": time_delta,
    }

    result_filename = f"{config_dict['experiment_name']}.json"
    results_file_path = os.path.join(results_dir, result_filename)
    with open(results_file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    # Create an artifact
    artifact = wandb.Artifact(result_filename, type="results")
    artifact.add_file(results_file_path)

    # Log artifact to the current run
    wandb.log_artifact(artifact)

    logger.info(f"Training results saved to {results_file_path}")

    top_portfolio_values = heapq.nlargest(
        10, trading_simulator.items(), key=lambda x: x[1]["portfolio_value"]
    )
    top_points = heapq.nlargest(10, points.items(), key=lambda x: x[1])

    top_portfolio_values_list = []
    logger.info("Top 10 strategies with highest portfolio values:")
    for strategy, value in top_portfolio_values:
        top_portfolio_values_list.append([strategy, value["portfolio_value"]])
        logger.info(f"{strategy} - {value['portfolio_value']}")

    wandb.log({"TRAIN_top_portfolio_values": top_portfolio_values_list})
    wandb.log({"TRAIN_top_points": top_points})

    logger.info("Top 10 strategies with highest points:")
    for strategy, value in top_points:
        logger.info(f"{strategy} - {value}")

    logger.info("Training completed.")
