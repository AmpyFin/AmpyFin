#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training module for the trading simulator.

This module handles the training phase of the trading simulation, including:
- Loading and processing ticker data
- Running simulated trading for each strategy
- Tracking performance metrics
- Saving results for later use in testing
"""

import heapq
import json
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import wandb

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# Local imports
from variables import config_dict
from control import (
    train_period_end,
    train_period_start,
    train_tickers,
    train_time_delta,
    train_time_delta_mode,
)
from utilities.ranking_trading_utils import strategies
import logging
import pandas as pd
from utilities.common_utils import (
    get_ndaq_tickers,
    simulate_trading_day, 
    update_time_delta, 
    fetch_price_from_db, 
    fetch_strategy_decisions,
    local_update_portfolio_values
)

from utilities.logging import setup_logging

logger = setup_logging(__name__)


def train() -> None:
    """
    Execute the training phase of the trading simulator.
    
    This function runs simulated trading for all strategies over the training period,
    evaluates their performance, and saves the results for later use in testing.
    
    Returns:
        None: Results are saved to disk and logged to W&B.
    """
    # Initialize tickers if not provided
    global train_tickers
    if not train_tickers:
        train_tickers = get_ndaq_tickers()
        logger.info(f"Training with {len(train_tickers)} tickers")

    # Initialize trading simulator data structure for each strategy
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

    # Initialize points tracker for each strategy
    points = {strategy.__name__: 0 for strategy in strategies}
    
    # Set initial time delta from configuration
    time_delta = train_time_delta

    logger.info(f"Starting training from {train_period_start} to {train_period_end}")
    
    # Parse date strings to datetime objects
    start_date = pd.to_datetime(train_period_start, format="%Y-%m-%d")
    end_date = pd.to_datetime(train_period_end, format="%Y-%m-%d")
    current_date = start_date

    # Fetch price data for the entire training period
    logger.info("Fetching historical price data and strategy decisions...")
    ticker_price_history = fetch_price_from_db(
        start_date - timedelta(days=1), end_date, train_tickers)
    ticker_price_history['Date'] = pd.to_datetime(ticker_price_history['Date'], format="%Y-%m-%d")
    ticker_price_history.set_index(['Ticker', 'Date'], inplace=True)

    # Preload all strategy decisions for the training period
    precomputed_decisions = fetch_strategy_decisions( 
        start_date - timedelta(days=1),
        end_date,
        train_tickers,
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
        
        # Simulate trading for this day
        trading_simulator, points = simulate_trading_day(
            current_date,
            ticker_price_history.copy(),
            precomputed_decisions.copy(),
            strategies,
            train_tickers,
            trading_simulator,
            points,
            time_delta,
            logger
        )
        
        # Update portfolio values for all strategies
        active_count, trading_simulator = local_update_portfolio_values(
            current_date, 
            strategies, 
            trading_simulator, 
            ticker_price_history.copy(), 
            logger
        )

        # Log weekly summary or at important milestones
        if current_date.weekday() == 4 or current_date == end_date:  # Friday or last day
            logger.info(f"Date: {date_str} | Active positions: {active_count} | Time delta: {time_delta}")

        # Update time delta according to specified mode
        time_delta = update_time_delta(time_delta, train_time_delta_mode)

        # Move to next day
        current_date += timedelta(days=1)
    
    # Prepare and save results
    results = {
        "trading_simulator": trading_simulator,
        "points": points,
        "date": current_date.strftime("%Y-%m-%d"),
        "time_delta": time_delta,
    }

    # Save results to file
    result_filename = f"{config_dict['experiment_name']}.json"
    results_dir = os.path.join('../artifacts', 'results')
    results_file_path = os.path.join(results_dir, result_filename)
    with open(results_file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    # Create W&B artifact for results
    artifact = wandb.Artifact(result_filename, type="results")
    artifact.add_file(results_file_path)
    wandb.log_artifact(artifact)
    logger.info(f"Training results saved to {results_file_path}")

    # Find top performing strategies by portfolio value
    top_portfolio_values = heapq.nlargest(
        10, trading_simulator.items(), key=lambda x: x[1]["portfolio_value"]
    )
    
    # Find top performing strategies by points
    top_points = heapq.nlargest(10, points.items(), key=lambda x: x[1])

    # Format and log results
    top_portfolio_values_list = []
    logger.info("Top 10 strategies by portfolio value:")
    for strategy, value in top_portfolio_values:
        top_portfolio_values_list.append([strategy, value["portfolio_value"]])
        logger.info(f"{strategy}: ${value['portfolio_value']:.2f}")

    # Log to W&B
    wandb.log({"TRAIN_top_portfolio_values": top_portfolio_values_list})
    wandb.log({"TRAIN_top_points": top_points})

    logger.info("Training completed.")