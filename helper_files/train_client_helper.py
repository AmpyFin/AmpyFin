import os
import sys
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs

from control import benchmark_asset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_historical_data(ticker, current_date, period, ticker_price_history):
    period_start_date = {
        "1mo": current_date - timedelta(days=30),
        "3mo": current_date - timedelta(days=90),
        "6mo": current_date - timedelta(days=180),
        "1y": current_date - timedelta(days=365),
        "2y": current_date - timedelta(days=730),
    }
    start_date = period_start_date[period]

    return ticker_price_history[ticker].loc[
        start_date.strftime("%Y-%m-%d") : current_date.strftime("%Y-%m-%d")
    ]


def local_update_portfolio_values(
    current_date, strategies, trading_simulator, ticker_price_history, logger
):
    """
    Updates portfolio values for all strategies and counts active strategies.
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

def calculate_metrics(account_values):
    # Fill non-leading NA values with the previous value using 'ffill' (forward fill)
    account_values_filled = account_values.ffill()
    returns = account_values_filled.pct_change().dropna()
    # Sharpe Ratio
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252)

    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    max_drawdown = (cumulative.cummax() - cumulative).max()

    # R Ratio
    r_ratio = returns.mean() / returns.std()

    return {
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "r_ratio": r_ratio,
    }


def plot_cash_growth(account_values):
    account_values = account_values.interpolate(
        method="linear"
    )  # Fill missing values by linear interpolation
    plt.figure(figsize=(10, 6))
    plt.plot(account_values.index, account_values.values, label="Account Cash Growth")
    plt.xlabel("Date")
    plt.ylabel("Account Value")
    plt.title("Account Cash Growth Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_tear_sheet(account_values, filename):
    # Create 'tearsheets' folder if it doesn't exist
    if not os.path.exists("tearsheets"):
        os.makedirs("tearsheets")

    # Fill missing values by linear interpolation
    account_values = account_values.interpolate(method="linear")

    # Generate quantstats report
    qs.reports.html(
        account_values.pct_change(),
        benchmark=benchmark_asset,
        title=f"Strategy vs {benchmark_asset}",
        output=f"tearsheets/{filename}.html",
    )
