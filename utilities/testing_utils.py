import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs
from control import benchmark_asset
import os

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
    # Fill missing values by linear interpolation
    account_values = account_values.interpolate(method="linear")
    output_path = os.path.join('../artifacts', 'tearsheets', f"{filename}.html")
    # Generate quantstats report
    qs.reports.html(
        account_values.pct_change(),
        benchmark=benchmark_asset,
        title=f"Strategy vs {benchmark_asset}",
        output=output_path,
    )
