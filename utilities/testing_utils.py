import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs
from control import benchmark_asset
import os
import pandas as pd

def calculate_metrics(account_values: pd.Series) -> dict:
    """Calculates various performance metrics for a given account value series.

        Args:
            account_values (pd.Series): A pandas Series representing the account values over time.

        Returns:
            dict: A dictionary containing the calculated metrics:
                - sharpe_ratio (float): The Sharpe Ratio.
                - sortino_ratio (float): The Sortino Ratio.
                - max_drawdown (float): The Max Drawdown.
                - r_ratio (float): The R Ratio (mean return / standard deviation).
        """
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


def generate_tear_sheet(account_values: pd.Series, filename: str) -> None:
    """Generates a tear sheet for the given account values.

        Args:
            account_values (pd.Series): A pandas Series containing the account values over time.
            filename (str): The name of the file to save the tear sheet to.

        Returns:
            None
        """
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
