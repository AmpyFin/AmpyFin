import random
import time

import pandas as pd


def retry_with_backoff(
    func,
    logger,
    max_retries=5,
    base_delay=1,
    max_delay=30,
    exceptions=(Exception,),
    jitter=True,
):
    """
    Retries a function with exponential backoff if it raises specified
     exceptions.

    Parameters:
    - func: The function to execute.
    - max_retries: Maximum number of retries before giving up.
    - base_delay: Initial delay between retries in seconds.
    - max_delay: Maximum delay between retries in seconds.
    - exceptions: A tuple of exception classes that trigger a retry.
    - jitter: Whether to add random jitter to the delay to avoid thundering
      herd problem.
    - logger: Optional logger object with .warning() and .error() methods.
      Falls back to print if None.

    Returns:
    - The return value of func if successful.

    Raises:
    - The last exception raised by func if all retries fail.
    """
    func_name = func.__name__
    attempt = 0
    while attempt <= max_retries:
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries:
                msg = f"""Max retries reached.
                  Function '{func_name}' failed with exception: {e}"""
                if logger:
                    logger.error(msg)
                else:
                    print(msg)
                raise
            else:
                delay = min(max_delay, base_delay * (2**attempt))
                if jitter:
                    delay = delay * (
                        0.5 + random.random() / 2
                    )  # random between 50% and 100% of delay
                msg = f"""Attempt {attempt + 1} for function '{func_name}'
                  failed with {e}. Retrying in {delay:.2f} seconds..."""
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
                time.sleep(delay)
                attempt += 1


def get_ndaq_tickers():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    df = tables[4]  # NASDAQ-100 companies table
    return df["Ticker"].tolist()


if __name__ == "__main__":
    ...
