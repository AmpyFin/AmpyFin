import heapq
import time
import os 
import sys

import certifi
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from pymongo import MongoClient

# Ensure sys.path manipulation is at the top, before other local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.common_utils import weighted_majority_decision_and_median_quantity, get_ndaq_tickers
from config import API_KEY, API_SECRET, MONGO_URL

from control import suggestion_heap_limit, trade_asset_limit, trade_liquidity_limit, train_tickers
from utilities.ranking_trading_utils import ( 
    get_latest_price,
    market_status,
    place_order,
    strategies,
)
# from strategies.categorise_talib_indicators_vect import strategies
from strategies.talib_indicators import get_data, simulate_strategy
from utilities.logging import setup_logging
logger = setup_logging(__name__)

buy_heap = []
suggestion_heap = []
sold = False

ca = certifi.where()

def process_ticker(ticker: str, trading_client: TradingClient, mongo_client: MongoClient, indicator_periods: dict, strategy_to_coefficient: dict) -> None:  
    """Processes a single ticker symbol, making trading decisions based on strategy evaluations and risk management.
        Args:
            ticker (str): The ticker symbol to process.
            trading_client ( Alpaca Trade API): The Alpaca trading client for placing orders.
            mongo_client (MongoClient): The MongoDB client for accessing historical data and storing trade information.
            indicator_periods (dict): A dictionary mapping strategy names to the period of historical data required.
            strategy_to_coefficient (dict): A dictionary mapping strategy names to their weighting coefficient.
        Returns:
            None
        Raises:
            Exception: If there is an error fetching the price or historical data.
        """
    logger.info(f"Processing ticker: {ticker}")
    global buy_heap, suggestion_heap, sold
    if sold is True:
        print("Sold boolean is True. Exiting process_ticker function.")
   
    # 1) fetch price
    try:
        current_price = get_latest_price(ticker)
    except Exception:
        logger.warning(f"Price fetch failed for {ticker}.")
        return

    # 2) check stop-loss/take-profit using the cached limits…
    asset_coll = mongo_client.trades.assets_quantities
    limits_coll = mongo_client.trades.assets_limit
    
    asset_info = asset_coll.find_one({"symbol": ticker})
    portfolio_qty = asset_info["quantity"] if asset_info else 0.0

    limit_info = limits_coll.find_one({"symbol": ticker})
    if limit_info:
        stop_loss_price = limit_info["stop_loss_price"]
        take_profit_price = limit_info["take_profit_price"]
        if (
            current_price <= stop_loss_price
            or current_price >= take_profit_price
        ):
            sold = True
            print(
                f"Executing SELL order for {ticker} due to stop-loss or take-profit condition"
            )
            quantity = portfolio_qty
            # Correct it such that once order is finished, then log it to mongoDB
            # Complete sync of Alpaca and MongoDB
            order = place_order(
                trading_client,
                symbol=ticker,
                side=OrderSide.SELL,
                quantity=quantity,
                mongo_client=mongo_client,
            )
            logger.info(f"Executed SELL order for {ticker}: {order}")
            return

    # 3) gather strategy decisions…
    decisions_and_quantities = []

    account = trading_client.get_account()
    buying_power = float(account.cash)
    portfolio_value = float(account.portfolio_value)
   
    for strategy in strategies:
        historical_data = None
        while historical_data is None:
            try:
                period = indicator_periods[strategy.__name__]
                # Get historical data from SQLite DBs - price DB
                # instead of using get_data()
                historical_data = get_data(ticker, mongo_client, period)
            except Exception as fetch_error:
                logger.warning(
                    f"Error fetching historical data for {ticker}. Retrying... {fetch_error}"
                )
                time.sleep(60)
        
        # In future no need to simulate strategy. 
        # We already have the decision in SQLite DB
        # Get decision from SQLite DB and compute quantity
        decision, quantity = simulate_strategy(
            strategy,
            ticker,
            current_price,
            historical_data,
            buying_power,
            portfolio_qty,
            portfolio_value,
        )
        print(
            f"Strategy: {strategy.__name__}, Decision: {decision}, Quantity: {quantity} for {ticker}"
        )
        weight = strategy_to_coefficient[strategy.__name__]
        decisions_and_quantities.append((decision, quantity, weight))

    # 4) weighted majority decision…
    decision, quantity, buy_w, sell_w, hold_w = weighted_majority_decision_and_median_quantity(decisions_and_quantities)
    print(
        f"Ticker: {ticker}, Decision: {decision}, Quantity: {quantity}, Weights: Buy: {buy_w}, Sell: {sell_w}, Hold: {hold_w}"
    )

    if (
        decision == "buy"
        and float(account.cash) > trade_liquidity_limit
        and (((quantity + portfolio_qty) * current_price) / portfolio_value)
        < trade_asset_limit
    ):
        heapq.heappush(buy_heap,(-(buy_w - (sell_w + (hold_w * 0.5))),quantity,ticker))
    elif decision == "sell" and portfolio_qty > 0:
        print(f"Executing SELL order for {ticker}")
        print(f"Executing quantity of {quantity} for {ticker}")
        sold = True
        quantity = max(quantity, 1)
        order = place_order(
            trading_client,
            symbol=ticker,
            side=OrderSide.SELL,
            quantity=quantity,
            mongo_client=mongo_client,
        )
        logger.info(f"Executed SELL order for {ticker}: {order}")
    elif (
        portfolio_qty == 0.0
        and buy_w > sell_w
        and (((quantity + portfolio_qty) * current_price) / portfolio_value)
        < trade_asset_limit
        and float(account.cash) > trade_liquidity_limit
        ):
        max_investment = portfolio_value * trade_asset_limit
        buy_quantity = min(
            int(max_investment // current_price),
            int(buying_power // current_price),
        )
        if buy_w > suggestion_heap_limit:
            buy_quantity = max(buy_quantity, 2)
            buy_quantity = buy_quantity // 2
            print(
                f"Suggestions for buying for {ticker} with a weight of {buy_w} and quantity of {buy_quantity}"
            )
            heapq.heappush(suggestion_heap,(-(buy_w - sell_w), buy_quantity, ticker))
        else:
            logger.info(f"Holding for {ticker}, no action taken.")
    else:
        logger.info(f"Holding for {ticker}, no action taken.")

def execute_buy_orders(mongo_client: MongoClient, trading_client: TradingClient) -> None:
    """Executes buy orders from the buy_heap and suggestion_heap.

        This function iterates through the buy_heap and suggestion_heap, placing buy orders
        as long as there is sufficient cash in the account and the 'sold' flag is not set.
        It uses the Alpaca trading client to place the orders and logs the execution.

        Args:
            mongo_client: A MongoDB client instance for database operations.
            trading_client: An Alpaca trading client instance for placing orders.

        Returns:
            None.

        Raises:
            Exception: If an error occurs during the execution of a buy order.
                The error is logged, and the loop continues.

        Notes:
            - The function uses global variables buy_heap, suggestion_heap, and sold.
            - It pauses for 5 seconds after each order to allow the order to propagate and
              ensure an accurate cash balance.
            - After processing all orders, it clears the buy_heap and suggestion_heap and
              resets the sold flag.
    """
    logger.info("Executing buy orders from heaps.")

    global buy_heap, suggestion_heap, sold
    account = trading_client.get_account()
    while (
        (buy_heap or suggestion_heap)
        and float(account.cash) > trade_liquidity_limit
        and not sold
    ):
        try:
            if buy_heap and float(account.cash) > trade_liquidity_limit:
                _, quantity, ticker = heapq.heappop(buy_heap)
                print(f"Executing BUY order for {ticker} of quantity {quantity}")

                order = place_order(
                    trading_client,
                    symbol=ticker,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    mongo_client=mongo_client,
                )
                logger.info(f"Executed BUY order for {ticker}")

            elif (suggestion_heap and float(account.cash) > trade_liquidity_limit):
                _, quantity, ticker = heapq.heappop(suggestion_heap)
                print(f"Executing BUY order for {ticker} of quantity {quantity}")

                order = place_order(
                    trading_client,
                    symbol=ticker,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    mongo_client=mongo_client,
                )
                logger.info(f"Executed BUY order for {ticker}")

            time.sleep(5)
            """
            This is here so order will propage through and we will have an accurate cash balance recorded
            """
        except Exception as e:
            print(f"Error occurred while executing buy order due to {e}. Continuing...")
            break
    buy_heap = []
    suggestion_heap = []
    sold = False

def load_indicator_periods(mongo_client: MongoClient) -> dict:
    """Loads indicator periods from MongoDB.

        Connects to the MongoDB database, retrieves indicator documents,
        and extracts the indicator name and ideal period for each.

        Args:
            mongo_client: A MongoDB client instance connected to the database.

        Returns:
            A dictionary where keys are indicator names and values are their
            corresponding ideal periods.
            For example: {'SMA': 20, 'RSI': 14}
        """
    coll = mongo_client.IndicatorsDatabase.Indicators
    # one query for all docs
    return {
        doc["indicator"]: doc["ideal_period"]
        for doc in coll.find({}, {"indicator": 1, "ideal_period": 1})
    }

def initialize_strategy_coefficients(mongo_client: MongoClient) -> dict:
    """Initializes a dictionary mapping strategy names to their corresponding coefficients.

        This function retrieves the rank associated with each strategy from the 'rank' collection
        in the MongoDB database, and then uses that rank to fetch the corresponding coefficient
        from the 'rank_to_coefficient' collection.  The strategy name and coefficient are then
        stored as a key-value pair in the returned dictionary.

        Args:
                mongo_client: A PyMongo MongoClient instance connected to the MongoDB database.
                        The database is expected to have collections named 'rank' and 'rank_to_coefficient'.

        Returns:
                A dictionary where keys are strategy names (strings) and values are their
                corresponding coefficients (numbers).

        Raises:
                KeyError: If a strategy's rank or coefficient cannot be found in the database.
        """
    strategy_to_coefficient = {}
    sim_db = mongo_client.trading_simulator
    rank_collection = sim_db.rank
    r_t_c_collection = sim_db.rank_to_coefficient

    for strategy in strategies:
        print(f"Processing strategy: {strategy.__name__}")
        rank = rank_collection.find_one({"strategy": strategy.__name__})["rank"]
        coefficient = r_t_c_collection.find_one({"rank": rank})["coefficient"]
        strategy_to_coefficient[strategy.__name__] = coefficient

    return strategy_to_coefficient

def process_market_open(mongo_client: MongoClient) -> None:
    """Processes the market open, including ticker processing and order execution.
        This function performs the following steps:
        1. Logs that the market is open and processing tickers.
        2. If there are no tickers to train, it pulls NASDAQ tickers.
        3. Loads indicator periods and initializes strategy coefficients from MongoDB.
        4. Initializes a TradingClient.
        5. Initializes empty lists for buy_heap and suggestion_heap, and sets sold to False.
        6. Iterates through each ticker in train_tickers, processing it and pausing briefly.
        7. Executes buy orders.
        8. Sleeps for 30 seconds.
        Args:
            mongo_client: A MongoDB client instance for database interactions.
        Returns:
            None.
        """
    logger.info("Market is open. Processing tickers.")
    global buy_heap, suggestion_heap, sold, train_tickers
    if not train_tickers:
        logger.info("No tickers to train. Pulling NASDAQ tickers.")
        train_tickers = get_ndaq_tickers()
    
    indicator_periods = load_indicator_periods(mongo_client)
    strategy_to_coefficient = initialize_strategy_coefficients(mongo_client)
    print(f"Strategy to coefficient mapping: {strategy_to_coefficient}")
    trading_client = TradingClient(API_KEY, API_SECRET)

    buy_heap = []
    suggestion_heap = []
    sold = False

    for ticker in train_tickers:
        process_ticker(ticker, trading_client, mongo_client, indicator_periods, strategy_to_coefficient)
        time.sleep(0.5)

    execute_buy_orders(mongo_client, trading_client)
    print("Sleeping for 30 seconds...")
    time.sleep(30)

def process_early_hours():
    """
    Handle operations during market early hours.
    """
    logger.info("Market is in early hours. Waiting for 30 seconds.")
    time.sleep(30)


def process_market_closed():
    """
    Handle operations when the market is closed.
    """
    logger.info("Market is closed. Performing post-market operations.")
    time.sleep(30)

def main():
    """Main function to run the trading simulation.

        This function continuously checks the market status and performs actions
        based on whether the market is open, in early hours, or closed. It also
        handles exceptions and logs any errors that occur.

        Raises:
            Exception: If an unexpected error occurs in the main trading loop.
        """
    logger.info("Trading mode is live.")
    mongo_client = MongoClient(MONGO_URL, tlsCAFile=ca)

    while True:
        try:
            status = market_status()
            print(f"Market status: {status}")
            market_db = mongo_client.market_data
            market_collection = market_db.market_status
            market_collection.update_one({}, {"$set": {"market_status": status}})

            if status == "open":
                process_market_open(mongo_client)
            elif status == "early_hours":
                process_early_hours()
            elif status == "closed":
                process_market_closed()
            else:
                logger.error("An error occurred while checking market status.")
                time.sleep(60)

        except Exception as e:
            logger.error(f"Unexpected error in main trading loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
