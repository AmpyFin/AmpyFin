import heapq
import logging
import threading
import time
from datetime import datetime

import certifi
from pymongo import MongoClient
from config import MONGO_URL
from control import (
    loss_price_change_ratio_d1,
    loss_price_change_ratio_d2,
    loss_profit_time_d1,
    loss_profit_time_d2,
    loss_profit_time_else,
    profit_price_change_ratio_d1,
    profit_price_change_ratio_d2,
    profit_profit_time_d1,
    profit_profit_time_d2,
    profit_profit_time_else,
    rank_asset_limit,
    rank_liquidity_limit,
    time_delta_balanced,
    time_delta_increment,
    time_delta_mode,
    time_delta_multiplicative,
)
from utilities.ranking_trading_utils import get_latest_price, strategies
from utilities.common_utils import get_ndaq_tickers
from strategies.talib_indicators import get_data, simulate_strategy

ca = certifi.where()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("rank_system.log"),  # Log messages to a file
        logging.StreamHandler(),  # Log messages to the console
    ],
)

def update_ranks(client: MongoClient) -> None:
    """Updates the ranking of trading strategies based on their performance.
        This function calculates a score for each strategy based on its total points,
        portfolio value, successful trades, and failed trades. It then ranks the
        strategies based on this score and stores the ranking in the 'rank' collection.
        It also clears the historical database after updating the ranks.
        Args:
            client (MongoClient): A MongoClient instance connected to the MongoDB database.
                The client should have access to the 'trading_simulator' and
                'HistoricalDatabase' databases.
        Returns:
            None. The function updates the 'rank' collection in the
            'trading_simulator' database and clears the 'HistoricalDatabase' database.
        Raises:
            pymongo.errors.PyMongoError: If there is an error interacting with the
                MongoDB database.

    """
    db = client.trading_simulator
    points_collection = db.points_tally
    rank_collection = db.rank
    algo_holdings = db.algorithm_holdings
   
    rank_collection.delete_many({})
  
    q = []
    for strategy_doc in algo_holdings.find({}):
       
        strategy_name = strategy_doc["strategy"]
        if strategy_name == "test" or strategy_name == "test_strategy":
            continue
        if points_collection.find_one({"strategy": strategy_name})["total_points"] > 0:
            heapq.heappush(
                q,
                (
                    points_collection.find_one({"strategy": strategy_name})[
                        "total_points"
                    ]
                    * 2
                    + (strategy_doc["portfolio_value"]),
                    strategy_doc["successful_trades"] - strategy_doc["failed_trades"],
                    strategy_doc["amount_cash"],
                    strategy_doc["strategy"],
                ),
            )
        else:
            heapq.heappush(
                q,
                (
                    strategy_doc["portfolio_value"],
                    strategy_doc["successful_trades"] - strategy_doc["failed_trades"],
                    strategy_doc["amount_cash"],
                    strategy_doc["strategy"],
                ),
            )
    rank = 1
    while q:
        _, _, _, strategy_name = heapq.heappop(q)
        rank_collection.insert_one({"strategy": strategy_name, "rank": rank})
        rank += 1

    db = client.HistoricalDatabase
    collection = db.HistoricalDatabase
    collection.delete_many({})
    print("Successfully updated ranks")
    print("Successfully deleted historical database")

def process_ticker(ticker, mongo_client):
    try:
        current_price = None
        while current_price is None:
            try:
                current_price = get_latest_price(ticker)
            except Exception as fetch_error:
                logging.warning(
                    f"Error fetching price for {ticker}. Retrying... {fetch_error}"
                )

                return

        indicator_tb = mongo_client.IndicatorsDatabase
        indicator_collection = indicator_tb.Indicators
        for strategy in strategies:
            historical_data = None
            while historical_data is None:
                try:
                    period = indicator_collection.find_one(
                        {"indicator": strategy.__name__}
                    )
                    historical_data = get_data(
                        ticker, mongo_client, period["ideal_period"]
                    )
                except Exception as fetch_error:
                    logging.warning(
                        f"Error fetching historical data for {ticker}. Retrying... {fetch_error}"
                    )
                    time.sleep(60)
            db = mongo_client.trading_simulator
            holdings_collection = db.algorithm_holdings
            print(f"Processing {strategy.__name__} for {ticker}")
            strategy_doc = holdings_collection.find_one({"strategy": strategy.__name__})
            if not strategy_doc:
                logging.warning(
                    f"Strategy {strategy.__name__} not in database. Skipping."
                )
                continue

            account_cash = strategy_doc["amount_cash"]
            total_portfolio_value = strategy_doc["portfolio_value"]

            portfolio_qty = strategy_doc["holdings"].get(ticker, {}).get("quantity", 0)

            simulate_trade(
                ticker,
                strategy,
                historical_data,
                current_price,
                account_cash,
                portfolio_qty,
                total_portfolio_value,
                mongo_client,
            )

        print(f"{ticker} processing completed.")
    except Exception as e:
        logging.error(f"Error in thread for {ticker}: {e}")


def simulate_trade(
    ticker,
    strategy,
    historical_data,
    current_price,
    account_cash,
    portfolio_qty,
    total_portfolio_value,
    mongo_client,
):
    """
    Simulates a trade based on the given strategy and updates MongoDB.
    """

    # Simulate trading action from strategy
    print(
        f"Simulating trade for {ticker} with strategy {strategy.__name__} and quantity of {portfolio_qty}"
    )
    action, quantity = simulate_strategy(
        strategy,
        ticker,
        current_price,
        historical_data,
        account_cash,
        portfolio_qty,
        total_portfolio_value,
    )

    # MongoDB setup

    db = mongo_client.trading_simulator
    holdings_collection = db.algorithm_holdings
    points_collection = db.points_tally

    # Find the strategy document in MongoDB
    strategy_doc = holdings_collection.find_one({"strategy": strategy.__name__})
    holdings_doc = strategy_doc.get("holdings", {})
    time_delta = db.time_delta.find_one({})["time_delta"]

    # Update holdings and cash based on trade action
    if (
        action in ["buy"]
        and strategy_doc["amount_cash"] - quantity * current_price
        > rank_liquidity_limit
        and quantity > 0
        and ((portfolio_qty + quantity) * current_price) / total_portfolio_value
        < rank_asset_limit
    ):
        logging.info(
            f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}"
        )
        # Calculate average price if already holding some shares of the ticker
        if ticker in holdings_doc:
            current_qty = holdings_doc[ticker]["quantity"]
            new_qty = current_qty + quantity
            average_price = (
                holdings_doc[ticker]["price"] * current_qty + current_price * quantity
            ) / new_qty
        else:
            new_qty = quantity
            average_price = current_price

        # Update the holdings document for the ticker.
        holdings_doc[ticker] = {"quantity": new_qty, "price": average_price}

        # Deduct the cash used for buying and increment total trades
        holdings_collection.update_one(
            {"strategy": strategy.__name__},
            {
                "$set": {
                    "holdings": holdings_doc,
                    "amount_cash": strategy_doc["amount_cash"]
                    - quantity * current_price,
                    "last_updated": datetime.now(),
                },
                "$inc": {"total_trades": 1},
            },
            upsert=True,
        )

    elif (
        action in ["sell"]
        and str(ticker) in holdings_doc
        and holdings_doc[str(ticker)]["quantity"] > 0
    ):
        logging.info(
            f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}"
        )
        current_qty = holdings_doc[ticker]["quantity"]

        # Ensure we do not sell more than we have
        sell_qty = min(quantity, current_qty)
        holdings_doc[ticker]["quantity"] = current_qty - sell_qty

        price_change_ratio = (
            current_price / holdings_doc[ticker]["price"]
            if ticker in holdings_doc
            else 1
        )

        if current_price > holdings_doc[ticker]["price"]:
            # increment successful trades
            holdings_collection.update_one(
                {"strategy": strategy.__name__},
                {"$inc": {"successful_trades": 1}},
                upsert=True,
            )

            # Calculate points to add if the current price is higher than the purchase price
            if price_change_ratio < profit_price_change_ratio_d1:
                points = time_delta * profit_profit_time_d1
            elif price_change_ratio < profit_price_change_ratio_d2:
                points = time_delta * profit_profit_time_d2
            else:
                points = time_delta * profit_profit_time_else

        else:
            # Calculate points to deduct if the current price is lower than the purchase price
            if holdings_doc[ticker]["price"] == current_price:
                holdings_collection.update_one(
                    {"strategy": strategy.__name__}, {"$inc": {"neutral_trades": 1}}
                )

            else:
                holdings_collection.update_one(
                    {"strategy": strategy.__name__},
                    {"$inc": {"failed_trades": 1}},
                    upsert=True,
                )

            if price_change_ratio > loss_price_change_ratio_d1:
                points = -time_delta * loss_profit_time_d1
            elif price_change_ratio > loss_price_change_ratio_d2:
                points = -time_delta * loss_profit_time_d2
            else:
                points = -time_delta * loss_profit_time_else

        # Update the points tally
        points_collection.update_one(
            {"strategy": strategy.__name__},
            {
                "$set": {"last_updated": datetime.now()},
                "$inc": {"total_points": points},
            },
            upsert=True,
        )
        if holdings_doc[ticker]["quantity"] == 0:
            del holdings_doc[ticker]
        # Update cash after selling
        holdings_collection.update_one(
            {"strategy": strategy.__name__},
            {
                "$set": {
                    "holdings": holdings_doc,
                    "amount_cash": strategy_doc["amount_cash"]
                    + sell_qty * current_price,
                    "last_updated": datetime.now(),
                },
                "$inc": {"total_trades": 1},
            },
            upsert=True,
        )

        # Remove the ticker if quantity reaches zero
        if holdings_doc[ticker]["quantity"] == 0:
            del holdings_doc[ticker]

    else:
        logging.info(
            f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}"
        )
    print(
        f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}"
    )
    # Close the MongoDB connection


def update_portfolio_values(client):
    """
    still need to implement.
    we go through each strategy and update portfolio value buy cash + summation(holding * current price)
    """

    db = client.trading_simulator
    holdings_collection = db.algorithm_holdings
    # Update portfolio values
    for strategy_doc in holdings_collection.find({}):
        # Calculate the portfolio value for the strategy
        portfolio_value = strategy_doc["amount_cash"]

        for ticker, holding in strategy_doc["holdings"].items():
            # The current price can be gotten through a cache system maybe
            # if polygon api is getting clogged - but that hasn't happened yet
            # Also implement in C++ or C instead of python
            # Get the current price of the ticker from the Polygon API
            # Use a cache system to store the latest prices
            # If the cache is empty, fetch the latest price from the Polygon API
            # Cache should be updated every 60 seconds
            current_price = None
            while current_price is None:
                try:
                    # get latest price shouldn't cache - we should also do a delay
                    current_price = get_latest_price(ticker)
                except (
                    Exception
                ) as e:  # Replace 'Exception' with a more specific error if possible
                    print(f"Error fetching price for {ticker} due to: {e}. Retrying...")
                    break

                    # Will sleep 120 seconds before retrying to get latest price
            print(f"Current price of {ticker}: {current_price}")
            if current_price is None:
                current_price = 0
            # Calculate the value of the holding
            holding_value = holding["quantity"] * current_price
            if current_price == 0:
                holding_value = 5000
            # Add the holding value to the portfolio value
            portfolio_value += holding_value

        # Update the portfolio value in the strategy document
        holdings_collection.update_one(
            {"strategy": strategy_doc["strategy"]},
            {"$set": {"portfolio_value": portfolio_value}},
            upsert=True,
        )

    # Update MongoDB with the modified strategy documents



def main():
    """
    Main function to control the workflow based on the market's status.
    """
    ndaq_tickers = []
    early_hour_first_iteration = True
    post_market_hour_first_iteration = True

    while True:
        mongo_client = MongoClient(MONGO_URL, tlsCAFile=ca)

        status = mongo_client.market_data.market_status.find_one({})["market_status"]

        if status == "open":
            # Connection pool is not thread safe. Create a new client for each thread.
            # We can use ThreadPoolExecutor to manage threads - maybe use this but this risks clogging
            # resources if we have too many threads or if a thread is on stall mode
            # We can also use multiprocessing.Pool to manage threads

            if not ndaq_tickers:
                logging.info("Market is open. Processing strategies.")
                ndaq_tickers = get_ndaq_tickers()

            threads = []

            for ticker in ndaq_tickers:
                thread = threading.Thread(
                    target=process_ticker, args=(ticker, mongo_client)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            logging.info("Finished processing all strategies. Waiting for 30 seconds.")
            time.sleep(30)

        elif status == "early_hours":
            # During early hour, currently we only support prep
            # However, we should add more features here like premarket analysis

            if early_hour_first_iteration is True:
                ndaq_tickers = get_ndaq_tickers()
                early_hour_first_iteration = False
                post_market_hour_first_iteration = True
                logging.info("Market is in early hours. Waiting for 30 seconds.")
            time.sleep(30)

        elif status == "closed":
            # Performs post-market analysis for next trading day
            # Will only run once per day to reduce clogging logging
            # Should self-implementing a delete log process after a certain time - say 1 year

            if post_market_hour_first_iteration is True:
                early_hour_first_iteration = True
                logging.info("Market is closed. Performing post-market analysis.")
                post_market_hour_first_iteration = False
                # Update time delta based on the mode

                if time_delta_mode == "additive":
                    mongo_client.trading_simulator.time_delta.update_one(
                        {}, {"$inc": {"time_delta": time_delta_increment}}
                    )
                elif time_delta_mode == "multiplicative":
                    mongo_client.trading_simulator.time_delta.update_one(
                        {}, {"$mul": {"time_delta": time_delta_multiplicative}}
                    )
                elif time_delta_mode == "balanced":
                    """
                    retrieve time_delta first
                    """
                    time_delta = mongo_client.trading_simulator.time_delta.find_one({})[
                        "time_delta"
                    ]
                    mongo_client.trading_simulator.time_delta.update_one(
                        {}, {"$inc": {"time_delta": time_delta_balanced * time_delta}}
                    )

                # Update ranks
                update_portfolio_values(mongo_client)
                # We keep reusing the same mongo client and never close to reduce the number within the connection pool

                update_ranks(mongo_client)
            time.sleep(60)
        else:
            logging.error("An error occurred while checking market status.")
            time.sleep(60)
        mongo_client.close()


if __name__ == "__main__":
    main()
