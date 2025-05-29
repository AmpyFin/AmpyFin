
'''
TODOS - 
 # The current price can be gotten through a cache system maybe
            # if polygon api is getting clogged - but that hasn't happened yet
            # Also implement in C++ or C instead of python
            # Get the current price of the ticker from the Polygon API
            # Use a cache system to store the latest prices
            # If the cache is empty, fetch the latest price from the Polygon API
            # Cache should be updated every 60 seconds
'''

import time
from datetime import datetime
import os
import sys

# Ensure sys.path manipulation is at the top, before other local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    train_tickers
)
from utilities.ranking_trading_utils import get_latest_price, update_ranks, strategies
from utilities.common_utils import get_ndaq_tickers
from strategies.talib_indicators import get_data, simulate_strategy
# from strategies.categorise_talib_indicators_vect import strategies
from utilities.logging import setup_logging
logger = setup_logging(__name__)

ca = certifi.where()

def process_ticker(ticker, mongo_client, indicator_periods):
    # 1) fetch price
    try:
        current_price = get_latest_price(ticker)
    except Exception:
        logger.warning(f"Price fetch failed for {ticker}.")
        return

    for strategy in strategies:
        try:
            historical_data = None
            while historical_data is None:
                try:
                    period = indicator_periods[strategy.__name__]
                    historical_data = get_data(ticker, mongo_client, period)
                except Exception as fetch_error:
                    logger.warning(
                        f"Error fetching historical data for {ticker}. Retrying... {fetch_error}"
                    )
                    time.sleep(60)

            holdings_coll = mongo_client.trading_simulator.algorithm_holdings
            strategy_doc = holdings_coll.find_one({"strategy": strategy.__name__})
            if not strategy_doc:
                logger.warning(
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
                strategy_doc,
                mongo_client,
            )
        except Exception as strat_error:
                logger.error(f"Error processing strategy {strategy.__name__} for {ticker}: {strat_error}")
    logger.info(f"{ticker} processing completed.")


def simulate_trade(
    ticker,
    strategy,
    historical_data,
    current_price,
    strat_doc,
    mongo_client,
):
    """
    Simulates a trade based on the given strategy and updates MongoDB.
    """
    portfolio_qty = strat_doc.get("holdings", {}).get(ticker, {}).get("quantity", 0)
  
    action, quantity = simulate_strategy(
        strategy,
        ticker,
        current_price,
        historical_data,
        strat_doc['amount_cash'],
        portfolio_qty,
        strat_doc['portfolio_value'],
    )

    holdings_coll = mongo_client.trading_simulator.algorithm_holdings
    points_coll = mongo_client.trading_simulator.points_tally
    time_delta = mongo_client.trading_simulator.time_delta.find_one({})["time_delta"]
    holdings_doc = strat_doc.get("holdings", {})
    

    # Update holdings and cash based on trade action
    if (
        action == "buy"
        and strat_doc["amount_cash"] - quantity * current_price
        > rank_liquidity_limit
        and quantity > 0
        and ((portfolio_qty + quantity) * current_price) / strat_doc["portfolio_value"]   
        < rank_asset_limit
    ):
        logger.info(
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
        holdings_coll.update_one(
            {"strategy": strategy.__name__},
            {
                "$set": {
                    "holdings": holdings_doc,
                    "amount_cash": strat_doc["amount_cash"]
                    - quantity * current_price,
                    "last_updated": datetime.now(),
                },
                "$inc": {"total_trades": 1},
            },
            upsert=True,
        )

    elif (
        action == "sell"
        and str(ticker) in holdings_doc
        and holdings_doc[ticker]["quantity"] > 0
    ):
        logger.info(
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
            holdings_coll.update_one(
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
                holdings_coll.update_one(
                    {"strategy": strategy.__name__}, {"$inc": {"neutral_trades": 1}}
                )

            else:
                holdings_coll.update_one(
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
        points_coll.update_one(
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
        holdings_coll.update_one(
            {"strategy": strategy.__name__},
            {
                "$set": {
                    "holdings": holdings_doc,
                    "amount_cash": strat_doc["amount_cash"]
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
        logger.info(
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

    holdings_coll = client.trading_simulator.algorithm_holdings
    # Update portfolio values
    for doc in holdings_coll.find({}):
        # Calculate the portfolio value for the strategy
        value = doc["amount_cash"]

        for ticker, holding in doc["holdings"].items():
           
            current_price = None
            while current_price is None:
                try:
                    # get latest price shouldn't cache - we should also do a delay
                    current_price = get_latest_price(ticker)
                except Exception as e:  # Replace 'Exception' with a more specific error if possible
                    print(f"Error fetching price for {ticker} due to: {e}. Retrying...")
                    break
                   
            print(f"Current price of {ticker}: {current_price}")
            if current_price is None:
                current_price = 0

            # Calculate the value of the holding
            holding_value = holding["quantity"] * current_price
            if current_price == 0:
                holding_value = 5000
            
            # Add the holding value to the portfolio value
            value += holding_value

        # Update the portfolio value in the strategy document
        holdings_coll.update_one(
            {"strategy": doc["strategy"]},
            {"$set": {"portfolio_value": value}}
        )

def load_indicator_periods(mongo_client):
    coll = mongo_client.IndicatorsDatabase.Indicators
    # one query for all docs
    return {
        doc["indicator"]: doc["ideal_period"]
        for doc in coll.find({}, {"indicator": 1, "ideal_period": 1})
    }

def process_early_hours(early_hour_first_iteration):
    """
    Market early hours phase: prep work.
    """
    if early_hour_first_iteration:
        logger.info("Market EARLY_HOURS: Refreshing ticker list.")
        get_ndaq_tickers()  # Refresh tickers
        early_hour_first_iteration = False
        
    logger.info("Market EARLY_HOURS: Waiting 30s.")
    time.sleep(30)
    return early_hour_first_iteration, True  # Return updated flag and set post_market flag True

def process_market_closed(client, post_market_hour_first_iteration):
    """
    Market closed phase: post-market updates.
    """
    if post_market_hour_first_iteration:
        logger.info("Market CLOSED: Performing post-market operations.")
        
        # Update time delta based on the mode
        td_coll = client.trading_simulator.time_delta
        td_doc = td_coll.find_one({})
        
        if not td_doc:
            logger.error("No time_delta document found!")
            td_coll.insert_one({"time_delta": 1.0})  # Create default
            # Check if this is correct or we want to set it to sys.float_info.min
        else:
            current_td = td_doc["time_delta"]
            
            if time_delta_mode == "additive":
                td_coll.update_one({}, {"$inc": {"time_delta": time_delta_increment}})
            elif time_delta_mode == "multiplicative":
                td_coll.update_one({}, {"$mul": {"time_delta": time_delta_multiplicative}})
            else:  # balanced mode
                td_coll.update_one({}, {"$inc": {"time_delta": current_td * time_delta_balanced}})

        update_portfolio_values(client)
        update_ranks(client, logger)
        post_market_hour_first_iteration = False
        
    time.sleep(60)
    return post_market_hour_first_iteration, True  # Return updated flag and set early_hour flag True

def process_market_open(mongo_client):  
    global train_tickers

    if not train_tickers:
        logger.info("No tickers to train. Pulling NASDAQ tickers.")
        train_tickers = get_ndaq_tickers()

    indicator_periods = load_indicator_periods(mongo_client)
    for ticker in train_tickers:
        process_ticker(ticker, mongo_client, indicator_periods)

    logger.info("Finished processing all strategies. Waiting for 30 seconds.")
    time.sleep(30)

def main():
    """
    Main function to control the workflow based on the market's status.
    """
    early_hour_first_iteration = True
    post_market_hour_first_iteration = True

   

    while True:
        mongo_client = MongoClient(MONGO_URL, tlsCAFile=ca)
        market_status = mongo_client.market_data.market_status.find_one({})["market_status"]
        if not market_status:
            logger.error("Market status not found in database.")
            time.sleep(60)
            continue
        
        print(f"Market status document: {market_status}")   
        status = market_status
        print(f"Current market status: {status}")

        if status == "open":
            process_market_open(mongo_client)
            # Reset flags when market closes
            early_hour_first_iteration = True
            post_market_hour_first_iteration = True
           
        elif status == "early_hours":
            # During early hour, currently we only support prep
            # However, we should add more features here like premarket analysis

            early_hour_first_iteration, post_market_hour_first_iteration = process_early_hours(early_hour_first_iteration)

        elif status == "closed":
            # Performs post-market analysis for next trading day
            # Will only run once per day to reduce clogger logger
            # Should self-implement a delete log process after a certain time - say 1 year
            post_market_hour_first_iteration, early_hour_first_iteration = process_market_closed(mongo_client, post_market_hour_first_iteration)

        else:
            logger.error("UNKNOWN market status. Waiting for 60 seconds.")
            time.sleep(60)
    
        mongo_client.close()


if __name__ == "__main__":
    main()
