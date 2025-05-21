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
from config import (
    API_KEY,
    API_SECRET,
    MONGO_URL,
)

from control import suggestion_heap_limit, trade_asset_limit, trade_liquidity_limit, train_tickers
from utilities.ranking_trading_utils import ( 
    get_latest_price,
    market_status,
    place_order,
    strategies
)
from strategies.talib_indicators import get_data, simulate_strategy
from utilities.logging import setup_logging
logger = setup_logging(__name__)

buy_heap = []
suggestion_heap = []
sold = False

ca = certifi.where()


def process_ticker(
    ticker, trading_client, mongo_client, indicator_periods, strategy_to_coefficient
):  
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



def execute_buy_orders(mongo_client, trading_client):
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

def load_indicator_periods(mongo_client):
    coll = mongo_client.IndicatorsDatabase.Indicators
    # one query for all docs
    return {
        doc["indicator"]: doc["ideal_period"]
        for doc in coll.find({}, {"indicator": 1, "ideal_period": 1})
    }

def initialize_strategy_coefficients(mongo_client):
    """
    Initialize strategy coefficients from MongoDB.
    """
    strategy_to_coefficient = {}
    sim_db = mongo_client.trading_simulator
    rank_collection = sim_db.rank
    r_t_c_collection = sim_db.rank_to_coefficient

    for strategy in strategies:
        rank = rank_collection.find_one({"strategy": strategy.__name__})["rank"]
        coefficient = r_t_c_collection.find_one({"rank": rank})["coefficient"]
        strategy_to_coefficient[strategy.__name__] = coefficient

    return strategy_to_coefficient

def process_market_open(mongo_client):
    logger.info("Market is open. Processing tickers.")
    global buy_heap, suggestion_heap, sold, train_tickers
    if not train_tickers:
        train_tickers = get_ndaq_tickers()
    
    indicator_periods = load_indicator_periods(mongo_client)
    strategy_to_coefficient = initialize_strategy_coefficients(mongo_client)

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
    """
    Main function to control the workflow based on the market's status.
    """

    logger.info("Trading mode is live.")
  
    trading_client = TradingClient(API_KEY, API_SECRET)
    mongo_client = MongoClient(MONGO_URL, tlsCAFile=ca)
   
    strategy_to_coefficient = {}
    while True:
        try:
            status = market_status()
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
