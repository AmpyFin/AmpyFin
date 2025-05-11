import heapq
import logging
import threading
import time

import certifi
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from polygon import RESTClient
from pymongo import MongoClient
from TradeSim.utils import weighted_majority_decision_and_median_quantity
from config import (
    API_KEY,
    API_SECRET,
    FINANCIAL_PREP_API_KEY,
    POLYGON_API_KEY,
    mongo_url,
)
from control import suggestion_heap_limit, trade_asset_limit, trade_liquidity_limit
from helper_files.client_helper import (
    get_latest_price,
    get_ndaq_tickers,
    market_status,
    place_order,
    strategies,
)
from strategies.talib_indicators import get_data, simulate_strategy

buy_heap = []
suggestion_heap = []
sold = False


ca = certifi.where()

# # Set up logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     handlers=[
#         logging.FileHandler('log/system.log'),  # Log messages to a file
#         logging.StreamHandler()             # Log messages to the console
#     ]
# )


def process_ticker(
    ticker, client, trading_client, data_client, mongo_client, strategy_to_coefficient
):
    global buy_heap
    global suggestion_heap
    global sold
    if sold is True:
        print("Sold boolean is True. Exiting process_ticker function.")
    else:
        try:
            decisions_and_quantities = []
            current_price = None

            while current_price is None:
                try:
                    current_price = get_latest_price(ticker)
                except Exception as fetch_error:
                    logging.warning(
                        f"Error fetching price for {ticker}. Retrying... {fetch_error}"
                    )
                    break
            if current_price is None:
                return
            print(f"Current price of {ticker}: {current_price}")

            asset_collection = mongo_client.trades.assets_quantities
            limits_collection = mongo_client.trades.assets_limit
            account = trading_client.get_account()
            buying_power = float(account.cash)
            portfolio_value = float(account.portfolio_value)
            # cash_to_portfolio_ratio = buying_power / portfolio_value

            asset_info = asset_collection.find_one({"symbol": ticker})
            portfolio_qty = asset_info["quantity"] if asset_info else 0.0
            print(f"Portfolio quantity for {ticker}: {portfolio_qty}")

            limit_info = limits_collection.find_one({"symbol": ticker})
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
                    order = place_order(
                        trading_client,
                        symbol=ticker,
                        side=OrderSide.SELL,
                        quantity=quantity,
                        mongo_client=mongo_client,
                    )
                    logging.info(f"Executed SELL order for {ticker}: {order}")
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

            (
                decision,
                quantity,
                buy_weight,
                sell_weight,
                hold_weight,
            ) = weighted_majority_decision_and_median_quantity(decisions_and_quantities)
            print(
                f"Ticker: {ticker}, Decision: {decision}, Quantity: {quantity}, Weights: Buy: {buy_weight}, Sell: {sell_weight}, Hold: {hold_weight}"
            )

            if (
                decision == "buy"
                and float(account.cash) > trade_liquidity_limit
                and (((quantity + portfolio_qty) * current_price) / portfolio_value)
                < trade_asset_limit
            ):
                heapq.heappush(
                    buy_heap,
                    (
                        -(buy_weight - (sell_weight + (hold_weight * 0.5))),
                        quantity,
                        ticker,
                    ),
                )
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
                logging.info(f"Executed SELL order for {ticker}: {order}")
            elif (
                portfolio_qty == 0.0
                and buy_weight > sell_weight
                and (((quantity + portfolio_qty) * current_price) / portfolio_value)
                < trade_asset_limit
                and float(account.cash) > trade_liquidity_limit
            ):
                max_investment = portfolio_value * trade_asset_limit
                buy_quantity = min(
                    int(max_investment // current_price),
                    int(buying_power // current_price),
                )
                if buy_weight > suggestion_heap_limit:
                    buy_quantity = max(buy_quantity, 2)
                    buy_quantity = buy_quantity // 2
                    print(
                        f"Suggestions for buying for {ticker} with a weight of {buy_weight} and quantity of {buy_quantity}"
                    )
                    heapq.heappush(
                        suggestion_heap,
                        (-(buy_weight - sell_weight), buy_quantity, ticker),
                    )
                else:
                    logging.info(f"Holding for {ticker}, no action taken.")
            else:
                logging.info(f"Holding for {ticker}, no action taken.")

        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")


def main():
    """
    Main function to control the workflow based on the market's status.
    """

    logging.info("Trading mode is live.")
    global buy_heap
    global suggestion_heap
    global sold
    ndaq_tickers = []
    early_hour_first_iteration = True
    post_hour_first_iteration = True
    client = RESTClient(api_key=POLYGON_API_KEY)
    trading_client = TradingClient(API_KEY, API_SECRET)
    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    mongo_client = MongoClient(mongo_url, tlsCAFile=ca)
    # db = mongo_client.trades
    # asset_collection = db.assets_quantities
    # limits_collection = db.assets_limit
    strategy_to_coefficient = {}
    sold = False
    while True:
        client = RESTClient(api_key=POLYGON_API_KEY)
        trading_client = TradingClient(API_KEY, API_SECRET)
        data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
        status = market_status(client)  # Use the helper function for market status
        # db = mongo_client.trades
        # asset_collection = db.assets_quantities
        # limits_collection = db.assets_limit
        market_db = mongo_client.market_data
        market_collection = market_db.market_status
        # indicator_tb = mongo_client.IndicatorsDatabase
        # indicator_collection = indicator_tb.Indicators

        market_collection.update_one({}, {"$set": {"market_status": status}})

        if status == "open":
            if not ndaq_tickers:
                logging.info("Market is open")
                ndaq_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
                sim_db = mongo_client.trading_simulator
                rank_collection = sim_db.rank
                r_t_c_collection = sim_db.rank_to_coefficient
                for strategy in strategies:
                    rank = rank_collection.find_one({"strategy": strategy.__name__})[
                        "rank"
                    ]
                    coefficient = r_t_c_collection.find_one({"rank": rank})[
                        "coefficient"
                    ]
                    strategy_to_coefficient[strategy.__name__] = coefficient
                    early_hour_first_iteration = False
                    post_hour_first_iteration = True
            trading_client = TradingClient(API_KEY, API_SECRET)
            account = trading_client.get_account()
            # buying_power = float(account.cash)
            portfolio_value = float(account.portfolio_value)
            # cash_to_portfolio_ratio = buying_power / portfolio_value
            qqq_latest = get_latest_price("QQQ")
            spy_latest = get_latest_price("SPY")
            buy_heap = []
            suggestion_heap = []

            trades_db = mongo_client.trades
            portfolio_collection = trades_db.portfolio_values

            portfolio_collection.update_one(
                {"name": "portfolio_percentage"},
                {"$set": {"portfolio_value": (portfolio_value - 50491.13) / 50491.13}},
            )
            portfolio_collection.update_one(
                {"name": "ndaq_percentage"},
                {"$set": {"portfolio_value": (qqq_latest - 518.58) / 518.58}},
            )
            portfolio_collection.update_one(
                {"name": "spy_percentage"},
                {"$set": {"portfolio_value": (spy_latest - 591.95) / 591.95}},
            )

            threads = []

            for ticker in ndaq_tickers:
                thread = threading.Thread(
                    target=process_ticker,
                    args=(
                        ticker,
                        client,
                        trading_client,
                        data_client,
                        mongo_client,
                        strategy_to_coefficient,
                    ),
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            trading_client = TradingClient(API_KEY, API_SECRET)
            account = trading_client.get_account()
            while (
                (buy_heap or suggestion_heap)
                and float(account.cash) > trade_liquidity_limit
                and sold is False
            ):
                try:
                    trading_client = TradingClient(API_KEY, API_SECRET)
                    account = trading_client.get_account()
                    print(f"Cash: {account.cash}")
                    if buy_heap and float(account.cash) > trade_liquidity_limit:
                        _, quantity, ticker = heapq.heappop(buy_heap)
                        print(
                            f"Executing BUY order for {ticker} of quantity {quantity}"
                        )

                        order = place_order(
                            trading_client,
                            symbol=ticker,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            mongo_client=mongo_client,
                        )
                        logging.info(f"Executed BUY order for {ticker}: {order}")

                    elif (
                        suggestion_heap and float(account.cash) > trade_liquidity_limit
                    ):
                        _, quantity, ticker = heapq.heappop(suggestion_heap)
                        print(
                            f"Executing BUY order for {ticker} of quantity {quantity}"
                        )

                        order = place_order(
                            trading_client,
                            symbol=ticker,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            mongo_client=mongo_client,
                        )
                        logging.info(f"Executed BUY order for {ticker}: {order}")

                    time.sleep(5)
                    """
                    This is here so order will propage through and we will have an accurate cash balance recorded
                    """
                except Exception as e:
                    print(
                        f"Error occurred while executing buy order due to {e}. Continuing..."
                    )
                    break
            buy_heap = []
            suggestion_heap = []
            sold = False
            print("Sleeping for 30 seconds...")
            time.sleep(30)

        elif status == "early_hours":
            if early_hour_first_iteration:
                ndaq_tickers = get_ndaq_tickers(mongo_client, FINANCIAL_PREP_API_KEY)
                sim_db = mongo_client.trading_simulator
                rank_collection = sim_db.rank
                r_t_c_collection = sim_db.rank_to_coefficient
                for strategy in strategies:
                    rank = rank_collection.find_one({"strategy": strategy.__name__})[
                        "rank"
                    ]
                    coefficient = r_t_c_collection.find_one({"rank": rank})[
                        "coefficient"
                    ]
                    strategy_to_coefficient[strategy.__name__] = coefficient
                    early_hour_first_iteration = False
                    post_hour_first_iteration = True
                logging.info("Market is in early hours. Waiting for 30 seconds.")
            time.sleep(30)

        elif status == "closed":
            if post_hour_first_iteration:
                early_hour_first_iteration = True
                post_hour_first_iteration = False
                logging.info("Market is closed. Performing post-market operations.")
            time.sleep(30)
        else:
            logging.error("An error occurred while checking market status.")
            time.sleep(60)


if __name__ == "__main__":
    main()
