import math
from datetime import datetime

from alpaca.trading.client import TradingClient
from pymongo import MongoClient, errors

from config import  mongo_url, API_KEY, API_SECRET
from utilities.ranking_trading_utils import get_latest_price, strategies
import subprocess
import os

indicator_periods = {
    "BBANDS_indicator": "1y",
    "DEMA_indicator": "1mo",
    "EMA_indicator": "1mo",
    "HT_TRENDLINE_indicator": "6mo",
    "KAMA_indicator": "1mo",
    "MA_indicator": "3mo",
    "MAMA_indicator": "6mo",
    "MAVP_indicator": "3mo",
    "MIDPOINT_indicator": "1mo",
    "MIDPRICE_indicator": "1mo",
    "SAR_indicator": "6mo",
    "SAREXT_indicator": "6mo",
    "SMA_indicator": "1mo",
    "T3_indicator": "1mo",
    "TEMA_indicator": "1mo",
    "TRIMA_indicator": "1mo",
    "WMA_indicator": "1mo",
    "ADX_indicator": "3mo",
    "ADXR_indicator": "3mo",
    "APO_indicator": "1mo",
    "AROON_indicator": "3mo",
    "AROONOSC_indicator": "3mo",
    "BOP_indicator": "1mo",
    "CCI_indicator": "1mo",
    "CMO_indicator": "1mo",
    "DX_indicator": "1mo",
    "MACD_indicator": "3mo",
    "MACDEXT_indicator": "3mo",
    "MACDFIX_indicator": "3mo",
    "MFI_indicator": "1mo",
    "MINUS_DI_indicator": "1mo",
    "MINUS_DM_indicator": "1mo",
    "MOM_indicator": "1mo",
    "PLUS_DI_indicator": "1mo",
    "PLUS_DM_indicator": "1mo",
    "PPO_indicator": "1mo",
    "ROC_indicator": "1mo",
    "ROCP_indicator": "1mo",
    "ROCR_indicator": "1mo",
    "ROCR100_indicator": "1mo",
    "RSI_indicator": "1mo",
    "STOCH_indicator": "1mo",
    "STOCHF_indicator": "1mo",
    "STOCHRSI_indicator": "1mo",
    "TRIX_indicator": "1mo",
    "ULTOSC_indicator": "6mo",
    "WILLR_indicator": "1mo",
    "AD_indicator": "1mo",
    "ADOSC_indicator": "1mo",
    "OBV_indicator": "1mo",
    "HT_DCPERIOD_indicator": "2y",
    "HT_DCPHASE_indicator": "2y",
    "HT_PHASOR_indicator": "2y",
    "HT_SINE_indicator": "2y",
    "HT_TRENDMODE_indicator": "2y",
    "AVGPRICE_indicator": "1mo",
    "MEDPRICE_indicator": "1mo",
    "TYPPRICE_indicator": "1mo",
    "WCLPRICE_indicator": "1mo",
    "ATR_indicator": "3mo",
    "NATR_indicator": "3mo",
    "TRANGE_indicator": "3mo",
    "CDL2CROWS_indicator": "1mo",
    "CDL3BLACKCROWS_indicator": "1mo",
    "CDL3INSIDE_indicator": "1mo",
    "CDL3LINESTRIKE_indicator": "1mo",
    "CDL3OUTSIDE_indicator": "1mo",
    "CDL3STARSINSOUTH_indicator": "1mo",
    "CDL3WHITESOLDIERS_indicator": "1mo",
    "CDLABANDONEDBABY_indicator": "1mo",
    "CDLADVANCEBLOCK_indicator": "1mo",
    "CDLBELTHOLD_indicator": "1mo",
    "CDLBREAKAWAY_indicator": "1mo",
    "CDLCLOSINGMARUBOZU_indicator": "1mo",
    "CDLCONCEALBABYSWALL_indicator": "1mo",
    "CDLCOUNTERATTACK_indicator": "1mo",
    "CDLDARKCLOUDCOVER_indicator": "1mo",
    "CDLDOJI_indicator": "1mo",
    "CDLDOJISTAR_indicator": "1mo",
    "CDLDRAGONFLYDOJI_indicator": "1mo",
    "CDLENGULFING_indicator": "1mo",
    "CDLEVENINGDOJISTAR_indicator": "1mo",
    "CDLEVENINGSTAR_indicator": "1mo",
    "CDLGAPSIDESIDEWHITE_indicator": "1mo",
    "CDLGRAVESTONEDOJI_indicator": "1mo",
    "CDLHAMMER_indicator": "1mo",
    "CDLHANGINGMAN_indicator": "1mo",
    "CDLHARAMI_indicator": "1mo",
    "CDLHARAMICROSS_indicator": "1mo",
    "CDLHIGHWAVE_indicator": "1mo",
    "CDLHIKKAKE_indicator": "1mo",
    "CDLHIKKAKEMOD_indicator": "1mo",
    "CDLHOMINGPIGEON_indicator": "1mo",
    "CDLIDENTICAL3CROWS_indicator": "1mo",
    "CDLINNECK_indicator": "1mo",
    "CDLINVERTEDHAMMER_indicator": "1mo",
    "CDLKICKING_indicator": "1mo",
    "CDLKICKINGBYLENGTH_indicator": "1mo",
    "CDLLADDERBOTTOM_indicator": "1mo",
    "CDLLONGLEGGEDDOJI_indicator": "1mo",
    "CDLLONGLINE_indicator": "1mo",
    "CDLMARUBOZU_indicator": "1mo",
    "CDLMATCHINGLOW_indicator": "1mo",
    "CDLMATHOLD_indicator": "1mo",
    "CDLMORNINGDOJISTAR_indicator": "1mo",
    "CDLMORNINGSTAR_indicator": "1mo",
    "CDLONNECK_indicator": "1mo",
    "CDLPIERCING_indicator": "1mo",
    "CDLRICKSHAWMAN_indicator": "1mo",
    "CDLRISEFALL3METHODS_indicator": "1mo",
    "CDLSEPARATINGLINES_indicator": "1mo",
    "CDLSHOOTINGSTAR_indicator": "1mo",
    "CDLSHORTLINE_indicator": "1mo",
    "CDLSPINNINGTOP_indicator": "1mo",
    "CDLSTALLEDPATTERN_indicator": "1mo",
    "CDLSTICKSANDWICH_indicator": "1mo",
    "CDLTAKURI_indicator": "1mo",
    "CDLTASUKIGAP_indicator": "1mo",
    "CDLTHRUSTING_indicator": "1mo",
    "CDLTRISTAR_indicator": "1mo",
    "CDLUNIQUE3RIVER_indicator": "1mo",
    "CDLUPSIDEGAP2CROWS_indicator": "1mo",
    "CDLXSIDEGAP3METHODS_indicator": "1mo",
    "BETA_indicator": "1y",
    "CORREL_indicator": "1y",
    "LINEARREG_indicator": "2y",
    "LINEARREG_ANGLE_indicator": "2y",
    "LINEARREG_INTERCEPT_indicator": "2y",
    "LINEARREG_SLOPE_indicator": "2y",
    "STDDEV_indicator": "1mo",
    "TSF_indicator": "2y",
    "VAR_indicator": "2y",
}


def insert_rank_to_coefficient(i):
    try:
        client = MongoClient(mongo_url)
        db = client.trading_simulator
        collections = db.rank_to_coefficient
        """
        Upsert rank coefficients from 1 to i
        """
        for i in range(1, i + 1):
            e = math.e
            rate = (e**e) / (e**2) - 1
            coefficient = rate ** (2 * i)
            collections.update_one(
                {"rank": i},
                {"$set": {"coefficient": coefficient}},
                upsert=True,
            )
        client.close()
        print("Successfully ensured rank to coefficient mapping")
    except Exception as exception:
        print(exception)


def initialize_rank():
    try:
        client = MongoClient(mongo_url)
        db = client.trading_simulator

        initialization_date = datetime.now()

        for strategy in strategies:
            strategy_name = strategy.__name__

            collections = db.algorithm_holdings

            if not collections.find_one({"strategy": strategy_name}):
                collections.insert_one(
                    {
                        "strategy": strategy_name,
                        "holdings": {},
                        "amount_cash": 50000,
                        "initialized_date": initialization_date,
                        "total_trades": 0,
                        "successful_trades": 0,
                        "neutral_trades": 0,
                        "failed_trades": 0,
                        "last_updated": initialization_date,
                        "portfolio_value": 50000,
                    }
                )

                collections = db.points_tally
                collections.insert_one(
                    {
                        "strategy": strategy_name,
                        "total_points": 0,
                        "initialized_date": initialization_date,
                        "last_updated": initialization_date,
                    }
                )

        client.close()
        print("Successfully initialized rank")
    except Exception as exception:
        print(exception)


def initialize_time_delta():
    try:
        client = MongoClient(mongo_url)
        db = client.trading_simulator
        collection = db.time_delta
        collection.update_one(
            {"_id": "time_delta_config"},
            {"$setOnInsert": {"time_delta": 0.01}},
            upsert=True,
        )
        client.close()
        print("Successfully initialized time delta")
    except Exception as exception:
        print(exception)


def initialize_market_setup():
    try:
        client = MongoClient(mongo_url)
        db = client.market_data
        collection = db.market_status
        collection.update_one(
            {"_id": "market_status_config"},
            {"$setOnInsert": {"market_status": "closed"}},
            upsert=True,
        )
        client.close()
        print("Successfully initialized market setup")
    except Exception as exception:
        print(exception)


def initialize_portfolio_percentages():
    try:
        client = MongoClient(mongo_url)
        trading_client = TradingClient(API_KEY, API_SECRET)
        account = trading_client.get_account()
        db = client.trades
        collection = db.portfolio_values

        portfolio_value = float(account.portfolio_value)
        collection.update_one(
            {"name": "portfolio_percentage"},
            {"$set": {"portfolio_value": (portfolio_value - 50000) / 50000}},
            upsert=True,
        )

        qqq_latest = get_latest_price("QQQ")
        collection.update_one(
            {"name": "ndaq_percentage"},
            {"$set": {"portfolio_value": (qqq_latest - 503.17) / 503.17}},
            upsert=True,
        )

        spy_latest = get_latest_price("SPY")
        collection.update_one(
            {"name": "spy_percentage"},
            {"$set": {"portfolio_value": (spy_latest - 590.50) / 590.50}},
            upsert=True,
        )

        client.close()
        print("Successfully initialized portfolio percentages")
    except Exception as exception:
        print(exception)


def initialize_indicator_setup():
    try:
        client = MongoClient(mongo_url)
        db = client["IndicatorsDatabase"]
        collection = db["Indicators"]

        for indicator, period in indicator_periods.items():
            collection.update_one(
                {"indicator": indicator},
                {"$set": {"ideal_period": period}},
                upsert=True,
            )

        print("Indicators and their ideal periods are ensured in MongoDB.")
    except Exception as e:
        print(e)
        return


def initialize_historical_database_cache():
    try:
        client = MongoClient(mongo_url)
        db = client["HistoricalDatabase"]
        collection = db["HistoricalDatabase"]
        print("Historical DB collection : ", collection)

    except errors.ConnectionError as e:
        print(f"Error connecting to the MongoDB server: {e}")
        return
    
def initialize_dbs():
    # Define the paths to the scripts
    store_price_data_path = os.path.join(os.path.dirname(__file__), 'dbs', 'store_price_data.py')
    compute_strategy_path = os.path.join(os.path.dirname(__file__), 'dbs', 'compute_store_strategy_decisions.py')

    # Construct the database paths
    price_data_db_path = os.path.join(os.path.dirname(__file__), 'dbs', 'databases', 'price_data.db')
    strategy_decisions_db_path = os.path.join(os.path.dirname(__file__), 'dbs', 'databases', 'strategy_decisions.db')

    # Check if price_data.db already exists and remove it
    if os.path.exists(price_data_db_path):
        os.remove(price_data_db_path)
        print(f"Removed existing database: {price_data_db_path}")
    else:
        print(f"{price_data_db_path} does not exist. Creating a new database...")
        # Logic to create a new directory if it doesn't exist
        price_data_dir = os.path.dirname(price_data_db_path)
        if not os.path.exists(price_data_dir):
            os.makedirs(price_data_dir)
            print(f"Created new directory: {price_data_dir}")

        # Logic to create a new database file if it doesn't exist
        with open(price_data_db_path, 'w') as db_file:
            db_file.write("")  # Create an empty file
        print(f"Created new database: {price_data_db_path}")

    # Call the first script: store_price_data.py
    print("Calling store_price_data.py...")
    try:
        subprocess.run(['python', store_price_data_path], check=True)
        print("store_price_data.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing store_price_data.py: {e}")
        return  # Exit if the first script fails

    # Check if price_data.db was created
    if not os.path.exists(price_data_db_path):
        print(f"Error: {price_data_db_path} was not created by store_price_data.py")
        return

    # Check if strategy_decisions.db already exists and remove it
    if os.path.exists(strategy_decisions_db_path):
        os.remove(strategy_decisions_db_path)
        print(f"Removed existing database: {strategy_decisions_db_path}")
    else: 
        print(f"{strategy_decisions_db_path} does not exist. Creating a new database...")
        # Logic to create a new directory if it doesn't exist
        price_data_dir = os.path.dirname(price_data_db_path)
        if not os.path.exists(price_data_dir):
            os.makedirs(price_data_dir)
            print(f"Created new directory: {price_data_dir}")
            
        # Logic to create a new database if it doesn't exist
        with open(strategy_decisions_db_path, 'w') as db_file:
            db_file.write("") # Create an empty file
        print(f"Created new database: {strategy_decisions_db_path}")

    # Call the second script: compute_store_strategy_decisions.py
    print("Calling compute_store_strategy_decisions.py...")
    try:
        subprocess.run(['python', compute_strategy_path], check=True)
        print("compute_store_strategy_decisions.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing compute_store_strategy_decisions.py: {e}")
        return

    


if __name__ == "__main__":
    # insert_rank_to_coefficient(200)

    # initialize_rank()

    # initialize_time_delta()

    # initialize_market_setup()

    # initialize_portfolio_percentages()

    # initialize_indicator_setup()

    # initialize_historical_database_cache()
    
    initialize_dbs()