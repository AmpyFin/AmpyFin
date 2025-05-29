import json
import os
from datetime import datetime

import certifi
from pymongo import MongoClient
from variables import config_dict

from config import MONGO_URL
from control import experiment_name
from utilities.ranking_trading_utils import update_ranks
from utilities.logging import setup_logging
logger = setup_logging(__name__)

ca = certifi.where()
# Load saved training results
results_dir = os.path.join('../artifacts', 'results')

def push() -> None:
    """Pushes trading simulator results and points to a MongoDB database.

        The function reads trading simulator data and points from a JSON file,
        then updates the corresponding collections in the MongoDB database.
        It updates the holdings, cash amount, trade statistics, portfolio value,
        and points for each strategy. It also updates the time delta.
        Finally, it calls the `update_ranks` function to update the ranks in the database.

        Args:
            None

        Returns:
            None

        Raises:
            FileNotFoundError: If the JSON file containing the results is not found.
            json.JSONDecodeError: If the JSON file is invalid.
            pymongo.errors.ConnectionFailure: If a connection to the MongoDB database cannot be established.
            pymongo.errors.PyMongoError: If any other MongoDB related error occurs.

        """
    with open(os.path.join(results_dir, f"{config_dict['experiment_name']}.json"), "r") as json_file:
        results = json.load(json_file)
        trading_simulator = results["trading_simulator"]
        points = results["points"]
        time_delta = results["time_delta"]

    # Push the trading simulator and points to the database
    mongo_client = MongoClient(MONGO_URL, tlsCAFile=ca)

    holdings_coll = mongo_client.trading_simulator.algorithm_holdings
    points_coll = mongo_client.trading_simulator.points_tally

    for strategy, value in trading_simulator.items():
        holdings_coll.update_one(
            {"strategy": strategy},
            {
                "$set": {
                    "holdings": value["holdings"],
                    "amount_cash": value["amount_cash"],
                    "total_trades": value["total_trades"],
                    "successful_trades": value["successful_trades"],
                    "neutral_trades": value["neutral_trades"],
                    "failed_trades": value["failed_trades"],
                    "portfolio_value": value["portfolio_value"],
                    "last_updated": datetime.now(),
                    "initialized_date": datetime.now(),
                }
            },
            upsert=True,
        )

    for strategy, value in points.items():
        points_coll.update_one(
            {"strategy": strategy},
            {
                "$set": {
                    "total_points": value,
                    "last_updated": datetime.now(),
                    "initialized_date": datetime.now(),
                }
            },
            upsert=True,
        )

    mongo_client.trading_simulator.time_delta.update_one({}, {"$set": {"time_delta": time_delta}}, upsert=True)
    update_ranks(mongo_client, logger)
