import json
import os
from datetime import datetime

import certifi
from pymongo import MongoClient
from variables import config_dict

from config import mongo_url
from control import experiment_name
from utils import update_ranks

ca = certifi.where()

results_dir = "results"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)


def push():
    with open(
        os.path.join(results_dir, f"{config_dict[experiment_name]}.json"), "r"
    ) as json_file:
        results = json.load(json_file)
        trading_simulator = results["trading_simulator"]
        points = results["points"]
        time_delta = results["time_delta"]

    # Push the trading simulator and points to the database
    mongo_client = MongoClient(mongo_url, tlsCAFile=ca)
    db = mongo_client.trading_simulator
    holdings_collection = db.algorithm_holdings
    points_collection = db.points_tally

    for strategy, value in trading_simulator.items():
        holdings_collection.update_one(
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
        points_collection.update_one(
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

    db.time_delta.update_one({}, {"$set": {"time_delta": time_delta}}, upsert=True)
    update_ranks(mongo_client)
