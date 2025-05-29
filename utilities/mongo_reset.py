import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymongo import MongoClient
from config import MONGO_URI

def reset_trading_collections():
    mongo_client = MongoClient(MONGO_URI)

    db = mongo_client.trades
    fills_coll = db.paper
    assets = db.assets_quantities
    limits = db.assets_limit

    deleted_fills = fills_coll.delete_many({})
    deleted_assets = assets.delete_many({})
    deleted_limits = limits.delete_many({})

    print(f"Deleted {deleted_fills.deleted_count} documents from 'paper'")
    print(f"Deleted {deleted_assets.deleted_count} documents from 'assets_quantities'")
    print(f"Deleted {deleted_limits.deleted_count} documents from 'assets_limit'")

if __name__ == "__main__":
    reset_trading_collections()
