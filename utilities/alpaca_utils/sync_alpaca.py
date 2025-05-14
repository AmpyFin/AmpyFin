import logging

import certifi
from alpaca.trading.client import TradingClient
from pymongo import MongoClient

from config import API_KEY, API_SECRET, mongo_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def sync_positions():
    """
    Sync MongoDB trades database with actual Alpaca long positions only
    """
    try:
        trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        mongo_client = MongoClient(mongo_url, tlsCAFile=certifi.where())
        db = mongo_client.trades
        print("\nCurrent MongoDB positions:")
        mongo_positions = {}
        for doc in db.assets_quantities.find():
            mongo_positions[doc["symbol"]] = doc["quantity"]
            print(f"  {doc['symbol']}: {doc['quantity']}")  # No space before colon

        print("\nCurrent Alpaca long positions:")
        alpaca_positions = trading_client.get_all_positions()
        alpaca_holdings = {}
        for position in alpaca_positions:
            qty = float(position.qty)
            if qty > 0:
                alpaca_holdings[position.symbol] = qty
                print(
                    f"  {position.symbol}: {qty} shares @ ${float(position.avg_entry_price): .2f}"
                )  # No extra space before colon

        print("\nDifferences in long positions:")
        all_symbols = set(mongo_positions.keys()) | set(alpaca_holdings.keys())
        has_differences = False
        for symbol in sorted(all_symbols):
            mongo_qty = mongo_positions.get(symbol, 0)
            alpaca_qty = alpaca_holdings.get(symbol, 0)
            if mongo_qty != alpaca_qty:
                has_differences = True
                print(f"  {symbol}: ")  # Removed space before colon
                print(f"    MongoDB: {mongo_qty}")
                print(f"    Alpaca: {alpaca_qty}")  # Removed extra space after colon

        if not has_differences:
            print("  No differences found in long positions")
            return

        # Update MongoDB to match Alpaca long positions
        if (
            input("\nUpdate MongoDB to match Alpaca long positions? (y/n): ").lower()
            == "y"
        ):
            # Clear existing positions
            db.assets_quantities.delete_many({})

            for symbol, quantity in alpaca_holdings.items():
                db.assets_quantities.insert_one(
                    {"symbol": symbol, "quantity": quantity}
                )

            account = trading_client.get_account()
            portfolio_value = float(account.portfolio_value)

            # Update portfolio value
            db.portfolio_values.insert_one(
                {"name": "portfolio_percentage", "portfolio_value": portfolio_value}
            )

            print("\nMongoDB updated successfully with long positions")
            # Remove spaces around the comma inside the format spec
            # Using ":,.2f" if you want thousand separators
            print(f"Portfolio Value: ${portfolio_value: ,.2f}")
        else:
            print("\nSync cancelled")

    except Exception as e:
        logging.error(f"Error syncing positions with Alpaca: {e}")
    finally:
        mongo_client.close()


if __name__ == "__main__":
    sync_positions()
