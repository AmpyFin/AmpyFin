import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import pandas as pd

import yfinance as yf
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from pymongo import MongoClient

import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime
import pytz

from control import stop_loss, take_profit
from strategies.talib_indicators import (
    AD_indicator,
    ADOSC_indicator,
    ADX_indicator,
    ADXR_indicator,
    APO_indicator,
    AROON_indicator,
    AROONOSC_indicator,
    ATR_indicator,
    AVGPRICE_indicator,
    BBANDS_indicator,
    BETA_indicator,
    BOP_indicator,
    CCI_indicator,
    CDL2CROWS_indicator,
    CDL3BLACKCROWS_indicator,
    CDL3INSIDE_indicator,
    CDL3LINESTRIKE_indicator,
    CDL3OUTSIDE_indicator,
    CDL3STARSINSOUTH_indicator,
    CDL3WHITESOLDIERS_indicator,
    CDLABANDONEDBABY_indicator,
    CDLADVANCEBLOCK_indicator,
    CDLBELTHOLD_indicator,
    CDLBREAKAWAY_indicator,
    CDLCLOSINGMARUBOZU_indicator,
    CDLCONCEALBABYSWALL_indicator,
    CDLCOUNTERATTACK_indicator,
    CDLDARKCLOUDCOVER_indicator,
    CDLDOJI_indicator,
    CDLDOJISTAR_indicator,
    CDLDRAGONFLYDOJI_indicator,
    CDLENGULFING_indicator,
    CDLEVENINGDOJISTAR_indicator,
    CDLEVENINGSTAR_indicator,
    CDLGAPSIDESIDEWHITE_indicator,
    CDLGRAVESTONEDOJI_indicator,
    CDLHAMMER_indicator,
    CDLHANGINGMAN_indicator,
    CDLHARAMI_indicator,
    CDLHARAMICROSS_indicator,
    CDLHIGHWAVE_indicator,
    CDLHIKKAKE_indicator,
    CDLHIKKAKEMOD_indicator,
    CDLHOMINGPIGEON_indicator,
    CDLIDENTICAL3CROWS_indicator,
    CDLINNECK_indicator,
    CDLINVERTEDHAMMER_indicator,
    CDLKICKING_indicator,
    CDLKICKINGBYLENGTH_indicator,
    CDLLADDERBOTTOM_indicator,
    CDLLONGLEGGEDDOJI_indicator,
    CDLLONGLINE_indicator,
    CDLMARUBOZU_indicator,
    CDLMATCHINGLOW_indicator,
    CDLMATHOLD_indicator,
    CDLMORNINGDOJISTAR_indicator,
    CDLMORNINGSTAR_indicator,
    CDLONNECK_indicator,
    CDLPIERCING_indicator,
    CDLRICKSHAWMAN_indicator,
    CDLRISEFALL3METHODS_indicator,
    CDLSEPARATINGLINES_indicator,
    CDLSHOOTINGSTAR_indicator,
    CDLSHORTLINE_indicator,
    CDLSPINNINGTOP_indicator,
    CDLSTALLEDPATTERN_indicator,
    CDLSTICKSANDWICH_indicator,
    CDLTAKURI_indicator,
    CDLTASUKIGAP_indicator,
    CDLTHRUSTING_indicator,
    CDLTRISTAR_indicator,
    CDLUNIQUE3RIVER_indicator,
    CDLUPSIDEGAP2CROWS_indicator,
    CDLXSIDEGAP3METHODS_indicator,
    CMO_indicator,
    CORREL_indicator,
    DEMA_indicator,
    DX_indicator,
    EMA_indicator,
    HT_DCPERIOD_indicator,
    HT_DCPHASE_indicator,
    HT_PHASOR_indicator,
    HT_SINE_indicator,
    HT_TRENDLINE_indicator,
    HT_TRENDMODE_indicator,
    KAMA_indicator,
    LINEARREG_ANGLE_indicator,
    LINEARREG_indicator,
    LINEARREG_INTERCEPT_indicator,
    LINEARREG_SLOPE_indicator,
    MA_indicator,
    MACD_indicator,
    MACDEXT_indicator,
    MACDFIX_indicator,
    MAMA_indicator,
    MAVP_indicator,
    MEDPRICE_indicator,
    MFI_indicator,
    MIDPOINT_indicator,
    MIDPRICE_indicator,
    MINUS_DI_indicator,
    MINUS_DM_indicator,
    MOM_indicator,
    NATR_indicator,
    OBV_indicator,
    PLUS_DI_indicator,
    PLUS_DM_indicator,
    PPO_indicator,
    ROC_indicator,
    ROCP_indicator,
    ROCR100_indicator,
    ROCR_indicator,
    RSI_indicator,
    SAR_indicator,
    SAREXT_indicator,
    SMA_indicator,
    STDDEV_indicator,
    STOCH_indicator,
    STOCHF_indicator,
    STOCHRSI_indicator,
    T3_indicator,
    TEMA_indicator,
    TRANGE_indicator,
    TRIMA_indicator,
    TRIX_indicator,
    TSF_indicator,
    TYPPRICE_indicator,
    ULTOSC_indicator,
    VAR_indicator,
    WCLPRICE_indicator,
    WILLR_indicator,
    WMA_indicator,
)

sys.path.append("..")

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


overlap_studies = [
    BBANDS_indicator,
    DEMA_indicator,
    EMA_indicator,
    HT_TRENDLINE_indicator,
    KAMA_indicator,
    MA_indicator,
    MAMA_indicator,
    MAVP_indicator,
    MIDPOINT_indicator,
    MIDPRICE_indicator,
    SAR_indicator,
    SAREXT_indicator,
    SMA_indicator,
    T3_indicator,
    TEMA_indicator,
    TRIMA_indicator,
    WMA_indicator,
]
momentum_indicators = [
    ADX_indicator,
    ADXR_indicator,
    APO_indicator,
    AROON_indicator,
    AROONOSC_indicator,
    BOP_indicator,
    CCI_indicator,
    CMO_indicator,
    DX_indicator,
    MACD_indicator,
    MACDEXT_indicator,
    MACDFIX_indicator,
    MFI_indicator,
    MINUS_DI_indicator,
    MINUS_DM_indicator,
    MOM_indicator,
    PLUS_DI_indicator,
    PLUS_DM_indicator,
    PPO_indicator,
    ROC_indicator,
    ROCP_indicator,
    ROCR_indicator,
    ROCR100_indicator,
    RSI_indicator,
    STOCH_indicator,
    STOCHF_indicator,
    STOCHRSI_indicator,
    TRIX_indicator,
    ULTOSC_indicator,
    WILLR_indicator,
]
volume_indicators = [AD_indicator, ADOSC_indicator, OBV_indicator]
cycle_indicators = [
    HT_DCPERIOD_indicator,
    HT_DCPHASE_indicator,
    HT_PHASOR_indicator,
    HT_SINE_indicator,
    HT_TRENDMODE_indicator,
]
price_transforms = [
    AVGPRICE_indicator,
    MEDPRICE_indicator,
    TYPPRICE_indicator,
    WCLPRICE_indicator,
]
volatility_indicators = [ATR_indicator, NATR_indicator, TRANGE_indicator]
pattern_recognition = [
    CDL2CROWS_indicator,
    CDL3BLACKCROWS_indicator,
    CDL3INSIDE_indicator,
    CDL3LINESTRIKE_indicator,
    CDL3OUTSIDE_indicator,
    CDL3STARSINSOUTH_indicator,
    CDL3WHITESOLDIERS_indicator,
    CDLABANDONEDBABY_indicator,
    CDLADVANCEBLOCK_indicator,
    CDLBELTHOLD_indicator,
    CDLBREAKAWAY_indicator,
    CDLCLOSINGMARUBOZU_indicator,
    CDLCONCEALBABYSWALL_indicator,
    CDLCOUNTERATTACK_indicator,
    CDLDARKCLOUDCOVER_indicator,
    CDLDOJI_indicator,
    CDLDOJISTAR_indicator,
    CDLDRAGONFLYDOJI_indicator,
    CDLENGULFING_indicator,
    CDLEVENINGDOJISTAR_indicator,
    CDLEVENINGSTAR_indicator,
    CDLGAPSIDESIDEWHITE_indicator,
    CDLGRAVESTONEDOJI_indicator,
    CDLHAMMER_indicator,
    CDLHANGINGMAN_indicator,
    CDLHARAMI_indicator,
    CDLHARAMICROSS_indicator,
    CDLHIGHWAVE_indicator,
    CDLHIKKAKE_indicator,
    CDLHIKKAKEMOD_indicator,
    CDLHOMINGPIGEON_indicator,
    CDLIDENTICAL3CROWS_indicator,
    CDLINNECK_indicator,
    CDLINVERTEDHAMMER_indicator,
    CDLKICKING_indicator,
    CDLKICKINGBYLENGTH_indicator,
    CDLLADDERBOTTOM_indicator,
    CDLLONGLEGGEDDOJI_indicator,
    CDLLONGLINE_indicator,
    CDLMARUBOZU_indicator,
    CDLMATCHINGLOW_indicator,
    CDLMATHOLD_indicator,
    CDLMORNINGDOJISTAR_indicator,
    CDLMORNINGSTAR_indicator,
    CDLONNECK_indicator,
    CDLPIERCING_indicator,
    CDLRICKSHAWMAN_indicator,
    CDLRISEFALL3METHODS_indicator,
    CDLSEPARATINGLINES_indicator,
    CDLSHOOTINGSTAR_indicator,
    CDLSHORTLINE_indicator,
    CDLSPINNINGTOP_indicator,
    CDLSTALLEDPATTERN_indicator,
    CDLSTICKSANDWICH_indicator,
    CDLTAKURI_indicator,
    CDLTASUKIGAP_indicator,
    CDLTHRUSTING_indicator,
    CDLTRISTAR_indicator,
    CDLUNIQUE3RIVER_indicator,
    CDLUPSIDEGAP2CROWS_indicator,
    CDLXSIDEGAP3METHODS_indicator,
]
statistical_functions = [
    BETA_indicator,
    CORREL_indicator,
    LINEARREG_indicator,
    LINEARREG_ANGLE_indicator,
    LINEARREG_INTERCEPT_indicator,
    LINEARREG_SLOPE_indicator,
    STDDEV_indicator,
    TSF_indicator,
    VAR_indicator,
]

strategies = (
    # overlap_studies
    # + momentum_indicators
     volume_indicators
    # + cycle_indicators
    # + price_transforms
    # + volatility_indicators
    # + pattern_recognition
    # + statistical_functions
)


def market_status() -> str:
    """Determines the current status of the market (open, closed, or early hours).

        The function checks the NYSE calendar to determine if the market is open,
        closed, or in the early trading hours (pre-market).

        Returns:
            str: 'open' if the market is currently open, 'closed' if the market is
                currently closed, and 'early_hours' if it is currently in the
                pre-market trading hours.
    """
    # Get the NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    # Current time in Eastern
    now = pd.Timestamp.now(tz='US/Eastern')
    # Today's schedule
    sched = nyse.schedule(start_date=now.date(), end_date=now.date())
    if sched.empty:
        return 'closed'
    open_time = sched.at[now.date(), 'market_open']
    close_time = sched.at[now.date(), 'market_close']
    pre_open = open_time - pd.Timedelta(hours=5.5)  # 4:00 ET
    if open_time <= now <= close_time:
        return 'open'
    if pre_open <= now < open_time:
        return 'early_hours'
    return 'closed'



def get_latest_price(ticker: str) -> float | None:
    """
        Fetches the latest closing price for a given stock ticker from Yahoo Finance.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').

        Returns:
            float: The latest closing price, rounded to 2 decimal places.
                   Returns None if there is an error fetching the price.

        Raises:
            Exception: Logs any exceptions encountered during the process.

    """
    try:
        ticker_yahoo = yf.Ticker(ticker, session=limiter)
        data = ticker_yahoo.history()

        return round(data["Close"].iloc[-1], 2)
    except Exception as e:
        logging.error(f"Error fetching latest price for {ticker}: {e}")
        return None


def place_order(trading_client: object, symbol: str, side: OrderSide, quantity: float, mongo_client: MongoClient) -> object:
    """Places a market order for a given symbol and logs the trade details.

        Args:
            trading_client: The Alpaca trading client.
            symbol (str): The symbol to trade.
            side (OrderSide): The side of the order (buy or sell).
            quantity (float): The quantity to trade.
            mongo_client: The MongoDB client.

        Returns:
            Order: The order object returned by the Alpaca API.

    """
    market_order_data = MarketOrderRequest(
        symbol=symbol, qty=quantity, side=side, time_in_force=TimeInForce.DAY
    )
    order = trading_client.submit_order(market_order_data)
    qty = round(quantity, 3)
    current_price = get_latest_price(symbol)
    stop_loss_price = round(current_price * (1 - stop_loss), 2)  # 3% loss
    take_profit_price = round(current_price * (1 + take_profit), 2)  # 5% profit

    # Log trade details to MongoDB
    db = mongo_client.trades
    db.paper.insert_one(
        {
            "symbol": symbol,
            "qty": qty,
            "side": side.name,
            "time_in_force": TimeInForce.DAY.name,
            "time": datetime.now(tz=timezone.utc),
        }
    )

    # Track assets as well
    assets = db.assets_quantities
    limits = db.assets_limit

    if side == OrderSide.BUY:
        assets.update_one({"symbol": symbol}, {"$inc": {"quantity": qty}}, upsert=True)
        limits.update_one(
            {"symbol": symbol},
            {
                "$set": {
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price,
                }
            },
            upsert=True,
        )
    elif side == OrderSide.SELL:
        assets.update_one({"symbol": symbol}, {"$inc": {"quantity": -qty}}, upsert=True)
        if assets.find_one({"symbol": symbol})["quantity"] == 0:
            assets.delete_one({"symbol": symbol})
            limits.delete_one({"symbol": symbol})

    return order
