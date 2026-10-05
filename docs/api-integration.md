# API Integration Guide

This document provides comprehensive information about the API integrations used in AmpyFin, including broker APIs, data providers, and external services.

## Overview

AmpyFin integrates with several external APIs:
- **Alpaca Markets**: Primary broker for trade execution
- **Interactive Brokers (IBKR)**: Alternative broker integration
- **yfinance**: Market data provider
- **Weights & Biases**: Experiment tracking and monitoring

## Alpaca Markets Integration

### Overview

Alpaca Markets is the primary broker used for trade execution in AmpyFin. It provides commission-free trading and a robust API for algorithmic trading.

### Setup and Configuration

#### 1. Account Setup
1. Sign up at [Alpaca Markets](https://alpaca.markets/)
2. Complete account verification
3. Generate API keys from the dashboard

#### 2. API Keys Configuration
```bash
# .env file
API_KEY=your_alpaca_api_key
API_SECRET=your_alpaca_secret_key
BASE_URL=https://paper-api.alpaca.markets  # Paper trading
# BASE_URL=https://api.alpaca.markets     # Live trading
```

#### 3. Client Initialization
```python
from alpaca.trading.client import TradingClient
from config import API_KEY, API_SECRET, BASE_URL

# Initialize trading client
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
```

### Trading Operations

#### Placing Orders
```python
from alpaca.trading.enums import OrderSide

def place_order(trading_client, symbol, side, quantity):
    """Place a market order through Alpaca."""
    try:
        order = trading_client.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,  # OrderSide.BUY or OrderSide.SELL
            type="market",
            time_in_force="day"
        )
        return order
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return None
```

#### Account Information
```python
def get_account_info(trading_client):
    """Get account information."""
    account = trading_client.get_account()
    return {
        'cash': float(account.cash),
        'portfolio_value': float(account.portfolio_value),
        'buying_power': float(account.buying_power),
        'equity': float(account.equity)
    }
```

#### Position Management
```python
def get_positions(trading_client):
    """Get current positions."""
    positions = trading_client.get_all_positions()
    return [
        {
            'symbol': pos.symbol,
            'qty': float(pos.qty),
            'avg_entry_price': float(pos.avg_entry_price),
            'market_value': float(pos.market_value)
        }
        for pos in positions
    ]
```

### Paper Trading vs Live Trading

#### Paper Trading (Recommended for Testing)
- **URL**: `https://paper-api.alpaca.markets`
- **Features**: Simulated trading with real market data
- **Benefits**: Risk-free testing, real market conditions
- **Limitations**: No real money, some features may differ

#### Live Trading (Production)
- **URL**: `https://api.alpaca.markets`
- **Features**: Real money trading
- **Requirements**: Thorough testing, risk management
- **Warning**: Uses real money - test extensively first

### Rate Limits and Best Practices

#### Rate Limits
- **Orders**: 200 requests per minute
- **Market Data**: 1000 requests per minute
- **Account Data**: 200 requests per minute

#### Best Practices
```python
import time
from functools import wraps

def rate_limit(max_calls=200, time_window=60):
    """Rate limiting decorator."""
    def decorator(func):
        calls = []
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls outside time window
            calls[:] = [call_time for call_time in calls if now - call_time < time_window]
            
            if len(calls) >= max_calls:
                sleep_time = time_window - (now - calls[0])
                time.sleep(sleep_time)
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=200, time_window=60)
def place_order_with_rate_limit(trading_client, symbol, side, quantity):
    return place_order(trading_client, symbol, side, quantity)
```

## Interactive Brokers (IBKR) Integration

### Overview

Interactive Brokers provides an alternative trading interface through their TWS API. This integration is primarily used for testing and advanced trading features.

### Setup and Configuration

#### 1. TWS Installation
1. Download and install [Trader Workstation (TWS)](https://www.interactivebrokers.com/en/trading/tws.php)
2. Enable API connections in TWS settings
3. Set up API permissions

#### 2. Connection Configuration
```python
from ib_insync import IB, Stock, MarketOrder

def connect_to_ibkr():
    """Connect to Interactive Brokers TWS."""
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)  # Paper trading
    # ib.connect('127.0.0.1', 7496, clientId=1)  # Live trading
    return ib
```

#### 3. Trading Operations
```python
def place_ibkr_order(ib, symbol, action, quantity):
    """Place order through IBKR."""
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    
    order = MarketOrder(action, quantity, outsideRth=True)
    trade = ib.placeOrder(contract, order)
    
    return trade
```

### Account Management
```python
def get_ibkr_account_values(ib):
    """Get account values from IBKR."""
    account_values = ib.accountValues()
    return {
        'cash': get_account_value(account_values, 'AvailableFunds'),
        'portfolio_value': get_account_value(account_values, 'NetLiquidation'),
        'buying_power': get_account_value(account_values, 'BuyingPower')
    }

def get_account_value(account_values, tag):
    """Extract specific account value."""
    for av in account_values:
        if av.tag == tag:
            return float(av.value)
    return 0.0
```

## yfinance Integration

### Overview

yfinance provides free market data for stocks, options, and other financial instruments. It's used as the primary data source for historical price data and technical indicators.

### Data Retrieval

#### Basic Price Data
```python
import yfinance as yf

def get_price_data(ticker, period="1y"):
    """Get historical price data."""
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

def get_current_price(ticker):
    """Get current stock price."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return info.get('currentPrice', 0)
```

#### Advanced Data Retrieval
```python
def get_comprehensive_data(ticker):
    """Get comprehensive stock data."""
    stock = yf.Ticker(ticker)
    
    return {
        'history': stock.history(period="1y"),
        'info': stock.info,
        'financials': stock.financials,
        'balance_sheet': stock.balance_sheet,
        'cashflow': stock.cashflow,
        'recommendations': stock.recommendations
    }
```

### Data Processing and Storage

#### MongoDB Integration
```python
def store_price_data(ticker, data, mongo_client):
    """Store price data in MongoDB."""
    db = mongo_client.HistoricalDatabase
    collection = db.HistoricalDatabase
    
    document = {
        'ticker': ticker,
        'data': data.to_dict(),
        'last_updated': datetime.now()
    }
    
    collection.update_one(
        {'ticker': ticker},
        {'$set': document},
        upsert=True
    )
```

#### SQLite Integration
```python
import sqlite3

def store_price_data_sqlite(ticker, data, db_path):
    """Store price data in SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for date, row in data.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO price_data 
            (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, date.strftime('%Y-%m-%d'),
            row['Open'], row['High'], row['Low'],
            row['Close'], row['Volume']
        ))
    
    conn.commit()
    conn.close()
```

## Weights & Biases Integration

### Overview

Weights & Biases (wandb) provides experiment tracking and monitoring capabilities for machine learning projects. It's used to track strategy performance and system metrics.

### Setup and Configuration

#### 1. Account Setup
1. Sign up at [Weights & Biases](https://wandb.ai/)
2. Generate API key from profile settings
3. Install wandb package: `pip install wandb`

#### 2. Configuration
```bash
# .env file
WANDB_API_KEY=your_wandb_api_key
```

#### 3. Initialization
```python
import wandb
from config import WANDB_API_KEY

def initialize_wandb(project_name, experiment_name, config):
    """Initialize Weights & Biases tracking."""
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=config
    )
```

### Experiment Tracking

#### Strategy Performance Tracking
```python
def log_strategy_performance(strategy_name, metrics):
    """Log strategy performance metrics."""
    wandb.log({
        f"{strategy_name}/total_trades": metrics['total_trades'],
        f"{strategy_name}/success_rate": metrics['success_rate'],
        f"{strategy_name}/total_return": metrics['total_return'],
        f"{strategy_name}/sharpe_ratio": metrics['sharpe_ratio']
    })
```

#### System Metrics Tracking
```python
def log_system_metrics(metrics):
    """Log system performance metrics."""
    wandb.log({
        "system/cpu_usage": metrics['cpu_usage'],
        "system/memory_usage": metrics['memory_usage'],
        "system/api_calls_per_minute": metrics['api_calls'],
        "system/database_query_time": metrics['db_query_time']
    })
```

#### Custom Metrics
```python
def log_custom_metrics(metrics):
    """Log custom trading metrics."""
    wandb.log({
        "trading/portfolio_value": metrics['portfolio_value'],
        "trading/daily_pnl": metrics['daily_pnl'],
        "trading/max_drawdown": metrics['max_drawdown'],
        "trading/win_rate": metrics['win_rate']
    })
```

## Error Handling and Resilience

### API Error Handling

#### Alpaca API Error Handling
```python
import requests
from alpaca.common.exceptions import APIError

def handle_alpaca_errors(func):
    """Decorator for handling Alpaca API errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            if e.status_code == 429:  # Rate limit
                time.sleep(60)  # Wait 1 minute
                return func(*args, **kwargs)
            elif e.status_code == 500:  # Server error
                time.sleep(30)  # Wait 30 seconds
                return func(*args, **kwargs)
            else:
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            time.sleep(10)
            return func(*args, **kwargs)
    return wrapper
```

#### yfinance Error Handling
```python
def safe_yfinance_call(ticker, max_retries=3):
    """Safely call yfinance with retry logic."""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")
            return data
        except Exception as e:
            logger.warning(f"yfinance call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"yfinance call failed after {max_retries} attempts")
                return None
```

### Connection Resilience

#### MongoDB Connection Resilience
```python
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

def resilient_mongo_operation(operation, max_retries=3):
    """Execute MongoDB operation with retry logic."""
    for attempt in range(max_retries):
        try:
            return operation()
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(f"MongoDB connection failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                logger.error(f"MongoDB operation failed after {max_retries} attempts")
                raise
```

## Monitoring and Logging

### API Usage Monitoring

#### Request Tracking
```python
import time
from functools import wraps

def track_api_usage(api_name):
    """Decorator to track API usage."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{api_name} API call successful: {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{api_name} API call failed: {e} ({duration:.2f}s)")
                raise
        return wrapper
    return decorator
```

#### Rate Limit Monitoring
```python
class RateLimitMonitor:
    def __init__(self, max_calls=200, time_window=60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def can_make_call(self):
        """Check if we can make an API call."""
        now = time.time()
        # Remove old calls
        self.calls[:] = [call_time for call_time in self.calls if now - call_time < self.time_window]
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Record an API call."""
        self.calls.append(time.time())
```

## Security Best Practices

### API Key Management

#### Environment Variables
```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Never hardcode API keys
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
```

#### Key Rotation
```python
def rotate_api_keys():
    """Rotate API keys for security."""
    # Generate new keys
    # Update environment variables
    # Notify system of key change
    # Gracefully handle transition
```

### Secure Communication

#### HTTPS/TLS
```python
import certifi
from pymongo import MongoClient

# Use SSL/TLS for MongoDB connections
mongo_client = MongoClient(MONGO_URL, tlsCAFile=certifi.where())
```

#### Request Signing
```python
import hmac
import hashlib
import base64

def sign_request(api_secret, request_data):
    """Sign API requests for authentication."""
    signature = hmac.new(
        api_secret.encode(),
        request_data.encode(),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode()
```

## Troubleshooting

### Common Issues

#### Alpaca API Issues
1. **Authentication Errors**: Check API keys and permissions
2. **Rate Limiting**: Implement proper rate limiting
3. **Market Hours**: Ensure trading during market hours
4. **Account Status**: Verify account is active and funded

#### yfinance Issues
1. **Data Availability**: Some tickers may not have data
2. **Rate Limiting**: Implement delays between requests
3. **Data Quality**: Validate data before processing
4. **Network Issues**: Handle network timeouts gracefully

#### MongoDB Issues
1. **Connection Timeouts**: Check network connectivity
2. **Authentication**: Verify credentials and permissions
3. **Index Performance**: Monitor query performance
4. **Memory Usage**: Monitor MongoDB memory consumption

### Debugging Tools

#### API Request Logging
```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('alpaca').setLevel(logging.DEBUG)
logging.getLogger('yfinance').setLevel(logging.DEBUG)
```

#### Performance Profiling
```python
import cProfile
import pstats

def profile_api_calls():
    """Profile API call performance."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your API calls here
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()
```
## FXMacroData Release Calendar

Strategies can import `fetch_fxmacrodata_calendar` to add official macro event
awareness before opening or resizing positions:

```python
from utilities.fxmacrodata_calendar import fetch_fxmacrodata_calendar, release_dates

events = fetch_fxmacrodata_calendar("usd", min_tier=1)
blocked_dates = release_dates(events)
```

Set `FXMACRODATA_API_KEY` when using authenticated FXMacroData endpoints. Public
USD calendar rows work without a key.
