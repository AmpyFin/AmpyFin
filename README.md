# AmpyFin Trading System

| Category    | Badges                                                                                                                                                                                                                                                                                                                                                   |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project** | [![Team Size](https://img.shields.io/badge/Team%20Size-6-blue)](https://github.com/AmpyFin/ampyfin/graphs/contributors) [![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen)](https://github.com/AmpyFin/ampyfin) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)         |
| **CI/CD**   | [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ampyfin/AmpyFin/pre-commit.yml?branch=main&label=linting)](https://github.com/AmpyFin/ampyfin/actions/workflows/pre-commit.yml) |
| **Quality** | [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Quality Gate Status](https://img.shields.io/badge/Quality%20Gate-Passing-success)](https://github.com/yeonholee50/AmpyFin)                                                                                                           |
| **Package** | [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)                                                                                                                                                                                                                                                   |
| **Meta**    | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)                                                                                                                                                                                                                                              |

## Introduction

AmpyFin is an advanced AI-powered trading system designed to trade within the NASDAQ-100. It leverages machine learning and algorithmic trading strategies to make data-driven investment decisions.

### Mission

The primary goal of AmpyFin as an open source project is to:

- **Democratize Algorithmic Trading**: Make proven trading frameworks available for everyone to use freely
- **Provide Transparency**: Offer insights into how machine learning can be applied to financial markets
- **Fill a Gap**: Contribute to the open source community where there are few published frameworks for effective trading systems
- **Enable Collaboration**: Create a platform for traders and developers to build upon and improve together

While there are many theoretical trading models in academic literature, AmpyFin aims to bridge the gap between theory and practice with a working implementation that the community can use, study, and enhance.

## Tutorial Playlist

For a comprehensive guide on how to use AmpyFin, check out our [YouTube tutorial playlist](https://www.youtube.com/playlist?list=PL7hzGb_OBFXaB7VGaXXiJk5uq2CXQqi-O).

## Features

### Data Collection

- **yfinance**: Primary source for historical price data and technical indicators

### Data Storage

- **MongoDB**: Securely stores all data and trading logs for historical analysis

### Machine Learning

AmpyFin uses a ranked ensemble learning system that dynamically evaluates each strategy's performance and adjusts their influence in the final decision accordingly.

### Trading Strategies

- **Mean Reversion**: Predicts asset prices will return to their historical average
- **Momentum**: Capitalizes on prevailing market trends
- **Arbitrage**: Identifies and exploits price discrepancies between related assets
- **AI-Driven Custom Strategies**: Continuously refined through machine learning

### Dynamic Ranking System

Each strategy is evaluated based on performance metrics and assigned a weight using a sophisticated ranking algorithm. The system adapts to changing market conditions by prioritizing high-performing strategies.

## Architecture

### Core Components

| Component             | Description                                     |
| --------------------- | ----------------------------------------------- |
| `control.py`          | Configuration interface for trading parameters  |
| `TradeSim/main.py`    | Training and testing environment for strategies |
| `TradeSim/trading.py` | Executes trades based on algorithmic decisions  |
| `TradeSim/ranking.py` | Evaluates and ranks trading strategies          |
| `strategies/*`        | Implementation of various trading algorithms    |
| `utilites/*`          | General Utility functions                       |
| `dbs/*`               | Create SQLite DBs of prices and decisions       |

## Installation

### Prerequisites

- Python 3.8+
- MongoDB
- TA-Lib

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/AmpyFin/ampyfin.git
cd ampyfin
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Install TA-Lib**

TA-Lib is required for technical indicators. Choose one of these installation options:

**Option 1**: Use pre-built binaries (Recommended for Windows users)

- Download the appropriate wheel file for your Python version from [here](https://github.com/cgohlke/talib-build/releases)
- Install it with pip:
  ```bash
  pip install <downloaded-wheel-file>.whl
  ```

**Option 2**: Build from source (For Linux/Mac users)

- Follow the instructions on the [TA-Lib Python](https://github.com/TA-Lib/ta-lib-python) repository

4. **Set up MongoDB**

MongoDB is used for storing trading data and strategy results. Here's how to set it up:

- **Option A**: MongoDB Atlas (Cloud-hosted, recommended for beginners)

  1. Create a free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
  2. Create a new cluster (the free tier is sufficient)
  3. Set up database access (username/password)
  4. Add your IP address to the network access whitelist
  5. Get your connection string, which will look like:
     ```
     mongodb+srv://<username>:<password>@cluster0.mongodb.net/ampyfin
     ```

- **Option B**: Local MongoDB Installation
  1. [Download and install MongoDB Community Edition](https://www.mongodb.com/try/download/community)
  2. Start the MongoDB service
  3. Your connection string will be:
     ```
     mongodb://localhost:27017/ampyfin
     ```

5. **Configure API keys**

You need to sign up for the following services to obtain API keys:

- [Alpaca](https://alpaca.markets/) - For trading execution
- [Weights & Biases](https://wandb.ai/) - For experiment tracking

Create a `.env` file based on the template:

```python
API_KEY = "your_alpaca_api_key"
API_SECRET = "your_alpaca_secret_key"
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading (safe for testing)
# BASE_URL = "https://api.alpaca.markets"      # Live trading (uses real money)
WANDB_API_KEY = "your_wandb_api_key"
MONGO_URL = "your_mongo_connection_string"
```

> ⚠️ **IMPORTANT**: The default configuration uses Alpaca's paper trading environment. To switch to live trading (using real money), change the BASE_URL to "https://api.alpaca.markets". Only do this once you've thoroughly tested your strategies and understand the risks involved.

6. **Run the setup script**

```bash
python setup.py
```

This initializes the database structure required for AmpyFin.

## Usage

### Running the System

Start the ranking and trading systems in separate terminals:

```bash
cd TradeSim
python ranking.py
python trading.py
```

### Training and Testing

These are separate operations that can be performed within the TradeSim module:

#### Training

1. Set the mode in `control.py`:

```python
mode = 'train'
```

2. Run the training module:

```bash
python TradeSim/main.py
```

#### Testing

1. Set the mode in `control.py`:

```python
mode = 'test'
```

2. Run the testing module:

```bash
python TradeSim/main.py
```

### Deploying a Model

1. Set the mode in `control.py`:

```python
mode = 'push'
```

2. Push your model to MongoDB:

```bash
python TradeSim/main.py
```

## Important Notes

For live trading, it's recommended to:

1. Train the system by running `ranking.py` for at least two weeks, or
2. Train using the TradeSim module and push changes to MongoDB before executing trades

This ensures the system has properly ranked strategies before making investment decisions.

## Contributing

Contributions are welcome! Please review the [Contributing Guide](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
