#!/usr/bin/env python3
"""
main.py

Entry point for AmpyFin training and testing workflows. Configures logging, MongoDB client,
Weights & Biases integration, and dispatches to train or test routines based on the `mode`
in `control.py`.
"""

import os
import sys
import logging
import certifi

# Ensure project root is on PYTHONPATH for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import wandb
from pymongo import MongoClient

# Local module imports
from push import push
from testing import test
from training import train
from variables import config_dict
from config import mongo_url
from control import mode

# Path to the CA certificate bundle for MongoDB TLS
CA_CERT_PATH = certifi.where()
# Directory where log files will be written
LOGS_DIR = "log"
# Log filename
LOG_FILE = "train_test.log"


def setup_logging(log_dir: str, log_file: str) -> logging.Logger:
    """
    Configure and return a logger that writes to a file in the given directory.

    Args:
        log_dir (str): Path to the directory where logs should be saved.
        log_file (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory if it does not exist
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_mongo_client(uri: str, ca_file: str) -> MongoClient:
    """
    Create and return a MongoDB client with TLS support.

    Args:
        uri (str): MongoDB connection URI.
        ca_file (str): Path to CA certificate bundle for TLS.

    Returns:
        MongoClient: Initialized MongoDB client.
    """
    return MongoClient(uri, tlsCAFile=ca_file)


def init_wandb(config: dict) -> None:
    """
    Initialize Weights & Biases run with the provided configuration.

    Args:
        config (dict): Configuration dictionary for W&B.
    """
    wandb.login()
    wandb.init(
        project=config.get("project_name"),
        config=config,
        name=config.get("experiment_name"),
    )

def main() -> None:
    """
    Main entry point for training or testing workflows.

    - Sets up logging
    - Initializes MongoDB client
    - Starts Weights & Biases run
    - Dispatches to train or test routines based on `mode`

    Returns:
        None
    """
    logger = setup_logging(LOGS_DIR, LOG_FILE)
    logger.info("Starting train/test runner")

    # Initialize MongoDB client
    mongo_client = get_mongo_client(mongo_url, CA_CERT_PATH)
    logger.info("Connected to MongoDB")

    # Initialize W&B
    init_wandb(config_dict)
    logger.info("Weights & Biases initialized")

    # Dispatch based on mode
    if mode == "train":
        logger.info("Entering training mode")
        train(logger)

    elif mode == "test":
        logger.info("Entering testing mode")
        test(mongo_client, logger)

    elif mode == "push":
        logger.info("Entering push mode")
        push()

    else:
        logger.error(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
