#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the trading simulator application.

This module handles the initialization of the MongoDB client, logging,
and calls the appropriate training or testing functions based on the mode.
"""

import logging
import os
import sys
import certifi
from pymongo import MongoClient
import wandb

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from config import mongo_url
from variables import config_dict
from control import mode, train_period_start, test_period_end, train_tickers
from helper_files.client_helper import get_ndaq_tickers
from training import train
from testing import test


def setup_logging():
    """
    Set up logging configuration for the application.
    
    Creates a logs directory if it doesn't exist and configures a file handler
    with appropriate formatting.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logs_dir = "log"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(os.path.join(logs_dir, "train_test.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def initialize_wandb():
    """
    Initialize and configure Weights & Biases for experiment tracking.
    
    Uses configuration from config_dict to set up the W&B run.
    """
    wandb.login()
    wandb.init(
        project=config_dict["project_name"],
        config=config_dict,
        name=config_dict["experiment_name"],
    )


def main():
    """
    Main function that initializes the application and runs the selected mode.
    
    Handles setup of MongoDB connection, logging, W&B initialization, and
    delegates to the appropriate module based on the selected mode.
    """
    # Set up logging
    logger = setup_logging()
    logger.info("Starting trading simulator application")
    
    # Initialize MongoDB client with certificate
    ca = certifi.where()
    mongo_client = MongoClient(mongo_url, tlsCAFile=ca)
    logger.info("MongoDB client initialized")
    
    # Initialize W&B for experiment tracking
    initialize_wandb()
    logger.info("Weights & Biases initialized")

    # Run in the appropriate mode
    if mode == "train":
        logger.info("Running in training mode")
        train(logger)
    elif mode == "test":
        logger.info("Running in testing mode")
        test(mongo_client, logger)
    elif mode == "push":
        logger.info("Running in push mode")
        # Placeholder for push mode functionality
        # push(mongo_client, logger)
        # This mode is not implemented yet
        logger.warning("Push mode is not implemented yet")
    # Additional modes can be added here
    else:
        logger.error(f"Invalid mode: {mode}")
        sys.exit(1)
    
    logger.info("Application completed successfully")


if __name__ == "__main__":
    main()