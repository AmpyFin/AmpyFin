#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the trading simulator application.

This module handles the initialization of the MongoDB client, logging,
and calls the appropriate training, testing, or live workflows.
"""

import logging
import os
import sys
import certifi
import subprocess
import signal
import time
from pymongo import MongoClient
import wandb

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from config import mongo_url
from variables import config_dict
from control import mode
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
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fmt = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    fh = logging.FileHandler(os.path.join(logs_dir, "train_test.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def initialize_wandb():
    """Initializes the Weights & Biases (wandb) integration.

        Logs into wandb, initializes a new wandb run, and configures it with the project name,
        configuration dictionary, and experiment name from the global config_dict.

        Args:
            None

        Returns:
            None
    """
    wandb.login()
    wandb.init(
        project=config_dict["project_name"],
        config=config_dict,
        name=config_dict["experiment_name"],
    )

# keep track of child processes globally so signal handler can see them
_procs = []

def _shutdown(signum=None, frame=None):
    """Terminate all child processes, then exit."""
    logging.info("Shutting down live mode processes…")
    for p in _procs:
        if p.poll() is None:
            logging.info(f" → terminating PID {p.pid}")
            p.terminate()
    # give them a moment
    time.sleep(1)
    for p in _procs:
        if p.poll() is None:
            logging.warning(f" → killing PID {p.pid}")
            p.kill()
    sys.exit(0)

def run_live(logger):
    """Spawn trading.py and ranking.py as standalone, long‑running services."""
    # install signal handlers to capture Ctrl+C / docker stop, etc.
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    scripts = [
        [sys.executable, "trading.py"],
        [sys.executable, "ranking.py"],
    ]

    for cmd in scripts:
        logger.info(f"Starting subprocess: {' '.join(cmd)}")
        p = subprocess.Popen(cmd)
        _procs.append(p)

    logger.info("Both trading.py and ranking.py started. Entering monitor loop.")
    # monitor loop: if either child exits, tear down everything
    try:
        while True:
            for p in _procs:
                if p.poll() is not None:
                    logger.error(f"Child PID {p.pid} exited (code {p.returncode}); shutting down.")
                    _shutdown()
            time.sleep(5)
    except Exception:
        _shutdown()

def main() -> None:
    """Main function that initializes the application and runs the selected mode.
        Args:
            None
        Returns:
            None
    """
    logger = setup_logging()
    logger.info("Starting trading simulator application")

    # Initialize MongoDB client
    ca = certifi.where()
    mongo_client = MongoClient(mongo_url, tlsCAFile=ca)
    logger.info("MongoDB client initialized")

    # Initialize W&B
    initialize_wandb()
    logger.info("Weights & Biases initialized")

    if mode == "train":
        logger.info("Running in training mode")
        train(logger=logger)

    elif mode == "test":
        logger.info("Running in testing mode")
        test(mongo_client=mongo_client, logger=logger)

    elif mode == "live":
        logger.info("Running in live mode (spawning trading & ranking)")
        run_live(logger)

    elif mode == "push":
        logger.warning("Push mode is not implemented yet")
        sys.exit(1)

    else:
        logger.error(f"Invalid mode: {mode}")
        sys.exit(1)

    logger.info("Application completed successfully")

if __name__ == "__main__":
    main()
