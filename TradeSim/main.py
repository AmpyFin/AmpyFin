#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the trading simulator application.

This module handles the initialization of the MongoDB client, artifact directory setup,
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

# Add the parent directory to the Python path (assumes this file is in TradeSim/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from config import mongo_url
from variables import config_dict
from control import mode
from training import train
from testing import test
from utilities.logging import setup_logging
logger = setup_logging(__name__)

# Base paths
def _get_paths():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ARTIFACTS = os.path.join(BASE_DIR, 'artifacts')
    return {
        'base': BASE_DIR,
        'artifacts': ARTIFACTS,
        'logs': os.path.join(ARTIFACTS, 'log'),
        'wandb': os.path.join(ARTIFACTS, 'wandb'),
        'results': os.path.join(ARTIFACTS, 'results'),
        'tearsheets': os.path.join(ARTIFACTS, 'tearsheets'),
    }

PATHS = _get_paths()
_procs = []  # keep track of child processes for shutdown handling


def create_artifacts_dirs() -> None:
    """
    Create the artifacts directory and all required subdirectories if they don't exist.
    """
    os.makedirs(PATHS['artifacts'], exist_ok=True)
    for key in ('logs', 'wandb', 'results', 'tearsheets'):
        os.makedirs(PATHS[key], exist_ok=True)


def initialize_wandb() -> None:
    """
    Initialize the Weights & Biases (wandb) integration.

    Sets the WANDB_DIR environment so wandb files go under artifacts.
    """
    os.environ['WANDB_DIR'] = PATHS['wandb']
    wandb.login()
    wandb.init(
        project=config_dict['project_name'],
        config=config_dict,
        name=config_dict['experiment_name'],
    )


def _shutdown(signum=None, frame=None) -> None:
    """
    Terminate all child processes, then exit.
    """
    logging.info("Shutting down live mode processes…")
    for p in _procs:
        if p.poll() is None:
            logging.info(f" → terminating PID {p.pid}")
            p.terminate()
    time.sleep(1)
    for p in _procs:
        if p.poll() is None:
            logging.warning(f" → killing PID {p.pid}")
            p.kill()
    sys.exit(0)


def run_live(logger: logging.Logger) -> None:
    """
    Spawn trading.py and ranking.py as standalone, long‑running services.
    """
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    scripts = [
        [sys.executable, os.path.join(PATHS['base'], 'TradeSim', 'trading.py')],
        [sys.executable, os.path.join(PATHS['base'], 'TradeSim', 'ranking.py')],
    ]
    for cmd in scripts:
        logger.info(f"Starting subprocess: {' '.join(cmd)}")
        p = subprocess.Popen(cmd)
        _procs.append(p)

    logger.info("Both trading.py and ranking.py started. Entering monitor loop.")
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
    """
    Main function that initializes the application and runs the selected mode.
    """
    create_artifacts_dirs()
    logger.info("Starting trading simulator application")

    # Initialize MongoDB client
    ca = certifi.where()
    mongo_client = MongoClient(mongo_url, tlsCAFile=ca)
    logger.info("MongoDB client initialized")

    # Initialize W&B
    initialize_wandb()
    logger.info("Weights & Biases initialized")

    if mode == 'train':
        logger.info("Running in training mode")
        train()
    elif mode == 'test':
        logger.info("Running in testing mode")
        test(mongo_client=mongo_client)
    elif mode == 'live':
        logger.info("Running in live mode (spawning trading & ranking)")
        run_live(logger)
    elif mode == 'push':
        logger.warning("Push mode is not implemented yet")
        sys.exit(1)
    else:
        logger.error(f"Invalid mode: {mode}")
        sys.exit(1)

    logger.info("Application completed successfully")


if __name__ == '__main__':
    main()
