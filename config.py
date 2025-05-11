import os
import sys
from os import environ as env

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

try:
    WANDB_API_KEY = env["WANDB_API_KEY"]
    mongo_url = env["MONGO_URL"]
except KeyError as e:
    print(f"[error]: {e} required environment variable missing")
    sys.exit(1)
