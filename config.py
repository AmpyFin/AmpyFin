import os
import sys
from os import environ as env

from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = os.getenv("BASE_URL")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
MONGO_URL = os.getenv("MONGO_URL")

# Check and fail explicitly if something is missing
required_vars = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "BASE_URL": BASE_URL,
    "WANDB_API_KEY": WANDB_API_KEY,
    "MONGO_URL": MONGO_URL,
}

missing = [k for k, v in required_vars.items() if not v]
if missing:
    print(f"[error]: Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)