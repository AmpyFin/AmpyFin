import logging
from pathlib import Path

# Absolute path to the project root (the directory that contains TradeSim/, artifacts/, utilities/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory where all log files will live (…/artifacts/log)
LOG_DIR = PROJECT_ROOT / "artifacts" / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Correct log message format (no stray space after %)
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _build_file_handler(module_basename: str, level: int) -> logging.Handler:
    """Return a :class:`logging.FileHandler` that writes to the correct file."""
    filename_map = {"__main__": "main.log", "training": "training.log", "testing": "testing.log"}
    log_filename = filename_map.get(module_basename, "other.log")
    filepath = LOG_DIR / log_filename

    handler = logging.FileHandler(filepath, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    return handler


def _build_console_handler(level: int) -> logging.Handler:
    """Return a :class:`logging.StreamHandler` that prints to stdout."""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    return handler


def setup_logging(name: str, *, level: int = logging.INFO, console: bool = True) -> logging.Logger:
    """Configure (once) and return a logger for *name*.

    Usage::
        from utilities.logging import setup_logging
        logger = setup_logging(__name__)
    """
    logger = logging.getLogger(name)

    # Only configure once
    if getattr(logger, "_is_configured", False):
        return logger

    logger.setLevel(level)

    module_basename = Path(name).name.split(".")[-1]  # e.g., "TradeSim.training" → "training"
    logger.addHandler(_build_file_handler(module_basename, level))

    if console:
        logger.addHandler(_build_console_handler(level))

    logger.propagate = False
    logger._is_configured = True  # type: ignore[attr-defined]
    return logger
