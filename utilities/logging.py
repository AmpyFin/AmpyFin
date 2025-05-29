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


import sys
from pathlib import Path
import logging

# … your existing constants …

def _build_file_handler(module_basename: str, level: int) -> logging.Handler:
    """
    Return a FileHandler that writes to `<caller>.log`.
    
    If invoked as a script (module_basename == "__main__"), we grab
    the script’s filename from sys.argv[0].
    """
    if module_basename == "__main__":
        # sys.argv[0] might be "trading.py" or a full path; take the stem
        name = Path(sys.argv[0]).stem or "main"
    else:
        # strip any leading underscores so "_foo" → "foo"
        name = module_basename.lstrip("_")
    filename = f"{name}.log"
    filepath = LOG_DIR / filename

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
