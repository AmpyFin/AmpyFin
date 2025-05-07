"""
Logging configuration for the AmpyFin project.

Defines logging formatters, handlers, and logger settings
 for both console and file outputs.
- Supports rotating file handlers for main logs, database logs,
 and dynamically named logs.
- Provides different formatting for console and file outputs.
- Allows dynamic assignment of log file names for specific modules.

Log files are stored in the 'log' directory.

Usage:
    Import LOG_CONFIG and use with logging.config.dictConfig(LOG_CONFIG)
      to initialize logging.

Example with dynamic naming
 (log file name will match the file name it is run from):
    import os
    import sys
    import logging
    import logging.config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from log_config import LOG_CONFIG

    # Get the current filename without extension
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    log_filename = f"log/{module_name}.log"
    LOG_CONFIG["handlers"]["file_dynamic"]["filename"] = log_filename

    logging.config.dictConfig(LOG_CONFIG)
    logger = logging.getLogger(__name__)
"""

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",  # noqa: E501
            "datefmt": "%Y-%m-%d %H:%M:%S %z",
        },
        "simple_console": {
            "format": "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",  # noqa: E501
            "datefmt": "%Y-%m-%d %H:%M:%S %z",
        },
    },
    "handlers": {
        "stderr": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple_console",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "log/main_log.log",
            "maxBytes": 2000000,  # 2000000=2MB
            "backupCount": 3,
        },
        "file_dbs": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "log/dbs_log.log",
            "maxBytes": 2000000,
            "backupCount": 3,
        },
        "file_dynamic": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            # "filename":  to be added dynamically
            "maxBytes": 2000000,
            "backupCount": 3,
            "mode": "w",  # 'w' overwrite file, 'a' append file.
        },
    },
    "loggers": {
        "": {"level": "DEBUG", "handlers": ["stderr", "file_dynamic"]},
        "dbs": {
            "level": "INFO",
            "handlers": ["stderr", "file_dbs"],
            "propagate": False,
        },
    },
}
