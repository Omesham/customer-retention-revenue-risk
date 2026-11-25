import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


def configure_logging() -> None:
    """
    Configure centralized console and rotating file logging.

    Returns:
        None.

    Raises:
        Exception: If logging configuration fails.
    """
    try:
        # Identify the project root and create the logs directory.
        root = Path(__file__).resolve().parents[2]
        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Retrieve the environment-specific logging level.
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        # Rotate log files when they reach the configured size.
        file_handler = RotatingFileHandler(
            log_dir / "application.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )

        # Configure logging for all application modules.
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[
                file_handler,
                logging.StreamHandler()
            ],
            force=True
        )

    except Exception:
        logging.exception("Logging configuration failed.")
        raise