import os
import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_config() -> dict:
    """
    Load and validate environment-specific database configuration.

    Returns:
        dict: PostgreSQL connection parameters.

    Raises:
        ValueError: If the environment or configuration is invalid.
    """
    try:
        # Identify the project root directory.
        root = Path(__file__).resolve().parents[2]

        # Select development unless another environment is specified.
        app_env = os.getenv("APP_ENV", "development")

        # Map application environments to configuration files.
        env_files = {
            "development": ".env.dev",
            "production": ".env.prod"
        }

        # Reject unsupported environments.
        if app_env not in env_files:
            raise ValueError(f"Invalid application environment: {app_env}")

        # Load configuration without overriding existing environment variables.
        load_dotenv(root / env_files[app_env], override=False)

        # Map environment variables to PostgreSQL connection parameters.
        config = {
            "host": os.getenv("POSTGRES_HOST"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "dbname": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD")
        }

        # Ensure all required configuration values are present.
        missing = [key for key, value in config.items() if not value]

        if missing:
            raise ValueError(f"Missing database configuration: {missing}")

        # Record the selected environment without exposing credentials.
        logger.info("Configuration loaded for %s", app_env)

        return config

    except Exception:
        logger.exception("Failed to load application configuration.")
        raise