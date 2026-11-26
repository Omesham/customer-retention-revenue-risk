import logging

from dataclasses_json import config
import psycopg
from psycopg.rows import dict_row

from src.retention_lab.config import load_config

logger = logging.getLogger(__name__)


def execute_query(query: str, params: tuple = ()) -> list[dict]:
    """
    Execute a database query and retrieve its results.

    Args:
        query (str): SQL statement to execute.
        params (tuple): Values for parameterized SQL placeholders.

    Returns:
        list[dict]: Database records represented as dictionaries.

    Raises:
        Exception: If the database connection or query fails.
    """
    try:
        # Retrieve database connection settings.
        config = load_config()

        # Return database rows as dictionaries using column names.
        # Limit database connection attempts to five seconds.
        with psycopg.connect(**config, row_factory=dict_row, connect_timeout=5) as connection:
            with connection.cursor() as cursor:

                # Execute the query with safely bound parameters.
                cursor.execute(query, params)

                return cursor.fetchall()

    except Exception:
        logger.exception("Database query failed.")
        raise