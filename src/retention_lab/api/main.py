import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.retention_lab.config import load_config
from src.retention_lab.logger import configure_logging
from src.retention_lab.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Args:
        app (FastAPI): FastAPI application instance.

    Yields:
        None: Control during the application's running lifetime.
    """
    try:
        # Load environment configuration and initialize centralized logging.
        load_config()
        configure_logging()

        logger.info("Customer Retention API started.")

        # Allow the application to process incoming requests.
        yield

    except Exception:
        logger.exception("Application lifecycle failed.")
        raise

    finally:
        logger.info("Customer Retention API stopped.")


# Initialize the application and its interactive documentation.
app = FastAPI(
    title="Customer Retention and Revenue Risk API",
    description="Customer churn predictions and estimated revenue exposure.",
    version="1.0.0",
    lifespan=lifespan
)

# Register the customer and revenue endpoints.
app.include_router(router)