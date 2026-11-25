import logging

from fastapi import APIRouter, HTTPException, Query

from src.retention_lab.database import execute_query
from src.retention_lab.api.schemas import CustomerPrediction, RevenueSummary, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> dict:
    """
    Check application and database availability.

    Returns:
        dict: Application and database health status.
    """
    try:
        # Execute a lightweight query to verify database connectivity.
        execute_query("SELECT 1")

        return {"status": "healthy", "database": "connected"}

    except Exception:
        logger.exception("Database health check failed.")
        raise HTTPException(status_code=503, detail="Database unavailable")


@router.get("/customers", response_model=list[CustomerPrediction])
def get_customers(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)) -> list[dict]:
    """
    Retrieve customer predictions using pagination.

    Args:
        limit (int): Maximum number of records to retrieve.
        offset (int): Number of records to skip.

    Returns:
        list[dict]: Customer prediction records.
    """
    try:
        # Retrieve the latest scoring snapshot in manageable batches.
        query = """
            SELECT *
            FROM customer_revenue_risk
            WHERE scoring_date = (
                SELECT MAX(scoring_date) FROM customer_revenue_risk
            )
            ORDER BY customer_id
            LIMIT %s OFFSET %s
        """

        return execute_query(query, (limit, offset))

    except Exception:
        logger.exception("Failed to retrieve customers.")
        raise HTTPException(status_code=500, detail="Unable to retrieve customers")


@router.get("/customers/{customer_id}", response_model=list[CustomerPrediction])
def get_customer(customer_id: str) -> list[dict]:
    """
    Retrieve prediction history for a specific customer.

    Args:
        customer_id (str): Unique customer identifier.

    Returns:
        list[dict]: Customer predictions ordered by scoring date.
    """
    try:
        # Retrieve all available predictions for the customer.
        query = """
            SELECT *
            FROM customer_revenue_risk
            WHERE customer_id = %s
            ORDER BY scoring_date DESC
        """

        records = execute_query(query, (customer_id,))

        # Return a not-found response when the customer does not exist.
        if not records:
            raise HTTPException(status_code=404, detail="Customer not found")

        return records

    except HTTPException:
        raise

    except Exception:
        logger.exception("Failed to retrieve customer %s.", customer_id)
        raise HTTPException(status_code=500, detail="Unable to retrieve customer")


@router.get("/revenue-risk/top", response_model=list[CustomerPrediction])
def get_top_risk_customers(limit: int = Query(10, ge=1, le=1000)) -> list[dict]:
    """
    Retrieve customers with the highest estimated revenue exposure.

    Args:
        limit (int): Maximum number of customers to retrieve.

    Returns:
        list[dict]: Customers ranked by estimated revenue exposure.
    """
    try:
        # Rank customers within the latest scoring snapshot.
        query = """
            SELECT *
            FROM customer_revenue_risk
            WHERE scoring_date = (
                SELECT MAX(scoring_date) FROM customer_revenue_risk
            )
            ORDER BY revenue_at_risk_90d DESC, customer_id
            LIMIT %s
        """

        return execute_query(query, (limit,))

    except Exception:
        logger.exception("Failed to retrieve high-risk customers.")
        raise HTTPException(status_code=500, detail="Unable to retrieve high-risk customers")


@router.get("/summary", response_model=RevenueSummary)
def get_summary() -> dict:
    """
    Calculate aggregate business metrics for the latest scoring date.

    Returns:
        dict: Customer count, average churn probability, and revenue metrics.
    """
    try:
        # Aggregate predictions without combining different scoring dates.
        query = """
            SELECT
                scoring_date,
                COUNT(*) AS total_customers,
                AVG(churn_probability) AS average_churn_probability,
                SUM(revenue_at_risk_90d) AS total_revenue_at_risk,
                SUM(expected_90d_revenue) AS total_expected_revenue
            FROM customer_revenue_risk
            WHERE scoring_date = (
                SELECT MAX(scoring_date) FROM customer_revenue_risk
            )
            GROUP BY scoring_date
        """

        records = execute_query(query)

        # Ensure predictions exist before returning aggregate metrics.
        if not records:
            raise HTTPException(status_code=404, detail="No predictions available")

        return records[0]

    except HTTPException:
        raise

    except Exception:
        logger.exception("Failed to retrieve revenue summary.")
        raise HTTPException(status_code=500, detail="Unable to retrieve revenue summary")