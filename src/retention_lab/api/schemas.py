from datetime import date

from pydantic import BaseModel, Field


class CustomerPrediction(BaseModel):
    """Represent customer-level churn and revenue predictions."""

    customer_id: str
    scoring_date: date
    churn_probability: float = Field(ge=0, le=1)
    estimated_90d_spend: float
    revenue_at_risk_90d: float
    expected_90d_revenue: float


class RevenueSummary(BaseModel):
    """Represent aggregate customer and revenue metrics."""

    scoring_date: date
    total_customers: int
    average_churn_probability: float
    total_revenue_at_risk: float
    total_expected_revenue: float


class HealthResponse(BaseModel):
    """Represent application and database availability."""

    status: str
    database: str