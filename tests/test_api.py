from unittest.mock import patch
from fastapi.testclient import TestClient
from src.retention_lab.api.main import app

client = TestClient(app)


def test_health_success():
    """Verify that the health endpoint reports database availability."""
    with patch("src.retention_lab.api.routes.execute_query", return_value=[{"result": 1}]):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["database"] == "connected"


def test_health_database_failure():
    """Verify that database failures return HTTP 503."""
    with patch("src.retention_lab.api.routes.execute_query", side_effect=Exception("Database unavailable")):
        response = client.get("/health")

        assert response.status_code == 503


def test_customer_not_found():
    """Verify that an unknown customer returns HTTP 404."""
    with patch("src.retention_lab.api.routes.execute_query", return_value=[]):
        response = client.get("/customers/unknown")

        assert response.status_code == 404


def test_invalid_customer_limit():
    """Verify that invalid pagination parameters are rejected."""
    response = client.get("/customers?limit=0")

    assert response.status_code == 422


def test_invalid_revenue_risk_limit():
    """Verify that invalid revenue risk limits are rejected."""
    response = client.get("/revenue-risk/top?limit=1001")

    assert response.status_code == 422