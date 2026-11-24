import pandas as pd


def calculate_revenue_risk(customers, model, feature_names):
    # Predict each customer's probability of churning.
    probabilities = model.predict_proba(customers[feature_names])[:, 1]

    # Estimate typical 90-day spending from the previous 365 days.
    # Use at least 30 days of exposure to avoid extreme estimates for new customers.
    observed_days = customers["tenure_days"].clip(lower=30, upper=365)
    expected_spend = customers["spend_last_365d"] / observed_days * 90

    # Create a customer-level revenue risk report.
    results = customers[["customer_id", "scoring_date"]].copy()
    results["churn_probability"] = probabilities
    results["estimated_90d_spend"] = expected_spend
    results["revenue_at_risk_90d"] = probabilities * expected_spend

    # Rank customers by estimated financial exposure.
    return results.sort_values("revenue_at_risk_90d", ascending=False)