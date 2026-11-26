def build_spending_snapshot(snapshot, future):
    """
    Build a regression dataset for returning customers.

    Args:
        snapshot (pd.DataFrame): Customer features and churn labels at a scoring date.
        future (pd.DataFrame): Customer orders over the next 90 days.

    Returns:
        pd.DataFrame: Returning customers with their actual 90-day spending.
    """
    # Keep only customers who returned.
    data = snapshot[snapshot["churn"] == 0].copy()

    # Calculate each customer's total spending over the next 90 days.
    future_spend = future.groupby("customer_id")["order_spend"].sum()

    # Assign the regression target.
    data["actual_90d_spend"] = data["customer_id"].map(future_spend)

    return data