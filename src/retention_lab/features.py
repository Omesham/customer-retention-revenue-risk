import pandas as pd


def build_customer_features(df, snapshot_date):
    """Calculate customer purchasing behavior up to a snapshot date."""

    # Convert the snapshot date to a pandas timestamp.
    snapshot_date = pd.Timestamp(snapshot_date)

    # Use only transactions before the snapshot date.
    history = df.loc[df["invoice_date"] < snapshot_date].copy()

    # Calculate revenue for each transaction line.
    history["revenue"] = history["quantity"] * history["unit_price"]

    # Group transactions by customer.
    customers = history.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"),
        frequency=("invoice", "nunique"),
        monetary=("revenue", "sum"),
    )

    # Calculate days since each customer's last purchase.
    customers["recency_days"] = (snapshot_date - customers["last_purchase"]).dt.days

    # Return customer-level features.
    return customers.reset_index()


def select_active_customers(features):
    # Keep customers who purchased within 124 days
    # before our prediction date (September 1, 2011).
    return features.loc[
        features["recency_days"] <= 124
    ].copy()