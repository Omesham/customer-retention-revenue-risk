
import pandas as pd


def build_snapshot_features(history, scoring_date):
    # Prepare transaction dates for the scoring snapshot.
    scoring_date = pd.Timestamp(scoring_date).normalize()
    history = history.copy()
    history["shopping_day"] = history["order_date"].dt.normalize()

    # Calculate customer purchase history, spending, and basket features.
    customers = history.groupby("customer_id").agg(
        first_purchase=("shopping_day", "min"),
        last_purchase=("shopping_day", "max"),
        order_count=("invoice", "size"),
        shopping_days=("shopping_day", "nunique"),
        total_spend=("order_spend", "sum"),
        avg_order_spend=("order_spend", "mean"),
        avg_order_units=("order_units", "mean"),
        avg_order_lines=("order_lines", "mean")
    )

    # Calculate days since the first and most recent purchases.
    customers["recency_days"] = (scoring_date - customers["last_purchase"]).dt.days
    customers["tenure_days"] = (scoring_date - customers["first_purchase"]).dt.days

    # Use distinct shopping days to avoid zero-day gaps between same-day orders.
    shopping_days = history[["customer_id", "shopping_day"]].drop_duplicates().sort_values(["customer_id", "shopping_day"])
    shopping_days["gap_days"] = shopping_days.groupby("customer_id")["shopping_day"].diff().dt.days

    # Summarize purchase gaps for each customer.
    gaps = shopping_days.groupby("customer_id")["gap_days"].agg(
        avg_purchase_gap_days="mean",
        median_purchase_gap_days="median",
        std_purchase_gap_days="std",
        latest_purchase_gap_days="last"
    )

    customers = customers.join(gaps)

    # Customers with insufficient purchase history have unavailable gap statistics.
    gap_columns = ["avg_purchase_gap_days", "median_purchase_gap_days", "std_purchase_gap_days", "latest_purchase_gap_days"]
    customers[gap_columns] = customers[gap_columns].fillna(0)

    # Calculate purchasing habits relative to each customer's history.
    customers["overdue_ratio"] = customers["recency_days"] / (1 + customers["avg_purchase_gap_days"])
    customers["shopping_day_rate"] = customers["shopping_days"] / (30 + customers["tenure_days"])

    # Calculate order count and spending over five historical windows.
    for days in [30, 60, 90, 180, 365]:
        recent = history[history["order_date"] >= scoring_date - pd.Timedelta(days=days)]
        recent_stats = recent.groupby("customer_id").agg(orders=("invoice", "size"), spend=("order_spend", "sum"))

        # Customers without purchases in this window receive zero.
        customers[f"orders_last_{days}d"] = recent_stats["orders"].reindex(customers.index).fillna(0).astype(int)
        customers[f"spend_last_{days}d"] = recent_stats["spend"].reindex(customers.index).fillna(0)

    # Keep identifiers for dataset construction, but not as model inputs.
    customers = customers.drop(columns=["first_purchase", "last_purchase"])
    customers["scoring_date"] = scoring_date

    return customers.reset_index()
