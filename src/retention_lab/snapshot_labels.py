from src.retention_lab.snapshot_features import build_snapshot_features


def build_labeled_snapshot(history, future, scoring_date):
    # Calculate 24 customer features using past purchases only.
    snapshot = build_snapshot_features(history, scoring_date)

    # Identify customers who made a purchase during the next 90 days.
    returned_customers = future["customer_id"].unique()

    # Assign churn = 1 if the customer made no purchase; otherwise 0.
    snapshot["churn"] = (~snapshot["customer_id"].isin(returned_customers)).astype(int)

    return snapshot