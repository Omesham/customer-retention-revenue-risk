import pandas as pd


def build_churn_labels(active_customers, df):
    # Identify purchases made during the 90-day prediction window.
    future_purchases = df.loc[
        (df["invoice_date"] >= "2011-09-01")
        & (df["invoice_date"] < "2011-11-30")
    ]

    # Get customer IDs that made at least one purchase.
    returned_ids = future_purchases["customer_id"].unique()

    # Start with customers who were active on September 1.
    labels = active_customers[["customer_id"]].copy()

    # 1 = No purchase within 90 days (churned).
    # 0 = Purchased within 90 days (retained).
    labels["churn"] = (
        ~labels["customer_id"].isin(returned_ids)
    ).astype(int)

    return labels