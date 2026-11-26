
import numpy as np
import pandas as pd


def build_customer_features(df, snapshot_date):
    # Use only transactions before the prediction date to prevent leakage.
    snapshot_date = pd.Timestamp(snapshot_date)
    history = df.loc[df["invoice_date"] < snapshot_date].copy()
    history["revenue"] = history["quantity"] * history["unit_price"]

    # Calculate each customer's purchasing history.
    customers = history.groupby("customer_id").agg(
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        frequency=("invoice", "nunique"),
        monetary=("revenue", "sum")
    )

    # Calculate traditional RFM and customer tenure.
    customers["recency_days"] = (snapshot_date - customers["last_purchase"]).dt.days
    customers["tenure_days"] = (snapshot_date - customers["first_purchase"]).dt.days
    customers["avg_order_value"] = customers["monetary"] / customers["frequency"]

    # Calculate purchase activity in recent time windows.
    for days in [30, 90]:
        recent = history.loc[history["invoice_date"] >= snapshot_date - pd.Timedelta(days=days)]
        customers[f"orders_last_{days}d"] = recent.groupby("customer_id")["invoice"].nunique()
        customers[f"orders_last_{days}d"] = customers[f"orders_last_{days}d"].fillna(0).astype(int)

    # Compare recent purchases against the preceding 90 days.
    previous = history.loc[
        (history["invoice_date"] >= snapshot_date - pd.Timedelta(days=180)) &
        (history["invoice_date"] < snapshot_date - pd.Timedelta(days=90))
    ]
    customers["orders_prev_90d"] = previous.groupby("customer_id")["invoice"].nunique()
    customers["orders_prev_90d"] = customers["orders_prev_90d"].fillna(0).astype(int)
    customers["order_change_90d"] = customers["orders_last_90d"] - customers["orders_prev_90d"]

    # Identify individual purchases and calculate time between orders.
    orders = history[["customer_id", "invoice", "invoice_date"]].drop_duplicates()
    orders = orders.sort_values(["customer_id", "invoice_date", "invoice"])
    orders["gap_days"] = orders.groupby("customer_id")["invoice_date"].diff().dt.total_seconds() / 86400

    # Ignore zero-day gaps caused by purchases at the same timestamp.
    gaps = orders.loc[orders["gap_days"] > 0]
    gap_stats = gaps.groupby("customer_id")["gap_days"].agg(["mean", "std", "count"])

    # Relative recency: how overdue is the customer compared with their normal buying interval?
    customers["avg_purchase_gap_days"] = gap_stats["mean"]
    customers["overdue_ratio"] = customers["recency_days"] / customers["avg_purchase_gap_days"]

    # Purchase regularity: variation in the customer's purchase intervals.
    customers["purchase_regularity_cv"] = gap_stats["std"] / gap_stats["mean"]
    # Align gap counts with every customer and assign zero when no gap exists.
    customers["observed_gap_count"] = gap_stats["count"].reindex(customers.index).fillna(0).astype(int)

    # Calculate customer spending during the most recent 90 days.
    recent_90 = history.loc[history["invoice_date"] >= snapshot_date - pd.Timedelta(days=90)]
    customers["spend_last_90d"] = recent_90.groupby("customer_id")["revenue"].sum()
    customers["spend_last_90d"] = customers["spend_last_90d"].fillna(0)

    # Compare recent spending with average spending over a 90-day period.
    # Require at least 90 days of history for a meaningful comparison.
    customers["spend_velocity_90d"] = np.where(
        customers["tenure_days"] >= 90,
        customers["spend_last_90d"] / (customers["monetary"] / customers["tenure_days"].clip(lower=1) * 90),
        np.nan
    )

    # Identify products purchased again in later orders.
    sku_orders = history[["customer_id", "invoice", "invoice_date", "stock_code"]].dropna(subset=["stock_code"])
    sku_orders = sku_orders.drop_duplicates(["customer_id", "invoice", "stock_code"])
    sku_orders = sku_orders.sort_values(["customer_id", "invoice_date", "invoice"])
    sku_orders["repeat_sku"] = sku_orders.duplicated(["customer_id", "stock_code"])

    # Proportion of orders containing at least one previously purchased product.
    repeat_orders = sku_orders.groupby(["customer_id", "invoice"])["repeat_sku"].any()
    customers["repeat_sku_ratio"] = repeat_orders.groupby(level=0).mean()

    # Preserve missing-history indicators before replacing undefined ratios.
    customers["has_gap_history"] = (customers["observed_gap_count"] > 0).astype(int)
    customers["has_spend_velocity_history"] = (customers["tenure_days"] >= 90).astype(int)

    # Use zero placeholders where customer history is insufficient.
    ratio_columns = [
        "avg_purchase_gap_days",
        "overdue_ratio",
        "purchase_regularity_cv",
        "spend_velocity_90d",
        "repeat_sku_ratio"
    ]
    customers[ratio_columns] = customers[ratio_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    return customers.reset_index()


def select_active_customers(features):
    # Retain customers considered active at the prediction date.
    return features.loc[features["recency_days"] <= 124].copy()
