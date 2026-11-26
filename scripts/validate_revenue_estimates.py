from pathlib import Path
import pandas as pd

from src.retention_lab.data import load_retail
from src.retention_lab.preprocessing import clean_retail
from src.retention_lab.snapshot_orders import build_orders


def main():
    root = Path(__file__).resolve().parents[1]
    customers = pd.read_csv(root / "data/processed/snapshots/validation.csv")
    orders = build_orders(clean_retail(load_retail(root / "data/raw/online_retail_II.xlsx")))

    # Use the validation scoring date and the following 90 days.
    scoring_date = pd.Timestamp("2011-06-12")
    end_date = scoring_date + pd.Timedelta(days=90)
    future = orders[(orders["order_date"] >= scoring_date) & (orders["order_date"] < end_date)]

    # Calculate actual spending from purchases in the following 90 days.
    future_spend = future.groupby("customer_id")["order_spend"].sum()
    customers["customer_id"] = customers["customer_id"].astype(str)
    customers["actual_90d_spend"] = customers["customer_id"].map(future_spend).fillna(0)

    # Reproduce our existing historical spending estimate.
    observed_days = customers["tenure_days"].clip(lower=30, upper=365)
    customers["estimated_90d_spend"] = customers["spend_last_365d"] / observed_days * 90

    # Measure how far our estimates differ from actual spending.
    customers["absolute_error"] = (customers["estimated_90d_spend"] - customers["actual_90d_spend"]).abs()

    print(f"Customers evaluated: {len(customers):,}")
    print(f"Estimated total spending: {customers['estimated_90d_spend'].sum():,.2f}")
    print(f"Actual total spending: {customers['actual_90d_spend'].sum():,.2f}")
    print(f"Mean absolute error: {customers['absolute_error'].mean():,.2f}")
    print(f"Median absolute error: {customers['absolute_error'].median():,.2f}")

    # Examine customers who actually returned, to assess the spending assumption separately.
    returned = customers[customers["churn"] == 0]
    print(f"\nReturning customers: {len(returned):,}")
    print(f"Returning customers' mean absolute error: {returned['absolute_error'].mean():,.2f}")
    print(f"Returning customers' actual spending: {returned['actual_90d_spend'].sum():,.2f}")


if __name__ == "__main__":
    main()