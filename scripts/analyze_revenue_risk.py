import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from src.retention_lab.revenue_risk import calculate_revenue_risk


def main():
    root = Path(__file__).resolve().parents[1]
    run_id = "a81c8ba692eb43af8eda738acccbb887"
    model_dir = root / "models" / run_id

    # Load the final model and its original feature order.
    model = XGBClassifier()
    model.load_model(str(model_dir / "xgboost_model.json"))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    # Use the September 10, 2011 customer snapshot.
    customers = pd.read_csv(root / "data/processed/snapshots/test.csv")

    # Calculate customer-level revenue exposure without future transaction data.
    results = calculate_revenue_risk(customers, model, config["feature_names"])
    # Inspect purchase histories for the 10 customers with the highest estimated risk.
    top_ids = results.head(10)["customer_id"]
    columns = ["customer_id", "tenure_days", "order_count", "avg_order_spend", "spend_last_365d"]

    print("\nPurchase history of highest-risk customers:")
    print(customers.loc[customers["customer_id"].isin(top_ids), columns].to_string(index=False))
    # Save the full report for analysis and dashboard development.
    report_path = root / "reports/revenue_at_risk.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_path, index=False)

    print(f"Customers analyzed: {len(results):,}")
    print(f"Estimated total 90-day revenue exposure: {results['revenue_at_risk_90d'].sum():,.2f}")
    print("\nTop 10 customers by estimated revenue at risk:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()