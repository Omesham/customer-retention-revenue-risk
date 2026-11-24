import logging
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor


def main() -> None:
    """
    Calculate customer revenue exposure using trained churn
    classification and spending regression models.

    Returns:
        None.
    """

    # Define project directories.
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs"
    report_dir = root / "reports"

    # Create output directories.
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Configure terminal and file logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "final_revenue_risk.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    try:
        # Define the saved model paths.
        churn_path = root / "models/a81c8ba692eb43af8eda738acccbb887/xgboost_model.json"
        spending_path = root / "models/48010bbc4a7f43a39be43683fdd98389/spending_model.json"

        # Load the complete test dataset, including churned customers.
        test_path = root / "data/processed/snapshots/test.csv"

        # Verify that all required files exist.
        for path in [churn_path, spending_path, test_path]:
            if not path.is_file():
                raise FileNotFoundError(f"Required file not found: {path}")

        customers = pd.read_csv(test_path)

        # Reject an empty test dataset.
        if customers.empty:
            raise ValueError("Test dataset is empty.")

        # Load the trained churn classification model.
        churn_model = XGBClassifier()
        churn_model.load_model(churn_path)

        # Load the trained spending regression model.
        spending_model = XGBRegressor()
        spending_model.load_model(spending_path)

        # Retrieve the features expected by each trained model.
        churn_features = churn_model.get_booster().feature_names
        spending_features = spending_model.get_booster().feature_names

        # Validate that all required model features exist.
        for features in [churn_features, spending_features]:
            if not features:
                raise ValueError("Model feature names are missing.")

            missing = set(features) - set(customers.columns)

            if missing:
                raise ValueError(f"Missing features: {missing}")

        # Predict each customer's probability of churning.
        churn_probability = churn_model.predict_proba(customers[churn_features])[:, 1]

        # Predict spending if each customer returns.
        estimated_spending = spending_model.predict(customers[spending_features])

        # Prevent negative spending predictions.
        estimated_spending = np.maximum(estimated_spending, 0)

        # Prepare the customer-level revenue report.
        results = customers[["customer_id", "scoring_date"]].copy()

        # Store predictions from both models.
        results["churn_probability"] = churn_probability
        results["estimated_90d_spend"] = estimated_spending

        # Calculate estimated revenue exposure.
        results["revenue_at_risk_90d"] = churn_probability * estimated_spending

        # Calculate expected spending after accounting for churn.
        results["expected_90d_revenue"] = (1 - churn_probability) * estimated_spending

        # Rank customers by revenue exposure.
        results = results.sort_values("revenue_at_risk_90d", ascending=False)

        # Save a new report without overwriting the original report.
        output_path = report_dir / "final_revenue_at_risk.csv"
        results.to_csv(output_path, index=False, mode="x")

        # Log aggregate business metrics.
        logger.info("Customers analyzed: %s", f"{len(results):,}")
        logger.info("Estimated revenue exposure: £%s", f"{results['revenue_at_risk_90d'].sum():,.2f}")
        logger.info("Expected revenue: £%s", f"{results['expected_90d_revenue'].sum():,.2f}")

        # Display the ten customers with the highest revenue exposure.
        logger.info("Top 10 customers:\n%s", results.head(10).to_string(index=False))

        logger.info("Report saved to: %s", output_path)
        logger.info("Revenue risk analysis completed successfully.")

    except Exception:
        # Record the error and complete stack trace.
        logger.exception("Revenue risk analysis failed.")
        raise


# Execute the revenue risk analysis.
if __name__ == "__main__":
    main()