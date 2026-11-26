from pathlib import Path
import pandas as pd

from src.retention_lab.model_tracking import configure_mlflow
from src.retention_lab.xgboost_model import train_xgboost


def main():
    # Load chronological training and validation datasets.
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data/processed/snapshots"

    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")

    # Exclude customer identifiers, scoring dates, and churn labels.
    feature_columns = train.drop(columns=["customer_id", "scoring_date", "churn"]).columns.tolist()

    X_train = train[feature_columns]
    y_train = train["churn"]

    X_val = validation[feature_columns]
    y_val = validation["churn"]

    # Ensure all 24 features are present and contain valid values.
    assert len(feature_columns) == 24, f"Expected 24 features, found {len(feature_columns)}"
    assert not X_train.isna().any().any(), "Training features contain missing values."
    assert not X_val.isna().any().any(), "Validation features contain missing values."

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Features: {len(feature_columns)}")

    # Train on historical snapshots and evaluate on June 12, 2011.
    # Save the experiment and model to DagsHub.
    configure_mlflow()
    train_xgboost(X_train, y_train, X_val, y_val, run_name="xgboost-24-features-temporal")


if __name__ == "__main__":
    main()