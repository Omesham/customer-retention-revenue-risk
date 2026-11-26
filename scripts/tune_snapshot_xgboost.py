from pathlib import Path
import pandas as pd

from src.retention_lab.model_tracking import configure_mlflow
from src.retention_lab.xgboost_tuning import tune_xgboost


def main():
    # Load the chronological training and validation datasets.
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data/processed/snapshots"

    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")

    # Keep customer IDs, scoring dates, and churn labels out of model features.
    features = train.drop(columns=["customer_id", "scoring_date", "churn"]).columns.tolist()

    # Run the existing six hyperparameter experiments.
    configure_mlflow()
    tune_xgboost(train[features], train["churn"], validation[features], validation["churn"])


if __name__ == "__main__":
    main()