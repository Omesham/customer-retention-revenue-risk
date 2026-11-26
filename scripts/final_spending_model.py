import logging
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.retention_lab.model_tracking import configure_mlflow


def main() -> None:
    """
    Retrain the selected spending model on training and validation
    datasets and evaluate its performance on unseen test data.

    Returns:
        None.

    Raises:
        Exception: If data loading, training or evaluation fails.
    """

    # Define project directories.
    root = Path(__file__).resolve().parents[1]
    source = root / "data/processed/spending"
    log_dir = root / "logs"

    # Create the logging directory.
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure console and file logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "final_spending_model.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    try:
        # Load all three regression datasets.
        train = pd.read_csv(source / "train.csv")
        val = pd.read_csv(source / "validation.csv")
        test = pd.read_csv(source / "test.csv")

        # Combine training and validation for final training.
        combined = pd.concat([train, val], ignore_index=True)

        # Exclude identifiers and future-derived target columns.
        excluded = ["customer_id", "scoring_date", "churn", "actual_90d_spend"]
        features = [col for col in train.columns if col not in excluded]

        # Validate the expected feature count.
        if len(features) != 24:
            raise ValueError(f"Expected 24 features, found {len(features)}.")

        # Separate training features and targets.
        X_train = combined[features]
        y_train = combined["actual_90d_spend"]

        # Separate test features and targets.
        X_test = test[features]
        y_test = test["actual_90d_spend"]

        logger.info("Final training samples: %s", f"{len(X_train):,}")
        logger.info("Test samples: %s", f"{len(X_test):,}")

        # Configure the best hyperparameters from Experiment 5.
        params = {
            "n_estimators": 310,               # Best iteration 309 + 1.
            "max_depth": 3,                    # Maximum tree depth.
            "min_child_weight": 10,            # Minimum child-node Hessian weight.
            "reg_lambda": 10,                  # L2 regularization.
            "learning_rate": 0.03,             # Contribution of each tree.
            "subsample": 0.8,                  # Fraction of training rows.
            "colsample_bytree": 0.8,           # Fraction of input features.
            "objective": "reg:squarederror",   # Squared-error regression.
            "eval_metric": "mae",              # Mean absolute error.
            "random_state": 42,                # Reproducibility.
            "n_jobs": 4                        # Parallel processing threads.
        }

        # Connect to the spending experiment.
        configure_mlflow()
        mlflow.set_experiment("customer-spending-prediction")

        with mlflow.start_run(run_name="spending-xgboost-final") as run:

            # Record the selected hyperparameters.
            mlflow.log_params(params)

            # Record the selected tuning experiment for traceability.
            mlflow.set_tag("parent_tuning_run", "440287f2f3e94f07b81b24da5dc32463")

            # Initialize and train the final model.
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            # Generate non-negative spending predictions.
            predictions = np.maximum(model.predict(X_test), 0)

            # Calculate aggregate predicted and actual spending.
            estimated_total = float(predictions.sum())
            actual_total = float(y_test.sum())

            # Prevent division by zero.
            if actual_total <= 0:
                raise ValueError("Actual total test spending must be positive.")

            # Calculate final test performance.
            metrics = {
                "test_mae": mean_absolute_error(y_test, predictions),
                "test_rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "test_r2": r2_score(y_test, predictions),
                "estimated_total_spending": estimated_total,
                "actual_total_spending": actual_total,
                "total_error_pct": (estimated_total - actual_total) / actual_total * 100
            }

            # Record and display test metrics.
            mlflow.log_metrics(metrics)

            for name, value in metrics.items():
                logger.info("%s: %.4f", name, value)

            # Create a unique directory without overwriting existing models.
            model_dir = root / "models" / run.info.run_id
            model_dir.mkdir(parents=True, exist_ok=False)

            # Save the final regression model locally.
            model.save_model(model_dir / "spending_model.json")

            # Save the feature names and model parameters.
            config = {"feature_names": features, "params": params}
            mlflow.log_dict(config, "config.json")

            # Register the model artifact in MLflow.
            mlflow.xgboost.log_model(model, name="model")

            logger.info("Final model directory: %s", model_dir)
            logger.info("Final run ID: %s", run.info.run_id)

        logger.info("Final spending model training completed successfully.")

    except Exception:
        # Log the exception and complete stack trace.
        logger.exception("Final spending model training failed.")
        raise


# Execute the final training pipeline.
if __name__ == "__main__":
    main()