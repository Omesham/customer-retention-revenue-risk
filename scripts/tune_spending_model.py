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
    Fine-tune the spending regression model using validation data.

    Returns:
        None.

    Raises:
        Exception: If dataset loading, training or experiment tracking fails.
    """

    # Define project directories.
    root = Path(__file__).resolve().parents[1]
    source = root / "data/processed/spending"
    log_dir = root / "logs"

    # Create the logging directory.
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure terminal and file logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "tune_spending_model.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    try:
        # Load training and validation datasets only.
        train = pd.read_csv(source / "train.csv")
        val = pd.read_csv(source / "validation.csv")

        # Exclude identifiers and target columns to prevent leakage.
        excluded = ["customer_id", "scoring_date", "churn", "actual_90d_spend"]
        features = [col for col in train.columns if col not in excluded]

        # Verify the expected number of input features.
        if len(features) != 24:
            raise ValueError(f"Expected 24 features, found {len(features)}.")

        # Separate historical features and regression targets.
        X_train = train[features]
        y_train = train["actual_90d_spend"]
        X_val = val[features]
        y_val = val["actual_90d_spend"]

        # Define parameters shared across all experiments.
        base_params = {
            "n_estimators": 1000,             # Maximum boosting rounds.
            "learning_rate": 0.03,            # Contribution of each tree.
            "subsample": 0.8,                 # Fraction of training rows.
            "colsample_bytree": 0.8,          # Fraction of input features.
            "objective": "reg:squarederror",  # Minimize squared errors.
            "eval_metric": "mae",             # Monitor validation MAE.
            "early_stopping_rounds": 40,      # Stop when MAE stops improving.
            "random_state": 42,               # Reproducibility.
            "n_jobs": 4                       # Parallel processing threads.
        }

        # Test different tree depths and regularization strengths.
        experiments = [
            {"max_depth": 2, "min_child_weight": 1, "reg_lambda": 1},
            {"max_depth": 3, "min_child_weight": 5, "reg_lambda": 5},
            {"max_depth": 4, "min_child_weight": 5, "reg_lambda": 5},
            {"max_depth": 5, "min_child_weight": 5, "reg_lambda": 5},
            {"max_depth": 3, "min_child_weight": 10, "reg_lambda": 10},
            {"max_depth": 4, "min_child_weight": 10, "reg_lambda": 10}
        ]

        # Connect to the existing spending prediction experiment.
        configure_mlflow()
        mlflow.set_experiment("customer-spending-prediction")

        # Track the best validation result.
        best_mae = float("inf")
        best_run_id = None

        # Train and evaluate each hyperparameter configuration.
        for index, experiment in enumerate(experiments, start=1):

            # Merge shared parameters with experiment-specific parameters.
            params = {**base_params, **experiment}

            logger.info("Running experiment %s/%s", index, len(experiments))
            logger.info("Parameters: %s", experiment)

            with mlflow.start_run(run_name=f"spending-tuning-{index}") as run:

                # Record hyperparameters.
                mlflow.log_params(params)

                # Initialize and train the regression model.
                model = XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                # Generate non-negative spending predictions.
                predictions = np.maximum(model.predict(X_val), 0)

                # Calculate individual customer prediction errors.
                mae = mean_absolute_error(y_val, predictions)
                rmse = np.sqrt(mean_squared_error(y_val, predictions))
                r2 = r2_score(y_val, predictions)

                # Calculate aggregate spending estimation error.
                estimated_total = float(predictions.sum())
                actual_total = float(y_val.sum())
                total_error_pct = (estimated_total - actual_total) / actual_total * 100

                # Record evaluation metrics in MLflow.
                metrics = {
                    "validation_mae": mae,
                    "validation_rmse": rmse,
                    "validation_r2": r2,
                    "estimated_total_spending": estimated_total,
                    "actual_total_spending": actual_total,
                    "total_error_pct": total_error_pct,
                    "best_iteration": model.best_iteration
                }

                mlflow.log_metrics(metrics)

                # Display experiment results.
                for name, value in metrics.items():
                    logger.info("%s: %.4f", name, value)

                # Save feature names and model configuration.
                mlflow.log_dict({
                    "feature_names": features,
                    "params": params,
                    "best_iteration": model.best_iteration
                }, "config.json")

                # Save the trained model locally using a unique run ID.
                model_dir = root / "models" / run.info.run_id
                model_dir.mkdir(parents=True, exist_ok=False)
                model.save_model(model_dir / "spending_model.json")

                # Save the model in MLflow.
                mlflow.xgboost.log_model(model, name="model")

                # Update the best model based on validation MAE.
                if mae < best_mae:
                    best_mae = mae
                    best_run_id = run.info.run_id
                    logger.info("New best validation MAE: %.4f", best_mae)

        # Report the best tuning experiment.
        logger.info("Best validation MAE: %.4f", best_mae)
        logger.info("Best run ID: %s", best_run_id)
        logger.info("Spending model tuning completed successfully.")

    except Exception:
        # Record the exception and complete stack trace.
        logger.exception("Spending model tuning failed.")
        raise


# Execute the tuning pipeline.
if __name__ == "__main__":
    main()