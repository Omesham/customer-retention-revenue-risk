import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Retrieve the module logger.
logger = logging.getLogger(__name__)


def train_spending_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> XGBRegressor:
    """
    Train and validate a model that predicts 90-day customer spending.

    Args:
        X_train (pd.DataFrame): Historical customer features for training.
        y_train (pd.Series): Actual 90-day spending for training customers.
        X_val (pd.DataFrame): Historical customer features for validation.
        y_val (pd.Series): Actual 90-day spending for validation customers.

    Returns:
        XGBRegressor: Trained spending regression model.

    Raises:
        Exception: If model training, evaluation or tracking fails.
    """

    try:
        logger.info("Starting spending model training.")

        # Define the initial regression model hyperparameters.
        params = {
            "n_estimators": 600,           # Maximum number of boosting trees.
            "max_depth": 3,                # Maximum depth of each tree.
            "learning_rate": 0.03,         # Contribution of each new tree.
            "subsample": 0.8,              # Fraction of training rows per tree.
            "colsample_bytree": 0.8,       # Fraction of features per tree.
            "objective": "reg:squarederror", # Minimize squared prediction errors.
            "eval_metric": "mae",          # Monitor mean absolute error.
            "early_stopping_rounds": 40,   # Stop after 40 rounds without improvement.
            "random_state": 42,            # Ensure reproducible results.
            "n_jobs": 4                    # Number of parallel processing threads.
        }

        # Initialize the XGBoost regression model.
        model = XGBRegressor(**params)

        # Create a new MLflow experiment run.
        with mlflow.start_run(run_name="spending-xgboost-baseline"):

            # Record model hyperparameters for experiment reproducibility.
            mlflow.log_params(params)

            # Train the model and monitor validation MAE for early stopping.
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Generate spending predictions for validation customers.
            predictions = model.predict(X_val)

            # Replace negative predictions with zero because spending cannot be negative.
            predictions = np.maximum(predictions, 0)

            # Calculate regression performance metrics.
            metrics = {
                # Average absolute difference between predicted and actual spending.
                "validation_mae": mean_absolute_error(y_val, predictions),

                # Prediction error metric that penalizes larger errors more heavily.
                "validation_rmse": np.sqrt(mean_squared_error(y_val, predictions)),

                # Measure how well the model explains variation in actual spending.
                "validation_r2": r2_score(y_val, predictions)
            }

            # Record validation metrics in MLflow.
            mlflow.log_metrics(metrics)

            # Display each evaluation metric in the terminal and log file.
            for name, value in metrics.items():
                logger.info("%s: %.4f", name, value)

            # Calculate aggregate predicted and actual spending.
            estimated_total = float(predictions.sum())
            actual_total = float(y_val.sum())

            # Record aggregate spending in MLflow.
            mlflow.log_metric("estimated_total_spending", estimated_total)
            mlflow.log_metric("actual_total_spending", actual_total)

            # Display aggregate spending for comparison.
            logger.info("Estimated total spending: %s", f"{estimated_total:,.2f}")
            logger.info("Actual total spending: %s", f"{actual_total:,.2f}")

            # Record the feature names required during future predictions.
            mlflow.log_dict({"feature_names": X_train.columns.tolist()}, "config.json")

            # Save the trained regression model to MLflow.
            mlflow.xgboost.log_model(model, artifact_path="model")

            logger.info("Spending model training completed successfully.")

        # Return the trained regression model.
        return model

    except Exception:
        # Record the error message and complete stack trace.
        logger.exception("Spending model training failed.")

        # Propagate the exception to the calling script.
        raise