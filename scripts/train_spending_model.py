import logging
from pathlib import Path
import pandas as pd
import mlflow

from src.retention_lab.model_tracking import configure_mlflow
from src.retention_lab.spending_model import train_spending_model


def main() -> None:
    """
    Load spending datasets, prepare model inputs and targets,
    configure MLflow tracking and train the regression model.

    Returns:
        None.

    Raises:
        FileNotFoundError: If a required dataset is missing.
        ValueError: If dataset validation fails.
        Exception: If the training pipeline fails.
    """

    # Resolve the project root directory.
    root = Path(__file__).resolve().parents[1]

    # Define the regression dataset and logging directories.
    source = root / "data/processed/spending"
    log_dir = root / "logs"

    # Create the logging directory if it does not exist.
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to the terminal and a persistent log file.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train_spending_model.log"),
            logging.StreamHandler()
        ]
    )

    # Retrieve the module logger.
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting spending model training pipeline.")

        # Define the training and validation dataset paths.
        train_path = source / "train.csv"
        val_path = source / "validation.csv"

        # Verify that both required datasets exist.
        for path in [train_path, val_path]:
            if not path.is_file():
                raise FileNotFoundError(f"Dataset not found: {path}")

        # Load the regression datasets into Pandas DataFrames.
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)

        # Ensure neither dataset is empty.
        if train.empty or val.empty:
            raise ValueError("Training or validation dataset is empty.")

        # Define columns that must not be used as model inputs.
        excluded = ["customer_id", "scoring_date", "churn", "actual_90d_spend"]

        # Verify that the required columns exist in both datasets.
        for name, dataset in [("train", train), ("validation", val)]:
            missing = set(excluded) - set(dataset.columns)

            if missing:
                raise ValueError(f"Missing required columns in {name}: {missing}")

        # Select historical customer features while excluding identifiers and targets.
        features = [col for col in train.columns if col not in excluded]

        # Verify that the expected 24 customer features are available.
        if len(features) != 24:
            raise ValueError(f"Expected 24 features but found {len(features)}.")

        # Ensure validation contains all required training features.
        missing_features = set(features) - set(val.columns)

        if missing_features:
            raise ValueError(f"Missing validation features: {missing_features}")

        # Separate training features from actual future spending.
        X_train = train[features]
        y_train = train["actual_90d_spend"]

        # Separate validation features from actual future spending.
        X_val = val[features]
        y_val = val["actual_90d_spend"]

        # Validate spending targets for both datasets.
        for name, target in [("train", y_train), ("validation", y_val)]:
            if target.isna().any() or (target <= 0).any():
                raise ValueError(f"Invalid spending targets in {name}.")

        # Log dataset dimensions before training.
        logger.info("Training samples: %s", f"{len(X_train):,}")
        logger.info("Validation samples: %s", f"{len(X_val):,}")
        logger.info("Input features: %s", len(features))

        # Establish the DagsHub MLflow tracking connection.
        configure_mlflow()

        # Select a separate experiment for spending regression.
        mlflow.set_experiment("customer-spending-prediction")

        # Train the regression model and evaluate validation performance.
        train_spending_model(X_train, y_train, X_val, y_val)

        logger.info("Spending training pipeline completed successfully.")

    except Exception:
        # Record the error message and complete stack trace.
        logger.exception("Spending training pipeline failed.")

        # Propagate the exception to indicate execution failure.
        raise


# Execute the training pipeline when the script is run directly.
if __name__ == "__main__":
    main()