import logging
from pathlib import Path
import pandas as pd

from src.retention_lab.data import load_retail
from src.retention_lab.preprocessing import clean_retail
from src.retention_lab.snapshot_orders import build_orders
from src.retention_lab.snapshots import split_scoring_window
from src.retention_lab.spending_labels import build_spending_snapshot


def main() -> None:
    """
    Generate spending regression datasets from existing customer snapshots.

    Args:
        None.

    Returns:
        None.

    Raises:
        FileNotFoundError: A required input file is missing.
        FileExistsError: An output dataset already exists.
        ValueError: Dataset validation fails.
    """

    # Resolve the project root directory.
    root = Path(__file__).resolve().parents[1]

    # Define input, output and logging directories.
    source = root / "data/processed/snapshots"
    destination = root / "data/processed/spending"
    log_dir = root / "logs"

    # Create the logging directory if it does not exist.
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to both the terminal and a persistent log file.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "build_spending_dataset.log"),
            logging.StreamHandler()
        ]
    )

    # Create a logger for this module.
    logger = logging.getLogger(__name__)

    try:
        # Create the regression dataset output directory.
        destination.mkdir(parents=True, exist_ok=True)
        logger.info("Starting spending dataset preparation.")

        # Define the original retail dataset location.
        retail_path = root / "data/raw/online_retail_II.xlsx"

        # Ensure the original dataset exists before loading it.
        if not retail_path.is_file():
            raise FileNotFoundError(f"Retail dataset not found: {retail_path}")

        # Load, clean and aggregate transaction lines into customer orders.
        orders = build_orders(clean_retail(load_retail(retail_path)))
        logger.info("Loaded %s customer orders.", f"{len(orders):,}")

        # Preserve the existing chronological dataset splits.
        for split in ["train", "validation", "test"]:
            logger.info("Processing %s split.", split)

            # Locate the corresponding churn snapshot dataset.
            snapshot_path = source / f"{split}.csv"

            # Ensure the snapshot file exists.
            if not snapshot_path.is_file():
                raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

            # Load customer features and churn labels.
            snapshots = pd.read_csv(snapshot_path, dtype={"customer_id": str})

            # Reject empty snapshot datasets.
            if snapshots.empty:
                raise ValueError(f"{split} snapshot is empty.")

            # Store regression samples generated for each scoring date.
            results = []

            # Group customer snapshots by their respective scoring dates.
            for scoring_date, snapshot in snapshots.groupby("scoring_date"):

                # Retrieve orders within 90 days after the scoring date.
                # The historical orders are not needed because features already exist.
                _, future = split_scoring_window(orders, scoring_date)

                # Standardize customer IDs for accurate matching.
                future["customer_id"] = future["customer_id"].astype(str)

                # Retain returning customers and assign actual future spending.
                data = build_spending_snapshot(snapshot, future)

                # Skip scoring dates without returning customers.
                if data.empty:
                    logger.warning("No returning customers for %s.", scoring_date)
                    continue

                # Store the generated regression samples.
                results.append(data)

                # Log the number of returning customers for this scoring date.
                logger.info("%s | %s | Returning customers: %s", split, scoring_date, f"{len(data):,}")

            # Ensure at least one regression sample was generated.
            if not results:
                raise ValueError(f"No regression samples generated for {split}.")

            # Combine regression samples from all scoring dates.
            dataset = pd.concat(results, ignore_index=True)

            # Reject customers with missing regression targets.
            if dataset["actual_90d_spend"].isna().any():
                raise ValueError(f"Missing spending targets in {split}.")

            # Returning customers must have positive actual spending.
            if (dataset["actual_90d_spend"] <= 0).any():
                raise ValueError(f"Non-positive spending targets in {split}.")

            # Ensure each customer appears only once per scoring date.
            if dataset.duplicated(["customer_id", "scoring_date"]).any():
                raise ValueError(f"Duplicate customer snapshots in {split}.")

            # Define the regression dataset output path.
            output_path = destination / f"{split}.csv"

            # Prevent accidental overwriting of existing regression datasets.
            if output_path.exists():
                raise FileExistsError(f"Dataset already exists: {output_path}")

            # Save the regression dataset without modifying the churn dataset.
            dataset.to_csv(output_path, index=False, mode="x")

            # Log the saved dataset location and sample count.
            logger.info("Saved %s with %s samples.", output_path, f"{len(dataset):,}")

            # Log descriptive statistics for the regression target.
            logger.info("Spending summary:\n%s", dataset["actual_90d_spend"].describe())

        logger.info("Spending dataset preparation completed successfully.")

    except Exception:
        # Record the exception message and complete stack trace.
        logger.exception("Spending dataset preparation failed.")

        # Propagate the exception to indicate execution failure.
        raise


# Execute main only when this script is run directly or as a module.
if __name__ == "__main__":
    main()