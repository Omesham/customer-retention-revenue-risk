import pandas as pd

from src.retention_lab.snapshots import split_scoring_window
from src.retention_lab.snapshot_labels import build_labeled_snapshot


def build_snapshot_datasets(orders):
    # Create 10 training scoring dates, spaced 28 days apart.
    train_dates = pd.date_range("2010-07-05", periods=10, freq="28D")

    # Create the validation and test scoring dates.
    validation_date = pd.Timestamp("2011-06-12")
    test_date = pd.Timestamp("2011-09-10")

    # Confirm that the dataset covers the full 90-day test window.
    if orders["order_date"].max() < test_date + pd.Timedelta(days=90):
        raise ValueError("The dataset does not cover the complete test window.")

    def create_snapshot(scoring_date):
        # Separate past purchases from the next 90 days.
        history, future = split_scoring_window(orders, scoring_date)

        # Calculate features and assign churn labels.
        return build_labeled_snapshot(history, future, scoring_date)

    # Combine all 10 historical training snapshots.
    train = pd.concat([create_snapshot(date) for date in train_dates], ignore_index=True)

    # Build validation and test sets using later scoring dates.
    validation = create_snapshot(validation_date)
    test = create_snapshot(test_date)

    return train, validation, test