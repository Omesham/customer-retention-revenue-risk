
"""Run duplicate analysis on cleaned retail transactions."""

from pathlib import Path

# Import our existing data-loading and validation functions.
from src.retention_lab.data import load_retail
from src.retention_lab import preprocessing, validation


def main():
    # __file__ is this script's location.
    # parents[1] moves up two levels to the project root.
    root = Path(__file__).resolve().parents[1]

    # Build the path to the original Excel dataset.
    retail_path = root / "data" / "raw" / "online_retail_II.xlsx"

    # Load both Excel sheets into a DataFrame.
    df = load_retail(retail_path)

    # Exclude transactions without identifiable customers.
    df = preprocessing.remove_missing_customer_ids(df)

    # Exclude cancelled transactions and negative quantities.
    df = preprocessing.remove_cancellations(df)

    # Exclude transactions with zero or negative prices.
    df = preprocessing.remove_nonpositive_prices(df)

    # Find potentially duplicate rows without removing them.
    duplicates = validation.check_duplicates(df)

    # Print the number of cleaned transactions.
    print("Cleaned rows:", len(df))

    # Print the number of rows flagged as possible duplicates.
    print("Potential duplicates:", len(duplicates))

    # Display the first 10 possible duplicates for inspection.
    print(duplicates.head(10).to_string())


# Run main() only when this file is executed as a script.
if __name__ == "__main__":
    main()
