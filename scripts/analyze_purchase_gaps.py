from pathlib import Path

from src.retention_lab.data import load_retail
from src.retention_lab.preprocessing import clean_retail
from src.retention_lab.purchase_gaps import calculate_purchase_gaps


def main():
    # Locate the retail dataset.
    root = Path(__file__).resolve().parents[1]
    path = root / "data/raw/online_retail_II.xlsx"

    # Load and clean the transactions.
    df = clean_retail(load_retail(path))
    # Only use purchases made before our prediction date.
    df = df.loc[df["invoice_date"] < "2011-09-01"].copy()
    # Calculate gaps between purchases.
    gaps = calculate_purchase_gaps(df)

    # Display the purchase-gap percentiles.
    print(gaps.quantile([0.50, 0.75, 0.90, 0.95]))


if __name__ == "__main__":
    main()