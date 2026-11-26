from pathlib import Path

from src.retention_lab.data import load_retail
from src.retention_lab.preprocessing import clean_retail
from src.retention_lab.snapshot_orders import build_orders
from src.retention_lab.snapshot_dataset import build_snapshot_datasets


def main():    
    # Load the Online Retail II workbook from the project's raw data folder.
    root = Path(__file__).resolve().parents[1]
    df = clean_retail(load_retail(root / "data" / "raw" / "online_retail_II.xlsx"))
    
    # Convert invoice items into customer orders.
    orders = build_orders(df)

    # Generate the chronological training, validation, and test datasets.
    train, validation, test = build_snapshot_datasets(orders)

    # Save datasets separately for modeling.
    output_dir = Path("data/processed/snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, dataset in [("train", train), ("validation", validation), ("test", test)]:
        dataset.to_csv(output_dir / f"{name}.csv", index=False)

        # Check that customer/date combinations are unique.
        duplicates = dataset.duplicated(["customer_id", "scoring_date"]).sum()

        print(f"{name}: {len(dataset):,} rows")
        print(f"Churn rate: {dataset['churn'].mean():.3f}")
        print(f"Duplicate customer/date rows: {duplicates}")


if __name__ == "__main__":
    main()