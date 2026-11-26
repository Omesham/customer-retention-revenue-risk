from pathlib import Path

from src.retention_lab.data import load_retail
from src.retention_lab.preprocessing import clean_retail
from src.retention_lab.features import build_customer_features
from src.retention_lab.features import select_active_customers
from src.retention_lab.labels import build_churn_labels
from src.retention_lab.validation import validate_training_data

def main():
    # Locate the project root.
    root = Path(__file__).resolve().parents[1]

    # Load and clean the retail dataset.
    df = clean_retail(load_retail(root / "data/raw/online_retail_II.xlsx"))

    # Calculate features using purchases before September 1, 2011.
    features = build_customer_features(df, "2011-09-01")
    # Select customers active on September 1, 2011.
    active_customers = select_active_customers(features)
    # Create churn labels using purchases in the next 90 days.
    labels = build_churn_labels(active_customers, df)
    # Match active customers with their churn labels.
    training_data = active_customers.merge(labels, on="customer_id", how="inner")

    # Save the labeled dataset for model training.
    training_data.to_csv("data/processed/customer_training_data.csv", index=False)

    # Display the first five rows.
    print(training_data.head())

    # Count customers who returned versus those who became inactive.
    print(labels["churn"].value_counts())

    # Display how many customers qualify.
    print(f"Active customers: {len(active_customers):,}")

    # Create the output directory if needed.
    output = root / "data/processed/customer_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save the customer features.
    features.to_csv(output, index=False)

    # Display the results.
    print("Customers:", len(features))
    print(features.head().to_string(index=False))

    
    # Validate the dataset before model training.
    validate_training_data(training_data)


if __name__ == "__main__":
    main()