import pandas as pd
from pathlib import Path
from src.retention_lab.eda import plot_correlation_heatmap
from src.retention_lab.eda import plot_recency_by_churn, plot_frequency_by_churn, plot_monetary_by_churn, plot_churn_rate_by_frequency

# Locate our saved training dataset.
root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data/processed/customer_training_data.csv")

# Display the correlation heatmap.
plot_correlation_heatmap(df)

# Compare customer recency across churn groups.
plot_recency_by_churn(df)

# Visualize purchase frequency by churn status.
plot_frequency_by_churn(df)

# Compare customer spending by churn status.
plot_monetary_by_churn(df)

# Visualize churn percentage across purchase-frequency groups.
plot_churn_rate_by_frequency(df)

# Group customers by their purchase frequency.
groups = pd.cut(df["frequency"], bins=[0, 2, 5, 10, float("inf")], labels=["1–2", "3–5", "6–10", "11+"])

# Count customers in each group.
print(df.groupby(groups, observed=True).size())