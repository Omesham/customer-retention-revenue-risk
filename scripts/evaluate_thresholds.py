from pathlib import Path
import pandas as pd
from src.retention_lab.modeling import split_customer_data, split_training_validation, train_baseline, evaluate_thresholds

# Load our labeled customer dataset.
root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data/processed/customer_training_data.csv")

# Keep the test set separate from threshold selection.
X_train, _, y_train, _ = split_customer_data(df)

# Divide the training customers into training and validation groups.
X_fit, X_val, y_fit, y_val = split_training_validation(X_train, y_train)

# Train logistic regression using training customers only.
model = train_baseline(X_fit, y_fit)

# Compare thresholds using validation customers only.
evaluate_thresholds(model, X_val, y_val)