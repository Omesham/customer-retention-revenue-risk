from pathlib import Path
import pandas as pd
from src.retention_lab.modeling import split_customer_data, split_training_validation
from src.retention_lab.model_tracking import configure_mlflow
from src.retention_lab.xgboost_model import train_xgboost

# Load our latest labeled customer dataset.
root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data/processed/customer_training_data.csv")

# Keep the test set untouched while training and selecting a threshold.
X_train, _, y_train, _ = split_customer_data(df)
X_fit, X_val, y_fit, y_val = split_training_validation(X_train, y_train)

# Train, evaluate, and save the model to DagsHub.
configure_mlflow()
train_xgboost(X_fit, y_fit, X_val, y_val)