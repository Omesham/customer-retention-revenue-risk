from pathlib import Path
import pandas as pd
import mlflow
from xgboost import XGBClassifier
from src.retention_lab.modeling import split_customer_data, evaluate_model
from src.retention_lab.model_tracking import configure_mlflow

# Load the dataset and recover the existing test split.
root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data/processed/customer_training_data.csv")
_, X_test, _, y_test = split_customer_data(df)

# Load the best saved XGBoost checkpoint.
run_id = "33a3994f37ec4e55a3fcb240e7300582"
model = XGBClassifier()
model.load_model(str(root / "models" / run_id / "xgboost_model.json"))

# Evaluate at the threshold selected using validation data.
probabilities = model.predict_proba(X_test)[:, 1]
metrics = evaluate_model(y_test, probabilities, threshold=0.2)

# Log evaluation results to the original DagsHub experiment.
configure_mlflow()
with mlflow.start_run(run_id=run_id):
    mlflow.log_metric("test_threshold", 0.2)
    for name, value in metrics.items():
        mlflow.log_metric(f"test_{name}", float(value))
        print(f"{name}: {value:.3f}")