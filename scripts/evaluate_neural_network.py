from pathlib import Path
import pandas as pd
import mlflow
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from src.retention_lab.modeling import split_customer_data, split_training_validation
from src.retention_lab.neural_network import load_churn_net, predict_churn_probabilities
from src.retention_lab.model_tracking import configure_mlflow

# Locate the dataset and previously saved neural network.
root = Path(__file__).resolve().parents[1]
run_id = "5a0e7a0364ac4a36bc984ae35cf413b5"
model_dir = root / "models" / run_id

# Load customer data and recreate the original validation split.
df = pd.read_csv(root / "data/processed/customer_training_data.csv")
X_train, _, y_train, _ = split_customer_data(df)
_, X_val, _, y_val = split_training_validation(X_train, y_train)

# Load the saved best-epoch model and scaler without retraining.
model, scaler = load_churn_net(model_dir)

# Predict churn probabilities for validation customers.
probabilities = predict_churn_probabilities(model, scaler, X_val)
auc = roc_auc_score(y_val, probabilities)

# Connect to DagsHub and update the original training run.
configure_mlflow()

with mlflow.start_run(run_id=run_id):
    mlflow.log_metric("val_roc_auc", auc)
    print(f"Validation ROC-AUC: {auc:.3f}")

    # Compare thresholds to determine the best balance of precision and recall.
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(y_val, predictions, zero_division=0)
        recall = recall_score(y_val, predictions, zero_division=0)
        f1 = f1_score(y_val, predictions, zero_division=0)

        # Log each threshold's metrics to the existing MLflow run.
        suffix = int(threshold * 100)
        # Display the evaluation results before sending them to DagsHub.
        print(f"Threshold: {threshold} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

        # Log metrics individually to identify any rejected metric.
        mlflow.log_metric(f"val_precision_t{suffix}", float(precision))
        mlflow.log_metric(f"val_recall_t{suffix}", float(recall))
        mlflow.log_metric(f"val_f1_t{suffix}", float(f1))

        print(f"Threshold: {threshold} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")