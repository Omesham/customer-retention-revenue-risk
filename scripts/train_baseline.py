from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.retention_lab.model_tracking import configure_mlflow
from src.retention_lab.modeling import split_customer_data, train_baseline

# Load the labeled customer dataset.
root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data/processed/customer_training_data.csv")

# Separate training and testing customers.
X_train, X_test, y_train, y_test = split_customer_data(df)

# Connect to DagsHub and track this experiment.
configure_mlflow()

with mlflow.start_run(run_name="logistic-regression-threshold-0.3"):
    # Train the model using training customers only.
    model = train_baseline(X_train, y_train)

    # Predict churn labels and probabilities on unseen customers.
    # Calculate churn probabilities and apply our selected threshold.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.3).astype(int)
    # Record model settings and evaluation metrics.
    # Record the model configuration and chosen threshold.
    mlflow.log_params({"model": "logistic_regression", "test_size": 0.2, "threshold": 0.3})
    mlflow.log_metrics({
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    })

    # Save the trained model to DagsHub through MLflow.
    mlflow.sklearn.log_model(model, name="logistic-regression")

    print("Baseline trained and logged to DagsHub successfully.")