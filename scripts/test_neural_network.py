from pathlib import Path
import pandas as pd
import mlflow
from src.retention_lab.modeling import split_customer_data, evaluate_model
from src.retention_lab.neural_network import load_churn_net, predict_churn_probabilities
from src.retention_lab.model_tracking import configure_mlflow

# Load the labeled customer dataset.
root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data/processed/customer_training_data.csv")

# Retrieve our untouched test customers.
_, X_test, _, y_test = split_customer_data(df)

# Load the previously trained neural network.
run_id = "5a0e7a0364ac4a36bc984ae35cf413b5"
model, scaler = load_churn_net(root / "models" / run_id)

# Evaluate using the threshold selected from validation data.
probabilities = predict_churn_probabilities(model, scaler, X_test)
metrics = evaluate_model(y_test, probabilities, threshold=0.3)

# Record test results in the original DagsHub run.
configure_mlflow()

with mlflow.start_run(run_id=run_id):
    mlflow.log_metric("test_threshold", 0.3)

    for name, value in metrics.items():
        mlflow.log_metric(f"test_{name}", float(value))
        print(f"{name}: {value:.3f}")