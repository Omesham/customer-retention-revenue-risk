from pathlib import Path
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

from src.retention_lab.modeling import split_customer_data
from src.retention_lab.neural_network import load_churn_net
from src.retention_lab.model_tracking import configure_mlflow

# Identify the existing training run and saved checkpoint.
root = Path(__file__).resolve().parents[1]
run_id = "5a0e7a0364ac4a36bc984ae35cf413b5"
model_dir = root / "models" / run_id

# Load the trained model and its fitted scaler.
model, scaler = load_churn_net(model_dir)

# Use actual training customers, not fabricated input values.
df = pd.read_csv(root / "data/processed/customer_training_data.csv")
X_train, _, _, _ = split_customer_data(df)

# Standardize five real customer records using the saved scaler.
input_example = scaler.transform(X_train.iloc[:5]).astype("float32")

# Generate model outputs to define the input/output signature.
with torch.no_grad():
    outputs = model(torch.from_numpy(input_example)).numpy()

signature = infer_signature(input_example, outputs)

# Reconnect to DagsHub and resume the existing training run.
configure_mlflow()

with mlflow.start_run(run_id=run_id):
    # Log the model with a real input example and explicit signature.
    model_info = mlflow.pytorch.log_model(
        model,
        name="churn-net",
        serialization_format="pt2",
        input_example=input_example,
        signature=signature
    )

    # Store the original weights and preprocessing scaler.
    mlflow.log_artifacts(str(model_dir), artifact_path="checkpoint")

    # Reload the MLflow model and verify that predictions match.
    loaded_model = mlflow.pytorch.load_model(model_info.model_uri)

    with torch.no_grad():
        reloaded_outputs = loaded_model(torch.from_numpy(input_example))

    assert torch.allclose(torch.from_numpy(outputs), reloaded_outputs, atol=1e-5), "Reloaded model predictions differ."

print("Model uploaded and verified successfully.")