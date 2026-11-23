import pandas as pd
import mlflow
from src.retention_lab.modeling import split_customer_data, split_training_validation
from src.retention_lab.neural_network import train_churn_net
from src.retention_lab.model_tracking import configure_mlflow
import torch
import joblib
import mlflow.pytorch
from pathlib import Path
import json
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score
from src.retention_lab.neural_network import predict_churn_probabilities
# Load our labeled customer data.
df = pd.read_csv("data/processed/customer_training_data.csv")

# Keep the test set untouched.
X_train, _, y_train, _ = split_customer_data(df)
X_fit, X_val, y_fit, y_val = split_training_validation(X_train, y_train)

# Connect to DagsHub.
configure_mlflow()

with mlflow.start_run(run_name="pytorch-churn-7-features"):
    # Train longer to see whether validation loss continues improving.
    model, scaler, train_losses, val_losses = train_churn_net(X_fit, y_fit, X_val, y_val, epochs=300)

    # Record all hyperparameters needed to reproduce the experiment.
    mlflow.log_params({"model": "ChurnNet", "epochs": 300, "learning_rate": 0.001, "weight_decay": 0.0, "hidden1": 16, "hidden2": 8, "dropout": 0.0, "input_dim": 7})

    # Track losses every 10 epochs.
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
        if epoch % 10 == 0:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)


    # Record the lowest validation loss and the epoch that achieved it.
    mlflow.log_metric("best_val_loss", min(val_losses))
    mlflow.log_metric("best_epoch", val_losses.index(min(val_losses)) + 1)
    
    # Display final losses.
    print(f"Training loss: {train_losses[-1]:.4f}")
    print(f"Validation loss: {val_losses[-1]:.4f}")    

    
    # Save the restored best-epoch model and fitted scaler.
    model_dir = Path("models") / mlflow.active_run().info.run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / "churn_net.pt")
    joblib.dump(scaler, model_dir / "scaler.joblib")

    # Save the model architecture and exact feature order.
    config = {
        "input_dim": X_fit.shape[1],
        "hidden1": 16,
        "hidden2": 8,
        "dropout": 0.0,
        "feature_names": X_fit.columns.tolist()
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Evaluate ROC-AUC using validation customers.
    probabilities = predict_churn_probabilities(model, scaler, X_val)
    auc = roc_auc_score(y_val, probabilities)
    mlflow.log_metric("val_roc_auc", float(auc))

    # Create an MLflow signature using real, scaled training records.
    input_example = scaler.transform(X_fit.iloc[:5]).astype("float32")
    model.eval()

    with torch.no_grad():
        outputs = model(torch.from_numpy(input_example)).numpy()

    signature = infer_signature(input_example, outputs)

    # Save the model and all supporting artifacts to DagsHub.
    mlflow.pytorch.log_model(model, name="churn-net", input_example=input_example, signature=signature, serialization_format="pt2")
    mlflow.log_artifacts(str(model_dir), artifact_path="checkpoint")

    print(f"Validation ROC-AUC: {auc:.3f}")


