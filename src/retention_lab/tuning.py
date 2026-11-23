import json
import joblib
import mlflow
import mlflow.pytorch
import torch
from pathlib import Path
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score
from src.retention_lab.neural_network import train_churn_net, predict_churn_probabilities


def tune_churn_net(X_fit, y_fit, X_val, y_val):
    # Compare the baseline with changes to individual hyperparameters.
    configs = [
        {"hidden1": 16, "hidden2": 8, "dropout": 0.0, "learning_rate": 0.001, "weight_decay": 0.0},
        {"hidden1": 16, "hidden2": 8, "dropout": 0.0, "learning_rate": 0.0005, "weight_decay": 0.0},
        {"hidden1": 16, "hidden2": 8, "dropout": 0.0, "learning_rate": 0.003, "weight_decay": 0.0},
        {"hidden1": 16, "hidden2": 8, "dropout": 0.0, "learning_rate": 0.001, "weight_decay": 0.001},
        {"hidden1": 16, "hidden2": 8, "dropout": 0.15, "learning_rate": 0.001, "weight_decay": 0.0},
        {"hidden1": 32, "hidden2": 16, "dropout": 0.0, "learning_rate": 0.001, "weight_decay": 0.0},
    ]

    root = Path(__file__).resolve().parents[2]

    for number, config in enumerate(configs, start=1):
        # Use the same random seed to make trials reproducible.
        torch.manual_seed(42)

        with mlflow.start_run(run_name=f"churn-tuning-trial-{number}"):
            # Train and restore the best validation-loss checkpoint.
            model, scaler, train_losses, val_losses = train_churn_net(X_fit, y_fit, X_val, y_val, epochs=300, **config)

            # Evaluate the model on validation customers only.
            probabilities = predict_churn_probabilities(model, scaler, X_val)
            auc = roc_auc_score(y_val, probabilities)

            # Record the experiment configuration and results.
            mlflow.log_params({**config, "epochs": 300, "seed": 42})
            mlflow.log_metric("val_roc_auc", float(auc))
            mlflow.log_metric("best_val_loss", min(val_losses))
            mlflow.log_metric("best_epoch", val_losses.index(min(val_losses)) + 1)

            # Save the checkpoint, scaler, and architecture for this trial.
            model_dir = root / "models" / mlflow.active_run().info.run_id
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "churn_net.pt")
            joblib.dump(scaler, model_dir / "scaler.joblib")
            (model_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

            # Use real, standardized training records for the model signature.
            example = scaler.transform(X_fit.iloc[:5]).astype("float32")
            model.eval()
            with torch.no_grad():
                outputs = model(torch.from_numpy(example)).numpy()

            signature = infer_signature(example, outputs)

            # Upload both the MLflow model and its reproducibility artifacts.
            mlflow.pytorch.log_model(model, name="churn-net", input_example=example, signature=signature, serialization_format="pt2")
            mlflow.log_artifacts(str(model_dir), artifact_path="checkpoint")

            print(f"Trial {number}: validation ROC-AUC = {auc:.3f}")