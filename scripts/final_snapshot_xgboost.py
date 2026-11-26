
import json
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from src.retention_lab.model_tracking import configure_mlflow


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data/processed/snapshots"

    # Load our chronological datasets.
    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")

    # Recover the configuration selected using validation ROC-AUC.
    best_run_id = "f610c27300a4411997c9492129a6a413"
    best_dir = root / "models" / best_run_id
    config = json.loads((best_dir / "config.json").read_text(encoding="utf-8"))
    features = config["feature_names"]
    threshold = float(config["threshold"])

    # Recover the number of trees selected by early stopping.
    selected_model = XGBClassifier()
    selected_model.load_model(str(best_dir / "xgboost_model.json"))
    best_rounds = selected_model.get_booster().num_boosted_rounds()

    # Refit using both training and validation snapshots.
    X_fit = pd.concat([train[features], validation[features]], ignore_index=True)
    y_fit = pd.concat([train["churn"], validation["churn"]], ignore_index=True)

    # Keep the final chronological test snapshot separate.
    X_test = test[features]
    y_test = test["churn"]

    # Train with the selected hyperparameters and number of trees.
    params = config["params"].copy()
    params["n_estimators"] = best_rounds
    model = XGBClassifier(**params)
    model.fit(X_fit, y_fit, verbose=False)

    # Evaluate the final model without tuning on test results.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0)
    }

    print(f"Training + validation rows: {len(X_fit):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Selected boosting rounds: {best_rounds}")
    print(f"Selected threshold: {threshold}")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Save the final model and test metrics as a new MLflow experiment.
    configure_mlflow()

    with mlflow.start_run(run_name="xgboost-24-features-temporal-final"):
        mlflow.log_params(params)
        mlflow.log_param("classification_threshold", threshold)
        mlflow.log_param("selected_from_run", best_run_id)
        mlflow.log_metrics({f"test_{name}": float(value) for name, value in metrics.items()})

        model_dir = root / "models" / mlflow.active_run().info.run_id
        model_dir.mkdir(parents=True, exist_ok=True)

        model.save_model(str(model_dir / "xgboost_model.json"))
        final_config = {"feature_names": features, "threshold": threshold, "params": params}
        (model_dir / "config.json").write_text(json.dumps(final_config, indent=2), encoding="utf-8")

        # Store an inference-ready model and its input schema.
        example = X_fit.iloc[:5].astype("float64")
        signature = infer_signature(example, model.predict(example))
        mlflow.xgboost.log_model(model, name="xgboost-final", input_example=example, signature=signature)
        mlflow.log_artifacts(str(model_dir), artifact_path="checkpoint")


if __name__ == "__main__":
    main()
