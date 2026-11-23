
import json
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping


def train_xgboost(X_fit, y_fit, X_val, y_val, params_override=None, run_name="xgboost-baseline"):
    # Start with regularized, shallow trees to reduce overfitting.
    params = {
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 5,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 4
    }
        # Apply the hyperparameters selected for this experiment.
    if params_override:
        params.update(params_override)
    # Stop after 40 rounds without validation AUC improvement.
    early_stop = EarlyStopping(rounds=40, metric_name="auc", data_name="validation_0", save_best=True)
    model = XGBClassifier(**params, callbacks=[early_stop])
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate ranking performance on validation data only.
    probabilities = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probabilities)
    print(f"Validation ROC-AUC: {auc:.3f}")
    print(f"Best boosting round: {model.best_iteration + 1}")

    # Compare classification thresholds.
    results = []
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_val, predictions, zero_division=0)
        recall = recall_score(y_val, predictions, zero_division=0)
        f1 = f1_score(y_val, predictions, zero_division=0)
        results.append((threshold, precision, recall, f1))
        print(f"Threshold: {threshold} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

    # Select a provisional threshold using validation F1.
    best_threshold, best_precision, best_recall, best_f1 = max(results, key=lambda result: result[3])

    with mlflow.start_run(run_name=run_name):
        # Log the complete experiment configuration.
        mlflow.log_params(params)
        mlflow.log_metric("val_roc_auc", float(auc))
        mlflow.log_metric("val_f1", float(best_f1))
        mlflow.log_metric("val_precision", float(best_precision))
        mlflow.log_metric("val_recall", float(best_recall))
        mlflow.log_metric("best_threshold", float(best_threshold))
        mlflow.log_metric("best_iteration", model.best_iteration + 1)

        # Save the trained model, feature order, and threshold locally.
        root = Path(__file__).resolve().parents[2]
        model_dir = root / "models" / mlflow.active_run().info.run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_dir / "xgboost_model.json"))

        config = {"feature_names": X_fit.columns.tolist(), "threshold": best_threshold, "params": params}
        (model_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        # Log a real input example and the model's input/output signature.
        # Use real customer records with consistent numerical types for deployment.
        example = X_fit.iloc[:5].astype("float64")
        signature = infer_signature(example, model.predict(example))
        model_info = mlflow.xgboost.log_model(model, name="xgboost-model", input_example=example, signature=signature)
        mlflow.log_artifacts(str(model_dir), artifact_path="checkpoint")

        # Verify that the model downloaded from MLflow gives matching predictions.
        restored = mlflow.xgboost.load_model(model_info.model_uri)
        assert np.allclose(model.predict_proba(example), restored.predict_proba(example)), "Saved model predictions differ."

    return model, auc
