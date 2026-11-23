
from src.retention_lab.xgboost_model import train_xgboost


def tune_xgboost(X_fit, y_fit, X_val, y_val):
    # Change one hyperparameter at a time to measure its effect.
    trials = [
        ("depth-2", {"max_depth": 2}),
        ("depth-4", {"max_depth": 4}),
        ("min-child-10", {"min_child_weight": 10}),
        ("lambda-10", {"reg_lambda": 10}),
        ("learning-rate-005", {"learning_rate": 0.05}),
        ("full-sampling", {"subsample": 1.0, "colsample_bytree": 1.0})
    ]

    best_auc = 0
    best_trial = None

    for name, params in trials:
        # Train, validate, and save each experiment to DagsHub.
        model, auc = train_xgboost(X_fit, y_fit, X_val, y_val, params_override=params, run_name=f"xgboost-{name}")

        if auc > best_auc:
            best_auc = auc
            best_trial = name

    print(f"Best tuning trial: {best_trial}")
    print(f"Best validation ROC-AUC: {best_auc:.3f}")
