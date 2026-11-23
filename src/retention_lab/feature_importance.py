import pandas as pd
from xgboost import XGBClassifier


def get_feature_importance(model_path):
    # Load the existing trained model.
    model = XGBClassifier()
    model.load_model(str(model_path))

    # Measure each feature's total contribution to improving tree splits.
    booster = model.get_booster()
    scores = booster.get_score(importance_type="total_gain")
    features = booster.feature_names

    importance = pd.DataFrame({
        "feature": features,
        "total_gain": [scores.get(feature, 0) for feature in features]
    })

    # Express importance as a percentage of total gain.
    importance["importance_pct"] = (
        importance["total_gain"] / importance["total_gain"].sum() * 100
    )

    return importance.sort_values("total_gain", ascending=False)