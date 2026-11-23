from pathlib import Path
from src.retention_lab.feature_importance import get_feature_importance

# Locate our best saved XGBoost model.
root = Path(__file__).resolve().parents[1]
run_id = "33a3994f37ec4e55a3fcb240e7300582"
model_path = root / "models" / run_id / "xgboost_model.json"

# Rank features and save results for reporting.
importance = get_feature_importance(model_path)

print(importance.head(10).to_string(index=False))

report_path = root / "reports" / "feature_importance.csv"
report_path.parent.mkdir(parents=True, exist_ok=True)
importance.to_csv(report_path, index=False)