import dagshub
import mlflow

def configure_mlflow():
    # Connect MLflow to our project's DagsHub repository.
    dagshub.init(repo_owner="Omesham", repo_name="customer-retention-revenue-risk", mlflow=True)

    # Create or select the churn prediction experiment.
    mlflow.set_experiment("customer-churn-prediction")