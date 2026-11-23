from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def train_baseline(X_train, y_train):
    # Scale features and train logistic regression.
    # Scaling is learned from training data only.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    return model

def split_customer_data(df):
    # X contains the customer features used to predict churn.
    # X = df[["recency_days", "frequency", "monetary"]]
    # Include the newly engineered customer behavior features.
    # X = df[["recency_days", "frequency", "monetary", "tenure_days", "orders_last_30d", "orders_last_90d", "avg_order_value"]]
    # Include purchasing trends alongside our existing customer features.
    # X = df[["recency_days", "frequency", "monetary", "tenure_days", "orders_last_30d", "orders_last_90d", "avg_order_value", "orders_prev_90d", "order_change_90d"]]
    # Exclude identifiers, dates, and the target from model inputs.
    X = df.drop(columns=["customer_id", "first_purchase", "last_purchase", "churn"])
    # y is our target: 1 = churned, 0 = retained.
    y = df["churn"]

    # Split customers into 80% training and 20% testing.
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def split_training_validation(X_train, y_train):
    # Reserve 20% of training data for threshold evaluation.
    # Keep the same churn proportions in both groups.
    return train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_thresholds(model, X_val, y_val):
    # Get each customer's predicted probability of churning.
    probabilities = model.predict_proba(X_val)[:, 1]
    # Measure logistic regression's performance on the same validation customers.
    print(f"Logistic regression validation ROC-AUC: {roc_auc_score(y_val, probabilities):.3f}")

    # Compare different thresholds against the default of 0.5.
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        # Classify customers as churned when their probability meets the threshold.
        predictions = (probabilities >= threshold).astype(int)

        # Measure precision, recall, and F1 for each threshold.
        precision = precision_score(y_val, predictions, zero_division=0)
        recall = recall_score(y_val, predictions, zero_division=0)
        f1 = f1_score(y_val, predictions, zero_division=0)

        print(f"Threshold: {threshold} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")



def evaluate_model(y_true, probabilities, threshold=0.3):
    # Convert predicted probabilities into churn labels.
    predictions = (probabilities >= threshold).astype(int)

    # Evaluate classification performance.
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0)
    }