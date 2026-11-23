import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler
import copy
import joblib
from pathlib import Path
import json

class ChurnNet(nn.Module):
    def __init__(self, input_dim=3, hidden1=16, hidden2=8, dropout=0.0):
        super().__init__()

        # Allow experiments with different network sizes and dropout rates.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        # Generate a raw churn prediction score.
        return self.network(x)


def train_churn_net(X_train, y_train, X_val, y_val, epochs=300, learning_rate=0.001, weight_decay=0.0, hidden1=16, hidden2=8, dropout=0.0):
    # Scale features using training data only.
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_val = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

    # Initialize the model, loss function, and optimizer.
    # Create the neural network using the selected hyperparameters.
    # Automatically match the network's input size to the number of features.
    model = ChurnNet(input_dim=X_train.shape[1], hidden1=hidden1, hidden2=hidden2, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss()
    # Make learning rate and L2 regularization configurable for tuning.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    val_losses = []

    # Track the best model and stop after 20 epochs without improvement.
    best_val_loss = float("inf")
    best_weights = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        # Update weights using training data.
        model.train()
        optimizer.zero_grad()
        train_loss = criterion(model(X_train).squeeze(1), y_train)
        train_loss.backward()
        optimizer.step()

        # Evaluate on validation data without updating weights.
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val).squeeze(1), y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Save weights whenever validation loss improves.
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Stop when validation loss hasn't improved for 20 epochs.
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Restore the weights with the lowest validation loss.
    model.load_state_dict(best_weights)
    print(f"Best validation loss: {best_val_loss:.4f}")

    return model, scaler, train_losses, val_losses


def predict_churn_probabilities(model, scaler, X):
    # Apply the same scaling learned during training.
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Switch to evaluation mode.
    model.eval()

    # Generate predictions without updating model weights.
    with torch.no_grad():
        logits = model(X_tensor).squeeze(1)
        probabilities = torch.sigmoid(logits)

    return probabilities.numpy()


def load_churn_net(model_dir):
    # Load the saved model weights.
    model_dir = Path(model_dir)
    weights = torch.load(model_dir / "churn_net.pt", map_location="cpu", weights_only=True)

    # Support checkpoints saved before dropout layers were introduced.
    if "network.2.weight" in weights:
        weights = {key.replace("network.2.", "network.3.").replace("network.4.", "network.6."): value for key, value in weights.items()}

    # Read saved hyperparameters when available.
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    # Reconstruct the architecture from the saved weight dimensions.
    input_dim = weights["network.0.weight"].shape[1]
    hidden1 = weights["network.0.weight"].shape[0]
    hidden2 = weights["network.3.weight"].shape[0]

    model = ChurnNet(input_dim=input_dim, hidden1=hidden1, hidden2=hidden2, dropout=config.get("dropout", 0.0))
    model.load_state_dict(weights)

    # Restore preprocessing and verify the feature count.
    scaler = joblib.load(model_dir / "scaler.joblib")
    if scaler.n_features_in_ != input_dim:
        raise ValueError("Model and scaler feature counts do not match.")

    model.eval()
    return model, scaler