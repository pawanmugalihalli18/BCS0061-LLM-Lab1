import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Create output directories
os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

# Load dataset (Wine Quality - Red)
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

# Features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Pre-processing: Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model: Ridge Regression
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Save trained model
joblib.dump(model, "outputs/model/model.pkl")

# Save metrics
metrics = {
    "MSE": mse,
    "R2": r2
}

with open("outputs/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Print metrics (for logs & CI)
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
