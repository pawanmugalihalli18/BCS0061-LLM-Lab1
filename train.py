import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create output directories
os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

# Load dataset (Wine Quality - Red)
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

# Features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model: Random Forest Regressor (tuned)
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)

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
