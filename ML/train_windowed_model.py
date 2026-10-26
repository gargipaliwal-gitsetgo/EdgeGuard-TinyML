import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load sensor data
data = pd.read_csv("Dataset/development_sensor_data.csv")

sensors = ["ax", "ay", "az", "gx", "gy", "gz"]
WINDOW_SIZE = 10

feature_rows = []

# Create windows
for start in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE):

    window = data.iloc[start:start + WINDOW_SIZE]

    row = {}

    # Extract vibration features
    for sensor in sensors:
        values = window[sensor].values

        row[sensor + "_mean"] = np.mean(values)
        row[sensor + "_std"] = np.std(values)
        row[sensor + "_rms"] = np.sqrt(np.mean(values ** 2))
        row[sensor + "_peak"] = np.max(values) - np.min(values)

    row["label"] = window["label"].mode()[0]

    feature_rows.append(row)

# Create feature dataset
features = pd.DataFrame(feature_rows)

# Save windowed features
features.to_csv("Dataset/development_windowed_features.csv", index=False)

print("\nWindowed Feature Dataset")
print("------------------------")
print("Windows:", len(features))

# Features and target
X = features.drop(columns=["label"])
y = features["label"].map({
    "normal": 0,
    "anomaly": 1
})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Decision Tree
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Anomaly"]
))

# Save model
joblib.dump(model, "ML/Models/edgeguard_windowed_model.pkl")

print("Windowed model saved successfully!")