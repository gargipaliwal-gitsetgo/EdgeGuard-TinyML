import pandas as pd
import numpy as np

# Load sensor data
data = pd.read_csv("Dataset/sample_vibration.csv")

sensors = ["ax", "ay", "az", "gx", "gy", "gz"]

WINDOW_SIZE = 5

feature_rows = []

# Create windows
for start in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE):

    window = data.iloc[start:start + WINDOW_SIZE]

    row = {}

    for sensor in sensors:
        values = window[sensor].values

        row[sensor + "_mean"] = np.mean(values)
        row[sensor + "_std"] = np.std(values)
        row[sensor + "_rms"] = np.sqrt(np.mean(values ** 2))
        row[sensor + "_peak"] = np.max(values) - np.min(values)

    # Label of the window
    row["label"] = window["label"].mode()[0]

    feature_rows.append(row)

# Create dataset
features = pd.DataFrame(feature_rows)

# Save
features.to_csv("Dataset/windowed_features.csv", index=False)

print("\nEdgeGuard Windowed Dataset")
print("--------------------------")
print(features)

print("\nSaved to Dataset/windowed_features.csv")