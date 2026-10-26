import pandas as pd
import numpy as np

# Load REAL motor dataset
data = pd.read_csv("Dataset/real_motor_data.csv")

sensor_columns = ["ax", "ay", "az", "gx", "gy", "gz"]

WINDOW_SIZE = 10

features = []

# Create non-overlapping windows
for start in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE):

    window = data.iloc[start:start + WINDOW_SIZE]

    row = {}

    # Extract 4 features from each sensor
    for sensor in sensor_columns:

        values = window[sensor].values

        row[f"{sensor}_mean"] = np.mean(values)
        row[f"{sensor}_std"] = np.std(values)
        row[f"{sensor}_rms"] = np.sqrt(np.mean(values ** 2))
        row[f"{sensor}_peak"] = np.max(np.abs(values))

    # Label of the window
    row["label"] = window["label"].iloc[0]

    features.append(row)

# Convert to DataFrame
feature_data = pd.DataFrame(features)

# Save extracted features
feature_data.to_csv("Dataset/real_features.csv", index=False)

print("===== FEATURE EXTRACTION COMPLETE =====")
print("Raw readings:", len(data))
print("Window size:", WINDOW_SIZE)
print("Feature dataset shape:", feature_data.shape)

print("\nLabel distribution:")
print(feature_data["label"].value_counts())

print("\nMissing values:")
print(feature_data.isna().sum().sum())

print("\nFirst 5 rows:")
print(feature_data.head())

print("\nSaved to:")
print("Dataset/real_features.csv")