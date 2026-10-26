import pandas as pd
import numpy as np

np.random.seed(42)

rows = []

# Generate NORMAL data
for i in range(250):
    rows.append({
        "ax": np.random.normal(0.15, 0.03),
        "ay": np.random.normal(0.10, 0.03),
        "az": np.random.normal(9.81, 0.05),
        "gx": np.random.normal(0.5, 0.15),
        "gy": np.random.normal(0.3, 0.10),
        "gz": np.random.normal(0.2, 0.10),
        "label": "normal"
    })

# Generate ANOMALY data
for i in range(250):
    rows.append({
        "ax": np.random.normal(1.5, 0.30),
        "ay": np.random.normal(1.2, 0.25),
        "az": np.random.normal(11.0, 0.30),
        "gx": np.random.normal(5.0, 0.60),
        "gy": np.random.normal(5.0, 0.60),
        "gz": np.random.normal(4.8, 0.60),
        "label": "anomaly"
    })

# Create dataset
data = pd.DataFrame(rows)

# Shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
data.to_csv("Dataset/development_sensor_data.csv", index=False)

print("Development dataset created!")
print("Rows:", len(data))
print("\nLabel distribution:")
print(data["label"].value_counts())