import pandas as pd
import numpy as np
import os
import joblib

# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ML",
    "Models",
    "edgeguard_real_model.pkl"
)

# ==============================
# LOAD MODEL
# ==============================

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
encoder = saved_data["encoder"]
feature_names = saved_data["features"]

print("EdgeGuard model loaded successfully!")

# ==============================
# SAMPLE 10-READING WINDOW
# ==============================
# Temporary test data from the collected dataset.
# Later, these values will come directly from ESP32.

sample_data = [
    [3316, -856, 15176, -18, 501, -120],
    [3376, -788, 15220, -4, 508, -136],
    [3352, -744, 15308, -23, 583, -90],
    [3252, -748, 15208, -23, 601, -113],
    [3296, -832, 15364, -44, 579, -109],
    [3260, -748, 15280, -26, 565, -115],
    [3348, -872, 15220, 11, 527, -140],
    [3352, -800, 15228, 16, 533, -163],
    [3332, -804, 15172, -94, 537, -90],
    [3296, -592, 15220, -183, 584, -135]
]

sensor_columns = ["ax", "ay", "az", "gx", "gy", "gz"]

# ==============================
# FEATURE EXTRACTION
# ==============================

features = {}

for i, sensor in enumerate(sensor_columns):

    values = np.array([row[i] for row in sample_data])

    features[sensor + "_mean"] = np.mean(values)
    features[sensor + "_std"] = np.std(values)
    features[sensor + "_rms"] = np.sqrt(np.mean(values ** 2))
    features[sensor + "_peak"] = np.max(np.abs(values))

# Convert to DataFrame
feature_df = pd.DataFrame([features])

# Make sure feature order is EXACTLY the same as training
feature_df = feature_df[feature_names]

# ==============================
# PREDICTION
# ==============================

prediction = model.predict(feature_df)

label = encoder.inverse_transform(prediction)[0]

print("\n===== EdgeGuard Real-Time Prediction =====")

if label == "anomaly":
    print("Machine Status: ANOMALY ⚠️")
else:
    print("Machine Status: NORMAL 🟢")

print("\nPredicted Label:", label)