import serial
import time
import numpy as np
import pandas as pd
import os
import joblib

# ==========================================
# ESP32 CONNECTION
# ==========================================

PORT = "COM3"
BAUD_RATE = 115200

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ML",
    "Models",
    "edgeguard_real_model.pkl"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(MODEL_PATH)

print("EdgeGuard model loaded successfully!")

# ==========================================
# CONNECT TO ESP32
# ==========================================

print(f"Connecting to ESP32 on {PORT}...")

ser = serial.Serial(
    PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print("ESP32 connected successfully!")
print("Collecting 10 sensor readings...\n")

# ==========================================
# SENSOR COLUMNS
# ==========================================

sensor_columns = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz"
]

# ==========================================
# FEATURE ORDER
# MUST MATCH TRAINING
# ==========================================

feature_names = [
    "ax_mean", "ax_std", "ax_rms", "ax_peak",
    "ay_mean", "ay_std", "ay_rms", "ay_peak",
    "az_mean", "az_std", "az_rms", "az_peak",
    "gx_mean", "gx_std", "gx_rms", "gx_peak",
    "gy_mean", "gy_std", "gy_rms", "gy_peak",
    "gz_mean", "gz_std", "gz_rms", "gz_peak"
]

# ==========================================
# CONTINUOUS PREDICTION
# ==========================================

try:

    while True:

        readings = []

        print("Collecting window...")

        # ==========================================
        # COLLECT 10 SENSOR READINGS
        # ==========================================

        while len(readings) < 10:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            # Ignore CSV header
            if line.startswith("ax"):
                continue

            values = line.split(",")

            # We need exactly 6 sensor values
            if len(values) != 6:
                continue

            try:

                values = [int(value) for value in values]

                readings.append(values)

            except ValueError:

                continue

        # ==========================================
        # FEATURE EXTRACTION
        # ==========================================

        features = {}

        for i, sensor in enumerate(sensor_columns):

            values = np.array(
                [row[i] for row in readings],
                dtype=float
            )

            features[sensor + "_mean"] = np.mean(values)

            features[sensor + "_std"] = np.std(values)

            features[sensor + "_rms"] = np.sqrt(
                np.mean(values ** 2)
            )

            features[sensor + "_peak"] = np.max(
                np.abs(values)
            )

        # ==========================================
        # CREATE FEATURE DATAFRAME
        # ==========================================

        feature_df = pd.DataFrame([features])

        # Make absolutely sure feature order
        # is identical to training
        feature_df = feature_df[feature_names]

        # ==========================================
        # PREDICTION
        # ==========================================

        prediction = model.predict(feature_df)

        label = prediction[0]
        print("\n===== LIVE FEATURES =====")

        for column in feature_names:
            print(f"{column}: {feature_df[column].iloc[0]:.2f}")

        print("\nModel prediction:", label)

        print("\n==============================")

        if label == "anomaly":

            print("Machine Status: ANOMALY ⚠️")

        else:

            print("Machine Status: NORMAL 🟢")

        print("==============================\n")


except KeyboardInterrupt:

    print("\nStopping EdgeGuard...")

finally:

    ser.close()