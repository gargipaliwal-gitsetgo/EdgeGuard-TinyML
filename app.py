"""
EdgeGuard: TinyML-Based Predictive Machine Health Monitoring
and Anomaly Detection System

Flask Backend + ESP32 Serial Data + Decision Tree ML
+ Physical LED/Buzzer Alert
"""

import os
import sys
import time
import threading
from datetime import datetime

import serial
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, jsonify


# ============================================================
# UTF-8 OUTPUT
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ML",
    "Models",
    "edgeguard_real_model.pkl"
)

PORT = os.environ.get("EDGEGUARD_PORT", "COM3")
BAUD = int(os.environ.get("EDGEGUARD_BAUD", 115200))

WINDOW_SIZE = 10

SENSOR_COLUMNS = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz"
]


# ============================================================
# EXACT 24 FEATURES USED DURING TRAINING
# ============================================================

FEATURE_NAMES = [

    "ax_mean",
    "ax_std",
    "ax_rms",
    "ax_peak",

    "ay_mean",
    "ay_std",
    "ay_rms",
    "ay_peak",

    "az_mean",
    "az_std",
    "az_rms",
    "az_peak",

    "gx_mean",
    "gx_std",
    "gx_rms",
    "gx_peak",

    "gy_mean",
    "gy_std",
    "gy_rms",
    "gy_peak",

    "gz_mean",
    "gz_std",
    "gz_rms",
    "gz_peak"
]


# ============================================================
# SYSTEM INFORMATION
# ============================================================

SYSTEM_INFO = {

    "system_name":
        "EdgeGuard Predictive Machine Health Monitor",

    "sensor":
        "MPU6050 6-DOF IMU",

    "controller":
        "ESP32 DevKit V1",

    "model_name":
        "Decision Tree Classifier",

    "model_dataset":
        "Real Motor Vibration Dataset",

    "training_windows":
        171,

    "test_windows":
        44,

    "test_accuracy":
        "86.36%",

    "test_precision":
        "90.00%",

    "test_recall":
        "81.82%",

    "test_f1":
        "85.71%",

    "baud_rate":
        BAUD,

    "port":
        PORT,

    "window_size":
        WINDOW_SIZE,

    "features_count":
        24,

    "i2c_sda":
        "GPIO 21",

    "i2c_scl":
        "GPIO 22",

    "alarm_led":
        "GPIO 23",

    "alarm_buzzer":
        "GPIO 25",

    "motor_driver":
        "L298N",

    "motor_in1":
        "GPIO 18",

    "motor_in2":
        "GPIO 19"
}


# ============================================================
# LOAD ML MODEL
# ============================================================

print("==================================================")
print("   EdgeGuard TinyML Machine Health Monitoring")
print("==================================================")

print()
print("Loading trained model...")
print(f"Model path: {MODEL_PATH}")

try:

    # The saved file is directly a DecisionTreeClassifier
    ml_model = joblib.load(MODEL_PATH)

    print()
    print("[OK] Model loaded successfully!")
    print(f"[OK] Model type: {type(ml_model).__name__}")
    print(f"[OK] Features: {len(FEATURE_NAMES)}")

    if hasattr(ml_model, "classes_"):
        print(
            f"[OK] Classes: "
            f"{list(ml_model.classes_)}"
        )

except Exception as error:

    print()
    print("[ERROR] Could not load model!")
    print(error)

    ml_model = None


# ============================================================
# SHARED STATE
# ============================================================

state_lock = threading.Lock()

state = {

    "connected":
        False,

    "connection_status":
        "ESP32 Disconnected",

    "connection_detail":
        f"Attempting connection to {PORT} @ {BAUD} baud...",

    "port":
        PORT,

    "baud":
        BAUD,

    "samples_received":
        0,

    "last_packet_time":
        None,

    "sensor": {

        "ax": None,
        "ay": None,
        "az": None,

        "gx": None,
        "gy": None,
        "gz": None,

        "accel_mag": None,
        "gyro_mag": None,

        "timestamp":
            "--"
    },

    "prediction": {

        "status":
            "WAITING",

        "label":
            "Waiting for prediction",

        "confidence":
            None,

        "model_name":
            "Decision Tree Classifier",

        "window_size":
            WINDOW_SIZE,

        "window_progress":
            0,

        "timestamp":
            "--"
    },

    "vibration_points":
        [],

    "history":
        []
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_window_features(window_readings):

    """
    Convert 10 raw sensor readings into 24 ML features.

    For each sensor:

        mean
        standard deviation
        RMS
        peak

    6 sensors × 4 features = 24 features
    """

    features = {}

    for i, sensor in enumerate(SENSOR_COLUMNS):

        values = np.array(
            [
                row[i]
                for row in window_readings
            ],
            dtype=float
        )

        features[f"{sensor}_mean"] = float(
            np.mean(values)
        )

        features[f"{sensor}_std"] = float(
            np.std(values)
        )

        features[f"{sensor}_rms"] = float(
            np.sqrt(np.mean(values ** 2))
        )

        features[f"{sensor}_peak"] = float(
            np.max(np.abs(values))
        )

    feature_df = pd.DataFrame([features])

    # IMPORTANT:
    # Keep exactly the same order used during training.

    feature_df = feature_df[FEATURE_NAMES]

    return feature_df


# ============================================================
# SERIAL WORKER
# ============================================================

def serial_worker():

    """
    Main EdgeGuard processing loop.

    ESP32
       ↓
    6 sensor values
       ↓
    10-reading window
       ↓
    24 features
       ↓
    Decision Tree
       ↓
    NORMAL / ANOMALY
       ↓
    ESP32 LED + Buzzer
    """

    buffer_window = []

    while True:

        ser = None

        try:

            # =================================================
            # CONNECTION STATUS
            # =================================================

            with state_lock:

                state["connected"] = False

                state["connection_status"] = \
                    "ESP32 Disconnected"

                state["connection_detail"] = \
                    f"Connecting to {PORT} at {BAUD} baud..."


            print()
            print(
                f"[EdgeGuard] Connecting to "
                f"{PORT} @ {BAUD}..."
            )


            # =================================================
            # OPEN SERIAL PORT
            # =================================================

            ser = serial.Serial(
                PORT,
                BAUD,
                timeout=1.0
            )

            time.sleep(1.5)

            ser.reset_input_buffer()


            # =================================================
            # CONNECTION SUCCESS
            # =================================================

            with state_lock:

                state["connected"] = True

                state["connection_status"] = \
                    "ESP32 Connected"

                state["connection_detail"] = \
                    f"Connected to {PORT} @ {BAUD} baud"


            print(
                f"[EdgeGuard] ESP32 connected "
                f"on {PORT} @ {BAUD} baud"
            )


            # Clear any old incomplete window

            buffer_window.clear()

            with state_lock:

                state[
                    "prediction"
                ][
                    "window_progress"
                ] = 0


            # =================================================
            # CONTINUOUS SERIAL READING
            # =================================================

            while True:

                raw_bytes = ser.readline()

                if not raw_bytes:
                    continue


                # -------------------------------------------------
                # Decode serial data
                # -------------------------------------------------

                line = raw_bytes.decode(
                    "utf-8",
                    errors="ignore"
                ).strip()


                if not line:
                    continue


                # -------------------------------------------------
                # Ignore header
                # -------------------------------------------------

                if line.startswith("ax"):
                    continue


                # -------------------------------------------------
                # Ignore ESP32 messages
                # -------------------------------------------------

                if "ESP32" in line:
                    continue

                if "Boot" in line:
                    continue

                if "Started" in line:
                    continue


                # -------------------------------------------------
                # Split CSV
                # -------------------------------------------------

                parts = line.split(",")


                if len(parts) != 6:
                    continue


                # -------------------------------------------------
                # Convert to numbers
                # -------------------------------------------------

                try:

                    ax, ay, az, gx, gy, gz = [
                        float(value.strip())
                        for value in parts
                    ]

                except ValueError:

                    continue


                # -------------------------------------------------
                # Reject invalid -1 readings
                # -------------------------------------------------

                if all(
                    value == -1
                    for value in [
                        ax,
                        ay,
                        az,
                        gx,
                        gy,
                        gz
                    ]
                ):

                    continue


                # =================================================
                # CURRENT TIME
                # =================================================

                now = datetime.now()

                time_str = now.strftime(
                    "%H:%M:%S"
                )


                # =================================================
                # SENSOR MAGNITUDES
                # =================================================

                accel_mag = round(
                    float(
                        np.sqrt(
                            ax ** 2 +
                            ay ** 2 +
                            az ** 2
                        )
                    ),
                    1
                )


                gyro_mag = round(
                    float(
                        np.sqrt(
                            gx ** 2 +
                            gy ** 2 +
                            gz ** 2
                        )
                    ),
                    1
                )


                # =================================================
                # UPDATE SENSOR DATA
                # =================================================

                with state_lock:

                    state["sensor"] = {

                        "ax":
                            int(round(ax)),

                        "ay":
                            int(round(ay)),

                        "az":
                            int(round(az)),

                        "gx":
                            int(round(gx)),

                        "gy":
                            int(round(gy)),

                        "gz":
                            int(round(gz)),

                        "accel_mag":
                            accel_mag,

                        "gyro_mag":
                            gyro_mag,

                        "timestamp":
                            time_str
                    }


                    state["samples_received"] += 1

                    state["last_packet_time"] = \
                        time.time()


                    # =================================================
                    # VIBRATION GRAPH DATA
                    # =================================================

                    state[
                        "vibration_points"
                    ].append({

                        "t":
                            time_str,

                        "ax":
                            int(round(ax)),

                        "ay":
                            int(round(ay)),

                        "az":
                            int(round(az))
                    })


                    # Keep only latest 40 points

                    if len(
                        state["vibration_points"]
                    ) > 40:

                        state[
                            "vibration_points"
                        ].pop(0)


                # =================================================
                # ADD READING TO ML WINDOW
                # =================================================

                buffer_window.append(
                    [
                        ax,
                        ay,
                        az,
                        gx,
                        gy,
                        gz
                    ]
                )


                # =================================================
                # UPDATE WINDOW PROGRESS
                # =================================================

                with state_lock:

                    state[
                        "prediction"
                    ][
                        "window_progress"
                    ] = len(buffer_window)


                # =================================================
                # RUN PREDICTION AFTER 10 READINGS
                # =================================================

                if (
                    len(buffer_window)
                    == WINDOW_SIZE
                    and
                    ml_model is not None
                ):

                    try:

                        # -----------------------------------------
                        # FEATURE EXTRACTION
                        # -----------------------------------------

                        feature_df = \
                            extract_window_features(
                                buffer_window
                            )


                        # -----------------------------------------
                        # DECISION TREE PREDICTION
                        # -----------------------------------------

                        raw_prediction = \
                            ml_model.predict(
                                feature_df
                            )


                        predicted_label = \
                            raw_prediction[0]


                        # -----------------------------------------
                        # CONFIDENCE
                        # -----------------------------------------

                        confidence = None


                        if hasattr(
                            ml_model,
                            "predict_proba"
                        ):

                            probabilities = \
                                ml_model.predict_proba(
                                    feature_df
                                )[0]


                            predicted_class = \
                                raw_prediction[0]


                            classes = list(
                                ml_model.classes_
                            )


                            class_index = \
                                classes.index(
                                    predicted_class
                                )


                            confidence = round(
                                float(
                                    probabilities[
                                        class_index
                                    ]
                                ) * 100,
                                1
                            )


                        # =================================================
                        # DETERMINE STATUS
                        # =================================================

                        if str(
                            predicted_label
                        ).lower() == "anomaly":

                            status_text = "ANOMALY"

                        else:

                            status_text = "NORMAL"


                        # =================================================
                        # SEND ALERT COMMAND TO ESP32
                        # =================================================

                        try:

                            if status_text == "ANOMALY":

                                # Turn LED + buzzer ON

                                ser.write(
                                    b"ANOMALY\n"
                                )

                                print()
                                print(
                                    "[EdgeGuard] "
                                    "Prediction: ANOMALY"
                                )

                                print(
                                    "[EdgeGuard] "
                                    "Alert sent -> "
                                    "LED ON + BUZZER ON"
                                )


                            else:

                                # Turn LED + buzzer OFF

                                ser.write(
                                    b"NORMAL\n"
                                )

                                print()
                                print(
                                    "[EdgeGuard] "
                                    "Prediction: NORMAL"
                                )

                                print(
                                    "[EdgeGuard] "
                                    "Normal signal sent -> "
                                    "LED OFF + BUZZER OFF"
                                )


                        except Exception as alert_error:

                            print(
                                "[EdgeGuard] "
                                "Alert command error: "
                                f"{alert_error}"
                            )


                        # =================================================
                        # UPDATE DASHBOARD PREDICTION
                        # =================================================

                        with state_lock:

                            state[
                                "prediction"
                            ][
                                "status"
                            ] = status_text


                            state[
                                "prediction"
                            ][
                                "label"
                            ] = str(
                                predicted_label
                            )


                            state[
                                "prediction"
                            ][
                                "confidence"
                            ] = confidence


                            state[
                                "prediction"
                            ][
                                "timestamp"
                            ] = time_str


                            state[
                                "prediction"
                            ][
                                "window_progress"
                            ] = WINDOW_SIZE


                            # =================================================
                            # ADD TO HISTORY
                            # =================================================

                            history_entry = {

                                "timestamp":
                                    time_str,

                                "status":
                                    status_text,

                                "prediction":
                                    status_text,

                                "confidence":
                                    (
                                        f"{confidence}%"
                                        if confidence
                                        is not None
                                        else "--"
                                    )
                            }


                            state[
                                "history"
                            ].insert(
                                0,
                                history_entry
                            )


                            # Keep last 10 predictions

                            if len(
                                state["history"]
                            ) > 10:

                                state[
                                    "history"
                                ].pop()


                        # =================================================
                        # CLEAR WINDOW
                        # =================================================

                        # Training used non-overlapping
                        # 10-reading windows.

                        buffer_window.clear()


                        with state_lock:

                            state[
                                "prediction"
                            ][
                                "window_progress"
                            ] = 0


                    except Exception as inference_error:

                        print()
                        print(
                            "[EdgeGuard] "
                            f"Inference error: "
                            f"{inference_error}"
                        )


                        with state_lock:

                            state[
                                "prediction"
                            ][
                                "status"
                            ] = "WAITING"


                            state[
                                "prediction"
                            ][
                                "label"
                            ] = \
                                "Waiting for prediction"


                            state[
                                "prediction"
                            ][
                                "confidence"
                            ] = None


                            state[
                                "prediction"
                            ][
                                "window_progress"
                            ] = 0


                        buffer_window.clear()


        # =========================================================
        # SERIAL CONNECTION ERROR
        # =========================================================

        except (
            serial.SerialException,
            OSError,
            FileNotFoundError
        ) as port_error:

            print()
            print(
                "[EdgeGuard] Serial error: "
                f"{port_error}"
            )


            with state_lock:

                state["connected"] = False

                state[
                    "connection_status"
                ] = "ESP32 Disconnected"

                state[
                    "connection_detail"
                ] = \
                    f"Port {PORT} unavailable. Retrying..."


                state[
                    "prediction"
                ][
                    "status"
                ] = "WAITING"


                state[
                    "prediction"
                ][
                    "label"
                ] = "Waiting for ESP32"


                state[
                    "prediction"
                ][
                    "window_progress"
                ] = 0


            if ser is not None:

                try:

                    if ser.is_open:
                        ser.close()

                except Exception:

                    pass


            buffer_window.clear()

            time.sleep(2)


        # =========================================================
        # UNEXPECTED ERROR
        # =========================================================

        except Exception as unexpected_error:

            print()
            print(
                "[EdgeGuard] Unexpected error: "
                f"{unexpected_error}"
            )


            buffer_window.clear()

            time.sleep(2)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# LIVE DATA API
# ============================================================

@app.route("/api/data")
def api_data():

    with state_lock:

        response_data = {

            "connected":
                state["connected"],

            "connection_status":
                state["connection_status"],

            "connection_detail":
                state["connection_detail"],

            "samples_received":
                state["samples_received"],

            "sensor":
                state["sensor"],

            "prediction":
                state["prediction"],

            "vibration_points":
                state["vibration_points"],

            "history":
                state["history"],

            "system_info":
                SYSTEM_INFO,


            # =================================================
            # BACKWARD COMPATIBILITY
            # =================================================

            "data": {

                "ax":
                    state["sensor"]["ax"],

                "ay":
                    state["sensor"]["ay"],

                "az":
                    state["sensor"]["az"],

                "gx":
                    state["sensor"]["gx"],

                "gy":
                    state["sensor"]["gy"],

                "gz":
                    state["sensor"]["gz"],

                "status":
                    state["prediction"]["status"],

                "prediction":
                    state["prediction"]["label"],

                "time":
                    state["sensor"]["timestamp"]
            }
        }


    return jsonify(
        response_data
    )


# ============================================================
# START BACKGROUND SERIAL THREAD
# ============================================================

serial_thread = threading.Thread(
    target=serial_worker,
    daemon=True
)

serial_thread.start()


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print()
    print("==================================================")
    print("       EDGEGUARD DASHBOARD STARTING")
    print("==================================================")

    print()
    print(f" Serial Port : {PORT}")
    print(f" Baud Rate   : {BAUD}")
    print(f" ML Model    : {MODEL_PATH}")
    print(f" Window      : {WINDOW_SIZE} readings")
    print(f" Features    : {len(FEATURE_NAMES)}")

    print()
    print(" Hardware Alerts:")
    print(" LED         : GPIO 23")
    print(" Buzzer      : GPIO 25")

    print()
    print(" Dashboard:")
    print(" http://127.0.0.1:5000")

    print()
    print("==================================================")
    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )