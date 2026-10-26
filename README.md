# EdgeGuard: TinyML-Based Predictive Machine Health Monitoring and Anomaly Detection System

EdgeGuard is an embedded machine-condition monitoring prototype that uses an **ESP32, MPU6050 vibration sensor, and a Machine Learning model** to detect abnormal motor vibration and provide real-time visual and audible alerts.

The system collects vibration data from a 5V DC motor, extracts statistical features from sensor windows, classifies the motor condition using a **Decision Tree**, and displays the result through a **Flask-based real-time dashboard**.

> **Project:** Embedded System Architecture  
> **Institution:** VIT Vellore  
> **Student:** Gargi Paliwal  
> **Branch:** B.Tech Computer Science and Engineering

---

## 🚀 Project Overview

Industrial motors can develop abnormal operating conditions before complete failure. Continuous vibration monitoring can help identify such changes at an early stage.

EdgeGuard demonstrates a low-cost prototype for:

- Real-time vibration monitoring
- Sensor-based condition analysis
- Statistical feature extraction
- Machine Learning-based anomaly detection
- Real-time dashboard visualization
- Physical LED and buzzer alerts

The current prototype focuses on **condition monitoring and anomaly detection**, rather than predicting an exact failure time or Remaining Useful Life (RUL).

---

## 🏗️ System Architecture

```text
        ┌──────────────┐
        │   DC Motor   │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │   MPU6050    │
        │  Vibration   │
        │    Sensor    │
        └──────┬───────┘
               │ I2C
               ▼
        ┌──────────────┐
        │    ESP32     │
        └──────┬───────┘
               │ Serial
               ▼
     ┌──────────────────────┐
     │ Python Feature       │
     │ Extraction Pipeline  │
     └──────────┬───────────┘
                │
                ▼
       ┌─────────────────┐
       │ Decision Tree   │
       │ Classification  │
       └────────┬────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌──────────────┐  ┌───────────────┐
│ Flask        │  │ ESP32 Alert   │
│ Dashboard    │  │ LED + Buzzer  │
└──────────────┘  └───────────────┘
```

---

## 🔧 Hardware

| Component | Purpose |
|---|---|
| ESP32 DevKit V1 | Main microcontroller |
| MPU6050 / GY-521 | Accelerometer and gyroscope |
| 5V DC Gear Motor | Machine/vibration source |
| L298N Motor Driver | Motor control |
| LED | Visual anomaly indication |
| Buzzer | Audible anomaly indication |
| 220Ω Resistor | LED current limiting |
| Breadboard | Circuit assembly |
| Jumper Wires | Connections |
| USB Power | ESP32 power |
| Regulated 5V Supply | Motor power |

### MPU6050 Connections

| MPU6050 | ESP32 |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | D21 / GPIO21 |
| SCL | D22 / GPIO22 |

### L298N Connections

| L298N | ESP32 / Motor |
|---|---|
| IN1 | D18 |
| IN2 | D19 |
| OUT1 | Motor |
| OUT2 | Motor |
| GND | Common GND |
| Motor Supply | External regulated 5V |

### Alert Connections

| Component | ESP32 |
|---|---|
| LED | D23 through 220Ω resistor |
| Buzzer | D25 |
| Ground | Common GND |

---

## 💻 Software Stack

### Embedded

- Arduino IDE
- ESP32
- MPU6050
- I2C communication

### Machine Learning

- Python 3.12
- NumPy
- Pandas
- scikit-learn
- Joblib
- PySerial

### Dashboard

- Flask
- HTML
- CSS
- JavaScript

---

## 🧠 Machine Learning Pipeline

The collected sensor readings contain:

```text
ax, ay, az
gx, gy, gz
```

The data is divided into **non-overlapping windows of 10 samples**.

For each sensor signal, four statistical features are calculated:

- Mean
- Standard deviation
- RMS
- Peak absolute value

With six sensor signals and four statistics per signal:

```text
6 signals × 4 features = 24 features
```

The resulting feature vector is passed to the Decision Tree classifier.

### Model

```text
Algorithm: Decision Tree Classifier
Maximum Depth: 5
Random State: 42
```

The model is trained to classify each vibration window as:

```text
NORMAL
ANOMALY
```

---

## 📊 Dataset

The final real-motor dataset contains:

- **2,154 raw sensor readings**
- **1,063 normal readings**
- **1,091 anomaly readings**
- **215 feature windows**
- **107 normal windows**
- **108 anomaly windows**

The dataset was collected using the physical motor and MPU6050 setup.

### Dataset Files

```text
Dataset/
├── real_motor_data.csv
├── real_motor_normal.csv
├── real_motor_anomaly.csv
└── real_features.csv
```

### Dataset

[EdgeGuard Dataset – Google Drive](https://drive.google.com/drive/folders/1Y9YWLhH9T27eDgzpXyMtqklWizHVSLd-?usp=sharing)

---

## 📈 Model Results

The final dataset was divided using a class-wise 80/20 holdout.

| Dataset | Samples |
|---|---:|
| Training | 171 |
| Testing | 44 |

### Performance

| Metric | Result |
|---|---:|
| Accuracy | **86.36%** |
| Precision | **90.00%** |
| Recall | **81.82%** |
| F1 Score | **85.71%** |

### Confusion Matrix

| Actual / Predicted | Normal | Anomaly |
|---|---:|---:|
| Normal | 20 | 2 |
| Anomaly | 4 | 18 |

The model correctly identified most normal and anomalous vibration windows while maintaining good precision.

> These results are preliminary because the dataset was collected from a single motor and experimental setup.

---

## 📊 Feature Importance

The Decision Tree identified vibration-related features as the most influential features for classification.

The most important feature in the collected dataset was:

```text
ay_rms ≈ 98.83%
```

followed by:

```text
gy_rms ≈ 1.17%
```

The remaining features had comparatively small importance.

This feature importance is **specific to the collected prototype dataset** and should not be interpreted as a universal indicator for all motors.

---

## 🖥️ Real-Time Dashboard

The Flask dashboard provides:

- ESP32 connection status
- Current motor condition
- AX, AY, AZ values
- GX, GY, GZ values
- Decision Tree prediction
- Prediction confidence
- Prediction history
- Live vibration graph
- Feature-window progress
- System information

### Dashboard Flow

```text
ESP32
  ↓
Serial Communication
  ↓
Python Backend
  ↓
Feature Extraction
  ↓
Decision Tree
  ↓
Flask Dashboard
```

---

## 🚨 Physical Alert System

EdgeGuard also provides physical alerts.

### Normal Condition

```text
Dashboard → NORMAL
LED       → OFF
Buzzer    → OFF
```

### Anomaly Condition

```text
Dashboard → ANOMALY
LED       → ON
Buzzer    → ON
```

When the vibration pattern returns to the normal condition, the alerts are turned off.

---

## 📁 Project Structure

```text
EdgeGuard-TinyML/
│
├── Dataset/
│   ├── cleaned_sensor_data.csv
│   ├── development_sensor_data.csv
│   ├── features.csv
│   ├── real_features.csv
│   ├── real_motor_anomaly.csv
│   ├── real_motor_data.csv
│   ├── real_motor_normal.csv
│   ├── sample_vibration.csv
│   ├── sensor_data.csv
│   └── windowed_features.csv
│
├── ML/
│   ├── preprocessing.py
│   ├── feature_pipeline.py
│   ├── windowing.py
│   ├── training.py
│   ├── predict.py
│   ├── live_predict.py
│   ├── generate_dataset.py
│   ├── model_comparison.py
│   ├── confusion_matrix.py
│   ├── feature_importance.py
│   ├── train_windowed_model.py
│   └── models/
│       ├── edgeguard_model.pkl
│       ├── edgeguard_real_model.pkl
│       └── edgeguard_windowed_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── app.py
├── collect_data.py
├── collect_anomaly.py
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/gargipaliwal-gitsetgo/EdgeGuard-TinyML.git
cd EdgeGuard-TinyML
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install required libraries

```bash
pip install numpy pandas matplotlib scikit-learn joblib pyserial flask
```

---

## 🔌 ESP32 Setup

1. Connect the MPU6050 to the ESP32.
2. Connect the L298N motor driver.
3. Connect the motor to the driver.
4. Connect the LED and buzzer.
5. Upload the ESP32 sensor program using Arduino IDE.
6. Open the correct serial port.
7. Use a baud rate of:

```text
115200
```

The ESP32 sends the six sensor values:

```text
ax, ay, az, gx, gy, gz
```

to the Python application through serial communication.

---

## ▶️ Running the Dashboard

After connecting the ESP32:

```bash
python app.py
```

Open the Flask dashboard in a browser:

```text
http://127.0.0.1:5000
```

The application reads the sensor stream, generates feature windows, runs the trained Decision Tree model, and updates the dashboard.

> The Arduino Serial Monitor should not be used simultaneously with the Flask application because both cannot reliably own the same serial connection.

---

## 🔬 Experimental Workflow

```text
1. Run the motor
       ↓
2. MPU6050 collects vibration
       ↓
3. ESP32 sends sensor readings
       ↓
4. Python receives serial data
       ↓
5. Data is divided into windows
       ↓
6. 24 statistical features are extracted
       ↓
7. Decision Tree performs classification
       ↓
8. Dashboard displays result
       ↓
9. ESP32 activates LED/buzzer for anomaly
```

---

## 📌 Limitations

The current prototype has several limitations:

- Dataset was collected from a single motor.
- Dataset size is relatively small.
- Anomaly conditions were experimentally controlled.
- Sensor mounting can affect vibration measurements.
- The current ML inference runs on the Python side rather than directly on the ESP32.
- Formal power, RAM, Flash, and inference-latency measurements were not instrumented.
- The system detects anomalous conditions but does not estimate an exact failure time.

---

## 🔮 Future Improvements

Future development could include:

- Larger multi-motor datasets
- Multiple motor speeds and load conditions
- FFT/frequency-domain features
- Additional sensors
- Direct ML inference on ESP32
- Model quantization
- Model pruning
- Formal RAM/Flash/latency measurements
- Power-consumption analysis
- Wi-Fi-based remote monitoring
- Cloud integration
- Multimodal predictive-maintenance architecture

---

## 🎯 Conclusion

EdgeGuard demonstrates a low-cost embedded condition-monitoring system combining **ESP32, MPU6050 vibration sensing, Machine Learning, and a Flask dashboard**.

Using real motor vibration data, the prototype achieved **86.36% accuracy, 90.00% precision, 81.82% recall, and 85.71% F1-score** on the final holdout test set.

The system also demonstrates real-time classification and physical anomaly alerts through an LED and buzzer.

Although the current implementation is a prototype with a limited single-motor dataset, it provides a foundation for future edge-based predictive-maintenance systems.

---

## 👩‍💻 Author

**Gargi Paliwal**  
B.Tech Computer Science and Engineering  
VIT Vellore

---

## ⭐ Project

**EdgeGuard – TinyML-Based Predictive Machine Health Monitoring and Anomaly Detection System**

Built as part of the **Embedded System Architecture** course.
