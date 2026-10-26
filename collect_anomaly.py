import serial
import csv
import time
from datetime import datetime

PORT = "COM3"
BAUD = 115200
OUTPUT_FILE = "Dataset/real_motor_anomaly.csv"

ser = serial.Serial(PORT, BAUD, timeout=1)

print("ESP32 connected!")
print("Collecting ANOMALY-condition data...")
print("Press Ctrl+C to stop.\n")

time.sleep(2)

with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "timestamp",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "label"
    ])

    count = 0

    try:
        while True:

            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                continue

            values = line.split(",")

            if len(values) != 6:
                continue

            try:
                ax, ay, az, gx, gy, gz = map(int, values)
            except ValueError:
                continue

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]

            writer.writerow([
                timestamp,
                ax,
                ay,
                az,
                gx,
                gy,
                gz,
                "anomaly"
            ])

            file.flush()

            count += 1

            print(
                f"{count:04d} | "
                f"AX={ax:6d} "
                f"AY={ay:6d} "
                f"AZ={az:6d} | "
                f"GX={gx:5d} "
                f"GY={gy:5d} "
                f"GZ={gz:5d}"
            )

    except KeyboardInterrupt:
        print("\n\nData collection stopped.")

    finally:
        ser.close()

print(f"Saved {count} readings to:")
print(OUTPUT_FILE)