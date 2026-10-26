import pandas as pd

normal = pd.read_csv("Dataset/real_motor_normal.csv")
anomaly = pd.read_csv("Dataset/real_motor_anomaly.csv")

data = pd.concat([normal, anomaly], ignore_index=True)

data.to_csv("Dataset/real_motor_data.csv", index=False)

print("===== REAL MOTOR DATASET =====")
print("Shape:", data.shape)
print("\nLabel distribution:")
print(data["label"].value_counts())
print("\nMissing values:")
print(data.isna().sum())
print("\nSaved to:")
print("Dataset/real_motor_data.csv")