import pandas as pd
import os

# Get the main project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to dataset
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "sensor_data.csv")

# Load dataset
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# Save cleaned dataset
OUTPUT_PATH = os.path.join(BASE_DIR, "Dataset", "cleaned_sensor_data.csv")
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")