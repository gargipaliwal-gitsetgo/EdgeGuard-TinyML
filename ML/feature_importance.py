import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load feature dataset
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "features.csv")

df = pd.read_csv(DATA_PATH)

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Convert labels to numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train Decision Tree
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X, y_encoded)

# Get feature importance
importance = model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== Feature Importance =====")
print(importance_df)

# Plot
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("EdgeGuard Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

# Save graph
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "feature_importance.png"
)

plt.savefig(OUTPUT_PATH, dpi=300)

print("\nFeature importance graph saved to:")
print(OUTPUT_PATH)

plt.show()