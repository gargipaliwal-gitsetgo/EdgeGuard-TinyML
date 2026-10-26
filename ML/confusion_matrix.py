import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "Dataset", "features.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

loo = LeaveOneOut()

actual = []
predicted = []

for train_index, test_index in loo.split(X):

    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    model.fit(
        X.iloc[train_index],
        y_encoded[train_index]
    )

    prediction = model.predict(X.iloc[test_index])

    actual.append(y_encoded[test_index][0])
    predicted.append(prediction[0])

# Create confusion matrix
cm = confusion_matrix(actual, predicted)

print("Confusion Matrix:")
print(cm)

# Display matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

display.plot()

plt.title("EdgeGuard Confusion Matrix")
plt.tight_layout()

# Save image
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "confusion_matrix.png"
)

plt.savefig(OUTPUT_PATH, dpi=300)

print("\nConfusion matrix saved to:")
print(OUTPUT_PATH)

plt.show()