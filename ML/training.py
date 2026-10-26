import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib


# ==========================================
# 1. LOAD FEATURE DATA
# ==========================================

data = pd.read_csv("Dataset/real_features.csv")

X = data.drop("label", axis=1)
y = data["label"]


# ==========================================
# 2. SESSION-AWARE TRAIN / TEST SPLIT
# ==========================================

# First 80% of each class -> training
# Last 20% of each class -> testing

normal = data[data["label"] == "normal"].reset_index(drop=True)
anomaly = data[data["label"] == "anomaly"].reset_index(drop=True)

normal_split = int(len(normal) * 0.8)
anomaly_split = int(len(anomaly) * 0.8)

normal_train = normal.iloc[:normal_split]
normal_test = normal.iloc[normal_split:]

anomaly_train = anomaly.iloc[:anomaly_split]
anomaly_test = anomaly.iloc[anomaly_split:]

train_data = pd.concat(
    [normal_train, anomaly_train],
    ignore_index=True
)

test_data = pd.concat(
    [normal_test, anomaly_test],
    ignore_index=True
)

X_train = train_data.drop("label", axis=1)
y_train = train_data["label"]

X_test = test_data.drop("label", axis=1)
y_test = test_data["label"]


print("===== EDGEGUARD REAL MOTOR MODEL =====")

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())


# ==========================================
# 3. TRAIN DECISION TREE
# ==========================================

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 4. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 5. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="anomaly",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="anomaly",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="anomaly",
    zero_division=0
)

print("\n===== MODEL PERFORMANCE =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ==========================================
# 6. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["normal", "anomaly"]
)

print("\n===== CONFUSION MATRIX =====")
print("              Predicted")
print("              Normal  Anomaly")
print(f"Actual Normal   {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"Actual Anomaly  {cm[1][0]:4d}    {cm[1][1]:4d}")


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))


# ==========================================
# 8. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\n===== TOP FEATURES =====")
print(importance.head(10))


# ==========================================
# 9. SAVE MODEL
# ==========================================

model_path = "ML/Models/edgeguard_real_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)