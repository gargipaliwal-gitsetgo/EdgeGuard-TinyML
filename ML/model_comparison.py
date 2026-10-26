import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load development dataset
data = pd.read_csv("Dataset/development_sensor_data.csv")

# Features and target
X = data[["ax", "ay", "az", "gx", "gy", "gz"]]
y = data["label"].map({"normal": 0, "anomaly": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42
    )
}

print("\nEdgeGuard Model Comparison")
print("--------------------------")

# Train and evaluate
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"\n{name}")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1-Score : {f1:.2f}")