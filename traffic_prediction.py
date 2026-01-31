import h5py
import numpy as np
import pandas as pd
import time
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

DATA_PATH = "metr-la.h5"
RESULTS_DIR = "results"
THRESHOLD_SPEED = 40

print("Traffic congestion prediction using energy-efficient models")

print("Loading METR-LA dataset")

try:
    with h5py.File(DATA_PATH, "r") as f:
        data = f["df"]["block0_values"][:]
except FileNotFoundError:
    print(f"Dataset file '{DATA_PATH}' not found")
    print("Download metr-la.h5 and place it in this directory")
    exit(1)

df = pd.DataFrame(data)
df.columns = [f"sensor_{i}" for i in range(df.shape[1])]

print(f"Dataset shape: {df.shape}")

print("Generating labels")

df["avg_speed"] = df.mean(axis=1)
df["congestion"] = (df["avg_speed"] < THRESHOLD_SPEED).astype(int)

print("Applying temporal train-test split")

X = df.drop(columns=["avg_speed", "congestion"])
y = df["congestion"].shift(-1)

X = X.iloc[:-1]
y = y.iloc[:-1]

split_idx = int(0.8 * len(X))
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print("Scaling features")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training and evaluating models")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50, random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

results = []
os.makedirs(RESULTS_DIR, exist_ok=True)

for name, model in models.items():
    print(f"Model: {name}")

    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train

    start_inf = time.time()
    y_pred = model.predict(X_test_scaled)
    inf_time = (time.time() - start_inf) / len(y_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    model_path = os.path.join(
        RESULTS_DIR, name.replace(" ", "_") + ".joblib"
    )
    joblib.dump(model, model_path)
    model_size_kb = os.path.getsize(model_path) / 1024

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training time: {train_time:.2f} s")
    print(f"Inference time: {inf_time:.2e} s/sample")
    print(f"Model size: {model_size_kb:.2f} KB")

    results.append(
        [name, acc, f1, train_time, inf_time, model_size_kb]
    )

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "F1 Score",
        "Training Time (s)",
        "Inference Time (s/sample)",
        "Model Size (KB)",
    ],
)

results_df.to_csv(
    os.path.join(RESULTS_DIR, "results_table.csv"), index=False
)

print("Experiment completed")
print(results_df.to_string(index=False))
