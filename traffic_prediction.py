import h5py
import numpy as np
import pandas as pd
import time
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
DATA_PATH = "metr-la.h5"  # Place dataset in same folder as this script
RESULTS_DIR = "results"
THRESHOLD_SPEED = 40  # km/h - congestion threshold

print("=" * 80)
print("Traffic Congestion Prediction - Green AI Implementation")
print("=" * 80)

print("\n[1/6] Loading METR-LA dataset...")
try:
    with h5py.File(DATA_PATH, 'r') as f:
        data = f['df']['block0_values'][:]
        columns = [c.decode('utf-8') for c in f['df']['axis0'][:]]
    
    df = pd.DataFrame(data, columns=columns)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} sensors")
except FileNotFoundError:
    print(f"ERROR: Dataset file '{DATA_PATH}' not found!")
    print("\nPlease download the METR-LA dataset:")
    print("1. Visit: https://www.kaggle.com/datasets/madmuthu/metr-la")
    print("2. Download 'metr-la.h5'")
    print(f"3. Place it in the same folder as this script")
    exit(1)

print("\n[2/6] Creating features and labels...")

# Rename columns for clarity
df = pd.DataFrame(data)
df.columns = [f"sensor_{i}" for i in range(df.shape[1])]

# Compute average speed across all sensors
df['avg_speed'] = df.mean(axis=1)

# Define congestion: average speed < 40 km/h
df['congestion'] = (df['avg_speed'] < THRESHOLD_SPEED).astype(int)

print(f"Class distribution:")
print(f"  Free-flow (0): {(df['congestion'] == 0).sum()} samples " +
      f"({100 * (df['congestion'] == 0).sum() / len(df):.2f}%)")
print(f"  Congestion (1): {(df['congestion'] == 1).sum()} samples " +
      f"({100 * (df['congestion'] == 1).sum() / len(df):.2f}%)")

print("\n[3/6] Creating temporal train-test split...")

# Features at time t
X = df.drop(columns=['congestion', 'avg_speed'])

# Target at time t+1 (temporal shift prevents label leakage)
y = df['congestion'].shift(-1)

# Drop last row (NaN target after shift)
X = X.iloc[:-1]
y = y.iloc[:-1]

# Chronological 80-20 split (no shuffling to preserve temporal order)
split_idx = int(0.8 * len(X))
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

print("\n[4/6] Applying StandardScaler...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled to zero mean and unit variance")

print("\n[5/6] Training and evaluating models...")
print("-" * 80)

# Define all models with their hyperparameters
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []
os.makedirs(RESULTS_DIR, exist_ok=True)

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Measure training time
    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train
    
    # Measure inference time
    start_inf = time.time()
    y_pred = model.predict(X_test_scaled)
    inf_time = (time.time() - start_inf) / len(y_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Measure model size
    model_path = f"{RESULTS_DIR}/{name.replace(' ', '_')}.joblib"
    joblib.dump(model, model_path)
    model_size = os.path.getsize(model_path) / 1024  # Convert to KB
    
    # Display results
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Training Time: {train_time:.2f}s")
    print(f"  Inference Time: {inf_time:.2e}s/sample")
    print(f"  Model Size: {model_size:.2f} KB")
    
    # Store results
    results.append([name, acc, f1, train_time, inf_time, model_size])

print("\n[6/6] Saving results...")

# Create results DataFrame
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "F1 Score",
        "Training Time (s)",
        "Inference Time (s/sample)",
        "Model Size (KB)"
    ]
)

# Save to CSV
results_df.to_csv(f"{RESULTS_DIR}/results_table.csv", index=False)
print(f"Results saved to {RESULTS_DIR}/results_table.csv")

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80 + "\n")
print(results_df.to_string(index=False))
print("\n" + "=" * 80)
print("Experiment completed successfully!")
print("=" * 80)

print("\n Key Findings:")
print(f"• Best F1-Score: {results_df['F1 Score'].max():.4f} " +
      f"({results_df.loc[results_df['F1 Score'].idxmax(), 'Model']})")
print(f"• Fastest Training: {results_df['Training Time (s)'].min():.4f}s " +
      f"({results_df.loc[results_df['Training Time (s)'].idxmin(), 'Model']})")
print(f"• Smallest Model: {results_df['Model Size (KB)'].min():.2f} KB " +
      f"({results_df.loc[results_df['Model Size (KB)'].idxmin(), 'Model']})")
