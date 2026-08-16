import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# 1. Generate Synthetic Dataset for Network Fault Detection
def generate_network_data(n_samples=1000):
    np.random.seed(42)
    latency = np.random.uniform(5, 500, n_samples)  # ms
    packet_loss = np.random.uniform(0, 20, n_samples)  # %
    throughput = np.random.uniform(1, 100, n_samples)  # Mbps
    jitter = np.random.uniform(1, 50, n_samples)  # ms
    error_rate = np.random.uniform(0, 10, n_samples)  # %

    # Define Fault Logic (1 = Fault Detected, 0 = Normal)
    fault = (
        (latency > 200) | (packet_loss > 5) | (throughput < 10) | (error_rate > 3)
    ).astype(int)

    df = pd.DataFrame(
        {
            "Latency_ms": latency,
            "Packet_Loss_pct": packet_loss,
            "Throughput_Mbps": throughput,
            "Jitter_ms": jitter,
            "Error_Rate_pct": error_rate,
            "Fault": fault,
        }
    )
    return df


df = generate_network_data()
X = df.drop(columns=["Fault"])
y = df["Fault"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Define and Train 3 Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

best_accuracy = 0
best_model = None
best_model_name = ""

print("--- Model Training Results ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc * 100:.2f}%")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Save the best model
joblib.dump(best_model, "best_fault_model.pkl")
print(f"\nSaved Best Model: {best_model_name} ({best_accuracy * 100:.2f}%)")