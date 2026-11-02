import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "latency_ms",
    "packet_loss_pct",
    "bandwidth_mbps",
    "cpu_usage_pct",
    "connections",
]
TARGET_COLUMN = "fault_type"

def generate_network_data(n_samples=2500, random_state=42):
    """Generate illustrative network telemetry with five labeled conditions."""
    rng = np.random.default_rng(random_state)
    classes = ["Normal", "High Latency", "Packet Loss", "Congestion", "Link Failure"]
    proportions = [0.42, 0.16, 0.16, 0.16, 0.10]
    labels = rng.choice(classes, size=n_samples, p=proportions)
    rows = []

    for label in labels:
        if label == "Normal":
            latency = rng.normal(28, 9)
            loss = rng.beta(1.2, 35) * 4
            bandwidth = rng.normal(650, 140)
            cpu = rng.normal(35, 13)
            connections = rng.normal(150, 55)
        elif label == "High Latency":
            latency = rng.normal(230, 65)
            loss = rng.beta(1.5, 18) * 8
            bandwidth = rng.normal(380, 150)
            cpu = rng.normal(48, 18)
            connections = rng.normal(210, 80)
        elif label == "Packet Loss":
            latency = rng.normal(95, 35)
            loss = rng.normal(12, 4.5)
            bandwidth = rng.normal(430, 150)
            cpu = rng.normal(52, 17)
            connections = rng.normal(180, 65)
        elif label == "Congestion":
            latency = rng.normal(155, 48)
            loss = rng.normal(4.5, 2.2)
            bandwidth = rng.normal(95, 45)
            cpu = rng.normal(88, 8)
            connections = rng.normal(850, 230)
        else:
            latency = rng.normal(850, 180)
            loss = rng.normal(48, 15)
            bandwidth = rng.normal(8, 8)
            cpu = rng.normal(12, 8)
            connections = rng.normal(8, 6)

        rows.append([
            np.clip(latency, 1, 2000),
            np.clip(loss, 0, 100),
            np.clip(bandwidth, 0, 2000),
            np.clip(cpu, 0, 100),
            int(np.clip(connections, 0, 5000)),
            label
        ])

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS + [TARGET_COLUMN]).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

def validate_dataset(df):
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return False, "Missing required columns: " + ", ".join(missing)
    if len(df) < 30:
        return False, "Please provide at least 30 rows."
    if df[TARGET_COLUMN].nunique() < 2:
        return False, "The target column must contain at least two classes."
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[FEATURE_COLUMNS].isna().all().any():
        return False, "At least one feature column contains no numeric values."
    if df[TARGET_COLUMN].isna().any():
        return False, "The fault_type column contains empty labels."
    if df[TARGET_COLUMN].value_counts().min() < 2:
        return False, "Each fault class must have at least two records."
    return True, "Dataset is valid."
