import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "Data", "social_features.csv"
)

MODEL_PATH = os.path.join(BASE_DIR, "social_model.pkl")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "social_feature_cols.pkl")

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

print("📊 Loaded social dataset:", df.shape)

# =====================================================
# CREATE SOCIAL RISK LABEL (SYNTHETIC)
# =====================================================
# High risk if:
# - Many cheater neighbors
# - Or high cheater ratio
# - Or very central in network

df["social_risk"] = (
    (df["cheater_ratio"] > 0.3) |
    (df["cheater_neighbors"] >= 2) |
    (df["pagerank_score"] > df["pagerank_score"].quantile(0.75))
).astype(int)

print("⚠️ Social risk label distribution:")
print(df["social_risk"].value_counts())

# =====================================================
# FEATURES & TARGET
# =====================================================
FEATURE_COLS = [
    "num_unique_connections",
    "total_interactions",
    "interaction_entropy",
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "pagerank_score",
    "cheater_neighbors",
    "cheater_ratio"
]

X = df[FEATURE_COLS]
y = df["social_risk"]

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =====================================================
# MODEL
# =====================================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================
y_pred = model.predict(X_test)

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# =====================================================
# SAVE MODEL
# =====================================================
joblib.dump(model, MODEL_PATH)
joblib.dump(FEATURE_COLS, FEATURE_COLS_PATH)

print("✅ Social Risk model saved")
print("📁 Model:", MODEL_PATH)
print("📁 Features:", FEATURE_COLS_PATH)
