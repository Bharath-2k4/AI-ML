import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Behaviour_view\Data\behaviour_features.csv"
MODEL_DIR = r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Behaviour_view"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

# ----------------------------------
# CREATE WEAK INACTIVITY FEATURE
# ----------------------------------
df["login_gap_bucket"] = pd.cut(
    df["days_since_last_login"],
    bins=[-1, 3, 7, 14, 30, 1000],
    labels=[0, 1, 2, 3, 4]
).astype(int)


# -----------------------------
# FEATURES & LABEL
# -----------------------------
feature_cols = [
    "login_gap_bucket",
    "avg_session_duration",
    "weekly_login_frequency",
    "social_interactions",
    "game_progress",
    "recent_spending"
]

X = df[feature_cols]
y = df["churn"]

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("\n===== BEHAVIOUR VIEW – CHURN MODEL =====")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, f"{MODEL_DIR}/behaviour_churn_model.pkl")
joblib.dump(feature_cols, f"{MODEL_DIR}/behaviour_feature_cols.pkl")

print("\n✅ Churn model saved successfully")
