import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "Backend\Portrait_stuff\Data\Portrait_view_data.csv"
MODEL_DIR = "Backend\Portrait_stuff\Models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "role_level","vip_level","account_age_days","battle_rank",
    "online_time","1d_login_cnt","1w_login_cnt","1m_login_cnt",
    "capability","online_get_exp_amt","recharge_amt",
    "online_get_money_amt","device_width","device_height","os_ver"
]

X = df[FEATURES]

# =============================
# CHEAT MODEL
# =============================
y_cheat = df["cheat"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cheat, test_size=0.2, random_state=42, stratify=y_cheat
)

cheat_model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

cheat_model.fit(X_train, y_train)

print("\n===== CHEAT MODEL =====")
print("Accuracy:", accuracy_score(y_test, cheat_model.predict(X_test)))
print(classification_report(y_test, cheat_model.predict(X_test)))

joblib.dump(cheat_model, f"{MODEL_DIR}/portrait_cheat_model.pkl")

# =============================
# CHURN MODEL
# =============================
y_churn = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)

churn_model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

churn_model.fit(X_train, y_train)

print("\n===== CHURN MODEL =====")
print("Accuracy:", accuracy_score(y_test, churn_model.predict(X_test)))
print(classification_report(y_test, churn_model.predict(X_test)))

joblib.dump(churn_model, f"{MODEL_DIR}/portrait_churn_model.pkl")

print("\n✅ Portrait training completed")
