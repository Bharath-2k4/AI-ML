import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "Backend\Portrait_stuff\Data\Portrait_view_data.csv"
CHEAT_MODEL_PATH = "Backend\Portrait_stuff\Models\portrait_cheat_model.pkl"
CHURN_MODEL_PATH = "Backend\Portrait_stuff\Models\portrait_churn_model.pkl"

# -----------------------------
# Load data & models
# -----------------------------
df = pd.read_csv(DATA_PATH)

cheat_model = joblib.load(CHEAT_MODEL_PATH)
churn_model = joblib.load(CHURN_MODEL_PATH)

FEATURES = [
    "role_level","vip_level","account_age_days","battle_rank",
    "online_time","1d_login_cnt","1w_login_cnt","1m_login_cnt",
    "capability","online_get_exp_amt","recharge_amt",
    "online_get_money_amt","device_width","device_height","os_ver"
]

# -----------------------------
# Test by player_id
# -----------------------------
def test_player(player_id: int):
    row = df[df["player_id"] == player_id]

    if row.empty:
        print(f"❌ Player ID {player_id} not found")
        return

    X = row[FEATURES]

    # Cheat prediction
    cheat_prob = cheat_model.predict_proba(X)[0][1]
    cheat_pred = int(cheat_prob >= 0.5)

    # Churn prediction
    churn_prob = churn_model.predict_proba(X)[0][1]
    churn_pred = int(churn_prob >= 0.5)

    # -----------------------------
    # Output
    # -----------------------------
    print("\n==============================")
    print(f"PLAYER ID: {player_id}")
    print("------------------------------")
    print(f"Cheat Prediction : {'YES' if cheat_pred else 'NO'}")
    print(f"Cheat Probability: {cheat_prob:.4f}")
    print("------------------------------")
    print(f"Churn Prediction : {'YES' if churn_pred else 'NO'}")
    print(f"Churn Probability: {churn_prob:.4f}")



# -----------------------------
# Example test
# -----------------------------
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter player_id: ")

        if user_input.lower() == "q":
            print("👋 Exiting test")
            break

        if not user_input.isdigit():
            print("❌ Please enter a valid numeric player_id")
            continue

        test_player(int(user_input))
