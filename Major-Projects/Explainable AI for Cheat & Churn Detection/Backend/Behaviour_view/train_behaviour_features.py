import pandas as pd
import numpy as np

# -----------------------------
# LOAD RAW SEQUENTIAL DATA
# -----------------------------
DATA_PATH = r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Behaviour_view\Data\behaviour_churn_features.csv"
data = pd.read_csv(DATA_PATH)

data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.sort_values(["player_id", "timestamp"])

# -----------------------------
# BUILD SESSIONS
# -----------------------------
sessions = []
current = {}

for _, row in data.iterrows():

    if row["action"] == "login":
        current = {
            "player_id": row["player_id"],
            "login_time": row["timestamp"],
            "session_duration": 0,
            "achievement": 0,
            "social": 0,
            "spending": 0
        }

    elif row["action"] == "session_duration":
        current["session_duration"] = float(row["value"])

    elif row["action"] == "achievement":
        current["achievement"] = 1

    elif row["action"] == "social_interaction":
        current["social"] += 1

    elif row["action"] == "spending":
        current["spending"] += float(row["value"])

    elif row["action"] == "logout" and current:
        current["logout_time"] = row["timestamp"]
        sessions.append(current)
        current = {}

df = pd.DataFrame(sessions)

# -----------------------------
# AGGREGATE PER PLAYER
# -----------------------------
features = []
now = df["login_time"].max()

for pid, g in df.groupby("player_id"):
    g = g.sort_values("login_time")

    days_since_last_login = (now - g["login_time"].max()).days
    avg_session_duration = g["session_duration"].mean()
    weekly_login_frequency = len(g) / max(
        1, ((g["login_time"].max() - g["login_time"].min()).days / 7))
    social_interactions = g["social"].sum()
    game_progress = g["achievement"].mean() * 100
    recent_spending = g["spending"].sum()

    # BASE churn rule (deterministic)
    churn = 1 if days_since_last_login > 14 else 0

    features.append([
        pid,
        days_since_last_login,
        avg_session_duration,
        weekly_login_frequency,
        social_interactions,
        game_progress,
        recent_spending,
        churn
    ])

feature_df = pd.DataFrame(
    features,
    columns=[
        "player_id",
        "days_since_last_login",
        "avg_session_duration",
        "weekly_login_frequency",
        "social_interactions",
        "game_progress",
        "recent_spending",
        "churn"
    ]
)

# -----------------------------
# 🔴 OPTION A: ADD LABEL NOISE
# -----------------------------
np.random.seed(42)

noise_ratio = 0.07   # 7% label noise (ideal)
flip_mask = np.random.rand(len(feature_df)) < noise_ratio

feature_df.loc[flip_mask, "churn"] = 1 - feature_df.loc[flip_mask, "churn"]

print(f"🔀 Injected label noise into {flip_mask.sum()} rows")

# -----------------------------
# SAVE DATASET
# -----------------------------
OUT_PATH = "Backend\Behaviour_view\Data\behaviour_churn_features.csv"
feature_df.to_csv(OUT_PATH, index=False)

print("✅ Behaviour training dataset saved at:")
print(OUT_PATH)
print(feature_df.head())
