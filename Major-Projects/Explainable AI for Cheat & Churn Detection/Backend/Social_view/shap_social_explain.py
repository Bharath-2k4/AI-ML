import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "Data", "social_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "social_model.pkl")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "social_feature_cols.pkl")

OUT_DIR = os.path.join(BASE_DIR, "shap_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# LOAD DATA & MODEL
# =====================================================
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
FEATURE_COLS = joblib.load(FEATURE_COLS_PATH)

X = df[FEATURE_COLS]

print("📊 Social SHAP data shape:", X.shape)

# =====================================================
# SHAP EXPLAINER
# =====================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# =====================================================
# NORMALIZE SHAP OUTPUT (CRITICAL FIX)
# =====================================================
if isinstance(shap_values, list):
    # Binary classification → use positive class
    shap_values_used = shap_values[1]
else:
    # New SHAP versions return array directly
    shap_values_used = shap_values

print("🧠 SHAP values shape:", shap_values_used.shape)

# =====================================================
# GLOBAL SHAP (BAR PLOT – SAFE)
# =====================================================
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_used,
    X,
    plot_type="bar",
    show=False
)

global_path = os.path.join(OUT_DIR, "social_global_shap.png")
plt.tight_layout()
plt.savefig(global_path, dpi=200)
plt.close()

print("✅ Global SHAP saved:", global_path)


# =====================================================
# LOCAL SHAP (SINGLE PLAYER — FIXED)
# =====================================================
high_risk_idx = df["cheater_ratio"].idxmax()

# Extract SHAP values for ONE sample & ONE class
local_shap_values = shap_values_used[high_risk_idx]

# If SHAP still returns multi-output, select class 1
if local_shap_values.ndim == 2:
    local_shap_values = local_shap_values[:, 1]

# Get correct base value
if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_value = explainer.expected_value[1]
else:
    base_value = explainer.expected_value

explanation = shap.Explanation(
    values=local_shap_values,
    base_values=base_value,
    data=X.iloc[high_risk_idx],
    feature_names=FEATURE_COLS
)

plt.figure(figsize=(10, 6))
shap.plots.waterfall(explanation, show=False)

local_path = os.path.join(OUT_DIR, "social_local_shap.png")
plt.tight_layout()
plt.savefig(local_path, dpi=200)
plt.close()

print("✅ Local SHAP saved:", local_path)
print("🎯 Explained player_id:", df.loc[high_risk_idx, "player_id"])
