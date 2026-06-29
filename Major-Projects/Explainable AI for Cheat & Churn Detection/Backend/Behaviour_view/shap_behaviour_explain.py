import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Behaviour_view\Data\behaviour_features.csv"
MODEL_PATH = r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Behaviour_view\behaviour_churn_model.pkl"
FEATURES_PATH = r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Behaviour_view\behaviour_feature_cols.pkl"

OUT_DIR = r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Behaviour_view\shap_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Recreate bucket feature (must match training)
df["login_gap_bucket"] = pd.cut(
    df["days_since_last_login"],
    bins=[-1, 3, 7, 14, 30, 1000],
    labels=[0, 1, 2, 3, 4]
).astype(int)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURES_PATH)

X = df[feature_cols]

# -----------------------------
# SHAP EXPLAINER (NEW API)
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Select churn class (class index = 1)
shap_vals_churn = shap_values.values[:, :, 1]
base_values = shap_values.base_values[:, 1]

# -----------------------------
# GLOBAL SHAP
# -----------------------------
plt.figure()
shap.summary_plot(
    shap_vals_churn,
    X,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "behaviour_churn_global_shap.png"),
    dpi=300
)
plt.close()

# -----------------------------
# LOCAL SHAP
# -----------------------------
sample_index = 0

plt.figure()
shap.force_plot(
    base_values[sample_index],
    shap_vals_churn[sample_index],
    X.iloc[sample_index],
    matplotlib=True,
    show=False
)
plt.savefig(
    os.path.join(OUT_DIR, "behaviour_churn_local_shap.png"),
    dpi=300
)
plt.close()

print("✅ SHAP explanations generated successfully")
print("Saved at:", OUT_DIR)
