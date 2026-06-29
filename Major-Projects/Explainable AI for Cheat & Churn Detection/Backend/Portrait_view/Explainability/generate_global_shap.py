# generate_global_shap.py
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

def get_positive_class_shap(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    DATA_PATH = os.path.join(BASE_DIR, "Data", "Portrait_view_data.csv")
    OUT_DIR = os.path.join(BASE_DIR, "Explainability")
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    FEATURE_COLS = [
        c for c in df.columns
        if c not in ["player_id", "cheat", "churn"]
    ]

    X = df[FEATURE_COLS].sample(
        n=min(2000, len(df)),
        random_state=42
    )

    print(f"🧠 SHAP rows: {X.shape[0]}, features: {X.shape[1]}")

    cheat_model = joblib.load(
        os.path.join(BASE_DIR, "Models", "portrait_cheat_model.pkl")
    )

    churn_model = joblib.load(
        os.path.join(BASE_DIR, "Models", "portrait_churn_model.pkl")
    )

    print("⚙️ Computing SHAP for Cheat model...")
    cheat_explainer = shap.TreeExplainer(cheat_model)
    cheat_shap = get_positive_class_shap(
        cheat_explainer.shap_values(X)
    )

    print("⚙️ Computing SHAP for Churn model...")
    churn_explainer = shap.TreeExplainer(churn_model)
    churn_shap = get_positive_class_shap(
        churn_explainer.shap_values(X)
    )

    print("📊 Saving plots...")

    shap.summary_plot(cheat_shap, X, plot_type="bar", show=False)
    plt.savefig(os.path.join(OUT_DIR, "shap_global_cheat.png"))
    plt.close()

    shap.summary_plot(churn_shap, X, plot_type="bar", show=False)
    plt.savefig(os.path.join(OUT_DIR, "shap_global_churn.png"))
    plt.close()

    print("✅ Global SHAP plots generated")
