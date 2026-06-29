import numpy as np

def get_portrait_explanation(model, X, feature_names, top_k=5):
    """
    Lightweight SHAP-style explanation for Flask usage.
    Uses model feature importance (tree-based models).
    
    Parameters:
    - model: trained ML model
    - X: DataFrame with a single row
    - feature_names: list of feature names
    - top_k: number of reasons to return
    """

    try:
        # Tree-based models only
        if not hasattr(model, "feature_importances_"):
            return ["Model does not support feature importance"]

        importances = model.feature_importances_

        # Sort features by importance
        idx = np.argsort(importances)[::-1][:top_k]

        reasons = []
        for i in idx:
            reasons.append(
                f"{feature_names[i]} (impact: {importances[i]:.4f})"
            )

        return reasons

    except Exception as e:
        return [f"Explanation failed: {str(e)}"]
