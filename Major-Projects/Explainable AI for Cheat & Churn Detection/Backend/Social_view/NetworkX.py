import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from flask import request, jsonify

# =================================================
# SOCIAL VIEW MAIN FUNCTION
# =================================================
def run_social_view(player_id):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # -------------------------------
        # PATHS
        # -------------------------------
        data_path = os.path.join(BASE_DIR, "Social_view", "Data", "social_features.csv")
        edges_path = os.path.join(BASE_DIR, "Social_view", "Data", "social_edges.csv")
        model_path = os.path.join(BASE_DIR, "Social_view", "social_model.pkl")
        feature_path = os.path.join(BASE_DIR, "Social_view", "social_feature_cols.pkl")

        # -------------------------------
        # LOAD DATA
        # -------------------------------
        df = pd.read_csv(data_path)
        df["player_id"] = df["player_id"].astype(str)

        edges_df = pd.read_csv(edges_path)
        edges_df["source_id"] = edges_df["source_id"].astype(str)
        edges_df["target_id"] = edges_df["target_id"].astype(str)

        model = joblib.load(model_path)
        feature_cols = joblib.load(feature_path)

        # -------------------------------
        # FIND PLAYER
        # -------------------------------
        input_id = str(player_id).strip()
        row = df[df["player_id"] == input_id]

        if row.empty:
            return {"error": "Player not found"}

        row = row.iloc[[0]]  # force single row

        # -------------------------------
        # PREDICTION
        # -------------------------------
        X = row[feature_cols]
        prob = float(model.predict_proba(X)[0][1])
        confidence = round(max(prob, 1 - prob), 4)

        if prob < 0.4:
            risk = "LOW SOCIAL RISK"
        elif prob < 0.6:
            risk = "MEDIUM SOCIAL RISK"
        else:
            risk = "HIGH SOCIAL RISK"

        social_risk_score = round(prob * 100, 2)


        # -------------------------------
        # SOCIAL INFLUENCE SCORE
        # -------------------------------
        pagerank = float(row["pagerank_score"].iloc[0])
        degree = float(row["degree_centrality"].iloc[0])
        between = float(row["betweenness_centrality"].iloc[0])

        influence_score = round(
    min(
        (pagerank * 0.4 + degree * 0.003 + between * 0.3) * 100,
        100
    ),
    2
)

        # -------------------------------
        # SHAP
        # -------------------------------
        local_shap, shap_features = generate_local_social_shap(
            model, X, feature_cols, input_id
        )

        plain_reasons = build_plain_english_reasons(shap_features, risk)


        # -------------------------------
        # SOCIAL GRAPH
        # -------------------------------
        graph_result = generate_social_network_graph(edges_df, input_id)
        social_graph = graph_result["image"]
        social_graph_message = graph_result["message"]

        return {
    "entity_id": input_id,                      # renamed for clarity
    "prediction": risk,                         # LOW / MEDIUM / HIGH SOCIAL RISK
    "confidence": confidence,                   # model confidence (0–1)

    # Explanation
    "plain_reasons": plain_reasons,

    # ✅ SINGLE authoritative score
    "social_risk_score": influence_score,       # 0–100 (what UI shows)

    # Optional transparency (for logs / research, not UI)
    "structural_influence_components": {
        "pagerank": pagerank,
        "degree_centrality": degree,
        "betweenness_centrality": between
    },

    # Explainability
    "local_shap": local_shap,
    "global_shap": "/static/social_shap/social_global_shap.png",

    # Network context
    "social_graph": social_graph,
    "social_graph_message": social_graph_message
}


    except Exception as e:
        return {"error": f"Social view failed internally: {str(e)}"}


# =================================================
# SHAP GENERATION
# =================================================
def generate_local_social_shap(model, X, feature_cols, player_id):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    if len(shap_values.values.shape) == 3:
        values = shap_values.values[0, :, 1]
        base_value = shap_values.base_values[0, 1]
    else:
        values = shap_values.values[0]
        base_value = shap_values.base_values[0]

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=X.iloc[0].values,
        feature_names=feature_cols
    )

    out_dir = os.path.join(BASE_DIR, "static", "social_shap")
    os.makedirs(out_dir, exist_ok=True)

    filename = f"local_{player_id}.png"
    out_path = os.path.join(out_dir, filename)

    plt.figure(figsize=(6, 3))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    abs_vals = np.abs(values)
    top_idx = np.argsort(abs_vals)[-3:][::-1]
    top_features = [feature_cols[i] for i in top_idx if abs_vals[i] > 0.01]

    return f"/static/social_shap/{filename}", top_features

# =================================================
# PLAIN ENGLISH EXPLANATION
# =================================================
def build_plain_english_reasons(features, risk_level):
    messages = []

    # -----------------------------
    # HIGH SOCIAL RISK
    # -----------------------------
    if risk_level == "HIGH SOCIAL RISK":
        if "pagerank_score" in features:
            messages.append(
                "Structurally central in the social network (high PageRank)."
            )

        if "betweenness_centrality" in features:
            messages.append(
                "Acts as a bridge between multiple communities."
            )

        if "degree_centrality" in features:
            messages.append(
                "High interaction reach despite no direct cheater neighbors."
            )

        # ✅ FILLER BUT VALID (prevents UI gap)
        messages.append(
            "Positioned close to multiple high-activity entities, increasing indirect exposure risk."
        )

    # -----------------------------
    # MEDIUM SOCIAL RISK
    # -----------------------------
    elif risk_level == "MEDIUM SOCIAL RISK":
        messages.append(
            "Moderate social connectivity with some exposure potential."
        )

        messages.append(
            "Occasional bridge behavior between nearby player groups."
        )

        messages.append(
            "Balanced network position limits large-scale influence spread."
        )

    # -----------------------------
    # LOW SOCIAL RISK
    # -----------------------------
    else:  # LOW SOCIAL RISK
        messages.append(
            "Peripheral position in the social network."
        )

        messages.append(
            "Limited interaction reach with other players."
        )

        messages.append(
            "Low exposure to risky social influence."
        )

        # ✅ FILLER BUT VALID (prevents UI gap)
        messages.append(
            "Minimal influence propagation due to low centrality and weak network reach."
        )

    return messages




# =================================================
# SOCIAL NETWORK GRAPH
# =================================================
def generate_social_network_graph(edges_df, target_id):
    G = nx.Graph()

    
    for _, row in edges_df.iterrows():
        G.add_edge(str(row["source_id"]), str(row["target_id"]))

    # 🔴 FIX: player has NO social edges
    if target_id not in G:
        return {
            "image": None,
            "message": "This player has no recorded social connections."
        }

    ego = nx.ego_graph(G, target_id, radius=1)
    pos = nx.spring_layout(ego, seed=42)


    for _, row in edges_df.iterrows():
        G.add_edge(str(row["source_id"]), str(row["target_id"]))

    # 🔴 FIX: player has NO social edges
    if target_id not in G:
        return {
            "image": None,
            "message": "This player has no recorded social connections."
        }
    plt.figure(figsize=(4, 3))

    # Identify risky neighbors (example rule)
    risky_nodes = [
        n for n in ego.nodes
        if n != target_id and n.startswith("team_")  # adjust rule if needed
    ]

    
    normal_nodes = [
            n for n in ego.nodes
            if n not in risky_nodes and n != target_id
        ]

# Normal nodes
    nx.draw_networkx_nodes(
        ego, pos,
        nodelist=normal_nodes,
        node_color="lightgray",
        node_size=80,
        alpha=0.6
    )

# Risky neighbors
    nx.draw_networkx_nodes(
        ego, pos,
        nodelist=risky_nodes,
        node_color="orange",
        node_size=140,
        alpha=0.9
    )

# Target player
    nx.draw_networkx_nodes(
        ego, pos,
        nodelist=[target_id],
        node_color="red",
        node_size=350
    )

    # Edges
    nx.draw_networkx_edges(
        ego, pos,
        alpha=0.4,
        edge_color="gray"
    )


    plt.title("Social Network Context")
    plt.axis("off")

    out_dir = os.path.join("static", "social_graphs")
    os.makedirs(out_dir, exist_ok=True)

    filename = f"social_graph_{target_id}.png"
    path = os.path.join(out_dir, filename)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

    return {
    "image": f"/static/social_graphs/{filename}",
    "message": None
}
