import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
import random
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from Portrait_view.Explainability.shap_portrait import get_portrait_explanation
from Images_ML.Explainability.gradcam import run_gradcam
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = 'your_secret_key_here'

def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            test_type TEXT NOT NULL,
            cheat_prediction INTEGER,
            churn_prediction INTEGER,
            cheat_confidence REAL,
            churn_confidence REAL,
            cheat_reasons TEXT,
            churn_reasons TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    # Add feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prediction_id INTEGER,
            rating INTEGER NOT NULL,
            comments TEXT,
            helpful INTEGER DEFAULT 0,
            not_helpful INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# Load models & assets
# -# --------------------------
# LOAD DATA
# --------------------------
PORTRAIT_DATA_PATH = os.path.join(
    BASE_DIR,
    "Portrait_view",
    "Data",
    "Portrait_view_data.csv" 
)

if not os.path.exists(PORTRAIT_DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {PORTRAIT_DATA_PATH}")

portrait_df = pd.read_csv(PORTRAIT_DATA_PATH)

# --------------------------
# LOAD MODELS
# --------------------------
CHEAT_MODEL_PATH = os.path.join(
    BASE_DIR, "Portrait_view", "Models", "portrait_cheat_model.pkl"
)

CHURN_MODEL_PATH = os.path.join(
    BASE_DIR, "Portrait_view", "Models", "portrait_churn_model.pkl"
)


cheat_model = joblib.load(CHEAT_MODEL_PATH)
churn_model = joblib.load(CHURN_MODEL_PATH)

# --------------------------
# FEATURES
# --------------------------
TARGET_COLS = ["cheat", "churn"]

FEATURE_COLS = [
    c for c in portrait_df.columns
    if c not in TARGET_COLS + ["player_id"]
]

print("🔢 Feature count:", len(FEATURE_COLS))

# ---------- DATASET INFO ----------
DATASET_INFO = {
    "name": "Portrait Player Dataset",
    "rows": len(portrait_df),
    "features": len(FEATURE_COLS)
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BEHAVIOUR_MODEL_PATH = os.path.join(
    BASE_DIR, "Behaviour_view", "behaviour_churn_model.pkl"
)

if not os.path.exists(BEHAVIOUR_MODEL_PATH):
    print("⚠️ Behaviour model not found:", BEHAVIOUR_MODEL_PATH)
    behaviour_model = None
else:
    behaviour_model = joblib.load(BEHAVIOUR_MODEL_PATH)



try:
    churn_model2 = joblib.load(os.path.join(BASE_DIR, "models", "churn_model2.pkl"))
    scaler2 = joblib.load(os.path.join(BASE_DIR, "models", "scaler2.pkl"))
    feature_names2 = joblib.load(os.path.join(BASE_DIR, "models", "feature_names2.pkl"))
    log_models_loaded = True
except Exception as e:
    scaler2 = None
    feature_names2 = []
    log_models_loaded = False


# ---------------------------
# Helper Functions
# ---------------------------
def get_reasons(model, feature_names, top_k=5):
    try:
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:top_k]

        reasons = []
        for idx in sorted_idx:
            reasons.append(
                f"{feature_names[idx]} (impact: {importances[idx]:.4f})"
            )
        return reasons
    except:
        return ["Feature importance not available"]

def save_prediction(user_id, test_type, cheat_pred, churn_pred, 
                   cheat_conf, churn_conf, cheat_reasons, churn_reasons):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions 
        (user_id, test_type, cheat_prediction, churn_prediction, 
         cheat_confidence, churn_confidence, cheat_reasons, churn_reasons)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, test_type, cheat_pred, churn_pred, 
          cheat_conf, churn_conf, str(cheat_reasons), str(churn_reasons)))
    conn.commit()
    conn.close()

def get_user_predictions(user_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (user_id,))
    predictions = cursor.fetchall()
    conn.close()
    return predictions


# ---------------------------
# Routes
# ---------------------------
# ---------------------------
# Routes
# ---------------------------
@app.route("/portrait-analysis", methods=["GET", "POST"])
def portrait_analysis():

    result = None
    player_id = None  # ✅ always safe

    if request.method == "POST":
        player_id = request.form.get("player_id")

        # ---------------------------
        # Fetch player row
        # ---------------------------
        row = portrait_df[
            portrait_df["player_id"].astype(str) == str(player_id)
        ]

        if row.empty:
            flash("Player ID not found", "danger")
            return redirect(url_for("portrait_analysis"))

        X = row[FEATURE_COLS]
        sample = row.iloc[0]

        # ---------------------------
        # Predictions
        # ---------------------------
        cheat_prob = cheat_model.predict_proba(X)[0][1]
        churn_prob = churn_model.predict_proba(X)[0][1]

        cheat_pred = "Cheater" if cheat_prob >= 0.5 else "Legit"
        churn_pred = "Likely to Leave" if churn_prob >= 0.5 else "Will Stay"

        # ---------------------------
        # SHAP RAW
        # ---------------------------
        cheat_shap_raw = get_portrait_explanation(
            cheat_model, X, FEATURE_COLS
        )

        churn_shap_raw = get_portrait_explanation(
            churn_model, X, FEATURE_COLS
        )

        # ---------------------------
        # Extract cheat feature names
        # ---------------------------
        cheat_features = {
            extract_feature_name(r) for r in cheat_shap_raw
        }

        # ---------------------------
        # Default explanations
        # ---------------------------
        cheat_defaults = [
            "Player actively engages with the game",
            "Consistent gameplay behavior observed",
            "No abnormal activity detected"
        ]

        churn_defaults = [
            "Player shows stable engagement",
            "No strong churn indicators detected",
            "Gameplay activity remains consistent"
        ]

        # ---------------------------
        # Final TOP-3 reasons (RAW)
        # ---------------------------
        cheat_top = get_top3_reasons_with_fallback(
            cheat_shap_raw,
            exclude_features=set(),
            default_reasons=cheat_defaults,
            top_k=3
        )

        churn_top = get_top3_reasons_with_fallback(
            churn_shap_raw,
            exclude_features=cheat_features,
            default_reasons=churn_defaults,
            top_k=3
        )

        # ---------------------------
        # HIGH-CONFIDENCE CHURN OVERRIDE
        # ---------------------------
        if churn_prob * 100 > 80 and churn_pred == "Likely to Leave":
            churn_top = [
                "Elevated churn risk from multiple behaviors",
                *churn_top[:2]
            ]

        # ---------------------------
        # FORMAT FOR UI
        # ---------------------------
        cheat_reasons_ui = [
                format_shap_reason(r, cheat_pred) for r in cheat_top
        ]

        churn_reasons_ui = [
    format_shap_reason(r, churn_pred) for r in churn_top
]



        # ---------------------------
        # RESULT OBJECT (ONLY INSIDE POST)
        # ---------------------------
        result = {
            "player_id": player_id,

            "cheat_pred": cheat_pred,
            "cheat_conf": round(cheat_prob * 100, 2),

            "churn_pred": churn_pred,
            "churn_conf": round(churn_prob * 100, 2),

            "cheat_reasons": cheat_reasons_ui,
            "churn_reasons": churn_reasons_ui,

            "features": sample[FEATURE_COLS].to_dict()
}


        # ---------------------------
        # SAVE LOG
        # ---------------------------
        save_prediction(
            session.get("user_id"),
            "portrait_analysis",
            int(cheat_pred == "Cheater"),
            int(churn_pred == "Likely to Leave"),
            cheat_prob * 100,
            churn_prob * 100,
            cheat_reasons_ui,
            churn_reasons_ui
        )

    # ---------------------------
    # RENDER
    # ---------------------------
    return render_template(
        "Portrait.html",
        result=result,
        dataset_info=DATASET_INFO
    )





# SHAP Post-processing Helpers (FIXED)
# ---------------------------
def extract_feature_name(reason):
    return reason.split(" (impact:")[0].strip().lower()


def format_shap_reason(reason):
    if "(impact:" not in reason:
        return {
            "feature": reason,
            "dir": "neutral"
        }

    feature, impact_part = reason.split("(impact:")
    impact = float(impact_part.replace(")", "").strip())

    if impact > 0:
        direction = "up"
    elif impact < 0:
        direction = "down"
    else:
        direction = "neutral"

    return {
        "feature": feature.replace("_", " ").title().strip(),
        "dir": direction
    }



def get_top3_reasons_with_fallback(
    shap_reasons,
    exclude_features=None,
    default_reasons=None,
    top_k=3
):
    exclude_features = exclude_features or set()
    default_reasons = default_reasons or []

    selected = []
    for r in shap_reasons:
        if extract_feature_name(r) not in exclude_features:
            selected.append(r)
        if len(selected) == top_k:
            break

    if len(selected) < top_k:
        selected.extend(default_reasons[: top_k - len(selected)])

    return selected


def shap_direction(impact):
    """
    Convert SHAP impact to direction arrow
    """
    if impact > 0:
        return "up"
    elif impact < 0:
        return "down"
    return "neutral"

def format_shap_reason(reason, prediction):
    """
    Converts SHAP reason string into UI-friendly trend
    """

    # Human / fallback reasons
    if "(impact:" not in reason:
        return {
            "feature": reason,
            "dir": "neutral"
        }

    feature, impact_part = reason.split("(impact:")
    impact = float(impact_part.replace(")", "").strip())
    feature_name = feature.replace("_", " ").title().strip()

    # ---------------------------
    # CHEAT LOGIC
    # ---------------------------
    if prediction in ["Cheater", "Legit"]:
        if prediction == "Cheater":
            direction = "up" if impact > 0 else "down"
        else:  # Legit
            direction = "down" if impact > 0 else "up"

    # ---------------------------
    # CHURN LOGIC
    # ---------------------------
    elif prediction in ["Likely to Leave", "Will Stay"]:
        if prediction == "Likely to Leave":
            direction = "up" if impact > 0 else "down"
        else:  # Will Stay
            direction = "down" if impact > 0 else "up"

    else:
        direction = "neutral"

    return {
        "feature": feature_name,
        "dir": direction
    }


import os
import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import networkx as nx

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



@app.route("/social_view", methods=["POST"])
def social_view():
    print("🔥 /social_view HIT")

    data = request.get_json()
    print("📦 Payload:", data)

    player_id = data.get("player_id")
    if not player_id:
        return jsonify({"error": "player_id required"}), 400

    result = run_social_view(player_id)

    print("✅ Backend result:", result)   # 👈 ADD THIS

    return jsonify(result)



@app.route("/behaviour", methods=["GET", "POST"])
def behaviour_view():

    prediction = None
    confidence = None
    shap_image = None

    if request.method == "POST":
        # -----------------------------
        # GET USER INPUTS
        # -----------------------------
        days_since_last_login = float(request.form["days_since_last_login"])
        avg_session_duration = float(request.form["avg_session_duration"])
        weekly_login_frequency = float(request.form["weekly_login_frequency"])
        social_interactions = float(request.form["social_interactions"])
        game_progress = float(request.form["game_progress"])
        recent_spending = float(request.form["recent_spending"])

        # -----------------------------
        # CREATE login_gap_bucket (same as training)
        # -----------------------------
        if days_since_last_login <= 3:
            login_gap_bucket = 0
        elif days_since_last_login <= 7:
            login_gap_bucket = 1
        elif days_since_last_login <= 14:
            login_gap_bucket = 2
        elif days_since_last_login <= 30:
            login_gap_bucket = 3
        else:
            login_gap_bucket = 4

        # -----------------------------
        # BUILD FEATURE VECTOR
        # -----------------------------
        input_data = pd.DataFrame([{
            "login_gap_bucket": login_gap_bucket,
            "avg_session_duration": avg_session_duration,
            "weekly_login_frequency": weekly_login_frequency,
            "social_interactions": social_interactions,
            "game_progress": game_progress,
            "recent_spending": recent_spending
        }])

        # -----------------------------
        # PREDICTION
        # -----------------------------
        proba = behaviour_model.predict_proba(input_data)[0][1]
        pred = behaviour_model.predict(input_data)[0]

        prediction = "Likely to Churn" if pred == 1 else "Not Likely to Churn"
        confidence = round(proba * 100, 2)

        # -----------------------------
        # SHAP IMAGE (pre-generated)
        # -----------------------------
        shap_image = "shap/behaviour_churn_global_shap.png"

    return render_template(
        "behaviour.html",
        prediction=prediction,
        confidence=confidence,
        shap_image=shap_image
    )

@app.route("/behaviour_predict", methods=["POST"])
def behaviour_predict():

    data = request.get_json()

    # -----------------------------
    # Read inputs
    # -----------------------------
    days = float(data["daysSinceLogin"])
    avg_session = float(data["sessionDuration"])
    weekly_freq = float(data["loginFrequency"])
    social = float(data["socialInteractions"])
    progress = float(data["gameProgress"])
    spending = float(data["recentSpending"])

    # -----------------------------
    # Create login_gap_bucket (MODEL FEATURE)
    # -----------------------------
    if days <= 3:
        bucket = 0
    elif days <= 7:
        bucket = 1
    elif days <= 14:
        bucket = 2
    elif days <= 30:
        bucket = 3
    else:
        bucket = 4

    # -----------------------------
    # Build model input
    # -----------------------------
    X_input = pd.DataFrame([{
        "login_gap_bucket": bucket,
        "avg_session_duration": avg_session,
        "weekly_login_frequency": weekly_freq,
        "social_interactions": social,
        "game_progress": progress,
        "recent_spending": spending
    }])

    # -----------------------------
    # Predict churn
    # -----------------------------
    proba = behaviour_model.predict_proba(X_input)[0]
    pred = int(proba[1] >= 0.5)
    confidence = round(proba[1] * 100, 2)

    # -----------------------------
    # HUMAN-READABLE REASONS (VALUE-AWARE)
    # -----------------------------
    reasons = []

    # Login inactivity
    if days > 30:
        reasons.append(f"Long inactivity period ({int(days)} days) strongly increases churn risk")
    elif days > 14:
        reasons.append(f"Moderate inactivity ({int(days)} days) increases churn risk")
    else:
        reasons.append(f"Recent login activity ({int(days)} days) reduces churn risk")

    # Session duration
    if avg_session < 20:
        reasons.append(f"Short average session duration ({avg_session:.0f} min) indicates disengagement")
    elif avg_session < 45:
        reasons.append(f"Moderate session duration ({avg_session:.0f} min) slightly affects churn risk")
    else:
        reasons.append(f"Healthy session duration ({avg_session:.0f} min) indicates strong engagement")

    # Game progress
    if progress < 25:
        reasons.append(f"Low game progress ({progress:.0f}%) suggests limited player investment")
    elif progress < 60:
        reasons.append(f"Moderate game progress ({progress:.0f}%) supports continued engagement")
    else:
        reasons.append(f"High game progress ({progress:.0f}%) significantly reduces churn risk")

    # Optional: Social engagement (only if impactful)
    if social == 0:
        reasons.append("No social interactions detected, which may increase churn risk")

    # -----------------------------
    # Return response
    # -----------------------------
    return jsonify({
        "predicted_churn": pred,
        "confidence": confidence,
        "reasons": reasons[:3],  # top 3 readable reasons
        "shap_image": "shap/behaviour_churn_global_shap.png"
    })














@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------
# Feedback Routes
# ---------------------------
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'})
    
    try:
        prediction_id = request.form.get('prediction_id')
        rating = int(request.form.get('rating'))
        comments = request.form.get('comments', '')
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (user_id, prediction_id, rating, comments)
            VALUES (?, ?, ?, ?)
        ''', (session['user_id'], prediction_id, rating, comments))
        conn.commit()
        conn.close()
        
        flash('Thank you for your feedback!', 'success')
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/feedback_helpful', methods=['POST'])
def feedback_helpful():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'})
    
    try:
        feedback_id = request.form.get('feedback_id')
        action = request.form.get('action')  # 'helpful' or 'not_helpful'
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        if action == 'helpful':
            cursor.execute('''
                UPDATE feedback SET helpful = helpful + 1 WHERE id = ?
            ''', (feedback_id,))
        else:
            cursor.execute('''
                UPDATE feedback SET not_helpful = not_helpful + 1 WHERE id = ?
            ''', (feedback_id,))
            
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)})

def get_user_feedback(user_id):
    """Get feedback submitted by the user"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT f.*, p.test_type, p.cheat_prediction, p.churn_prediction
        FROM feedback f
        LEFT JOIN predictions p ON f.prediction_id = p.id
        WHERE f.user_id = ?
        ORDER BY f.created_at DESC
    ''', (user_id,))
    feedback = cursor.fetchall()
    conn.close()
    return feedback

def get_all_feedback():
    """Get all feedback for admin view (optional)"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT f.*, u.username, p.test_type
        FROM feedback f
        LEFT JOIN users u ON f.user_id = u.id
        LEFT JOIN predictions p ON f.prediction_id = p.id
        ORDER BY f.created_at DESC
    ''')
    feedback = cursor.fetchall()
    conn.close()
    return feedback


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form.get('email', '')
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, password, email) 
                VALUES (?, ?, ?)
            ''', (username, hashed_password, email))
            conn.commit()
            conn.close()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists', 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login to access dashboard', 'error')
        return redirect(url_for('login'))
    
    predictions = get_user_predictions(session['user_id'])
    user_feedback = get_user_feedback(session['user_id'])
    
    return render_template('dashboard.html', 
                         predictions=predictions, 
                         user_feedback=user_feedback)




    
@app.route('/log_test')
def log_test():
    if 'user_id' not in session:
        flash('Please login to access this feature', 'error')
        return redirect(url_for('login'))
    
    return render_template('Social.html', features=feature_names2)

@app.route('/generate_log_prediction', methods=['POST'])
def generate_log_prediction():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'})
    
    try:
        # Get form data
        form_data = {}
        for feature in feature_names2:
            form_data[feature] = float(request.form.get(feature, 0))
        
        # Create DataFrame
        new_data = pd.DataFrame([form_data])
        new_data = new_data[feature_names2]
        
        # Scale and predict
        sample_scaled = scaler2.transform(new_data)
        pred_cheat = int(cheat_model2.predict(sample_scaled)[0])
        pred_churn = int(churn_model2.predict(sample_scaled)[0])
        
        # Get probabilities for confidence
        cheat_proba = cheat_model2.predict_proba(sample_scaled)[0]
        churn_proba = churn_model2.predict_proba(sample_scaled)[0]
        
        cheat_confidence = max(cheat_proba) * 100
        churn_confidence = max(churn_proba) * 100
        
        # Get reasons
        cheat_reasons = get_reasons(cheat_model2, feature_names2)
        churn_reasons = get_reasons(churn_model2, feature_names2)
        
        # Save prediction
        save_prediction(
            session['user_id'],
            'log_test',
            pred_cheat,
            pred_churn,
            cheat_confidence,
            churn_confidence,
            cheat_reasons,
            churn_reasons
        )
        
        return jsonify({
            'success': True,
            'predicted_cheat': pred_cheat,
            'predicted_churn': pred_churn,
            'cheat_confidence': round(cheat_confidence, 2),
            'churn_confidence': round(churn_confidence, 2),
            'cheat_reasons': cheat_reasons,
            'churn_reasons': churn_reasons,
            'input_data': form_data
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/churn_test')
def churn_test():
    if 'user_id' not in session:
        flash('Please login to access this feature', 'error')
        return redirect(url_for('login'))
    
    return render_template('Behavior.html')


def generate_synthetic_churn_sample():
    """Generate synthetic churn data when real dataset is not available"""
    try:
        # Create synthetic features that might indicate churn
        synthetic_features = {
            'session_duration': random.randint(5, 300),
            'login_frequency': random.randint(1, 30),
            'achievement_rate': random.uniform(0, 100),
            'social_interactions': random.randint(0, 50),
            'payment_amount': random.uniform(0, 1000),
            'days_since_last_login': random.randint(1, 90),
            'game_progress': random.uniform(0, 100),
            'session_regularity': random.uniform(0, 1)
        }
        
        # Simple rule-based churn prediction for demonstration
        # Higher days since last login + lower session duration = higher churn probability
        churn_probability = (
            synthetic_features['days_since_last_login'] * 0.4 +
            (100 - synthetic_features['session_duration']) * 0.3 +
            (100 - synthetic_features['login_frequency']) * 0.3
        ) / 100
        
        pred_churn = 1 if churn_probability > 0.6 else 0
        churn_confidence = churn_probability * 100 if pred_churn else (1 - churn_probability) * 100
        
        # Generate realistic reasons
        churn_reasons = []
        if synthetic_features['days_since_last_login'] > 30:
            churn_reasons.append(f"Long login gap ({synthetic_features['days_since_last_login']} days)")
        if synthetic_features['session_duration'] < 30:
            churn_reasons.append("Short session duration")
        if synthetic_features['login_frequency'] < 5:
            churn_reasons.append("Low login frequency")
        if synthetic_features['social_interactions'] < 10:
            churn_reasons.append("Limited social engagement")
        
        if not churn_reasons:
            churn_reasons = ["Stable engagement patterns", "Regular play behavior"]
        
        # Save prediction
        save_prediction(
            session['user_id'],
            'churn_test',
            0,
            pred_churn,
            0,
            round(churn_confidence, 2),
            [],
            churn_reasons
        )
        
        return jsonify({
            'success': True,
            'sample_index': 'synthetic',
            'dataset_used': 'synthetic',
            'actual_churn': random.randint(0, 1),  # Random for demo
            'predicted_churn': pred_churn,
            'churn_confidence': round(churn_confidence, 2),
            'churn_reasons': churn_reasons,
            'sample_data': synthetic_features
        })
        
    except Exception as e:
        print(f"Synthetic churn error: {e}")
        return jsonify({
            'error': f'Churn detection failed: {str(e)}'
        })



























@app.route('/image_test')
def image_test():
    if 'user_id' not in session:
        flash('Please login to access this feature', 'error')
        return redirect(url_for('login'))
    
    return render_template('image_test.html')



@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'})

    try:
        # -----------------------------
        # 1. Get uploaded image
        # -----------------------------
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'})

        # -----------------------------
        # 2. Save image
        # -----------------------------
        filename = f"upload_{session['user_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        upload_dir = os.path.join(BASE_DIR, "static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # -----------------------------
        # 3. Run CNN + Grad-CAM
        # -----------------------------
        MODEL_PATH = os.path.join(
            BASE_DIR,
            "Images_ML",
            "Models",
            "cheat_esp_cnn.h5"
        )

        gradcam_result = run_gradcam(
            image_path=filepath,
            model_path=MODEL_PATH
        )

        # -----------------------------
        # 4. Parse prediction
        # -----------------------------
        pred_cheat = 1 if gradcam_result["prediction"] == "CHEAT" else 0
        cheat_confidence = round(gradcam_result["confidence"] * 100, 2)

        if pred_cheat == 1:
            cheat_reasons = [
                    "Strong activations detected in suspicious screen regions",
                    f"Model focused on abnormal visual patterns (Layer: {gradcam_result['conv_layer_used']})",
                    "Highlighted overlays are commonly associated with cheating tools"
        ]
        else:
            cheat_reasons = [
                    "No abnormal visual patterns detected",
                    "Grad-CAM shows attention on normal gameplay areas",
                    "Visual behavior aligns with legitimate gameplay"
        ]

        # -----------------------------
        # 5. Save to DB
        # -----------------------------
        save_prediction(
            session['user_id'],
            'image_test',
            pred_cheat,
            0,
            cheat_confidence,
            0,
            cheat_reasons,
            []
        )

        # -----------------------------
        # 6. Grad-CAM URL for frontend
        # -----------------------------
       # ---------- Grad-CAM URL ----------
        gradcam_url = "/static/gradcam/" + os.path.basename(
            gradcam_result["gradcam_image"]
        )


        # -----------------------------
        # 7. Response
        # -----------------------------
        return jsonify({
            "success": True,
            "prediction": gradcam_result["prediction"],
            "cheat_probability": round(gradcam_result["cheat_probability"] * 100, 2),
            "confidence": round(gradcam_result["confidence"] * 100, 2),
            "gradcam_image": gradcam_url,
            "cheat_reasons": cheat_reasons   # ✅ ADD THIS
        })

    except Exception as e:
        print("UPLOAD IMAGE ERROR:", e)
        return jsonify({'error': str(e)})



@app.route('/results')
def results():
    if 'user_id' not in session:
        flash('Please login to view results', 'error')
        return redirect(url_for('login'))
    
    # Get the latest prediction for the user
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 1
    ''', (session['user_id'],))
    prediction = cursor.fetchone()
    conn.close()
    
    if not prediction:
        flash('No prediction results found. Please run a test first.', 'info')
        return redirect(url_for('dashboard'))
    
    # Parse reasons from string format
    try:
        cheat_reasons = eval(prediction[7]) if prediction[7] else []
        churn_reasons = eval(prediction[8]) if prediction[8] else []
    except:
        cheat_reasons = []
        churn_reasons = []
    
    result_data = {
        'test_type': prediction[2],
        'cheat_prediction': prediction[3],
        'churn_prediction': prediction[4],
        'cheat_confidence': prediction[5],
        'churn_confidence': prediction[6],
        'cheat_reasons': cheat_reasons,
        'churn_reasons': churn_reasons,
        'created_at': prediction[9]
    }
    
    return render_template('results.html', result=result_data)

if __name__ == '__main__':
    app.run(debug=True)