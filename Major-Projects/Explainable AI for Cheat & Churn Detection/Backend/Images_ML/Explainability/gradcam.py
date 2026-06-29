import tensorflow as tf
import numpy as np
import cv2
import os

# =============================
# CONFIG
# =============================
IMG_SIZE = 150
CHEAT_THRESHOLD = 0.5          # Sigmoid threshold
LAST_CONV_LAYER_NAME = None    # Auto-detect last Conv2D layer


# =============================
# BASE PATH
# =============================
# Path resolution:
# Backend/Images_ML/Explainability/gradcam.py → Backend/
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

STATIC_GRADCAM_DIR = os.path.join(PROJECT_ROOT, "static", "gradcam")


# =============================
# LOAD MODEL
# =============================
def load_model(model_path):
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    # FIX: prevent loading incompatible optimizer
    model = tf.keras.models.load_model(model_path, compile=False)

    model.trainable = False
    return model

# =============================
# PREPROCESS IMAGE
# =============================
def preprocess_image(img_path):
    if not os.path.isabs(img_path):
        img_path = os.path.join(PROJECT_ROOT, img_path)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ Image not found: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ OpenCV failed to read image: {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img, img_path


# =============================
# FIND LAST CONV LAYER
# =============================
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise RuntimeError("❌ No Conv2D layer found for Grad-CAM")


# =============================
# GRAD-CAM CORE (SIGMOID SAFE)
# =============================
def generate_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        cheat_score = predictions[:, 0]   # Sigmoid output → P(CHEAT)

    grads = tape.gradient(cheat_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), float(predictions[0][0])


# =============================
# OVERLAY HEATMAP
# =============================
def overlay_heatmap(original_img_path, heatmap, output_path):
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, overlay)


# =============================
# MAIN API FUNCTION (FINAL)
# =============================
def run_gradcam(image_path, model_path):
    """
    Runs CNN prediction + Grad-CAM and saves overlay image.

    Returns:
        {
            prediction: "CHEAT" | "NON-CHEAT",
            cheat_probability: float (0–1),
            confidence: float (0–1),
            threshold: float,
            gradcam_image: absolute path,
            conv_layer_used: str
        }
    """

    os.makedirs(STATIC_GRADCAM_DIR, exist_ok=True)

    model = load_model(model_path)
    img_array, resolved_img_path = preprocess_image(image_path)

    conv_layer = LAST_CONV_LAYER_NAME or get_last_conv_layer(model)
    heatmap, cheat_prob = generate_gradcam(model, img_array, conv_layer)

    # ---------- SAFETY ----------
    if cheat_prob is None or np.isnan(cheat_prob):
        cheat_prob = 0.0

    cheat_prob = float(np.clip(cheat_prob, 0.0, 1.0))

    prediction = "CHEAT" if cheat_prob >= CHEAT_THRESHOLD else "NON-CHEAT"
    confidence = max(cheat_prob, 1.0 - cheat_prob)

    # ---------- SAVE IMAGE ----------
    image_name = os.path.splitext(os.path.basename(resolved_img_path))[0]
    gradcam_filename = f"gradcam_{image_name}.jpg"
    output_img_path = os.path.join(STATIC_GRADCAM_DIR, gradcam_filename)

    overlay_heatmap(resolved_img_path, heatmap, output_img_path)

    # ---------- RETURN ----------
    return {
        "prediction": prediction,
        "cheat_probability": cheat_prob,   # RAW (0–1)
        "confidence": confidence,           # RAW (0–1)
        "threshold": CHEAT_THRESHOLD,
        "gradcam_image": output_img_path.replace("\\", "/"),
        "conv_layer_used": conv_layer
    }
