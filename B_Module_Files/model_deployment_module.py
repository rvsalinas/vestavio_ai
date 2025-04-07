import os
import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify
from typing import Dict, Any, List

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------------
# CONFIG / ENV
# -------------------------------------------------------------------------
MODEL_DIR = "/Users/robertsalinas/Desktop/energy_optimization_project/C_Model_Files"

# Classification Model
CLASSIFICATION_MODEL_PATH = os.path.join(
    MODEL_DIR, "classification_model", "classification_model.joblib"
)

# Energy Efficiency Model
ENERGY_MODEL_PATH = os.path.join(
    MODEL_DIR, "energy_efficiency_optimization_model", "energy_efficiency_optimization_model.joblib"
)

# Dictionary to store loaded models + expected_features
LOADED_MODELS: Dict[str, Any] = {}

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------
def load_model(model_path: str):
    """
    Loads a scikit-learn joblib model from disk.
    Returns the loaded model or None on failure.
    """
    if not os.path.exists(model_path):
        logging.error(f"[ERROR] Model file not found: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        logging.info(f"[INFO] Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"[ERROR] Could not load model from {model_path}: {e}", exc_info=True)
        return None


def align_features(input_data: Dict[str, float], expected_features: List[str]) -> np.ndarray:
    """
    Align and pad/truncate feature data to the model's expected order/length.
    Missing features are filled with 0.0; extra features are ignored.
    Returns a 2D numpy array for model.predict().
    """
    aligned_values = []
    for feat in expected_features:
        val = input_data.get(feat, 0.0)  # default to 0.0 if not provided
        aligned_values.append(val)

    # Convert to shape (1, num_features)
    return np.array([aligned_values], dtype=float)

# -------------------------------------------------------------------------
# FLASK ROUTES
# -------------------------------------------------------------------------
@app.route("/available_models", methods=["GET"])
def available_models():
    """
    Lists all loaded models.
    """
    return jsonify({"loaded_models": list(LOADED_MODELS.keys())}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST endpoint for predictions. JSON format:
      {
        "model_name": "<model_key>",
        "features": {
            "<feat1>": <val1>,
            "<feat2>": <val2>,
            ...
        }
      }
    """
    data = request.get_json() or {}
    model_name = data.get("model_name", "")
    input_features = data.get("features", {})

    if model_name not in LOADED_MODELS:
        msg = f"No such model: '{model_name}'. Loaded models: {list(LOADED_MODELS.keys())}"
        logging.warning(msg)
        return jsonify({"error": msg}), 400

    model_info = LOADED_MODELS[model_name]
    model_obj = model_info["model"]
    expected_feats = model_info["expected_features"]

    # Align input
    X_aligned = align_features(input_features, expected_feats)

    # Predict
    try:
        preds = model_obj.predict(X_aligned)
        return jsonify({"predictions": preds.tolist()}), 200
    except Exception as e:
        msg = f"Prediction error: {e}"
        logging.error(msg, exc_info=True)
        return jsonify({"error": msg}), 500

# -------------------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------------------
def load_all_models():
    """
    Loads each model into LOADED_MODELS with 'model' and 'expected_features'.
    Adjust feature lists to match your training pipeline for each.
    """

    # Classification Model (Titanic example)
    classification_model = load_model(CLASSIFICATION_MODEL_PATH)
    if classification_model:
        LOADED_MODELS["classification"] = {
            "model": classification_model,
            "expected_features": [
                "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
            ]
        }

    # Energy Efficiency Model
    energy_model = load_model(ENERGY_MODEL_PATH)
    if energy_model:
        # Suppose the model expects 8 features, but you only use 4 in real usage:
        energy_feats = [
            "T_outdoor", "T_indoor", "Occupancy", "HVAC_setting",
            "extra1", "extra2", "extra3", "extra4"
        ]
        LOADED_MODELS["energy_efficiency"] = {
            "model": energy_model,
            "expected_features": energy_feats
        }


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("Starting Model Deployment module...")
    load_all_models()
    logging.info(f"Models loaded: {list(LOADED_MODELS.keys())}")

    # Start Flask dev server
    app.run(host="0.0.0.0", port=8080, debug=False)