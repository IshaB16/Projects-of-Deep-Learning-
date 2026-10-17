import io, os
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from src.common import load_model_bundle, predict_image

app = Flask(__name__)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model_bundle(os.getenv("MODEL_PATH", "models/pneumonia_model.keras"))
    return _model

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Upload image using the 'file' field."}), 400
    try:
        image = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Uploaded file is not a supported image."}), 400
    try:
        label, probability = predict_image(get_model(), image)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"prediction": label, "p_pneumonia": round(probability, 6),
                    "educational_only": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
