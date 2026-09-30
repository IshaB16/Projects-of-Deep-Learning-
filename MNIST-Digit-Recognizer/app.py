from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "mnist_model.keras")

try:
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
except Exception as exc:
    model = None
    MODEL_ERROR = str(exc)


def preprocess_image(image_bytes):
    """Convert a canvas PNG into the 28x28 format expected by MNIST."""
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Canvas is black background with white drawing.
    # Crop to the drawn digit so the model is less sensitive to position.
    arr = np.array(image)
    coords = np.argwhere(arr > 20)

    if coords.size == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = image.crop((x0, y0, x1, y1))

    # Add padding and resize while preserving aspect ratio.
    w, h = cropped.size
    scale = 20 / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cropped.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    left = (28 - resized.width) // 2
    top = (28 - resized.height) // 2
    canvas.paste(resized, (left, top))

    data = np.array(canvas, dtype=np.float32) / 255.0
    return data.reshape(1, 28, 28, 1)


@app.route("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({
            "error": "Model not found. Run `python train.py` first.",
            "details": MODEL_ERROR
        }), 500

    if "image" not in request.files:
        return jsonify({"error": "No image was uploaded."}), 400

    image_bytes = request.files["image"].read()
    processed = preprocess_image(image_bytes)

    if processed is None:
        return jsonify({"error": "Please draw a digit first."}), 400

    probabilities = model.predict(processed, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])

    return jsonify({
        "digit": digit,
        "confidence": round(confidence * 100, 2),
        "probabilities": [round(float(p) * 100, 2) for p in probabilities]
    })


if __name__ == "__main__":
    app.run(debug=True)
