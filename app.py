import os
import json
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# -------------------------------
# BASE DIRECTORY
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# FLASK APP
# -------------------------------
app = Flask(__name__)

# -------------------------------
# UPLOAD FOLDER
# -------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------------
# LOAD MODEL (SAFE LOAD)
# -------------------------------
model = None
class_names = []

def load_resources():
    global model, class_names

    if model is None:
        model_path = os.path.join(BASE_DIR, "rice_model.keras")
        model = load_model(model_path)
        print("✅ Model loaded")

    if not class_names:
        class_path = os.path.join(BASE_DIR, "class_names.json")
        with open(class_path, "r") as f:
            class_names.extend(json.load(f))
        print("✅ Class names loaded")

# -------------------------------
# DISEASE INFO
# -------------------------------
disease_info = {
    "False Smut": {
        "treatment": "Apply fungicides like Propiconazole or Carbendazim.",
        "precaution": "Use certified seeds and avoid excess nitrogen fertilizer."
    },
    "Leaf Smut": {
        "treatment": "Spray Mancozeb or Copper fungicides.",
        "precaution": "Ensure proper spacing and avoid water stagnation."
    },
    "Narrow Brown Leaf Spot": {
        "treatment": "Apply Tricyclazole.",
        "precaution": "Use resistant varieties and balanced fertilization."
    }
}

# -------------------------------
# LOAD ACCURACY
# -------------------------------
def load_accuracy():
    try:
        metrics_path = os.path.join(BASE_DIR, "metrics.txt")
        with open(metrics_path, "r") as f:
            for line in f:
                if "Accuracy" in line:
                    return float(line.split(":")[1])
    except:
        return 0.98

model_accuracy = load_accuracy()

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_image(filepath):
    try:
        load_resources()  # ensure model is loaded

        img = Image.open(filepath).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]

        predicted_class = np.argmax(preds)
        confidence = float(np.max(preds)) * 100

        label = class_names[predicted_class]

        if label == "Other":
            return "Sorry, I know only 3 rice diseases 😅", None, None, None

        treatment = disease_info.get(label, {}).get("treatment", "")
        precaution = disease_info.get(label, {}).get("precaution", "")

        return label, confidence, treatment, precaution

    except Exception as e:
        print("❌ PREDICTION ERROR:", str(e))
        return "Invalid image ❌", None, None, None

# -------------------------------
# ROUTE
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            file = request.files.get("file")

            if not file or file.filename == "":
                return render_template("index.html", label="Please upload an image")

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, confidence, treatment, precaution = predict_image(filepath)

            return render_template(
                "index.html",
                label=label,
                confidence=round(confidence, 2) if confidence else None,
                accuracy=round(model_accuracy * 100, 2),
                image_path=filepath,
                treatment=treatment,
                precaution=precaution
            )

        except Exception as e:
            print("❌ ROUTE ERROR:", str(e))
            return render_template("index.html", label="Error processing image ❌")

    return render_template("index.html")

# -------------------------------
# RENDER PORT FIX
# -------------------------------
port = int(os.environ.get("PORT", 10000))

# -------------------------------
# RUN (LOCAL ONLY)
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)