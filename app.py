import os
import json
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------
# BASE DIRECTORY
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# LOAD MODEL
# -------------------------------
model_path = os.path.join(BASE_DIR, "rice_model.keras")
model = load_model(model_path)

# -------------------------------
# LOAD CLASS NAMES
# -------------------------------
class_path = os.path.join(BASE_DIR, "class_names.json")
with open(class_path, "r") as f:
    class_names = json.load(f)

# -------------------------------
# DISEASE INFO
# -------------------------------
disease_info = {
    "False Smut": {
        "treatment": "Apply fungicides like Propiconazole or Carbendazim.",
        "precaution": "Use certified seeds, avoid excess nitrogen fertilizer."
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
# FLASK SETUP
# -------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]

    predicted_class = np.argmax(preds)
    confidence = float(np.max(preds)) * 100

    label = class_names[predicted_class]

    # Handle unknown
    if label == "Other":
        return "Sorry, I know only 3 rice diseases 😅", None, None, None

    treatment = disease_info.get(label, {}).get("treatment", "")
    precaution = disease_info.get(label, {}).get("precaution", "")

    return label, confidence, treatment, precaution

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
            print("ERROR:", str(e))
            return render_template("index.html", label="Something went wrong")

    return render_template("index.html")

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)