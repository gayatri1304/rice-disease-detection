from PIL import Image

def predict_image(img_path):
    try:
        # 🔥 Load image safely using PIL
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))

        # Convert to array
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        preds = model.predict(img_array)[0]
        confidence = float(np.max(preds)) * 100
        predicted_class = np.argmax(preds)

        label = class_names[predicted_class]

        # 🎯 HANDLE "OTHER"
        if label == "Other":
            return "Sorry, I know only 3 rice diseases 😅", None, None, None

        # ✅ Normal case
        treatment = disease_info.get(label, {}).get("treatment", "")
        precaution = disease_info.get(label, {}).get("precaution", "")

        return label, confidence, treatment, precaution

    except Exception as e:
        print("PREDICTION ERROR:", str(e))
        return "Invalid image or unsupported file ❌", None, None, None