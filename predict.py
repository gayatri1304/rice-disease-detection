def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    confidence = float(np.max(preds)) * 100
    predicted_class = np.argmax(preds)

    label = class_names[predicted_class]

    # 🎯 HANDLE "OTHER" CLASS
    if label == "Other":
        return "Unpredictable disease", None, None, None

    # ✅ Normal case
    treatment = disease_info.get(label, {}).get("treatment", "")
    precaution = disease_info.get(label, {}).get("precaution", "")

    return label, confidence, treatment, precaution