import os
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

# -------------------------------
# PATHS
# -------------------------------
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# -------------------------------
# DATA GENERATORS
# -------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# -------------------------------
# LOAD DATA
# -------------------------------
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# -------------------------------
# CLASS WEIGHTS
# -------------------------------
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
weights = dict(enumerate(weights))

# -------------------------------
# MODEL
# -------------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------------
# COMPILE
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# CALLBACKS
# -------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2)
]

# -------------------------------
# TRAIN
# -------------------------------
print("\n🚀 Training started...\n")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8,
    class_weight=weights,
    callbacks=callbacks
)

# -------------------------------
# VALIDATION REPORT
# -------------------------------
val_gen.reset()

y_true_val = val_gen.classes
y_pred_val = model.predict(val_gen)
y_pred_val_classes = np.argmax(y_pred_val, axis=1)

print("\n📊 Validation Classification Report:\n")
print(classification_report(
    y_true_val,
    y_pred_val_classes,
    target_names=list(val_gen.class_indices.keys())
))

# -------------------------------
# TEST EVALUATION (FINAL)
# -------------------------------
print("\n🧪 Evaluating on TEST DATA...\n")

test_loss, test_acc = model.evaluate(test_gen)

print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# -------------------------------
# TEST REPORT
# -------------------------------
test_gen.reset()

y_true_test = test_gen.classes
y_pred_test = model.predict(test_gen)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)

print("\n📊 Test Classification Report:\n")
print(classification_report(
    y_true_test,
    y_pred_test_classes,
    target_names=list(test_gen.class_indices.keys())
))

# -------------------------------
# SAVE DATA FOR GRAPHS
# -------------------------------
np.save("y_true_test.npy", y_true_test)
np.save("y_pred_test.npy", y_pred_test_classes)

print("\n✅ Saved y_true_test.npy and y_pred_test.npy")

# -------------------------------
# SAVE MODEL + CLASS NAMES
# -------------------------------
model.save("rice_model.keras")

with open("class_names.json", "w") as f:
    json.dump(list(train_gen.class_indices.keys()), f)

print("\n🎉 Training completed successfully!")