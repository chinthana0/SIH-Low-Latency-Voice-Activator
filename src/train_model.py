import os
import numpy as np

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

from dataset import load_dataset


# -----------------------------
# 1. Load real dataset
# -----------------------------

DATASET_PATH = r"C:\Users\chint\Downloads\output-20260926T115017Z-1-001\output\hey_mist"

X, y = load_dataset(DATASET_PATH)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)


# -----------------------------
# 2. Add channel dimension
# -----------------------------

X = X[..., np.newaxis]

print("Input shape:", X.shape)


# -----------------------------
# 3. Train/test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# 4. Build small CNN
# -----------------------------

model = models.Sequential([
    layers.Input(shape=(13, 32, 1)),

    layers.Conv2D(
        8,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        16,
        (3, 3),
        activation="relu"
    ),

    layers.Flatten(),

    layers.Dense(
        16,
        activation="relu"
    ),

    layers.Dense(
        2,
        activation="softmax"
    )
])


# -----------------------------
# 5. Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# 6. Train
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)


# -----------------------------
# 7. Evaluate
# -----------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n==============================")
print("FINAL TEST ACCURACY:", test_accuracy)
print("==============================")


# -----------------------------
# 8. Save model
# -----------------------------

os.makedirs("../models", exist_ok=True)

model.save(
    "../models/kws_real_model.keras"
)

print("Model saved successfully!")