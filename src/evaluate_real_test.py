import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import load_audio, extract_mfcc

DATASET_PATH = r"C:\Users\chint\Downloads\output-20260926T115017Z-1-001\output\hey_mist"
MODEL_PATH = r"..\models\kws_real_model.keras"

# Load OUR trained model
model = tf.keras.models.load_model(MODEL_PATH)

X = []
y = []

# Separate TEST folders
test_folders = [
    ("positive_test", 1),
    ("negative_test", 0),
    ("background_test", 0),
]

for folder_name, label in test_folders:

    folder = os.path.join(DATASET_PATH, folder_name)

    print(f"\nLoading: {folder_name}")

    if not os.path.exists(folder):
        print("Folder not found!")
        continue

    count = 0

    for filename in os.listdir(folder):

        if filename.lower().endswith(".wav"):

            file_path = os.path.join(folder, filename)

            try:
                audio = load_audio(file_path)
                mfcc = extract_mfcc(audio)

                X.append(mfcc)
                y.append(label)

                count += 1

            except Exception as e:
                print("Error:", filename, e)

    print("Samples loaded:", count)


# Convert to numpy
X = np.array(X)
y = np.array(y)

# Add channel dimension
X = X[..., np.newaxis]

print("\n==============================")
print("TEST DATA")
print("==============================")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Positive samples:", np.sum(y == 1))
print("Negative samples:", np.sum(y == 0))


# Predict
predictions = model.predict(X, verbose=1)

y_pred = np.argmax(predictions, axis=1)


# Confusion matrix
cm = confusion_matrix(y, y_pred)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)


# Classification report
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y,
        y_pred,
        target_names=["Non-Wake-Word", "Wake-Word"],
        digits=4
    )
)


# Calculate values manually
TN, FP, FN, TP = cm.ravel()

print("\n==============================")
print("IMPORTANT SIH METRICS")
print("==============================")

print("True Positives :", TP)
print("True Negatives :", TN)
print("False Positives:", FP)
print("False Negatives:", FN)

false_activation_rate = FP / (FP + TN)
false_negative_rate = FN / (FN + TP)

print("\nFalse Activation Rate :", false_activation_rate)
print("False Negative Rate   :", false_negative_rate)