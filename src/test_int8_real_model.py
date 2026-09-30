import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from preprocess import load_audio, extract_mfcc


DATASET_PATH = r"C:\Users\chint\Downloads\output-20260926T115017Z-1-001\output\hey_mist"

MODEL_PATH = r"..\models\kws_real_model_int8.tflite"


# --------------------------------------------------
# Load INT8 TFLite model
# --------------------------------------------------

print("Loading INT8 TFLite model...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput details:")
print(input_details)

print("\nOutput details:")
print(output_details)


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

X = []
y = []

test_folders = [
    ("positive_test", 1),
    ("negative_test", 0),
    ("background_test", 0),
]

print("\n==============================")
print("LOADING TEST DATA")
print("==============================")


for folder_name, label in test_folders:

    folder = os.path.join(DATASET_PATH, folder_name)

    print("\nLoading:", folder_name)

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

    print("Samples:", count)


X = np.array(X, dtype=np.float32)

y = np.array(y)

X = X[..., np.newaxis]

print("\nTotal test samples:", len(X))


# --------------------------------------------------
# Get INT8 quantization parameters
# --------------------------------------------------

input_scale, input_zero_point = input_details[0]["quantization"]

output_scale, output_zero_point = output_details[0]["quantization"]

print("\n==============================")
print("QUANTIZATION PARAMETERS")
print("==============================")

print("Input scale:", input_scale)
print("Input zero point:", input_zero_point)

print("Output scale:", output_scale)
print("Output zero point:", output_zero_point)


# --------------------------------------------------
# Run inference
# --------------------------------------------------

predictions = []

print("\nRunning INT8 inference...")

for i in range(len(X)):

    sample = X[i]

    # Convert float MFCC → INT8
    sample_int8 = np.round(
        sample / input_scale + input_zero_point
    ).astype(np.int8)

    sample_int8 = np.expand_dims(sample_int8, axis=0)

    interpreter.set_tensor(
        input_details[0]["index"],
        sample_int8
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    # Convert INT8 output back to float
    output_float = (
        output.astype(np.float32) - output_zero_point
    ) * output_scale

    predicted_class = np.argmax(output_float)

    predictions.append(predicted_class)


y_pred = np.array(predictions)


# --------------------------------------------------
# Results
# --------------------------------------------------

cm = confusion_matrix(y, y_pred)

TN, FP, FN, TP = cm.ravel()


print("\n==============================")
print("INT8 CONFUSION MATRIX")
print("==============================")

print(cm)


print("\n==============================")
print("INT8 CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y,
        y_pred,
        target_names=[
            "Non-Wake-Word",
            "Wake-Word"
        ],
        digits=4
    )
)


# --------------------------------------------------
# SIH metrics
# --------------------------------------------------

accuracy = (TP + TN) / (TP + TN + FP + FN)

false_activation_rate = FP / (FP + TN)

false_negative_rate = FN / (FN + TP)


print("\n==============================")
print("INT8 SIH METRICS")
print("==============================")

print("Accuracy:", accuracy)

print("True Positives :", TP)
print("True Negatives :", TN)

print("False Positives:", FP)
print("False Negatives:", FN)

print("\nFalse Activation Rate:",
      false_activation_rate)

print("False Negative Rate:",
      false_negative_rate)