import os
import numpy as np
import tensorflow as tf

from preprocess import load_audio, extract_mfcc

MODEL_PATH = r"..\models\kws_real_model.keras"
OUTPUT_PATH = r"..\models\kws_real_model_int8.tflite"

DATASET_PATH = r"C:\Users\chint\Downloads\output-20260926T115017Z-1-001\output\hey_mist"


print("Loading OUR trained model...")
model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Collect real MFCC samples for calibration
# --------------------------------------------------

samples = []

folders = [
    "positive_train",
    "negative_train",
    "background_train"
]

print("\nCollecting real representative samples...")

for folder_name in folders:

    folder = os.path.join(DATASET_PATH, folder_name)

    print("Reading:", folder_name)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]

    # Only use up to 100 samples from each folder
    files = files[:100]

    for filename in files:

        file_path = os.path.join(folder, filename)

        try:
            audio = load_audio(file_path)
            mfcc = extract_mfcc(audio)

            # Add channel dimension
            mfcc = mfcc[..., np.newaxis]

            samples.append(mfcc.astype(np.float32))

        except Exception as e:
            print("Error:", filename, e)


print("\nRepresentative samples:", len(samples))


# --------------------------------------------------
# Representative dataset generator
# --------------------------------------------------

def representative_dataset():

    for sample in samples:

        sample = np.expand_dims(sample, axis=0)

        yield [sample]


# --------------------------------------------------
# INT8 conversion
# --------------------------------------------------

print("\nStarting proper INT8 quantization...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


tflite_model = converter.convert()


# --------------------------------------------------
# Save
# --------------------------------------------------

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)


print("\n==============================")
print("PROPER INT8 QUANTIZATION COMPLETE")
print("==============================")

print("Saved to:", OUTPUT_PATH)

print("Model size:", len(tflite_model), "bytes")

print("Model size:", len(tflite_model) / 1024, "KB")