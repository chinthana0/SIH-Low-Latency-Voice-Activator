import tensorflow as tf

MODEL_PATH = r"..\models\kws_real_model.keras"
OUTPUT_PATH = r"..\models\kws_real_model.tflite"

print("Loading OUR trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Converting to TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("\n==============================")
print("TFLITE CONVERSION COMPLETE")
print("==============================")
print("Saved to:", OUTPUT_PATH)
print("Model size:", len(tflite_model), "bytes")
print("Model size:", len(tflite_model) / 1024, "KB")