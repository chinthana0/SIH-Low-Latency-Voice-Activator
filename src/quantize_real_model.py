import tensorflow as tf
import numpy as np

MODEL_PATH = r"..\models\kws_real_model.keras"
OUTPUT_PATH = r"..\models\kws_real_model_int8.tflite"

print("Loading OUR trained model...")

model = tf.keras.models.load_model(MODEL_PATH)


# Representative dataset
def representative_dataset():
    for _ in range(100):
        sample = np.random.randn(1, 13, 32, 1).astype(np.float32)
        yield [sample]


print("Starting INT8 quantization...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()


with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)


print("\n==============================")
print("INT8 QUANTIZATION COMPLETE")
print("==============================")
print("Saved to:", OUTPUT_PATH)
print("Model size:", len(tflite_model), "bytes")
print("Model size:", len(tflite_model) / 1024, "KB")