import tensorflow as tf

# Load your trained Keras model
model = tf.keras.models.load_model("model.h5")

# Create a TFLiteConverter object from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Allow TensorFlow Select Ops (useful for LSTM or unsupported ops)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,        # Basic TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS           # Fallback to TF ops if needed
]

# Fix for TensorList ops used in LSTM models
converter._experimental_lower_tensor_list_ops = False

# Optional: Add dynamic range quantization to reduce size (good for edge devices)
# Comment this out if you want to keep the original float model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the converted model to disk
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Successfully converted to 'model.tflite'")
