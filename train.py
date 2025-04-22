import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Parameters
WINDOW_SIZE = 100  # number of time steps per window
STEP_SIZE = 20     # stride of sliding window

# Load merged dataset (from accel + gyro with labels)
print("[INFO] Loading data...")
data = pd.read_csv("merged_edge_impulse.csv")

# Features and labels
FEATURE_COLUMNS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
LABEL_COLUMN = 'label'

# Encode labels
print("[INFO] Encoding labels...")
le = LabelEncoder()
data[LABEL_COLUMN] = le.fit_transform(data[LABEL_COLUMN])
label_classes = le.classes_
num_classes = len(label_classes)

# Save label mapping
np.save("label_map.npy", label_classes)

# Create sliding windows
print("[INFO] Creating windowed data...")
def create_windows(df, window_size, step_size, feature_cols, label_col):
    X, y = [], []
    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i:i + window_size]
        label = window[label_col].mode()[0]  # majority label
        X.append(window[feature_cols].values)
        y.append(label)
    return np.array(X), np.array(y)

X, y = create_windows(data, WINDOW_SIZE, STEP_SIZE, FEATURE_COLUMNS, LABEL_COLUMN)

# One-hot encode labels
y = to_categorical(y, num_classes=num_classes)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"[INFO] Total samples: {len(X)}, Time steps: {WINDOW_SIZE}, Features: {len(FEATURE_COLUMNS)}")

# Build LSTM model
print("[INFO] Building model...")
model = Sequential([
    LSTM(64, input_shape=(WINDOW_SIZE, len(FEATURE_COLUMNS))),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
print("[INFO] Training model...")
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=64)

# Export model in SavedModel format for TFLite conversion
print("[INFO] Exporting model to SavedModel format...")
model.export("saved_model")

# Convert to TensorFlow Lite
print("[INFO] Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional quantization
tflite_model = converter.convert()

with open("activity_model.tflite", "wb") as f:
    f.write(tflite_model)

print("[DONE] Saved TFLite model to activity_model.tflite")
