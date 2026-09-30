import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixels from 0-255 to 0-1 and add the channel dimension.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

# A small CNN is accurate and still easy to understand.
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nTraining...")
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

print("\nEvaluating...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy * 100:.2f}%")

model.save(MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
