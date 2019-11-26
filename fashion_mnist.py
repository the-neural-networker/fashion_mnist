import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist.load_data()
X_train, y_train, X_test, y_test = fashion_mnist[0][0], fashion_mnist[0][1], fashion_mnist[1][0], fashion_mnist[1][1]

X_train, X_test = X_train / 255.0, X_test / 255.0

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

optimizer = keras.optimizers.SGD(momentum=0.9, nesterov=True)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 2)
plt.show()

model.save("fashion_mnist.h5")