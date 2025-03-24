import tensorflow as tf
import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=float)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=100, verbose=0)

test_signal = np.array([5.5])
prediction = model.predict(test_signal)
print(f"Signal {test_signal[0]} -> Threat probability: {prediction[0][0]:.2f}")
