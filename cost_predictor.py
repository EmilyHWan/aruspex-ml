import tensorflow as tf
import numpy as np
X = np.array([1, 2, 3, 4, 5], dtype=float)  # Months
y = np.array([100, 200, 300, 400, 500], dtype=float)  # Costs
model = tf.keras.Sequential([tf.keras.Input(shape=(1,)), tf.keras.layers.Dense(1)])
model.compile(optimizer='sgd', loss='mse')
model.fit(X, y, epochs=100, verbose=0)
print(f"Month 6 cost: {model.predict(np.array([6]))[0][0]:.2f}")
