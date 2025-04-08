import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import keras_tuner as kt
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic data
n_samples = 500  # Increased dataset size
signals = np.random.rand(n_samples) * 10  # Random signals between 0 and 10
labels = (signals > 5).astype(int)  # Threat if signal > 5

# Add new feature: square of the signal values
signals_squared = signals ** 2  # Square of each signal value
X = np.column_stack((signals, signals_squared))  # Combine original signals and squared signals
y = labels

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the model-building function for Keras Tuner
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2,)))  # Updated input shape for 2 features
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units_1', min_value=1, max_value=16, step=1),
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_1', min_value=1e-4, max_value=1e-1, sampling='log'))  # Increased regularization
    ))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.3, step=0.1)))
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units_2', min_value=1, max_value=16, step=1),
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_2', min_value=1e-4, max_value=1e-1, sampling='log'))  # Increased regularization
    ))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.3, step=0.1)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')  # Narrowed learning rate range
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Set up Keras Tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='my_dir',
    project_name='threat_classifier'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=1)

# Get the best model
best_model = tuner.get_best_models()[0]  # Fixed: Removed num_models=1
best_hps = tuner.get_best_hyperparameters()[0]  # Fixed: Removed num_models=1

# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.2f}")
print(f"Test Loss: {test_loss:.2f}")

# Train the best model again to get history for plotting, with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
history = best_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('accuracy_loss_plot.png')
plt.show()

# Save the best model
best_model.save('threat_classifier_model_v2.keras')

# Logistic Regression baseline
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"\nLogistic Regression Test Accuracy: {lr_accuracy:.2f}")

# Predict for a new signal
test_signal = np.array([[5.5, 5.5**2]])  # Include the squared value for the test signal
nn_pred = best_model.predict(test_signal, verbose=0)
lr_pred = lr.predict(test_signal)
print(f"\nNeural Network - Signal 5.5 -> Threat probability: {nn_pred[0][0]:.2f}")
print(f"Logistic Regression - Signal 5.5 -> Threat probability: {lr_pred[0]:.2f}")