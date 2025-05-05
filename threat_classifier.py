import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import keras_tuner as kt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
# ... rest of your code ...
plt.savefig('accuracy_loss_plot.png')
# Remove plt.show() since it fails without a displ
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
    model.add(tf.keras.Input(shape=(2,)))  # Input shape for 2 features
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units_1', min_value=1, max_value=16, step=1),
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_1', min_value=1e-5, max_value=1e-2, sampling='log'))
    ))
    model.add(tf.keras.layers.BatchNormalization())  # Added batch normalization
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units_2', min_value=1, max_value=16, step=1),
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_2', min_value=1e-5, max_value=1e-2, sampling='log'))
    ))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='log')  # Extended learning rate range
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Set up Keras Tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=100,  # Increased max epochs
    factor=3,
    directory='my_dir',
    project_name='threat_classifier'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=1)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]

# Rebuild the model with the best hyperparameters
best_model = build_model(best_hps)

# Evaluate the rebuilt model on the test set (before retraining)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nInitial Test Accuracy (before retraining): {test_accuracy:.2f}")
print(f"Initial Test Loss (before retraining): {test_loss:.2f}")

# Custom callback to print test accuracy and loss at the end of each epoch
class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        super(TestCallback, self).__init__()
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        x_test, y_test = self.test_data
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"\nEpoch {epoch + 1} - Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")

# Train the rebuilt model with early stopping and test callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
test_callback = TestCallback((X_test, y_test))
history = best_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping, test_callback], verbose=1)

# Print average validation accuracy and loss
avg_val_accuracy = np.mean(history.history['val_accuracy'])
avg_val_loss = np.mean(history.history['val_loss'])
print(f"\nAverage Validation Accuracy: {avg_val_accuracy:.2f}")
print(f"Average Validation Loss: {avg_val_loss:.2f}")

# Evaluate the model on the test set (after retraining)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy (after retraining): {test_accuracy:.2f}")
print(f"Test Loss (after retraining): {test_loss:.2f}")

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
test_signal = np.array([[5.5, 5.5**2]])  # Raw signal and squared value
nn_pred = best_model.predict(test_signal, verbose=0)
lr_pred = lr.predict(test_signal)
print(f"\nNeural Network - Signal 5.5 -> Threat probability: {nn_pred[0][0]:.2f}")
print(f"Logistic Regression - Signal 5.5 -> Threat probability: {lr_pred[0]:.2f}")