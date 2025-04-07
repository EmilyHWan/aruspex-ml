import numpy as np
import os

# Generate 10 random signals between 0 and 10
print("Starting signal generation...")
signals = np.random.rand(10) * 10

# Fake labels: signal > 5 = threat (1), else no threat (0)
print(f"Signals: {signals}")
labels = (signals > 5).astype(int)

# Pair signals and labels
print(f"Labels: {labels}")
data = np.column_stack((signals, labels))

# Save to file
print(f"Data to save: {data}")
np.save("signal_data.npy", data)
print(f"Saved to: {os.path.abspath('signal_data.npy')}")
print("Done!")