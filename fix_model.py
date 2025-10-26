import tensorflow as tf
from tensorflow import keras
from src.model_training.lstm_model import build_lstm_model
import os

# Create clean model with same architecture
print("Building model architecture...")
model = build_lstm_model(input_shape=(50, 21))

# Compile it
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Save in H5 format
save_path = 'models/lstm/clean_model.h5'
model.save(save_path, save_format='h5')
print(f"✅ Clean model saved to {save_path}")

# Verify it loads
test_model = keras.models.load_model(save_path)
print("✅ Model verified - loads successfully!")

