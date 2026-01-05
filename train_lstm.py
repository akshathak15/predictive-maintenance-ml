import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("real_time_sensor_data.csv")

# Prepare data
X = df.drop(columns=["failure"]).values
y = df["failure"].values

# Reshape for LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Define LSTM model
model = Sequential([
    LSTM(50, activation="relu", input_shape=(1, X.shape[2])),
    Dense(1, activation="sigmoid")
])

# Compile & train
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=10, batch_size=16)

# Save model
model.save("models/lstm_model.h5")
print("LSTM Model Trained and Saved!")
