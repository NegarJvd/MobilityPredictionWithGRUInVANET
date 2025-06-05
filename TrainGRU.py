import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
from keras.layers import Dropout


def load_data(npz_path, seq_len=5):
    data = np.load(npz_path)['data']

    # Normalize (min-max scaling between 0 and 1)
    min_vals = data.reshape(-1, 2).min(axis=0)
    max_vals = data.reshape(-1, 2).max(axis=0)
    data = (data - min_vals) / (max_vals - min_vals + 1e-8)

    X = data[:, :-1, :]
    Y = data[:, -1, :]
    return X, Y

def build_gru_model(input_shape):
    model = Sequential([
        GRU(64, return_sequences=False, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(2)  # Predicting x, y
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return model

def train_and_save(X, Y, model_path='models/gru_position_predictor.h5'):
    model = Sequential()
    model.add(GRU(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2))  # خروجی: مختصات x و y

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=30, batch_size=64, validation_split=0.1, verbose=1)

    model.save(model_path)
    print(f"Model saved to '{model_path}'")

if __name__ == '__main__':
    npz_file = './data/vehicle_sequences.npz'
    model_file = './models/gru_position_predictor.h5'

    print("Loading data...")
    X, Y = load_data(npz_file)

    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    print("Training model...")
    train_and_save(X, Y, model_file)
