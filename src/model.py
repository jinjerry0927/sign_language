"""LSTM classifier for sign-language landmark sequences.

Input  : (B, T=32, 225) - normalized landmark frames
Output : (B, n_classes)  - softmax probabilities over target words
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(seq_len: int, n_features: int, n_classes: int,
                lstm_units: int = 64, dense_units: int = 32,
                dropout: float = 0.4) -> tf.keras.Model:
    inputs = layers.Input(shape=(seq_len, n_features), name="landmarks")
    x = layers.LSTM(lstm_units, return_sequences=False)(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    return models.Model(inputs, outputs, name="ksl_lstm")
