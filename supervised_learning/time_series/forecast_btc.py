#!/usr/bin/env python3
"""
Creates, trains, and validates a Keras model for BTC price forecasting.
"""
import numpy as np
import tensorflow as tf


def forecast_btc():
    """
    Builds and trains an RNN model to predict BTC prices.
    """
    # Load preprocessed data
    try:
        data = np.load('preprocessed_data.npz')
    except FileNotFoundError:
        print("Error: preprocessed_data.npz not found.")
        print("Please run preprocess_data.py first.")
        return

    train_data = data['train']
    val_data = data['val']

    # Configuration
    # 24 steps (hours) of input data
    input_width = 24
    # The 'Close' column is at index 3 based on our aggregation dictionary
    target_idx = 3
    batch_size = 64

    # Create tf.data.Datasets
    # We use the past 24 hours to predict the 25th hour's close price
    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=train_data[:-1],
        targets=train_data[input_width:, target_idx],
        sequence_length=input_width,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=val_data[:-1],
        targets=val_data[input_width:, target_idx],
        sequence_length=input_width,
        batch_size=batch_size,
        shuffle=False
    )

    # Define the RNN (LSTM) Architecture
    model = tf.keras.Sequential([
        # LSTM layer to process the 24-hour sequence
        tf.keras.layers.LSTM(64, return_sequences=False,
                             input_shape=(input_width, train_data.shape[1])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        # Output layer for the single continuous prediction (Close price)
        tf.keras.layers.Dense(1)
    ])

    # Compile with Mean Squared Error (MSE) cost function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    model.summary()

    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[early_stopping]
    )

    # Save the trained model
    model.save('btc_forecast_model.h5')
    print("Model saved as btc_forecast_model.h5")


if __name__ == "__main__":
    forecast_btc()
