# Time Series Forecasting: Bitcoin Price

This project aims to forecast the value of Bitcoin (BTC) using historical data from the Coinbase and Bitstamp exchanges. The model leverages a Recurrent Neural Network (RNN) built with Keras/TensorFlow.

## Task Overview

We use the past 24 hours of BTC data to predict the close price of BTC for the following hour. 

## Files
* `preprocess_data.py`: A script to clean, resample, and scale the raw 1-minute interval dataset into hourly windows. It saves the preprocessed splits into a `.npz` file.
* `forecast_btc.py`: A script that loads the preprocessed data, creates a `tf.data.Dataset`, and builds, trains, and validates an LSTM-based model using Mean Squared Error (MSE).

## Requirements
* `numpy`
* `pandas`
* `tensorflow`
* `scikit-learn`

## Usage
1. First, run the preprocessing script on your raw CSV file (e.g., `coinbase.csv`):
   ```bash
   ./preprocess_data.py bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv