#!/usr/bin/env python3
"""
Preprocesses raw Bitcoin transaction data for time-series forecasting.
"""
import sys
import pandas as pd
import numpy as np


def preprocess_data(file_path):
    """
    Reads, preprocesses, resamples, and saves the BTC dataset.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Convert Unix timestamps to datetime objects and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    print("Handling missing values and resampling to hourly intervals...")
    # Missing data in this dataset implies no trades occurred.
    # Forward-fill prices, and fill volumes with 0.
    df['Close'] = df['Close'].ffill()
    df['Open'] = df['Open'].ffill()
    df['High'] = df['High'].ffill()
    df['Low'] = df['Low'].ffill()
    df['Weighted_Price'] = df['Weighted_Price'].ffill()
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    # Drop any remaining NaNs at the beginning of the dataset
    df.dropna(inplace=True)

    # Resample to hourly data to reduce noise and align with the 24h window
    df_hourly = df.resample('H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    })

    # Forward fill any gaps introduced by resampling
    df_hourly.ffill(inplace=True)
    df_hourly.dropna(inplace=True)

    print("Scaling the data...")
    # Scale features to be between 0 and 1 using basic Min-Max scaling
    # We save min and max values to allow inverse transformation later
    min_val = df_hourly.min()
    max_val = df_hourly.max()
    scaled_data = (df_hourly - min_val) / (max_val - min_val)

    print("Splitting data into train, validation, and test sets...")
    n = len(scaled_data)
    train_df = scaled_data.iloc[0:int(n * 0.7)].values
    val_df = scaled_data.iloc[int(n * 0.7):int(n * 0.9)].values
    test_df = scaled_data.iloc[int(n * 0.9):].values

    # Save to compressed numpy file
    out_file = 'preprocessed_data.npz'
    np.savez(out_file, train=train_df, val=val_df, test=test_df)
    print(f"Preprocessed data successfully saved to {out_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./preprocess_data.py <path_to_raw_csv>")
        sys.exit(1)

    raw_file = sys.argv[1]
    preprocess_data(raw_file)
