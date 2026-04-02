#!/usr/bin/env python3
""" rename """
import pandas as pd


def rename(df):
    """ rename """
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df["Datetime"], unit='s')
    df = df[['Datetime', 'Close']]
    return df
