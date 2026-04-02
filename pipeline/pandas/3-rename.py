#!/usr/bin/env python3
""" rename """
import pandas as pd


def rename(df):
    """ rename """
    df['Timestamp'] = df['Datetime']
    df['Datetime'] = pd.to_datetime(df["Datetime"], unit='s')
    df = df[['Datetime', 'Close']]
    return df
