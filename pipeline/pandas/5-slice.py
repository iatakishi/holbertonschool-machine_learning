#!/usr/bin/env python3
""" slice """


def slice(df):
    """ slice """
    return df[['High', 'Low', 'Close', 'Volume_BTC']].iloc[::60]
