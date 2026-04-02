#!/usr/bin/env python3
""" slice """


def slice(df):
    """ slice """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
