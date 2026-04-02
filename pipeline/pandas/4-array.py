#!/usr/bin/env python3
""" array """


def array(df):
    """ array """
    df = df[['High', 'Close']]
    df = df.tail(10)
    df = df.to_numpy()
    return df
