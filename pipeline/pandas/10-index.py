#!/usr/bin/env python3
""" index """


def index(df):
    """ index """
    df.set_index('Timestamp', inplace=True)
    return df
