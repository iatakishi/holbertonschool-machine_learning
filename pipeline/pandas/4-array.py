#!/usr/bin/env python3
""" array """
import pandas as pd


def array(df):
    """ array """
    df = df[['High', 'Close']]
    df = df.bottom(10)
    df = df.to_numpy
    return df
