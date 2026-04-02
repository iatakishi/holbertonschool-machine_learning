#!/usr/bin/env python3
""" from numpy """
import pandas as pd


def from_numpy(array):
    """ from numpy """
    cols = [chr(i) for i in range(65, 65 + array.shape[1])]
    return pd.DataFrame(array, columns=cols)
