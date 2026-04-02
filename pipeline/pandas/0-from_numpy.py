#!/usr/bin/env python3
""" from numpy """
import pandas as pd
import string

def from_numpy(array):
    cols = list(string.ascii_uppercase[:array.shape[1]])
    return pd.DataFrame(array, columns=cols)
