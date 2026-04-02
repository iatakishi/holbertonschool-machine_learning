#!/usr/bin/env python3
""" from file """
import pandas as pd


def from_file(filename, delimiter):
    """ loads data from file """
    return pd.read_csv(filename, sep=delimiter)
