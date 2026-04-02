#!/usr/bin/env python3
""" prune """


def prune(df):
    """ prune """
    return df.dropna(subset=['Close'])
