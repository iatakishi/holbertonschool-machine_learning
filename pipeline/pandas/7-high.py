#!/usr/bin/env python3
""" high """


def high(df):
    """ high """
    return df.sort_values(by='High', ascending=False)
