#!/usr/bin/env python3
""" analyze """


def analyze(df):
    """ analyze """
    return df.drop(columns=['Timestamp']).describe()
