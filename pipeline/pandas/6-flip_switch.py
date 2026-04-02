#!/usr/bin/env python3
""" flip-switch """


def flip_switch(df):
    """ flip switch """
    df = df.sort_index(ascending=False)
    return df.T
