#!/usr/bin/env python3
""" high """


def flip_switch(df):
    """ high """
    return df.sort_values(by='High', ascending=False)
