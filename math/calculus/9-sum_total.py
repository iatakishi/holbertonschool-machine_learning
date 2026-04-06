#!/usr/bin/env python3
""" sum total """


def summation_i_squared(n):
    """ sum total """
    if not isinstance(n, int) or n < 1:
        return None
    else:
        return n*(n+1)*(2*n+1) // 6
