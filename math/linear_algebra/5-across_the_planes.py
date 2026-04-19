#!/usr/bin/env python3
""" anything """


def add_matrices2D(mat1, mat2):
    """ anything """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
