#!/usr/bin/env python3
"""Module containing cat_matrices2D function"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along a specific axis"""
    if axis == 0:
        if len(mat1) == 0 or len(mat2) == 0:
            return None
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [r1[:] + r2[:] for r1, r2 in zip(mat1, mat2)]
    return None
