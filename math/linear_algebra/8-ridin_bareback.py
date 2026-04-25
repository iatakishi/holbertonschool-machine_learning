#!/usr/bin/env python3
"""Module containing mat_mul function"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication of two 2D matrices"""
    if len(mat1[0]) != len(mat2):
        return None

    # Transpose mat2 to easily access columns as rows
    mat2_t = [[mat2[j][i] for j in range(len(mat2))]
              for i in range(len(mat2[0]))]

    result = []
    for row1 in mat1:
        new_row = []
        for col2 in mat2_t:
            # Dot product of row1 and col2
            val = sum(a * b for a, b in zip(row1, col2))
            new_row.append(val)
        result.append(new_row)

    return result
