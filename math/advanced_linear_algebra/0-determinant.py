#!/usr/bin/env python3
"""Module containing determinant function"""


def determinant(matrix):
    """Calculates the determinant of a matrix"""
    # Validation: must be a list of lists
    if (not isinstance(matrix, list) or not matrix
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    # Special case: 0x0 matrix represented as [[]]
    if n == 1 and len(matrix[0]) == 0:
        return 1

    # Validation: must be square
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Recursive Laplace expansion along the first row
    det = 0
    for c in range(n):
        # Create minor matrix by removing row 0 and column c
        minor = [row[:c] + row[c + 1:] for row in matrix[1:]]
        sign = (-1) ** c
        det += sign * matrix[0][c] * determinant(minor)

    return det
