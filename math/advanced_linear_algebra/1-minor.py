#!/usr/bin/env python3
"""Module containing minor function"""


def minor(matrix):
    """Calculates the minor matrix of a matrix"""
    # Validation: must be a list of lists
    if (not isinstance(matrix, list) or not matrix
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    # Validation: must be square
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Helper function to calculate determinant recursively
    def determinant(m):
        if len(m) == 1:
            return m[0][0]
        if len(m) == 0:  # 0x0 matrix determinant is defined as 1
            return 1

        det = 0
        for c in range(len(m)):
            # Remove row 0 and column c
            sub = [row[:c] + row[c + 1:] for row in m[1:]]
            det += ((-1) ** c) * m[0][c] * determinant(sub)
        return det

    # Build the minor matrix
    minors = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # Create submatrix by removing row i and column j
            sub_matrix = [row[:j] + row[j + 1:] for row
                          in (matrix[:i] + matrix[i + 1:])]
            minors[i][j] = determinant(sub_matrix)

    return minors
