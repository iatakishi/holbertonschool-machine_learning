#!/usr/bin/env python3
"""Module containing adjugate function"""


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix"""
    # Validation: must be a non-empty list of lists
    if (type(matrix) is not list or len(matrix) == 0
            or any(type(row) is not list for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    # Validation: must be square
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Helper function to calculate determinant recursively
    def determinant(m):
        if len(m) == 1:
            return m[0][0]
        if len(m) == 0:  # 0x0 matrix determinant is 1 by convention
            return 1

        det = 0
        for c in range(len(m)):
            sub = [row[:c] + row[c + 1:] for row in m[1:]]
            det += ((-1) ** c) * m[0][c] * determinant(sub)
        return det

    # Step 1: Compute cofactor matrix
    cofactor_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # Create submatrix by removing row i and column j
            sub_matrix = [row[:j] + row[j + 1:] for k, row
                          in enumerate(matrix) if k != i]
            minor = determinant(sub_matrix)
            cofactor_matrix[i][j] = ((-1) ** (i + j)) * minor

    # Step 2: Transpose cofactor matrix to get adjugate
    adjugate_matrix = [[cofactor_matrix[j][i] for j in range(n)] for i
                       in range(n)]
    return adjugate_matrix
