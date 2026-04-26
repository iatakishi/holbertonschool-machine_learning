#!/usr/bin/env python3
"""Module containing inverse function"""


def inverse(matrix):
    """Calculates the inverse of a matrix"""
    # Validation: must be a non-empty list of lists
    if (type(matrix) is not list or len(matrix) == 0
            or any(type(row) is not list for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Helper: determinant calculation
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

    det = determinant(matrix)
    if det == 0:
        return None  # Singular matrix, no inverse

    # Step 1: Calculate cofactor matrix
    cofactor_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sub = [row[:j] + row[j + 1:] for k, row
                   in enumerate(matrix) if k != i]
            minor = determinant(sub)
            cofactor_matrix[i][j] = ((-1) ** (i + j)) * minor

    # Step 2: Transpose cofactor to get adjugate
    adjugate_matrix = [[cofactor_matrix[j][i] for j
                        in range(n)] for i in range(n)]

    # Step 3: Divide adjugate by determinant to get inverse
    inverse_matrix = [[adjugate_matrix[i][j] / det for j
                       in range(n)] for i in range(n)]

    return inverse_matrix
