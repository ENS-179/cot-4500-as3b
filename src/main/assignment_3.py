import numpy as np


# 1. Guassian Elimination and Backward Substitution
def gaussian_elimination(A, b):
    n = len(A)
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        if i != max_row:
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]
    return A, b


def backward_substitution(U, y):
    n = len(U)
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        sum_ax = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_ax) / U[i][i]
    return x


def solve_system():
    A = [[2, -1, 1], [1, 3, 1], [-1, 5, 4]]
    b = [6, 0, -3]

    U, y = gaussian_elimination([row[:] for row in A], b[:])
    x = backward_substitution(U, y)
    return x


# 2. L U Factorization
def lu_factorization(matrix):
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for k in range(i, n):
            sum_ = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = matrix[i][k] - sum_

        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0
            else:
                sum_ = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (matrix[k][i] - sum_) / U[i][i]

    determinant = 1.0
    for i in range(n):
        determinant *= U[i][i]

    return L, U, determinant


# 3. Diagonal Dominance Check
def is_diagonally_dominant(matrix):
    n = len(matrix)
    for i in range(n):
        diag = abs(matrix[i][i])
        off_diag_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if diag < off_diag_sum:
            return False
    return True


# 4. Positive Definiteness Check
def is_positive_definite(matrix):
    n = len(matrix)
    for k in range(1, n + 1):
        minor = [[matrix[i][j] for j in range(k)] for i in range(k)]
        if np.linalg.det(minor) <= 0:
            return False
    return True
