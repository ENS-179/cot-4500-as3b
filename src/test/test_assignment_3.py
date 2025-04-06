import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../main")))

from assignment_3 import (
    solve_system,
    lu_factorization,
    is_diagonally_dominant,
    is_positive_definite,
)


def format_row(row):
    return "[" + " ".join(str(int(round(val))) for val in row) + "]"


def print_matrix(matrix, name):
    print(f"{name}:")
    for row in matrix:
        print(format_row(row))
    print()


def main():
    # 1. Guassian Elimination and Backward Substitution
    solution = solve_system()
    print("1.")
    print(format_row(solution))
    print()

    # 2. L U Factorization
    matrix = [[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]]

    L, U, determinant = lu_factorization(matrix)

    print("2.")
    print(determinant)
    print_matrix(L, "L")
    print_matrix(U, "U")

    # 3. Diagonal Dominance Check
    print("3.")
    diag_matrix = [
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8],
    ]

    print(is_diagonally_dominant(diag_matrix))

    # 4. Positive Definiteness Check
    print("4.")
    pos_def_matrix = [[2, 2, 1], [2, 3, 0], [1, 0, 2]]
    print(is_positive_definite(pos_def_matrix))


if __name__ == "__main__":
    main()
