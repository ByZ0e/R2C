import numpy as np


def split_matrix(matrix, size):
    m, n = matrix.shape
    rows = m // size
    cols = n // size
    result = []

    for i in range(rows):
        for j in range(cols):
            sub_matrix = matrix[i*size:(i+1)*size, j*size:(j+1)*size]
            result.append(sub_matrix)
            result = np.any(result, axis=(0, 1))

    return result

large_matrix = np.array([[0, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]])

map = large_matrix==False
size_of_submatrix = 2
submatrices = split_matrix(map, size_of_submatrix)

for submatrix in submatrices:
    print(submatrix)
