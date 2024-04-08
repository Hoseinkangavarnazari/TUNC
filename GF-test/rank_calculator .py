import galois
# look into this https://mhostetter.github.io/galois/v0.3.8 for more information
import numpy as np

GF8 = galois.GF(2**8)
GF8.ufunc_mode


# define coefficient matrix 
coeffMatrix = [[1, 0, 1, 0, 1, 0, 1, 0],
               [0, 1, 1, 0, 0, 1, 1, 0],
               [0, 0, 0, 1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1]]

coeffMatrixInGF8 = GF8(coeffMatrix)
# return rank
print("Rank", np.linalg.matrix_rank(coeffMatrixInGF8))

