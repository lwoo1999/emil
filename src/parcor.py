import numpy as np
from numba import jit, float64

@jit(float64[:,:](float64[:,:]), nopython=True)
def normalise(m):
    n = len(m)
    res = np.zeros_like(m)
    for i in range(n):
        for j in range(n):
            if i == j:
                res[i, j] = 1
            else:
                res[i, j] = -m[i, j] / np.sqrt(m[i, i] * m[j, j])
    
    return res

def parcor(*args):
    return normalise(np.linalg.inv(np.cov(*args)))