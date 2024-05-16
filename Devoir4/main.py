import numpy as np
import numba as nb
import scipy as sp
import matplotlib.pyplot as plt
 
@nb.njit(nopython=True,fastmath=True)
def mat_mult(A,B):
    m = A.shape[0]
    n = A.shape[1]
    p = B.shape[1]
    result = np.zeros((m,p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result
 
@nb.njit(nopython=True,fastmath=True)
def mat_vec_mult_hor(A,B):
    n = A.shape[1]
    p = B.shape[1]
    result = np.zeros((1,p))
    for j in range(p):
        for k in range(n):
            result[0][j] += A[0][k] * B[k][j]
    return result
 
@nb.njit(nopython=True,fastmath=True)
def egenval(A):
    C = mat_mult(A.T,A)
    eigvals, eigvecs = np.linalg.eig(C)
    return eigvals ,eigvecs
 
def SVD(A):
    eigvals,eigvecs = egenval(A)
    s = np.sqrt(eigvals)
    U = eigvecs
    V = mat_vec_mult_hor(A,U)/s
    return U,s,V