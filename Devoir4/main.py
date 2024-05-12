import numpy as np
import numba as nb
import matplotlib.pyplot as plt



def SVD(A):
    C = np.dot(A.T,A)
    eigvals, eigvecs = np.linalg.eig(C)
    s = np.sqrt(eigvals)
    U = eigvecs
    V = np.dot(A,U)/s
    return U,s,V