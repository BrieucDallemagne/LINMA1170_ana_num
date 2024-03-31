import numba as nb
import numpy as np

@nb.jit(nopython=True)
def norme(V):
    # retourne la norme euclidienne d'un vecteur V
    squared_sum = 0
    for i in range(len(V)):
        squared_sum += V[i] ** 2
    euclidean_norm = np.sqrt(squared_sum)
    return euclidean_norm

@nb.jit(nopython=True)
def transpose(A):
    # transpose une matrice A
    m = len(A)
    n = len(A[0])
    B = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            B[i,j] = A[j,i]
    return B

@nb.jit(nopython=True)
def  qr(A):
    #qui renvoie deux matrices Q et R correspondant à la décomposition QR réduite de A
    # A: matrice de taille m x n
    # Q: matrice de taille m x n
    # R: matrice de taille n x n
    # return Q, R
    M, N = A.shape
    Q = np.zeros((M, N))
    for i in range(N):
        Q[:, i] = A[:, i]
    R = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            V1 = Q[:, j]
            V2 = Q[:, i]
            for k in range(M):
                R[j, i] += V1[k] * V2[k]
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]

        R[i, i] = norme(Q[:, i])
        if R[i, i] == 0:
            Q[:, i] /= R[i, i]
        Q[:, i] = Q[:, i] / R[i, i]

    Q = -Q
    R = -R

    return Q, R

def lstsq(A, B):
    # A: matrice de taille m x n
    # B: matrice de taille m x n
    # return X
    M, N = A.shape
    Y,Z = B.shape
    Q, R = qr(A)
    X = np.zeros((N, Z))
    QTB = np.dot(transpose(Q), B)
    for k in range(Z):
        for i in range(N-1, -1, -1):
            sum = 0
            for j in range(i+1, N):
                sum += R[i][ j] * X[j][  k]
            X[i][  k] = (QTB[i][  k] - sum) / R[i][  i]

    return X

