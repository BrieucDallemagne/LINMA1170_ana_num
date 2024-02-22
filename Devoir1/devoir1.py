import numba as nb
import numpy as np

@nb.jit(nopython=True)
def norme(V):
    # retourne la norme euclidienne d'un vecteur V
    squared_sum = 0
    for i in nb.prange(len(V)):
        squared_sum += V[i] ** 2
    euclidean_norm = np.sqrt(squared_sum)
    return euclidean_norm

@nb.jit(nopython=True)
def transpose(A):
    # transpose une matrice A
    m = len(A)
    n = len(A[0])
    B = np.zeros((n,m))
    for i in nb.prange(n):
        for j in nb.prange(m):
            B[i,j] = A[j,i]
    return B

@nb.jit(nopython=True)
def dot_matrix(A, B):
    # A: matrice de taille m x n
    # B: matrice de taille n x p
    # return C: matrice de taille m x p
    if len(A[0]) != len(B):
        return None
    
    result = []
    for i in nb.prange(len(A)):
        row = []
        for j in nb.prange(len(B[0])):
            sum = 0
            for k in nb.prange(len(B)):
                sum += A[i][k] * B[k][j]
            row.append(sum)
        result.append(row)
    
    return result


@nb.jit(nopython=True)
def  qr(A):
    #qui renvoie deux matrices Q et R correspondant à la décomposition QR réduite de A
    # A: matrice de taille m x n
    # Q: matrice de taille m x n
    # R: matrice de taille n x n
    # return Q, R
    M, N = A.shape
    Q = np.zeros((M, N))
    for i in nb.prange(N):
        Q[:, i] = A[:, i]
    R = np.zeros((N, N))
    for i in nb.prange(N):
        for j in nb.prange(i):
            V1 = Q[:, j]
            V2 = Q[:, i]
            for k in nb.prange(M):
                R[j, i] += V1[k] * V2[k]
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]

        R[i, i] = norme(Q[:, i])
        if R[i, i] == 0:
            Q[:, i] /= R[i, i]
        Q[:, i] = Q[:, i] / R[i, i]

    Q = -Q
    R = -R

    return Q, R

@nb.jit(nopython=True)
def multiply(matrix, vector):
    # mumtiply a matrix by a vector

    result = [0] * len(matrix)

    for i in nb.prange(len(matrix)):
        for j in nb.prange(len(vector)):
            result[i] += matrix[i][j] * vector[j]

    return result



@nb.jit(nopython=True)
def lstsq(A, B):
    # A: matrice de taille m x n
    # B: matrice de taille m x n
    # return X
    M, N = A.shape
    Y,Z = B.shape
    Q, R = qr(A)
    X = np.zeros((N, Z))
    QTB = dot_matrix(transpose(Q), B)
    #resolution back substitution
    for i in nb.prange(N):
        x = np.zeros(N)
        for j in nb.prange(N - 1, -1, -1):
            sum = 0
            for k in nb.prange(j+1, N):
                sum += R[j, k] * x[k]
            x[j] = (QTB[j, i] - sum) / R[j, j]
        X[:,i] = x

            
    return X



