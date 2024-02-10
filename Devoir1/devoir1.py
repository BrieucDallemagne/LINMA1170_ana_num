import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as clock


@nb.jit(nopython=True, parallel=True)
def norme(V):
    squared_sum = 0
    for i in nb.prange(len(V)):
        squared_sum += V[i] ** 2
    euclidean_norm = np.sqrt(squared_sum)
    return euclidean_norm


@nb.jit(nopython=True, parallel=True)
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
        Q[:, i] = Q[:, i] / R[i, i]

    Q = -Q
    R = -R

    return Q, R


@nb.jit(nopython=True, parallel=True)
def lstsq(A, B):
    # A: matrice de taille m x n
    # B: vecteur de taille m

    # return Frobenius norm of A x - B

    diag_B = np.zeros((len(B), len(B)))
    for i in nb.prange(len(B)):
        diag_B[i, i] = B[i]
    M = len(A)
    N = len(A[0])
    
    sum = 0.0
    for i in nb.prange(M):
        for j in nb.prange(N):
            sum += (A[i, j] - diag_B[i, i])**2

    return np.sqrt(sum)

def plot_QR():
    # plot complexity of function QR for different scale of matrix with a loglog scale
    # return None
    N = [10, 100,200, 300, 400, 500, 600, 700, 800, 900, 1000]

    A = np.random.rand(2, 2)
    qr(A) # run 1 time for numba compilation


    time1 = np.array([],dtype=float)
    for n in N:
        A = np.random.rand(n, n)
        t = clock()
        qr(A)
        dt = clock() - t
        time1 = np.append(time1, dt)
        print(n, dt)
    
    plt.loglog(N, time1)

    plt.xlabel('Matrix size')
    plt.ylabel('Time')
    plt.xticks(N)
    plt.title('Complexity of QR function')
    plt.show()

def plot_lstsq():
    # plot complexity of function lstsq for different scale of matrix with a loglog scale
    # return None
    N = [10, 100,200, 300, 400, 500, 600, 700, 800, 900, 1000]

    A = np.random.rand(2, 2)
    B = np.random.rand(2)
    lstsq(A, B) # run 1 time for numba compilation


    time1 = np.array([],dtype=float)
    for n in N:
        A = np.random.rand(n, n)
        B = np.random.rand(n)
        t = clock()
        lstsq(A, B)
        dt = clock() - t
        time1 = np.append(time1, dt)
        print(n, dt)
    
    plt.loglog(N, time1)
    plt.xlabel('Matrix size')
    plt.ylabel('Time')
    plt.xticks(N)

    plt.title('Complexity of lstsq function')
    plt.show()

plot_QR()

plot_lstsq()