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
    N = [10,20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    A = np.random.rand(2, 2)
    qr(A) # run 1 time for numba compilation


    time1 = np.array([],dtype=float)
    for n in N:
        dt = 0
        for i in range(5):
            A = np.random.rand(n, n)
            t = clock()
            qr(A)
            dt += clock() - t

        dt = dt / 5
        time1 = np.append(time1, dt)
        print(n, dt)
    
    plt.loglog(N, time1)

    plt.xlabel('Matrix size')
    plt.ylabel('Time')
    plt.xticks(N)
    plt.title('Complexity of QR function')
    plt.savefig('QR.png')
    plt.show()

def plot_lstsq():
    # plot complexity of function lstsq for different scale of matrix with a loglog scale
    # return None
    N = [10,20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    A = np.random.rand(2, 2)
    B = np.random.rand(2)
    lstsq(A, B) # run 1 time for numba compilation


    time1 = np.array([],dtype=float)
    for n in N:
        dt = 0
        for i in range(5):
            A = np.random.rand(n, n)
            B = np.random.rand(n)
            t = clock()
            lstsq(A, B)
            dt += clock() - t

        dt = dt /5
        time1 = np.append(time1, dt)
        print(n, dt)
    
    plt.loglog(N, time1)
    plt.xlabel('Matrix size')
    plt.ylabel('Time')
    plt.xticks(N)

    plt.title('Complexity of lstsq function')
    plt.savefig('lstsq.png')
    plt.show()

#plot_QR()

#plot_lstsq()
    
def rank(A):
    # A: matrice de taille m x n with m >= n
    # return true si la matrice A est de rang n
    return np.linalg.matrix_rank(A) == len(A[0])
    
def noeuds_controle( n, data):
    # n: nombre de points de controle
    # data: array de taille m x 2
    # return array of size m+n x 2
    m = len(data)
    controle_points = np.zeros((m+n, 2))
    for i in range(m):
        controle_points[i] = data[i]
    for i in range(m, m+n):
        controle_points[i] = data[m-1]
    return controle_points
    
   
data = np.genfromtxt('data.csv', delimiter=',')

noeuds = noeuds_controle(3, data)
print(noeuds)
    
def plot_draw_between_points(data, controle_points):
    # data: array de taille m x 2
    # controle_points: array de taille m+n x 2
    # plot the curve between the control points and show the data points
    # return None
    X_data = data[:, 0]
    Y_data = data[:, 1]
    X_controle = controle_points[:, 0]
    Y_controle = controle_points[:, 1]
    plt.plot(X_data, Y_data, 'red', marker='o')
    plt.plot(X_controle, Y_controle, 'blue', marker='o')
    plt.scatter(X_controle, Y_controle, color='blue')
    plt.show()





plot_draw_between_points(data, noeuds)






















