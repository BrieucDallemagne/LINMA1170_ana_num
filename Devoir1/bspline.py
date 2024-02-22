import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as clock
import matplotlib 

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
    Q, R = qr(A)
    X = np.zeros((N, N))
    for i in nb.prange(N):
        QTB = multiply(transpose(Q), B[:, i])
        x = np.zeros(N)
        for j in nb.prange(N - 1, -1, -1):
            x[j] = QTB[j]
            for k in nb.prange(j + 1, N):
                x[j] -= R[j, k] * x[k]
            x[j] /= R[j, j]
        X[:, i] = x
    return X



def rank(A):
    # A: matrice de taille m x n with m >= n
    # return true si la matrice A est de rang n
    Q, R = qr(A)
    M, N = A.shape
    for i in range(N):
        if R[i, i] == 0:
            return False
    return True
    
def bspline(t,T,i,p):

    if p == 0:
        return (T[i] <= t)*(t < T[i+1])
    else:
        u  = 0.0 if T[i+p ]  == T[i]   else (t-T[i])/(T[i+p]- T[i]) * bspline(t,T,i,p-1)
        u += 0.0 if T[i+p+1] == T[i+1] else (T[i+p+1]-t)/(T[i+p+1]-T[i+1]) * bspline(t,T,i+1,p-1)
    return u

def create_A(t,T,p):
    n = len(T) - 1
    B = np.zeros((n-p,len(t)))
    for i in range(0,n-p):
        B[i,:] = bspline(t,T,i,p)
    return B

def create_B(X,Y,T,p):
    n = len(T) - 1
    B = np.zeros((n-p,2))
    for i in range(0,n-p):
        B[i,0] = X[i]
        B[i,1] = Y[i]
    return B



def find_ti(X,Y):
    # X: array de taille m
    # Y: array de taille m
    # return t array of size m who's an approximation of X,Y on [0;1] interval
    m = len(X)
    t = np.zeros(m)
    t[0] = 0
    for i in range(1, m):
        t[i] = t[i-1] + np.sqrt((X[i] - X[i-1])**2 + (Y[i] - Y[i-1])**2)
    t = t / t[-1]
    return t

def find_TI(ti):
    m = len(ti)
    T = np.zeros(m+4)
    T[0] = T[1] = T[2] = T[3] = 0
    T[m] = T[m+1] = T[m+2] = T[m+3] = 1
    for j in range(2, m-3):
        i = int(j)
        alpha = j - i
        T[j+3] = (1 - alpha) * ti[i-1] + alpha * ti[i]
    return T

def plot_QR():
    # plot complexity of function QR for different scale of matrix with a loglog scale
    # return None
    N = [10,20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    A = np.random.rand(2, 2)
    qr(A) # run 1 time for numba compilation

    def ref_fun(N):
        return [x**3/(3*10**8) for x in N]
    
    fun = ref_fun(N)


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
    
    plt.loglog(N, time1, label = 'QR')
    plt.loglog(N, fun, 'r--', label = '0(n^3)')

    plt.xlabel('Matrix size')
    plt.ylabel('Time')
    plt.xticks(N)
    plt.title('Complexity of QR function')
    plt.legend()
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
    
def plot_Bspline(B,X, Y, x,y):
    fig = plt.figure("Approximation avec des B-splines")
    plt.plot(X,Y,'.r',markersize=10, label='Points')
    plt.plot([*X,X[0]],[*Y,Y[0]],'--r')
    plt.plot(x,y,'-b')
    x = len(x)
    plt.axis("equal"); 
    plt.gca().invert_yaxis()
    plt.show()
    plt.savefig('Bspline_with' + str(x) + 'points.png')


def test_QR():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 8]])
    Q, R = qr(A)
    np.testing.assert_allclose(dot_matrix(Q,R), A)
    print("Test QR passed")




#plot_QR()

#plot_lstsq()

data = np.genfromtxt('data.csv', delimiter=',')
X = data[:, 0]
Y = data[:, 1]
ti = find_ti(X,Y)
T = find_TI(ti)
A = create_A(ti, T, 3)
B = create_B(X,Y,T,3)
X, Y = lstsq(A, B)
t = np.linspace(0,len(X),len(X)*100 + 1)
x,y = bspline(t,T,3)
plot_Bspline(B,X,Y,x,y)




#t,T, Bspline,leastsq,plot


