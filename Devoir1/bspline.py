import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as clock
import matplotlib 

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
            continue

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
            if R[i][i] == 0:
                continue
            X[i][  k] = (QTB[i][  k] - sum) / R[i][  i]

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
    if (i >= len(T) - p - 1) :
        return 0.0
    if (t == 1 and i == len(T) - 4 - 1) :
        return 1.0
    if p == 0:
        return (T[i] <= t)*(t < T[i+1])
    else:
        u  = 0.0 if T[i+p ]  == T[i]   else (t-T[i])/(T[i+p]- T[i]) * bspline(t,T,i,p-1)
        u += 0.0 if T[i+p+1] == T[i+1] else (T[i+p+1]-t)/(T[i+p+1]-T[i+1]) * bspline(t,T,i+1,p-1)
    return u

def create_A(t,T,p):
    n = len(T) - 1
    A = np.zeros((n-p,len(t)))
    for i in range(0,n-p):
        for j in range(len(t)):
            A[i,j] = bspline(t[j],T,i,p)

    A = transpose(A)
    A[-1,-1] = 1

    return A




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

def find_TI(n,ti):
    m = len(ti)
    d = m / (n - 3)
    T = [0, 0, 0, 0]  
    for j in range(1, n - 3):
        i = int(np.floor(j * d))
        alpha = j * d - i
        T.append((1 - alpha) * ti[i - 1] + alpha * ti[i])

    T.extend([1, 1, 1, 1])  
    return T

def plot_QR():
    # plot complexity of function QR for different scale of matrix with a loglog scale
    # return None
    N = [10,20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    M = 1000

    A = np.random.rand(2, 2)
    qr(A) # run 1 time for numba compilation

    def ref_fun(N):
        return [x**2/(3*10**5) for x in N]
    
    fun = ref_fun(N)


    time1 = np.array([],dtype=float)
    for n in N:
        dt = 0
        for i in range(5):
            A = np.random.rand(M, n)
            t = clock()
            qr(A)
            dt += clock() - t

        dt = dt / 5
        time1 = np.append(time1, dt)
        #print(n, dt)
    
    plt.loglog(N, time1, label = 'QR')
    plt.loglog(N, fun, 'r--', label = '0(n^2.m)')

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
    B = np.random.rand(2, 1)
    lstsq(A, B) # run 1 time for numba compilation

    def ref_fun(N):
        return [x**2/(10**6) for x in N]
    
    fun = ref_fun(N)


    time1 = np.array([],dtype=float)
    for n in N:
        dt = 0
        for i in range(5):
            A = np.random.rand(n, n)
            B = np.random.rand(n, 2)
            t = clock()
            lstsq(A, B)
            dt += clock() - t

        dt = dt /5
        time1 = np.append(time1, dt)
        #print(n, dt)
    
    plt.loglog(N, time1)
    plt.loglog(N, fun, 'r--', label = '0(n^2)')
    plt.xlabel('Matrix size')
    plt.ylabel('Time')
    plt.xticks(N)

    plt.title('Complexity of lstsq function')
    plt.savefig('lstsq.png')
    plt.show()


    
def plot_Bspline(X, Y, x,y,x_val, y_val):

    # return None
    plt.plot(X, Y, 'ro', label = 'data', markersize = 2)

    plt.plot(x, y, 'bo', label = 'Least square')
    plt.plot(x_val, y_val, 'b-', label = 'B-spline avec {} points'.format(len(x)))
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('B-spline')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig('Bspline.png')
    plt.show()






data = np.genfromtxt('data.csv', delimiter=',')

def main(data,N):
    X = data[:,0]
    Y = data[:,1]
    ti = find_ti(X,Y)
    #print(ti)
    T = find_TI(N,ti)
    #print(T)
    A = create_A(ti, T, 3)
    #print(A)
    B = np.array([X,Y]).T
    Xlst = lstsq(A, B)
    #print(Xlst)
    x = Xlst [:, 0]
    y = Xlst [:, 1]
    x_val = np.zeros(len(ti))
    y_val = np.zeros(len(ti))
    for i in range(len(ti)):
        x_val[i] = sum([bspline(ti[i],T,j,3)*x[j] for j in range(len(x))])
        y_val[i] = sum([bspline(ti[i],T,j,3)*y[j] for j in range(len(y))])

    #plot_QR()
    #plot_lstsq()
    plot_Bspline(X,Y,x,y, x_val, y_val)
    return None 

main(data, 40)






