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
        if R[i, i] == 0:
            Q[:, i] /= R[i, i]
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

def rank(A):
    # A: matrice de taille m x n with m >= n
    # return true si la matrice A est de rang n
    Q, R = qr(A)
    M, N = A.shape
    for i in range(N):
        if R[i, i] == 0:
            return False
    return True
    
def control_points(t,T,i,p):

    if p == 0:
        return (T[i] <= t)*(t < T[i+1])
    else:
        u  = 0.0 if T[i+p ]  == T[i]   else (t-T[i])/(T[i+p]- T[i]) * control_points(t,T,i,p-1)
        u += 0.0 if T[i+p+1] == T[i+1] else (T[i+p+1]-t)/(T[i+p+1]-T[i+1]) * control_points(t,T,i+1,p-1)
    return u

def bspline(X,Y,t): 

#
# -2.1- Definition des noeuds et duplication de 3 premiers points de controle
#       pour avoir une courbe fermée
#

  T = range(-3,len(X)+4)
  X = [*X,*X[0:3]]
  Y = [*Y,*Y[0:3]]


  
  p = 3; n = len(T)-1  
  B = np.zeros((n-p,len(t)))  
  for i in range(0,n-p):
    B[i,:] = control_points(t,T,i,p) 

    
  x = X @ B
  y = Y @ B  
  return x,y

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




    
def plot_draw_between_points( datax, datay, noeudsx, noeudsy):
    # data: array de taille m x 2
    # controle_points: array de taille m+n x 2
    # plot the curve between the control points and show the data points
    # return None
    plt.scatter(noeudsx, noeudsy, label='control points')
    plt.axis("equal")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()

#plot_QR()

#plot_lstsq()


data = np.genfromtxt('data.csv', delimiter=',')
datax = data[:, 0]
datay = data[:, 1]
    
ti = find_ti(datax, datay)
Ti = find_TI(ti)
n = len(datax)
p = 3
m = n - p - 1
x = np.zeros(m)
y = np.zeros(m)
for i in range(m):
    x[i] = np.dot(datax, bspline(datax, datay, Ti[i])[0])
    y[i] = np.dot(datay, bspline(datax, datay, Ti[i])[1])

plot_draw_between_points(datax, datay, x, y)

