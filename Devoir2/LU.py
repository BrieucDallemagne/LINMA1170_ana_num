import numba as nb
import numpy as np
from time import perf_counter as clock
import matplotlib.pyplot as plt


@nb.jit(nopython=True, parallel=True)
def LU_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))    


    for k in range(n):
        U[k][k] = A[k][k]
        for i in nb.prange(k + 1, n):
            L[i][k] = A[i][k] / U[k][k]
            U[k][i] = A[k][i]
            for j in nb.prange(k + 1, n):
                A[i][j] -= L[i][k] * U[k][j]

    return L, U


nb.jit(nopython=True)
def LU_decomposition2(A):
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))    


    for k in nb.prange(n):
        U[k][k] = A[k][k]
        for i in nb.prange(k + 1, n):
            L[i][k] = A[i][k] / U[k][k]
            U[k][i] = A[k][i]
            for j in nb.prange(k + 1, n):
                A[i][j] -= L[i][k] * U[k][j]

    return L, U

@nb.jit(nopython=True, parallel=True)
def Cholesky_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            s = 0
            for k in range(j):
                s += L[i][k] * L[j][k]
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]

    return L



def plot_LU_decomposition():
        # plot complexity of function QR for different scale of matrix with a loglog scale
    # return None
    N = [10,20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    A = np.random.rand(2, 2)
    LU_decomposition(A) # run 1 time for numba compilation
    LU_decomposition2(A) # run 1 time for numba compilation

    def ref_fun(N):
        return [x**3/(5*10**8) for x in N]
    
    fun = ref_fun(N)


    time1 = np.array([],dtype=float)
    #time2 = np.array([],dtype=float)
    for n in N:
        dt = 0
        dt2 = 0
        for i in range(5):
            A = np.random.rand(n, n)
            t = clock()
            LU_decomposition(A)
            dt += clock() - t

            #B = np.random.rand(n, n)
            #t2 = clock()
            #LU_decomposition2(B)
            #dt2 += clock() - t2

        #dt2 = dt2 / 5
        dt = dt / 5
        #time2 = np.append(time2, dt2)
        time1 = np.append(time1, dt)
        #print(n, dt)
    
    plt.loglog(N, time1, label = 'LU with numba')
    #plt.loglog(N, time2, label = 'LU without numba')
    plt.loglog(N, fun, 'r--', label = '0(n^3)')

    plt.xlabel('Matrix size')
    plt.ylabel('Time')
    plt.xticks(N)
    plt.title('Complexity of LU function')
    plt.legend()
    plt.savefig('LU.png')

    plt.show()




def hist_rapport_LU_cholesky():

    A = np.random.rand(2, 2)
    LU_decomposition(A) # run 1 time for numba compilation
    Cholesky_decomposition(A) # run 1 time for numba compilation

    rapport = np.array([],dtype=float)

    for i in range(3000):
        n = np.random.randint(100, 1000)
        A = np.random.rand(n, n)
        t1 = clock()
        LU_decomposition(A)
        dt1 = clock() - t1

        B = np.random.rand(n, n)
        t2 = clock()
        Cholesky_decomposition(B)
        dt2 = clock() - t2

        rapport = np.append(rapport, int(round(dt1/dt2, 0)))

    lst = [i for i in range(10)]
    for i in rapport:
        if i < 10:
            lst[int(i)] += 1

    # Plot histogram
    plt.bar([str(i) for i in range(10)], lst)
    plt.xlabel('Rapport')
    plt.ylabel('Occurence')
    plt.title('occurence of each rounded rapport of LU/cholesky')
    plt.savefig('rapport_LU_cholesky.png')



    #plt.show()

#plot_LU_decomposition()
#hist_rapport_LU_cholesky()
