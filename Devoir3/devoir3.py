import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.jit(nopython=True)
def norme(V):
    # retourne la norme euclidienne d'un vecteur V
    squared_sum = 0
    for i in nb.prange(len(V)):
        squared_sum += abs(V[i]) ** 2
    euclidean_norm = np.sqrt(squared_sum)
    return euclidean_norm

@nb.jit(nopython=True)
def transpose(A):
    # transpose une matrice A
    m = len(A)
    n = len(A[0])
    B = np.zeros((n,m), dtype=complex)
    for i in nb.prange(n):
        for j in nb.prange(m):
            B[i,j] = A[j,i]
    return B

@nb.jit(nopython=True, parallel=True)
def dot_product(M1,M2):
    # retourne le produit scalaire de deux matrices M1 et M2
    m,n = np.shape(M1)
    p = len(M2[0])
    result = np.zeros((m,p)) + np.zeros((m,p))*1j
    for i in nb.prange(m):
        for j in nb.prange(p):
            for k in nb.prange(n):
                result[i,j] += M1[i,k] * M2[k,j]
    return result

@nb.jit(nopython=True)
def dot_product_vector(V1,V2):
    # retourne le produit scalaire de deux vecteurs V1 et V2
    n = len(V1)
    result = 0 + 0j
    for i in nb.prange(n):
        result += V1[i] * V2[i]
    return result

@nb.jit(nopython=True)
def mult_matrix_vector(M,V):
    # retourne le produit d'une matrice M et d'un vecteur V
    m,n = np.shape(M)
    result = np.zeros(n) + np.zeros(n)*1j
    for i in nb.prange(m):
        for j in nb.prange(n):
            result[i] += M[i,j] * V[j]
    return result

@nb.jit(nopython=True)
def mult_vector_matrix(V,M):
    # retourne le produit d'un vecteur V et d'une matrice M
    m,n = np.shape(M)
    result = np.zeros(n) + np.zeros(n)*1j
    for i in nb.prange(n):
        for j in nb.prange(m):
            result[i] += V[j] * M[j,i]
    return result

@nb.jit(nopython=True)
def hypot(a, b):
    return np.sqrt(np.abs(a)**2 + np.abs(b)**2)

def eye(n):
    A = np.zeros((n,n)) + np.zeros((n,n))*1j
    for i in range(n):
        A[i,i] = 1
    return A


@nb.jit(nopython=True)
def hessenberg(A,P):
    n = A.shape[0]
    for k in nb.prange(n-2):
        v = A[k+1:,k].copy()
        e = v[0]
        v[0] += np.sign(e.real)*norme(v) 
            
        v[0] += e*norme(v)
        v /= norme(v)
        A[k+1:,k:] -= 2*np.outer(v, mult_vector_matrix(v, A[k+1:,k:]))
        A[:,k+1:] -= 2*np.outer(mult_matrix_vector(A[:,k+1:], v), v)
        P[:,k+1:] -= 2*np.outer(mult_matrix_vector(P[:,k+1:], v), v)
    return None


def step_qr(H, Q):

    n = H.shape[0]
    for i in range(n-1):
        j = 0
        c = H[i,i]/hypot(H[i,i], H[i+1,i])
        s = H[i+1,i]/hypot(H[i,i], H[i+1,i])
        H[i:i+2,i:] = dot_product(np.array([[c, s], [-s, c]]), H[i:i+2,i:])
        if j == 0:
            print(Q[:,i:i+2])
            print(np.array([[c, s], [-s, c]]))
            j += 1
        Q[:,i:i+2] = dot_product(Q[:,i:i+2], np.array([[c, s], [-s, c]]))
    return None

@nb.jit(nopython=True)
def step_qr_shift(H, Q, m):
    n = H.shape[0]
    c = H[n-2,n-2]
    s = H[n-1,n-2]
    d = H[n-1,n-1]
    delta = (c-d)/2
    mu = d - delta - np.sign(delta)*hypot(delta, s)
    c = (d-mu)/hypot(delta, s)
    s = -s/hypot(delta, s)
    H[n-2:n,n-2:] = np.array(dot_product(np.array([[c, s], [-s, c]]), H[n-2:n,n-2:]))
    Q[:,n-2:n] = dot_product(Q[:,n-2:n], np.array([[c, s], [-s, c]]))
    return mu


@nb.jit(nopython=True)
def solve_qr(A, use_shifts, eps, max_iter):
    n = A.shape[0]
    H = A.copy()
    Q = np.eye(n)
    for i in nb.prange(max_iter):
        if use_shifts:
            m = H[n-1,n-1]
            H, Q = step_qr_shift(H, Q, m)
        else:
            H, Q = step_qr(H, Q)
        if norme(np.triu(H, 1)) < eps:
            break
    return np.diag(H), Q

def test():
    """test les 4 fonctions hessenberg, step_qr, step_qr_shift et solve_qr par rapport aux fonctions de np.linalg"""
    # crée un np.array contenant des complexes
    random_matrix = np.random.rand(5,5) + 1j*np.random.rand(5,5)
    A = random_matrix.copy()
    P = np.eye(5) + np.zeros((5,5))*1j
    A2 = A.copy()
    P2 = P.copy()
    hessenberg(A, P)
    H = A.copy()
    Q = P.copy()
    mu = step_qr(H, Q)
    H, Q = step_qr_shift(H, Q, 0)
    eigenvalues, eigenvectors = solve_qr(random_matrix, False, 1e-10, 100)
    np_eigenvalues = np.linalg.eig(random_matrix)[0]
    np_eigenvectors = np.linalg.eig(random_matrix)[1]
    assert np.allclose(eigenvalues, np_eigenvalues)
    assert np.allclose(np.abs(eigenvectors), np.abs(np_eigenvectors))
    print("Les tests ont réussi")

test()
