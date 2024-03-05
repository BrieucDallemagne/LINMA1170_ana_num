import numpy as np
import matplotlib.pyplot as plt

m = 2

A = np.random.randn(m,m)
b = np.random.randn(m)
x = np.linalg.solve(A, b)

kappa = np.linalg.cond(A)

# print(f'{np.linalg.cond(A) = }')

p = 1000
delta = np.zeros((p,2))
for k in range(p):
	Ap = A + 1e-10 * np.random.randn(m,m)
	xp = np.linalg.solve(Ap, b)
	delta[k,:] = ((xp - x) / np.linalg.norm(x)) / (np.linalg.norm(Ap - A) / np.linalg.norm(A))

fig,ax = plt.subplots()
ax.scatter(delta[:,0], delta[:,1])
circle = plt.Circle((0.0,0.0), kappa, fill=False)
ax.add_patch(circle)

print(f'{kappa = }')
print(f'{np.max(np.linalg.norm(delta, axis=1)) = }')

plt.show()

m = 100; n = 50

# On génère Q et R, et on calcule A = QR
R = np.triu(np.random.randn(n,n))
Q,_ = np.linalg.qr(np.random.randn(m,n))
A = Q @ R

# Q,R: f(~y)

# On calculer la décomposition QR de A
Q2, R2 = np.linalg.qr(A) # ~f(y)

# Q2, R2: ~f(y)

# Est-ce que l'algo est stable ?
print(f'{np.linalg.norm(Q2-Q) / np.linalg.norm(Q) = }')
print(f'{np.linalg.norm(R2-R) / np.linalg.norm(R) = }')

# Est-ce que l'algo est inversement stable ?
print(f'{np.linalg.norm(A - Q2 @ R2) / np.linalg.norm(A) = }')
# ||~y - y|| / ||y|| <= o(eps)

def gram_schmidt(A):
	m,n = A.shape
	Q = np.zeros((m,n))
	R = np.zeros((n,n))
	for k in range(n):
		Q[:,k] = A[:,k]
		for j in range(k):
			R[j,k] = Q[:,j] @ A[:,k]
			Q[:,k] -= R[j,k] * Q[:,j]
		R[k,k] = np.linalg.norm(Q[:,k])
		Q[:,k] /= R[k,k]
	return Q,R

def modified_gram_schmidt(A):
	m,n = A.shape
	Q = A.copy()
	R = np.zeros((n,n))
	for k in range(n):
		R[k,k] = np.linalg.norm(Q[:,k])
		Q[:,k] /= R[k,k]
		for j in range(k+1,n):
			R[k,j] = Q[:,k] @ Q[:,j]
			Q[:,j] -= R[k,j] * Q[:,k]
	return Q,R

# A = np.random.randn(20,15)

eps = 1e-8
A = np.array([
	[1, 1, 1],
	[eps, 0, 0],
	[0, eps, 0],
	[0, 0, eps]
])

Q,R = modified_gram_schmidt(A)

# Q.T @ Q == I
# R est triangulaire supÃ©rieure

fig,axs = plt.subplots(1,2)
axs[0].imshow(Q.T @ Q, cmap='Blues')
axs[1].imshow(np.abs(R), cmap='Blues')
plt.show()