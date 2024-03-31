import numpy as np
import matplotlib.pyplot as plt

m = 2

A = np.random.randn(m,m)
B = np.random.randn(m)
X = np.linalg.solve(A.T@A, A.T@B)

kappa = np.linalg.cond(A.T@A)

print(f'{np.linalg.cond(A) = }')
p = 1000
delta = np.zeros((p,2))
for k in range(p):
    Bp = A.T@B + 1e-10 * np.random.randn(m)
    xp = np.linalg.solve(A.T@A, Bp)
    delta[k,:] = ((xp - X) / np.linalg.norm(X)) / (np.linalg.norm(Bp - A.T@B) / np.linalg.norm(A.T@A))

fig,ax = plt.subplots()
ax.scatter(delta[:,0], delta[:,1])
circle = plt.Circle((0.0,0.0), kappa, fill=False)
ax.add_patch(circle)

print(f'{kappa = }')
print(f'{np.max(np.linalg.norm(delta, axis=1)) = }')

plt.savefig('conditionB.pdf')

plt.show()

