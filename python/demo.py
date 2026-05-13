import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from optimalControlSolver import optimalControlSolver

# Symbols
x1, x2, u1, u2 = sp.symbols('x1 x2 u1 u2')
x = sp.Matrix([x1, x2])
u = sp.Matrix([u1, u2])

# Dynamics f(x,u)
A = sp.Matrix([[0, 1], [-1, 0]])
B = sp.eye(2)
f = A*x + B*u

# Running cost g(x,u)
Q = sp.diag(1, 1)
R = sp.diag(sp.Rational(1, 10), sp.Rational(1, 10))
g = x.T @ Q @ x + u.T @ R @ u
g = g[0]  # extract scalar

# Terminal cost Phi(x)
Phi = 10 * (x.T @ x)[0]

# Time grid and initial conditions
T = 5.0
N = 501
tGrid = np.linspace(0, T, N)
x0 = np.array([1.0, 0.0])

# Initial control guess U0
U0 = np.zeros((N, 2))

# Options
opts = {
    'maxIters': 50,
    'alpha': 1.0,
    'beta': 0.5,
    'c1': 1e-4,
    'tol': 1e-6,
    'interp': 'linear',
    'uLower': None,
    'uUpper': None,
    'odeOptions': {'RelTol': 1e-7, 'AbsTol': 1e-9},
    'verbose': True
}

# Solve
sol, info = optimalControlSolver(f, g, Phi, x, u, tGrid, x0, U0, opts)

# Plots
fig, ax = plt.subplots()
ax.plot(sol['t'], sol['X'], linewidth=1.5)
ax.grid(True)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$x$')
ax.set_title('States')
labels = [f'$x_{i+1}$' for i in range(sol['X'].shape[1])]
ax.legend(labels)

fig, ax = plt.subplots()
ax.plot(sol['t'], sol['U'], linewidth=1.5)
ax.grid(True)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$u$')
ax.set_title('Controls')
labels = [f'$u_{i+1}$' for i in range(sol['U'].shape[1])]
ax.legend(labels)

fig, ax = plt.subplots()
ax.plot(sol['t'], sol['P'], linewidth=1.5)
ax.grid(True)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$p$')
ax.set_title('Costates')
labels = [f'$p_{i+1}$' for i in range(sol['P'].shape[1])]
ax.legend(labels)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(range(1, len(sol['J_hist']) + 1), sol['J_hist'], '-o', linewidth=1.5)
ax1.grid(True)
ax1.set_xlabel('iteration')
ax1.set_ylabel(r'$J$')
ax1.set_title('Cost per iteration')

ax2.plot(range(1, len(sol['grad_norm_hist']) + 1), sol['grad_norm_hist'], '-o', linewidth=1.5)
ax2.grid(True)
ax2.set_xlabel('iteration')
ax2.set_ylabel(r'$\|dH/du\|_F$')
ax2.set_title('Gradient norm per iteration')

plt.show()

print(f'Done. Final cost J = {sol["J"]:.6f}, iterations = {info["iters"]}')
