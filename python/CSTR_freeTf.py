import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from optimalControlSolver import optimalControlSolver

# Symbols
x1, x2, u = sp.symbols('x1 x2 u')
x = sp.Matrix([x1, x2])

# Parameters
R = 0.1        # control weight in running cost
Tf0 = 0.78     # initial guess for final time
N = 1001       # time grid points

tGrid = np.linspace(0, Tf0, N)

# Dynamics f(x,u)
expTerm = sp.exp(25*x1/(x1+2))
f1 = -2*(x1 + 0.25) + (x2 + 0.5)*expTerm - (x1 + 0.25)*u
f2 = 0.5 - x2 - (x2 + 0.5)*expTerm
f = sp.Matrix([f1, f2])

# Running cost g(x, u)
g = x1**2 + x2**2 + R*u**2

# Terminal cost Phi(x)
Phi = x1**2 + x2**2

# Initial condition
x0 = np.array([0.05, 0])

# Initial control guess U0
U0 = np.zeros((N, 1))

# Options (enable free-final-time)
opts = {
    'maxIters': 1000,
    'alpha': 1.0,
    'beta': 0.5,
    'c1': 1e-4,
    'tol': 1e-6,
    'interp': 'linear',
    'uLower': -2,
    'uUpper': 2,
    'odeOptions': {'RelTol': 1e-7, 'AbsTol': 1e-9},
    'verbose': True,
    'freeFinalTime': True,
    'tfAlpha': 0.5,
    'tfBeta': 0.5,
    'tfC1': 1e-4,
    'tfLower': 0.2,
    'tfUpper': 5.0
}

# Call solver
sol, info = optimalControlSolver(f, g, Phi, x, u, tGrid, x0, U0, opts)

# Plot results
fig, ax = plt.subplots()
ax.plot(sol['t'], sol['X'], linewidth=1.5)
ax.grid(True)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$x$')
ax.set_title(f"States (tf = {sol['tf']:.4f})")
ax.legend([r'$x_1$', r'$x_2$'])

fig, ax = plt.subplots()
ax.plot(sol['t'], sol['U'], linewidth=1.5)
ax.grid(True)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$u$')
ax.set_title('Control Input')

fig, ax = plt.subplots()
ax.plot(sol['t'], sol['P'], linewidth=1.5)
ax.grid(True)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$p$')
ax.set_title('Costates')
ax.legend([r'$p_1$', r'$p_2$'])

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

print(f'CSTR free-tf demo complete. Final cost J = {sol["J"]:.6f}, tf = {sol["tf"]:.6f}, iterations = {info["iters"]}')
