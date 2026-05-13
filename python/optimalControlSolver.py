import numpy as np
from scipy.integrate import odeint, trapezoid
from scipy.interpolate import interp1d
import sympy as sp


def optimalControlSolver(symF, symG, symPhi, xSym, uSym, tGrid, x0, U0, opts=None):
    """
    Optimal control solver using gradient descent with line search.
    
    Args:
        symF: SymPy column matrix [n x 1] - dynamics
        symG: SymPy scalar - running cost
        symPhi: SymPy scalar - terminal cost
        xSym: SymPy column matrix [n x 1] - state symbols
        uSym: SymPy column matrix [m x 1] - control symbols
        tGrid: array [N] - time grid (strictly increasing)
        x0: array [n] - initial condition
        U0: array [N x m] - initial control guess
        opts: dict - options
    
    Returns:
        sol: dict with keys t, tf, X, U, P, J, J_hist, grad_norm_hist
        info: dict with key iters
    """
    if opts is None:
        opts = {}
    
    # Dimensions
    n = len(xSym)
    m = len(uSym)
    
    # Basic checks
    if hasattr(symF, 'shape'):
        assert symF.shape == (n, 1), f'symF must be [{n} x 1] matching xSym.'
    assert isinstance(symG, sp.Basic), 'symG must be a SymPy expression.'
    assert isinstance(symPhi, sp.Basic), 'symPhi must be a SymPy expression.'
    
    # Make sure time grid is strictly increasing
    tGrid = np.asarray(tGrid, dtype=float).flatten()
    N = len(tGrid)
    assert N >= 2 and np.all(np.diff(tGrid) > 0), 'tGrid must be strictly increasing with >=2 points.'
    tf = tGrid[-1] # final time
    
    # Initial Condition
    x0 = np.asarray(x0, dtype=float).flatten()
    assert len(x0) == n, f'x0 must be [{n}] vector.'
    
    # Initial Guess for the Control Input
    U = np.asarray(U0, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    assert U.shape == (N, m), f'U0 must have shape ({N}, {m}).'
    
    # Options defaults
    opts = setDefault(opts, 'maxIters', 50)
    opts = setDefault(opts, 'alpha', 1.0)
    opts = setDefault(opts, 'beta', 0.5)
    opts = setDefault(opts, 'c1', 1e-4)
    opts = setDefault(opts, 'tol', 1e-6)
    opts = setDefault(opts, 'odeOptions', {})
    opts = setDefault(opts, 'interp', 'linear')
    opts = setDefault(opts, 'uLower', None)
    opts = setDefault(opts, 'uUpper', None)
    opts = setDefault(opts, 'maxLineSearch', 10)
    opts = setDefault(opts, 'verbose', True)
    opts = setDefault(opts, 'freeFinalTime', False)
    opts = setDefault(opts, 'tfAlpha', 0.5)
    opts = setDefault(opts, 'tfBeta', 0.5)
    opts = setDefault(opts, 'tfC1', 1e-4)
    opts = setDefault(opts, 'tfLower', None)
    opts = setDefault(opts, 'tfUpper', None)
    
    # Build symbolic gradients
    pSym = sp.Matrix(sp.symbols(f'p0:{n}'))
    tSym = sp.Symbol('t')
    
    # G is the running cost and f is the system's dynamics
    dGdx_sym = sp.Matrix([sp.diff(symG, xi) for xi in xSym])
    dGdu_sym = sp.Matrix([sp.diff(symG, ui) for ui in uSym])
    Jfx_sym = sp.Matrix([[sp.diff(symF[i], xSym[j]) for j in range(n)] for i in range(n)])
    Jfu_sym = sp.Matrix([[sp.diff(symF[i], uSym[j]) for j in range(m)] for i in range(n)])
    
    dHdx_sym = dGdx_sym + Jfx_sym.T @ pSym
    dHdu_sym = dGdu_sym + Jfu_sym.T @ pSym
    gradPhi_sym = sp.Matrix([sp.diff(symPhi, xi) for xi in xSym])
    
    if tSym in symPhi.free_symbols:
        dPhi_dt_sym = sp.diff(symPhi, tSym)
    else:
        dPhi_dt_sym = sp.Integer(0)
    
    # Convert to numeric functions
    f_num = sp.lambdify((xSym, uSym), symF, 'numpy')
    g_num = sp.lambdify((xSym, uSym), symG, 'numpy')
    dHdx_num = sp.lambdify([xSym, uSym, pSym], dHdx_sym, 'numpy')
    dHdu_num = sp.lambdify([xSym, uSym, pSym], dHdu_sym, 'numpy')
    
    has_t_in_phi = tSym in symPhi.free_symbols or any(tSym in e.free_symbols for e in gradPhi_sym)
    if has_t_in_phi:
        gradPhi_num = sp.lambdify([xSym, tSym], gradPhi_sym, 'numpy')
    else:
        gradPhi_num = sp.lambdify([xSym], gradPhi_sym, 'numpy')
    
    # Prepare output histories
    J_hist = []
    grad_norm_hist = []
    
    # Helper for projection
    def projU(Ui):
        return projectU(Ui, opts['uLower'], opts['uUpper'])
    
    # Terminal cost functions
    if tSym in symPhi.free_symbols:
        Phi_num = sp.lambdify([xSym, tSym], symPhi, 'numpy')
        dPhi_dt_num = sp.lambdify([xSym, tSym], dPhi_dt_sym, 'numpy')
    else:
        Phi_num = sp.lambdify([xSym], symPhi, 'numpy')
        dPhi_dt_num = lambda x, t: 0.0
    
    # Initial forward pass
    X, _ = forwardSim(tGrid, x0, U, f_num, opts['odeOptions'], opts['interp'])
    J = computeCost(tGrid, X, U, g_num, Phi_num)
    
    if opts['verbose']:
        print(f'Iter {0:3d} | J = {J:.6e} (initial)')
    
    # Main loop
    for k in range(1, opts['maxIters'] + 1):
        # Forward: x(t)
        X, x_of_t = forwardSim(tGrid, x0, U, f_num, opts['odeOptions'], opts['interp'])
        
        # Backward: p(t) with terminal condition
        if has_t_in_phi:
            pTf = gradPhi_num(X[-1], tGrid[-1])
        else:
            pTf = gradPhi_num(X[-1])
        pTf = np.asarray(pTf).reshape(-1)
        P, _ = backwardSim(tGrid, pTf, x_of_t, U, dHdx_num, opts['odeOptions'], opts['interp'])
        
        # Compute gradient wrt u
        dHdu = np.zeros((N, m))
        for i in range(N):
            xi = X[i]
            ui = U[i]
            pi = P[i]
            gi = dHdu_num(xi, ui, pi)
            dHdu[i, :] = np.asarray(gi).flatten()
        
        grad_norm = np.linalg.norm(dHdu)
        grad_norm_hist.append(grad_norm)
        
        # Cost at current iterate
        J = computeCost(tGrid, X, U, g_num, Phi_num)
        J_hist.append(J)
        
        if opts['verbose']:
            print(f'Iter {k:3d} | J = {J:.6e} | ||grad_u||_F = {grad_norm:.3e}')
        
        # Stopping criterion
        if grad_norm < opts['tol']:
            if opts['verbose']:
                print(f'Converged: gradient norm below tol {opts["tol"]:.3e}.')
            break
        
        # Gradient descent with backtracking line search (Armijo)
        alpha = opts['alpha']
        accepted = False
        for ls in range(opts['maxLineSearch']):
            U_try = projU(U - alpha * dHdu)
            
            # Forward simulate
            X_try, _ = forwardSim(tGrid, x0, U_try, f_num, opts['odeOptions'], opts['interp'])
            J_try = computeCost(tGrid, X_try, U_try, g_num, Phi_num)
            
            # Armijo condition
            if J_try <= J - opts['c1'] * alpha * (grad_norm ** 2):
                U = U_try
                J = J_try
                accepted = True
                break
            else:
                alpha = alpha * opts['beta']
        
        if not accepted:
            if opts['verbose']:
                print('Line search failed to improve J; stopping.')
            break
        
        # Update free final time via transversality if requested
        if opts['freeFinalTime']:
            xf = X[-1]
            uf = U[-1]
            pf = P[-1]
            tf_curr = tGrid[-1]
            
            # Hamiltonian at tf
            H_end = g_num(xf, uf) + pf @ f_num(xf, uf)
            
            # dPhi/dt if Phi depends on time
            if tSym in symPhi.free_symbols:
                dPhi_dt_val = dPhi_dt_num(xf, tf_curr)
            else:
                dPhi_dt_val = 0.0
            dJdtf = H_end + dPhi_dt_val
            
            if abs(dJdtf) < max(opts['tol'], 1e-8) and grad_norm < opts['tol']:
                if opts['verbose']:
                    print('Converged: small dJ/dtf and control gradient.')
                break
            
            eta = opts['tfAlpha']
            acceptedTf = False
            for ls in range(opts['maxLineSearch']):
                tf_try = projectTf(tf_curr - eta * dJdtf, opts['tfLower'], opts['tfUpper'])
                if tf_try <= 0:
                    eta = eta * opts['tfBeta']
                    continue
                if abs(tf_try - tf_curr) < 1e-12:
                    break
                
                # Resample U to new final time
                tGrid_try, U_try = resampleUOnNewTf(tGrid, U, tf_curr, tf_try, opts['interp'])
                X_try, _ = forwardSim(tGrid_try, x0, U_try, f_num, opts['odeOptions'], opts['interp'])
                J_try = computeCost(tGrid_try, X_try, U_try, g_num, Phi_num)
                
                if J_try <= J - opts['tfC1'] * eta * (dJdtf ** 2):
                    tGrid = tGrid_try
                    U = projU(U_try)
                    J = J_try
                    acceptedTf = True
                    break
                else:
                    eta = eta * opts['tfBeta']
            
            if not acceptedTf and opts['verbose']:
                print('Final time step not accepted this iteration.')
    
    # Final forward/backward to report solution
    X, _ = forwardSim(tGrid, x0, U, f_num, opts['odeOptions'], opts['interp'])
    if has_t_in_phi:
        pTf = gradPhi_num(X[-1], tGrid[-1])
    else:
        pTf = gradPhi_num(X[-1])
    pTf = np.asarray(pTf).reshape(-1)
    P, _ = backwardSim(tGrid, pTf, x_of_t, U, dHdx_num, opts['odeOptions'], opts['interp'])
    J = computeCost(tGrid, X, U, g_num, Phi_num)
    
    # Package outputs
    sol = {
        't': tGrid,
        'tf': tGrid[-1],
        'X': X,
        'U': U,
        'P': P,
        'J': J,
        'J_hist': np.array(J_hist),
        'grad_norm_hist': np.array(grad_norm_hist)
    }
    
    info = {
        'iters': len(J_hist)
    }
    
    return sol, info


# Helpers

def setDefault(d, key, value):
    """Set default value in dict if key is missing or None."""
    if key not in d or d[key] is None:
        d[key] = value
    return d


def forwardSim(tGrid, x0, U, f_num, odeOptions, interpMode):
    """
    Forward simulation: integrate x_dot = f(x, u(t)).
    
    Returns:
        X: array [N x n] - state trajectory
        x_of_t: callable - interpolant for x(t)
    """
    u_of_t = makeInterpolant(tGrid, U, interpMode)
    
    # Default ODE options
    if odeOptions is None:
        odeOptions = {}
    rtol = odeOptions.get('RelTol', 1e-7)
    atol = odeOptions.get('AbsTol', 1e-9)
    
    def odefun(x, t):
        return np.asarray(f_num(x, u_of_t(t))).flatten()
    
    X = odeint(odefun, x0, tGrid, rtol=rtol, atol=atol)
    x_of_t = makeInterpolant(tGrid, X, 'linear')
    
    return X, x_of_t


def backwardSim(tGrid, pTf, x_of_t, U, dHdx_num, odeOptions, interpMode):
    """
    Backward simulation: integrate p_dot = -dH/dx.
    
    Returns:
        P: array [N x n] - costate trajectory
        p_of_t: callable - interpolant for p(t)
    """
    u_of_t = makeInterpolant(tGrid, U, interpMode)
    
    if odeOptions is None:
        odeOptions = {}
    rtol = odeOptions.get('RelTol', 1e-7)
    atol = odeOptions.get('AbsTol', 1e-9)
    
    def odefun(p, t):
        return -np.asarray(dHdx_num(x_of_t(t), u_of_t(t), p)).flatten()
    
    # Integrate backward from tf to 0
    tRev = tGrid[::-1]
    P_rev = odeint(odefun, pTf, tRev, rtol=rtol, atol=atol)
    P = P_rev[::-1]  # Reverse to match tGrid
    
    p_of_t = makeInterpolant(tGrid, P, 'linear')
    
    return P, p_of_t


def makeInterpolant(tGrid, Y, mode):
    """
    Create an interpolant function for Y(t).
    
    Args:
        tGrid: array [N] - time points
        Y: array [N x d] - values
        mode: str - 'linear' or 'zoh' (zero-order-hold)
    
    Returns:
        callable - interpolation function
    """
    tGrid = np.asarray(tGrid).flatten()
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    assert Y.shape[0] == len(tGrid), 'Y must have same number of rows as tGrid.'
    
    kind = 'previous' if mode.lower() == 'zoh' else 'linear'
    f = interp1d(tGrid, Y, kind=kind, axis=0, bounds_error=False, fill_value='extrapolate')
    
    def interp_wrapper(t):
        if np.isscalar(t):
            return f(t).flatten()
        else:
            return f(t)
    
    return interp_wrapper


def computeCost(tGrid, X, U, g_num, Phi_num):
    """
    Compute total cost J = Phi(x(tf)) + int g(x,u) dt.
    """
    N = len(tGrid)
    g_vals = np.zeros(N)
    for i in range(N):
        g_val = g_num(X[i], U[i])
        g_vals[i] = float(np.asarray(g_val).flat[0])
    
    intG = trapezoid(g_vals, tGrid)
    intG = float(np.asarray(intG).flat[0])
    
    # Terminal cost: handle both Phi(x) and Phi(x,t)
    if hasattr(Phi_num, '__code__'):
        nargs = Phi_num.__code__.co_argcount
    else:
        nargs = 1
    
    if nargs == 2:
        term = Phi_num(X[-1], tGrid[-1])
    else:
        term = Phi_num(X[-1])
    
    term = float(np.asarray(term).flat[0])
    J = float(term) + float(intG)
    return J


def projectTf(tf, tfLower, tfUpper):
    """Project tf to bounds."""
    tfp = tf
    if tfLower is not None:
        tfp = max(tfp, tfLower)
    if tfUpper is not None:
        tfp = min(tfp, tfUpper)
    return max(tfp, np.finfo(float).eps)


def resampleUOnNewTf(tGrid_old, U_old, tf_old, tf_new, interpMode):
    """
    Resample control to new final time while keeping N fixed.
    """
    N = len(tGrid_old)
    s = np.linspace(0, 1, N)
    tGrid_new = s * tf_new
    u_of_t_old = makeInterpolant(tGrid_old, U_old, interpMode)
    
    U_new = np.zeros_like(U_old)
    for i in range(N):
        U_new[i] = u_of_t_old(s[i] * tf_old)
    
    return tGrid_new, U_new


def projectU(U, uLower, uUpper):
    """Project each row of U to be within bounds."""
    Uproj = U.copy()
    m = U.shape[1]
    
    if uLower is not None:
        if np.isscalar(uLower):
            uLower_vec = np.full(m, uLower)
        else:
            uLower_vec = np.asarray(uLower).reshape(1, m)
        Uproj = np.maximum(Uproj, uLower_vec)
    
    if uUpper is not None:
        if np.isscalar(uUpper):
            uUpper_vec = np.full(m, uUpper)
        else:
            uUpper_vec = np.asarray(uUpper).reshape(1, m)
        Uproj = np.minimum(Uproj, uUpper_vec)
    
    return Uproj
