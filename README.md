 # Optimal-Control
This repo contains MATLAB codes for solving a general nonlinear optimal control problem using the gradient descent approach.

In optimal contorl theory a standard optimal control is defined as

$$
   \begin{align}
      \text{minimize} &\phi(x(t_f) + \int_0^{t_f} g(x,u) dt \\
      s.t. &\dot{x} = f(x,u)
   \end{align}
$$

The solution to the problem above comes from calculus of variations. A Hamiltonian function is defined as

$$
   \mathcal{H} = g(x,u) + p^T f(x,u)
$$

And the optimal control input, in absence of input limits, can be calcuated using the relationships below:

$$
   \begin{align}
      \dot{x} = \frac{\partial \mathcal{H}}{\partial p} = f(x,u) \\
      \dot{p} = - \frac{\partial \mathcal{H}}{\partial x} \\
      \frac{\partial \mathcal{H}}{\partial u} = 0 
   \end{align}
$$

Here `p` denotes the costates. This set of equations is generally hard to solve, because they are two point boundary nonlinear equations. The initial values of `x` and the final values of `p` are known.
One way to solve this set of equations is using the gradient descent algorithm. An initial guess of the control input `u` is selected and the equations are solved for `x` and `p`, given the boundary values. Then `u` is corrected using the gradient of the Hamiltonian. 

## Free Final Time Problems

When the final time $t_f$ is not specified and can be freely chosen to minimize the cost, an additional optimality condition must be satisfied. This condition is known as the _transversality condition_ for free final time.

For a problem with free $t_f$, the total cost is:

$$
   J = \phi(x(t_f), t_f) + \int_0^{t_f} g(x,u) \, dt
$$

At the optimal solution, not only must the control satisfy $\frac{\partial \mathcal{H}}{\partial u} = 0$ (in unconstrained input case), but the final time must also satisfy:

$$
   \frac{dJ}{dt_f} = \mathcal{H}(x(t_f), u(t_f), p(t_f)) + \frac{\partial \phi}{\partial t}\bigg|_{t_f} = 0
$$

This condition arises because when $t_f$ is perturbed by a small amount $\delta t_f$, the cost changes due to two effects:  
- the terminal cost changes as $\frac{\partial \phi}{\partial t} \delta t_f$
-  the integral cost changes by adding a time slice of width $\delta t_f$ with integrand $g + p^T f = \mathcal{H}$.

At optimality, these combined effects must sum to zero.

The solver implements this by performing alternating optimization: first updating the control `u` via gradient descent on $\frac{\partial \mathcal{H}}{\partial u}$, then updating the final time $t_f$ via gradient descent on $\frac{dJ}{dt_f}$. Both steps use Armijo backtracking line search to ensure cost reduction. For more details on the transversality condition, see the [Wikipedia article on Pontryagin's Maximum Principle](https://en.wikipedia.org/wiki/Pontryagin%27s_maximum_principle).

This exact logic has been implemented in the function [optimalControlSolver](Codes/optimalControlSolver.m). Here, we go over the variables, inputs and outputs of the function.

Problem:
   
   $$
   \begin{aligned}
    \text{minimize}& J = Phi(x(tf)) + ∫_0^{tf} g(x(t), u(t)) dt \\
    \text{subject to}& ẋ = f(x,u),\quad x(0) = x0
   \end{aligned}
   $$

 Usage:
  ```
   [sol, info] = optimalControlSolver(symF, symG, symPhi, xSym, uSym, tGrid, x0, U0, opts)
 ```
 
 Inputs:
- `symF`   : symbolic vector field f(x,u) of size [n x 1]
- `symG`   : symbolic scalar running cost g(x,u)
- `symPhi` : symbolic scalar terminal cost Phi(x)
- `xSym`   : symbolic state vector [x1; x2; ...; xn]
- `uSym`   : symbolic control vector [u1; u2; ...; um]
- `tGrid`  : time grid (column or row) of size [N x 1] or [1 x N], increasing, with tGrid(1) = 0
- `x0`     : initial state (numeric) [n x 1]
- `U0`     : initial control trajectory over tGrid [N x m]
- `opts`   : options struct (all optional fields):
   - `maxIters`      (default 50)
   - `alpha`         (default 1.0)     initial step size for gradient descent
   - `beta`          (default 0.5)     backtracking reduction factor (0<beta<1)
   - `c1`            (default 1e-4)    Armijo condition constant
   - `tol`           (default 1e-6)    stopping tolerance on ||grad_u||_F
   - `odeOptions`    (default [])      options set by odeset
   - `interp`        (default 'linear') 'linear' or 'zoh' for u/x interpolation
   - `uLower`        (default [])      lower bounds on u (1x m) or scalar
   - `uUpper`        (default [])      upper bounds on u (1x m) or scalar
   - `maxLineSearch` (default 10)
   - `verbose`       (default true)

Outputs:
- `sol.t`   : time grid [N x 1]
- `sol.X`   : state trajectory along tGrid [N x n]
- `sol.U`   : control trajectory along tGrid [N x m]
- `sol.P`   : costate trajectory along tGrid [N x n]
- `sol.J`   : final cost value at solution
- `sol.J_hist` : cost history per iteration
- `sol.grad_norm_hist` : gradient-norm history per iteration
- `info.iters` : number of iterations performed

Requirements:
- MATLAB Symbolic Math Toolbox
- Free time support has been added. You can use the available options to set it up. Take a look at [Free time CSTR](Codes/CSTR_freeTf.m) to see how to use it 

Notes:
- Instead of a simple gradient descent with a constant step size, the Armijo condition is checked every time and backtracking is used to find an appropriate step size
- Three sample scripts have been provided:
   1. [demo.m](Codes/demo.m) contains a linear system with two states and two inputs.
   2. [CSTR.m](Codes/CSTR.m) solves the optimal control problem for a CSTR system (example 6.2-2 from [Kirk's book](https://books.google.com/books?hl=en&lr=&id=fCh2SAtWIdwC&oi=fnd&pg=PA1&dq=optimal+control+Kirk&ots=xziIYkb-e4&sig=fJ9SZLSPHDYESgw6TMzxO8K2HBo)).
   3. [Free time CSTR](Codes/CSTR_freeTf.m) solves the same optimal control problem as [CSTR.m](Codes/CSTR.m), but with free final time 
