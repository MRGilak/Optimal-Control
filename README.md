# Optimal-Control
This repo contains MATLAB codes for solving a general nonlinear optimal control problem using the gradient descent approach.

In optimal contorl theory a standard optimal control is defined as

![optimal control problem](https://quicklatex.com/cache3/4b/ql_f0a92f6c2743c22d543188b723fcfd4b_l3.png)

The solution to the problem above comes from calculus of variations. A Hamiltonian function is defined as

![Hamiltonian](https://quicklatex.com/cache3/dc/ql_b68dd567e3dc907186ed3255194ff0dc_l3.png)

And the optimal control input can be calcuated using the relationships below:

![optimal control solution](https://quicklatex.com/cache3/58/ql_86439e83486d7be8d7d7b1925d61d758_l3.png)

Here `p` denotes the costates. This set of equations is generally hard to solve, because they are two point boundary nonlinear equations. The initial values of `x` and the final values of `p` are known.
One way to solve this set of equations is using the gradient descent algorithm. An initial guess of the control input `u` is selected and the equations are solved for `x` and `p`, given the boundary values. Then `u` is corrected using the gradient of the Hamiltonian. 
This logic has been implemented in this function. 
You can write the Hamiltonian yourself and pass the necessary varriables to the function (numerical method) or you can define `f`, `g` and `h` symbolically and pass that to the optimal control solver.
Future versions will include:
- handling infinite horizon problems
- handling the final value being on a curve in space 

