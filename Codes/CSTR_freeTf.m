clear; clc;

% Symbols
syms x1 x2 u
x = [x1; x2];

% Parameters
R = 0.1;        % control weight in running cost
Tf0 = 0.78;     % initial guess for final time
N = 1001;       % time grid points

tGrid = linspace(0, Tf0, N).';

% Dynamics f(x,u)
expTerm = exp(25*x1/(x1+2));
f1 = -2*(x1 + 0.25) + (x2 + 0.5)*expTerm - (x1 + 0.25)*u;
f2 = 0.5 - x2 - (x2 + 0.5)*expTerm;
f = [f1; f2];

% Running cost g(x, u)
g = x1^2 + x2^2 + R*u^2;

% Terminal cost Phi(x)
Phi = sym(x1^2 + x2^2);
5
% Initial condition
x0 = [0.05; 0];

% Initial control guess U0
U0 = zeros(N, 1);

% Options (enable free-final-time)
opts = struct();
opts.maxIters = 1000;
opts.alpha = 1.0;
opts.beta = 0.5;
opts.c1 = 1e-4;
opts.tol = 1e-6;
opts.interp = 'linear';
opts.uLower = -2;       
opts.uUpper =  2;        
opts.odeOptions = odeset('RelTol',1e-7,'AbsTol',1e-9);
opts.verbose = true;

% Free-final-time settings
opts.freeFinalTime = true;
opts.tfAlpha = 0.5;      % step size for tf update
opts.tfBeta  = 0.5;      % backtracking factor
opts.tfC1    = 1e-4;     % Armijo constant for tf step
opts.tfLower = 0.2;      % optional bounds on tf
opts.tfUpper = 5.0;

% Call solver
[sol, info] = optimalControlSolver(f, g, Phi, x, u, tGrid, x0, U0, opts);

%% Plot results
figure('Name','States (free tf)','Color','w');
plot(sol.t, sol.X, 'LineWidth', 1.5); grid on;
xlabel('$t$', 'interpreter', 'latex');
ylabel('$x$', 'interpreter', 'latex');
title(sprintf('States (tf = %.4f)', sol.tf));
legend({'$x_1$','$x_2$'}, 'interpreter', 'latex');

figure('Name','Control Input (free tf)','Color','w');
plot(sol.t, sol.U, 'LineWidth', 1.5); grid on;
xlabel('$t$', 'interpreter', 'latex');
ylabel('$u$', 'interpreter', 'latex');
title('Control Input');

figure('Name','Costates (free tf)','Color','w');
plot(sol.t, sol.P, 'LineWidth', 1.5); grid on;
xlabel('$t$', 'interpreter', 'latex');
ylabel('$p$', 'interpreter', 'latex');
title('Costates');
legend({'$p_1$','$p_2$'}, 'interpreter', 'latex');

figure('Name','Optimization History (free tf)','Color','w');
subplot(2,1,1); plot(1:numel(sol.J_hist), sol.J_hist, '-o', 'LineWidth', 1.5);
grid on; xlabel('iteration'); ylabel('$J$', 'interpreter', 'latex'); title('Cost per iteration');
subplot(2,1,2); plot(1:numel(sol.grad_norm_hist), sol.grad_norm_hist, '-o', 'LineWidth', 1.5);
grid on; xlabel('iteration'); ylabel('$\|dH/du\|_F$', 'interpreter', 'latex'); title('Gradient norm per iteration');

fprintf('CSTR free-tf demo complete. Final cost J = %.6f, tf = %.6f, iterations = %d\n', sol.J, sol.tf, info.iters);
