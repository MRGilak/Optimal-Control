clear; clc;

% Symbols
syms x1 x2 u1 u2
x = [x1; x2];
u = [u1; u2];

% Dynamics f(x,u)
A = [0 1; -1 0];   
B = eye(2);        
f = A*x + B*u;     

% Running cost g(x,u)
Q = diag([1, 1]);
R = diag([0.1, 0.1]);
g = x.'*Q*x + u.'*R*u;  

% Terminal cost Phi(x)
Phi = 10*(x.'*x);

% Time grid and initial conditions
T = 5.0;
N = 501;
tGrid = linspace(0, T, N).';
x0 = [1; 0];

% Initial control guess U0
U0 = zeros(N, 2); 

% Options
opts = struct();
opts.maxIters = 50;
opts.alpha = 1.0;
opts.beta = 0.5;
opts.c1 = 1e-4;
opts.tol = 1e-6;
opts.interp = 'linear';
opts.uLower = [];           
opts.uUpper = [];
opts.odeOptions = odeset('RelTol',1e-7,'AbsTol',1e-9);
opts.verbose = true;

% Solve
[sol, info] = optimalControlSolver(f, g, Phi, x, u, tGrid, x0, U0, opts);

%% Plots
figure('Name','states','Color','w');
plot(sol.t, sol.X, 'LineWidth', 1.5);
grid on;
xlabel('$t$', 'Interpreter', 'latex'); 
ylabel('$x$', 'Interpreter', 'latex');
title('States');
legend(arrayfun(@(i) sprintf('x_%d',i), 1:size(sol.X,2), 'UniformOutput', false));

figure('Name','Control Inputs','Color','w');
plot(sol.t, sol.U, 'LineWidth', 1.5);
grid on; xlabel('$t$', 'Interpreter', 'latex');
ylabel('$u$', 'Interpreter', 'latex');
title('Controls');
legend(arrayfun(@(i) sprintf('u_%d',i), 1:size(sol.U,2), 'UniformOutput', false));

figure('Name','Costates','Color','w');
plot(sol.t, sol.P, 'LineWidth', 1.5);
grid on; xlabel('$t$', 'Interpreter', 'latex');
ylabel('$p$', 'Interpreter', 'latex');
title('Costates');
legend(arrayfun(@(i) sprintf('p_%d',i), 1:size(sol.P,2), 'UniformOutput', false));

figure('Name','Optimization History','Color','w');
subplot(2,1,1);
plot(1:numel(sol.J_hist), sol.J_hist, '-o', 'LineWidth', 1.5);
grid on;
xlabel('iteration');
ylabel('$J$', 'Interpreter', 'latex');
title('Cost per iteration');

subplot(2,1,2);
plot(1:numel(sol.grad_norm_hist), sol.grad_norm_hist, '-o', 'LineWidth', 1.5);
grid on;
xlabel('iteration');
ylabel('$\|dH/du\|_F$', 'Interpreter', 'latex');
title('Gradient norm per iteration');

fprintf('Done. Final cost J = %.6f, iterations = %d\n', sol.J, info.iters);
