%% test_optimalController_symbolic.m
% Test optimalController using the symbolic method for a simple LTI system
% System:  dx/dt = Ax + Bu,  Cost: J = int(x'*Q*x + u'*R*u) + x(tf)'*H*x(tf)

% Symbolic variables
syms x1 x2 real
syms u [1 1] real % symbolic vector (m=1)
x = [x1; x2];
% System definition
A = [0 1; -1 -1];
B = [0; 1];
f = A*x + B*u;
% Quadratic cost and terminal cost
Q = eye(2);
H = 10 * eye(2);
R = 1;
g = x.'*Q*x + u.'*R*u;
h = x.'*H*x;
% Symbolic variables
syms x1 x2 real
syms u [1 1] real % define u as a symbolic vector (m=1)
x = [x1; x2];

%% System: dx/dt = Ax + Bu
A = [0 1; -1 -1];
B = [0; 1];
f = A*x + B*u;

%% Quadratic cost: g(x,u) = x'*Q*x + u'*R*u, J = \int g(x,u)
Q = eye(2);
H = 10 * eye(2);
R = 1;
g = x.'*Q*x + u.'*R*u;
h = x.'*H*x;

%% Initial condition and simulation parameters
x0 = [1; 1];
% Initial condition and simulation parameters
x0 = [1; 1];
tf = 5;
dt = 0.01;

[t, x_traj, u_traj, p_traj, J_history] = optimalController(f, g, [], [], tf, dt, x0, h);

%% Plot results 

% State trajectories
figure;
plot(t, x_traj(1,:), 'LineWidth', 1.5, 'Color', [0.85 0.33 0.10]); hold on;
plot(t, x_traj(2,:), 'LineWidth', 1.5, 'Color', [0.00 0.45 0.74]);
xlabel('Time (s)'); ylabel('States $x_1$, $x_2$', 'interpreter', 'latex');
title('State Trajectories');
legend('$x_1$','$x_2$', 'interpreter', 'latex');
grid on;

% Control input
figure;
if size(u_traj,1) == 1
	plot(t, u_traj, 'LineWidth', 1.5, 'Color', [0.47 0.67 0.19]);
	legend('$u$', 'interpreter', 'latex');
else
	plot(t, u_traj', 'LineWidth', 1.5);
	legend(arrayfun(@(i) sprintf('$u_{%d}$',i), 1:size(u_traj,1), 'UniformOutput', false), 'interpreter', 'latex');
end
xlabel('Time (s)'); ylabel('$u$', 'interpreter', 'latex');
title('Control Input');
grid on;

% Costate trajectories
figure;
plot(t, p_traj(1,:), 'LineWidth', 1.5, 'Color', [0.49 0.18 0.56]); hold on;
plot(t, p_traj(2,:), 'LineWidth', 1.5, 'Color', [0.93 0.69 0.13]);
xlabel('Time (s)'); ylabel('Costates $p_1$, $p_2$', 'interpreter', 'latex');
title('Costate Trajectories');
legend('$p_1$','$p_2$', 'interpreter', 'latex');
grid on;

% Cost history
figure;
plot(1:length(J_history), J_history, 'LineWidth', 1.5, 'Color', [0.30 0.75 0.93]);
xlabel('Iteration'); ylabel('Cost $J$', 'interpreter', 'latex');
title('Cost History');
grid on;