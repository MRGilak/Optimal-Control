
%% test_optimalController_numeric.m
% Test optimalController using the numeric (function handle) method for a simple LTI system
% System:  dx/dt = Ax + Bu,  Cost: J = int(x'*Q*x + u'*R*u) + x(tf)'*H*x(tf)

% System parameters
A = [0 1; -1 -1];
B = [0; 1];
Q = eye(2);
H = 10 * eye(2);
R = 1;

x0 = [1; 1];
tf = 5;
dt = 0.01;
N = tf/dt;

% Dimensions
n = 2;
m = 1;

%% Function handles for system and cost
f = @(x, u) A*x + B*u;
g = @(x, u) x.'*Q*x + u.'*R*u;
h = @(x) x.'*H*x;

%% Hamiltonian and its gradients
Hfun = @(x, u, p) g(x, u) + p.'*f(x, u);
Hgrad_u = @(x, u, p) jacobian_numeric(@(uu) Hfun(x, uu, p), u);
Hgrad_p = @(x, u, p) -jacobian_numeric(@(xx) Hfun(xx, u, p), x);

% Terminal costate (gradient of h) as a function handle
p_end_fun = @(x) jacobian_numeric(h, x);

[t, x_traj, u_traj, p_traj, J_history] = optimalController(f, g, Hgrad_u, Hgrad_p, tf, dt, x0, p_end_fun);

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
%% Helper function: Central difference numerical Jacobian

% --- Helper function for numerical Jacobian 
function grad = jacobian_numeric(fun, x)
    % Central difference approximation for gradient
    eps = 1e-6;
    fx0 = fun(x);
    grad = zeros(length(x), length(fx0));
    for k = 1:length(x)
        x1 = x; x2 = x;
        x1(k) = x1(k) + eps;
        x2(k) = x2(k) - eps;
        grad(k,:) = (fun(x1) - fun(x2)) / (2*eps);
    end
    if size(grad,2) == 1
        grad = grad(:); % return as column for scalar output
    else
        grad = grad.'; % for vector output, transpose to match MATLAB's jacobian
    end
end
