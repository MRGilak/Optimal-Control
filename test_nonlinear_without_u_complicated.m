clear; close all; clc;

%% Simulation Parameters
q1 = 2;
q2 = 1;
Q = diag([q1, q2]);
r = 0.1;
tf = 5;
dt = 0.001;
N = tf/dt;
x0 = [1; 0.1];

%% Original System
f = @(x, u) [x(2); -0.33*x(1)*exp(-x(1)) - 1.1*x(2) + u];
g = @(x, u) x'*Q*x + r*u^2;
Hgrad_u = @(x, u, p) 2*r*u + p(2);
Hgrad_p = @(x, u, p) [-(2*q1*x(1) -0.33*p(2)*exp(-x(1)) + 0.33*p(2)*(x(1))^2 * exp(-x(1))); ...
                            -(2*q2*x(2) + p(1) - 1.1*p(2))];

% Solve the optimal control problem
[~, x_true, u_true, p_true, ~] = forward_optimal_general(f, g, Hgrad_u, Hgrad_p, tf, dt, x0);
x1_true = x_true(1, :);
x2_true = x_true(2, :);
p1_true = p_true(1,: );
p2_true = p_true(2, :);

l = 3;
W = [q1; q2; 0];

%% Define the new system dynamics
A_func = @(x1, x2) [0, ((0.33*exp(-x1)) - (0.33*x1^2*exp(-x1))), -2*x1, 0, x2; ...
                   -1, 1.1, 0, -2*x2, x1; ...
                    0, 0, 0,   0, 0; ...
                    0, 0, 0,   0, 0; ...
                    0, 0, 0,   0, 0];

C_func = @(zeta1, zeta2, zeta5) [0, -1/(2*zeta5), 0, 0, 0];

% Discretize system
A = A_func(x1_true(1), x2_true(1));
Ad = expm(A*dt);        

%% Filter Parameters 
Qd = diag([0.001, 0.001, 0.001, 0.001, 0.001]);  % Discrete process noise covariance
R = 0.00001;

% Initialize
zeta_true = zeros(5, N);          % True state [p; W]
zeta_true(1, :) = p1_true;
zeta_true(2, :) = p2_true;
zeta_true(3:5, :) = W * ones(1, N);

y_meas = u_true;                  % Measurements
zeta_est = zeros(5, N);           % Estimated state
K_history = zeros(5, N);          % Kalman gain history

% Noise
w = sqrt(Qd) * randn(5, N);       % Process noise
v = sqrt(R) * randn(1, N);        % Measurement noise

% Initial guess
zeta_est(:, 1) = [2; -1; 0.1; 2.9; 1];  % Initial estimate
P = 10000 * eye(5);                        % Initial error covariance

% main loop
for k = 1:N-1
    A = A_func(x1_true(k), x2_true(k));
    Ad = expm(A*dt);

    % Prediction
    zeta_pred = Ad * zeta_est(:, k);
    P_pred = Ad * P * Ad' + Qd;
    
    % Update if measurement available
    H = C_func(zeta_pred(1), zeta_pred(2), r);

    K = zeros(4, 1);  % Default gain (no update)
    if ~isnan(y_meas(k))
        K = P_pred * H' / (H * P_pred * H' + R);
        zeta_est(:, k) = zeta_pred + K * (y_meas(k) - H * zeta_pred);
        P = (eye(5) - K * H) * P_pred;
    else
        zeta_est(:, k) = zeta_pred;
        P = P_pred;
    end
    
    % Store for next step and history
    zeta_est(:, k+1) = zeta_est(:, k);
    K_history(:, k) = K;
end
K_history(:, N) = K_history(:, N-1);  % Extend for plotting

%% Results
W_hat = zeta_est(3:5, N);
q1 = W_hat(1);
q2 = W_hat(2);
c = W_hat(3);
R = r;
Q = diag([q1, q2]);
g_new = @(x, u) x'*Q*x + R*u^2;
Hgrad_u_new = @(x, u, p) 2*R*u + p(2);
Hgrad_p_new = @(x, u, p) [-(2*q1*x(1) - c*x(2) - 0.33*p(2)*exp(-x(1)) + 0.33*p(2)*(x(1))^2 * exp(-x(1))); ...
                            -(2*q2*x(2) - c*x(1) + p(1) - 1.1*p(2))];

[~, x_est, u_est, p_est, J_history] = forward_optimal_general(f, g_new, Hgrad_u_new, Hgrad_p_new, tf, dt, x0);
x1_est = x_est(1, :); x2_est = x_est(2, :);
p1_est = p_est(1, :); p2_est = p_est(2, :);

%% Initial Guess
W_guess = zeta_est(3:5, 1);
q1 = W_guess(1);
q2 = W_guess(2);
c = W_guess(3);
R = r;
Q = diag([q1, q2]);
g_guess = @(x, u) x'*Q*x + R*u^2;
Hgrad_u_guess = @(x, u, p) 2*R*u + p(2);
Hgrad_p_guess = @(x, u, p) [-(2*q1*x(1) - c*x(2) - 0.33*p(2)*exp(-x(1)) + 0.33*p(2)*(x(1))^2 * exp(-x(1))); ...
                            -(2*q2*x(2) - c*x(1) + p(1) - 1.1*p(2))];

[t, x_guess, u_guess, p_guess, J_history_guess] = forward_optimal_general(f, g_guess, Hgrad_u_guess, Hgrad_p_guess, tf, dt, x0);
x1_guess = x_guess(1, :); x2_guess = x_guess(2, :);
p1_guess = p_guess(1, :); p2_guess = p_guess(2, :);

%% Compute performance measure
J_original = 0;
for i = 1:N
    J_original = J_original + g(x_true(:,i), u_true(i)) * dt;
end
J_new = 0;
for i = 1:N
    J_new = J_new + g(x_est(:,i), u_est(i)) * dt;
end
J_guess = 0;
for i = 1:N
    J_guess = J_guess + g(x_guess(:,i), u_guess(i)) * dt;
end

%% Display results
disp('Original Performance Measure: '); disp(J_original);
disp('Initial Guess Performance Measure: '); disp(J_guess);
disp('New Performance Measure: '); disp(J_new);
disp('Original Weights: '); disp(W);
disp('Initial Guess Weights: '); disp(W_guess);
disp('New Weights: '); disp(W_hat);

%% Plots
close all;

% x1
figure();
subplot(2, 1, 1);
plot(t, x1_true, 'b', 'LineWidth', 1.5); hold on;
plot(t, x1_est, 'r', 'LineWidth', 1.5);
plot(t, x1_guess, 'g', 'LineWidth', 1.5);
grid on;
xlabel('time (sec)');
ylabel('$x_1$', 'Interpreter', 'latex');
title('$x_1$ comparison', 'Interpreter', 'latex');
legend('$x_1$ true', '$x_1$ estimate', '$x_1$ initial guess', 'interpreter', 'latex');

% x2
subplot(2, 1, 2);
plot(t, x2_true, 'b', 'LineWidth', 1.5); hold on;
plot(t, x2_est, 'r', 'LineWidth', 1.5);
plot(t, x2_guess, 'g', 'LineWidth', 1.5);
grid on;
xlabel('time (sec)');
ylabel('$x_2$', 'Interpreter', 'latex');
title('$x_2$ comparison', 'Interpreter', 'latex');
legend('$x_2$ true', '$x_2$ estimate', '$x_2$ initial guess', 'interpreter', 'latex');

% p1
figure();
subplot(2, 1, 1);
plot(t, p1_true, 'b', 'LineWidth', 1.5); hold on;
plot(t, p1_est, 'r', 'LineWidth', 1.5);
plot(t, p1_guess, 'g', 'LineWidth', 1.5);
grid on;
xlabel('time (sec)');
ylabel('$p_1$', 'Interpreter', 'latex');
title('$p_1$ comparison', 'Interpreter', 'latex');
legend('$p_1$ true', '$p_1$ estimate', '$p_1$ initial guess', 'interpreter', 'latex');

% p2
subplot(2, 1, 2);
plot(t, p2_true, 'b', 'LineWidth', 1.5); hold on;
plot(t, p2_est, 'r', 'LineWidth', 1.5);
plot(t, p2_guess, 'g', 'LineWidth', 1.5);
grid on;
xlabel('time (sec)');
ylabel('$p_2$', 'Interpreter', 'latex');
title('$p_2$ comparison', 'Interpreter', 'latex');
legend('$p_2$ true', '$p_2$ estimate', '$p_2$ initial guess', 'interpreter', 'latex');

% u
figure();
plot(t, u_true, 'b', 'LineWidth', 1.5); hold on;
plot(t, u_est, 'r', 'LineWidth', 1.5);
plot(t, u_guess, 'g', 'LineWidth', 1.5);
grid on;
xlabel('time (sec)');
ylabel('$u$', 'Interpreter', 'latex');
title('$u$ comparison', 'Interpreter', 'latex');
legend('$u$ true', '$u$ estimate', '$u$ initial guess', 'interpreter', 'latex');

% zeta1
figure();
subplot(2, 1, 1);
plot(t, zeta_true(1, :), 'b', 'LineWidth', 1.5); hold on;
plot(t, zeta_est(1, :), 'r', 'LineWidth', 1.5);
grid on;
xlabel('time (sec)');
ylabel('$\zeta_1$', 'Interpreter', 'latex');
title('$\zeta_1$ comparison', 'Interpreter', 'latex');
legend('$\zeta_1$ true', '$\zeta_1$ estimate', 'interpreter', 'latex');

% zeta2
subplot(2, 1, 2);
plot(t, zeta_true(2, :), 'b', 'LineWidth', 1.5); hold on;
plot(t, zeta_est(2, :), 'r', 'LineWidth', 1.5);
grid on;
xlabel('time (sec)');
ylabel('$\zeta_2$', 'Interpreter', 'latex');
title('$\zeta_2$ comparison', 'Interpreter', 'latex');
legend('$\zeta_2$ true', '$\zeta_2$ estimate', 'interpreter', 'latex');

% zeta3
figure();
subplot(3, 1, 1);
plot(t, zeta_true(3, :), 'b', 'LineWidth', 1.5); hold on;
plot(t, zeta_est(3, :), 'r', 'LineWidth', 1.5);
grid on;
xlabel('time (sec)');
ylabel('$\zeta_3$', 'Interpreter', 'latex');
title('$\zeta_3$ comparison', 'Interpreter', 'latex');
legend('$\zeta_3$ true', '$\zeta_3$ estimate', 'interpreter', 'latex');

% zeta4
subplot(3, 1, 2);
plot(t, zeta_true(4, :), 'b', 'LineWidth', 1.5); hold on;
plot(t, zeta_est(4, :), 'r', 'LineWidth', 1.5);
grid on;
xlabel('time (sec)');
ylabel('$\zeta_4$', 'Interpreter', 'latex');
title('$\zeta_4$ comparison', 'Interpreter', 'latex');
legend('$\zeta_4$ true', '$\zeta_4$ estimate', 'interpreter', 'latex');

% zeta5
subplot(3, 1, 3);
plot(t, zeta_true(5, :), 'b', 'LineWidth', 1.5); hold on;
plot(t, zeta_est(5, :), 'r', 'LineWidth', 1.5);
grid on;
xlabel('time (sec)');
ylabel('$\zeta_5$', 'Interpreter', 'latex');
title('$\zeta_5$ comparison', 'Interpreter', 'latex');
legend('$\zeta_5$ true', '$\zeta_5$ estimate', 'interpreter', 'latex');
