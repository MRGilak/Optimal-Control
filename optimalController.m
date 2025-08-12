function [t, x, u, p, J_history] = optimalController(f, g, Hgrad_u, Hgrad_p, tf, dt, x0, varargin)
    % OptimalController Finite-horizon optimal control for general nonlinear systems.
    %   Supports symbolic (f, g, h) or numeric (f, g, Hgrad_u, Hgrad_p, p_end) modes.
    %
    %   [t, x, u, p, J_history] = optimalController(f, g, Hgrad_u, Hgrad_p, tf, dt, x0, h_or_pend)
    %
    %   Inputs:
    %     f         - System dynamics (symbolic or function handle)
    %     g         - Running cost (symbolic or function handle)
    %     Hgrad_u   - (numeric mode) Hamiltonian gradient wrt u (function handle)
    %     Hgrad_p   - (numeric mode) Hamiltonian gradient wrt p (function handle)
    %     tf        - Final time
    %     dt        - Time step
    %     x0        - Initial state (numeric vector)
    %     h_or_pend - (symbolic) terminal cost h(x) (symbolic), (numeric) terminal costate or function handle
    %
    %   Outputs:
    %     t         - Time vector
    %     x         - State trajectory
    %     u         - Control trajectory
    %     p         - Costate trajectory
    %     J_history - Cost at each iteration

    if isa(f, 'sym') && isa(g, 'sym') && nargin >= 8 && isa(varargin{1}, 'sym')
        % Symbolic mode
        h = varargin{1};
        n = length(x0);
        u_syms = symvar(g);
        u_syms = setdiff(u_syms, sym('x', [n 1]));
        if isempty(u_syms)
            error('Could not infer symbolic control variable u. Please use a symbolic vector u in your cost and dynamics.');
        end

        m = length(u_syms); % control dimension
        syms x [n 1] real
        syms u [m 1] real
        syms p [n 1] real

        % Hamiltonian
        H = g + p.' * f;
        Hgrad_u = matlabFunction(jacobian(H, u), 'Vars', {x, u, p});
        Hgrad_p = matlabFunction(-jacobian(H, x).', 'Vars', {x, u, p});
        f = matlabFunction(f, 'Vars', {x, u});
        g = matlabFunction(g, 'Vars', {x, u});
        dhdx_fun = matlabFunction(jacobian(h, x).', 'Vars', {x});
        p_end_mode = 'symbolic';

    elseif isa(f, 'function_handle') && isa(g, 'function_handle') && isa(Hgrad_u, 'function_handle') && isa(Hgrad_p, 'function_handle') && nargin >= 8
        % Numeric mode
        p_end = varargin{1};
        p_end_mode = 'numeric';
        n = length(x0);
        
    else
        error('Inputs must be either all symbolic (f, g, h) or all numeric/function handles (f, g, Hgrad_u, Hgrad_p, p_end).');
    end

    N = tf/dt;
    t = linspace(0, tf, N);
    x = zeros(n, N); x(:, 1) = x0;
    p = zeros(n, N);
    if exist('m', 'var')
        u = 0.01 * ones(m, N);
    else
        u = 0.01 * ones(1, N);
    end
    max_iter = 1000;
    tau = 0.025;
    gamma = 1e-6;
    J_history = [];

    for iter = 1:max_iter
        % Forward integration (states)
        for i = 2:N
            x(:,i) = x(:,i-1) + dt * f(x(:,i-1), u(:,i-1));
        end

        % Backward integration (costates)
        if strcmp(p_end_mode, 'symbolic')
            p(:,end) = dhdx_fun(x(:,end));
        elseif strcmp(p_end_mode, 'numeric')
            if isa(p_end, 'function_handle')
                p(:,end) = p_end(x(:,end));
            else
                p(:,end) = p_end;
            end
        end
        for i = N-1:-1:1
            dp = Hgrad_p(x(:,i), u(:,i), p(:,i+1));
            p(:,i) = p(:,i+1) - dt * dp;
        end

        % Gradient descent update for control
        for i = 1:N
            grad_u = Hgrad_u(x(:,i), u(:,i), p(:,i));
            u(:,i) = u(:,i) - tau * grad_u;
        end

        % Compute performance
        J_val = 0;
        for i = 1:N
            J_val = J_val + g(x(:,i), u(:,i)) * dt;
        end
        J_history = [J_history J_val]; 
        % fprintf('Iteration %d: J = %.4f\n', iter, J_val);

        if length(J_history) > 1 && abs(J_history(end-1) - J_val) < gamma
            break;
        end
    end
end
