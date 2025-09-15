function [sol, info] = optimalControlSolver(symF, symG, symPhi, xSym, uSym, tGrid, x0, U0, opts)
    %% Validate Input Sizes and Set Defaults
    if nargin < 9, opts = struct(); end

    % Dimensions
    n = length(xSym);
    m = length(uSym);

    % Basic checks
    assert(isvector(xSym) && size(symF,1) == n && size(symF,2) == 1, 'symF must be [n x 1] matching xSym.');
    assert(isscalar(symG), 'symG must be a scalar symbolic expression.');
    assert(isscalar(symPhi), 'symPhi must be a scalar symbolic expression.');

    tGrid = tGrid(:); % ensure column
    N = numel(tGrid);
    assert(N >= 2 && all(diff(tGrid) > 0), 'tGrid must be strictly increasing with at least 2 points.');
    tf = tGrid(end);

    x0 = x0(:);
    assert(numel(x0) == n, 'x0 must be an [n x 1] vector.');

    U = U0;
    if isvector(U) && m == 1
        U = U(:);
    end
    assert(all(size(U) == [N, m]), 'U0 must have size [numel(tGrid) x m].');

    % Options defaults
    opts = setDefault(opts, 'maxIters', 50);
    opts = setDefault(opts, 'alpha', 1.0);
    opts = setDefault(opts, 'beta', 0.5);
    opts = setDefault(opts, 'c1', 1e-4);
    opts = setDefault(opts, 'tol', 1e-6);
    opts = setDefault(opts, 'odeOptions', []);
    opts = setDefault(opts, 'interp', 'linear');
    opts = setDefault(opts, 'uLower', []);
    opts = setDefault(opts, 'uUpper', []);
    opts = setDefault(opts, 'maxLineSearch', 10);
    opts = setDefault(opts, 'verbose', true);

    %% Build symbolic gradients and numeric function handles
    pSym = sym('p', size(xSym));                     % symbolic costate vector 

    % Hamiltonian gradients
    dGdx_sym = jacobian(symG, xSym).';               % [n x 1]
    dGdu_sym = jacobian(symG, uSym).';               % [m x 1]
    Jfx_sym  = jacobian(symF, xSym);                 % [n x n]
    Jfu_sym  = jacobian(symF, uSym);                 % [n x m]

    dHdx_sym = dGdx_sym + Jfx_sym.' * pSym;          % [n x 1]
    dHdu_sym = dGdu_sym + Jfu_sym.' * pSym;          % [m x 1]
    gradPhi_sym = jacobian(symPhi, xSym).';          % [n x 1]

    % Numeric function handles
    f_num = matlabFunction(symF, 'Vars', {xSym, uSym});
    g_num = matlabFunction(symG, 'Vars', {xSym, uSym});
    dHdx_num = matlabFunction(dHdx_sym, 'Vars', {xSym, uSym, pSym});
    dHdu_num = matlabFunction(dHdu_sym, 'Vars', {xSym, uSym, pSym});
    gradPhi_num = matlabFunction(gradPhi_sym, 'Vars', {xSym});

    % Prepare outputs
    J_hist = zeros(opts.maxIters,1);
    grad_norm_hist = zeros(opts.maxIters,1);

    % Helper for projection
    project = @(Ui) projectU(Ui, opts.uLower, opts.uUpper);

    % Prebuild terminal cost Phi(x)
    Phi_num = matlabFunction(symPhi, 'Vars', {xSym});

    % Initial forward pass to get a baseline cost
    [X, ~] = forwardSim(tGrid, x0, U, f_num, opts.odeOptions, opts.interp);
    J = computeCost(tGrid, X, U, g_num, Phi_num);

    if opts.verbose
        fprintf('Iter %3d | J = %.6e (initial)\n', 0, J);
    end

    %% main loop
    for k = 1:opts.maxIters
        % Forward: x(t)
        [X, x_of_t] = forwardSim(tGrid, x0, U, f_num, opts.odeOptions, opts.interp);

        % Backward: p(t) with terminal condition p(tf) = ∂Phi/∂x(x(tf))
        pTf = gradPhi_num(X(end,:).');
        [P, ~] = backwardSim(tGrid, pTf, x_of_t, U, dHdx_num, opts.odeOptions, opts.interp);

        % Compute gradient wrt u
        dHdu = zeros(N, m);
        for i = 1:N
            xi = X(i,:).';
            ui = U(i,:).';
            pi = P(i,:).';
            gi = dHdu_num(xi, ui, pi);
            dHdu(i,:) = gi.'; % row vector
        end

        grad_norm = norm(dHdu(:));
        grad_norm_hist(k) = grad_norm;

        % Cost at current iterate
        J = computeCost(tGrid, X, U, g_num, Phi_num);
        J_hist(k) = J;

        if opts.verbose
            fprintf('Iter %3d | J = %.6e | ||grad_u||_F = %.3e\n', k, J, grad_norm);
        end

        % Stopping criterion
        if grad_norm < opts.tol
            if opts.verbose
                fprintf('Converged: gradient norm below tol %.3e.\n', opts.tol);
            end
            break;
        end

        % Gradient descent step with backtracking line search (Armijo)
        alpha = opts.alpha;
        accepted = false;
        for ls = 1:opts.maxLineSearch
            U_try = project(U - alpha * dHdu);

            % Forward simulate to evaluate cost
            [X_try, ~] = forwardSim(tGrid, x0, U_try, f_num, opts.odeOptions, opts.interp);
            J_try = computeCost(tGrid, X_try, U_try, g_num, Phi_num);

            % Armijo condition: J(U - a grad) <= J(U) - c1 * a * ||grad||^2
            if J_try <= J - opts.c1 * alpha * (grad_norm^2)
                U = U_try;
                J = J_try;
                accepted = true;
                break;
            else
                alpha = alpha * opts.beta;
            end
        end

        if ~accepted
            % Could not find improving step; stop
            if opts.verbose
                fprintf('Line search failed to improve J; stopping.\n');
            end
            break;
        end
    end

    % Final forward/backward to report solution
    [X, x_of_t] = forwardSim(tGrid, x0, U, f_num, opts.odeOptions, opts.interp);
    pTf = gradPhi_num(X(end,:).');
    [P, ~] = backwardSim(tGrid, pTf, x_of_t, U, dHdx_num, opts.odeOptions, opts.interp);
    J = computeCost(tGrid, X, U, g_num, Phi_num);

    % Trim histories to performed iterations
    lastIter = find(J_hist ~= 0, 1, 'last');
    if isempty(lastIter), lastIter = 0; end
    J_hist = J_hist(1:lastIter);
    grad_norm_hist = grad_norm_hist(1:lastIter);

    % Package outputs
    sol = struct();
    sol.t = tGrid;
    sol.X = X;
    sol.U = U;
    sol.P = P;
    sol.J = J;
    sol.J_hist = J_hist;
    sol.grad_norm_hist = grad_norm_hist;

    info = struct();
    info.iters = lastIter;
end

% ===================== Helpers =====================

function S = setDefault(S, field, value)
	if ~isfield(S, field) || isempty(S.(field))
		S.(field) = value;
	end
end

function [X, x_of_t] = forwardSim(tGrid, x0, U, f_num, odeOptions, interpMode)
	% Interpolant for u(t)
	u_of_t = makeInterpolant(tGrid, U, interpMode);
	if isempty(odeOptions), odeOptions = odeset(); end

	% ODE: ẋ = f(x, u(t))
	odefun = @(t,x) f_num(x, u_of_t(t));
	[~, X] = ode45(odefun, tGrid, x0, odeOptions);
	x_of_t = makeInterpolant(tGrid, X, 'linear');
end

function [P, p_of_t] = backwardSim(tGrid, pTf, x_of_t, U, dHdx_num, odeOptions, interpMode)
	% Interpolants for x(t), u(t)
	u_of_t = makeInterpolant(tGrid, U, interpMode);

	% Backward ODE: ṗ = -∂H/∂x(x(t), u(t), p(t))
	if isempty(odeOptions), odeOptions = odeset(); end
	odefun = @(t,p) -dHdx_num(x_of_t(t), u_of_t(t), p);

	% Integrate backward: from tf to 0
	tRev = flipud(tGrid);
	[~, P_rev] = ode45(odefun, tRev, pTf, odeOptions);

	% The solver returns times in decreasing order; reverse to match tGrid
	P = flipud(P_rev);

	% Create interpolant p(t)
	p_of_t = makeInterpolant(tGrid, P, 'linear');
end

function u_of_t = makeInterpolant(tGrid, Y, mode)
	tGrid = tGrid(:);
	if size(Y,1) ~= numel(tGrid)
		error('Interpolant: Y must have same number of rows as tGrid length.');
	end
	switch lower(mode)
		case 'zoh'
			method = 'previous';
		otherwise
			method = 'linear';
	end
	u_of_t = @(t) interpRow(tGrid, Y, t, method);
end

function y = interpRow(tGrid, Y, t, method)
	yi = interp1(tGrid, Y, t, method, 'extrap'); % 1 x d or d vector
	if isrow(yi)
		y = yi.';
	else
		y = yi;
	end
end

function J = computeCost(tGrid, X, U, g_num, Phi_num)
	% Compute J = Phi(x(tf)) + ∫ g(x,u) dt via trapz using prebuilt Phi_num(x).

	N = size(X,1);
	g_vals = zeros(N,1);
	for i = 1:N
		g_vals(i) = g_num(X(i,:).', U(i,:).');
	end
	intG = trapz(tGrid, g_vals);
	term = Phi_num(X(end,:).');
	J = term + intG;
end

function Uproj = projectU(U, uLower, uUpper) 
    % Project each row of U to be within [uLower, uUpper] if bounds are given.

	Uproj = U;
	m = size(U,2);
	if ~isempty(uLower)
		if isscalar(uLower)
			uLower = repmat(uLower, 1, m);
		else
			uLower = reshape(uLower, 1, m);
		end
		Uproj = max(Uproj, repmat(uLower, size(U,1), 1));
	end
	if ~isempty(uUpper)
		if isscalar(uUpper)
			uUpper = repmat(uUpper, 1, m);
		else
			uUpper = reshape(uUpper, 1, m);
		end
		Uproj = min(Uproj, repmat(uUpper, size(U,1), 1));
	end
end

