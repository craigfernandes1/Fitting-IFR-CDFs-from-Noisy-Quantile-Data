import numpy as np
import sympy as sp
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB, nlfunc
from utils import linear_interpolation, schumaker_spline, discrete_interpolation
from scipy.stats import linregress

def fit_cdf(x, n, y, l, u, interpolation_method='linear', **kwargs):
    """Fit an IFR CDF to noisy quantile data.

    Applies the log-survival transform t(x) = ln(1 - F(x)), which converts the
    increasing-failure-rate (IFR) shape constraint into concavity of t. A convex
    maximum-likelihood problem is solved for t at the observed knots, and the
    result is back-transformed via F(x) = 1 - exp(t(x)).

    Args:
        x (np.ndarray): Strictly increasing observation points.
        n (np.ndarray): Number of Bernoulli trials at each observation point.
        y (np.ndarray): Number of successes (observations <= x) at each point.
        l (float): Lower bound of the CDF support; F(l) = 0 is enforced.
        u (float): Upper bound of the CDF support; F(u) = 1 is enforced.
        interpolation_method (str): Interpolation strategy for t between knots.
            One of ``'linear'``, ``'schumaker'``, or ``'discrete'``.
            Defaults to ``'linear'``.
        **kwargs: Additional options passed to the interpolation routine.
            terminal_M (float): Large positive constant M such that t(u) = -M,
                making F(u) ≈ 1. Defaults to 50.
            num_subdivisions (int): Grid density for ``'discrete'`` interpolation.
            return_type (str): ``'values'`` or ``'sympy'`` (for ``'discrete'`` only).

    Returns:
        sympy.Piecewise: Fitted CDF as a symbolic piecewise expression, when
            ``interpolation_method`` is ``'linear'`` or ``'schumaker'``, or when
            ``interpolation_method='discrete'`` and ``return_type='sympy'``.
        tuple[np.ndarray, np.ndarray]: ``(x_dense, F_dense)`` arrays on a fine
            grid, when ``interpolation_method='discrete'`` and
            ``return_type='values'`` (the default for discrete).

    Raises:
        ValueError: If ``interpolation_method`` is not one of the supported options.
        AssertionError: If the input arrays fail any sanity check (length mismatch,
            non-integer counts, out-of-domain points, etc.).
    """

    # data sanity checks
    assert len(x) == len(n) == len(y), "x, n, and y must have the same length"
    assert np.all(np.diff(x) > 0), "x must be strictly increasing"
    assert l < u, "l must be less than u"
    assert np.all(n > 0), "n must be positive"
    assert np.all(y >= 0), "y must be non-negative"
    assert np.all(y <= n), "y must be less than or equal to n"
    assert np.all(x >= l), "x must be greater than or equal to l"
    assert np.all(x <= u), "x must be less than or equal to u"
    assert np.all(n == np.round(n)), "n must be integer"
    assert np.all(y == np.round(y)), "y must be integer"

    # Use original x, n, y for the optimization part
    x_opt = x.copy()
    n_opt = n.copy()
    y_opt = y.copy()

    # Add l as the first knot if not already present (t(l)=0 is fixed).
    # u is NOT added here — the terminal segment [x_k, u] is handled in Step 2.
    if not np.isclose(l, x_opt[0]):
        x_opt = np.insert(x_opt, 0, l)
        n_opt = np.insert(n_opt, 0, 0)
        y_opt = np.insert(y_opt, 0, 0)


    # STEP 1: SOLVE THE CONVEX OPTIMIZATION PROBLEM

    k = len(x_opt)
    t = cp.Variable(k)

    objective = cp.sum(cp.multiply(y_opt, cp.log1p(-cp.exp(t))) + cp.multiply(n_opt - y_opt, t))
    constraints = []

    # t_i <= 0
    constraints += [t <= 0]

    # Monotonicity: t_i >= t_{i+1}
    for i in range(k - 1):
        constraints += [t[i] >= t[i + 1]]

    # IFR: t is concave, or slope must be non-increasing
    # (t[i] - t[i-1]) / (x[i] - x[i-1]) >= (t[i+1] - t[i]) / (x[i+1] - x[i])
    for i in range(1, k - 1):
        # Multiply crosswise to avoid division: (t[i] - t[i-1]) * (x[i+1]-x[i]) >= (t[i+1] - t[i]) * (x[i]-x[i-1])
        constraints += [(t[i] - t[i-1]) * (x_opt[i+1] - x_opt[i]) >= (t[i+1] - t[i]) * (x_opt[i] - x_opt[i-1])]

    # Endpoints: fix t(l) = 0.  t(x_k) is free (the terminal segment to u is
    # handled in Step 2 by appending (u, -M) for linear/Schumaker).
    constraints += [t[0] == 0]

    # Solve the problem
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()

    t_vals = t.value

    # STEP 2: INTERPOLATE t(x) and return the CDF
    # The optimization covers [l, x_k].  For linear and Schumaker we append the
    # terminal point (u, -M) so the final segment extrapolates t -> -M (F -> 1).
    # M=50 gives F(u) = 1 - e^{-50} ~ 2e-22, indistinguishable from 1.
    # Discrete interpolation covers [l, x_k] only; F(u)=1 is enforced explicitly.
    M = kwargs.get('terminal_M', 50)

    if interpolation_method == 'linear':
        x_interp = np.append(x_opt, u)
        t_interp = np.append(t_vals, -M)
        t_func = linear_interpolation(x_interp, t_interp, **kwargs)
        cdf = sp.piecewise_fold(1 - sp.exp(t_func))
        return cdf
    elif interpolation_method == 'schumaker':
        x_interp = np.append(x_opt, u)
        t_interp = np.append(t_vals, -M)
        t_func = schumaker_spline(x_interp, t_interp, **kwargs)
        cdf = sp.piecewise_fold(1 - sp.exp(t_func))
        return cdf
    elif interpolation_method == 'discrete':
        discrete_interp = discrete_interpolation(x_opt, t_vals, x_end=u, **kwargs)
        if kwargs.get('return_type') == 'sympy':
            t_func = discrete_interp
            cdf = sp.piecewise_fold(1 - sp.exp(t_func))
            return cdf
        else:
            x_dense, t_dense = discrete_interp
            F_dense = 1 - np.exp(t_dense)
            F_dense = np.clip(F_dense, 0, 1)
            F_dense[0] = 0.0   # F(l) = 0
            F_dense[-1] = 1.0  # F(x_k) declared as 1 (x_k is the last observed knot)
            return x_dense, F_dense
    else:
        raise ValueError(f"Invalid interpolation method: {interpolation_method}. Choose 'linear', 'schumaker', or 'discrete'.")


def fit_cdf_nonconvex(x, n, y, l, u, d, ifr_constraint=False, verbose=False, return_type='sympy'):
    """Fit a CDF to noisy quantile data using a nonconvex Gurobi solver.

    Discretizes [l, u] into a fine grid controlled by ``d`` and solves a
    nonlinear maximum-likelihood problem directly in F-space. An optional IFR
    constraint is enforced via bilinear (nonconvex quadratic) inequalities.
    Requires a Gurobi license with nonlinear/bilinear support.

    Args:
        x (np.ndarray): Observation points (must lie strictly between l and u).
        n (np.ndarray): Number of Bernoulli trials at each observation point.
        y (np.ndarray): Number of successes (observations <= x) at each point.
        l (float): Lower bound of the CDF support; F(l) = 0 is enforced.
        u (float): Upper bound of the CDF support; F(u) = 1 is enforced.
        d (int): Number of equidistant points inserted between consecutive knots.
            ``d=0`` uses only the original knots; ``d=1`` inserts one midpoint
            per interval. Higher values give a smoother solution at the cost of
            a larger optimization problem.
        ifr_constraint (bool): If ``True``, enforce the increasing-failure-rate
            (IFR) property via bilinear constraints. Defaults to ``False``.
        verbose (bool): If ``True``, display Gurobi solver output. Defaults to
            ``False``.
        return_type (str): Output format. ``'values'`` returns numeric arrays;
            ``'sympy'`` returns a symbolic piecewise step function.
            Defaults to ``'sympy'``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(x_grid, F_values)`` when
            ``return_type='values'``.
        sympy.Piecewise: Fitted CDF as a symbolic step function when
            ``return_type='sympy'``.

    Raises:
        RuntimeError: If Gurobi fails to find a feasible solution within the
            600-second time limit.
        ValueError: If ``return_type`` is not ``'values'`` or ``'sympy'``.
    """
    model = gp.Model()
    model.setParam('OutputFlag', verbose)

    grid = np.concatenate(([l], x, [u]))
    fine_grid = np.concatenate(
        [np.linspace(grid[i], grid[i + 1], d + 1, endpoint=False) for i in range(len(grid) - 1)])
    n_points = len(fine_grid)
    # Decision variables: F(i) for i from l to u with fine grid
    F = model.addVars(n_points+1, lb=0, ub=1, name="F")

    # Fix F(l) = 0 and F(u) = 1
    F[0].lb = 0
    F[0].ub = 0
    F[n_points].lb = 1
    F[n_points].ub = 1

    # Monotonicity constraints: F(i) <= F(i+1)
    for i in range(n_points):
        model.addConstr(F[i] <= F[i + 1])

    # IFR constraint (rearranged to bilinear, ie nonconvex quadratic)
    if ifr_constraint:
        for i in range(1, n_points):
            model.addConstr(
                (F[i] - F[i - 1]) * (1 - F[i]) <= (F[i + 1] - F[i]) * (1 - F[i - 1]),
                name=f"ifr_quad_{i}"
            )


    # Objective: log-likelihood
    # In gurobi, to do nonlinear objective, have to introduce a new variable and make a nonlinear constraint
    z = model.addVar(name="z", lb=-GRB.INFINITY, ub=GRB.INFINITY)

    # we will appropriately bound z to help the solver
    # upper bound is the usual binomial log-likelihood
    n_arr, y_arr = np.array(n), np.array(y)
    p_hat = np.clip(y_arr/n_arr, 1e-8, 1 - 1e-8) # if its exactly 0 or 1, we clip it to avoid log(0)
    z_max = np.sum(y_arr * np.log(p_hat) + (n_arr - y_arr) * np.log(1 - p_hat))
    model.addConstr(z <= z_max)
    # lower bound is...

    # lastly define z as the log-likelihood
    model.addConstr(z == gp.quicksum(
        y[i] * nlfunc.log(F[(i+1)*(d+1)]) + (n[i] - y[i]) * nlfunc.log(1 - F[(i+1)*(d+1)])
        for i in range(len(x))
    ))

    model.setObjective(z, GRB.MAXIMIZE)
    model.setParam("TimeLimit", 300)

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(
            f"Gurobi failed to find a solution (status={model.Status}). "
            "Consider increasing TimeLimit or relaxing constraints."
        )

    estimated_cdf = np.array([F[i].X for i in range(n_points + 1)])
    if return_type == 'values':
        return np.append(fine_grid, u), estimated_cdf
    elif return_type == 'sympy':
        # Create a sympy step function for plotting discrete CDF
        x_sym = sp.symbols('x')
        pieces = []
        pieces.append((estimated_cdf[0], x_sym <= fine_grid[0]))
        for i in range(n_points - 1):
            pieces.append((estimated_cdf[i+1], (x_sym > fine_grid[i]) & (x_sym <= fine_grid[i+1])))
        pieces.append((estimated_cdf[-1], x_sym > fine_grid[-1]))
        pieces.insert(0, (0, x_sym < fine_grid[0]))
        pieces.append((1, x_sym > fine_grid[-1]))
        return sp.Piecewise(*pieces, (0, True))
    else:
        raise ValueError(f"Invalid return type: {return_type}. Choose 'sympy' or 'values'.")


def fit_weibull_cdf(x, n, y, l, u, x_sym):
    """Fit a truncated Weibull CDF to noisy quantile data.

    Estimates Weibull shape and scale parameters by log-log linearization of the
    empirical quantiles followed by ordinary least-squares regression on log(x).
    The resulting Weibull CDF is normalized to the truncated support [l, u] so
    that F(l) = 0 and F(u) = 1.

    Args:
        x (np.ndarray): Strictly increasing observation points.
        n (np.ndarray): Number of Bernoulli trials at each observation point.
        y (np.ndarray): Number of successes (observations <= x) at each point.
        l (float): Lower bound of the CDF support.
        u (float): Upper bound of the CDF support.
        x_sym (sympy.Symbol): Symbolic variable used in the returned expression.

    Returns:
        sympy.Expr: Fitted truncated Weibull CDF as a symbolic expression in
            ``x_sym``.
    """

    # linear regression set up
    F_hat = np.clip(y / n, 1e-8, 1 - 1e-8)  # if its exactly 0 or 1, we clip it to avoid log(0)
    y_regress = np.log(-np.log(1 - F_hat))
    res = linregress(np.log(x), y_regress)

    # recover parameters
    sh_hat = res.slope
    lamb_hat = np.exp(-res.intercept / res.slope)

    # create sympy representation of estimated weibull
    weibull_cdf_result_nontrunc = 1 - sp.exp(-(x_sym / lamb_hat) ** sh_hat)
    # Normalize the CDF for truncation support (0, u)
    F_l = weibull_cdf_result_nontrunc.subs(x_sym, l)  # CDF at 0 (always 0 for Weibull)
    F_u = weibull_cdf_result_nontrunc.subs(x_sym, u)  # CDF at u
    weibull_cdf_result = (weibull_cdf_result_nontrunc - F_l) / (F_u - F_l)

    return weibull_cdf_result

