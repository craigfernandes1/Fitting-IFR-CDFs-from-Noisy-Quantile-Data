import numpy as np
import sympy as sp
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB, nlfunc
from utils import linear_interpolation, schumaker_spline, discrete_interpolation
from scipy.stats import linregress

def fit_cdf(x, n, y, l, u, interpolation_method="linear", **kwargs):
    """Fit the IFR cdf estimator specified in Algorithm 1.

    Step 1 solves the finite-dimensional convex likelihood-maximization problem
    for the log-survival values at the observed knots.  Step 2 linearly
    interpolates those values in log-survival space and appends the analytic
    logarithmic tail required to obtain a proper IFR cdf.

    If the final observed knot has all successes, the finite program has only a
    supremum.  The implementation then fits through the final non-all-success
    knot and defines an analytic tail that reaches one at the first terminal
    all-success knot.
    """
    # The revised Algorithm 1 uses piecewise-linear interpolation in t-space.
    # Retaining this argument prevents old calls from silently using an
    # estimator different from the one now described in the manuscript.
    if interpolation_method != "linear":
        raise ValueError(
            "The revised Algorithm 1 implements only piecewise-linear "
            "log-survival interpolation. Use interpolation_method='linear'."
        )

    # Standardize the inputs before checking dimensions and numerical conditions.
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)

    # Each observed knot must have one sample size and one success count.
    if x.ndim != 1 or n.ndim != 1 or y.ndim != 1:
        raise ValueError("x, n, and y must be one-dimensional arrays.")
    if len(x) == 0 or len(x) != len(n) or len(x) != len(y):
        raise ValueError("x, n, and y must be nonempty arrays of equal length.")

    # Algorithm 1 is stated for finite lower and upper support endpoints.
    if not np.isfinite(l) or not np.isfinite(u) or l >= u:
        raise ValueError("This implementation requires finite endpoints with l < u.")

    # The likelihood knots must be strictly ordered.
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")

    # The artificial lower knot x_0 = l is added internally.  Every observed
    # knot must therefore be strictly inside the support; in particular, x_k<u
    # is needed for the ordinary analytic upper tail.
    if not (l < x[0] and x[-1] < u):
        raise ValueError("Observed knots must lie strictly inside (l, u).")

    # The binomial model requires positive integer sample sizes.
    if not np.all(n > 0) or not np.all(n == np.round(n)):
        raise ValueError("n must contain positive integers.")

    # The observed counts must be valid binomial success counts.
    if not np.all(y == np.round(y)) or not np.all((0 <= y) & (y <= n)):
        raise ValueError("y must contain integers satisfying 0 <= y_i <= n_i.")

    def solve_knot_problem(x_fit, n_fit, y_fit):
        """Solve the finite convex program on a selected prefix of the knots.

        The artificial lower knot (l, 0) is prepended here.  It represents the
        support condition F(l)=0, equivalently t(l)=log(1-F(l))=0.
        """
        # Add the artificial lower knot x_0=l.  No data are observed there.
        x_opt = np.concatenate(([l], x_fit))

        # t[i] represents log survival at x_opt[i].
        t = cp.Variable(len(x_opt))

        objective_terms = []

        # The finite log-likelihood is
        #
        #   sum_i y_i log(1-exp(t_i)) + (n_i-y_i)t_i.
        #
        # Zero-coefficient terms are omitted deliberately.  In particular,
        # y_i=0 permits t_i=0 and corresponds to the conventional value
        # 0*log(0)=0 rather than an artificial numerical domain restriction.
        for i in range(len(x_fit)):
            knot_index = i + 1

            # Include the success term only when y_i is positive.
            if y_fit[i] > 0:
                objective_terms.append(
                    y_fit[i] * cp.log1p(-cp.exp(t[knot_index]))
                )

            # Include the failure term only when n_i-y_i is positive.
            if n_fit[i] > y_fit[i]:
                objective_terms.append(
                    (n_fit[i] - y_fit[i]) * t[knot_index]
                )

        objective = cp.sum(objective_terms)

        # The cdf range restriction F(x) in [0,1] becomes t(x)<=0.
        # The lower endpoint restriction is t(l)=0.
        constraints = [t <= 0, t[0] == 0]

        # Since F is nondecreasing, t=log(1-F) must be nonincreasing.
        for i in range(len(x_opt) - 1):
            constraints.append(t[i] >= t[i + 1])

        # IFR is equivalent to concavity of t.  On an irregular knot grid,
        # concavity means that consecutive secant slopes are nonincreasing.
        #
        # The expression below is the cross-multiplied form of
        #
        #   (t_i-t_{i-1})/(x_i-x_{i-1})
        #       >= (t_{i+1}-t_i)/(x_{i+1}-x_i).
        #
        # Cross multiplication avoids unnecessary divisions in CVXPY.
        for i in range(1, len(x_opt) - 1):
            left_width = x_opt[i] - x_opt[i - 1]
            right_width = x_opt[i + 1] - x_opt[i]

            constraints.append(
                (t[i] - t[i - 1]) * right_width
                >= (t[i + 1] - t[i]) * left_width
            )

        # This is a concave maximization objective over a polyhedral feasible
        # region, hence a convex optimization problem in the usual convention.
        problem = cp.Problem(cp.Maximize(objective), constraints)

        # Allow a caller to choose a CVXPY solver or pass solver-specific options,
        # without exposing those options to the interpolation routine.
        solve_options = dict(kwargs.get("solver_options", {}))
        if kwargs.get("solver") is not None:
            solve_options["solver"] = kwargs["solver"]

        problem.solve(**solve_options)

        # An optimal-inaccurate status is accepted because exponential-cone
        # solvers can report it for numerically difficult but usable solutions.
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"Finite optimization failed with status: {problem.status}."
            )

        # Return both the augmented knot vector and its fitted log-survival values.
        return x_opt, np.asarray(t.value, dtype=float).reshape(-1)

    # The returned cdf uses this package-wide symbolic variable.
    x_sym = sp.symbols("x")

    # -------------------------------------------------------------------------
    # Ordinary branch: y_k < n_k.
    #
    # In this case the finite knot optimization has an attained optimum.  We
    # fit all observed knots and append an analytic tail on [x_k,u).
    # -------------------------------------------------------------------------
    if y[-1] < n[-1]:
        x_fit, t_fit = solve_knot_problem(x, n, y)

        # The last fitted knot is x_k.
        join_x = x_fit[-1]
        join_t = t_fit[-1]

        # Compute the final piecewise-linear slope.  The analytic tail must have
        # an initial slope no larger than this value to preserve concavity.
        previous_slope = (
            (t_fit[-1] - t_fit[-2]) / (x_fit[-1] - x_fit[-2])
        )

        # The manuscript's deterministic implementation rule:
        #
        #   a = max{1, -(u-x_k) * previous_slope}.
        #
        # It guarantees that -a/(u-x_k) <= previous_slope.
        tail_end = u
        a = max(1.0, -(tail_end - join_x) * previous_slope)

    # -------------------------------------------------------------------------
    # Terminal all-success branch: y_k = n_k.
    #
    # The original finite problem has a supremum but no finite optimizer because
    # the likelihood improves as t(x_k) tends to minus infinity.  We therefore
    # identify the final non-all-success knot and construct a finite estimator
    # that reaches F=1 at the next all-success knot.
    # -------------------------------------------------------------------------
    else:
        # Search only before the terminal knot.  The zero-based index j below
        # corresponds to the manuscript's last index satisfying y_j<n_j.
        eligible_indices = np.flatnonzero(y[:-1] < n[:-1])

        # Every observed knot is all-success.  Use the lower endpoint as the
        # joining point and let the analytic tail reach one at x_1.
        if len(eligible_indices) == 0:
            x_fit = np.array([l], dtype=float)
            t_fit = np.array([0.0], dtype=float)

            join_x = l
            join_t = 0.0
            tail_end = x[0]

            # The manuscript permits any positive a in this case; choose one
            # deterministically for reproducibility.
            a = 1.0

        # There is at least one non-all-success knot before the all-success run.
        else:
            # Fit only through the final such knot.  The selected final knot
            # satisfies y_j<n_j, so this smaller finite program attains an optimum.
            j = eligible_indices[-1]
            x_fit, t_fit = solve_knot_problem(
                x[:j + 1],
                n[:j + 1],
                y[:j + 1],
            )

            join_x = x_fit[-1]
            join_t = t_fit[-1]

            # The tail reaches minus infinity at the immediately following knot,
            # which is necessarily an all-success knot by construction.
            tail_end = x[j + 1]

            # Match the concavity condition at the join as in the ordinary branch.
            previous_slope = (
                (t_fit[-1] - t_fit[-2]) / (x_fit[-1] - x_fit[-2])
            )
            a = max(1.0, -(tail_end - join_x) * previous_slope)

    # Construct the PWL log-survival estimator on [l, join_x].
    observed_t = linear_interpolation(x_fit, t_fit)

    # On the analytic-tail interval, the cdf can be written directly as
    #   F(x) = 1 - exp(join_t) * ((tail_end - x)/(tail_end - join_x))**a.
    # This is algebraically equivalent to transforming
    #   t(x) = join_t + a log((tail_end-x)/(tail_end-join_x)).
    # The Max prevents NumPy from evaluating a negative base to a noninteger
    # power when lambdify eagerly evaluates this branch for x > tail_end.
    # On the actual tail interval x in (join_x, tail_end), it changes nothing.
    safe_tail_gap = sp.Max(tail_end - x_sym, 0)

    tail_cdf = 1 - sp.exp(join_t) * (
        safe_tail_gap / (tail_end - join_x)
    ) ** a

    # Assemble the full cdf.  The final branch covers both the finite upper
    # endpoint and, in the terminal-all-success case, every point after the
    # first terminal all-success knot.
    cdf = sp.Piecewise(
        (0, x_sym < l),
        (1 - sp.exp(observed_t), (x_sym >= l) & (x_sym <= join_x)),
        (tail_cdf, (x_sym > join_x) & (x_sym < tail_end)),
        (1, x_sym >= tail_end),
    )

    # Flatten the nested Piecewise expression for stable use with lambdify.
    return sp.piecewise_fold(cdf)

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
            300-second time limit.
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

