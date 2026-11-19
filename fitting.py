import numpy as np
import sympy as sp
import cvxpy as cp
from utils import linear_interpolation
from scipy.stats import linregress
from functools import reduce
from fractions import Fraction
from math import gcd as _math_gcd


def eval_step_cdf(x_grid, F_grid, x):
    """Evaluate a right-continuous step cdf at x.

    Uses the convention that the fitted cdf equals ``F_grid[j]`` on the
    half-open interval ``[x_grid[j], x_grid[j+1])``, with ``F = F_grid[-1]`` for
    ``x >= x_grid[-1]`` (``F_grid[-1] = 1`` by construction). This is the single
    shared evaluator for every step-cdf benchmark (``fit_cdf_nonconvex`` output):
    sup-norm error, downstream decisions, and plots.

    Args:
        x_grid (np.ndarray): Strictly increasing grid breakpoints.
        F_grid (np.ndarray): Step values, same length as ``x_grid``.
        x (float or np.ndarray): Query point(s).

    Returns:
        float or np.ndarray: Step-cdf value(s); float for scalar ``x``.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    F_grid = np.asarray(F_grid, dtype=float)
    idx = np.clip(np.searchsorted(x_grid, x, side='right') - 1, 0, len(F_grid) - 1)
    return F_grid[idx]


def l_inf_step_vs_cdf(x_grid, F_grid, true_cdf, x_max=None):
    """Sup-norm distance between a right-continuous step cdf and a monotone cdf.

    The step cdf is constant at ``F_grid[j]`` on ``[x_grid[j], x_grid[j+1])``, so
    the worst-case error on each interval is attained at one of its endpoints
    (the true cdf is monotone). Intervalwise evaluation therefore uses
    ``F_grid[:-1]`` against the true cdf at both endpoints. If ``x_max`` is given,
    only intervals starting before ``x_max`` are included and right endpoints are
    capped at ``x_max``.

    Args:
        x_grid (np.ndarray): Strictly increasing grid breakpoints.
        F_grid (np.ndarray): Step values, same length as ``x_grid``.
        true_cdf (callable): Vectorized true cdf accepting a NumPy array.
        x_max (float, optional): Upper cutoff for the comparison region.

    Returns:
        float: The sup-norm distance.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    F_grid = np.asarray(F_grid, dtype=float)
    step   = F_grid[:-1]
    x_left = x_grid[:-1]
    x_right = x_grid[1:]
    if x_max is not None:
        mask    = x_left < x_max
        step    = step[mask]
        x_left  = x_left[mask]
        x_right = np.minimum(x_right[mask], x_max)
    return float(np.max(np.maximum(
        np.abs(step - true_cdf(x_left)),
        np.abs(step - true_cdf(x_right)),
    )))


# Numerical tolerances shared by the estimators below. Conic solvers return knot
# values accurate to roughly 1e-8, so a constraint that is active at the optimum
# (t_i = 0, or t_i = t_{i+1}) comes back violated or satisfied by roundoff.
_ZERO_SNAP = 1e-6   # |tau_i| below this is treated as exactly 0 (F(x_i) = 0).
_SLOPE_TOL = 1e-6   # a fitted slope/exponent within this of 0 counts as flat.


def _clean_knot_values(t_val):
    """Project solver output onto the constraints it was meant to satisfy.

    Three roundoff artifacts matter downstream. A value slightly above zero gives
    ``F = 1 - exp(t) < 0``; a value of order ``1e-8`` where the optimum is exactly
    zero makes an "all-failure" knot look informative and can trigger the
    degenerate-case perturbations spuriously; and a monotonicity violation of
    order ``1e-8`` gives a locally decreasing cdf. The projection clips to
    ``t <= 0``, snaps near-zero values to zero, and enforces ``t`` nonincreasing
    by a running minimum. Each step moves values by at most the solver's own
    error, so concavity/convexity are preserved to the same order.
    """
    t_val = np.asarray(t_val, dtype=float).reshape(-1)
    t_val = np.minimum(t_val, 0.0)
    t_val[np.abs(t_val) < _ZERO_SNAP] = 0.0
    return np.minimum.accumulate(t_val)


def fit_cdf_ifr(x, n, y, l, u, interpolation_method="linear", **kwargs):
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

        # Return both the augmented knot vector and its fitted log-survival values,
        # projected onto the constraints to remove solver roundoff.
        return x_opt, _clean_knot_values(t.value)

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


def fit_cdf_dfr(x, n, y, l=0.0, epsilon=1e-3, **kwargs):
    """Fit a decreasing-failure-rate (DFR) cdf from noisy quantile data.

    DFR analogue of :func:`fit_cdf_ifr` (manuscript Section 5.1). DFR is equivalent to
    *convexity* of the log-survival transform ``t(x) = log(1 - F(x))`` (recall
    ``t'(x) = -h(x)``, so a decreasing hazard means a non-decreasing ``t'``, i.e.
    convex ``t``). The only structural change to the finite convex program of
    Algorithm 1 is therefore that the IFR concavity constraints on the knot values
    are replaced by convexity constraints; the binomial log-likelihood, the
    range constraint ``t <= 0``, the endpoint ``t(l) = 0``, and the monotonicity
    constraints are unchanged. The problem remains convex (a concave objective over
    a polyhedron).

    The reconstruction is piecewise-linear in t-space through the fitted knot
    values ``(x_i, tau_i)`` (including the artificial lower knot ``(l, 0)``),
    followed by a *linear* upper tail that continues the final fitted slope to
    minus infinity. Because the final slope is the largest (least negative) slope
    of a convex non-increasing ``t``, extending it as a straight line preserves
    both convexity and monotonicity while giving ``t(x) -> -inf`` and hence
    ``F(x) -> 1`` as ``x -> inf``. The returned cdf is therefore a proper DFR cdf
    on the unbounded support ``[l, inf)``.

    Unlike the IFR estimator, the terminal all-success case ``y_k = n_k`` needs no
    special handling. IFR concavity lets ``tau_k -> -inf`` (an unattained
    supremum), which is why :func:`fit_cdf_ifr` splices in a finite-support tail. DFR
    convexity instead bounds the final knot value from below -- the last secant
    slope cannot fall below the preceding one -- so the ordinary convex program
    attains an optimum with ``F(x_k) < 1``. This is correct: a DFR cdf cannot reach
    one at a finite point (a convex ``t`` cannot diverge to minus infinity at a
    finite ``x``), so an all-success knot is regularized toward the DFR shape
    rather than driven to ``F = 1``. The only configuration with no attained
    optimum is when *every* observed knot is all-success, for which the likelihood
    supremum (zero) is approached only as all ``t_i -> -inf``; that case is
    rejected.

    A convex tail decreasing to minus infinity needs a strictly negative final
    fitted slope (equivalently ``tau_{k-1} > tau_k``, with ``tau_0 = 0`` when
    ``k = 1``). If the final two fitted values coincide the slope is zero, and the
    manuscript's DFR discussion resolves this by perturbing the fit toward an
    exponential distribution. We implement that perturbation: the fitted knot
    values are tilted by a tiny exponential component, ``tau_i -> tau_i -
    epsilon * (x_i - l)`` (hazard ``epsilon`` measured from ``l``, so ``t(l) = 0``
    is preserved), which keeps ``t`` convex while making the final slope
    ``-epsilon < 0``. The log-likelihood loss vanishes as
    ``epsilon -> 0``; a smaller ``epsilon`` is closer to the optimum but yields a
    slower (flatter) tail that reaches ``F = 1`` only far out.

    Args:
        x (np.ndarray): Strictly increasing observed knots, each ``> l``.
        n (np.ndarray): Positive integer sample sizes at each knot.
        y (np.ndarray): Integer success counts with ``0 <= y_i <= n_i``.
        l (float): Finite lower support endpoint; ``F(l) = 0`` is enforced via the
            artificial lower knot ``(l, 0)``. Defaults to ``0.0``, the standard
            domain for the failure-rate extensions.
        epsilon (float): Exponential-perturbation hazard used only in the
            degenerate (flat final segment) case. It also sets the flatness
            threshold: the final segment counts as flat when its slope is at
            least ``-epsilon``, since solver output for a truly flat segment is
            only accurate to about ``1e-5``. Must be positive. Defaults to
            ``1e-3``.
        **kwargs: Optional ``solver`` (a CVXPY solver) and ``solver_options`` (a
            dict passed to ``problem.solve``), handled as in :func:`fit_cdf_ifr`.

    Returns:
        sympy.Piecewise: The fitted DFR cdf as a symbolic expression on
            ``[l, inf)``.

    Raises:
        ValueError: If the inputs are malformed, if every observed knot is
            all-success (no attained DFR optimum), or if ``epsilon <= 0``.
        RuntimeError: If the convex program fails to solve (e.g. an unbounded
            configuration).
    """
    # Standardize the inputs before checking dimensions and numerical conditions.
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)

    # Each observed knot must have one sample size and one success count.
    if x.ndim != 1 or n.ndim != 1 or y.ndim != 1:
        raise ValueError("x, n, and y must be one-dimensional arrays.")
    if len(x) == 0 or len(x) != len(n) or len(x) != len(y):
        raise ValueError("x, n, and y must be nonempty arrays of equal length.")

    # The support is [l, inf); only the lower endpoint is finite.
    if not np.isfinite(l):
        raise ValueError("This implementation requires a finite lower endpoint l.")

    # The exponential-perturbation hazard (degenerate case only) must be positive.
    if not (epsilon > 0):
        raise ValueError("epsilon must be positive.")

    # The likelihood knots must be strictly ordered and strictly inside (l, inf).
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")
    if not (l < x[0]):
        raise ValueError("Observed knots must lie strictly inside (l, inf).")

    # The binomial model requires positive integer sample sizes.
    if not np.all(n > 0) or not np.all(n == np.round(n)):
        raise ValueError("n must contain positive integers.")

    # The observed counts must be valid binomial success counts.
    if not np.all(y == np.round(y)) or not np.all((0 <= y) & (y <= n)):
        raise ValueError("y must contain integers satisfying 0 <= y_i <= n_i.")

    # Unlike IFR, DFR convexity bounds the last knot value from below, so a
    # trailing all-success knot (y_k = n_k) is normally attained by the ordinary
    # program with F(x_k) < 1. The one configuration with no attained optimum is
    # when every observed knot is all-success: the likelihood supremum (zero) is
    # approached only as all t_i -> -inf. Reject that explicitly; any remaining
    # unbounded configuration is caught by the solver-status check below.
    if np.all(y >= n):
        raise ValueError(
            "fit_cdf_dfr: every observed knot is all-success (y_i = n_i); the DFR "
            "likelihood supremum is not attained, since a DFR cdf cannot reach one "
            "at a finite point. Provide at least one knot with y_i < n_i."
        )

    # Add the artificial lower knot x_0 = l, representing F(l) = 0, i.e. t(l) = 0.
    x_opt = np.concatenate(([l], x))

    # t[i] represents log survival at x_opt[i].
    t = cp.Variable(len(x_opt))

    objective_terms = []

    # The finite log-likelihood is
    #
    #   sum_i y_i log(1-exp(t_i)) + (n_i-y_i)t_i,
    #
    # identical to the IFR formulation. Zero-coefficient terms are omitted so that
    # a boundary count uses the convention 0*log(0)=0 rather than restricting the
    # domain.
    for i in range(len(x)):
        knot_index = i + 1
        if y[i] > 0:
            objective_terms.append(y[i] * cp.log1p(-cp.exp(t[knot_index])))
        if n[i] > y[i]:
            objective_terms.append((n[i] - y[i]) * t[knot_index])

    objective = cp.sum(objective_terms)

    # The cdf range restriction F(x) in [0,1] becomes t(x) <= 0, and the lower
    # endpoint restriction is t(l) = 0.
    constraints = [t <= 0, t[0] == 0]

    # Since F is nondecreasing, t = log(1-F) must be nonincreasing.
    for i in range(len(x_opt) - 1):
        constraints.append(t[i] >= t[i + 1])

    # DFR is equivalent to convexity of t. On an irregular knot grid, convexity
    # means that consecutive secant slopes are nondecreasing. This is the
    # cross-multiplied form of
    #
    #   (t_i-t_{i-1})/(x_i-x_{i-1}) <= (t_{i+1}-t_i)/(x_{i+1}-x_i),
    #
    # i.e. the reversed inequality relative to the IFR concavity constraint.
    for i in range(1, len(x_opt) - 1):
        left_width = x_opt[i] - x_opt[i - 1]
        right_width = x_opt[i + 1] - x_opt[i]
        constraints.append(
            (t[i] - t[i - 1]) * right_width
            <= (t[i + 1] - t[i]) * left_width
        )

    # Concave maximization over a polyhedron: a convex optimization problem.
    problem = cp.Problem(cp.Maximize(objective), constraints)

    solve_options = dict(kwargs.get("solver_options", {}))
    if kwargs.get("solver") is not None:
        solve_options["solver"] = kwargs["solver"]

    problem.solve(**solve_options)

    # Optimal-inaccurate is accepted because exponential-cone solvers can report it
    # for numerically difficult but usable solutions.
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            f"Finite optimization failed with status: {problem.status}."
        )

    # Project onto the constraints to remove solver roundoff (see _clean_knot_values).
    t_fit = _clean_knot_values(t.value)

    x_sym = sp.symbols("x")

    # The final piecewise-linear slope is the largest (least negative) slope of the
    # convex, non-increasing fit. Continuing it as a straight line preserves
    # convexity and sends t -> -inf. A non-negative slope means tau_{k-1} = tau_k
    # (a flat final segment), for which no such tail exists. The manuscript's DFR
    # discussion resolves this by perturbing the fit toward an exponential: tilt the
    # fitted values by tau_i -> tau_i - epsilon*(x_i - l) (hazard epsilon from l).
    # This keeps t convex (subtracting an affine term), lowers every segment slope
    # by epsilon, makes the final slope -epsilon < 0, and leaves t(l) = 0 intact
    # (the x_0 = l term is unchanged), at a log-likelihood cost that vanishes as
    # epsilon -> 0.
    # Flatness is judged against the perturbation hazard epsilon, not machine
    # precision: a genuinely flat segment comes back from the solver with a slope
    # of order 1e-5 (observed), which would pass a tighter test and yield a tail
    # that reaches F = 1 only astronomically far out. A final slope in
    # [-epsilon, 0] is "flat up to the perturbation's own scale", and tilting it
    # costs at most the epsilon hazard the manuscript already accepts.
    final_slope = (t_fit[-1] - t_fit[-2]) / (x_opt[-1] - x_opt[-2])
    if final_slope >= -epsilon:
        t_fit = t_fit - epsilon * (x_opt - l)
        final_slope = (t_fit[-1] - t_fit[-2]) / (x_opt[-1] - x_opt[-2])

    # The join point is the final observed knot x_k.
    join_x = x_opt[-1]
    join_t = t_fit[-1]

    # Piecewise-linear log-survival estimator on [l, x_k].
    observed_t = linear_interpolation(x_opt, t_fit)

    # Linear convex tail on [x_k, inf): t(x) = join_t + final_slope*(x - x_k),
    # equivalently F(x) = 1 - exp(join_t) * exp(final_slope*(x - x_k)) -> 1.
    tail_cdf = 1 - sp.exp(join_t + final_slope * (x_sym - join_x))

    # Assemble the full cdf on [l, inf). No finite upper endpoint is imposed; the
    # tail branch carries F(x) -> 1 as x -> inf.
    cdf = sp.Piecewise(
        (0, x_sym < l),
        (1 - sp.exp(observed_t), (x_sym >= l) & (x_sym <= join_x)),
        (tail_cdf, x_sym > join_x),
    )

    # Flatten the nested Piecewise expression for stable use with lambdify.
    return sp.piecewise_fold(cdf)


def fit_cdf_ifra(x, n, y, l=0.0, **kwargs):
    """Fit an increasing-failure-rate-average (IFRA) cdf from noisy quantile data.

    IFRA analogue of :func:`fit_cdf_ifr` (manuscript Section 5.2). A distribution is
    IFRA if its average failure rate ``H(x)/x`` is non-decreasing, where
    ``H(x) = -t(x)`` is the cumulative hazard and ``t(x) = log(1 - F(x))``. Since
    ``t(0) = 0``, this is equivalent to ``-t(x)/x`` being non-decreasing for
    ``x > 0``. IFRA is strictly weaker than IFR: the instantaneous hazard may dip
    locally as long as the running average from the origin does not decrease. The
    support is ``[0, inf)``, so the lower endpoint must be ``l = 0``.

    The finite program is Algorithm 1's convex program with the IFR concavity
    constraints replaced by the linear IFRA constraints
    ``-tau_i / x_i <= -tau_{i+1} / x_{i+1}`` for ``i = 1, ..., k-1`` (the origin
    knot ``(0, 0)`` is excluded from these ratios). The binomial log-likelihood,
    ``t <= 0``, ``t(0) = 0``, and the monotonicity constraints are unchanged, and
    the problem stays convex.

    Reconstruction depends on the terminal observation:

    * Ordinary case ``y_k < n_k``. Piecewise-linear interpolation in t-space
      through the fitted knots, then a *linear* tail ``t(x) = tau_k - a (x - x_k)``
      with ``a = max(1/x_k, -tau_k / x_k)``. The condition ``a >= -tau_k / x_k``
      keeps ``-t(x)/x`` non-decreasing beyond the last knot, the floor ``1/x_k``
      (a hazard of one per unit of the observed range, hence scale-free) makes
      ``a > 0`` when ``tau_k = 0``, and ``t(x) -> -inf`` as ``x -> inf``, giving a
      proper IFRA cdf on ``[0, inf)``. Piecewise-linear interpolation preserves
      IFRA between knots.
    * Terminal all-success case ``y_k = n_k``. The finite program's supremum is
      unattained (``tau_k -> -inf``), so, as in :func:`fit_cdf_ifr`, the fit uses the
      last non-all-success knot ``x_j`` (``j = 0`` if none) and appends the tail
      ``t(x) = tau_j + a log((x_{j+1} - x)/(x_{j+1} - x_j))`` on ``[x_j, x_{j+1})``
      with ``a = max(1, -tau_j (x_{j+1} - x_j)/x_j)`` (this ``a`` is dimensionless,
      so the floor is 1), reaching ``F = 1`` at the first all-success knot
      ``x_{j+1}``. The bound on ``a`` is exactly the threshold that
      keeps ``-t(x)/x`` non-decreasing across the join and through the tail, so the
      reconstruction is a valid (finite-support) IFRA cdf.

    Args:
        x (np.ndarray): Strictly increasing observed knots, each ``> 0``.
        n (np.ndarray): Positive integer sample sizes at each knot.
        y (np.ndarray): Integer success counts with ``0 <= y_i <= n_i``.
        l (float): Lower support endpoint; must be ``0`` for IFRA (the average
            failure rate is measured from the origin). Defaults to ``0.0``.
        **kwargs: Optional ``solver`` and ``solver_options``, handled as in
            :func:`fit_cdf_ifr`.

    Returns:
        sympy.Piecewise: The fitted IFRA cdf as a symbolic expression, on
            ``[0, inf)`` in the ordinary case or ``[0, x_{j+1}]`` in the terminal
            all-success case.

    Raises:
        ValueError: If the inputs are malformed or ``l != 0``.
        RuntimeError: If the convex program fails to solve.
    """
    # Standardize the inputs before checking dimensions and numerical conditions.
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)

    # Each observed knot must have one sample size and one success count.
    if x.ndim != 1 or n.ndim != 1 or y.ndim != 1:
        raise ValueError("x, n, and y must be one-dimensional arrays.")
    if len(x) == 0 or len(x) != len(n) or len(x) != len(y):
        raise ValueError("x, n, and y must be nonempty arrays of equal length.")

    # IFRA measures the average failure rate from the origin, so the support must
    # be [0, inf). The origin knot (0, 0) is added internally.
    if l != 0.0:
        raise ValueError("fit_cdf_ifra requires l = 0 (support [0, inf)).")

    # The likelihood knots must be strictly ordered and strictly positive.
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")
    if not (x[0] > 0.0):
        raise ValueError("Observed knots must lie strictly inside (0, inf).")

    # The binomial model requires positive integer sample sizes.
    if not np.all(n > 0) or not np.all(n == np.round(n)):
        raise ValueError("n must contain positive integers.")

    # The observed counts must be valid binomial success counts.
    if not np.all(y == np.round(y)) or not np.all((0 <= y) & (y <= n)):
        raise ValueError("y must contain integers satisfying 0 <= y_i <= n_i.")

    x_sym = sp.symbols("x")

    def solve_knot_problem(x_fit, n_fit, y_fit):
        """Solve the finite IFRA convex program on a prefix of the knots.

        The origin knot (0, 0) is prepended, representing F(0)=0, i.e. t(0)=0.
        """
        # Add the origin knot x_0 = 0. No data are observed there.
        x_opt = np.concatenate(([0.0], x_fit))

        # t[i] represents log survival at x_opt[i].
        t = cp.Variable(len(x_opt))

        # Binomial log-likelihood at the observed knots, identical to IFR; zero-
        # coefficient terms are omitted (0*log(0)=0 convention).
        objective_terms = []
        for i in range(len(x_fit)):
            knot_index = i + 1
            if y_fit[i] > 0:
                objective_terms.append(y_fit[i] * cp.log1p(-cp.exp(t[knot_index])))
            if n_fit[i] > y_fit[i]:
                objective_terms.append((n_fit[i] - y_fit[i]) * t[knot_index])
        objective = cp.sum(objective_terms)

        # Range restriction t <= 0 and the origin restriction t(0) = 0.
        constraints = [t <= 0, t[0] == 0]

        # F nondecreasing  <=>  t = log(1-F) nonincreasing.
        for i in range(len(x_opt) - 1):
            constraints.append(t[i] >= t[i + 1])

        # IFRA: the average failure rate -t(x)/x is nondecreasing across adjacent
        # observed knots. Cross-multiplying by the positive knots x_i, x_{i+1}:
        #
        #   -tau_i / x_i <= -tau_{i+1} / x_{i+1}
        #       <=>  -t_i * x_{i+1} <= -t_{i+1} * x_i.
        #
        # The origin knot (index 0) is excluded, since -t(0)/0 is undefined; the
        # monotonicity constraint handles the [0, x_1] segment.
        for i in range(1, len(x_opt) - 1):
            constraints.append(-t[i] * x_opt[i + 1] <= -t[i + 1] * x_opt[i])

        # Concave maximization over a polyhedron: a convex optimization problem.
        problem = cp.Problem(cp.Maximize(objective), constraints)

        solve_options = dict(kwargs.get("solver_options", {}))
        if kwargs.get("solver") is not None:
            solve_options["solver"] = kwargs["solver"]

        problem.solve(**solve_options)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"Finite optimization failed with status: {problem.status}."
            )

        return x_opt, _clean_knot_values(t.value)

    # -------------------------------------------------------------------------
    # Ordinary branch: y_k < n_k. Fit all knots and append a linear t-space tail
    # to infinity, yielding a proper IFRA cdf on [0, inf).
    # -------------------------------------------------------------------------
    if y[-1] < n[-1]:
        x_fit, t_fit = solve_knot_problem(x, n, y)

        join_x = x_fit[-1]
        join_t = t_fit[-1]

        # a >= -tau_k / x_k keeps -t(x)/x nondecreasing on the tail. The floor
        # 1/x_k guarantees a > 0 (hence t -> -inf) even when tau_k = 0; it is a
        # hazard of one per unit of the observed range, so the rule is scale-free
        # (a has units of 1/x, so a floor of 1 would not be). This mirrors the
        # NBU rule s = min(s_cap, -1/x_k); dimensionless tail exponents (IFR,
        # the IFRA terminal tail, IGFR) use the floor 1 instead.
        a = max(1.0 / join_x, -join_t / join_x)

        observed_t = linear_interpolation(x_fit, t_fit)

        # Linear tail t(x) = tau_k - a (x - x_k); F -> 1 as x -> inf.
        tail_cdf = 1 - sp.exp(join_t - a * (x_sym - join_x))

        cdf = sp.Piecewise(
            (0, x_sym < 0),
            (1 - sp.exp(observed_t), (x_sym >= 0) & (x_sym <= join_x)),
            (tail_cdf, x_sym > join_x),
        )
        return sp.piecewise_fold(cdf)

    # -------------------------------------------------------------------------
    # Terminal all-success branch: y_k = n_k. The supremum is unattained, so fit
    # through the last non-all-success knot x_j and drive F to one at the first
    # all-success knot x_{j+1} via an IFRA-preserving log tail.
    # -------------------------------------------------------------------------
    eligible_indices = np.flatnonzero(y[:-1] < n[:-1])

    if len(eligible_indices) == 0:
        # Every observed knot is all-success. The IFRA MLE sets F = 1 at the first
        # knot x_1 (attaining log-likelihood 0); fit nothing and tail from origin.
        x_fit = np.array([0.0], dtype=float)
        t_fit = np.array([0.0], dtype=float)
        join_x = 0.0
        join_t = 0.0
        tail_end = x[0]
        a = 1.0
    else:
        j = eligible_indices[-1]
        x_fit, t_fit = solve_knot_problem(x[:j + 1], n[:j + 1], y[:j + 1])
        join_x = x_fit[-1]
        join_t = t_fit[-1]
        tail_end = x[j + 1]

        # a >= -tau_j (x_{j+1} - x_j) / x_j is exactly the IFRA threshold that keeps
        # -t(x)/x nondecreasing across the join and through the log tail.
        a = max(1.0, -join_t * (tail_end - join_x) / join_x)

    observed_t = linear_interpolation(x_fit, t_fit)

    # Log tail on [x_j, x_{j+1}): t(x) = tau_j + a log((x_{j+1}-x)/(x_{j+1}-x_j)),
    # equivalently F(x) = 1 - exp(tau_j) * ((x_{j+1}-x)/(x_{j+1}-x_j))**a. The Max
    # guards the fractional power against a negative base when lambdify eagerly
    # evaluates this branch outside its interval.
    safe_tail_gap = sp.Max(tail_end - x_sym, 0)
    tail_cdf = 1 - sp.exp(join_t) * (safe_tail_gap / (tail_end - join_x)) ** a

    cdf = sp.Piecewise(
        (0, x_sym < 0),
        (1 - sp.exp(observed_t), (x_sym >= 0) & (x_sym <= join_x)),
        (tail_cdf, (x_sym > join_x) & (x_sym < tail_end)),
        (1, x_sym >= tail_end),
    )
    return sp.piecewise_fold(cdf)


def fit_cdf_igfr(x, n, y, l=0.0, epsilon=1e-3, **kwargs):
    """Fit an increasing-generalized-failure-rate (IGFR) cdf from noisy quantile data.

    IGFR analogue of :func:`fit_cdf_ifr` (manuscript Section 5.3). The generalized
    failure rate is ``g(x) = x h(x)``; a distribution is IGFR if ``g`` is
    non-decreasing. Writing ``t(x) = log(1 - F(x))``, ``v = log x``, and
    ``phi(v) = t(e^v)``, one has ``phi'(v) = -g(x)``, so IGFR is exactly concavity
    of ``phi`` -- i.e. IGFR is the IFR problem in the log-coordinate ``v = log x``.
    This estimator therefore runs Algorithm 1 in ``v``-space and maps back via
    ``x = e^v``. The support is ``[0, inf)`` (so ``l = 0``), which is unbounded at
    both ends in ``v``; the knots must be strictly positive.

    The finite program maximizes the same binomial log-likelihood over the knot
    values ``tau_1, ..., tau_k`` (no artificial lower knot, since ``v -> -inf`` as
    ``x -> 0``), subject to ``tau_i <= 0``, monotonicity, and the IGFR concavity
    constraints in ``v``,
    ``(tau_i - tau_{i-1})/(ln x_i - ln x_{i-1}) >= (tau_{i+1} - tau_i)/(ln x_{i+1} - ln x_i)``
    for ``i = 2, ..., k-1``. It is convex.

    Reconstruction (all pieces are linear/concave in ``v``, hence IGFR):

    * Interior ``[x_1, x_k]``: linear interpolation of ``(ln x_i, tau_i)``.
    * Lower tail ``[0, x_1]``: if ``tau_1 = 0``, set ``t = 0`` (``F = 0``);
      otherwise a power law ``t(x) = tau_1 (x/x_1)**a`` (exponential in ``v``,
      ``-> 0`` as ``x -> 0``). With at least two fitted knots and
      ``tau_2 < tau_1``, ``a = (tau_2 - tau_1)/(tau_1 (ln x_2 - ln x_1)) > 0``
      matches the ``v``-slopes at ``x_1`` (a ``C^1`` concave join); with a single
      fitted knot, ``a = 1``.
    * Let ``s_i`` be the left slope of ``phi`` at ``v = ln x_i`` (the
      lower-endpoint segment's slope when ``i = 1``).
    * Ordinary upper tail ``y_k < n_k``: log-linear ``t(x) = tau_k + b ln(x/x_k)``
      (linear in ``v``) with ``b = min(-1, s_k) < 0``. Since ``b <= s_k`` the join
      stays concave, and ``b < 0`` sends ``t -> -inf`` as ``x -> inf`` -- a proper
      IGFR cdf on ``[0, inf)`` -- even when the final fitted slope ``s_k`` is zero.
    * Terminal all-success ``y_k = n_k``: as in :func:`fit_cdf_ifr`, but in ``v``-space
      -- fit through the last non-all-success knot ``x_j`` (possibly ``j = 1``, a
      single fitted knot) and append the tail
      ``t(x) = tau_j + c log((ln x_{j+1} - ln x)/(ln x_{j+1} - ln x_j))`` with
      ``c = max(1, -s_j (ln x_{j+1} - ln x_j))``, which keeps ``phi`` concave at
      the join and drives ``F = 1`` at the first all-success knot ``x_{j+1}``.

    Degenerate lower endpoint. The power-law lower tail needs ``tau_2 < tau_1``
    when ``tau_1 < 0``; if instead ``tau_2 = tau_1 < 0`` the ``v``-slope at ``x_1``
    is zero and no continuous IGFR extension to the origin preserves both fitted
    values. The manuscript's IGFR footnote resolves this by perturbing the fit
    toward an exponential distribution, ``tau_i -> tau_i - epsilon * x_i``
    (hazard ``epsilon``): subtracting ``epsilon e^v`` keeps ``phi`` concave in
    ``v`` while making ``tau_2 < tau_1 < 0``. The log-likelihood loss vanishes as
    ``epsilon -> 0``. A flat *final* segment needs no perturbation, since the
    upper-tail rule ``b = min(-1, s_k)`` already yields a decreasing tail.

    Args:
        x (np.ndarray): Strictly increasing observed knots, each ``> 0``.
        n (np.ndarray): Positive integer sample sizes at each knot.
        y (np.ndarray): Integer success counts with ``0 <= y_i <= n_i``.
        l (float): Lower support endpoint; must be ``0`` for IGFR. Defaults to
            ``0.0``.
        epsilon (float): Exponential-perturbation hazard used only when the lower
            endpoint is degenerate. It also sets the flatness threshold: the
            endpoint counts as degenerate when ``tau_1 < 0`` and the first
            log-space slope ``(tau_2 - tau_1)/(ln x_2 - ln x_1)`` is at least
            ``-epsilon x_1`` (the slope the perturbation itself contributes at
            ``x_1``), since solver output for a truly flat segment is only
            accurate to about ``1e-5``. Must be positive. Defaults to ``1e-3``.
        **kwargs: Optional ``solver`` and ``solver_options``, handled as in
            :func:`fit_cdf_ifr`.

    Returns:
        sympy.Piecewise: The fitted IGFR cdf as a symbolic expression, on
            ``[0, inf)`` in the ordinary case or ``[0, x_{j+1}]`` in the terminal
            all-success case.

    Raises:
        ValueError: If the inputs are malformed, ``l != 0``, ``epsilon <= 0``,
            every observed knot is all-success (no fitted knot to anchor the
            reconstruction), or the lower endpoint is still degenerate after the
            perturbation (a defensive guard reachable only with an extremely small
            ``epsilon``).
        RuntimeError: If the convex program fails to solve.
    """
    # Standardize the inputs before checking dimensions and numerical conditions.
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)

    # Each observed knot must have one sample size and one success count.
    if x.ndim != 1 or n.ndim != 1 or y.ndim != 1:
        raise ValueError("x, n, and y must be one-dimensional arrays.")
    if len(x) == 0 or len(x) != len(n) or len(x) != len(y):
        raise ValueError("x, n, and y must be nonempty arrays of equal length.")

    # IGFR is defined through the log-axis v = log x, so the support is [0, inf)
    # and all knots must be strictly positive.
    if l != 0.0:
        raise ValueError("fit_cdf_igfr requires l = 0 (support [0, inf)).")
    if not (epsilon > 0):
        raise ValueError("epsilon must be positive.")

    # The likelihood knots must be strictly ordered and strictly positive.
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")
    if not (x[0] > 0.0):
        raise ValueError("Observed knots must lie strictly inside (0, inf).")

    # The binomial model requires positive integer sample sizes.
    if not np.all(n > 0) or not np.all(n == np.round(n)):
        raise ValueError("n must contain positive integers.")

    # The observed counts must be valid binomial success counts.
    if not np.all(y == np.round(y)) or not np.all((0 <= y) & (y <= n)):
        raise ValueError("y must contain integers satisfying 0 <= y_i <= n_i.")

    x_sym = sp.symbols("x")
    # Tolerance for the "tau_1 = 0" test and the defensive exponent guard below.
    # It must exceed the solver's accuracy at a bound (~1e-8); a tighter value lets
    # roundoff at an all-failure knot (tau_1 = +7e-9 instead of 0) send the fit
    # down the power-law branch. The flatness test itself is scaled by epsilon
    # (see perturb_if_degenerate), since a truly flat segment comes back from the
    # solver with a slope of order 1e-5.
    tol = _SLOPE_TOL

    def guarded_log(expr):
        # Keep the argument strictly positive so lambdify never evaluates log(<=0)
        # in a branch that is not selected (sympy Piecewise evaluates all branches).
        return sp.log(sp.Max(expr, sp.Float(1e-12)))

    def solve_knot_problem(x_fit, n_fit, y_fit):
        """Solve the finite IGFR convex program (IFR in v = log x) on the knots."""
        v = np.log(x_fit)
        m = len(x_fit)

        # t[i] = log survival at x_fit[i]; no artificial lower knot (v -> -inf).
        t = cp.Variable(m)

        objective_terms = []
        for i in range(m):
            if y_fit[i] > 0:
                objective_terms.append(y_fit[i] * cp.log1p(-cp.exp(t[i])))
            if n_fit[i] > y_fit[i]:
                objective_terms.append((n_fit[i] - y_fit[i]) * t[i])
        objective = cp.sum(objective_terms)

        # Range t <= 0 and monotonicity (F nondecreasing => t nonincreasing).
        constraints = [t <= 0]
        for i in range(m - 1):
            constraints.append(t[i] >= t[i + 1])

        # IGFR: phi concave in v, i.e. consecutive secant slopes in v nonincreasing.
        # Cross-multiplied by the positive log-gaps:
        #   (t_i - t_{i-1})(v_{i+1} - v_i) >= (t_{i+1} - t_i)(v_i - v_{i-1}).
        for i in range(1, m - 1):
            constraints.append(
                (t[i] - t[i - 1]) * (v[i + 1] - v[i])
                >= (t[i + 1] - t[i]) * (v[i] - v[i - 1])
            )

        problem = cp.Problem(cp.Maximize(objective), constraints)

        solve_options = dict(kwargs.get("solver_options", {}))
        if kwargs.get("solver") is not None:
            solve_options["solver"] = kwargs["solver"]
        problem.solve(**solve_options)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"Finite optimization failed with status: {problem.status}."
            )
        return _clean_knot_values(t.value)

    def lower_and_interior_pieces(x_fit, t_fit):
        """Build the (0 <= x < x_last) cdf pieces: lower tail on [0, x_1) plus the
        interior log-linear interpolation. Returns (pieces, x_last, t_last, s_last),
        where s_last is the manuscript's left slope of phi at the last fitted knot:
        the final interior secant slope in v when two or more knots are fitted, and
        the lower-endpoint segment's slope when only x_1 is fitted."""
        v = np.log(x_fit)
        m = len(x_fit)
        x1, tau1 = x_fit[0], t_fit[0]
        pieces = [(0, x_sym < 0)]

        # Lower tail on [0, x_1).
        if abs(tau1) < tol:
            # tau_1 = 0  =>  F = 0 on [0, x_1]; phi is flat there, so s_1 = 0.
            pieces.append((0, (x_sym >= 0) & (x_sym < x1)))
            lower_slope = 0.0
        else:
            # Power law t(x) = tau_1 (x/x_1)**a_low (= tau_1 e^{a_low (v - v_1)}),
            # whose left slope in v at x_1 is a_low * tau_1.
            if m >= 2:
                # Tight value matching the v-slopes at x_1 (smooth concave join):
                # a_low = (tau_2 - tau_1) / (tau_1 (v_2 - v_1)); needs tau_2 < tau_1.
                a_low = (t_fit[1] - t_fit[0]) / (t_fit[0] * (v[1] - v[0]))
                if a_low <= tol:
                    raise ValueError(
                        "Degenerate IGFR lower tail: the first two fitted values "
                        "coincide (tau_1 = tau_2 < 0), so no power-law tail to the "
                        "origin preserves IGFR. See the IGFR degenerate-case footnote."
                    )
            else:
                # Single fitted knot: the manuscript takes a = 1.
                a_low = 1.0
            base = sp.Max(x_sym, 0) / x1
            pieces.append((1 - sp.exp(tau1 * base ** a_low), (x_sym >= 0) & (x_sym < x1)))
            lower_slope = a_low * tau1

        # Interior segments [x_i, x_{i+1}) : t(x) = tau_i + s_i (log x - v_i).
        for i in range(m - 1):
            s_i = (t_fit[i + 1] - t_fit[i]) / (v[i + 1] - v[i])
            expr = t_fit[i] + s_i * (guarded_log(x_sym) - v[i])
            pieces.append((1 - sp.exp(expr), (x_sym >= x_fit[i]) & (x_sym < x_fit[i + 1])))

        # Left slope of phi at the last fitted knot (the lower-endpoint segment's
        # slope when only x_1 is fitted).
        if m >= 2:
            s_last = (t_fit[-1] - t_fit[-2]) / (v[-1] - v[-2])
        else:
            s_last = lower_slope
        return pieces, x_fit[-1], t_fit[-1], s_last

    def perturb_if_degenerate(x_fit, t_fit):
        """Apply the exponential perturbation tau_i -> tau_i - epsilon*x_i when the
        lower endpoint is degenerate (tau_1 < 0 with tau_2 = tau_1, i.e. a zero
        v-slope at x_1, so no power-law tail to the origin exists). Subtracting
        epsilon*e^v keeps phi concave in v while making tau_2 < tau_1 < 0. A single
        fitted knot uses a = 1 and never needs the perturbation; a flat final
        segment is handled by the upper-tail rule b = min(-1, s_k) instead."""
        if len(x_fit) < 2 or abs(t_fit[0]) <= tol:
            return t_fit
        v = np.log(x_fit)
        # First log-space slope. The perturbation -epsilon*e^v has slope
        # -epsilon*x_1 at v_1, so a slope no steeper than that is flat up to the
        # perturbation's own scale; after tilting, the slope is strictly below
        # -epsilon*x_1 and the power-law exponent is strictly positive.
        s_1 = (t_fit[1] - t_fit[0]) / (v[1] - v[0])
        if s_1 >= -epsilon * x_fit[0]:
            t_fit = t_fit - epsilon * x_fit
        return t_fit

    # -------------------------------------------------------------------------
    # Ordinary branch: y_k < n_k. Log-linear upper tail to infinity in x.
    # -------------------------------------------------------------------------
    if y[-1] < n[-1]:
        t_fit = solve_knot_problem(x, n, y)
        t_fit = perturb_if_degenerate(x, t_fit)
        pieces, x_last, t_last, s_last = lower_and_interior_pieces(x, t_fit)

        # Upper tail t(x) = tau_k + b log(x/x_k) (linear in v) with the manuscript's
        # b = min{-1, s_k} < 0: b <= s_k keeps phi concave across the join, and
        # b < 0 sends t -> -inf as x -> inf even when the final fitted slope is 0.
        b = min(-1.0, s_last)
        base = sp.Max(x_sym, sp.Float(1e-12)) / x_last
        upper_cdf = 1 - sp.exp(t_last) * base ** b
        pieces.append((upper_cdf, x_sym >= x_last))

        return sp.piecewise_fold(sp.Piecewise(*pieces))

    # -------------------------------------------------------------------------
    # Terminal all-success branch: y_k = n_k. Fit through the last non-all-success
    # knot x_j (a single fitted knot when j = 1) and drive F to one at the first
    # all-success knot x_{j+1} with an IFR-in-v log tail.
    # -------------------------------------------------------------------------
    eligible_indices = np.flatnonzero(y[:-1] < n[:-1])
    if len(eligible_indices) == 0:
        raise ValueError(
            "fit_cdf_igfr: every observed knot is all-success; there is no fitted "
            "knot to anchor the log-axis reconstruction."
        )
    j = eligible_indices[-1]

    t_fit = solve_knot_problem(x[:j + 1], n[:j + 1], y[:j + 1])
    # Only the lower (origin) join can be degenerate here; the upper end uses the
    # finite-support log tail below, whose c = max(1, ...) never degenerates.
    t_fit = perturb_if_degenerate(x[:j + 1], t_fit)
    pieces, x_last, t_last, s_last = lower_and_interior_pieces(x[:j + 1], t_fit)

    # Finite-support (in v) tail reaching F = 1 at x_{j+1}. In v-space this is the
    # IFR terminal log tail; c is the threshold that preserves concavity at the
    # join: c >= -s_j (v_{j+1} - v_j), with s_j the left slope of phi at x_j.
    tail_end = x[j + 1]
    v_last = np.log(x_last)
    v_end = np.log(tail_end)
    c_term = max(1.0, -(v_end - v_last) * s_last)

    gap = sp.Max(v_end - guarded_log(x_sym), 0)
    term_cdf = 1 - sp.exp(t_last) * (gap / (v_end - v_last)) ** c_term
    pieces.append((term_cdf, (x_sym >= x_last) & (x_sym < tail_end)))
    pieces.append((1, x_sym >= tail_end))

    return sp.piecewise_fold(sp.Piecewise(*pieces))


def _infer_lattice_delta(x):
    """Largest spacing Delta such that every knot is an integer multiple of it.

    This is the gcd of the knots viewed as rationals: gcd of the numerators over
    the lcm of the denominators of their reduced fractions. It is well defined
    whenever the knots are commensurable (e.g. integers or simple rationals), per
    the common-lattice assumption in the manuscript's NBU construction.
    """
    fracs = [Fraction(float(xi)).limit_denominator(10 ** 6) for xi in x]
    num_gcd = reduce(_math_gcd, [f.numerator for f in fracs])
    den_lcm = reduce(lambda a, b: a * b // _math_gcd(a, b), [f.denominator for f in fracs])
    return float(Fraction(num_gcd, den_lcm))


def fit_cdf_nbu(x, n, y, l=0.0, delta=None, **kwargs):
    """Fit a new-better-than-used (NBU) cdf from noisy quantile data.

    NBU analogue of :func:`fit_cdf_ifr` (manuscript Section 5.4). A distribution is NBU
    if ``1 - F(x' + x'') <= (1 - F(x'))(1 - F(x''))`` for all ``x', x'' >= 0``,
    which in terms of ``t(x) = log(1 - F(x))`` is subadditivity:
    ``t(x' + x'') <= t(x') + t(x'')`` (equivalently the cumulative hazard
    ``H = -t`` is superadditive). NBU is weaker than both IFR and IFRA. The support
    is ``[0, inf)`` (so ``l = 0``).

    Subadditivity is not a one- or two-point shape condition, so the estimator is
    built on a common lattice. Let ``Delta`` be a spacing such that every observed
    knot is an integer multiple of it, and let the lattice be
    ``z_j = j*Delta`` for ``j = 0, ..., m`` with ``z_m = x_k``. The decision
    variables are the transformed values ``tau_j = t(z_j)`` at *all* lattice points
    (knot points carry a likelihood term; the rest are free fillers). The finite
    program maximizes the binomial log-likelihood subject to ``tau_j <= 0``,
    ``tau_0 = 0``, monotonicity, and the subadditivity constraints
    ``tau_{p+q} <= tau_p + tau_q`` for ``1 <= p <= q``, ``p + q <= m`` (these grow
    quadratically in ``m``). It is convex (linear constraints).

    The reconstruction is piecewise-linear in ``t``-space through the lattice
    values (which preserves subadditivity on an equally spaced lattice), followed
    by a linear tail ``t(x) = tau_m + s (x - x_k)`` for ``x >= x_k`` with
    ``s <= s_cap := min_{p+q>m} (tau_p + tau_q - tau_m)/((p+q-m) Delta)`` and
    ``s < 0``. This slope keeps subadditivity intact when an argument sum crosses
    ``z_m`` (including sums with one or both arguments in the tail) and sends
    ``t -> -inf`` as ``x -> inf``, yielding a proper NBU cdf on ``[0, inf)``. The
    bound is an *upper* limit on ``s``, and its ``p = q = m`` term is
    ``tau_m / x_k <= 0``, so an admissible negative slope always exists; any
    steeper slope remains admissible. The implementation takes the deterministic
    ``s = min(s_cap, -1/x_k)``, the NBU analogue of the IGFR rule
    ``b = min(-1, s_k)``: ``s_cap`` continues the fit as slowly as subadditivity
    allows, and ``-1/x_k`` (a hazard of one per unit of the observed range, so
    scale-free) guarantees decay even when the fit is identically zero, where
    ``s_cap = 0``.

    Terminal all-success case ``y_k = n_k``. The supremum is unattained
    (``tau -> -inf`` at the all-success knots), so, as in :func:`fit_cdf_ifr`, the fit
    is taken through the last non-all-success knot ``x_j`` and ``F`` is driven to
    one at the first all-success knot ``x_{j+1}``. The reconstruction uses the
    ordinary NBU reconstruction on the informative prefix (the manuscript's
    ``t_init``, a subadditive function on ``[0, inf)``) plus a cap ``g`` that is zero on
    ``[0, x_{j+1}-Delta]``, a log dive ``log((x_{j+1}-x)/Delta)`` on the single
    final cell ``[x_{j+1}-Delta, x_{j+1})``, and ``-inf`` beyond. Because the cap
    is confined to one lattice cell, any two points where it is active sum past
    ``x_{j+1}`` (as ``x_{j+1} >= 2 Delta`` whenever an informative knot exists), so
    ``g`` is subadditive; the sum ``t = t_init + g`` is therefore subadditive, is
    continuous, reaches ``F = 1`` at ``x_{j+1}``, and matches the fit at every
    informative knot. The all-knots-all-success case has no informative anchor and
    raises.

    Args:
        x (np.ndarray): Strictly increasing observed knots, each ``> 0``, lying on
            a common lattice anchored at zero.
        n (np.ndarray): Positive integer sample sizes at each knot.
        y (np.ndarray): Integer success counts with ``0 <= y_i <= n_i``.
        l (float): Lower support endpoint; must be ``0`` for NBU. Defaults to
            ``0.0``.
        delta (float, optional): Lattice spacing. If ``None`` (default) it is
            inferred as the gcd of the knots. If given, every knot must be an
            integer multiple of it.
        **kwargs: Optional ``solver`` and ``solver_options``, handled as in
            :func:`fit_cdf_ifr`.

    Returns:
        sympy.Piecewise: The fitted NBU cdf as a symbolic expression, on
            ``[0, inf)`` in the ordinary case or ``[0, x_{j+1}]`` in the terminal
            all-success case.

    Raises:
        ValueError: If the inputs are malformed, ``l != 0``, the knots are not on a
            common lattice of spacing ``delta``, the implied lattice is too fine,
            or every knot is all-success.
        RuntimeError: If the convex program fails to solve.
    """
    # Standardize the inputs before checking dimensions and numerical conditions.
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)

    # Each observed knot must have one sample size and one success count.
    if x.ndim != 1 or n.ndim != 1 or y.ndim != 1:
        raise ValueError("x, n, and y must be one-dimensional arrays.")
    if len(x) == 0 or len(x) != len(n) or len(x) != len(y):
        raise ValueError("x, n, and y must be nonempty arrays of equal length.")

    # NBU is defined through subadditivity from the origin, so the support is
    # [0, inf) and all knots must be strictly positive.
    if l != 0.0:
        raise ValueError("fit_cdf_nbu requires l = 0 (support [0, inf)).")

    # The likelihood knots must be strictly ordered and strictly positive.
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")
    if not (x[0] > 0.0):
        raise ValueError("Observed knots must lie strictly inside (0, inf).")

    # The binomial model requires positive integer sample sizes.
    if not np.all(n > 0) or not np.all(n == np.round(n)):
        raise ValueError("n must contain positive integers.")

    # The observed counts must be valid binomial success counts.
    if not np.all(y == np.round(y)) or not np.all((0 <= y) & (y <= n)):
        raise ValueError("y must contain integers satisfying 0 <= y_i <= n_i.")

    # Lattice spacing: inferred gcd of the knots, or a caller-supplied value.
    if delta is None:
        delta = _infer_lattice_delta(x)
    else:
        delta = float(delta)
        if not (delta > 0):
            raise ValueError("delta must be positive.")

    # Map each knot onto the lattice and confirm it lands on a lattice point.
    ratios = x / delta
    k_idx = np.round(ratios).astype(int)
    if np.any(np.abs(ratios - k_idx) > 1e-6 * np.maximum(1.0, ratios)):
        raise ValueError(
            "Knots are not integer multiples of delta; supply a compatible lattice "
            "spacing via delta (NBU requires the knots on a common lattice)."
        )
    if np.any(k_idx < 1):
        raise ValueError("Each knot must map to a positive lattice index.")

    x_sym = sp.symbols("x")

    def solve_lattice(x_fit, n_fit, y_fit):
        """Solve the ordinary NBU lattice program on knots whose last entry is the
        lattice endpoint. Returns (z_lattice, tau_values)."""
        kk = np.round(x_fit / delta).astype(int)
        mm = int(kk[-1])  # z_mm = x_fit[-1]
        if mm > 5000:
            raise ValueError(
                f"The implied lattice has m = {mm} points (quadratically many "
                "subadditivity constraints); pass a coarser delta."
            )
        zz = delta * np.arange(mm + 1)
        t = cp.Variable(mm + 1)

        # Binomial log-likelihood at the knot lattice points only (fillers carry no
        # data); zero-coefficient terms omitted (0*log(0)=0 convention).
        obj = []
        for i in range(len(x_fit)):
            ji = int(kk[i])
            if y_fit[i] > 0:
                obj.append(y_fit[i] * cp.log1p(-cp.exp(t[ji])))
            if n_fit[i] > y_fit[i]:
                obj.append((n_fit[i] - y_fit[i]) * t[ji])

        # Range t <= 0, origin t(0) = 0, monotonicity, and lattice subadditivity.
        cons = [t <= 0, t[0] == 0]
        for jj in range(mm):
            cons.append(t[jj] >= t[jj + 1])
        for p in range(1, mm + 1):
            for q in range(p, mm - p + 1):
                cons.append(t[p + q] <= t[p] + t[q])

        problem = cp.Problem(cp.Maximize(cp.sum(obj)), cons)
        solve_options = dict(kwargs.get("solver_options", {}))
        if kwargs.get("solver") is not None:
            solve_options["solver"] = kwargs["solver"]
        problem.solve(**solve_options)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"Finite optimization failed with status: {problem.status}."
            )
        return zz, _clean_knot_values(t.value)

    def tail_slope(z_lat, tau_val):
        """Linear tail slope for the lattice z_lat with fitted values tau_val.

        The manuscript admits any s < 0 with
            s <= s_cap := min_{p+q>last} (tau_p+tau_q-tau_last)/((p+q-last)Delta),
        which preserves subadditivity whenever an argument sum crosses the last
        lattice point. s_cap <= tau_last/z_last <= 0 always (take p = q = last), so
        the bound never excludes every negative slope, and steeper slopes stay
        admissible. Return the deterministic s = min(s_cap, -1/z_last): the
        slowest admissible decay, floored at a hazard of one per unit of the
        observed range so that t -> -inf even when the fit is identically zero.
        """
        last = len(tau_val) - 1
        s_cap = np.inf
        for p in range(last + 1):
            for q in range(last + 1):
                if p + q > last:
                    s_cap = min(
                        s_cap,
                        (tau_val[p] + tau_val[q] - tau_val[-1]) / ((p + q - last) * delta),
                    )
        return min(s_cap, -1.0 / z_lat[-1])

    # -------------------------------------------------------------------------
    # Ordinary branch: y_k < n_k. Lattice fit over [0, x_k] + linear tail to inf.
    # -------------------------------------------------------------------------
    if y[-1] < n[-1]:
        z, tau_val = solve_lattice(x, n, y)
        s = tail_slope(z, tau_val)
        x_k = z[-1]
        tau_m = tau_val[-1]
        observed_t = linear_interpolation(z, tau_val)
        tail_cdf = 1 - sp.exp(tau_m + s * (x_sym - x_k))
        cdf = sp.Piecewise(
            (0, x_sym < 0),
            (1 - sp.exp(observed_t), (x_sym >= 0) & (x_sym <= x_k)),
            (tail_cdf, x_sym > x_k),
        )
        return sp.piecewise_fold(cdf)

    # -------------------------------------------------------------------------
    # Terminal all-success branch: y_k = n_k. Fit the informative prefix through
    # the last non-all-success knot x_j, then cap F to one at the first all-success
    # knot x_{j+1}. The cap is confined to the single final lattice cell so that
    # t = t_head + cap stays subadditive (NBU); see the docstring.
    # -------------------------------------------------------------------------
    eligible_indices = np.flatnonzero(y[:-1] < n[:-1])
    if len(eligible_indices) == 0:
        raise ValueError(
            "fit_cdf_nbu: every observed knot is all-success; there is no "
            "informative anchor for a finite-support NBU reconstruction."
        )
    j = int(eligible_indices[-1])

    z, tau_val = solve_lattice(x[:j + 1], n[:j + 1], y[:j + 1])
    s_head = tail_slope(z, tau_val)

    x_j = z[-1]
    tau_j = tau_val[-1]
    z_end = float(x[j + 1])        # first all-success knot = support end
    z_cell = z_end - delta         # start of the final lattice cell

    observed_t = linear_interpolation(z, tau_val)      # PWL on [0, x_j]
    lin = tau_j + s_head * (x_sym - x_j)               # t_head linear tail (beyond x_j)
    gap = sp.Max(z_end - x_sym, 0)
    cap_expr = lin + sp.log(gap / delta)               # t_head + log cap on final cell

    cdf = sp.Piecewise(
        (0, x_sym < 0),
        (1 - sp.exp(observed_t), (x_sym >= 0) & (x_sym <= x_j)),
        (1 - sp.exp(lin), (x_sym > x_j) & (x_sym < z_cell)),
        (1 - sp.exp(cap_expr), (x_sym >= z_cell) & (x_sym < z_end)),
        (1, x_sym >= z_end),
    )
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
    # Imported lazily so the pure-CVXPY estimators in this module (fit_cdf_ifr and
    # the failure-rate extensions) can be imported and run without a Gurobi
    # installation; only this benchmark requires it.
    import gurobipy as gp
    from gurobipy import GRB, nlfunc

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

    # IFR constraint as a width-weighted bilinear (nonconvex quadratic) inequality.
    # Discrete IFR = discrete hazard rate nondecreasing across adjacent intervals:
    #   (F_i - F_{i-1}) / [ (x_i - x_{i-1}) (1 - F_{i-1}) ]
    #       <= (F_{i+1} - F_i) / [ (x_{i+1} - x_i) (1 - F_i) ].
    # Cross-multiplying (widths and survival terms are positive) gives the bilinear
    # form below. The interval widths are required on a nonuniform grid; on a
    # uniform grid they cancel and this reduces to the usual equal-width condition
    # (1 - F_i)^2 >= (1 - F_{i-1})(1 - F_{i+1}).
    if ifr_constraint:
        x_grid_full = np.append(fine_grid, u)
        for i in range(1, n_points):
            w_l = x_grid_full[i] - x_grid_full[i - 1]
            w_r = x_grid_full[i + 1] - x_grid_full[i]
            model.addConstr(
                (F[i] - F[i - 1]) * w_r * (1 - F[i])
                <= (F[i + 1] - F[i]) * w_l * (1 - F[i - 1]),
                name=f"ifr_quad_{i}"
            )


    # Objective: log-likelihood
    # In gurobi, to do nonlinear objective, have to introduce a new variable and make a nonlinear constraint
    z = model.addVar(name="z", lb=-GRB.INFINITY, ub=GRB.INFINITY)

    # Bound z to help the solver. The upper bound is the saturated (unconstrained)
    # binomial log-likelihood, computed with the exact convention 0*log(0)=0 and no
    # clipping: an all-failure knot (y_i=0) or all-success knot (y_i=n_i) contributes
    # exactly 0, so the bound never falls below the true optimum.
    n_arr, y_arr = np.asarray(n, dtype=float), np.asarray(y, dtype=float)
    z_max = 0.0
    for yi, ni in zip(y_arr, n_arr):
        if yi > 0:
            z_max += yi * np.log(yi / ni)
        if ni - yi > 0:
            z_max += (ni - yi) * np.log(1 - yi / ni)
    model.addConstr(z <= z_max)

    # Define z as the binomial log-likelihood at the observed knots. A term is
    # appended only when its count is positive, matching Algorithm 1's treatment of
    # boundary counts (0*log(0)=0) and avoiding a 0*log(0)=NaN term at all-success
    # or all-failure knots, where F_i is driven to 1 or 0.
    ll_terms = []
    for i in range(len(x)):
        knot_idx = (i + 1) * (d + 1)
        if y[i] > 0:
            ll_terms.append(y[i] * nlfunc.log(F[knot_idx]))
        if n[i] - y[i] > 0:
            ll_terms.append((n[i] - y[i]) * nlfunc.log(1 - F[knot_idx]))
    model.addConstr(z == gp.quicksum(ll_terms))

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
        # Right-continuous step cdf, matching eval_step_cdf: value estimated_cdf[j]
        # on [x_grid[j], x_grid[j+1]), with F = 1 for x >= u.
        x_sym = sp.symbols('x')
        x_grid_full = np.append(fine_grid, u)
        pieces = [(0, x_sym < x_grid_full[0])]
        for j in range(len(x_grid_full) - 1):
            pieces.append((estimated_cdf[j],
                           (x_sym >= x_grid_full[j]) & (x_sym < x_grid_full[j + 1])))
        pieces.append((1, x_sym >= x_grid_full[-1]))
        return sp.Piecewise(*pieces)
    else:
        raise ValueError(f"Invalid return type: {return_type}. Choose 'sympy' or 'values'.")


def fit_weibull_cdf(x, n, y, l, u, x_sym):
    """Fit a truncated Weibull CDF to noisy quantile data.

    Estimates Weibull shape and scale parameters by log-log linearization of the
    empirical quantiles followed by ordinary least-squares regression on log(x).
    Boundary proportions (y_i = 0 or y_i = n_i) are replaced by a half-count,
    i.e. y_i/n_i is clipped to [1/(2 n_i), 1 - 1/(2 n_i)], before the transform.
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

    Raises:
        ValueError: If the log-log regression is degenerate -- a non-positive
            shape estimate (slope <= 0) or a non-finite scale estimate. This
            occurs for the 3-knot probability-plot fit when the noisy empirical
            quantiles are (near-)collinear with zero or negative trend in
            log-log space. The caller should treat such a fit as a failed run.
    """

    # Probability-plot transform of the empirical proportions. An all-failure or
    # all-success knot (y_i = 0 or y_i = n_i) has no finite transform, so it is
    # replaced by a half-count: y_i/n_i is clipped to [1/(2 n_i), 1 - 1/(2 n_i)].
    # This is the usual boundary correction for a probability plot and keeps the
    # transformed value on the scale of the data (a clip to 1 - 1e-8 maps to 2.9
    # in log(-log(1 - F)) space, far above any attainable interior value, which
    # would dominate a three-point regression).
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    F_hat = np.clip(y / n, 0.5 / n, 1.0 - 0.5 / n)
    y_regress = np.log(-np.log(1 - F_hat))
    res = linregress(np.log(x), y_regress)

    # recover parameters
    sh_hat = res.slope

    # A valid Weibull requires a positive shape. A non-positive slope means the
    # empirical quantiles have no (or a decreasing) log-log trend, giving a
    # degenerate fit: slope == 0 divides by zero below, and slope < 0 yields a
    # non-increasing "cdf". Reject so the caller records a failed run.
    if not np.isfinite(sh_hat) or sh_hat <= 0:
        raise ValueError(
            f"Degenerate Weibull fit: non-positive log-log slope (shape={sh_hat})."
        )

    # Even with a positive slope, a near-zero slope can overflow the scale. Compute
    # it quietly and reject a non-finite result rather than returning a garbage cdf.
    with np.errstate(over='ignore', invalid='ignore'):
        lamb_hat = np.exp(-res.intercept / res.slope)
    if not np.isfinite(lamb_hat) or lamb_hat <= 0:
        raise ValueError(
            f"Degenerate Weibull fit: non-finite scale estimate (scale={lamb_hat})."
        )

    # create sympy representation of estimated weibull
    weibull_cdf_result_nontrunc = 1 - sp.exp(-(x_sym / lamb_hat) ** sh_hat)
    # Normalize the CDF for truncation support (0, u)
    F_l = weibull_cdf_result_nontrunc.subs(x_sym, l)  # CDF at 0 (always 0 for Weibull)
    F_u = weibull_cdf_result_nontrunc.subs(x_sym, u)  # CDF at u
    weibull_cdf_result = (weibull_cdf_result_nontrunc - F_l) / (F_u - F_l)

    return weibull_cdf_result

