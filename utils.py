import sympy as sp
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar
import mpmath
from scipy.integrate import quad
from scipy import stats



def linear_interpolation(x, y, **kwargs):
    """
    Construct a sympy piecewise function representing the linear interpolation of y(x)
    given data points (x, y). The resulting function is only defined for
    x in [x[0], x[-1]].

    Args:
        x (list or np.ndarray): A sequence of x-values in strictly increasing order.
        y (list or np.ndarray): A sequence of corresponding y-values.
        **kwargs: Additional keyword arguments (none are actually used)

    Returns:
        sp.Piecewise: A sympy piecewise function which linearly interpolates y over the
                      interval [x[0], x[-1]].
    """
    # Data sanity checks
    assert len(x) == len(y), "x and y must have the same length"
    x, y = np.array(x), np.array(y)
    assert np.all(np.diff(x) > 0), "x must be strictly increasing"

    x_sym = sp.symbols('x')
    pieces = []
    n = len(x)

    # If there's only a single point, return a constant function.
    if n == 1:
        return sp.Piecewise((y[0], True))

    # Create linear interpolation pieces for each interval between adjacent points.
    for i in range(n - 1):
        left = x[i]
        right = x[i+1]
        # Compute the slope between the two points.
        slope = (y[i+1] - y[i]) / (right - left)
        # Linear expression: y = y_points[i] + slope*(x - left)
        expr = y[i] + slope * (x_sym - left)
        # Define the condition for this piece. For the first n-2 intervals, we use [left, right),
        # and for the final interval we include the right endpoint.
        if i < n - 2:
            condition = (x_sym >= left) & (x_sym < right)
        else:
            condition = (x_sym >= left) & (x_sym <= right)
        pieces.append((expr, condition))

    piecewise_function = sp.Piecewise(*pieces)
    return piecewise_function

'''
THE BELOW IMPLEMENTATION OF THE SCHUMAKER SPLINE IS COURTESY OF:
https://github.com/mmaaz-git/schumaker-spline
'''
def schumaker_spline(x, y, s = None, **kwargs):
    """
    Schumaker spline interpolation

    The Schumaker spline is a quadratic spline that is co-monotone and co-convex with the data.

    There is a slight modification in the slope estimation calculation: we set the slope of the first endpoint
    to be 0. This way there are no spurious new minima/maxima in the spline.

    Args:
        x (list or np.ndarray): A sequence of x-values in strictly increasing order.
        y (list or np.ndarray): A sequence of corresponding y-values.
        s (list or np.ndarray, optional): A sequence of slopes, optional.
        **kwargs: Additional keyword arguments (ignored for compatibility)
            - return_type (str, optional): 'coeffs' or 'sympy', default is 'sympy', if 'sympy', return a Sympy piecewise expression, otherwise return a matrix of the coefficients of the polynomials with associated x-coordinates

    Returns:
        array of coefficients for the Schumaker spline with corresponding x-coordinates, or a Sympy piecewise expression
    """
    x, y = np.array(x), np.array(y)
    n = len(x)
    assert len(y) == n, "x and y must have the same length"
    assert np.all(np.diff(x) > 0), "x must be in ascending order"
    if s is not None:
        s = np.array(s)
        assert len(s) == n, "s must have the same length as x and y"

    # if not given, estimate the slopes at the data points
    if s is None:
        L = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        delta =  np.diff(y) / np.diff(x)
        s = np.zeros(n)
        for i in range(1, n-1):
            if delta[i-1] * delta[i] > 0:
                s[i] = (L[i-1] * delta[i-1] + L[i] * delta[i]) / (L[i-1] + L[i])
            else:
                s[i] = 0
        #s[0] = (3*delta[0] - s[1]) / 2
        s[0] = 0
        s[n-1] = (3*delta[n-2] - s[n-2]) / 2

    # now loop over each interval and compute the knot and the quadratic spline(s)
    knots, coeffs = [], []
    for i in range(n-1):
        knots.append(x[i])

        # check if lemma is satisfied
        if (s[i] + s[i+1]) / 2 == (y[i+1] - y[i]) / (x[i+1] - x[i]):
            coeffs.append([y[i], # (x-x_i)^0 term
                           s[i], # (x-x_i)^1 term
                           (s[i+1] - s[i]) / (2 * (x[i+1] - x[i]))]) # (x-x_i)^2 term
        else:
            # have to add a new knot, depends on delta conditions
            if (s[i] - delta[i]) * (s[i+1] - delta[i]) >= 0:
                e = (x[i] + x[i+1]) / 2
            elif abs(s[i+1] - delta[i]) < abs(s[i] - delta[i]):
                e_temp = x[i] + 2*(x[i+1] - x[i])*(s[i+1] - delta[i]) / (s[i+1] - s[i])
                e = (x[i] + e_temp) / 2
            else:
                e_temp = x[i+1] + 2*(x[i+1] - x[i])*(s[i] - delta[i]) / (s[i+1] - s[i])
                e = (x[i+1] + e_temp) / 2

            knots.append(e)

            # compute the coefficients of the two splines, x[i] to e and e to x[i+1]
            # need some helper expressions
            alpha = e - x[i]
            beta = x[i+1] - e
            s_bar = (2*(y[i+1] - y[i]) - (alpha*s[i] + beta*s[i+1])) / (x[i+1] - x[i])
            A1 = y[i]
            B1 = s[i]
            C1 = (s_bar - s[i]) / (2 * alpha)
            A2 = A1 + alpha * B1 + alpha**2 * C1
            B2 = s_bar
            C2 = (s[i+1] - s_bar) / (2 * beta)
            # x[i] to e
            coeffs.append([A1,
                           B1,
                           C1])
            # e to x[i+1]
            coeffs.append([A2,
                           B2,
                           C2])

    knots.append(x[n-1])

    return_type = kwargs.get('return_type', 'sympy')
    if return_type == 'coeffs':
        return np.array(knots), np.array(coeffs)
    elif return_type == 'sympy':
        x_sym = sp.symbols('x')
        pieces = []
        for i in range(len(knots)-1):
            t = knots[i]
            a, b, c = coeffs[i]
            quadratic = a + b*(x_sym - t) + c*(x_sym - t)**2
            condition = (x_sym >= t) & (x_sym <= knots[i+1])
            pieces.append((quadratic, condition))
        return sp.Piecewise(*pieces)

def discrete_interpolation(x_knots, y_knots, **kwargs):
    """
    Interpolates between given knots (x_knots, y_knots) using a dense grid,
    enforcing monotonicity (decreasing) and convexity constraints via optimization.

    Args:
        x_knots (np.ndarray): X-coordinates of the knots (must be increasing).
        y_knots (np.ndarray): Y-coordinates of the knots (corresponding to t-values).
        **kwargs: Additional keyword arguments
            - num_subdivisions (int): Number of points to add between each pair of knots.
            - return_type (str): 'values' or 'sympy', default is 'values', if 'sympy', return a Sympy piecewise (stepwise) expression

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - x_dense: The dense grid of x-coordinates.
            - y_dense: The optimized y-coordinates (interpolated t-values) on the dense grid.
    """
    x_knots = np.asarray(x_knots)
    y_knots = np.asarray(y_knots, dtype=float)
    # Clamp extreme t-values so the LP coefficients stay numerically tractable.
    # Any M >= ~50 approximates t(u) -> -inf to machine precision (e^{-50} ~ 2e-22),
    # and fitting.py already forces F_dense[-1] = 1.0 after this call.
    y_knots = np.clip(y_knots, -50.0, 0.0)
    n_knots = len(x_knots)

    if n_knots <= 1:
        return x_knots, y_knots # Cannot interpolate with 0 or 1 knot

    num_subdivisions = kwargs.get('num_subdivisions', 15)

    # Create the dense grid
    x_dense_list = []
    original_knot_indices = [] # Keep track of indices corresponding to original knots
    current_dense_idx = 0
    for i in range(n_knots - 1):
        # Add points between x_knots[i] and x_knots[i+1]
        # Include the start point, add subdivisions-1 points in between, exclude end point (added by next iteration)
        points = np.linspace(x_knots[i], x_knots[i+1], num_subdivisions + 1, endpoint=False)
        x_dense_list.append(points)
        original_knot_indices.append(current_dense_idx)
        current_dense_idx += len(points)

    # Add the final knot and its index
    x_dense_list.append([x_knots[-1]])
    original_knot_indices.append(current_dense_idx)
    current_dense_idx += 1

    # Optionally extend the dense grid to x_end without fixing t(x_end).
    # This lets the LP choose a feasible t(u) freely (monotone + concave + <=0).
    x_end = kwargs.get('x_end', None)
    if x_end is not None and x_end > x_knots[-1]:
        extra = np.linspace(x_knots[-1], x_end, num_subdivisions + 1, endpoint=False)[1:]
        x_dense_list.append(extra)
        x_dense_list.append([x_end])

    x_dense = np.concatenate(x_dense_list)
    n_dense = len(x_dense)

    # Optimization variable for y-values on the dense grid
    y_dense_var = cp.Variable(n_dense)

    # Just want feasible solution
    objective = cp.Minimize(0)

    constraints = []

    # Constraint: Match the original knots
    constraints.append(y_dense_var[original_knot_indices] == y_knots)

    # Constraint: Monotonicity (decreasing)
    for i in range(n_dense - 1):
        constraints.append(y_dense_var[i] >= y_dense_var[i+1])

    # Constraint: Concavity (non-increasing slope)
    for i in range(1, n_dense - 1):
        # Directly apply the cross-multiplied slope condition
        # (y[i] - y[i-1]) * (x[i+1]-x[i]) >= (y[i+1] - y[i]) * (x[i]-x[i-1])
        constraints.append(
            (y_dense_var[i] - y_dense_var[i-1]) * (x_dense[i+1] - x_dense[i]) >=
            (y_dense_var[i+1] - y_dense_var[i]) * (x_dense[i] - x_dense[i-1])
        )

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    # Consider using a different solver if SCS struggles
    problem.solve(solver=cp.SCS)

    if y_dense_var.value is None:
        print("Warning: Discrete interpolation optimization failed. Status:", problem.status)

    return_type = kwargs.get('return_type', 'values')
    if return_type == 'values':
        return x_dense, y_dense_var.value
    elif return_type == 'sympy':
        # Create a sympy step function for plotting discrete CDF
        x_sym = sp.symbols('x')
        pieces = []
        pieces.append((y_dense_var.value[0], x_sym <= x_dense[0]))
        for i in range(len(x_dense) - 1):
            pieces.append((y_dense_var.value[i+1], (x_sym > x_dense[i]) & (x_sym <= x_dense[i+1])))
        pieces.append((y_dense_var.value[-1], x_sym > x_dense[-1]))
        pieces.insert(0, (0, x_sym < x_dense[0]))
        pieces.append((1, x_sym > x_dense[-1]))
        return sp.Piecewise(*pieces, (0, True))

def minimize_fn(fn, bounds):
    """Minimizes a lambdified function over a bounded interval.

    Uses scipy.optimize.minimize_scalar with method='bounded'.

    Args:
        fn: A callable (e.g. from sp.lambdify) accepting a scalar and returning a scalar.
        bounds (tuple): A (min, max) pair bounding the search interval.

    Returns:
        tuple[float, float]: The minimizing x-value and the minimum function value.
    """
    def objective(x): return float(fn(x))
    result = minimize_scalar(objective,
                     bounds=bounds,
                     method='bounded')

    return result.x, result.fun

def maximize_fn(fn, bounds):
    """Maximizes a lambdified function over a bounded interval.

    Uses scipy.optimize.minimize_scalar with method='bounded' on the negated function.

    Args:
        fn: A callable (e.g. from sp.lambdify) accepting a scalar and returning a scalar.
        bounds (tuple): A (min, max) pair bounding the search interval.

    Returns:
        tuple[float, float]: The maximizing x-value and the maximum function value.
    """
    def objective(x): return -float(fn(x))
    result = minimize_scalar(objective,
                     bounds=bounds,
                     method='bounded')

    return result.x, -result.fun

def compute_tvd(pdf_a, pdf_b, sym, l, u):
    """Computes the total variation distance between two pdfs.

    Args:
        pdf_a: First PDF (sympy expression).
        pdf_b: Second PDF (sympy expression).
        sym: The sympy symbol.
        l: Lower bound of the interval.
        u: Upper bound of the interval.

    Returns:
        float: The total variation distance, 0.5 * integral of |pdf_a - pdf_b| over [l, u].
    """
    integrand = 0.5 * sp.Abs(pdf_a - pdf_b)
    # mpmath is more precise but slower, and handles all sorts of functions
    integrand_mp = sp.lambdify(sym, integrand, 'mpmath')
    return float(mpmath.quad(integrand_mp, (l, u)))

def t_mean_confidence_interval(data, alpha):
    """Computes a t-distribution based confidence interval for the mean of a sample.

    Args:
        data (list or np.ndarray): Sample data.
        alpha (float): Significance level (e.g. 0.05 for a 95% confidence interval).

    Returns:
        tuple[float, float]: The sample mean and the half-width of the confidence interval.
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), np.std(a, ddof=1)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * se / np.sqrt(n)
    return m, h

# compute the L_inf distance between two cdfs
# aka the maximum absolute difference between the two cdfs
# aka the kolmogorov-smirnov distance
# added interval splitting to handle local maxima
def compute_l_inf(cdf_a, cdf_b, sym, l, u, num_intervals=10):
    """Computes the L-infinity distance by maximizing the absolute difference.
    Uses interval splitting to improve chances of finding global max.

    Args:
        cdf_a: First CDF (sympy expression).
        cdf_b: Second CDF (sympy expression).
        sym: The sympy symbol.
        l: Lower bound of the interval.
        u: Upper bound of the interval.
        num_intervals (int): Number of sub-intervals to check for maximization.

    Returns:
        float: The estimated L-infinity distance.
    """
    diff_fn = sp.lambdify(sym, sp.Abs(cdf_a - cdf_b), 'mpmath')

    if num_intervals <= 0:
        num_intervals = 1

    sub_bounds = np.linspace(l, u, num_intervals + 1)
    overall_max_val = -np.inf

    for i in range(num_intervals):
        interval = (sub_bounds[i], sub_bounds[i+1])
        try:
            # maximize_fn returns (x_loc, max_val)
            _, max_val_interval = maximize_fn(diff_fn, interval)
            overall_max_val = max(overall_max_val, float(max_val_interval))
        except Exception as e:
            try:
                val_l = diff_fn(interval[0])
                val_r = diff_fn(interval[1])
                overall_max_val = max(overall_max_val, float(val_l), float(val_r))
            except Exception:
                pass

    # If no successful maximization occurred, return 0 or handle error
    # Check against -np.inf ensures we return 0 if all optimizations/evaluations failed
    return overall_max_val if overall_max_val > -np.inf else 0.0

