import sympy as sp
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar
import mpmath
from scipy.integrate import quad
from scipy import stats



def linear_interpolation(x, y, **kwargs):
    """Construct a symbolic piecewise-linear interpolant.

    The interpolant is defined only on the closed interval [x[0], x[-1]].
    Outside that interval it evaluates to NaN, allowing the calling function
    to specify its own tail behavior explicitly.
    """
    # Convert inputs to floating-point NumPy arrays.  This ensures that slope
    # calculations use standard numeric arithmetic rather than list operations.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Check that the supplied knot locations and values are compatible.
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional arrays of equal length.")
    if len(x) == 0:
        raise ValueError("x and y must be nonempty.")

    # Piecewise-linear interpolation requires strictly increasing knot locations.
    if len(x) > 1 and not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")

    # Use the same symbolic variable convention as the rest of the package.
    x_sym = sp.symbols("x")

    # A one-point "interpolant" is defined only at that point.  This case occurs
    # in Algorithm 1 when every observed knot is an all-success knot.
    if len(x) == 1:
        return sp.Piecewise(
            (sp.Float(y[0]), sp.Eq(x_sym, sp.Float(x[0]))),
            (sp.nan, True),
        )

    pieces = []

    # Construct one affine expression on each adjacent interval [x_i, x_{i+1}].
    for i in range(len(x) - 1):
        left = x[i]
        right = x[i + 1]

        # The slope and affine expression through (left, y_i).
        slope = (y[i + 1] - y[i]) / (right - left)
        expression = y[i] + slope * (x_sym - left)

        # Use left-closed/right-open intervals except for the last interval.
        # This gives every point exactly one active branch while including x[-1].
        if i < len(x) - 2:
            condition = (x_sym >= left) & (x_sym < right)
        else:
            condition = (x_sym >= left) & (x_sym <= right)

        pieces.append((expression, condition))

    # Do not extrapolate linearly beyond the fitted knots.  The estimator's
    # analytic tail is constructed separately in fit_cdf_ifr.
    pieces.append((sp.nan, True))

    return sp.Piecewise(*pieces)

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

