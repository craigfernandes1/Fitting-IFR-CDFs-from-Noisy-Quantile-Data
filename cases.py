#%%

##############################################################################
##############################################################################
######################    PRICING EXPERIMENT    ##############################
##############################################################################
##############################################################################

#%%

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.stats import chi2, truncnorm, gamma as gamma_dist, logistic
from fitting import fit_cdf_ifr, fit_cdf_nonconvex, eval_step_cdf, l_inf_step_vs_cdf
from tqdm.auto import tqdm
import os

plt.rcParams.update({
    'axes.labelsize':  22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/data', exist_ok=True)


# ── Problem parameters ──────────────────────────────────────────────────────────
l, u     = 20, 50.0
x_knots  = np.array([25.0, 30.0, 35.0])   # sale, regular, high prices
n_trials = np.array([50, 100, 50])          # observations at each price
num_runs = 150  # simulation runs per ground-truth cdf (matches the paper)
d_ifr    = 2   # Discretized-IFR grid density; d=2 matches Section 7.1 of paper

x_sym     = sp.Symbol('x')
# Sup-norm / price grid: a dense grid plus the knots and endpoints, so the L-inf error
# cannot miss a fitted-knot or endpoint discrepancy
x_fine    = np.unique(np.concatenate((np.linspace(l, u, 5000), x_knots, [l, u])))

# ── Ground-truth distributions ──────────────────────────────────────────────────

class TruncatedDist:
    """Wraps a scipy distribution, normalizing CDF/PDF to [l, u].

    F_trunc(x) = (F_base(x) - F_base(l)) / (F_base(u) - F_base(l))
    """
    def __init__(self, base_dist, l, u):
        self._base = base_dist
        self.l, self.u = l, u
        self._Fl = base_dist.cdf(l)
        self._Fu = base_dist.cdf(u)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip((self._base.cdf(x) - self._Fl) / (self._Fu - self._Fl), 0.0, 1.0)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(
            (x >= self.l) & (x <= self.u),
            self._base.pdf(x) / (self._Fu - self._Fl),
            0.0,
        )


# All four are IFR (log-concave density) truncated to [l, u] = [20, 50].
# Revenue-optimal prices p* = argmax p (1 - F0(p)) on the truncated support:
#   Chi-Sq(38) = Gamma(19, scale=2)   p* ~ 28.1      Gamma(8, 6)       p* ~ 29.1
#   Normal(38, 6)                     p* ~ 30.3      Logistic(36, 5)   p* ~ 28.0
# The knots 25/30/35 bracket every p*; the Normal's p* sits on the regular price.
DISTRIBUTIONS = {
    'ChiSq':    {'label': 'Chi-Sq (df=38)',
                 'display_name': 'Chi-Sq',
                 'dist': TruncatedDist(chi2(38), l, u)},
    'Normal':   {'label': 'Normal (mu=38, sigma=6)',
                 'display_name': 'Normal',
                 'dist': truncnorm((l - 38.0) / 6.0, (u - 38.0) / 6.0, loc=38.0, scale=6.0)},
    'Gamma':    {'label': 'Gamma (alpha=8, theta=6)',
                 'display_name': 'Gamma',
                 'dist': TruncatedDist(gamma_dist(8, scale=6), l, u)},
    'Logistic': {'label': 'Logistic (mu=36, s=5)',
                 'display_name': 'Logistic',
                 'dist': TruncatedDist(logistic(loc=36, scale=5), l, u)},
}

# ── Method styling (shared across all figure cells) ─────────────────────────────
method_keys_list = ['algo1', 'discretized_ifr', 'discretized_non_ifr']
method_labels = {
    'algo1':               'Algorithm 1',
    'discretized_ifr':     r'Discretized-IFR ($d=2$)',
    'discretized_non_ifr': 'Discretized-non-IFR',
}
colors = {'algo1': 'green', 'discretized_ifr': 'red', 'discretized_non_ifr': 'purple'}


def grouped_mean_sd_bars(ax, arr_3d, methods, labels, colors, x_pos, width=0.22):
    """Grouped bar chart of per-group means with +-1 sample-sd error bars.

    Shared by every bar panel in this script (pricing, reliability, Weibull).

    Args:
        ax: Target axes.
        arr_3d (np.ndarray): Shape (n_groups, len(methods), n_runs). NaN entries
            are failed runs and are dropped before averaging.
        methods (list[str]): Method keys in plotting order (one bar per group).
        labels (dict): Method key -> legend label.
        colors (dict): Method key -> bar colour.
        x_pos (np.ndarray): Group centres on the x-axis.
        width (float): Centre-to-centre spacing of bars within a group.
    """
    n_m     = len(methods)
    offsets = np.linspace(-(n_m - 1) * width / 2, (n_m - 1) * width / 2, n_m)
    for j, m in enumerate(methods):
        means = np.full(len(x_pos), np.nan)
        sds   = np.full(len(x_pos), np.nan)
        for g in range(len(x_pos)):
            vals = arr_3d[g, j, :]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                means[g] = vals.mean()
                sds[g]   = vals.std(ddof=1) if vals.size > 1 else 0.0
        ax.bar(
            x_pos + offsets[j], means, width=width * 0.9,
            yerr=sds, color=colors[m], alpha=0.7,
            edgecolor='black', linewidth=0.5,
            error_kw=dict(ecolor='black', elinewidth=1.2, capsize=3),
            label=labels[m],
        )


def _l_inf_step(x_nc, F_nc, true_cdf_fn, x_max=None):
    """L-inf between a step CDF (fit_cdf_nonconvex output) and the true CDF.

    Thin adapter to the shared right-continuous step-cdf sup-norm evaluator in
    fitting.py, so the step-cdf convention is defined in exactly one place.
    """
    return l_inf_step_vs_cdf(x_nc, F_nc, true_cdf_fn, x_max)

#%%

x_plot = np.linspace(l, u, 500)

fig, axes = plt.subplots(3, len(DISTRIBUTIONS), figsize=(6 * len(DISTRIBUTIONS), 11))

for j, (dist_key, dist_cfg) in enumerate(DISTRIBUTIONS.items()):
    dist     = dist_cfg['dist']
    pdf_vals = dist.pdf(x_plot)
    cdf_vals = dist.cdf(x_plot)
    rev_vals = x_plot * (1 - cdf_vals)

    opt_idx  = int(np.argmax(rev_vals))
    opt_p    = x_plot[opt_idx]
    opt_rev  = rev_vals[opt_idx]

    ax_pdf = axes[0, j]
    ax_cdf = axes[1, j]
    ax_rev = axes[2, j]

    ax_pdf.plot(x_plot, pdf_vals, color='steelblue', linewidth=2)
    for xk in x_knots:
        ax_pdf.axvline(xk, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax_pdf.set_title(dist_cfg['label'], fontsize=14)
    ax_pdf.set_xlabel('x')
    ax_pdf.set_ylim(bottom=0)
    ax_pdf.grid(True, ls='--', alpha=0.5)

    ax_cdf.plot(x_plot, cdf_vals, color='steelblue', linewidth=2)
    for xk in x_knots:
        ax_cdf.axvline(xk, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax_cdf.scatter([xk], [dist.cdf(np.array([xk]))],
                       color='red', zorder=5, s=80, marker='x', linewidths=2)
    ax_cdf.set_xlabel('x')
    ax_cdf.set_ylim(0, 1)
    ax_cdf.grid(True, ls='--', alpha=0.5)

    ax_rev.plot(x_plot, rev_vals, color='steelblue', linewidth=2)
    ax_rev.axvline(opt_p, color='black', linestyle=':', linewidth=1.5, alpha=0.8)
    ax_rev.scatter([opt_p], [opt_rev], color='black', zorder=5, s=60)
    ax_rev.annotate(f'$p^*={opt_p:.1f}$', xy=(opt_p, opt_rev),
                    xytext=(6, -12), textcoords='offset points', fontsize=12)
    ax_rev.set_xlabel('Price $p$')
    ax_rev.set_ylim(bottom=0)
    ax_rev.grid(True, ls='--', alpha=0.5)

axes[0, 0].set_ylabel('PDF', fontsize=18)
axes[1, 0].set_ylabel('CDF', fontsize=18)
axes[2, 0].set_ylabel(r'Revenue $p(1-F(p))$', fontsize=18)

plt.suptitle(f'Ground-Truth Distributions on [{l:.0f}, {u:.0f}]  (red crosses = knot positions)', fontsize=16)
plt.tight_layout()
plt.savefig('results/plots/pricing_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

#%%

np.random.seed(0)

dist_names_list = list(DISTRIBUTIONS.keys())
n_dists         = len(dist_names_list)
n_methods       = len(method_keys_list)
shape           = (n_dists, n_methods, num_runs)

all_errors      = np.full(shape, np.nan)
all_revenues    = np.full(shape, np.nan)

for di, (dist_name, dist_cfg) in enumerate(DISTRIBUTIONS.items()):
    dist      = dist_cfg['dist']
    true_p    = dist.cdf(x_knots)
    true_fine = dist.cdf(x_fine)

    # True optimal price and revenue (oracle)
    true_rev      = x_fine * (1 - true_fine)
    true_opt_idx  = int(np.argmax(true_rev))
    true_opt_rev  = float(true_rev[true_opt_idx])
    print(f"{dist_name}: opt price = {x_fine[true_opt_idx]:.2f}, opt rev = {true_opt_rev:.4f}")

    for run_idx in tqdm(range(num_runs), desc=dist_name, leave=False):
        y_samples = np.random.binomial(n_trials, true_p)

        # ── Algorithm 1 (PWL, CVXPY) ─────────────────────────────────────────────
        try:
            fitted = fit_cdf_ifr(x_knots, n_trials, y_samples, l, u, 'linear')
            fn     = sp.lambdify(x_sym, fitted, 'numpy')
            with np.errstate(invalid='ignore', over='ignore'):
                vals = np.asarray(fn(x_fine), dtype=float)
            # Detect a numerical failure rather than masking a nonfinite value to 1
            if not np.all(np.isfinite(vals)):
                raise RuntimeError("Nonfinite fitted-cdf evaluation.")
            vals = np.clip(vals, 0.0, 1.0)
            all_errors[di, 0, run_idx]      = float(np.max(np.abs(vals - true_fine)))
            opt_idx = int(np.argmax(x_fine * (1 - vals)))
            all_revenues[di, 0, run_idx] = float(x_fine[opt_idx] * (1 - true_fine[opt_idx])) / true_opt_rev
        except Exception as e:
            print(f"  [{dist_name}, run {run_idx}] algo1 failed: {e}")

        # ── Discretized-IFR (Gurobi, d=d_ifr) ───────────────────────────────────
        try:
            x_nc, F_nc = fit_cdf_nonconvex(
                x_knots, n_trials, y_samples, l, u, d_ifr,
                ifr_constraint=True, verbose=False, return_type='values',
            )
            all_errors[di, 1, run_idx]      = _l_inf_step(x_nc, F_nc, dist.cdf)
            F_nc_fine = eval_step_cdf(x_nc, F_nc, x_fine)
            opt_idx = int(np.argmax(x_fine * (1 - F_nc_fine)))
            all_revenues[di, 1, run_idx] = float(x_fine[opt_idx] * (1 - true_fine[opt_idx])) / true_opt_rev
        except Exception as e:
            print(f"  [{dist_name}, run {run_idx}] discretized_ifr failed: {e}")

        # ── Discretized-non-IFR (Gurobi, monotonicity only) ─────────────────────
        try:
            x_ni, F_ni = fit_cdf_nonconvex(
                x_knots, n_trials, y_samples, l, u, 0,
                ifr_constraint=False, verbose=False, return_type='values',
            )
            all_errors[di, 2, run_idx]      = _l_inf_step(x_ni, F_ni, dist.cdf)
            F_ni_fine = eval_step_cdf(x_ni, F_ni, x_fine)
            opt_idx = int(np.argmax(x_fine * (1 - F_ni_fine)))
            all_revenues[di, 2, run_idx] = float(x_fine[opt_idx] * (1 - true_fine[opt_idx])) / true_opt_rev
        except Exception as e:
            print(f"  [{dist_name}, run {run_idx}] discretized_non_ifr failed: {e}")

np.savez('results/data/pricing_case_study.npz',
    dist_names=np.array(dist_names_list),
    method_keys=np.array(method_keys_list),
    num_runs=num_runs,
    all_errors=all_errors,
    all_revenues=all_revenues,
)
print('Saved results/data/pricing_case_study.npz')

#%%

data = np.load('results/data/pricing_case_study.npz', allow_pickle=True)
dist_names_list  = data['dist_names'].tolist()
method_keys_list = data['method_keys'].tolist()
all_errors      = data['all_errors']       # (n_dists, n_methods, num_runs)

n_dists   = len(dist_names_list)
x_pos     = np.arange(n_dists)
display_names = [DISTRIBUTIONS[k]['display_name'] for k in dist_names_list]

# --- Combined pricing figure: full-support error | revenue ratio | illustrative curve ---
all_revenues = data['all_revenues']

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

# Panel 1: full-support L-inf error (mean ± SD bars)
ax_full = axes[0]
grouped_mean_sd_bars(ax_full, all_errors, method_keys_list, method_labels, colors, x_pos)
ax_full.set_xticks(x_pos)
ax_full.set_xticklabels(display_names)
ax_full.set_ylabel(r'$\|F_0 - \hat{F}\|_{[l,u]}$')
ax_full.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax_full.set_ylim([0, 0.85])
ax_full.grid(True, which='both', ls='--')

# Panel 2: revenue ratio (mean ± SD bars)
ax_ratio = axes[1]
grouped_mean_sd_bars(ax_ratio, all_revenues, method_keys_list, method_labels, colors, x_pos)

ax_ratio.axhline(1.0, color='black', linewidth=1, alpha=0.5)
ax_ratio.set_xticks(x_pos)
ax_ratio.set_xticklabels(display_names)
ax_ratio.set_ylabel('Revenue Ratio')
ax_ratio.set_ylim([-0.05, 1.1])
ax_ratio.grid(True, which='both', ls='--')

# Panel 3: illustrative revenue curves for one simulated instance (ground truth
# and seed set just below). The draw is a fresh dataset, not one of the num_runs
# tallied above; state the family and seed in the figure caption.
np.random.seed(3)
ax_ex = axes[2]

ex_dist = DISTRIBUTIONS['Gamma']['dist']
true_fine = ex_dist.cdf(x_fine)
true_p    = ex_dist.cdf(x_knots)
y_samples = np.random.binomial(n_trials, true_p)

true_rev     = x_fine * (1 - true_fine)
true_opt_idx = int(np.argmax(true_rev))

ax_ex.plot(x_fine, true_rev, 'k-', linewidth=2, label='True', zorder=5)
ax_ex.axvline(x_fine[true_opt_idx], color='black', linestyle=':', linewidth=1.5, alpha=0.7)

try:
    fitted = fit_cdf_ifr(x_knots, n_trials, y_samples, l, u, 'linear')
    fn = sp.lambdify(x_sym, fitted, 'numpy')
    with np.errstate(invalid='ignore', over='ignore'):
        vals = np.asarray(fn(x_fine), dtype=float)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError("Nonfinite fitted-cdf evaluation.")
    vals = np.clip(vals, 0.0, 1.0)
    rev = x_fine * (1 - vals)
    ax_ex.plot(x_fine, rev, color='green', linewidth=1.5, label='Algorithm 1')
    ax_ex.axvline(x_fine[int(np.argmax(rev))], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
except Exception as e:
    print(f"algo1 failed: {e}")

try:
    x_nc, F_nc = fit_cdf_nonconvex(x_knots, n_trials, y_samples, l, u, d_ifr,
                                    ifr_constraint=True, verbose=False, return_type='values')
    F_nc_fine = eval_step_cdf(x_nc, F_nc, x_fine)
    rev = x_fine * (1 - F_nc_fine)
    ax_ex.plot(x_fine, rev, color='red', linewidth=1.5, label=r'Discretized-IFR ($d=2$)')
    ax_ex.axvline(x_fine[int(np.argmax(rev))], color='red', linestyle=':', linewidth=1.5, alpha=0.7)
except Exception as e:
    print(f"discretized_ifr failed: {e}")

try:
    x_ni, F_ni = fit_cdf_nonconvex(x_knots, n_trials, y_samples, l, u, 0,
                                    ifr_constraint=False, verbose=False, return_type='values')
    F_ni_fine = eval_step_cdf(x_ni, F_ni, x_fine)
    rev = x_fine * (1 - F_ni_fine)
    ax_ex.plot(x_fine, rev, color='purple', linewidth=1.5, label='Discretized-non-IFR')
    ax_ex.axvline(x_fine[int(np.argmax(rev))], color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
except Exception as e:
    print(f"discretized_non_ifr failed: {e}")

ax_ex.set_xlabel('Price')
ax_ex.set_ylabel('Expected Revenue')
ax_ex.set_xlim(l, u)
ax_ex.grid(True, linestyle='--', alpha=0.4)

# One shared legend floating beneath all three panels
handles, labels = ax_ex.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
           ncol=len(labels), frameon=True)
plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig('results/plots/pricing_case_study.png', dpi=150, bbox_inches='tight')
plt.show()

#%%

##############################################################################
##############################################################################
####################    RELIABILITY  EXPERIMENT    ###########################
##############################################################################
##############################################################################

#%%

import warnings
from scipy.integrate import quad, IntegrationWarning
from fitting import fit_weibull_cdf
from scipy.stats import weibull_min as weibull_scipy

# ─────────────────────────────────────────────────────────────────────────────
# Preventive-maintenance decision search.
#
#   C(kappa; F) = ( c_p (1 - F(kappa)) + c_f F(kappa) ) / int_l^kappa (1 - F(t)) dt
#
# The replacement age kappa is selected by a global search over a common candidate
# grid, applied identically to the true cdf and every estimator. A continuous
# unimodal optimizer is not used because the discretized-benchmark cost is a
# discontinuous function of kappa. The denominator survival integral is computed
# exactly (interval by interval) for step cdfs and by cumulative quadrature over
# consecutive candidate gaps for continuous cdfs. A non-finite segment integral is
# reported as a failure; a benign roundoff IntegrationWarning on a finite integral
# is suppressed. A run whose decision cannot be evaluated is recorded as NaN (a
# dropped run), never as a ratio of zero.
# ─────────────────────────────────────────────────────────────────────────────

KAPPA_GRID_N    = 1000     # density of the common global candidate grid on [l+0.01, u]
_LEFT_LIMIT_EPS = 1e-6     # left-limit offset (breakpoint - eps) for step-cdf jumps

def kappa_candidates(l, u, breakpoints=None):
    ## Common ascending candidate grid on [l + 0.01, u]: a dense grid, the endpoint u,
    ## and (for step cdfs or piecewise kinks) all breakpoints plus their left-limits.
    ## Ascending order makes np.argmin select the smallest minimizer on a tie.
    cand = [np.linspace(l + 0.01, u, KAPPA_GRID_N), np.array([float(u)])]
    if breakpoints is not None:
        bp = np.asarray(breakpoints, dtype=float)
        cand.append(bp)
        cand.append(bp - _LEFT_LIMIT_EPS)
    c = np.unique(np.concatenate(cand))
    return c[(c >= l + 0.01) & (c <= u)]

def best_kappa(cand, cost_vals):
    ## Smallest minimizer of cost_vals over the ascending grid cand (inf/NaN-safe).
    cost_vals = np.where(np.isfinite(cost_vals), cost_vals, np.inf)
    if not np.any(np.isfinite(cost_vals)):
        return np.nan, np.nan
    i = int(np.argmin(cost_vals))
    return float(cand[i]), float(cost_vals[i])

def _step_survival_integral(x_grid, F_grid, l, kappa):
    ## Exact int_l^kappa (1 - F_step(t)) dt for a right-continuous step cdf
    ## (value F_grid[j] on [x_grid[j], x_grid[j+1])).
    x_grid = np.asarray(x_grid, dtype=float)
    S  = 1.0 - np.asarray(F_grid, dtype=float)[:-1]
    lo = np.maximum(x_grid[:-1], l)
    hi = np.minimum(x_grid[1:], kappa)
    return float(np.sum(S * np.clip(hi - lo, 0.0, None)))

def costs_step(x_grid, F_grid, c_p, c_f, l, cand):
    ## C(kappa; F) over the candidate grid for a step cdf, using the exact integral.
    integ  = np.array([_step_survival_integral(x_grid, F_grid, l, k) for k in cand])
    F_cand = eval_step_cdf(x_grid, F_grid, cand)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(integ > 0, (c_p * (1 - F_cand) + c_f * F_cand) / integ, np.inf)

def costs_continuous(cdf_fn, c_p, c_f, l, cand):
    ## C(kappa; F) over the candidate grid for a continuous cdf. The survival integral
    ## is accumulated by quadrature over consecutive candidate gaps. A genuine failure
    ## is a *non-finite* segment integral; that is recorded (and reported). A benign
    ## roundoff IntegrationWarning on a near-step but finite integrand (e.g. a Weibull
    ## fit with very large shape) still returns a usable value, so it is suppressed
    ## rather than flagged.
    cand   = np.asarray(cand, dtype=float)
    F_cand = np.array([float(cdf_fn(k)) for k in cand])
    integ  = np.empty(len(cand))
    run, prev, n_fail = 0.0, float(l), 0
    for m, k in enumerate(cand):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', IntegrationWarning)
            seg, _ = quad(lambda t: 1.0 - cdf_fn(t), prev, k, limit=200)
        if not np.isfinite(seg):
            n_fail += 1
            seg = 0.0   # keep the running total finite; this candidate becomes unusable
        run += seg
        integ[m] = run
        prev = k
    if n_fail:
        print(f"  [cost] {n_fail} non-finite integral segment(s) during kappa search")
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(integ > 0, (c_p * (1 - F_cand) + c_f * F_cand) / integ, np.inf)

def cost_continuous_at(cdf_fn, c_p, c_f, l, kappa):
    ## C(kappa; F) at a single kappa for a continuous cdf; used to evaluate the TRUE
    ## cost at an estimator's chosen replacement age. Returns inf only on a genuine
    ## failure (non-finite kappa or non-positive/non-finite integral); a benign
    ## roundoff IntegrationWarning on a finite integral is suppressed.
    if not np.isfinite(kappa):
        return np.inf
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntegrationWarning)
        integral, _ = quad(lambda t: 1.0 - cdf_fn(t), l, kappa, limit=400)
    if not np.isfinite(integral) or integral <= 0:
        return np.inf
    return (c_p * (1 - cdf_fn(kappa)) + c_f * cdf_fn(kappa)) / integral

def true_cost_ratio(true_cdf_fn, c_p, c_f, l, cost_star, kappa_hat):
    ## Cost ratio C(kappa*; F0) / C(kappa_hat; F0) of an estimator's chosen
    ## replacement age under the ground truth. Raises (so the caller records the
    ## run as failed, i.e. NaN) when kappa_hat is undefined or its true cost is
    ## non-finite; without this guard such a run would be stored as a ratio of 0.
    c_hat = cost_continuous_at(true_cdf_fn, c_p, c_f, l, kappa_hat)
    if not np.isfinite(c_hat) or c_hat <= 0:
        raise RuntimeError(f"True cost at kappa_hat={kappa_hat} is not finite and positive.")
    return cost_star / c_hat

# ── Part 1 parameters ─────────────────────────────────────────────────────────────
l_r, u_r      = 0.0, 12.0
x_knots_r     = np.array([2.0, 4.0, 6.0])
n_per_knot_r  = 50
n_trials_r    = n_per_knot_r * np.ones(len(x_knots_r), dtype=int)
num_runs_r    = 150
c_p_r, c_f_r  = 300, 1200
x_fine_r      = np.unique(np.concatenate((np.linspace(l_r + 1e-4, u_r, 3000), x_knots_r, [l_r, u_r])))
d_ifr_r       = 2

# All four are IFR, truncated to [l_r, u_r] = [0, 12]. Optimal replacement ages
# kappa* = argmin C(kappa; F0) under c_p = 300, c_f = 1200 lie between the 4- and
# 6-month knots:
#   Chi-Sq(9)   = Gamma(4.5, scale=2): mode 7,  kappa* ~ 4.4
#   Normal(9, 3):                      mode 9,  kappa* ~ 5.4
#   Gamma(5, 2):                       mode 8,  kappa* ~ 4.8
#   Logistic(9, 2):                    mode 9,  kappa* ~ 5.6  (increasing hazard)
DIST_RELIABILITY = {
    'ChiSq':    {'label': 'Chi-Sq (df=9)',        'display_name': 'Chi-Sq',
                 'dist': TruncatedDist(chi2(9), l_r, u_r)},
    'Normal':   {'label': r'Normal ($\mu$=9, $\sigma$=3)', 'display_name': 'Normal',
                 'dist': truncnorm((l_r - 9.0) / 3.0, (u_r - 9.0) / 3.0, loc=9.0, scale=3.0)},
    'Gamma':    {'label': r'Gamma ($\alpha$=5, $\theta$=2)', 'display_name': 'Gamma',
                 'dist': TruncatedDist(gamma_dist(5, scale=2), l_r, u_r)},
    'Logistic': {'label': r'Logistic ($\mu$=9, s=2)', 'display_name': 'Logistic',
                 'dist': TruncatedDist(logistic(loc=9.0, scale=2.0), l_r, u_r)},
}

rel_method_keys = ['algo1', 'discretized_ifr', 'discretized_non_ifr']
rel_colors      = {'algo1': 'green', 'discretized_ifr': 'red', 'discretized_non_ifr': 'purple'}
rel_labels      = {
    'algo1':               'Algorithm 1',
    'discretized_ifr':     r'Discretized-IFR ($d=2$)',
    'discretized_non_ifr': 'Discretized-non-IFR',
}

#%%

x_plot_r = np.linspace(l_r, u_r, 500)
x_cost_plot_r = x_plot_r[x_plot_r > l_r + 0.05]

fig, axes = plt.subplots(3, len(DIST_RELIABILITY), figsize=(6 * len(DIST_RELIABILITY), 11))

for j, (dist_key, dist_cfg) in enumerate(DIST_RELIABILITY.items()):
    dist        = dist_cfg['dist']
    pdf_vals    = dist.pdf(x_plot_r)
    cdf_vals    = dist.cdf(x_plot_r)
    true_cdf_fn = lambda t, _d=dist: float(np.clip(_d.cdf(np.array([t])), 0, 1).flat[0])
    cost_vals   = costs_continuous(true_cdf_fn, c_p_r, c_f_r, l_r, x_cost_plot_r)

    cand_true     = kappa_candidates(l_r, u_r)
    kappa_star, _ = best_kappa(cand_true, costs_continuous(true_cdf_fn, c_p_r, c_f_r, l_r, cand_true))
    cost_star     = cost_continuous_at(true_cdf_fn, c_p_r, c_f_r, l_r, kappa_star)

    ax_pdf  = axes[0, j]
    ax_cdf  = axes[1, j]
    ax_cost = axes[2, j]

    ax_pdf.plot(x_plot_r, pdf_vals, color='steelblue', linewidth=2)
    for xk in x_knots_r:
        ax_pdf.axvline(xk, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax_pdf.set_title(dist_cfg['label'], fontsize=14)
    ax_pdf.set_xlabel('x')
    ax_pdf.set_ylim(bottom=0)
    ax_pdf.grid(True, ls='--', alpha=0.5)

    ax_cdf.plot(x_plot_r, cdf_vals, color='steelblue', linewidth=2)
    for xk in x_knots_r:
        ax_cdf.axvline(xk, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax_cdf.scatter([xk], [dist.cdf(np.array([xk]))],
                       color='red', zorder=5, s=80, marker='x', linewidths=2)
    ax_cdf.set_xlabel('x')
    ax_cdf.set_ylim(0, 1)
    ax_cdf.grid(True, ls='--', alpha=0.5)

    ax_cost.plot(x_cost_plot_r, cost_vals, color='steelblue', linewidth=2)
    ax_cost.axvline(kappa_star, color='black', linestyle=':', linewidth=1.5, alpha=0.8)
    ax_cost.scatter([kappa_star], [cost_star], color='black', zorder=5, s=60)
    ax_cost.annotate(fr'$\kappa^*={kappa_star:.1f}$', xy=(kappa_star, cost_star),
                     xytext=(6, 8), textcoords='offset points', fontsize=12)
    ax_cost.set_xlabel(r'Replacement interval $\kappa$')
    ax_cost.set_ylim(bottom=0, top = 250)
    ax_cost.grid(True, ls='--', alpha=0.5)

axes[0, 0].set_ylabel('PDF', fontsize=18)
axes[1, 0].set_ylabel('CDF', fontsize=18)
axes[2, 0].set_ylabel(r'Cost $C(\kappa;F)$', fontsize=18)

plt.suptitle(f'Ground-Truth Distributions on [{l_r:.0f}, {u_r:.0f}]  (red crosses = knot positions)', fontsize=16)
plt.tight_layout()
plt.savefig('results/plots/reliability_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

#%%

np.random.seed(0)

rel_dist_names    = list(DIST_RELIABILITY.keys())
n_rd = len(rel_dist_names)
n_rm = len(rel_method_keys)
all_errors_r      = np.full((n_rd, n_rm, num_runs_r), np.nan)
all_costs_r       = np.full((n_rd, n_rm, num_runs_r), np.nan)

for di, (dist_name, dist_cfg) in enumerate(DIST_RELIABILITY.items()):
    dist           = dist_cfg['dist']
    true_cdf_fn    = lambda t, _d=dist: float(np.clip(_d.cdf(np.array([t])), 0, 1).flat[0])
    true_p         = dist.cdf(x_knots_r)
    true_fine      = dist.cdf(x_fine_r)

    cand_true      = kappa_candidates(l_r, u_r)
    kappa_star, _  = best_kappa(cand_true, costs_continuous(true_cdf_fn, c_p_r, c_f_r, l_r, cand_true))
    cost_star      = cost_continuous_at(true_cdf_fn, c_p_r, c_f_r, l_r, kappa_star)
    print(f"{dist_name}: kappa* = {kappa_star:.2f} months, C(kappa*) = {cost_star:.4f}")

    for run in tqdm(range(num_runs_r), desc=dist_name, leave=False):
        y_samples = np.random.binomial(n_trials_r, true_p)

        # ── Algorithm 1 (PWL) ─────────────────────────────────────────────────────
        try:
            fitted = fit_cdf_ifr(x_knots_r, n_trials_r, y_samples, l_r, u_r, 'linear')
            fn_raw = sp.lambdify(x_sym, fitted, 'numpy')
            with np.errstate(invalid='ignore', over='ignore'):
                vals = np.asarray(fn_raw(x_fine_r), dtype=float)
            # Detect a numerical failure rather than masking a nonfinite value to 1
            if not np.all(np.isfinite(vals)):
                raise RuntimeError("Nonfinite fitted-cdf evaluation.")
            vals = np.clip(vals, 0.0, 1.0)
            all_errors_r[di, 0, run]      = float(np.max(np.abs(vals - true_fine)))
            def fn_alg(t, _f=fn_raw):
                with np.errstate(invalid='ignore', over='ignore'):
                    return float(np.clip(np.asarray(_f(t), dtype=float).flat[0], 0.0, 1.0))
            cand_alg      = kappa_candidates(l_r, u_r, breakpoints=x_knots_r)
            kappa_alg, _  = best_kappa(cand_alg, costs_continuous(fn_alg, c_p_r, c_f_r, l_r, cand_alg))
            all_costs_r[di, 0, run] = true_cost_ratio(true_cdf_fn, c_p_r, c_f_r, l_r, cost_star, kappa_alg)
        except Exception as e:
            print(f"  [{dist_name}, run {run}] algo1 failed: {e}")

        # ── Discretized-IFR (Gurobi, d=d_ifr_r) ─────────────────────────────────
        try:
            x_nc, F_nc = fit_cdf_nonconvex(
                x_knots_r, n_trials_r, y_samples, l_r, u_r, d_ifr_r,
                ifr_constraint=True, verbose=False, return_type='values',
            )
            all_errors_r[di, 1, run]      = _l_inf_step(x_nc, F_nc, dist.cdf)
            cand_disc     = kappa_candidates(l_r, u_r, breakpoints=x_nc)
            kappa_disc, _ = best_kappa(cand_disc, costs_step(x_nc, F_nc, c_p_r, c_f_r, l_r, cand_disc))
            all_costs_r[di, 1, run] = true_cost_ratio(true_cdf_fn, c_p_r, c_f_r, l_r, cost_star, kappa_disc)
        except Exception as e:
            print(f"  [{dist_name}, run {run}] discretized_ifr failed: {e}")

        # ── Discretized-non-IFR (Gurobi, monotonicity only) ─────────────────────
        try:
            x_ni, F_ni = fit_cdf_nonconvex(
                x_knots_r, n_trials_r, y_samples, l_r, u_r, 0,
                ifr_constraint=False, verbose=False, return_type='values',
            )
            all_errors_r[di, 2, run]      = _l_inf_step(x_ni, F_ni, dist.cdf)
            cand_nifr     = kappa_candidates(l_r, u_r, breakpoints=x_ni)
            kappa_nifr, _ = best_kappa(cand_nifr, costs_step(x_ni, F_ni, c_p_r, c_f_r, l_r, cand_nifr))
            all_costs_r[di, 2, run] = true_cost_ratio(true_cdf_fn, c_p_r, c_f_r, l_r, cost_star, kappa_nifr)
        except Exception as e:
            print(f"  [{dist_name}, run {run}] discretized_non_ifr failed: {e}")

np.savez('results/data/reliability_case_study_multi_dist.npz',
    dist_names=np.array(rel_dist_names),
    all_errors=all_errors_r,
    all_costs=all_costs_r,
)
print('Saved results/data/reliability_case_study_multi_dist.npz')

#%%

data_r = np.load('results/data/reliability_case_study_multi_dist.npz', allow_pickle=True)
rel_dist_names    = data_r['dist_names'].tolist()
all_errors_r      = data_r['all_errors']
all_costs_r       = data_r['all_costs']

n_rd  = len(rel_dist_names)
x_pos_r         = np.arange(n_rd)
display_names_r = [DIST_RELIABILITY[k]['display_name'] for k in rel_dist_names]

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

# Panel 1: full-support L-inf error (mean ± SD bars)
ax_full = axes[0]
grouped_mean_sd_bars(ax_full, all_errors_r, rel_method_keys, rel_labels, rel_colors, x_pos_r)
ax_full.set_xticks(x_pos_r)
ax_full.set_xticklabels(display_names_r)
ax_full.set_ylabel(r'$\|F_0 - \hat{F}\|_{[l,u]}$')
ax_full.set_ylim(0, 0.85)
ax_full.grid(True, axis='y', ls='--')

# Panel 2: cost ratio (mean ± SD bars)
ax_cost = axes[1]
grouped_mean_sd_bars(ax_cost, all_costs_r, rel_method_keys, rel_labels, rel_colors, x_pos_r)
ax_cost.axhline(1.0, color='black', linewidth=1, alpha=0.5)
ax_cost.set_xticks(x_pos_r)
ax_cost.set_xticklabels(display_names_r)
ax_cost.set_ylabel('Cost Ratio')
ax_cost.set_ylim(0.5, 1.025)
ax_cost.grid(True, axis='y', ls='--')

# Panel 3: illustrative cost curves for one simulated instance (ground truth and
# seed set just below). The draw is a fresh dataset, not one of the num_runs_r
# tallied above; state the family and seed in the figure caption.
np.random.seed(0)
ax_ex = axes[2]
ex_dist      = DIST_RELIABILITY['ChiSq']['dist']
true_p_ex    = ex_dist.cdf(x_knots_r)
y_samples_ex = np.random.binomial(n_trials_r, true_p_ex)

true_cdf_fn_ex   = lambda t, _d=ex_dist: float(np.clip(_d.cdf(np.array([t])), 0, 1).flat[0])
cand_ex          = kappa_candidates(l_r, u_r)
kappa_star_ex, _ = best_kappa(cand_ex, costs_continuous(true_cdf_fn_ex, c_p_r, c_f_r, l_r, cand_ex))

x_cost_plot = x_fine_r[x_fine_r > l_r + 0.05]
true_cv = costs_continuous(true_cdf_fn_ex, c_p_r, c_f_r, l_r, x_cost_plot)

ax_ex.plot(x_cost_plot, true_cv, 'k-', linewidth=2, label='True', zorder=5)
ax_ex.axvline(kappa_star_ex, color='black', linestyle=':', linewidth=1.5, alpha=0.7)

try:
    fitted   = fit_cdf_ifr(x_knots_r, n_trials_r, y_samples_ex, l_r, u_r, 'linear')
    fn_raw   = sp.lambdify(x_sym, fitted, 'numpy')
    def fn_s(t, _f=fn_raw):
        with np.errstate(invalid='ignore', over='ignore'):
            return float(np.clip(np.asarray(_f(t), dtype=float).flat[0], 0, 1))
    cv       = costs_continuous(fn_s, c_p_r, c_f_r, l_r, x_cost_plot)
    cand_hat     = kappa_candidates(l_r, u_r, breakpoints=x_knots_r)
    kappa_hat, _ = best_kappa(cand_hat, costs_continuous(fn_s, c_p_r, c_f_r, l_r, cand_hat))
    m = np.isfinite(cv)
    ax_ex.plot(x_cost_plot[m], cv[m], color=rel_colors['algo1'], linewidth=1.5, label=rel_labels['algo1'])
    ax_ex.axvline(kappa_hat, color=rel_colors['algo1'], linestyle=':', linewidth=1.5, alpha=0.7)
except Exception as e:
    print(f"algo1 failed: {e}")

try:
    x_nc, F_nc = fit_cdf_nonconvex(x_knots_r, n_trials_r, y_samples_ex, l_r, u_r, d_ifr_r,
                                    ifr_constraint=True, verbose=False, return_type='values')
    cv       = costs_step(x_nc, F_nc, c_p_r, c_f_r, l_r, x_cost_plot)
    cand_d   = kappa_candidates(l_r, u_r, breakpoints=x_nc)
    kappa_hat, _ = best_kappa(cand_d, costs_step(x_nc, F_nc, c_p_r, c_f_r, l_r, cand_d))
    m = np.isfinite(cv)
    ax_ex.plot(x_cost_plot[m], cv[m], color=rel_colors['discretized_ifr'], linewidth=1.5, label=rel_labels['discretized_ifr'])
    ax_ex.axvline(kappa_hat, color=rel_colors['discretized_ifr'], linestyle=':', linewidth=1.5, alpha=0.7)
except Exception as e:
    print(f"discretized_ifr failed: {e}")

try:
    x_ni, F_ni = fit_cdf_nonconvex(x_knots_r, n_trials_r, y_samples_ex, l_r, u_r, 0,
                                    ifr_constraint=False, verbose=False, return_type='values')
    cv       = costs_step(x_ni, F_ni, c_p_r, c_f_r, l_r, x_cost_plot)
    cand_n   = kappa_candidates(l_r, u_r, breakpoints=x_ni)
    kappa_hat, _ = best_kappa(cand_n, costs_step(x_ni, F_ni, c_p_r, c_f_r, l_r, cand_n))
    m = np.isfinite(cv)
    ax_ex.plot(x_cost_plot[m], cv[m], color=rel_colors['discretized_non_ifr'], linewidth=1.5, label=rel_labels['discretized_non_ifr'])
    ax_ex.axvline(kappa_hat, color=rel_colors['discretized_non_ifr'], linestyle=':', linewidth=1.5, alpha=0.7)
except Exception as e:
    print(f"discretized_non_ifr failed: {e}")

ax_ex.set_xlabel(r'Replacement interval $\kappa$ (months)')
ax_ex.set_ylabel(r'$C(\kappa;\,\hat{F})$')
ax_ex.set_xlim(l_r + 0.05, u_r)
ax_ex.set_ylim(bottom=0, top=175)
ax_ex.grid(True, linestyle='--', alpha=0.4)

# One shared legend floating beneath all three panels
handles, labels = ax_ex.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
           ncol=len(labels), frameon=True)
plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig('results/plots/reliability_case_study.png', dpi=150, bbox_inches='tight')
plt.show()

#%%

##############################################################################
##############################################################################
################    RELIABILITY WEIBULL EXPERIMENT    ########################
##############################################################################
##############################################################################

#%%

np.random.seed(0)

# ── Part 2 parameters ─────────────────────────────────────────────────────────────
l_w, u_w         = 0.0, 100.0
x_knots_w        = np.array([24.0, 30.0, 36.0])
n_per_knot_list  = [30,40, 50]
num_runs_w       = 150
c_p_w, c_f_w     = 600, 1800
x_fine_w         = np.unique(np.concatenate((np.linspace(l_w + 1e-4, u_w, 3000), x_knots_w, [l_w, u_w])))

class TwoPhaseHazardDist:
    """Two-phase (constant, then linearly increasing) hazard, truncated to [l, u].

    h(x) = h0 + s * max(x - x0, 0): a constant random-failure rate h0 until the
    wear-out onset age x0, then a hazard rising linearly at rate s. The law is
    IFR with a continuously differentiable cdf, but its hazard has a kink that a
    Weibull power-law hazard cannot represent, so a Weibull fitted through the
    three knots is decision-biased even with unlimited data (the Gamma, Normal
    and Logistic families are not: their log-log cdf plots are close to linear
    across three nearby knots, so a Weibull through the knots reproduces the cdf
    where the replacement decision lives).

    Cumulative hazard H(x) = h0 x + s max(x - x0, 0)^2 / 2; the untruncated cdf
    1 - exp(-H) is renormalised to [l, u] exactly as TruncatedDist does.
    """
    def __init__(self, h0, x0, s, l, u):
        self.h0, self.x0, self.s = float(h0), float(x0), float(s)
        self.l, self.u = l, u
        self._Fl = self._raw_cdf(np.array([l]))[0]
        self._Fu = self._raw_cdf(np.array([u]))[0]

    def _H(self, x):
        return self.h0 * x + 0.5 * self.s * np.maximum(x - self.x0, 0.0) ** 2

    def _raw_cdf(self, x):
        return 1.0 - np.exp(-self._H(x))

    def hazard(self, x):
        x = np.asarray(x, dtype=float)
        return self.h0 + self.s * np.maximum(x - self.x0, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip((self._raw_cdf(x) - self._Fl) / (self._Fu - self._Fl), 0.0, 1.0)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(
            (x >= self.l) & (x <= self.u),
            self.hazard(x) * np.exp(-self._H(x)) / (self._Fu - self._Fl),
            0.0,
        )


# Experiment 1: Weibull(shape=5, scale=30) truncated to [0,100] — correct spec
#   mode = 30*(4/5)^(1/5) ≈ 28.7 months; F(100) ≈ 1 so truncation is negligible
# Experiment 2: two-phase hazard h(x) = 0.005 + 0.02 (x - 28)^+ on [0,100] —
#   misspecification. Random failures at rate 0.005/month until wear-out begins
#   at 28 months; F0(knots) ≈ 0.11, 0.17, 0.56; kappa* ≈ 28.9 (between the knots).
#   Infinite-data decision cost ratio of the Weibull probability-plot fit ≈ 0.93
#   versus ≈ 0.99 for Algorithm 1; the Weibull fit fails (non-increasing
#   proportions) in < 0.4% of runs at n_i = 15.
EXP_DISTS = {
    'Weibull':  TruncatedDist(weibull_scipy(5, scale=30.0), l_w, u_w),
    'TwoPhase': TwoPhaseHazardDist(h0=0.005, x0=28.0, s=0.02, l=l_w, u=u_w),
}

results_exp = {name: {} for name in EXP_DISTS}
#%%
for exp_name, exp_dist in EXP_DISTS.items():
    true_cdf_fn_w = lambda t, _d=exp_dist: float(np.clip(_d.cdf(np.array([t])), 0, 1).flat[0])
    true_p_w      = exp_dist.cdf(x_knots_w)
    true_fine_w   = exp_dist.cdf(x_fine_w)

    cand_true_w     = kappa_candidates(l_w, u_w)
    kappa_star_w, _ = best_kappa(cand_true_w, costs_continuous(true_cdf_fn_w, c_p_w, c_f_w, l_w, cand_true_w))
    cost_star_w     = cost_continuous_at(true_cdf_fn_w, c_p_w, c_f_w, l_w, kappa_star_w)
    print(f"{exp_name}: kappa* = {kappa_star_w:.2f} months, C(kappa*) = {cost_star_w:.4f}")

    for n in n_per_knot_list:
        n_trials_w = n * np.ones(len(x_knots_w), dtype=int)
        ratios_alg, ratios_weib = [], []
        errors_alg, errors_weib = [], []
        n_fail_alg, n_fail_weib = 0, 0

        for _ in tqdm(range(num_runs_w), desc=f"{exp_name} n={n}", leave=False):
            y_samples = np.random.binomial(n_trials_w, true_p_w)

            # Algorithm 1 (PWL)
            try:
                fitted_alg   = fit_cdf_ifr(x_knots_w, n_trials_w, y_samples, l_w, u_w, 'linear')
                fn_raw_alg   = sp.lambdify(x_sym, fitted_alg, 'numpy')
                with np.errstate(invalid='ignore', over='ignore'):
                    vals_alg = np.asarray(fn_raw_alg(x_fine_w), dtype=float)
                # Detect a numerical failure rather than masking a nonfinite value to 1
                if not np.all(np.isfinite(vals_alg)):
                    raise RuntimeError("Nonfinite fitted-cdf evaluation.")
                vals_alg = np.clip(vals_alg, 0.0, 1.0)
                errors_alg.append(float(np.max(np.abs(vals_alg - true_fine_w))))
                def fn_alg(t, _f=fn_raw_alg):
                    with np.errstate(invalid='ignore', over='ignore'):
                        return float(np.clip(np.asarray(_f(t), dtype=float).flat[0], 0.0, 1.0))
                cand_alg_w   = kappa_candidates(l_w, u_w, breakpoints=x_knots_w)
                kappa_alg, _ = best_kappa(cand_alg_w, costs_continuous(fn_alg, c_p_w, c_f_w, l_w, cand_alg_w))
                ratios_alg.append(true_cost_ratio(true_cdf_fn_w, c_p_w, c_f_w, l_w, cost_star_w, kappa_alg))
            except Exception as e:
                n_fail_alg += 1
                # The error may already have been appended if the failure occurred
                # in the decision step; keep the two lists aligned run by run.
                if len(errors_alg) < len(ratios_alg) + 1:
                    errors_alg.append(np.nan)
                ratios_alg.append(np.nan)

            # Weibull Regression. fit_weibull_cdf raises on a degenerate probability
            # plot (non-positive log-log slope, i.e. empirical proportions that do
            # not increase in x); such runs are counted and dropped.
            try:
                fitted_weib   = fit_weibull_cdf(x_knots_w, n_trials_w, y_samples, l_w, u_w, x_sym)
                fn_raw_weib   = sp.lambdify(x_sym, fitted_weib, 'numpy')
                def fn_weib(t, _f=fn_raw_weib):
                    with np.errstate(invalid='ignore', over='ignore'):
                        return float(np.clip(np.asarray(_f(t), dtype=float).flat[0], 0, 1))
                vals_weib = np.array([fn_weib(t) for t in x_fine_w])
                errors_weib.append(float(np.max(np.abs(vals_weib - true_fine_w))))
                cand_weib_w   = kappa_candidates(l_w, u_w)
                kappa_weib, _ = best_kappa(cand_weib_w, costs_continuous(fn_weib, c_p_w, c_f_w, l_w, cand_weib_w))
                ratios_weib.append(true_cost_ratio(true_cdf_fn_w, c_p_w, c_f_w, l_w, cost_star_w, kappa_weib))
            except Exception as e:
                n_fail_weib += 1
                if len(errors_weib) < len(ratios_weib) + 1:   # as above
                    errors_weib.append(np.nan)
                ratios_weib.append(np.nan)

        print(f"  {exp_name}, n_i={n}: failed runs  Algorithm 1 = {n_fail_alg}/{num_runs_w}, "
              f"Weibull regression = {n_fail_weib}/{num_runs_w}")

        results_exp[exp_name][n] = {
            'algo1':          np.array(ratios_alg),
            'weibull':        np.array(ratios_weib),
            'errors_algo1':   np.array(errors_alg),
            'errors_weibull': np.array(errors_weib),
            'n_failed_algo1':   n_fail_alg,
            'n_failed_weibull': n_fail_weib,
        }

np.savez('results/data/reliability_case_study_weibull.npz',
    exp_names=np.array(list(EXP_DISTS.keys())),
    n_values=np.array(n_per_knot_list),
    **{f'{en}_{n}_{k}': results_exp[en][n][k]
       for en in EXP_DISTS for n in n_per_knot_list
       for k in ['algo1', 'weibull', 'errors_algo1', 'errors_weibull',
                 'n_failed_algo1', 'n_failed_weibull']},
)
print('Saved results/data/reliability_case_study_weibull.npz')

#%%

# Ground-truth figure for Part 2. Reuses EXP_DISTS from the experiment cell; only
# the plotting labels and colours are defined here.
x_plot_w = np.linspace(0.0, 105.0, 1000)
x_cost_w = np.linspace(0.5, u_w, 500)

dist_cfg = {
    'Weibull': {
        'dist':  EXP_DISTS['Weibull'],
        'label': r'Weibull(shape$=5$, scale$=30$)  [mode $\approx 28.7$, mean $\approx 27.5$]',
        'color': 'steelblue'
    },
    'TwoPhase': {
        'dist':  EXP_DISTS['TwoPhase'],
        'label': r'Two-phase hazard $h(x) = 0.005 + 0.02\,(x-28)^+$  [$\kappa^* \approx 28.9$]',
        'color': 'darkorange'
    },
}

fig, (ax_pdf, ax_cdf, ax_cost) = plt.subplots(1, 3, figsize=(21, 5))

for name, cfg in dist_cfg.items():
    d     = cfg['dist']
    color = cfg['color']
    label = cfg['label']

    ax_pdf.plot(x_plot_w, d.pdf(x_plot_w), color=color, linewidth=2, label=label)
    ax_cdf.plot(x_plot_w, d.cdf(x_plot_w), color=color, linewidth=2, label=label)

    # Cost curve C(kappa; F_0) using the true (truncated) CDF
    cdf_scalar = lambda t, _d=d: float(np.clip(_d.cdf(np.array([t])), 0, 1).flat[0])
    cand_plot_w        = kappa_candidates(l_w, u_w)
    kappa_star_plot, _ = best_kappa(cand_plot_w, costs_continuous(cdf_scalar, c_p_w, c_f_w, l_w, cand_plot_w))
    cv = costs_continuous(cdf_scalar, c_p_w, c_f_w, l_w, x_cost_w)
    m  = np.isfinite(cv)
    ax_cost.plot(x_cost_w[m], cv[m], color=color, linewidth=2, label=label)
    ax_cost.axvline(kappa_star_plot, color=color, linewidth=1.2, linestyle=':', alpha=0.85)
    ax_cost.scatter([kappa_star_plot], [cost_continuous_at(cdf_scalar, c_p_w, c_f_w, l_w, kappa_star_plot)],
                    color=color, zorder=5, s=60)

# Mark knots on all panels
for ax in (ax_pdf, ax_cdf, ax_cost):
    for xk in x_knots_w:
        ax.axvline(xk, color='gray', linewidth=0.9, linestyle='--', alpha=0.6)
    ax.grid(True, ls='--', alpha=0.4)

ax_pdf.set_xlabel('Failure time (months)')
ax_pdf.set_ylabel('PDF')
ax_pdf.set_ylim(bottom=0)
ax_pdf.set_xlim(0, 105)

ax_cdf.set_xlabel('Failure time (months)')
ax_cdf.set_ylabel('CDF')
ax_cdf.set_ylim(0, 1.02)
ax_cdf.set_xlim(0, 105)
for name, cfg in dist_cfg.items():
    for xk in x_knots_w:
        ax_cdf.scatter([xk], [cfg['dist'].cdf(xk)], color=cfg['color'],
                       zorder=5, s=60, marker='x', linewidths=2)

ax_cost.set_xlabel(r'Replacement interval $\kappa$ (months)')
ax_cost.set_ylabel(r'$C(\kappa;\, F_0)$')
ax_cost.set_xlim(0, u_w)
ax_cost.set_ylim(bottom=0, top = 100)

handles, labels_leg = ax_pdf.get_legend_handles_labels()
fig.legend(handles, labels_leg, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))
plt.suptitle(
    f'Part 2 ground-truth distributions  (dashed = knots {x_knots_w.tolist()}, dotted = mode / $\\kappa^*$)',
    fontsize=13,
)
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig('results/plots/reliability_part2_dists.png', dpi=150, bbox_inches='tight')
plt.show()

#%%

data_w  = np.load('results/data/reliability_case_study_weibull.npz', allow_pickle=True)
exp_names_w  = data_w['exp_names'].tolist()
n_values_w   = data_w['n_values'].tolist()

weib_methods = ['weibull', 'algo1']
weib_labels  = {'weibull': 'Weibull Regression', 'algo1': 'Algorithm 1'}
weib_colors  = {'weibull': 'steelblue', 'algo1': 'green'}

for exp_name in exp_names_w:
    # Dropped (failed) runs per sample size, as recorded by the experiment cell.
    for n in n_values_w:
        print(f"{exp_name}, n_i={n}: dropped runs  Algorithm 1 = {int(data_w[f'{exp_name}_{n}_n_failed_algo1'])}, "
              f"Weibull regression = {int(data_w[f'{exp_name}_{n}_n_failed_weibull'])}")

    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(n_values_w))

    # Shape (n_sample_sizes, n_methods, num_runs) for the shared bar helper.
    ratios = np.stack([
        np.stack([data_w[f'{exp_name}_{n}_{m}'] for m in weib_methods])
        for n in n_values_w
    ])
    grouped_mean_sd_bars(ax, ratios, weib_methods, weib_labels, weib_colors, x_pos, width=0.38)

    ax.axhline(1.0, color='black',  linewidth=1, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'$n_i = {n}$' for n in n_values_w])
    ax.set_ylabel('Cost Ratio')
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, axis='y', ls='--')

    # Legend floating below the figure, single column
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=True)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(f'results/plots/reliability_fig_weibull_{exp_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()

#%%

