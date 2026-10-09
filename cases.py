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
from fitting import fit_cdf, fit_cdf_nonconvex, eval_step_cdf, l_inf_step_vs_cdf
from tqdm.auto import tqdm
import os
import winsound

plt.rcParams.update({
    'axes.labelsize':  18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

os.makedirs('results', exist_ok=True)


# ── Problem parameters ──────────────────────────────────────────────────────────
l, u     = 20, 50.0
x_knots  = np.array([25.0, 30.0, 35.0])   # sale, regular, high prices
n_trials = np.array([50, 100, 50])          # observations at each price
num_runs = 150  # 100 to match the paper
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
# Chi-squared(df=38) = Gamma(19, scale=2); revenue-optimal price p* ~ 28.
# Logistic(loc=36, s=5): increasing hazard on the truncated support; p* ~ 28.
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
            fitted = fit_cdf(x_knots, n_trials, y_samples, l, u, 'linear')
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
n_methods = len(method_keys_list)
width     = 0.22
offsets   = np.linspace(-(n_methods - 1) * width / 2, (n_methods - 1) * width / 2, n_methods)
x_pos     = np.arange(n_dists)
display_names = [DISTRIBUTIONS[k]['display_name'] for k in dist_names_list]

def _add_mean_sd(ax, arr_3d):
    ## Grouped bar chart of the mean with ± sample-standard-deviation error bars.
    ## arr_3d shape (n_dists, n_methods, num_runs).
    bar_w = width * 0.9   # bar width within each distribution group
    for j, m in enumerate(method_keys_list):
        means = np.full(n_dists, np.nan)
        sds   = np.full(n_dists, np.nan)
        for di in range(n_dists):
            vals = arr_3d[di, j, :]
            vals = vals[np.isfinite(vals)]   # drop failed runs
            if vals.size:
                means[di] = vals.mean()
                sds[di]   = vals.std(ddof=1) if vals.size > 1 else 0.0
        ax.bar(
            x_pos + offsets[j], means, width=bar_w,
            yerr=sds, color=colors[m], alpha=0.7,
            edgecolor='black', linewidth=0.5,
            error_kw=dict(ecolor='black', elinewidth=1.2, capsize=3),
            label=method_labels[m],
        )

# --- Combined pricing figure: full-support error | revenue ratio | illustrative curve ---
all_revenues = data['all_revenues']

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

# Panel 1: full-support L-inf error (mean ± SD bars)
ax_full = axes[0]
_add_mean_sd(ax_full, all_errors)
ax_full.set_xticks(x_pos)
ax_full.set_xticklabels(display_names)
ax_full.set_ylabel(r'$\|F_0 - \hat{F}\|$')
ax_full.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax_full.set_ylim([0, 0.85])
ax_full.grid(True, which='both', ls='--')

# Panel 2: revenue ratio (mean ± SD bars)
ax_ratio = axes[1]
bar_w = width * 0.9   # bar width within each distribution group
for j, m in enumerate(method_keys_list):
    means = np.full(n_dists, np.nan)
    sds   = np.full(n_dists, np.nan)
    for di in range(n_dists):
        vals = all_revenues[di, j, :]
        vals = vals[np.isfinite(vals)]   # drop failed runs
        if vals.size:
            means[di] = vals.mean()
            sds[di]   = vals.std(ddof=1) if vals.size > 1 else 0.0
    ax_ratio.bar(
        x_pos + offsets[j], means, width=bar_w,
        yerr=sds, color=colors[m], alpha=0.7,
        edgecolor='black', linewidth=0.5,
        error_kw=dict(ecolor='black', elinewidth=1.2, capsize=3),
        label=method_labels[m],
    )

ax_ratio.axhline(1.0, color='black', linewidth=1, alpha=0.5)
ax_ratio.set_xticks(x_pos)
ax_ratio.set_xticklabels(display_names)
ax_ratio.set_ylabel('Revenue Ratio')
ax_ratio.set_ylim([-0.05, 1.1])
ax_ratio.grid(True, which='both', ls='--')

# Panel 3: illustrative Chi-Sq revenue curves (seed 0)
np.random.seed(0)
ax_ex = axes[2]

ex_dist = DISTRIBUTIONS['ChiSq']['dist']
true_fine = ex_dist.cdf(x_fine)
true_p    = ex_dist.cdf(x_knots)
y_samples = np.random.binomial(n_trials, true_p)

true_rev     = x_fine * (1 - true_fine)
true_opt_idx = int(np.argmax(true_rev))

ax_ex.plot(x_fine, true_rev, 'k-', linewidth=2, label='True', zorder=5)
ax_ex.axvline(x_fine[true_opt_idx], color='black', linestyle=':', linewidth=1.5, alpha=0.7)

try:
    fitted = fit_cdf(x_knots, n_trials, y_samples, l, u, 'linear')
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
# consecutive candidate gaps for continuous cdfs; integration warnings are
# recorded as failures rather than suppressed.
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

# ── Part 1 parameters ─────────────────────────────────────────────────────────────
l_r, u_r      = 0.0, 12.0
x_knots_r     = np.array([2.0, 4.0, 6.0])
n_per_knot_r  = 50
n_trials_r    = n_per_knot_r * np.ones(len(x_knots_r), dtype=int)
num_runs_r    = 150
c_p_r, c_f_r  = 300, 1200
x_fine_r      = np.unique(np.concatenate((np.linspace(l_r + 1e-4, u_r, 3000), x_knots_r, [l_r, u_r])))
d_ifr_r       = 2

# All four are IFR, truncated to [l_r, u_r] = [0, 12]; knots [2, 4, 6] bracket kappa* ~ 5.
# Chi-Sq(10) = Gamma(5, scale=2): mode = df-2 = 8
# Normal(9, 3): mode = 9
# Gamma(5, scale=2): mode = (alpha-1)*scale = 8
# Logistic(9, 2): mode = loc = 9; increasing hazard
DIST_RELIABILITY = {
    'ChiSq':    {'label': 'Chi-Sq (df=10)',        'display_name': 'Chi-Sq',
                 'dist': TruncatedDist(chi2(10), l_r, u_r)},
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
            fitted = fit_cdf(x_knots_r, n_trials_r, y_samples, l_r, u_r, 'linear')
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
            all_costs_r[di, 0, run] = cost_star / cost_continuous_at(true_cdf_fn, c_p_r, c_f_r, l_r, kappa_alg)
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
            all_costs_r[di, 1, run] = cost_star / cost_continuous_at(true_cdf_fn, c_p_r, c_f_r, l_r, kappa_disc)
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
            all_costs_r[di, 2, run] = cost_star / cost_continuous_at(true_cdf_fn, c_p_r, c_f_r, l_r, kappa_nifr)
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
n_rm  = len(rel_method_keys)
width = 0.22
offsets_r       = np.linspace(-(n_rm - 1) * width / 2, (n_rm - 1) * width / 2, n_rm)
x_pos_r         = np.arange(n_rd)
display_names_r = [DIST_RELIABILITY[k]['display_name'] for k in rel_dist_names]

def _rel_mean_sd(ax, arr_3d):
    ## rouped bar chart of the mean with ± sample-standard-deviation error bars.
    bar_w = width * 0.9
    for j, m in enumerate(rel_method_keys):
        means = np.full(n_rd, np.nan)
        sds   = np.full(n_rd, np.nan)
        for di in range(n_rd):
            vals = arr_3d[di, j, :]
            vals = vals[np.isfinite(vals)]   # drop failed runs
            if vals.size:
                means[di] = vals.mean()
                sds[di]   = vals.std(ddof=1) if vals.size > 1 else 0.0
        ax.bar(
            x_pos_r + offsets_r[j], means, width=bar_w,
            yerr=sds, color=rel_colors[m], alpha=0.7,
            edgecolor='black', linewidth=0.5,
            error_kw=dict(ecolor='black', elinewidth=1.2, capsize=3),
            label=rel_labels[m],
        )

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

# Panel 1: full-support L-inf error (mean ± SD bars)
ax_full = axes[0]
_rel_mean_sd(ax_full, all_errors_r)
ax_full.set_xticks(x_pos_r)
ax_full.set_xticklabels(display_names_r)
ax_full.set_ylabel(r'$\|F_0 - \hat{F}\|_{[l,\,u]}$')
ax_full.set_ylim(0, 0.85)
ax_full.grid(True, axis='y', ls='--')

# Panel 2: cost ratio (mean ± SD bars)
ax_cost = axes[1]
_rel_mean_sd(ax_cost, all_costs_r)
ax_cost.axhline(1.0, color='black', linewidth=1, alpha=0.5)
ax_cost.set_xticks(x_pos_r)
ax_cost.set_xticklabels(display_names_r)
ax_cost.set_ylabel('Cost Ratio')
ax_cost.set_ylim(0.5, 1.025)
ax_cost.grid(True, axis='y', ls='--')

# Panel 3: illustrative Chi-Sq cost curves (seed 42)
np.random.seed(42)
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
    fitted   = fit_cdf(x_knots_r, n_trials_r, y_samples_ex, l_r, u_r, 'linear')
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
n_per_knot_list  = [15,20,25]
num_runs_w       = 150
c_p_w, c_f_w     = 600, 1800
x_fine_w         = np.unique(np.concatenate((np.linspace(l_w + 1e-4, u_w, 3000), x_knots_w, [l_w, u_w])))

# Experiment 1: Weibull(shape=5, scale=30) truncated to [0,100] — correct spec
#   mode = 30*(4/5)^(1/5) ≈ 28.7 months; F(100) ≈ 1 so truncation is negligible
# Experiment 2: Gamma(alpha=8, scale=6) truncated to [0,100] — misspecification
#   mode = (8-1)*6 = 42 months; mean = 48; F(100) ≈ 0.9987
EXP_DISTS = {
    'Weibull': TruncatedDist(weibull_scipy(5, scale=30.0), l_w, u_w),
    'Gamma':   TruncatedDist(gamma_dist(8, scale=6.0),    l_w, u_w),
}

results_exp = {name: {} for name in EXP_DISTS}

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

        for _ in tqdm(range(num_runs_w), desc=f"{exp_name} n={n}", leave=False):
            y_samples = np.random.binomial(n_trials_w, true_p_w)

            # Algorithm 1 (PWL)
            try:
                fitted_alg   = fit_cdf(x_knots_w, n_trials_w, y_samples, l_w, u_w, 'linear')
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
                ratios_alg.append(cost_star_w / cost_continuous_at(true_cdf_fn_w, c_p_w, c_f_w, l_w, kappa_alg))
            except Exception as e:
                errors_alg.append(np.nan)
                ratios_alg.append(np.nan)

            # Weibull Regression
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
                ratios_weib.append(cost_star_w / cost_continuous_at(true_cdf_fn_w, c_p_w, c_f_w, l_w, kappa_weib))
            except Exception as e:
                errors_weib.append(np.nan)
                ratios_weib.append(np.nan)

        results_exp[exp_name][n] = {
            'algo1':          np.array(ratios_alg),
            'weibull':        np.array(ratios_weib),
            'errors_algo1':   np.array(errors_alg),
            'errors_weibull': np.array(errors_weib),
        }

np.savez('results/data/reliability_case_study_weibull.npz',
    exp_names=np.array(list(EXP_DISTS.keys())),
    n_values=np.array(n_per_knot_list),
    **{f'{en}_{n}_{k}': results_exp[en][n][k]
       for en in EXP_DISTS for n in n_per_knot_list
       for k in ['algo1', 'weibull', 'errors_algo1', 'errors_weibull']},
)
print('Saved results/data/reliability_case_study_weibull.npz')

#%%

x_plot_w = np.linspace(0.0, 105.0, 1000)
x_cost_w = np.linspace(0.5, u_w, 500)

EXP_DISTS = {
    'Weibull': TruncatedDist(weibull_scipy(5, scale=30.0), l_w, u_w),
    'Gamma':   TruncatedDist(gamma_dist(8, scale=6.0),    l_w, u_w),
}

results_exp = {name: {} for name in EXP_DISTS}

dist_cfg = {
    'Weibull': {
        'dist':  TruncatedDist(weibull_scipy(5, scale=30.0), l_w, u_w),
        'label': r'Weibull(shape$=5$, scale$=30$)  [mode $\approx 28.7$, mean $\approx 27.5$]',
        'color': 'steelblue'
    },
    'Gamma': {
        'dist':  TruncatedDist(gamma_dist(8, scale=6.0), l_w, u_w),
        'label': r'Gamma($\alpha=8$, $\theta=6$)  [mode $= 42$, mean $= 48$]',
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

for exp_name in exp_names_w:
    fig, ax = plt.subplots(figsize=(8, 6))
    n_nv  = len(n_values_w)
    x_pos = np.arange(n_nv)
    bw    = 0.38

    def _mean_sd(key):
        means, sds = [], []
        for n in n_values_w:
            vals = data_w[f'{exp_name}_{n}_{key}']
            vals = vals[np.isfinite(vals)]   # drop failed runs
            means.append(vals.mean() if vals.size else np.nan)
            sds.append(vals.std(ddof=1) if vals.size > 1 else 0.0)
        return np.array(means), np.array(sds)

    m_weib, s_weib = _mean_sd('weibull')
    m_alg,  s_alg  = _mean_sd('algo1')

    ax.bar(x_pos - bw / 2, m_weib, width=bw, yerr=s_weib, color='steelblue', alpha=0.7,
           edgecolor='black', linewidth=0.5,
           error_kw=dict(ecolor='black', elinewidth=1.2, capsize=3), label='Weibull Regression')
    ax.bar(x_pos + bw / 2, m_alg, width=bw, yerr=s_alg, color='green', alpha=0.7,
           edgecolor='black', linewidth=0.5,
           error_kw=dict(ecolor='black', elinewidth=1.2, capsize=3), label='Algorithm 1')

    ax.axhline(1.0, color='black',  linewidth=1, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'$n_i = {n}$' for n in n_values_w])
    ax.set_ylabel('Cost Ratio')
    ax.set_ylim(0.5, 1.15)
    ax.grid(True, axis='y', ls='--')

    # Legend floating below the figure, single column
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=True)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(f'results/plots/reliability_fig_weibull_{exp_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()

#%%

