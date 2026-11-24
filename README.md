# Fitting-IFR-CDFs-from-Noisy-Quantile-Data

Code and data to reproduce the algorithms, numerical experiments and case studies in:

> *Estimating Distributions with Failure Rate Properties from Noisy Quantile Data (2026)*

> Timothy C. Y. Chan (MIE, UToronto)  
> Ningyuan Chen (Rotman, UToronto)  
> Craig Fernandes (Wharton, UPenn)  
> Muhammad Maaz (MIE, UToronto)   

**Setting.** Given noisy binomial observations at a handful of knots $x_1 < \dots < x_k$ (i.e., $y_i$ successes out of $n_i$ trials at each $x_i$, with $y_i \sim \mathrm{Binomial}(n_i, F_0(x_i))$), the paper fits a cdf $\hat F$ that is guaranteed to have increasing failure rate (IFR). The key idea is a change of variables $t(x) = \ln(1 - F(x))$, under which the IFR constraint becomes concavity of $t$. This turns a non-convex, infinite-dimensional MLE into a finite-dimensional convex program over the knot values (**Algorithm 1**), solved with CVXPY. The fitted knot values are interpolated piecewise linearly in $t$-space and completed with an analytic tail, which gives a continuous IFR cdf that attains the optimal likelihood. The same framework extends to decreasing failure rate (DFR), increasing failure rate average (IFRA), increasing generalized failure rate (IGFR), and new-better-than-used (NBU) distributions.

## Repository contents

```
.
├── fitting.py            # Algorithm 1 (fit_cdf_ifr); extensions (fit_cdf_dfr, fit_cdf_ifra, fit_cdf_igfr, fit_cdf_nbu);
│                         # benchmarks (fit_cdf_nonconvex: discretized-IFR / discretized-non-IFR via Gurobi; fit_weibull_cdf)
├── utils.py              # Symbolic piecewise-linear interpolation used by every estimator, plus legacy numerical helpers
├── example.ipynb         # Minimal worked example: Algorithm 1 on two ground truths, and Algorithm 1 vs. the discretized benchmarks
├── numerics.ipynb        # Section 6 numerical experiments (knot allocation at a fixed budget; benchmark comparison and solve times)
├── cases.ipynb           # Section 7 case studies (monopolistic pricing; preventive maintenance incl. the Weibull benchmark)
├── extensions.ipynb      # Section 5 worked examples: one DFR, IFRA, IGFR and NBU fit each, and the combined 2x2 figure
├── numerics.py           # Script versions of the three notebooks above; each `#%%` block is one notebook code cell
├── cases.py
├── extensions.py
├── presentation_plots.py # Slide figures (not used in the paper); writes to results/presentations/
└── results/
    ├── data/             # .npz files with raw Monte Carlo output (pre-computed, so figures reproduce without rerunning simulations)
    ├── plots/            # .png figures and the .tex solve-time table, as they appear in the paper
    └── presentations/    # slide figures from presentation_plots.py
```

The notebooks and the scripts contain the same code: every code cell of `numerics.ipynb`, `cases.ipynb` and `extensions.ipynb` is a `#%%` block of the corresponding `.py` file. Edit one and copy to the other.

## Setup

```bash
pip install numpy scipy sympy cvxpy mpmath matplotlib tqdm gurobipy
```

`fit_cdf_ifr` (**Algorithm 1**) and the four extension estimators only need CVXPY's default open-source solvers — no license required. The **discretized-IFR** and **discretized-non-IFR** benchmarks (`fit_cdf_nonconvex`) solve a non-convex program with **Gurobi** (version 12 or later, for its nonlinear-constraint support), which requires a license (free for academic use at [gurobi.com](https://www.gurobi.com)). `gurobipy` is imported lazily inside `fit_cdf_nonconvex`, so everything else imports and runs without it, and every plotting cell only loads pre-computed results from `results/data/`.

The figure code sets `matplotlib`'s `text.usetex=True`, which requires a working LaTeX installation on the system path. If you don't have one, set `text.usetex=False` in the `plt.rcParams.update({...})` block at the top of each notebook or script (the figures still render, just with matplotlib's default math fonts).

Solve times for the discretized-IFR benchmark, and the number of instances that hit its 300-second limit, depend on the machine and the Gurobi version; the accuracy results do not.

## Notebooks

- **`example.ipynb`** — Two figures from Section 3 and Section 6.3 of the paper: Algorithm 1 fitted to a Beta(3, 5) and a truncated Normal ground truth under two knot designs, and a single Beta(3, 5) dataset fitted with Algorithm 1, discretized-IFR ($d \in \{2, 6\}$) and discretized-non-IFR. Start here if you just want to see the method run end-to-end.

- **`numerics.ipynb`** — Reproduces Section 6:
  - **Knot allocation at a fixed total budget** — how error trades off between more knots vs. more observations per knot, for Beta(3, b) ground truths with $b \in \{5, 7, 12\}$.
  - **Benchmark comparison** — Algorithm 1 vs. discretized-IFR ($d \in \{2, 4, 6\}$) and discretized-non-IFR on accuracy, plus the solve-time table (Table 1), written as both a PNG and a LaTeX `tabular`.

- **`cases.ipynb`** — Reproduces Section 7:
  - **Monopolistic pricing** — a firm estimates the willingness-to-pay distribution $F_0$ from sale/no-sale observations at three prices and sets the revenue-maximizing price $\hat p = \arg\max_p\, p(1-\hat F(p))$; performance is the revenue ratio relative to the true optimum.
  - **Preventive maintenance** — a component is replaced at age $\kappa$ or at failure, whichever comes first; $\hat F$ is used to choose the cost-minimizing replacement age $\hat\kappa$, evaluated by the realized cost ratio. Part 1 compares Algorithm 1 against the discretized benchmarks across four distributions; Part 2 compares Algorithm 1 against a parametric Weibull-regression benchmark under a correctly specified Weibull truth and a misspecified two-phase-hazard truth.

- **`extensions.ipynb`** — Reproduces the Section 5 visualization: one worked example per failure-rate extension (DFR, IFRA, IGFR, NBU), each fitted to a ground truth that has the property but is not IFR, and the combined figure.

Each simulation cell saves its raw per-run results to `results/data/*.npz`, and each plotting cell reads them back, so figures can be regenerated without rerunning the (often slow, Gurobi-dependent) Monte Carlo simulations.

## Questions

Contact the corresponding author at craigfer@wharton.upenn.edu
