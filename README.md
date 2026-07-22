# Fitting-IFR-CDFs-from-Noisy-Quantile-Data

Code and data to reproduce the algorithms, numerical experiments and case studies in:

> *Estimating Distributions with Failure Rate Properties from Noisy Quantile Data (2026)*

> Timothy C. Y. Chan (MIE, UToronto)  
> Ningyuan Chen (Rotman, UToronto)  
> Craig Fernandes (Wharton, UPenn)  
> Muhammad Maaz (MIE, UToronto)   

**Setting.** Given noisy binomial observations at a handful of knots $x_1 < \dots < x_k$ (i.e., $y_i$ successes out of $n_i$ trials at each $x_i$), the paper fits a CDF $\hat F$ that is guaranteed to have increasing failure rate (IFR). The key idea is a change of variables $t(x) = \ln(1 - F(x))$, under which the IFR constraint becomes concavity of $t$. This turns a non-convex, infinite-dimensional MLE into a finite-dimensional convex program (**Algorithm 1**), solved with CVXPY and interpolated back to a continuous CDF via a Schumaker spline.

## Repository contents

```
.
├── fitting.py    # Algorithm 1 (fit_cdf) and the discretized-grid benchmarks (fit_cdf_nonconvex, fit_weibull_cdf)
├── utils.py      # Interpolation routines and shared numerical helpers
├── example.ipynb # Minimal worked example: fit all three methods to one simulated dataset
├── numerics.ipynb # Section 6 numerical experiments (convergence rates, knot allocation, method comparison)
├── cases.ipynb   # Section 7 case studies (monopolistic pricing, preventive maintenance)
└── results/
    ├── data/     # .npz files with raw Monte Carlo output (pre-computed, so figures reproduce without rerunning simulations)
    └── plots/    # .png figures and .tex tables, as they appear in the paper
```

## Setup

```bash
pip install numpy scipy sympy cvxpy mpmath matplotlib tqdm gurobipy
```

`fitting.py`'s `fit_cdf` (**Algorithm 1**) only needs CVXPY's default open-source solver — no license required. The **discretized-IFR** and **discretized-non-IFR** benchmarks used throughout for comparison (`fit_cdf_nonconvex`) solve a non-convex program with **Gurobi**, which requires a license (free for academic use at [gurobi.com](https://www.gurobi.com)). If you don't have a license, you can still run every cell that only loads pre-computed results from `results/data/`.

A couple of cells call `winsound.Beep(...)` to signal when a long-running simulation finishes. This is Windows-only — comment it out (or wrap it in a `try/except`) on macOS/Linux.

## Notebooks

- **`example.ipynb`** — Fits a single simulated Beta(3, 5) dataset with Algorithm 1, discretized-IFR (multiple grid densities), and discretized-non-IFR, and plots all three CDF estimates against the ground truth and the noisy observations. Start here if you just want to see the method run end-to-end.

- **`numerics.ipynb`** — Reproduces Section 6 of the paper:
  - **Empirical convergence rates** — estimation error vs. $n_i$ and interpolation error vs. $k$, matching the $O(1/\sqrt{n_i})$ and $O(1/k^p)$ rates from Propositions 1–2.
  - **Optimal knot allocation** — how error trades off between more knots vs. more observations per knot, at a fixed total sampling budget.
  - **Method comparison** — Algorithm 1 vs. the discretized-IFR/non-IFR benchmarks, on accuracy and solve time (Table 1), plus an appendix comparison across several additional ground-truth distributions.

- **`cases.ipynb`** — Reproduces Section 7:
  - **Monopolistic pricing** — a firm estimates the willingness-to-pay distribution $F_0$ from limited price experimentation and sets the revenue-maximizing price $\hat p = \arg\max_p\, p(1-\hat F(p))$; performance is measured by the revenue ratio relative to the true optimum.
  - **Preventive maintenance** — a component is replaced at age $\kappa$ or at failure, whichever comes first; $\hat F$ is used to choose the cost-minimizing replacement age $\hat\kappa$, evaluated by the realized cost ratio. Part 1 compares Algorithm 1 against the discretized benchmarks across four distributions; Part 2 reproduces Figure 8, comparing Algorithm 1 against a parametric Weibull-regression benchmark under correct specification and misspecification.

Each notebook's computation cells are separated from its plotting cells, so figures can be regenerated from `results/data/*.npz` without rerunning the (often slow, Gurobi-dependent) Monte Carlo simulations.

## Questions

Contact the corresponding author at craigfer@wharton.upenn.edu
