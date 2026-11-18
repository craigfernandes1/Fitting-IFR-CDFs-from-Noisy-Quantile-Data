import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTDIR = os.path.join("results", "presentations")
os.makedirs(OUTDIR, exist_ok=True)

x = np.linspace(0, 10, 500)
cdf = 1 / (1 + np.exp(-1.35 * (x - 5.5)))

# Three observation locations, spaced evenly across the range
x1, x2, x3 = 3.5, 5.5, 7.5
xs = [x1, x2, x3]

# True CDF values at x1, x2, x3
ys_true = [1 / (1 + np.exp(-1.35 * (xi - 5.5))) for xi in xs]
# Observed (noisy) quantile values, offset off the curve
offsets = [0.08, -0.06, -0.05]
ys = [yt + off for yt, off in zip(ys_true, offsets)]


def make_figure(stage, filename=None, show_fit=False):
    """Build the CDF figure as a staged reveal.
    stage 0: just the axes and the black curve.
    stage 1: also the x_i tick labels.
    stage 2: also the red markers and y/n ticks (full figure).
    show_fit: also overlay the green estimated cdf F_hat.
    If filename is given, save the figure as a PNG."""
    show_xticks = stage >= 1
    show_data = stage >= 2

    fig, ax = plt.subplots(figsize=(12, 7))

    # True CDF curve
    ax.plot(x, cdf, color="black", linewidth=3)

    if show_data:
        # Dashed guide lines (omitted on the final fitted figure)
        if not show_fit:
            ax.vlines(xs, 0, ys, colors="grey", linestyles="--", linewidth=2)
            ax.hlines(ys, 0, xs, colors="grey", linestyles="--", linewidth=2)

        # Observation markers
        ax.scatter(xs, ys, marker="X", s=550, color="black", zorder=5)

    if show_fit:
        # Estimated cdf F_hat = 1 - exp(t_hat)
        ax.plot(x, Fhat, color=GREEN, linewidth=3, zorder=4)
        # Green fitted knot values F_hat(x_i) = 1 - exp(tau_hat_i), incl. (l, 0)
        Fhat_knots = 1 - np.exp(knot_t)
        ax.scatter(knot_x, Fhat_knots, marker="o", s=300, color=GREEN, zorder=6)

    # Legend (bottom right): true cdf F_0, estimate F_hat, and the data markers
    _handles = [Line2D([0], [0], color="black", lw=3, label=r"$F_0$")]
    if show_fit:
        _handles.append(Line2D([0], [0], color=GREEN, lw=3, label=r"$\hat{F}$"))
        _handles.append(Line2D([0], [0], color=GREEN, marker="o",
                               linestyle="None", markersize=14,
                               label=r"$\hat{F}(x_i)$"))
    if show_data:
        _handles.append(Line2D([0], [0], color="black", marker="X",
                               linestyle="None", markersize=16, label="Data"))
    ax.legend(handles=_handles, fontsize=26, loc="lower right",
              frameon=False, bbox_to_anchor=(0.98, 0.06))

    # Limits
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.03, 1.12)

    # X-axis ticks
    if show_xticks:
        ax.set_xticks([0] + xs + [10])
        ax.set_xticklabels(
            [r"$l$", r"$x_1$", r"$x_2$", r"$x_3$", r"$u$"], fontsize=28)
    else:
        ax.set_xticks([0, 10])
        ax.set_xticklabels([r"$l$", r"$u$"], fontsize=28)

    # Y-axis ticks (no 0 tick; none at all on the final fitted figure)
    if show_fit:
        ax.set_yticks([])
    elif show_data:
        ax.set_yticks(ys + [1])
        ax.set_yticklabels([
            r"$\frac{y_1}{n_1}$",
            r"$\frac{y_2}{n_2}$",
            r"$\frac{y_3}{n_3}$",
            r"$1$"
        ], fontsize=25)
    else:
        ax.set_yticks([1])
        ax.set_yticklabels([r"$1$"], fontsize=25)

    # Axis labels. Pin both label positions in axes coordinates so they
    # do NOT move between figures (matplotlib's default places them off the
    # tick labels, whose widths differ between the figures). The x label
    # is dropped a little lower to leave extra white space below it.
    ax.set_xlabel(r"$x$", fontsize=28)
    ax.set_ylabel(r"$F(x)$", fontsize=28, rotation=0)
    ax.xaxis.set_label_coords(0.5, -0.11)
    ax.yaxis.set_label_coords(-0.10, 0.5)

    # Arrowed axes
    ax.annotate("", xy=(10.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=3, color="black"))
    ax.annotate("", xy=(0, 1.12), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=3, color="black"))

    # Styling
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=0, width=2, pad=12)
    ax.tick_params(axis="y", length=0, width=2, pad=12)

    # Pin the axes to a fixed rectangle (identical in every figure) so the
    # plot does NOT shift when flipping between them. Do not use
    # tight_layout here: it re-sizes the axes from each figure's own tick
    # labels, so the wider y/n fractions would move the box.
    ax.set_position([0.14, 0.20, 0.80, 0.75])

    if filename is not None:
        fig.savefig(filename, dpi=300)

    plt.show()

    return fig, ax


# Figure 1: just the axes and the black curve
make_figure(stage=0,
            filename=os.path.join(OUTDIR, "presentation_cdf_curve.png"))

# Figure 2: add the x_i tick labels
make_figure(stage=1,
            filename=os.path.join(OUTDIR, "presentation_cdf_xticks.png"))

# Figure 3: full info (markers and y/n ticks)
make_figure(stage=2,
            filename=os.path.join(OUTDIR, "presentation_cdf_full.png"))

#%%

# ---------------------------------------------------------------------------
# t-space figures: the change of variables t(x) = ln(1 - F(x)).
# Under IFR, t is non-increasing and concave, with t(l) = 0 and t -> -inf as
# x -> u. The noisy quantile y_i/n_i maps to the transformed value
# ln(1 - y_i/n_i). The curve descends from the top-left, so the x-axis sits
# at the top (t = 0) and the y-axis arrow points downward toward -inf.
# ---------------------------------------------------------------------------

# True transformed curve and transformed noisy observations
t_curve_x = x
t_curve = np.log(1 - cdf)
t_obs = [np.log(1 - yi) for yi in ys]

T_BOTTOM = -6.8  # lower limit of the t-axis (stands in for -inf)

# ---------------------------------------------------------------------------
# "Optimized" knot values -- a stand-in for the solution of Problem (2). We
# pull each noisy observation a fraction of the way back toward the true
# transformed curve. The result is non-increasing and concave (checked at
# import time), i.e., feasible for the convex program over the knots.
# ---------------------------------------------------------------------------
_t_true_knots = np.array([np.log(1 - yt) for yt in ys_true])
_t_obs_knots = np.array(t_obs)
_SHRINK = 0.4  # fraction pulled from the red observation toward the truth
tau_hat = _t_obs_knots + _SHRINK * (_t_true_knots - _t_obs_knots)

# Knot vector including tau_0 = t(l) = 0 at x0 = l
knot_x = np.array([0.0] + xs)
knot_t = np.concatenate([[0.0], tau_hat])

# Feasibility check: non-increasing and concave (slopes non-increasing)
_slopes = np.diff(knot_t) / np.diff(knot_x)
assert np.all(np.diff(knot_t) <= 0), "knot values must be non-increasing"
assert np.all(np.diff(_slopes) <= 1e-9), "knot values must be concave"

# Analytic log-tail appended after the last knot: tau_k + a*log((u-x)/(u-x_k)),
# with slope a/(u-x_k) >= |last PWL slope| so the full t stays concave.
U_END = 10.0
A_TAIL = max(1.0, -_slopes[-1] * (U_END - xs[2]))

# t-space tail samples (for plotting)
_xt = np.linspace(xs[2], U_END - 1e-3, 200)
_tt = tau_hat[2] + A_TAIL * np.log((U_END - _xt) / (U_END - xs[2]))

# Fitted cdf F_hat(x) = 1 - exp(t_hat(x)) over the full grid (PWL body + tail)
_body = x <= xs[2]
Fhat = np.empty_like(x)
Fhat[_body] = 1 - np.exp(np.interp(x[_body], knot_x, knot_t))
Fhat[~_body] = 1 - np.exp(tau_hat[2]) * ((U_END - x[~_body]) / (U_END - xs[2])) ** A_TAIL

GREEN = "#2ca02c"


def make_figure_t(stage, filename=None):
    """Build the t-space figure t(x) = ln(1 - F(x)) as a staged reveal.
    stage 0: just the axes and the black curve.
    stage 1: also the x_i tick labels.
    stage 2: also the red markers and the 0 tick (data in t-space).
    stage 3: also the green optimized knot values (Problem (2)) and t_hat ticks.
    stage 4: also the piecewise-linear interpolation between the green dots.
    stage 5: also the analytic log-tail descending to -inf as x -> u.
    If filename is given, save the figure as a PNG."""
    show_xticks = stage >= 1
    show_data = stage >= 2
    show_green = stage >= 3
    show_pwl = stage >= 4
    show_tail = stage >= 5

    fig, ax = plt.subplots(figsize=(12, 7))

    # Transformed (true) curve
    ax.plot(t_curve_x, t_curve, color="black", linewidth=3)

    if show_data:
        # Red observation markers
        if stage == 2:
            # Dashed guide lines only while the data is the focus
            ax.vlines(xs, 0, t_obs, colors="grey", linestyles="--", linewidth=2)
            ax.hlines(t_obs, 0, xs, colors="grey", linestyles="--", linewidth=2)
        ax.scatter(xs, t_obs, marker="X", s=550, color="black", zorder=5)

    if show_green:
        # Green piecewise-linear interpolation through the optimized knots
        if show_pwl:
            ax.plot(knot_x, knot_t, color=GREEN, linewidth=3, zorder=4)
        # Green analytic log-tail toward -inf
        if show_tail:
            ax.plot(_xt, _tt, color=GREEN, linewidth=3, zorder=4)
        # Dashed guide lines to the knot values while they are the focus
        if stage == 3:
            ax.hlines(tau_hat, 0, xs, colors=GREEN, linestyles="--",
                      linewidth=2, alpha=0.7)
        # Green optimized knot values (including tau_0 = 0 at x = l)
        ax.scatter(knot_x, knot_t, marker="o", s=300, color=GREEN, zorder=6)

    # Legend (bottom left): true transform t_0, t_hat, and the data markers.
    # The interpolated curve t_hat(x) only enters the legend once it is drawn.
    _handles = [Line2D([0], [0], color="black", lw=3, label=r"$t_0$")]
    if show_pwl:
        _handles.append(Line2D([0], [0], color=GREEN, lw=3, label=r"$\hat{t}(x)$"))
    if show_green:
        _handles.append(Line2D([0], [0], color=GREEN, marker="o",
                               linestyle="None", markersize=14,
                               label=r"$\hat{t}(x_i)$"))
    if show_data:
        _handles.append(Line2D([0], [0], color="black", marker="X",
                               linestyle="None", markersize=16, label="Data"))
    ax.legend(handles=_handles, fontsize=26, loc="lower left",
              frameon=False, bbox_to_anchor=(0.04, 0.04))

    # Limits (tight left margin so y-tick labels sit near the y-axis)
    ax.set_xlim(-0.2, 10.5)
    ax.set_ylim(T_BOTTOM, 0.25)

    # X-axis lives at the top (t = 0)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    if show_xticks:
        ax.set_xticks([0] + xs + [10])
        ax.set_xticklabels(
            [r"$l$", r"$x_1$", r"$x_2$", r"$x_3$", r"$u$"], fontsize=28)
    else:
        ax.set_xticks([0, 10])
        ax.set_xticklabels([r"$l$", r"$u$"], fontsize=28)

    # Y-axis ticks (no 0 tick)
    if show_green:
        ax.set_yticks(list(tau_hat))
        ax.set_yticklabels([
            r"$\hat{t}(x_1)$",
            r"$\hat{t}(x_2)$",
            r"$\hat{t}(x_3)$",
        ], fontsize=24)
    else:
        ax.set_yticks([])

    # Axis labels, pinned so they do not shift between figures
    ax.set_xlabel(r"$x$", fontsize=28)
    ax.set_ylabel(r"$t(x)$", fontsize=28, rotation=0)
    ax.xaxis.set_label_coords(0.5, 1.14)
    ax.yaxis.set_label_coords(-0.08, 0.40)

    # Arrowed axes: x to the right along t = 0, y downward toward -inf
    ax.annotate("", xy=(10.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=3, color="black"))
    ax.annotate("", xy=(0, T_BOTTOM), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=3, color="black"))

    # Styling
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=0, width=2, pad=12)
    ax.tick_params(axis="y", length=0, width=2, pad=4)

    # Pin the axes to a fixed rectangle (identical in every t-space figure).
    ax.set_position([0.12, 0.10, 0.82, 0.72])

    if filename is not None:
        fig.savefig(filename, dpi=300)

    plt.show()

    return fig, ax


# Figure 1 (t-space): just the axes and the black curve
make_figure_t(stage=0,
              filename=os.path.join(OUTDIR, "presentation_t_curve.png"))

# Figure 2 (t-space): add the x_i tick labels
make_figure_t(stage=1,
              filename=os.path.join(OUTDIR, "presentation_t_xticks.png"))

# Figure 3 (t-space): full info (red markers and 0 tick)
make_figure_t(stage=2,
              filename=os.path.join(OUTDIR, "presentation_t_full.png"))

# Figure 4 (t-space): green optimized knot values (Problem (2)) + t_hat ticks
make_figure_t(stage=3,
              filename=os.path.join(OUTDIR, "presentation_t_optdots.png"))

# Figure 5 (t-space): piecewise-linear interpolation between the green dots
make_figure_t(stage=4,
              filename=os.path.join(OUTDIR, "presentation_t_pwl.png"))

# Figure 6 (t-space): append the analytic log-tail toward -inf as x -> u
make_figure_t(stage=5,
              filename=os.path.join(OUTDIR, "presentation_t_tail.png"))

# Figure 7 (F-space): the estimated cdf F_hat (green) mapped back from t-space
make_figure(stage=2, show_fit=True,
            filename=os.path.join(OUTDIR, "presentation_cdf_fit.png"))


#%%
