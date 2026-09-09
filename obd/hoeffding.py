"""The Hoeffding bound: how P(|nu - mu| > epsilon) shrinks with N and epsilon.

    P(|nu - mu| > epsilon) <= 2 exp(-2 epsilon^2 N)

A probability lives in [0, 1], so the bound is only informative once it drops
below 1 — above that it's a trivially true (and uninteresting) statement.
Below about 1e-3 it's trivially *zero* for any practical purpose. Both panels
below are therefore windowed to that informative middle stretch, on linear
axes rather than log, so the shape of the decay is the whole story:

    left:  bound vs N, epsilon fixed at 0.05  — sample size buys you
    right: bound vs epsilon, N fixed at 30    — margin buys you the other way

Two panels, saved as one figure to figures/hoeffding.png.
Run directly: `python3 hoeffding.py`.
"""

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).parent / "figures"
TRIVIAL = 1e-3  # below this the bound is uninformatively close to zero

BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

STYLE = {
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
}


def bound(epsilon, n):
    return 2 * np.exp(-2 * epsilon ** 2 * n)


def n_at(p, epsilon):
    """N at which the bound equals p, for fixed epsilon."""
    return -math.log(p / 2) / (2 * epsilon ** 2)


def epsilon_at(p, n):
    """epsilon at which the bound equals p, for fixed N."""
    return math.sqrt(-math.log(p / 2) / (2 * n))


def _grid(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_vs_n(ax, epsilon=0.05):
    n_lo, n_hi = n_at(1.0, epsilon), n_at(TRIVIAL, epsilon)
    n = np.linspace(n_lo, n_hi, 500)
    ax.plot(n, bound(epsilon, n), color=BLUE, linewidth=2)
    ax.set_xlim(n_lo, n_hi)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("N (sample size)")
    ax.set_ylabel(r"P($|\nu - \mu| > \epsilon$)  (bound)")
    ax.set_title(f"vs N, at $\\epsilon$ = {epsilon}", fontsize=11, pad=12)


def plot_vs_epsilon(ax, n=30):
    eps_lo, eps_hi = epsilon_at(1.0, n), epsilon_at(TRIVIAL, n)
    epsilon = np.linspace(eps_lo, eps_hi, 500)
    ax.plot(epsilon, bound(epsilon, n), color=BLUE, linewidth=2)
    ax.set_xlim(eps_lo, eps_hi)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\epsilon$ (margin of error)")
    ax.set_ylabel(r"P($|\nu - \mu| > \epsilon$)  (bound)")
    ax.set_title(f"vs $\\epsilon$, at N = {n}", fontsize=11, pad=12)


def make_figure():
    plt.rcParams.update(STYLE)
    fig, (ax_n, ax_eps) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("Hoeffding bound: 2 exp(-2 $\\epsilon^2$ N)", fontsize=12, y=1.03)
    plot_vs_n(ax_n)
    plot_vs_epsilon(ax_eps)
    for ax in (ax_n, ax_eps):
        _grid(ax)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)
    fig = make_figure()
    path = FIGURES_DIR / "hoeffding.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {path}")
