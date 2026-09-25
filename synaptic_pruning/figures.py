"""Every figure, as a pure function of the cached results dicts (see
main.py:load_results). `FIGURES` is the registry main.py iterates when
drawing all of them; each function returns a matplotlib Figure or None if
its required results aren't present yet.
"""

import numpy as np
import matplotlib.pyplot as plt

from pruning import PruneConfig, cubic_schedule

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

METHOD_COLORS = {
    "none": "#898781",
    "dropout": "#2a78d6",
    "mc_dropout": "#7a5cd6",
    "pruning": "#d64545",
}
METHOD_LABELS = {
    "none": "No regularization", "dropout": "Dropout",
    "mc_dropout": "MC Dropout", "pruning": "Synaptic Pruning",
}

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


def _grid(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# the schedule itself
# --------------------------------------------------------------------------

def fig_sparsity_schedule(results):
    """s(t) — the cubic sparsity ramp (Algorithm 2), independent of any run."""
    plt.rcParams.update(STYLE)
    cfg = PruneConfig()
    epochs = np.arange(1, cfg.schedule_epochs + 1)
    sparsity = [cubic_schedule(int(e), cfg) for e in epochs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, sparsity, color=METHOD_COLORS["pruning"], linewidth=2, marker="o", markersize=3)
    ax.axvline(cfg.warmup_epochs, color=MUTED, linestyle="--", linewidth=1)
    ax.text(cfg.warmup_epochs + 0.2, 0.02, "warmup ends", color=MUTED, fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("target sparsity s(t)")
    ax.set_title(f"Cubic sparsity schedule (smin={cfg.smin}, smax={cfg.smax})",
                fontsize=11, pad=12)
    ax.set_ylim(0, 0.8)
    _grid(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# replication grid
# --------------------------------------------------------------------------

def fig_replication_by_seqlen(results):
    rep = results.get("replication")
    if rep is None:
        return None
    plt.rcParams.update(STYLE)
    summary = rep["summary"]
    model_types = sorted({v["model_type"] for v in summary.values()})
    methods = rep["config"]["methods"]

    fig, axes = plt.subplots(1, len(model_types), figsize=(6 * len(model_types), 4.5), sharey=False)
    if len(model_types) == 1:
        axes = [axes]

    for ax, model_type in zip(axes, model_types):
        cells = sorted((v for v in summary.values() if v["model_type"] == model_type),
                       key=lambda v: v["seq_len"])
        seq_lens = [c["seq_len"] for c in cells]
        for method in methods:
            means = [c["methods"][method]["mean"] for c in cells]
            los = [c["methods"][method]["ci95"][0] for c in cells]
            his = [c["methods"][method]["ci95"][1] for c in cells]
            ax.plot(seq_lens, means, label=METHOD_LABELS[method],
                   color=METHOD_COLORS[method], linewidth=2, marker="o", markersize=4)
            ax.fill_between(seq_lens, los, his, color=METHOD_COLORS[method], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xticks(seq_lens)
        ax.set_xticklabels([str(s) for s in seq_lens])
        ax.set_xlabel("input sequence length")
        ax.set_ylabel("test MAE (standardized units)")
        ax.set_title(model_type.upper(), fontsize=11, pad=12)
        _grid(ax)

    axes[-1].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Air Quality (CO forecast): MAE vs sequence length, by regularization method",
                fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


def fig_replication_bars(results):
    """Mean MAE per method, averaged across sequence lengths — one bar
    group per model, mirroring the paper's summary tables/bar charts."""
    rep = results.get("replication")
    if rep is None:
        return None
    plt.rcParams.update(STYLE)
    summary = rep["summary"]
    model_types = sorted({v["model_type"] for v in summary.values()})
    methods = rep["config"]["methods"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / len(methods)
    x = np.arange(len(model_types))
    for i, method in enumerate(methods):
        means, errs = [], []
        for model_type in model_types:
            cells = [v for v in summary.values() if v["model_type"] == model_type]
            vals = [c["methods"][method]["mean"] for c in cells]
            means.append(float(np.mean(vals)))
            errs.append(float(np.std(vals)))
        ax.bar(x + i * width, means, width=width, yerr=errs, capsize=3,
              label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in model_types])
    ax.set_ylabel("mean test MAE across sequence lengths")
    ax.set_title("Method comparison, averaged over sequence lengths", fontsize=11, pad=12)
    ax.legend(frameon=False, fontsize=9)
    _grid(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# bitter-lesson ladder
# --------------------------------------------------------------------------

def fig_bitter_lesson_ladder(results):
    """The compute-scaling ladder: MAE (none vs pruning) and pruning's
    improvement over baseline at each tier, tiers ordered by increasing
    compute (hidden_size/epochs/n_rows all scale up together per tier)."""
    bl = results.get("bitter_lesson")
    if bl is None:
        return None
    if not bl["cells"] or "tier_index" not in bl["cells"][0]:
        print("  note: results/bitter_lesson.json is from the old 3x3x3 grid — "
              "rerun `python3 main.py --bitter-lesson` for the ladder figure")
        return None
    plt.rcParams.update(STYLE)
    cells = sorted(bl["cells"], key=lambda c: c["tier_index"])
    labels = [f"h={c['hidden_size']}\ne={c['epochs']}\nn={c['n_rows']}" for c in cells]
    x = np.arange(len(cells))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for method in ("none", "pruning"):
        means = [c[method]["mean"] for c in cells]
        los = [c[method]["ci95"][0] for c in cells]
        his = [c[method]["ci95"][1] for c in cells]
        ax1.plot(x, means, label=METHOD_LABELS[method], color=METHOD_COLORS[method],
                 linewidth=2, marker="o", markersize=4)
        ax1.fill_between(x, los, his, color=METHOD_COLORS[method], alpha=0.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_xlabel("compute tier (hidden size / epochs / training rows)")
    ax1.set_ylabel("test MAE (standardized units)")
    ax1.set_title("MAE vs compute tier", fontsize=11, pad=12)
    ax1.legend(frameon=False, fontsize=9)
    _grid(ax1)

    improvements = [c["improvement_pct"] for c in cells]
    colors = [METHOD_COLORS["pruning"] if v >= 0 else BASELINE for v in improvements]
    ax2.bar(x, improvements, color=colors)
    ax2.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_xlabel("compute tier (hidden size / epochs / training rows)")
    ax2.set_ylabel("pruning MAE improvement over baseline (%)")
    ax2.set_title("Improvement vs compute tier", fontsize=11, pad=12)
    _grid(ax2)

    fig.suptitle("Bitter-lesson ladder: does synaptic pruning's benefit survive scale?", fontsize=12, y=1.04)
    fig.tight_layout()
    return fig


FIGURES = {
    "sparsity_schedule": fig_sparsity_schedule,
    "replication_by_seqlen": fig_replication_by_seqlen,
    "replication_bars": fig_replication_bars,
    "bitter_lesson_ladder": fig_bitter_lesson_ladder,
}
