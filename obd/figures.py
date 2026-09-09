"""Figures, as pure functions from results/<name>.json to matplotlib.

No torch here, so plots can be iterated on without rerunning anything. Every
figure returns None (with a printed note) when its data is missing from the
results dict, so a partial run still plots what it has.

    fig2  loss vs weights remaining, no retraining        (paper Fig. 2)
    fig3  OBD's predicted loss increase vs the measured    (paper Fig. 3)
    fig4  prune-retrain loop, OBD vs magnitude control     (paper Fig. 4)
    fig5  how much the two rankings' kept sets overlap
    fig6  saliency vs magnitude, per layer
    fig7  where each ranking spends its pruning budget

Colours are three slots of a CVD-validated categorical palette, held to a fixed
meaning throughout: blue = OBD saliency (or "train"), red = magnitude (or
"test"), violet = the layer-mean saliency control. Every series also carries a
distinct marker and linestyle, so identity is never colour alone.
"""

import math

import matplotlib.pyplot as plt

BLUE = "#2a78d6"    # OBD saliency / train
RED = "#e34948"     # magnitude / test
VIOLET = "#4a3aa7"  # layer-mean saliency (the control between the two)
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

LOSS_LABEL = {"mse": "MSE", "ce": "cross-entropy loss"}


def _axes(title, figsize=(6, 4.2)):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=11, pad=12)
    _grid(ax)
    return fig, ax


def _panels(title, n, ncols=3, panel=(2.5, 2.2)):
    """A grid of small multiples with a shared figure title.

    Returns the axes flattened in row-major order, plus the helpers needed to
    label only the left column and the bottom row of the grid.
    """
    plt.rcParams.update(STYLE)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, grid = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(panel[0] * ncols, panel[1] * nrows))
    axes = [ax for row in grid for ax in row]
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[:n]:
        _grid(ax)
    fig.suptitle(title, fontsize=11, y=1.0)
    axes = axes[:n]
    left = axes[::ncols]
    bottom = [ax for i, ax in enumerate(axes) if i + ncols >= n]
    return fig, axes, left, bottom


def _grid(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _series(ax, results, key, y="train_loss", **style):
    """Plot one results series against weights remaining. No-op if absent."""
    if key not in results:
        return False
    points = sorted(results[key], key=lambda p: p["remaining"])
    ax.plot([p["remaining"] for p in points], [p[y] for p in points], **style)
    return True


def _missing(name, keys):
    print(f"  {name}: no data for {keys} — skipping")
    return None


def fig2_magnitude_vs_obd(results, exp):
    """Fig 2: training loss vs remaining weights, no retraining.

    Three ranking rules, in increasing order of what they assume about the
    curvature: magnitude (h constant everywhere), layer-mean saliency (h
    constant within a layer), full OBD saliency. Dashed = ranked once on the
    trained net; solid = re-ranked after every deletion.
    """
    series = [
        ("magnitude", dict(color=RED, marker="s", linestyle="-",
                           label="magnitude")),
        ("saliency", dict(color=BLUE, marker="o", linestyle="--",
                          label="OBD saliency (one-shot)")),
        ("saliency_recomputed", dict(color=BLUE, marker="o", linestyle="-",
                                     label="OBD saliency (re-ranked)")),
        ("saliency_layermean", dict(color=VIOLET, marker="^", linestyle="--",
                                    label="layer-mean saliency (one-shot)")),
        ("saliency_layermean_recomputed", dict(color=VIOLET, marker="^", linestyle="-",
                                               label="layer-mean saliency (re-ranked)")),
    ]
    if not any(key in results for key, _ in series):
        return _missing("fig2", [key for key, _ in series])

    fig, ax = _axes(f"Deleting weights without retraining — {exp.name} (paper Fig. 2)")
    for key, style in series:
        _series(ax, results, key, markersize=4, linewidth=2 if style["linestyle"] == "-" else 1.5,
                markerfacecolor="none" if style["linestyle"] == "--" else style["color"],
                **style)
    ax.set_xscale("log")
    ax.set_xlabel("weights remaining")
    ax.set_ylabel(f"training {LOSS_LABEL[exp.loss]}")
    ax.legend(frameon=False, fontsize=8)
    return fig


def fig3_predicted_vs_actual(results, exp):
    """Fig 3: the second-order forecast of the damage vs the measured damage."""
    sources = [("saliency_recomputed", "saliency", BLUE, "o", "OBD saliency"),
               ("saliency_layermean_recomputed", "saliency_layermean", VIOLET, "^",
                "layer-mean saliency")]
    plotted, bounds = [], []
    for preferred, fallback, color, marker, label in sources:
        source = results.get(preferred, results.get(fallback))
        if source is None:
            continue
        points = [(p["predicted_increase"], p["actual_increase"]) for p in source
                  if p.get("predicted_increase", 0) > 0 and p.get("actual_increase", 0) > 0]
        if points:
            plotted.append((points, color, marker, label))
            bounds += [v for point in points for v in point]
    if not plotted:
        return _missing("fig3", [s[0] for s in sources])

    fig, ax = _axes(f"Predicted vs. actual loss increase — {exp.name} (paper Fig. 3)")
    lo, hi = min(bounds), max(bounds)
    ax.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1, linestyle="--",
            label="perfect prediction")
    for points, color, marker, label in plotted:
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=22,
                   color=color, marker=marker, zorder=3, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted increase  (sum of deleted saliencies)")
    ax.set_ylabel("actual increase")
    ax.legend(frameon=False, fontsize=9)
    return fig


def fig4_retraining(results, exp):
    """Fig 4: iterative pruning with retraining after each deletion —
    OBD saliency ranking vs the magnitude-ranking control."""
    series = [("retrain", BLUE, "o", "OBD saliency"),
              ("retrain_magnitude", RED, "s", "magnitude (control)")]
    if not any(key in results for key, *_ in series):
        return _missing("fig4", [key for key, *_ in series])

    fig, ax = _axes(f"Pruning with retraining — {exp.name} (paper Fig. 4)")
    for key, color, marker, label in series:
        _series(ax, results, key, y="train_loss", color=color, marker=marker,
                markersize=4, linewidth=1.5, linestyle="--",
                markerfacecolor="none", label=f"{label} — train")
        _series(ax, results, key, y="test_loss", color=color, marker=marker,
                markersize=4, linewidth=2, label=f"{label} — test")
    ax.set_xscale("log")
    ax.set_xlabel("weights remaining")
    ax.set_ylabel(LOSS_LABEL[exp.loss])
    ax.legend(frameon=False, fontsize=9)
    return fig


# --------------------------------------------------------------------------
# The overlap analysis (results["overlap"], see experiments.overlap_analysis)
# --------------------------------------------------------------------------

def fig5_overlap(results, exp):
    """How much of the kept set do the two rankings agree on?

    The chance line is what the overlap would be if the rankings were
    independent: keeping n of N weights at random twice shares n/N of them.
    """
    overlap = results.get("overlap")
    if overlap is None:
        return _missing("fig5", ["overlap"])
    points = sorted(overlap["overlap_curve"], key=lambda p: p["remaining"])

    fig, ax = _axes(f"OBD and magnitude keep the same weights — {exp.name}")
    ax.plot([p["remaining"] for p in points], [p["chance"] for p in points],
            color=BASELINE, linewidth=1, linestyle="--",
            label="chance (independent rankings)")
    ax.plot([p["remaining"] for p in points], [p["overlap_frac"] for p in points],
            color=BLUE, marker="o", markersize=4, linewidth=2,
            label="shared fraction of the kept set")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("weights remaining")
    ax.set_ylabel("fraction of kept weights shared")
    # the curve runs bottom-left to top-right, so the chrome goes in the corners
    # it leaves free
    stats = overlap["global"]
    ax.annotate(f"Spearman(s, |w|) = {stats['spearman']:.3f}\n"
                f"curvature variance between layers = {stats['between_layer_frac']:.0%}",
                xy=(0.98, 0.04), xycoords="axes fraction", fontsize=8, color=MUTED,
                ha="right")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return fig


def fig6_saliency_vs_magnitude(results, exp):
    """One panel per layer: saliency against magnitude, log-log.

    s = ½·h·w², so on log-log axes every layer must lie on a slope-2 line
    whose *intercept* is that layer's curvature. The blue line is the layer's
    own fit; the grey line is the network-wide one, repeated in every panel.
    The gap between them is the layer's curvature offset — the entire
    difference between ranking by saliency and ranking by |w|.
    """
    overlap = results.get("overlap")
    if overlap is None:
        return _missing("fig6", ["overlap"])
    scatter, layers = overlap["scatter"], overlap["layers"]
    if not scatter["log_w"]:
        return _missing("fig6", ["overlap.scatter"])

    # c = s / w^2 = h/2 is the curvature factor a slope-2 line's intercept
    # encodes; in logs it is just log_s - 2*log_w.
    def median(values):
        return sorted(values)[len(values) // 2]

    by_layer = {i: ([], [], []) for i in range(len(layers))}
    for i, log_w, log_s in zip(scatter["layer"], scatter["log_w"], scatter["log_s"]):
        by_layer[i][0].append(math.exp(log_w))
        by_layer[i][1].append(math.exp(log_s))
        by_layer[i][2].append(log_s - 2 * log_w)
    c_global = math.exp(median([log_s - 2 * log_w for log_w, log_s
                                in zip(scatter["log_w"], scatter["log_s"])]))

    present = [i for i in range(len(layers)) if by_layer[i][0]]
    fig, axes, left, bottom = _panels(
        f"Saliency is magnitude, offset by a per-layer curvature — {exp.name}",
        len(present))
    lo = min(min(by_layer[i][0]) for i in present)
    hi = max(max(by_layer[i][0]) for i in present)
    for ax, i in zip(axes, present):
        w, s, log_c = by_layer[i]
        c_layer = math.exp(median(log_c))
        ax.scatter(w, s, s=3, color=BLUE, alpha=0.25, linewidths=0)
        ax.plot([lo, hi], [c_global * lo ** 2, c_global * hi ** 2],
                color=BASELINE, linewidth=1.2, linestyle="--")
        ax.plot([lo, hi], [c_layer * lo ** 2, c_layer * hi ** 2],
                color=BLUE, linewidth=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{layers[i]}\n{math.log10(c_layer / c_global):+.1f} decades",
                     fontsize=8, pad=6)
        ax.tick_params(labelsize=7)
    for ax in left:
        ax.set_ylabel("saliency  s", fontsize=8)
    for ax in bottom:
        ax.set_xlabel("|w|", fontsize=8)
    fig.tight_layout()
    return fig


def fig7_layer_allocation(results, exp):
    """One panel per layer: what fraction of it does each ranking keep?

    Within a layer the two rankings agree almost perfectly (fig 6), so this
    is where they actually differ — how the global pruning budget is split
    across layers.
    """
    overlap = results.get("overlap")
    if overlap is None:
        return _missing("fig7", ["overlap"])
    allocation = sorted(overlap["layer_allocation"], key=lambda p: p["remaining"])
    layers = overlap["layers"]

    fig, axes, left, bottom = _panels(
        f"Where each ranking spends its budget — {exp.name}", len(layers))
    remaining = [p["remaining"] for p in allocation]
    for ax, name in zip(axes, layers):
        size = allocation[0]["layer_size"][name]
        ax.plot(remaining, [p["kept_mag"][name] / size for p in allocation],
                color=RED, marker="s", markersize=3, linewidth=1.5,
                linestyle="--", markerfacecolor="none", label="magnitude")
        ax.plot(remaining, [p["kept_obd"][name] / size for p in allocation],
                color=BLUE, marker="o", markersize=3, linewidth=2,
                label="OBD saliency")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{name}  ({size:,} weights)", fontsize=8, pad=6)
        ax.tick_params(labelsize=7)
    for ax in left:
        ax.set_ylabel("fraction kept", fontsize=8)
    for ax in bottom:
        ax.set_xlabel("weights remaining", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


FIGURES = {
    "fig2_magnitude_vs_obd": fig2_magnitude_vs_obd,
    "fig3_predicted_vs_actual": fig3_predicted_vs_actual,
    "fig4_retraining": fig4_retraining,
    "fig5_overlap": fig5_overlap,
    "fig6_saliency_vs_magnitude": fig6_saliency_vs_magnitude,
    "fig7_layer_allocation": fig7_layer_allocation,
}
