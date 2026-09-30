"""Figures from the aggregated log (aggregate.Data). Each function takes the
data and returns a matplotlib figure, or None if the data it needs is missing.
Every curve is a mean over seeds with a 95% CI band."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import aggregate as agg

COLORS = {"saliency": "#d1495b", "saliency_recomputed": "#edae49",
          "saliency_layermean": "#00798c", "magnitude": "#30638e",
          "taylor": "#66a182", "random": "#8d8d8d"}
CRITERIA = ("saliency", "saliency_layermean", "taylor", "magnitude", "random")


def _panels(n, cols=4):
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows),
                             squeeze=False, sharex=True)
    return fig, axes.flatten()


def _band(ax, df, x, y, label, color):
    g = df.groupby(x)[y]
    mean, ci = g.mean(), g.apply(agg.ci95).fillna(0)
    ax.plot(mean.index, mean.values, color=color, label=label, lw=1.6)
    ax.fill_between(mean.index, mean - ci, mean + ci, color=color, alpha=0.18, lw=0)


def _curve_grid(df, criteria, y, ylabel, title):
    """One panel per dataset/arch, one line per criterion, x = weights kept."""
    models = sorted(df["model"].unique())
    if not models:
        return None
    fig, axes = _panels(len(models))
    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        for c in criteria:
            part = sub[sub["criterion"] == c]
            if len(part):
                _band(ax, part, "keep", y, c, COLORS[c])
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_title(model, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
    for ax in axes[len(models):]:
        ax.axis("off")
    for ax in axes[-4:]:
        ax.set_xlabel("fraction of weights kept")
    axes[0].legend(fontsize=7)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


def retrain_curves(data):
    df = agg.center(data.curves[data.curves["kind"] == "retrain"])
    return _curve_grid(df, CRITERIA, "test_acc", "test accuracy",
                       "Prune-retrain loop at the centre point (width 1, full data, 1x epochs)")


def sweep_curves(data):
    df = agg.center(data.curves[data.curves["kind"] == "sweep"])
    fig = _curve_grid(df, CRITERIA + ("saliency_recomputed",), "train_loss", "train loss",
                      "No retraining: train loss vs weights kept (centre point)")
    if fig is not None:
        for ax in fig.axes:
            ax.set_yscale("log")
    return fig


def paired_heatmap(data):
    """saliency - magnitude test accuracy after retraining, per model and width."""
    d = agg.paired(data.curves, "retrain", [("saliency", "magnitude")], "test_acc",
                   ["model", "width", "keep"])
    if d.empty:
        return None
    d["row"] = d["model"] + "  w" + d["width"].astype(str)
    table = d.pivot(index="row", columns="keep", values="mean_diff")
    sig = d.pivot(index="row", columns="keep", values="mean_diff").abs() > \
        d.pivot(index="row", columns="keep", values="ci")
    fig, ax = plt.subplots(figsize=(1 + 0.6 * table.shape[1], 0.32 * table.shape[0] + 1.5))
    lim = np.nanmax(np.abs(table.values)) or 1
    im = ax.imshow(table.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [f"{k:g}" for k in table.columns], fontsize=7)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=7)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if sig.values[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=9)
    ax.set_xlabel("fraction of weights kept")
    fig.colorbar(im, label="test acc: OBD - magnitude (retrained)")
    ax.set_title("* = 95% t-interval across seeds excludes 0", fontsize=9)
    fig.tight_layout()
    return fig


def scale_effects(data, keep=0.05):
    """The OBD - magnitude gap (retrained test accuracy) against each scale
    axis: width, training-set fraction, and epochs."""
    curves = data.curves
    curves = curves[curves["keep"] == keep]
    axes_ = (("width", "model width"), ("data_frac", "training-set fraction"),
             ("epoch_mult", "epochs (x base)"), ("weight_decay", "weight decay"))
    fig, axs = plt.subplots(1, len(axes_), figsize=(4 * len(axes_), 3.2))
    any_data = False
    for ax, (col, label) in zip(axs, axes_):
        d = agg.paired(curves, "retrain", [("saliency", "magnitude")], "test_acc",
                       ["model", col])
        for model, part in d.groupby("model"):
            part = part.sort_values(col)
            x = np.arange(len(part)) if col == "weight_decay" else part[col]
            ax.errorbar(x, part["mean_diff"], yerr=part["ci"].fillna(0), marker="o",
                        ms=3, lw=1, capsize=2, label=model)
            any_data = True
        if col == "weight_decay":
            ax.set_xticks(range(len(part)), [f"{v:g}" for v in part[col]])
        elif col != "data_frac":
            ax.set_xscale("log", base=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(label)
    axs[0].set_ylabel(f"test acc gap at {keep:g} kept")
    axs[-1].legend(fontsize=6, ncol=2)
    fig.suptitle("OBD - magnitude, prune-retrain, across scale axes", y=1.0)
    fig.tight_layout()
    return fig if any_data else None


def overlap_vs_weight_decay(data):
    df = data.overlap
    if df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for model, part in df.groupby("model"):
        s = part.groupby("weight_decay")["spearman"]
        axes[0].errorbar(range(len(s.mean())), s.mean(), yerr=s.apply(agg.ci95).fillna(0),
                         marker="o", ms=3, lw=1, capsize=2, label=model)
        v = part.groupby("weight_decay")["between_layer_frac"]
        axes[1].plot(range(len(v.mean())), v.mean(), marker="o", ms=3, lw=1)
    wds = sorted(df["weight_decay"].unique())
    for ax in axes:
        ax.set_xticks(range(len(wds)), [f"{w:g}" for w in wds])
        ax.set_xlabel("weight decay")
    axes[0].set_ylabel("Spearman(saliency, |w|)")
    axes[1].set_ylabel("share of Var(log h) between layers")
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def overlap_curves(data):
    df = agg.center(data.overlap_curves)
    return _curve_grid(df.assign(criterion="jaccard_kept"), ("saliency",), "jaccard_kept",
                       "Jaccard of kept sets",
                       "Overlap of the OBD and magnitude kept sets") if len(df) else None


FIGURES = {"retrain_curves": retrain_curves, "sweep_curves": sweep_curves,
           "paired_heatmap": paired_heatmap, "scale_effects": scale_effects,
           "overlap_vs_weight_decay": overlap_vs_weight_decay,
           "overlap_curves": overlap_curves}


def make_all(data, out_dir):
    for name, fn in FIGURES.items():
        fig = fn(data)
        if fig is None:
            print(f"skipped {name} (no data)")
            continue
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")
