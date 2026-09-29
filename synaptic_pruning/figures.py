"""Figures from the aggregated log (aggregate.Data). Each function returns a
matplotlib figure, or None if the data it needs is missing. Points are means
over seeds with 95% CI bars; improvement is the % reduction in test MAE
against the reference method, paired by configuration and seed."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import aggregate as agg
from design import BASE_SEQ

ARCH_COLORS = {"rnn": "#8d8d8d", "lstm": "#30638e", "gru": "#00798c",
               "cnn": "#edae49", "transformer": "#d1495b"}
METHOD_ORDER = ["dropout", "mc_dropout", "l2", "pruning", "random_pruning", "oneshot", "narrow"]


def _base(runs):
    return runs[runs["base_cell"]]


def methods_core(data):
    """Improvement over no regularization at the base width, data and epochs,
    per dataset and architecture (averaged over sequence lengths)."""
    imp = agg.improvements(_base(data.runs), "none")
    imp = imp[imp["variant"].isin(agg.DEFAULT_VARIANTS.values())]
    if imp.empty:
        return None
    t = agg.summarize(imp, ["dataset", "arch", "variant"])
    datasets = sorted(t["dataset"].unique())
    variants = [agg.DEFAULT_VARIANTS[m] for m in METHOD_ORDER if agg.DEFAULT_VARIANTS[m] in set(t["variant"])]
    archs = [a for a in ARCH_COLORS if a in set(t["arch"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.6), squeeze=False, sharey=True)
    width = 0.8 / max(1, len(archs))
    for ax, ds in zip(axes[0], datasets):
        for j, arch in enumerate(archs):
            part = t[(t["dataset"] == ds) & (t["arch"] == arch)].set_index("variant").reindex(variants)
            ax.bar(np.arange(len(variants)) + j * width, part["mean"], width, yerr=part["ci"].fillna(0),
                   color=ARCH_COLORS[arch], label=arch, capsize=1.5, error_kw={"lw": 0.7})
        ax.set_xticks(np.arange(len(variants)) + 0.4 - width / 2,
                      [v.split(":")[0] for v in variants], rotation=40, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(ds, fontsize=10)
    axes[0][0].set_ylabel("test MAE improvement vs none (%)")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def scale_axes(data):
    """The edge of synaptic pruning (solid) and dropout (dashed) over no
    regularization, along each scale axis. Other axes are averaged out."""
    runs = data.runs[data.runs["seq_len"] == BASE_SEQ]
    imp = agg.improvements(runs, "none")
    imp = imp[imp["variant"].isin([agg.PRUNING, agg.DROPOUT_REF])]
    axes_ = (("width", "model width"), ("train_frac", "fraction of training rows"),
             ("epoch_mult", "epochs (x base)"))
    datasets = sorted(imp["dataset"].unique()) if len(imp) else []
    if not datasets:
        return None
    fig, axs = plt.subplots(len(datasets), 3, figsize=(11, 2.8 * len(datasets)), squeeze=False,
                            sharey="row")
    for i, ds in enumerate(datasets):
        for k, (col, label) in enumerate(axes_):
            ax = axs[i][k]
            t = agg.summarize(imp[imp["dataset"] == ds], ["arch", "variant", col])
            for (arch, variant), part in t.groupby(["arch", "variant"]):
                part = part.sort_values(col)
                ax.errorbar(part[col], part["mean"], yerr=part["ci"].fillna(0), marker="o", ms=3,
                            lw=1.2, capsize=2, color=ARCH_COLORS.get(arch),
                            ls="-" if variant == agg.PRUNING else "--",
                            label=f"{arch} {variant.split(':')[0]}")
            ax.set_xscale("log", base=2 if col != "train_frac" else 10)
            ax.axhline(0, color="k", lw=0.5)
            if i == len(datasets) - 1:
                ax.set_xlabel(label)
            if k == 0:
                ax.set_ylabel(f"{ds}\nimprovement vs none (%)")
    axs[0][-1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def sweep_heatmap(data):
    """Pruning improvement over none for each (smin, smax, ramp horizon)."""
    imp = agg.improvements(_base(data.runs[data.runs["seq_len"] == BASE_SEQ]), "none")
    imp = imp[imp["variant"].str.startswith("pruning:")]
    if imp["variant"].nunique() < 2:
        return None
    parts = imp["variant"].str.extract(r"pruning:(?P<smin>[\d.]+)-(?P<smax>[\d.]+):h(?P<h>\d+)")
    imp = imp.assign(**parts.astype(float))
    horizons = sorted(imp["h"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 3.2), squeeze=False)
    lim = None
    tabs = []
    for h in horizons:
        t = agg.summarize(imp[imp["h"] == h], ["smin", "smax"])
        tabs.append(t.pivot(index="smin", columns="smax", values="mean"))
    lim = max(np.nanmax(np.abs(t.values)) for t in tabs) or 1
    for ax, h, t in zip(axes[0], horizons, tabs):
        im = ax.imshow(t.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(t.shape[1]), [f"{c:g}" for c in t.columns])
        ax.set_yticks(range(t.shape[0]), [f"{r:g}" for r in t.index])
        for a in range(t.shape[0]):
            for b in range(t.shape[1]):
                ax.text(b, a, f"{t.values[a, b]:.1f}", ha="center", va="center", fontsize=8)
        ax.set_xlabel("smax")
        ax.set_ylabel("smin")
        ax.set_title("ramp over the whole run" if h == 0 else f"ramp over {int(h)} epochs", fontsize=9)
    fig.colorbar(im, ax=list(axes[0]), label="improvement vs none (%)")
    return fig


def controls(data):
    """Synaptic pruning against its controls: random pruning at the same
    schedule, one-shot magnitude pruning, and a narrower dense net with the
    same parameter count. Positive = synaptic pruning is better."""
    imp = [agg.contrast(_base(data.runs), agg.PRUNING, agg.DEFAULT_VARIANTS[c]).assign(control=c)
           for c in agg.CONTROLS]
    imp = [i for i in imp if len(i)]
    if not imp:
        return None
    import pandas as pd
    imp = pd.concat(imp, ignore_index=True)
    t = agg.summarize(imp, ["control", "dataset", "arch"])
    datasets = sorted(t["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.4), squeeze=False, sharey=True)
    ctrls = [c for c in agg.CONTROLS if c in set(t["control"])]
    archs = [a for a in ARCH_COLORS if a in set(t["arch"])]
    width = 0.8 / max(1, len(archs))
    for ax, ds in zip(axes[0], datasets):
        for j, arch in enumerate(archs):
            part = t[(t["dataset"] == ds) & (t["arch"] == arch)].set_index("control").reindex(ctrls)
            ax.bar(np.arange(len(ctrls)) + j * width, part["mean"], width, yerr=part["ci"].fillna(0),
                   color=ARCH_COLORS[arch], label=arch, capsize=1.5, error_kw={"lw": 0.7})
        ax.set_xticks(np.arange(len(ctrls)) + 0.4 - width / 2, ctrls, fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(ds, fontsize=10)
    axes[0][0].set_ylabel("pruning vs control: MAE improvement (%)")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


FIGURES = {"methods_core": methods_core, "scale_axes": scale_axes,
           "sweep_heatmap": sweep_heatmap, "controls": controls}


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
