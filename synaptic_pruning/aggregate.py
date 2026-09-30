"""Analysis of results/*.jsonl: tidy frames, paired contrasts, summary tables.

    data = load(path)       -> Data(runs)
    write_tables(data, dir) -> CSVs under results/tables/

No prose is generated. The outcome is `test_mae` (test MAE at the epoch of the
best validation MAE, standardized units, lower is better). A method is compared
with a baseline by its improvement 100 * (baseline - method) / baseline, paired
by (configuration, seed): positive = better than the baseline.

Seeds are the blocking factor. A summary over a factor first averages the
paired improvements within each seed over every setting it marginalises over,
so the replicates in every interval are seeds, never the dependent cells of
one seed. Intervals are 95% t-intervals. The Friedman table's p-values are
Holm-adjusted within the table.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from design import BASE_SEQ, BASE_WIDTH

CFG = ["dataset", "arch", "width", "train_frac", "epochs", "seq_len"]
AXES = ["dataset", "arch", "width", "train_frac", "epoch_mult", "seq_len"]
DROPOUT_REF = "dropout:p=0.3"
CONTROLS = ("random_pruning", "oneshot", "narrow")


@dataclass
class Data:
    runs: pd.DataFrame


def variant_label(u):
    m = u["method"]
    if m in ("dropout", "mc_dropout"):
        return f"{m}:p={u['dropout']:g}"
    if m == "l2":
        return f"l2:wd={u['weight_decay']:g}"
    if m in ("pruning", "random_pruning"):
        return f"{m}:{u['smin']:g}-{u['smax']:g}:h{u['horizon']}"
    if m in ("oneshot", "narrow"):
        return f"{m}:{u['smax']:g}"
    return m


DEFAULT_VARIANTS = {"none": "none", "dropout": DROPOUT_REF, "mc_dropout": "mc_dropout:p=0.3",
                    "l2": "l2:wd=0.001", "pruning": "pruning:0.3-0.7:h0",
                    "random_pruning": "random_pruning:0.3-0.7:h0",
                    "oneshot": "oneshot:0.7", "narrow": "narrow:0.7"}
PRUNING = DEFAULT_VARIANTS["pruning"]


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        u = r["unit"]
        rows.append({**u, "variant": variant_label(u), "key": r["key"],
                     "test_mae": r["test_mae"], "val_mae": r["val_mae"],
                     "test_mae_final": r["test_mae_final"], "best_epoch": r["best_epoch"],
                     "persistence": r["persistence"]["test"], "n_prunable": r["n_prunable"],
                     "sparsity": r["sparsity"], "duration_s": r["_meta"]["duration_s"]})
    if not rows:
        raise SystemExit(f"{path}: no finished runs yet")
    df = pd.DataFrame(rows)
    # the design's centre epoch count (BASE_EPOCHS unless --set epochs=... was used)
    centre = df[(df["width"] == BASE_WIDTH) & (df["train_frac"] == 1.0)]["epochs"].mode().iloc[0]
    df["epoch_mult"] = (df["epochs"] / centre).round(3)
    df["skill"] = 1.0 - df["test_mae"] / df["persistence"]   # > 0: beats last-value persistence
    df["base_cell"] = (df["width"] == BASE_WIDTH) & (df["train_frac"] == 1.0) & (df["epochs"] == centre)
    return Data(df)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def ci95(x):
    """Half-width of the 95% t-interval of the mean (NaN below 2 samples)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return stats.sem(x) * stats.t.ppf(0.975, len(x) - 1)


def holm(pvalues):
    """Holm step-down adjustment; NaNs stay NaN."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(order) - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def improvements(runs, baseline="none", value="test_mae"):
    """Paired improvement of every other variant over `baseline`, per
    (configuration, seed): 100 * (baseline - variant) / baseline, plus the raw
    difference. Rows without a baseline partner are dropped."""
    base = runs[runs["variant"] == baseline][CFG + ["seed", value]].rename(columns={value: "base"})
    other = runs[runs["variant"] != baseline]
    df = other.merge(base, on=CFG + ["seed"])
    df["improvement_pct"] = 100.0 * (df["base"] - df[value]) / df["base"]
    df["diff"] = df["base"] - df[value]
    return df


def summarize(imp, by, value="improvement_pct"):
    """Mean improvement per group of `by`, with the CI across seeds after
    averaging within a seed over everything else."""
    per_seed = imp.groupby(by + ["seed"], dropna=False)[value].mean().reset_index()
    g = per_seed.groupby(by, dropna=False)[value]
    return pd.DataFrame({"mean": g.mean(), "ci": g.apply(ci95), "n_seeds": g.count(),
                         "wins": g.apply(lambda x: float((x > 0).mean()))}).reset_index()


def contrast(runs, a, b, value="test_mae"):
    """Paired improvement of variant `a` over variant `b` (b is the reference)."""
    imp = improvements(runs, baseline=b, value=value)
    return imp[imp["variant"] == a]


def variance_explained(df, value, factors):
    """eta^2 of each factor alone: the share of the variance of `value` that
    lies between that factor's levels. A descriptive ranking of effect sizes,
    not a significance test."""
    total = ((df[value] - df[value].mean()) ** 2).sum()
    rows = []
    for f in factors:
        if df[f].nunique() < 2 or total == 0:
            continue
        g = df.groupby(f)[value]
        rows.append({"factor": f, "levels": df[f].nunique(),
                     "eta_sq": (g.count() * (g.mean() - df[value].mean()) ** 2).sum() / total})
    return pd.DataFrame(rows).sort_values("eta_sq", ascending=False) if rows else pd.DataFrame()


def friedman_by_cell(runs, variants):
    """Friedman test across `variants`, blocks = seeds, per configuration."""
    rows = []
    sub = runs[runs["variant"].isin(variants)]
    for cfg, g in sub.groupby(CFG):
        wide = g.pivot(index="seed", columns="variant", values="test_mae").dropna()
        if wide.shape[1] < 3 or len(wide) < 3:
            continue
        stat, p = stats.friedmanchisquare(*(wide[c] for c in wide))
        best = wide.mean().idxmin()
        rows.append({**dict(zip(CFG, cfg)), "n_seeds": len(wide), "statistic": stat,
                     "p": p, "best_variant": best})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"])
    return out


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def write_tables(data, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = data.runs
    defaults = list(DEFAULT_VARIANTS.values())
    tables = {}

    g = runs.groupby(CFG + ["variant"])
    tables["cell_summary"] = pd.DataFrame({
        "test_mae": g["test_mae"].mean(), "ci": g["test_mae"].apply(ci95),
        "n_seeds": g["test_mae"].count(), "skill_vs_persistence": g["skill"].mean(),
        "best_epoch": g["best_epoch"].mean()}).reset_index()

    vs_none = improvements(runs, "none")
    vs_drop = improvements(runs, DROPOUT_REF)
    tables["vs_none_by_cell"] = summarize(vs_none, CFG + ["variant"])
    tables["vs_dropout_by_cell"] = summarize(vs_drop, CFG + ["variant"])

    # how does a method's edge move along each axis, marginalising the others
    axis_rows = []
    for ref, imp in (("none", vs_none), (DROPOUT_REF, vs_drop)):
        imp = imp[imp["variant"].isin(defaults)]
        for axis in AXES:
            t = summarize(imp, [axis, "variant"])
            t.insert(0, "reference", ref)
            t.insert(1, "axis", axis)
            axis_rows.append(t.rename(columns={axis: "level"}))
    tables["by_axis"] = pd.concat(axis_rows, ignore_index=True)

    # is pruning's edge the pruning, or just sparsity / capacity / search noise?
    ctrl = [contrast(runs, PRUNING, DEFAULT_VARIANTS[c]).assign(control=c) for c in CONTROLS]
    ctrl = pd.concat(ctrl, ignore_index=True) if ctrl else pd.DataFrame()
    if len(ctrl):
        tables["pruning_vs_controls"] = summarize(ctrl, ["control", "dataset", "arch"])

    tables["pruning_sweep"] = summarize(
        vs_none[vs_none["variant"].str.match(r"pruning:") & vs_none["base_cell"]
                & (vs_none["seq_len"] == BASE_SEQ)], ["dataset", "arch", "variant"])

    fried = friedman_by_cell(runs, [v for v in defaults])
    if len(fried):
        tables["friedman_core"] = fried

    eff = vs_none[(vs_none["variant"] == PRUNING) & (vs_none["seq_len"] == BASE_SEQ)]
    if len(eff):
        tables["effects"] = variance_explained(eff, "improvement_pct", AXES)

    tables["runs"] = runs.drop(columns=["key", "base_cell"])
    for name, df in tables.items():
        if len(df):
            df.to_csv(out_dir / f"{name}.csv", index=False)
            print(f"wrote {out_dir / name}.csv  ({len(df)} rows)")
