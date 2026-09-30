"""Analysis of results/*.jsonl: tidy frames, summary statistics, paired contrasts.

    data = load(path)       -> Data(units, curves, overlap, overlap_curves)
    write_tables(data, dir) -> CSVs under results/tables/

No prose is generated. Every summary is mean +- a 95% t-interval over seeds,
and every comparison of two criteria is paired (same unit config, same seed).
Seeds are the blocking factor: a paired diff is first averaged within a seed
over any factor a summary marginalises over, so the replicates in every test
are seeds, never dependent cells of one seed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from datasets import DATASETS
from experiments import FRACTIONS

LEVELS = np.array((1.0,) + FRACTIONS)

CONFIG = ["dataset", "arch", "width", "data_frac", "epochs", "weight_decay", "retrain_epochs"]
FACTORS = ["dataset", "arch", "width", "data_frac", "epoch_mult", "weight_decay", "retrain_mult"]
CENTER = {"width": 1.0, "data_frac": 1.0, "epoch_mult": 1.0, "retrain_mult": 1.0,
          "default_wd": True}

# (a, b): a - b, the pairs worth testing
RETRAIN_CONTRASTS = (("saliency", "magnitude"), ("saliency_layermean", "magnitude"),
                     ("saliency", "saliency_layermean"), ("taylor", "magnitude"),
                     ("magnitude", "random"), ("saliency", "random"))
SWEEP_CONTRASTS = RETRAIN_CONTRASTS + (("saliency_recomputed", "magnitude"),
                                       ("saliency_recomputed", "saliency"))


@dataclass
class Data:
    units: pd.DataFrame           # one row per unit: config, base metrics, timing
    curves: pd.DataFrame          # long: config, seed, kind, criterion, keep, metrics
    overlap: pd.DataFrame         # one row per unit: ranking-agreement statistics
    overlap_curves: pd.DataFrame  # long: config, seed, keep, overlap curves


def _annotate(df):
    info = df["dataset"].map(DATASETS)
    df["epoch_mult"] = (df["epochs"] / info.map(lambda i: i.epochs)).round(3)
    df["retrain_mult"] = (df["retrain_epochs"] / info.map(lambda i: i.retrain_epochs)).round(3)
    df["default_wd"] = df["weight_decay"] == info.map(lambda i: i.weight_decay)
    df["model"] = df["dataset"] + "/" + df["arch"]
    return df


def load(path):
    records = []
    for line in Path(path).read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not records:
        raise SystemExit(f"{path}: no finished units yet")

    units, curves, overlap, overlap_curves = [], [], [], []
    for r in records:
        u = r["unit"]
        ident = {c: u[c] for c in CONFIG + ["seed"]}
        units.append({**ident, "key": r["key"], "n_prunable": r["n_prunable"],
                      "n_params": r["n_params"], "duration_s": r["_meta"]["duration_s"],
                      **{f"base_{k}": v for k, v in r["base"].items() if k != "remaining"}})
        for kind, table in (("sweep", r["sweeps"]), ("retrain", r["retrain"])):
            for criterion, cols in table.items():
                df = pd.DataFrame(cols)
                curves.append(df.assign(kind=kind, criterion=criterion, **ident))
        ov = r["overlap"]
        overlap.append({**ident, **ov["global"], "n_prunable": ov["n_prunable"]})
        overlap_curves.append(pd.DataFrame(ov["overlap_curve"]).rename(
            columns={"remaining": "n_remaining"}).assign(
                keep=lambda d: d["chance"], **ident))
    frames = [pd.DataFrame(units), pd.concat(curves, ignore_index=True),
              pd.DataFrame(overlap), pd.concat(overlap_curves, ignore_index=True)]
    for df in frames[1::2]:   # actual kept fraction -> the nominal level it belongs to
        df["keep"] = LEVELS[np.abs(df["keep"].values[:, None] - LEVELS).argmin(axis=1)]
    return Data(*(_annotate(f) for f in frames))


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


def summarize(df, by, values):
    """Mean, 95% CI half-width and n (seeds) of each value column, by group."""
    g = df.groupby(by, dropna=False)
    out = {}
    for v in values:
        out[f"{v}"] = g[v].mean()
        out[f"{v}_ci"] = g[v].apply(ci95)
    out["n"] = g[values[0]].count()
    return pd.DataFrame(out).reset_index()


def paired(curves, kind, contrasts, value, by):
    """For each (a, b) in `contrasts`: the paired difference a - b of `value`,
    summarised over seeds within each group of `by` (columns of `curves`,
    which must include "keep"). Reports mean, CI and seeds. No p-value: with
    the few seeds here a signed-rank test cannot reach 0.05."""
    df = curves[curves["kind"] == kind]
    keys = CONFIG + ["seed", "keep"]
    wide = df.pivot_table(index=keys, columns="criterion", values=value)
    rows = []
    for a, b in contrasts:
        if a not in wide or b not in wide:
            continue
        diff = (wide[a] - wide[b]).rename("diff").reset_index().dropna(subset=["diff"])
        diff = _annotate(diff)
        per_seed = diff.groupby(by + ["seed"], dropna=False)["diff"].mean().reset_index()
        g = per_seed.groupby(by, dropna=False)["diff"]
        out = pd.DataFrame({"mean_diff": g.mean(), "ci": g.apply(ci95),
                            "n_seeds": g.count()}).reset_index()
        out.insert(0, "contrast", f"{a} - {b}")
        out.insert(1, "metric", value)
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def variance_explained(df, value, factors):
    """eta^2 of each factor taken alone: the share of the variance of `value`
    (across the rows of `df`) that is between that factor's levels. A
    descriptive effect-size ranking, not a significance test."""
    total = ((df[value] - df[value].mean()) ** 2).sum()
    rows = []
    for f in factors:
        if df[f].nunique() < 2 or total == 0:
            continue
        g = df.groupby(f)[value]
        between = (g.count() * (g.mean() - df[value].mean()) ** 2).sum()
        rows.append({"factor": f, "levels": df[f].nunique(), "eta_sq": between / total})
    return pd.DataFrame(rows).sort_values("eta_sq", ascending=False) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

OVERLAP_VALUES = ["spearman", "kendall_subsample", "pearson_log", "var_log_h",
                  "var_log_w", "cov_log_h_log_w", "between_layer_frac"]


def write_tables(data, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    curves = data.curves

    tables = {
        "retrain_summary": summarize(curves[curves["kind"] == "retrain"],
                                     cfg + ["criterion", "keep"],
                                     ["test_acc", "val_acc", "test_loss", "pre_test_acc"]),
        "sweep_summary": summarize(curves[curves["kind"] == "sweep"],
                                   cfg + ["criterion", "keep"],
                                   ["train_loss", "test_acc", "test_loss"]),
        "paired_retrain": paired(curves, "retrain", RETRAIN_CONTRASTS, "test_acc",
                                 cfg + ["keep"]),
        "paired_sweep": paired(curves, "sweep", SWEEP_CONTRASTS, "train_loss",
                               cfg + ["keep"]),
        "overlap_summary": summarize(data.overlap, cfg, OVERLAP_VALUES),
        "overlap_curves_summary": summarize(data.overlap_curves, cfg + ["keep"],
                                            ["overlap_frac", "jaccard_kept", "jaccard_pruned"]),
        "units": data.units.drop(columns=["key"]),
    }

    # which factors move the OBD-vs-magnitude gap (and the ranking agreement)?
    gap = paired(curves, "retrain", [("saliency", "magnitude")], "test_acc",
                 cfg + ["epoch_mult", "retrain_mult", "keep"])
    gap = gap[gap["n_seeds"] > 0].rename(columns={"mean_diff": "gap"})
    parts = []
    if len(gap):
        parts.append(variance_explained(_annotate(gap), "gap", FACTORS + ["keep"]).assign(
            outcome="retrain test_acc: saliency - magnitude"))
    ov = data.overlap
    if len(ov):
        parts.append(variance_explained(ov, "spearman", FACTORS).assign(
            outcome="spearman(saliency, magnitude)"))
    if parts:
        tables["effects"] = pd.concat(parts, ignore_index=True)

    for name, df in tables.items():
        if len(df):
            df.to_csv(out_dir / f"{name}.csv", index=False)
            print(f"wrote {out_dir / name}.csv  ({len(df)} rows)")


def center(df):
    """Rows at the design's centre point (width 1, full data, 1x epochs,
    default weight decay and retrain budget)."""
    mask = pd.Series(True, index=df.index)
    for col, val in CENTER.items():
        mask &= df[col] == val
    return df[mask]
