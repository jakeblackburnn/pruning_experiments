"""The pruning experiments.

Three families, all keyed in PRUNE_EXPERIMENTS and written to
results/<name>.json by main.run_experiments:

  sweep_no_retrain      delete more and more weights, no retraining (Figs 2, 3)
  iterative_prune_retrain   prune a little, retrain a little, repeat (Fig 4)
  overlap_analysis      how much do the OBD and magnitude rankings agree,
                        and where does the curvature term carry information?
                        (Figs 5, 6, 7)

The core math (models, diagonal Hessian, saliencies, mask primitives) lives
in obd.py; plotting in figures.py.
"""

import copy
import math

import torch

from obd import (apply_masks, diagonal_hessian, evaluate, flatten, initial_masks,
                 keep_mask, prunable_names, prune_to, scores_for, train)


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def sweep_targets(n_total, exp):
    """Log-spaced weight counts to prune down to, largest first.

    Log spacing because the interesting behaviour is spread over orders of
    magnitude: nothing happens between 100% and 80% remaining, and everything
    happens in the last decade. Excludes n_total itself (that's the baseline
    point, which callers record separately).
    """
    lo = float(min(exp.sweep_min_remaining, n_total))
    hi = float(n_total)
    targets = torch.logspace(math.log10(lo), math.log10(hi), exp.sweep_points)
    descending = targets.long().unique(sorted=True).flip(0).tolist()
    return [n for n in descending if n < n_total]


def _setup(model, exp):
    """A prunable copy of the model plus its mask bookkeeping."""
    pruned = copy.deepcopy(model)
    names = prunable_names(pruned)
    masks = initial_masks(pruned)
    n_total = int(flatten(masks, names).sum().item())
    return pruned, names, masks, n_total


def _point(model, splits, exp, n_remaining):
    """The train/test loss and accuracy record written for one pruning level."""
    (x_tr, y_tr, l_tr), (x_te, y_te, l_te) = splits
    loss_tr, acc_tr = evaluate(model, x_tr, y_tr, l_tr, exp.loss)
    loss_te, acc_te = evaluate(model, x_te, y_te, l_te, exp.loss)
    return {"remaining": n_remaining, "train_loss": loss_tr, "test_loss": loss_te,
            "train_acc": acc_tr, "test_acc": acc_te}


# --------------------------------------------------------------------------
# Experiment 1 (Figs 2 & 3): delete weights without retraining
# --------------------------------------------------------------------------

def sweep_no_retrain(model, splits, rank_by, exp, recompute=False):
    """Delete increasing numbers of weights without retraining (Figs 2, 3).

    With recompute=False the ranking is computed once on the trained net.
    With recompute=True the scores are recomputed on the pruned net before
    each further deletion - still no retraining. The distinction matters:
    the saliency is a local second-order estimate, so a ranking computed at
    the full network goes stale as weights are deleted. (Magnitude ranking is
    unaffected - deletion doesn't change the surviving |w|.)
    """
    x_tr = splits[0][0]
    label = rank_by + ("-recomputed" if recompute else "")
    pruned, names, masks, n_total = _setup(model, exp)
    scores = scores_for(pruned, x_tr, rank_by, exp)

    base = _point(model, splits, exp, n_total)
    results = [base]
    predicted = 0.0
    for n_remaining in sweep_targets(n_total, exp):
        if recompute and len(results) > 1:
            scores = scores_for(pruned, x_tr, rank_by, exp)
        before = flatten(masks, names) > 0
        masks = prune_to(masks, scores, n_remaining, names)
        newly_deleted = before & (flatten(masks, names) == 0)
        if rank_by.startswith("saliency"):
            # the second-order forecast of the damage: sum of deleted saliencies
            predicted += flatten(scores, names)[newly_deleted].sum().item()
        apply_masks(pruned, masks)

        point = _point(pruned, splits, exp, n_remaining)
        if rank_by.startswith("saliency"):
            point["predicted_increase"] = predicted
            point["actual_increase"] = point["train_loss"] - base["train_loss"]
        results.append(point)
        print(f"  [{label}] {n_remaining:7d} weights left  "
              f"train loss {point['train_loss']:.4f}")
    return results


# --------------------------------------------------------------------------
# Experiment 2 (Fig 4): the full OBD loop
# --------------------------------------------------------------------------

def iterative_prune_retrain(model, splits, exp, rank_by="saliency"):
    """Prune, retrain, repeat (Fig 4).

    rank_by="saliency" is OBD proper; rank_by="magnitude" is the control
    that isolates how much of Fig 4's result is the saliency ranking and
    how much is just prune-a-little-retrain-a-little.
    """
    x_tr, y_tr = splits[0][0], splits[0][1]
    model, names, masks, n_remaining = _setup(model, exp)

    results = []
    while n_remaining >= exp.retrain_min_remaining:
        point = _point(model, splits, exp, n_remaining)
        results.append(point)
        print(f"  [retrain-{rank_by}] {n_remaining:7d} weights left  "
              f"train loss {point['train_loss']:.4f}  "
              f"test loss {point['test_loss']:.4f}  test acc {point['test_acc']:.3f}")

        n_remaining = int(n_remaining * exp.shrink)
        scores = scores_for(model, x_tr, rank_by, exp)
        masks = prune_to(masks, scores, n_remaining, names)
        apply_masks(model, masks)
        train(model, x_tr, y_tr, exp, epochs=exp.retrain_epochs, masks=masks)
    return results


# --------------------------------------------------------------------------
# Experiment 3 (Figs 5-7): does OBD just rediscover magnitude?
#
# OBD scores a weight by s_k = 1/2 h_kk w_k^2; magnitude pruning scores it by
# w_k^2. The two rankings are *identical* whenever h_kk is constant, so
# magnitude pruning is not an approximation to OBD - it is OBD under an
# isotropic-curvature prior. Everything below measures how far from constant
# h_kk actually is, and where that variation lives.
#
# In logs the saliency decomposes additively,
#
#     log s_k = 2 log|w_k| + log(h_kk / 2),
#
# so with variances sigma_w^2, sigma_h^2 and covariance sigma_hw,
#
#     corr(log s, log|w|) = (2 sigma_w^2 + sigma_hw)
#                           / (sigma_w sqrt(4 sigma_w^2 + sigma_h^2 + 4 sigma_hw))
#
# The two rankings agree closely when sigma_h << 2 sigma_w, and positive
# covariance (which weight decay induces - see README) pushes agreement higher
# still. `between_layer_frac` then asks where sigma_h^2 comes from: if most of
# it is differences between layer means rather than spread within a layer,
# the curvature term acts as a per-layer offset, and OBD's only real advantage
# over magnitude is how it splits the pruning budget across layers.
# --------------------------------------------------------------------------

SCATTER_POINTS = 4000   # subsampled for fig6; the full vectors are far too many
KENDALL_POINTS = 2048   # tau is O(n^2), so it gets a subsample; Spearman is exact


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float(a.dot(b) / denom) if denom > 0 else float("nan")


def _ranks(v):
    order = torch.argsort(v)
    ranks = torch.empty_like(v)
    ranks[order] = torch.arange(len(v), dtype=v.dtype, device=v.device)
    return ranks


def _spearman(a, b):
    """Rank correlation. Exact ties are measure-zero for these floats, so the
    ordinal ranking above is enough - no tie-correction term."""
    if len(a) < 2:
        return float("nan")
    return _pearson(_ranks(a), _ranks(b))


def _kendall_subsample(a, b, generator):
    """Kendall's tau-a on a random subsample (the full statistic is O(n^2))."""
    n = min(KENDALL_POINTS, len(a))
    if n < 2:
        return float("nan")
    idx = torch.randperm(len(a), generator=generator)[:n]
    a, b = a[idx], b[idx]
    concordant = torch.sign(a[:, None] - a[None, :]) * torch.sign(b[:, None] - b[None, :])
    return float(concordant.sum() / (n * (n - 1)))


def _json_safe(value):
    """Replace non-finite floats with null so the results file stays valid JSON.

    They only arise in degenerate corners (a layer with one live weight, the
    unpruned baseline point where the "pruned set" is empty), but json.dumps
    would otherwise emit a bare `NaN` that no strict parser accepts.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _layer_index(masks, names):
    """Per-entry layer id, aligned with flatten(..., names)."""
    return torch.cat([torch.full((masks[name].numel(),), i, dtype=torch.long)
                      for i, name in enumerate(names)])


def overlap_analysis(model, splits, exp):
    """Compare the OBD and magnitude rankings of the trained network.

    Everything is computed once, at the trained weights, over the live
    prunable entries (structurally absent connection-table entries excluded).
    """
    x_tr = splits[0][0]
    names = prunable_names(model)
    params = dict(model.named_parameters())
    masks = initial_masks(model)

    print("  [overlap] computing the diagonal Hessian...")
    h_by_name = diagonal_hessian(model, x_tr, loss=exp.loss,
                                 max_samples=exp.hessian_samples, names=names)

    live = (flatten(masks, names) > 0).cpu()
    layer = _layer_index(masks, names)[live]
    w = flatten({n: params[n].detach().abs() for n in names}, names).cpu().double()[live]
    h = flatten(h_by_name, names).cpu().double()[live]
    s = 0.5 * h * w.pow(2)
    n_total = int(live.sum().item())

    # log-space statistics need strictly positive entries; a dead ReLU path or
    # an exactly-zero weight has no logarithm and is reported, not silently kept
    ok = (w > 0) & (h > 0)
    n_dropped = int((~ok).sum().item())
    log_w, log_h, log_s = w[ok].log(), h[ok].log(), s[ok].log()
    layer_ok = layer[ok]

    var_w = float(log_w.var(unbiased=False))
    var_h = float(log_h.var(unbiased=False))
    cov = float(((log_h - log_h.mean()) * (log_w - log_w.mean())).mean())
    # the identity above, evaluated from the moments; must match pearson_log
    var_s = 4 * var_w + var_h + 4 * cov
    predicted = ((2 * var_w + cov) / math.sqrt(var_w * var_s)
                 if var_w > 0 and var_s > 0 else float("nan"))

    pearson_log = _pearson(log_s, log_w)
    spearman = _spearman(s, w)
    # what the rank correlation would be if (log h, log|w|) were jointly
    # Gaussian; the gap to `spearman` is a measure of how heavy the tails are
    gaussian_estimate = (6.0 / math.pi) * math.asin(max(-1.0, min(1.0, pearson_log)) / 2.0)

    # where does the curvature variance live: between layers, or within them?
    layer_means = torch.stack([log_h[layer_ok == i].mean()
                               for i in range(len(names))
                               if int((layer_ok == i).sum()) > 0])
    layer_sizes = torch.tensor([float((layer_ok == i).sum())
                                for i in range(len(names))
                                if int((layer_ok == i).sum()) > 0], dtype=torch.double)
    grand = (layer_means * layer_sizes).sum() / layer_sizes.sum()
    between = float((layer_sizes * (layer_means - grand).pow(2)).sum() / layer_sizes.sum())

    per_layer = []
    for i, name in enumerate(names):
        sel = layer == i
        sel_ok = layer_ok == i
        if int(sel.sum()) == 0:
            continue
        per_layer.append({
            "name": name,
            "n": int(sel.sum().item()),
            "spearman_within": _spearman(s[sel], w[sel]),
            "log_h_mean": float(log_h[sel_ok].mean()) if int(sel_ok.sum()) else float("nan"),
            "log_h_std": float(log_h[sel_ok].std(unbiased=False)) if int(sel_ok.sum()) > 1 else 0.0,
            "log_w_std": float(log_w[sel_ok].std(unbiased=False)) if int(sel_ok.sum()) > 1 else 0.0,
            "h_median": float(h[sel].median()),
        })

    # how much of each ranking's kept set is shared, at every pruning level
    overlap_curve, allocation = [], []
    for n_remaining in [n_total] + sweep_targets(n_total, exp):
        keep_obd = keep_mask(s, n_remaining)
        keep_mag = keep_mask(w, n_remaining)
        shared = int((keep_obd & keep_mag).sum().item())
        union = int((keep_obd | keep_mag).sum().item())
        dead_shared = int((~keep_obd & ~keep_mag).sum().item())
        dead_union = int((~keep_obd | ~keep_mag).sum().item())
        overlap_curve.append({
            "remaining": n_remaining,
            "overlap_frac": shared / n_remaining,
            "jaccard_kept": shared / union if union else float("nan"),
            "jaccard_pruned": dead_shared / dead_union if dead_union else float("nan"),
            "chance": n_remaining / n_total,
        })
        allocation.append({
            "remaining": n_remaining,
            "kept_obd": {names[i]: int((keep_obd & (layer == i)).sum().item())
                         for i in range(len(names))},
            "kept_mag": {names[i]: int((keep_mag & (layer == i)).sum().item())
                         for i in range(len(names))},
            "layer_size": {names[i]: int((layer == i).sum().item())
                           for i in range(len(names))},
        })
        print(f"  [overlap] {n_remaining:7d} weights left  "
              f"shared {shared / n_remaining:.3f}  (chance {n_remaining / n_total:.3f})")

    generator = torch.Generator().manual_seed(0)
    kendall = _kendall_subsample(s, w, generator)
    pick = torch.randperm(int(ok.sum().item()), generator=generator)[:SCATTER_POINTS]

    return _json_safe({
        "n_prunable": n_total,
        "n_dropped_zero": n_dropped,
        "layers": names,
        "global": {
            "spearman": spearman,
            "spearman_gaussian_estimate": gaussian_estimate,
            "kendall_subsample": kendall,
            "pearson_log": pearson_log,
            "predicted_pearson": predicted,
            "var_log_h": var_h,
            "var_log_w": var_w,
            "cov_log_h_log_w": cov,
            "between_layer_frac": between / var_h if var_h > 0 else float("nan"),
            "weight_decay": exp.weight_decay,
        },
        "per_layer": per_layer,
        "overlap_curve": overlap_curve,
        "layer_allocation": allocation,
        "scatter": {
            "layer": layer_ok[pick].tolist(),
            "log_w": log_w[pick].tolist(),
            "log_s": log_s[pick].tolist(),
        },
    })


# --------------------------------------------------------------------------
# The experiments run_experiments() can run, keyed as in results/<name>.json.
# --------------------------------------------------------------------------

PRUNE_EXPERIMENTS = {
    "magnitude":
        lambda m, s, e: sweep_no_retrain(m, s, "magnitude", e),
    "saliency":
        lambda m, s, e: sweep_no_retrain(m, s, "saliency", e),
    "saliency_recomputed":
        lambda m, s, e: sweep_no_retrain(m, s, "saliency", e, recompute=True),
    "saliency_layermean":
        lambda m, s, e: sweep_no_retrain(m, s, "saliency_layermean", e),
    "saliency_layermean_recomputed":
        lambda m, s, e: sweep_no_retrain(m, s, "saliency_layermean", e, recompute=True),
    "retrain":
        lambda m, s, e: iterative_prune_retrain(m, s, e, "saliency"),
    "retrain_magnitude":
        lambda m, s, e: iterative_prune_retrain(m, s, e, "magnitude"),
    "overlap":
        lambda m, s, e: overlap_analysis(m, s, e),
}
