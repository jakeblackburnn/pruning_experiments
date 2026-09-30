"""The pruning experiments, run on one trained network (a `Unit`).

  sweep_no_retrain          delete more and more weights, no retraining
  iterative_prune_retrain   prune a little, retrain a little, repeat
  overlap_analysis          how much do the OBD and magnitude rankings agree,
                            and where does the curvature term carry information?

All experiments visit the same `FRACTIONS` of the prunable weights, so curves
from different architectures and datasets share a grid. The prune-retrain loop
takes fewer, larger steps (`RETRAIN_FRACTIONS`, a subset): it is most of a
unit's time, and above 8% kept the criteria do not differ. The core math (models,
diagonal Hessian, saliencies, mask primitives) lives in obd.py.
"""

import copy
import math

import torch

from obd import (apply_masks, diagonal_hessian, evaluate, flatten, initial_masks,
                 keep_mask, prunable_names, prune_to, scores_for, train)

# fractions of the prunable weights kept, in the order they are visited
FRACTIONS = (0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005)
RETRAIN_FRACTIONS = (0.5, 0.25, 0.12, 0.05, 0.02, 0.01, 0.005)

# criteria run in each experiment
SWEEP_CRITERIA = (("magnitude", False), ("random", False), ("taylor", False),
                  ("saliency", False), ("saliency", True),
                  ("saliency_layermean", False))
RETRAIN_CRITERIA = ("magnitude", "random", "taylor", "saliency_layermean", "saliency")


def _r(x):
    return round(float(x), 6)


def sweep_targets(n_total, fractions=FRACTIONS):
    """Weight counts to prune down to, largest first (all < n_total)."""
    counts = [max(1, round(f * n_total)) for f in fractions]
    out = []
    for n in counts:
        if n < n_total and (not out or n < out[-1]):
            out.append(n)
    return out


def _setup(model):
    """A prunable copy of the model plus its mask bookkeeping."""
    pruned = copy.deepcopy(model)
    names = prunable_names(pruned)
    masks = initial_masks(pruned)
    n_total = int(flatten(masks, names).sum().item())
    return pruned, names, masks, n_total


def _point(model, splits, unit, n_remaining, n_total, which=("train", "val", "test")):
    """Loss and accuracy on the `which` splits for one pruning level."""
    point = {"remaining": n_remaining, "keep": _r(n_remaining / n_total)}
    for split in which:
        x, y, labels = splits[split]
        loss, acc = evaluate(model, x, y, labels, unit.loss)
        point[f"{split}_loss"], point[f"{split}_acc"] = _r(loss), _r(acc)
    return point


def _train_data(splits):
    return splits["train"][0], splits["train"][1]


# --------------------------------------------------------------------------
# Delete weights without retraining
# --------------------------------------------------------------------------

def sweep_no_retrain(model, splits, rank_by, unit, recompute=False):
    """Delete increasing numbers of weights without retraining.

    With recompute=False the ranking is computed once on the trained net.
    With recompute=True the scores are recomputed on the pruned net before
    each further deletion - still no retraining. The distinction matters:
    the saliency is a local second-order estimate, so a ranking computed at
    the full network goes stale as weights are deleted.
    """
    data = _train_data(splits)
    pruned, names, masks, n_total = _setup(model)
    scores = scores_for(pruned, data, rank_by, unit)

    base = _point(model, splits, unit, n_total, n_total)
    results = [base]
    predicted = 0.0
    for n_remaining in sweep_targets(n_total):
        if recompute and len(results) > 1:
            scores = scores_for(pruned, data, rank_by, unit)
        before = flatten(masks, names) > 0
        masks = prune_to(masks, scores, n_remaining, names)
        newly_deleted = before & (flatten(masks, names) == 0)
        if rank_by.startswith("saliency"):
            # the second-order forecast of the damage: sum of deleted saliencies
            predicted += flatten(scores, names)[newly_deleted].sum().item()
        apply_masks(pruned, masks)

        point = _point(pruned, splits, unit, n_remaining, n_total)
        if rank_by.startswith("saliency"):
            point["predicted_increase"] = predicted
            point["actual_increase"] = _r(point["train_loss"] - base["train_loss"])
        results.append(point)
    return results


# --------------------------------------------------------------------------
# The prune-retrain loop
# --------------------------------------------------------------------------

def iterative_prune_retrain(model, splits, unit, rank_by="saliency"):
    """Prune to the next fraction, retrain, repeat.

    Each level records the damage right after pruning (`pre_*`) and the state
    after retraining. rank_by="magnitude" is the control that isolates how
    much of the result is the ranking and how much is just
    prune-a-little-retrain-a-little; "random" is the floor.
    """
    x_tr, y_tr = _train_data(splits)
    model, names, masks, n_total = _setup(model)

    results = [_point(model, splits, unit, n_total, n_total)]
    for n_remaining in sweep_targets(n_total, RETRAIN_FRACTIONS):
        scores = scores_for(model, (x_tr, y_tr), rank_by, unit)
        masks = prune_to(masks, scores, n_remaining, names)
        apply_masks(model, masks)
        pre = _point(model, splits, unit, n_remaining, n_total, which=("val", "test"))
        train(model, x_tr, y_tr, unit, epochs=unit.retrain_epochs, masks=masks)
        point = _point(model, splits, unit, n_remaining, n_total)
        for k in ("val_loss", "val_acc", "test_loss", "test_acc"):
            point["pre_" + k] = pre[k]
        results.append(point)
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
# covariance (which weight decay induces) pushes agreement higher
# still. `between_layer_frac` then asks where sigma_h^2 comes from: if most of
# it is differences between layer means rather than spread within a layer,
# the curvature term acts as a per-layer offset, and OBD's only real advantage
# over magnitude is how it splits the pruning budget across layers.
# --------------------------------------------------------------------------

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


def overlap_analysis(model, splits, unit):
    """Compare the OBD and magnitude rankings of the trained network.

    Everything is computed once, at the trained weights, over the live
    prunable entries (structurally absent connection-table entries excluded).
    """
    x_tr = splits["train"][0]
    names = prunable_names(model)
    params = dict(model.named_parameters())
    masks = initial_masks(model)

    h_by_name = diagonal_hessian(model, x_tr, loss=unit.loss,
                                 max_samples=unit.hessian_samples, names=names)

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
    for n_remaining in [n_total] + sweep_targets(n_total):
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

    generator = torch.Generator().manual_seed(0)
    kendall = _kendall_subsample(s, w, generator)
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
            "weight_decay": unit.weight_decay,
        },
        "per_layer": per_layer,
        "overlap_curve": overlap_curve,
        "layer_allocation": allocation,
    })
