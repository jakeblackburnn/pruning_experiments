"""One training run (a `Unit`): build a model, train it under one
regularization method, and record validation and test MAE after every epoch.

Methods:

    none            no regularization: the baseline
    dropout         dropout on the last hidden state (off at eval)
    mc_dropout      the same module kept on at eval, averaged over K passes
    l2              decoupled weight decay (AdamW)
    pruning         synaptic pruning: the paper's cubic schedule, global magnitude
    random_pruning  the same schedule and sparsity, but weights chosen at random
    oneshot         train dense for half the epochs, prune once to smax, fine-tune
    narrow          a dense net narrowed to the parameter count `pruning` leaves

The headline number is the test MAE at the epoch with the best validation
MAE (early-stopping model selection; the test set never picks anything).
The final-epoch test MAE is recorded too.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from datasets import load
from models import build, count_prunable_params
from pruning import PruneConfig, SynapticPruner

METHODS = ("none", "dropout", "mc_dropout", "l2", "pruning", "random_pruning",
           "oneshot", "narrow")
MC_SAMPLES = 20


@dataclass(frozen=True)
class Unit:
    """One run. Fields a method does not use are held at their defaults, so a
    unit's key identifies exactly what was run."""
    dataset: str
    arch: str
    width: int
    train_frac: float
    epochs: int
    seq_len: int
    method: str
    seed: int
    dropout: float = 0.0        # dropout / mc_dropout
    weight_decay: float = 0.0   # l2
    smin: float = 0.3           # pruning family and narrow
    smax: float = 0.7
    horizon: int = 0            # cubic ramp length in epochs; 0 = the whole run
    warmup: int = 2
    prune_every: int = 5
    lr: float = 1e-3
    batch_size: int = 64
    n_cap: int | None = None    # cap on windows per split (smoke tests)


def narrow_width(width, smax):
    """Width whose weight-matrix parameter count matches a width-`width` net at
    sparsity smax (weights scale with width squared), in multiples of 4."""
    return max(4, 4 * round(width * math.sqrt(1.0 - smax) / 4))


def _batches(x, y, batch_size, generator):
    perm = torch.randperm(x.shape[0], generator=generator).to(x.device)
    for i in range(0, x.shape[0], batch_size):
        idx = perm[i:i + batch_size]
        yield x[idx], y[idx]


@torch.no_grad()
def _mae(model, x, y, mc=False):
    if mc:  # MC dropout: dropout stays active, average MC_SAMPLES passes
        model.train()
        pred = torch.stack([model(x) for _ in range(MC_SAMPLES)]).mean(dim=0)
    else:
        model.eval()
        pred = model(x)
    return (pred - y).abs().mean().item()


def train_one(unit, device, verbose=False):
    """Train one model. Returns the record written to the log (minus identity)."""
    torch.manual_seed(unit.seed)
    generator = torch.Generator().manual_seed(unit.seed)
    data = load(unit.dataset, unit.seq_len, unit.train_frac, unit.n_cap)
    x_tr, y_tr = (torch.from_numpy(a).to(device) for a in data.train)
    x_va, y_va = (torch.from_numpy(a).to(device) for a in data.val)
    x_te, y_te = (torch.from_numpy(a).to(device) for a in data.test)

    width = narrow_width(unit.width, unit.smax) if unit.method == "narrow" else unit.width
    model = build(unit.arch, data.n_features, width, unit.seq_len,
                  dropout=unit.dropout if unit.method in ("dropout", "mc_dropout") else 0.0).to(device)
    mc = unit.method == "mc_dropout"
    optimizer = torch.optim.AdamW(model.parameters(), lr=unit.lr,
                                  weight_decay=unit.weight_decay if unit.method == "l2" else 0.0)
    loss_fn = nn.L1Loss()

    pruner = None
    if unit.method in ("pruning", "random_pruning", "oneshot"):
        pruner = SynapticPruner(model, PruneConfig(
            smin=unit.smin, smax=unit.smax, warmup_epochs=unit.warmup,
            schedule_epochs=unit.horizon or unit.epochs, prune_every=unit.prune_every,
            kind="oneshot" if unit.method == "oneshot" else "cubic",
            select="random" if unit.method == "random_pruning" else "magnitude",
            oneshot_epoch=unit.epochs // 2 + 1))

    history = {"train_loss": [], "val_mae": [], "test_mae": [], "sparsity": []}
    for epoch in range(unit.epochs):
        model.train()
        total, n_batches = 0.0, 0
        for xb, yb in _batches(x_tr, y_tr, unit.batch_size, generator):
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if pruner is not None:
                pruner.step(epoch + 1)
            optimizer.step()
            if pruner is not None:
                pruner.apply_masks()
            total += loss.item()
            n_batches += 1
        history["train_loss"].append(total / max(1, n_batches))
        history["val_mae"].append(_mae(model, x_va, y_va, mc))
        history["test_mae"].append(_mae(model, x_te, y_te, mc))
        if pruner is not None:
            history["sparsity"].append(pruner.sparsity_stats()["_overall"]["sparsity"])
        if verbose:
            print(f"  epoch {epoch:3d}  train {history['train_loss'][-1]:.4f}  "
                  f"val {history['val_mae'][-1]:.4f}  test {history['test_mae'][-1]:.4f}")

    best = min(range(unit.epochs), key=history["val_mae"].__getitem__)
    return {
        "history": {k: [round(v, 5) for v in vs] for k, vs in history.items() if vs},
        "test_mae": history["test_mae"][best], "val_mae": history["val_mae"][best],
        "best_epoch": best + 1,
        "test_mae_final": history["test_mae"][-1], "val_mae_final": history["val_mae"][-1],
        "persistence": data.persistence, "n_train": len(y_tr),
        "n_prunable": count_prunable_params(model),
        "sparsity": history["sparsity"][-1] if history["sparsity"] else 0.0,
    }
