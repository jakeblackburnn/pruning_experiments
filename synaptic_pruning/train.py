"""One training run: build a model, train it under one of the four
regularization methods the paper compares, evaluate MAE. This is the unit
both experiments (replication grid, bitter-lesson grid) sweep over.

Methods, matching the paper's four-way comparison:

    none        no dropout, no pruning — the unregularized baseline
    dropout     standard dropout (disabled at eval)
    mc_dropout  same dropout module, kept active at eval, averaged over K
                stochastic forward passes
    pruning     no dropout; a SynapticPruner (pruning.py) permanently zeros
                the globally-smallest-magnitude weights on the paper's
                cubic schedule
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from models import build
from pruning import PruneConfig, SynapticPruner

METHODS = ("none", "dropout", "mc_dropout", "pruning")
DROPOUT_RATE = 0.3  # paper doesn't publish its dropout baseline rate; 0.3
                     # matches smin so the two methods start from a
                     # comparable sparsity floor
MC_SAMPLES = 20


@dataclass
class RunConfig:
    model_type: str = "lstm"       # "rnn" | "lstm"
    method: str = "none"           # one of METHODS
    hidden_size: int = 32
    num_layers: int = 1
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    prune_smin: float = 0.3
    prune_smax: float = 0.7
    prune_warmup_epochs: int = 2
    prune_every: int = 5
    prune_schedule_epochs: int = 20  # ramp horizon, independent of `epochs`
                                      # (see pruning.py:PruneConfig)


def _batches(x, y, batch_size, generator):
    n = x.shape[0]
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield x[idx], y[idx]


@torch.no_grad()
def _mae(model, x, y):
    model.eval()
    pred = model(x)
    return (pred - y).abs().mean().item()


@torch.no_grad()
def predict_mc(model, x, k=MC_SAMPLES):
    """MC Dropout: force dropout active at eval time, average k stochastic
    forward passes."""
    model.train()  # keeps nn.Dropout stochastic
    preds = torch.stack([model(x) for _ in range(k)], dim=0)
    return preds.mean(dim=0)


@torch.no_grad()
def _mae_mc(model, x, y, k=MC_SAMPLES):
    pred = predict_mc(model, x, k=k)
    return (pred - y).abs().mean().item()


def train_one(cfg: RunConfig, x_tr, y_tr, x_te, y_te, verbose=False):
    """Train one model under one method. Returns a dict with the loss curve,
    per-epoch test MAE, final test MAE, and (for pruning) the sparsity
    trajectory and final sparsity stats."""
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    dropout = DROPOUT_RATE if cfg.method in ("dropout", "mc_dropout") else 0.0
    model = build(cfg.model_type, x_tr.shape[-1], hidden_size=cfg.hidden_size,
                  num_layers=cfg.num_layers, dropout=dropout).to(cfg.device)
    x_tr, y_tr = x_tr.to(cfg.device), y_tr.to(cfg.device)
    x_te, y_te = x_te.to(cfg.device), y_te.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.L1Loss()

    pruner = None
    if cfg.method == "pruning":
        pcfg = PruneConfig(smin=cfg.prune_smin, smax=cfg.prune_smax,
                            warmup_epochs=cfg.prune_warmup_epochs,
                            schedule_epochs=cfg.prune_schedule_epochs,
                            prune_every=cfg.prune_every)
        pruner = SynapticPruner(model, pcfg)

    history, sparsity_trace = [], []
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in _batches(x_tr, y_tr, cfg.batch_size, generator):
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if pruner is not None:
                pruner.step(epoch + 1)  # paper's schedule is 1-indexed (1..total_epochs)
            optimizer.step()
            if pruner is not None:
                pruner.apply_masks()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(1, n_batches)
        test_mae = (_mae_mc(model, x_te, y_te) if cfg.method == "mc_dropout"
                    else _mae(model, x_te, y_te))
        record = {"epoch": epoch, "train_loss": train_loss, "test_mae": test_mae}
        if pruner is not None:
            sparsity = pruner.sparsity_stats()["_overall"]["sparsity"]
            record["sparsity"] = sparsity
            sparsity_trace.append(sparsity)
        history.append(record)
        if verbose:
            print(f"  epoch {epoch:3d}  train_loss {train_loss:.4f}  "
                  f"test_mae {test_mae:.4f}"
                  + (f"  sparsity {record['sparsity']:.2f}" if pruner else ""))

    result = {"history": history, "final_test_mae": history[-1]["test_mae"]}
    if pruner is not None:
        result["sparsity_trace"] = sparsity_trace
        result["sparsity_stats"] = pruner.sparsity_stats()
    return result
