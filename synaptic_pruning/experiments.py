"""The two experiment grids.

replication     as close a reproduction of the paper's method-comparison
                 as this machine and one publicly downloadable dataset
                 reasonably allow: RNN and LSTM, all four regularization
                 methods, the paper's own sequence-length sweep, on the full
                 Air Quality series.

bitter_lesson    a small ladder of individually large experiments — not a
                 grid. Each tier scales network size (LSTM hidden width),
                 compute (training epochs), and dataset size (training rows)
                 together, roughly geometrically, comparing pruning against
                 the unregularized baseline at every tier, to see whether
                 the marginal benefit of synaptic pruning grows, holds, or
                 shrinks as real compute scales up. If it shrinks — "bitter
                 lesson pilled" — the method looks more like a small-data/
                 small-model band-aid than a technique worth scaling.

Both write plain dicts (JSON-able) — disk I/O and caching live in main.py.
"""

import itertools
from dataclasses import dataclass, field

import torch

import datasets
from models import resolve_device
from stats import friedman, summarize
from train import METHODS, RunConfig, train_one


def _tensors(seq_len, n_rows=None):
    x_tr, y_tr, x_te, y_te = datasets.load(seq_len=seq_len, n_rows=n_rows)
    return tuple(torch.tensor(a) for a in (x_tr, y_tr, x_te, y_te))


# --------------------------------------------------------------------------
# replication grid
# --------------------------------------------------------------------------

@dataclass
class ReplicationConfig:
    seq_lens: tuple = (1, 14, 60)  # short / medium / long, trimmed from the
                                     # paper's full 6-point sweep to essentials
    model_types: tuple = ("rnn", "lstm")
    methods: tuple = ("none", "dropout", "pruning")  # mc_dropout dropped: a
                                     # supplementary variant, not needed to
                                     # test the central pruning-vs-baseline claim
    trials: int = 10          # matches the paper (was 5) — the old 5-trial
                               # count left the Friedman test underpowered
    epochs: int = 20          # matches the paper's max epochs per trial
    hidden_size: int = 32
    batch_size: int = 64
    lr: float = 1e-3
    prune_schedule_epochs: int = 20  # ramp horizon; equals `epochs` here so
                                       # this is a no-op today, but keeps the
                                       # two knobs independent (see pruning.py)
    device: str = "auto"


def run_replication(cfg: ReplicationConfig = None, progress=True):
    cfg = cfg or ReplicationConfig()
    device = resolve_device(cfg.device)
    runs, summary = [], {}

    for seq_len, model_type in itertools.product(cfg.seq_lens, cfg.model_types):
        x_tr, y_tr, x_te, y_te = _tensors(seq_len)
        key = f"{model_type}_seq{seq_len}"
        method_trials = {}

        for method in cfg.methods:
            trial_maes = []
            for trial in range(cfg.trials):
                if progress:
                    print(f"[replication] {key} {method} trial {trial}...")
                run_cfg = RunConfig(model_type=model_type, method=method,
                                     hidden_size=cfg.hidden_size, epochs=cfg.epochs,
                                     batch_size=cfg.batch_size, lr=cfg.lr,
                                     prune_schedule_epochs=cfg.prune_schedule_epochs,
                                     seed=trial, device=device)
                result = train_one(run_cfg, x_tr, y_tr, x_te, y_te)
                mae = result["final_test_mae"]
                trial_maes.append(mae)
                runs.append({"seq_len": seq_len, "model_type": model_type,
                             "method": method, "seed": trial, "final_test_mae": mae,
                             "sparsity": result.get("sparsity_stats", {}).get(
                                 "_overall", {}).get("sparsity")})
            method_trials[method] = trial_maes

        summary[key] = {
            "seq_len": seq_len, "model_type": model_type,
            "methods": {m: summarize(v) for m, v in method_trials.items()},
            "friedman": friedman(method_trials),
        }

    return {"config": cfg.__dict__, "runs": runs, "summary": summary}


# --------------------------------------------------------------------------
# bitter-lesson ladder: a handful of individually large, jointly-scaled runs
# --------------------------------------------------------------------------

@dataclass
class BitterLessonConfig:
    # Each tier is (hidden_size, epochs, n_rows), scaling roughly
    # geometrically together (~8x more model-weights x epochs per step).
    # Tier 1 matches the replication grid's own hidden_size/epochs — a
    # known-viable starting point, not the destructively-tiny hidden=8
    # regime the old grid showed is capacity-starved by 70% sparsity.
    # n_rows hits the dataset's ~9,357-row ceiling at tier 3 and stays there.
    tiers: tuple = (
        (32, 20, 4000),
        (64, 40, 8000),
        (128, 80, 9000),
        (256, 160, 9000),
        (512, 320, 9000),
        (1024, 640, 9000),
    )
    seq_len: int = 14           # held fixed so only tier scale varies
    model_type: str = "lstm"    # the paper's most consistent performer
    methods: tuple = ("none", "pruning")
    trials: int = 5
    prune_schedule_epochs: int = 10  # deliberately smaller than every
                                       # tier's epoch count (min is 20 at
                                       # tier 1) so every tier gets genuine
                                       # post-ramp fine-tuning time, not a
                                       # ramp that completes on its last epoch
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "auto"


def run_bitter_lesson(cfg: BitterLessonConfig = None, progress=True):
    cfg = cfg or BitterLessonConfig()
    device = resolve_device(cfg.device)
    runs, cells = [], []

    for tier_index, (hidden_size, epochs, n_rows) in enumerate(cfg.tiers):
        x_tr, y_tr, x_te, y_te = _tensors(cfg.seq_len, n_rows=n_rows)
        method_trials = {}

        for method in cfg.methods:
            trial_maes = []
            for trial in range(cfg.trials):
                if progress:
                    print(f"[bitter_lesson] tier {tier_index} hidden={hidden_size} "
                          f"epochs={epochs} n_rows={n_rows} {method} trial {trial}...")
                run_cfg = RunConfig(model_type=cfg.model_type, method=method,
                                     hidden_size=hidden_size, epochs=epochs,
                                     batch_size=cfg.batch_size, lr=cfg.lr,
                                     prune_schedule_epochs=cfg.prune_schedule_epochs,
                                     seed=trial, device=device)
                result = train_one(run_cfg, x_tr, y_tr, x_te, y_te)
                mae = result["final_test_mae"]
                trial_maes.append(mae)
                runs.append({"tier_index": tier_index, "hidden_size": hidden_size,
                             "epochs": epochs, "n_rows": n_rows, "method": method,
                             "seed": trial, "final_test_mae": mae})
            method_trials[method] = trial_maes

        none_stats = summarize(method_trials["none"])
        prune_stats = summarize(method_trials["pruning"])
        improvement = ((none_stats["mean"] - prune_stats["mean"]) /
                        none_stats["mean"] * 100.0 if none_stats["mean"] else 0.0)
        cells.append({
            "tier_index": tier_index, "hidden_size": hidden_size,
            "epochs": epochs, "n_rows": n_rows,
            "none": none_stats, "pruning": prune_stats,
            "improvement_pct": improvement,
        })

    return {"config": cfg.__dict__, "runs": runs, "cells": cells}


EXPERIMENTS = {"replication": run_replication, "bitter_lesson": run_bitter_lesson}
