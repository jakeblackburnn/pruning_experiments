"""The factorial design: which runs exist, and in what order they execute.

A run is a `Unit`: one model trained under one method. Axes: dataset, arch,
width (model size), train_frac (data), epochs (compute), seq_len, method and
the method's own parameters; seeds are the blocking factor.

Three blocks, overlapping where they coincide (deduplicated by key):

  core   dataset x arch x seq_len x method    at the default width, data, epochs
  scale  dataset x SCALE_ARCHS x a star around the centre (width, train_frac,
         epochs one at a time) x SCALE_METHODS; width x method is crossed,
         which is the interaction the scale question asks about, and
         random_pruning runs along width as the capacity control
  sweep  dataset x arch x method parameters   6 pruning variants (the paper's,
                                              one factor off it, one corner),
                                              dropout rate, weight decay

`ordered()` yields every unit of seed 0, then seed 1, ..., shuffled within a
seed, so stopping at any point leaves complete seeds and a balanced design.
"""

import dataclasses
import hashlib
import json
import random

from datasets import DATASETS
from models import ARCHS
from train import Unit

BASE_EPOCHS, BASE_WIDTH, BASE_SEQ = 20, 64, 14
SEQ_LENS = (14, 60)
WIDTHS = (16, 64, 256)
TRAIN_FRACS = (0.1, 0.3)       # off-centre levels of the scale star
EPOCH_MULTS = (0.5, 3.0)
SCALE_ARCHS = ("lstm", "cnn", "transformer")
SWEEP_ARCHS = ("lstm", "cnn")
BLOCKS = ("core", "scale", "sweep")
SEEDS = 4

# name -> the method-specific fields; the defaults of the paper's setting
CORE_METHODS = {
    "none": {}, "dropout": {"dropout": 0.3}, "mc_dropout": {"dropout": 0.3},
    "l2": {"weight_decay": 1e-3}, "pruning": {}, "random_pruning": {},
    "oneshot": {}, "narrow": {},
}
SCALE_METHODS = ("none", "dropout", "pruning")


def _unit(dataset, arch, seed, method, params=None, width=BASE_WIDTH, train_frac=1.0,
          epoch_mult=1.0, seq_len=BASE_SEQ):
    return Unit(dataset=dataset, arch=arch, width=width, train_frac=train_frac,
                epochs=max(1, round(BASE_EPOCHS * epoch_mult)), seq_len=seq_len,
                method=method, seed=seed, **(params if params is not None else CORE_METHODS[method]))


def block_units(block, seed):
    if block == "core":
        return [_unit(d, a, seed, m, seq_len=s) for d in DATASETS for a in ARCHS
                for s in SEQ_LENS for m in CORE_METHODS]
    if block == "scale":
        return ([_unit(d, a, seed, m, width=w) for d in DATASETS for a in SCALE_ARCHS
                 for w in WIDTHS for m in SCALE_METHODS + ("random_pruning",)]
                + [_unit(d, a, seed, m, train_frac=f) for d in DATASETS for a in SCALE_ARCHS
                   for f in TRAIN_FRACS for m in SCALE_METHODS]
                + [_unit(d, a, seed, m, epoch_mult=e) for d in DATASETS for a in SCALE_ARCHS
                   for e in EPOCH_MULTS for m in SCALE_METHODS])
    if block == "sweep":
        variants = [("pruning", {"smin": lo, "smax": hi, "horizon": h})
                    for lo, hi, h in ((0.3, 0.7, 0), (0.0, 0.7, 0), (0.3, 0.5, 0),
                                      (0.3, 0.9, 0), (0.3, 0.7, 10), (0.0, 0.9, 10))]
        variants += [("dropout", {"dropout": p}) for p in (0.1, 0.5)]
        variants += [("l2", {"weight_decay": w}) for w in (1e-4, 1e-2)]
        return [_unit(d, a, seed, m, params) for d in DATASETS for a in SWEEP_ARCHS
                for m, params in variants]
    raise ValueError(f"unknown block {block!r}; choose from {BLOCKS}")


def key(unit):
    """Stable identity of a unit's configuration (includes overrides)."""
    blob = json.dumps(dataclasses.asdict(unit), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def all_units(blocks, seeds, overrides=None, only=None):
    """Every distinct unit of the chosen blocks and seeds, with `overrides`
    (a dict of Unit fields) applied. `only` maps a Unit field to the allowed
    values and filters the design."""
    only = only or {}
    seen, out = set(), []
    for seed in range(seeds):
        for block in blocks:
            for u in block_units(block, seed):
                if any(getattr(u, f) not in v for f, v in only.items() if v):
                    continue
                u = dataclasses.replace(u, **(overrides or {}))
                if key(u) not in seen:
                    seen.add(key(u))
                    out.append(u)
    return out


def ordered(units):
    """Seed-major order, deterministic shuffle within a seed."""
    out = []
    for seed in sorted({u.seed for u in units}):
        chunk = [u for u in units if u.seed == seed]
        random.Random(seed).shuffle(chunk)
        out.extend(chunk)
    return out
