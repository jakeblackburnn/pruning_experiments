"""The factorial design: which runs exist, and in what order they execute.

A run is a `Unit`: one model trained under one method. Axes: dataset, arch,
width (model size), train_frac (data), epochs (compute), seq_len, method and
the method's own parameters; seeds are the blocking factor.

Three blocks, each crossed within itself, overlapping where they coincide
(deduplicated by key):

  core   dataset x arch x seq_len x method    at the default width, data, epochs
  scale  dataset x arch x width x train_frac x epochs x method
                                              for a subset of archs and methods
  sweep  dataset x arch x method parameters   pruning smin x smax x horizon,
                                              dropout rate, weight decay

`ordered()` yields every unit of seed 0, then seed 1, ..., shuffled within a
seed, so stopping at any point leaves complete seeds and a balanced design.
"""

import dataclasses
import hashlib
import json
import random
import statistics

from datasets import DATASETS
from models import ARCHS
from train import Unit

BASE_EPOCHS, BASE_WIDTH, BASE_SEQ = 20, 64, 14
SEQ_LENS = (1, 14, 60)
WIDTHS = (16, 64, 256)
TRAIN_FRACS = (0.1, 0.3, 1.0)
EPOCH_MULTS = (0.5, 1.0, 3.0)
SCALE_ARCHS = ("lstm", "cnn", "transformer")
SWEEP_ARCHS = ("lstm", "cnn")
BLOCKS = ("core", "scale", "sweep")

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
        return [_unit(d, a, seed, m, width=w, train_frac=f, epoch_mult=e)
                for d in DATASETS for a in SCALE_ARCHS for w in WIDTHS
                for f in TRAIN_FRACS for e in EPOCH_MULTS for m in SCALE_METHODS]
    if block == "sweep":
        variants = [("pruning", {"smin": lo, "smax": hi, "horizon": h})
                    for lo in (0.0, 0.3) for hi in (0.5, 0.7, 0.9) for h in (0, 10)]
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


class CostModel:
    """Expected seconds per unit, learned from completed records.

    The estimate is the median duration of finished units with the same
    (dataset, arch, width), scaled by train_frac * epochs (the work per unit);
    it falls back to the same (dataset, arch), then to None (unknown).
    """

    def __init__(self, records=()):
        self.by_group, self.by_arch = {}, {}
        for r in records:
            self.add(r)

    @staticmethod
    def _work(u):
        return u.train_frac * u.epochs

    def add(self, record):
        u, d = Unit(**record["unit"]), record["_meta"]["duration_s"]
        self.by_group.setdefault((u.dataset, u.arch, u.width), []).append((self._work(u), d))
        self.by_arch.setdefault((u.dataset, u.arch), []).append((self._work(u), d))

    def estimate(self, unit):
        for peers in (self.by_group.get((unit.dataset, unit.arch, unit.width)),
                      self.by_arch.get((unit.dataset, unit.arch))):
            if peers:
                # duration per unit of work, so a median over mixed work is fair
                per_work = statistics.median(d / max(w, 1e-9) for w, d in peers)
                return per_work * self._work(unit)
        return None
