"""The factorial design: which units exist, and in what order they run.

Factors (per unit): dataset x arch (COMBOS), width, data_frac, epochs (as a
multiple of the dataset's base epochs), weight_decay, retrain_epochs, seed.
Model size (width), data (data_frac) and compute (epochs) are separate axes.

Three blocks, each crossed within itself. Together they are far smaller than
the full cross, and they overlap at the centre point (deduplicated by key):

  core     COMBOS x width x weight_decay          at 1x data, 1x epochs
  scale    COMBOS x width x data_frac x epochs    at the default weight decay
  retrain  COMBOS x retrain_epochs                at width 1, 1x data, 1x epochs

Seeds are the blocking factor. `ordered()` yields every unit of seed 0, then
seed 1, ..., shuffled within a seed, so stopping at any point leaves complete
seeds (plus part of one) and the design stays balanced.
"""

import dataclasses
import hashlib
import json
import random
import statistics

from datasets import DATASETS
from obd import Unit

COMBOS = (("mnist", "paper"), ("mnist", "mlp"), ("mnist", "vgg"), ("mnist", "resnet"),
          ("fmnist", "mlp"), ("fmnist", "vgg"), ("fmnist", "resnet"),
          ("cifar10", "mlp"), ("cifar10", "vgg"), ("cifar10", "resnet"))
WIDTHS = (0.5, 1.0, 2.0)
WEIGHT_DECAYS = (0.0, 1e-4, 1e-3, 1e-2)
DATA_FRACS = (0.1, 0.3, 1.0)
EPOCH_MULTS = (1 / 3, 1.0, 3.0)
RETRAIN_MULTS = (0.5, 1.0, 2.0, 4.0)   # of the dataset's default retrain epochs
BLOCKS = ("core", "scale", "retrain")


def _widths(arch):
    return (1.0,) if arch == "paper" else WIDTHS   # the paper net has one size


def _unit(dataset, arch, seed, width=1.0, data_frac=1.0, epoch_mult=1.0,
          weight_decay=None, retrain_mult=1.0):
    info = DATASETS[dataset]
    return Unit(
        dataset=dataset, arch=arch, width=width, data_frac=data_frac,
        epochs=max(1, round(info.epochs * epoch_mult)),
        weight_decay=info.weight_decay if weight_decay is None else weight_decay,
        retrain_epochs=max(1, round(info.retrain_epochs * retrain_mult)),
        seed=seed)


def block_units(block, seed):
    if block == "core":
        return [_unit(d, a, seed, width=w, weight_decay=wd)
                for d, a in COMBOS for w in _widths(a) for wd in WEIGHT_DECAYS]
    if block == "scale":
        return [_unit(d, a, seed, width=w, data_frac=f, epoch_mult=m)
                for d, a in COMBOS for w in _widths(a)
                for f in DATA_FRACS for m in EPOCH_MULTS]
    if block == "retrain":
        return [_unit(d, a, seed, retrain_mult=r) for d, a in COMBOS
                for r in RETRAIN_MULTS]
    raise ValueError(f"unknown block {block!r}; choose from {BLOCKS}")


def key(unit):
    """Stable identity of a unit's configuration (includes overrides)."""
    blob = json.dumps(dataclasses.asdict(unit), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def all_units(blocks, seeds, overrides=None, only=None):
    """Every distinct unit of the chosen blocks and seeds, with `overrides`
    (a dict of Unit fields) applied. `only` = {"dataset": [...], "arch": [...]}
    filters the design."""
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
    (dataset, arch, width), scaled by data_frac * epochs with an assumed 30%
    fixed overhead (the pruning phase does not scale with epochs). It falls
    back to the same (dataset, arch), then to None (unknown).
    """

    def __init__(self, records=()):
        self.by_group, self.by_arch = {}, {}
        for r in records:
            self.add(r)

    @staticmethod
    def _work(u):
        return u.data_frac * u.epochs

    def add(self, record):
        u, d = Unit(**record["unit"]), record["_meta"]["duration_s"]
        self.by_group.setdefault((u.dataset, u.arch, u.width), []).append((self._work(u), d))
        self.by_arch.setdefault((u.dataset, u.arch), []).append((self._work(u), d))

    def estimate(self, unit):
        for peers in (self.by_group.get((unit.dataset, unit.arch, unit.width)),
                      self.by_arch.get((unit.dataset, unit.arch))):
            if peers:
                ref = statistics.median(w for w, _ in peers)
                base = statistics.median(d for _, d in peers)
                return base * (0.3 + 0.7 * self._work(unit) / ref)
        return None
