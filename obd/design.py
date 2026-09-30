"""The factorial design: which units exist, and in what order they run.

Factors (per unit): dataset x arch (COMBOS), width, data_frac, epochs (as a
multiple of the dataset's base epochs), weight_decay, retrain_epochs, seed.
Model size (width), data (data_frac) and compute (epochs) are separate axes.

Four blocks. They overlap at the centre point (deduplicated by key):

  core     COMBOS x width x weight_decay          at 1x data, 1x epochs
  scale    COMBOS x a star around the centre      width, data_frac, epochs one
                                                  at a time, default weight decay
  retrain  COMBOS x retrain_epochs                at the centre otherwise
  centre   COMBOS at the centre point             with more seeds (SEEDS)

Which prune-retrain loops a unit runs depends on its cell (`_retrain_criteria`):
the retrain arm is most of a unit's time, and the overlap question (core's
weight-decay axis) does not use it.

Seeds are the blocking factor. `ordered()` yields every unit of seed 0, then
seed 1, ..., shuffled within a seed, so stopping at any point leaves complete
seeds (plus part of one) and the design stays balanced. Run the centre block
first (`--design centre`): its seeds measure the seed-to-seed noise.
"""

import dataclasses
import hashlib
import json
import random

from datasets import DATASETS
from experiments import RETRAIN_CRITERIA
from obd import Unit

COMBOS = (("mnist", "paper"), ("mnist", "mlp"),
          ("fmnist", "mlp"), ("fmnist", "vgg"), ("fmnist", "resnet"),
          ("cifar10", "mlp"), ("cifar10", "vgg"), ("cifar10", "resnet"))
WIDTHS = (0.5, 1.0, 2.0)
WEIGHT_DECAYS = (0.0, None, 1e-2)      # None: the dataset's default
DATA_FRACS = (0.1, 0.3)                # off-centre levels of the scale star
EPOCH_MULTS = (1 / 3, 3.0)
RETRAIN_MULTS = (0.5, 1.0, 4.0)        # of the dataset's default retrain epochs
BLOCKS = ("core", "scale", "retrain", "centre")
SEEDS = {"core": 3, "scale": 3, "retrain": 3, "centre": 5}
# taylor (a modern baseline) and random (the floor) run at the centre only
MAIN_RETRAIN = ("magnitude", "saliency_layermean", "saliency")


def _widths(arch):
    return (1.0,) if arch == "paper" else WIDTHS   # the paper net has one size


def _retrain_criteria(width, data_frac, epoch_mult, default_wd, retrain_mult):
    if not default_wd:
        return ()
    if (width, data_frac, epoch_mult, retrain_mult) == (1.0, 1.0, 1.0, 1.0):
        return RETRAIN_CRITERIA
    return MAIN_RETRAIN


def _unit(dataset, arch, seed, width=1.0, data_frac=1.0, epoch_mult=1.0,
          weight_decay=None, retrain_mult=1.0):
    info = DATASETS[dataset]
    return Unit(
        dataset=dataset, arch=arch, width=width, data_frac=data_frac,
        epochs=max(1, round(info.epochs * epoch_mult)),
        weight_decay=info.weight_decay if weight_decay is None else weight_decay,
        retrain_epochs=max(1, round(info.retrain_epochs * retrain_mult)),
        seed=seed,
        retrain_criteria=_retrain_criteria(width, data_frac, epoch_mult,
                                           weight_decay is None, retrain_mult))


def block_units(block, seed):
    if block == "core":
        return [_unit(d, a, seed, width=w, weight_decay=wd)
                for d, a in COMBOS for w in _widths(a) for wd in WEIGHT_DECAYS]
    if block == "scale":
        return ([_unit(d, a, seed, width=w) for d, a in COMBOS for w in _widths(a)]
                + [_unit(d, a, seed, data_frac=f) for d, a in COMBOS for f in DATA_FRACS]
                + [_unit(d, a, seed, epoch_mult=m) for d, a in COMBOS for m in EPOCH_MULTS])
    if block == "retrain":
        return [_unit(d, a, seed, retrain_mult=r) for d, a in COMBOS
                for r in RETRAIN_MULTS]
    if block == "centre":
        return [_unit(d, a, seed) for d, a in COMBOS]
    raise ValueError(f"unknown block {block!r}; choose from {BLOCKS}")


def key(unit):
    """Stable identity of a unit's configuration (includes overrides)."""
    blob = json.dumps(dataclasses.asdict(unit), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def all_units(blocks, seeds=None, overrides=None, only=None):
    """Every distinct unit of the chosen blocks, each with its SEEDS (or
    `seeds` for every block), with `overrides` (a dict of Unit fields)
    applied. `only` = {"dataset": [...], "arch": [...]} filters the design."""
    only = only or {}
    seen, out = set(), []
    for block in blocks:
        for seed in range(seeds or SEEDS[block]):
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
