"""Datasets for the OBD experiments, held fully in memory as tensors.

    load_splits(unit, device) -> {"train": (x, y, labels), "val": ..., "test": ...}

`y` is whatever the loss consumes (a +-1 one-hot matrix for MSE, the integer
labels for cross-entropy); `labels` are always integer classes, for accuracy.

Splits: the official test set is used for final numbers only. A seeded 10% of
the official training set is held out as validation; the training subset is
the first `data_frac` of the remainder (nested across fractions for a seed).
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets

DATA_DIR = Path(__file__).parent / "data"
VAL_FRAC = 0.1


@dataclass(frozen=True)
class Info:
    channels: int
    size: int
    classes: int
    epochs: int          # base training epochs at the 1x compute level
    weight_decay: float  # default weight decay
    retrain_epochs: int  # default retraining epochs per pruning step


DATASETS = {
    # the 1989 zip-code set isn't public; MNIST stands in at the paper's sizes
    "mnist": Info(channels=1, size=16, classes=10, epochs=60, weight_decay=1e-3,
                  retrain_epochs=4),
    "fmnist": Info(channels=1, size=16, classes=10, epochs=30, weight_decay=1e-4,
                   retrain_epochs=2),
    "cifar10": Info(channels=3, size=32, classes=10, epochs=30, weight_decay=1e-4,
                    retrain_epochs=2),
}

# the paper's sizes for MNIST (7,291 train / 2,007 test)
MNIST_SIZES = (7291, 2007)


def _gray(dataset, n, mean=None, std=None):
    x = dataset.data[:n].float().unsqueeze(1) / 255.0        # (N, 1, 28, 28)
    x = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
    x = x * 2.0 - 1.0 if mean is None else (x - mean) / std
    return x, dataset.targets[:n].clone()


@lru_cache(maxsize=None)
def _raw(name):
    """(x_train, labels_train), (x_test, labels_test) on the CPU, normalised."""
    if name == "mnist":
        tr = datasets.MNIST(DATA_DIR, train=True, download=True)
        te = datasets.MNIST(DATA_DIR, train=False, download=True)
        return _gray(tr, MNIST_SIZES[0]), _gray(te, MNIST_SIZES[1])
    if name == "fmnist":
        tr = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
        te = datasets.FashionMNIST(DATA_DIR, train=False, download=True)
        return (_gray(tr, len(tr), 0.2860, 0.3530),
                _gray(te, len(te), 0.2860, 0.3530))
    if name == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

        def prep(ds):
            x = torch.from_numpy(ds.data).float().permute(0, 3, 1, 2) / 255.0
            return (x - mean) / std, torch.tensor(ds.targets)
        return (prep(datasets.CIFAR10(DATA_DIR, train=True, download=True)),
                prep(datasets.CIFAR10(DATA_DIR, train=False, download=True)))
    raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")


def _targets(labels, loss, classes):
    if loss == "ce":
        return labels
    y = -torch.ones(len(labels), classes)   # +1 for the correct class, -1 elsewhere
    y[torch.arange(len(labels)), labels] = 1.0
    return y


def load_splits(unit, device):
    """The unit's train / val / test tensors, on `device`."""
    info = DATASETS[unit.dataset]
    (x_tr, l_tr), (x_te, l_te) = _raw(unit.dataset)

    gen = torch.Generator().manual_seed(unit.seed)
    perm = torch.randperm(len(x_tr), generator=gen)
    n_val = round(VAL_FRAC * len(x_tr))
    val_idx, pool = perm[:n_val], perm[n_val:]
    n_train = max(1, round(unit.data_frac * len(pool)))
    if unit.n_cap is not None:
        n_train = min(n_train, unit.n_cap)
        val_idx = val_idx[:unit.n_cap]
        x_te, l_te = x_te[:unit.n_cap], l_te[:unit.n_cap]
    train_idx = pool[:n_train]

    out = {}
    for split, (x, labels) in {"train": (x_tr[train_idx], l_tr[train_idx]),
                               "val": (x_tr[val_idx], l_tr[val_idx]),
                               "test": (x_te, l_te)}.items():
        out[split] = tuple(t.to(device) for t in
                           (x, _targets(labels, unit.loss, info.classes), labels))
    return out
