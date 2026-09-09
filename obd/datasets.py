"""Dataset loading for the OBD experiments: MNIST resized to 16x16, and CIFAR-10.

    load(exp) -> ((x_train, y_train, labels_train), (x_test, y_test, labels_test))

The convention: `y` is whatever the loss consumes, `labels` are always integer
classes for computing accuracy. For digits `y` is a +-1 one-hot matrix (the
paper trains tanh outputs with MSE to +-1 targets); for CIFAR cross-entropy
consumes the integer labels directly, so `y` *is* `labels`.

Both datasets are small enough to hold fully in memory as single tensors — no
DataLoader, batching is manual slicing everywhere downstream.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

# the 1989 zip code data isn't public, so MNIST stands in at the paper's sizes
N_TRAIN = 7291
N_TEST = 2007


def _prepare_digits(dataset, n: int):
    x = dataset.data[:n].float().unsqueeze(1) / 255.0        # (N, 1, 28, 28) in [0, 1]
    x = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
    x = x * 2.0 - 1.0

    # targets: +1 for the correct class, -1 elsewhere
    labels = dataset.targets[:n]
    y = -torch.ones(n, 10)
    y[torch.arange(n), labels] = 1.0
    return x, y, labels


def _prepare_cifar(dataset):
    x = torch.from_numpy(dataset.data).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    x = (x - mean) / std
    labels = torch.tensor(dataset.targets)
    return x, labels, labels  # cross-entropy consumes the labels directly


def _load_digits():
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True)
    return _prepare_digits(train_set, N_TRAIN), _prepare_digits(test_set, N_TEST)


def _load_cifar():
    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True)
    return _prepare_cifar(train_set), _prepare_cifar(test_set)


LOADERS = {"digits": _load_digits, "cifar": _load_cifar}


def load(exp):
    if exp.name not in LOADERS:
        raise ValueError(f"no data loader for experiment {exp.name!r}; "
                         f"choose from {sorted(LOADERS)}")
    return LOADERS[exp.name]()
