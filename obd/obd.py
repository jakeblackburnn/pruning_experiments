"""Optimal Brain Damage — LeCun, Denker & Solla (NIPS 1989), reproduced,
plus a modern CIFAR-10 counterpart. See RESULTS.md and obd.ipynb for the story.

Core calculations only: experiment configs, device selection, model
definitions, the training step, the diagonal-Hessian / OBD-saliency math, and
the pruning primitives. The pruning experiments themselves live in
experiments.py, plotting in figures.py, and the CLI in main.py.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacrev, vmap


@dataclass(frozen=True)
class Experiment:
    name: str            # checkpoint / results / figure prefix
    loss: str            # "mse" (paper setup) or "ce" (conventional)
    device: str          # "auto" (see resolve_device) or an explicit torch device
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    retrain_epochs: int  # retraining between pruning steps (Fig 4)
    hessian_samples: int | None  # subsample for the diagonal Hessian (None = all)
    sweep_points: int            # pruning levels in the no-retrain sweeps
    sweep_min_remaining: int     # smallest weight count in the no-retrain sweep
    retrain_min_remaining: int   # stop the prune-retrain loop below this
    shrink: float                # fraction of weights kept per prune-retrain step


# simple 1990 network, 16x16 digits, MSE (mirrors OBD paper) for MNIST
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
DIGITS = Experiment(
    name="digits", loss="mse", device="auto",
    epochs=60, lr=1e-3, weight_decay=1e-3, batch_size=32,
    retrain_epochs=4, hessian_samples=None, sweep_points=25,
    sweep_min_remaining=50, retrain_min_remaining=100, shrink=0.8,
)

# VGG-style convnet for CIFAR-10
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
CIFAR = Experiment(
    name="cifar", loss="ce", device="auto",
    epochs=30, lr=1e-3, weight_decay=1e-4, batch_size=128,
    retrain_epochs=2, hessian_samples=2048, sweep_points=18,
    sweep_min_remaining=3000, retrain_min_remaining=3000, shrink=0.7,
)

EXPERIMENTS = {e.name: e for e in (DIGITS, CIFAR)}

SEED = 1989


def resolve_device(prefer: str = "auto") -> str:
    """Metal first, then CUDA, then CPU. An explicit name is honoured as-is.

    The digits net is tiny (8.8k parameters at batch 32), so kernel-launch
    overhead can make it *slower* on an accelerator than on the CPU — use
    `--set device=cpu` if that is the case on your machine.
    """
    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sigmoid(x):
    """LeCun's scaled tanh 1.7159 * tanh(2x/3): f(+-1) is well inside the
    linear range, so the net can fit +-1 targets without saturating. With a
    plain tanh the weights grow until every unit saturates, the curvature
    (and hence every OBD saliency) collapses to ~0, and the ranking becomes
    noise."""
    return 1.7159 * torch.tanh(2.0 / 3.0 * x)


def _connection_table():
    """(12, 12) binary matrix: table[j, i] = 1 if H2 map j sees H1 map i.
    The paper's exact table was never published, so we use a rotating scheme:
    H2 map j reads H1 maps j..j+7 (mod 12)."""
    table = torch.zeros(12, 12)
    for j in range(12):
        for k in range(8):
            table[j, (j + k) % 12] = 1.0
    return table


class MnistNet(nn.Module):
    """The same architecture of (LeCun et al. 1989, "Backpropagation Applied
    to Handwritten Zip Code Recognition") adapted to MNIST:

        input   1 @ 16x16
        H1     12 @ 8x8    5x5 conv, stride 2
        H2     12 @ 4x4    5x5 conv, stride 2, each map sees only 8 of the
                           12 H1 maps (sparse connection table)
        H3     30          fully connected
        output 10          fully connected, tanh, trained to +-1 targets

    Free parameters: 312 + 2412 + 5790 + 310 = 8824
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(12 * 4 * 4, 30)
        self.fc2 = nn.Linear(30, 10)

        # fixed sparse connection table for H1 -> H2, broadcast to kernel shape
        mask = _connection_table()[:, :, None, None].expand(12, 12, 5, 5)
        self.register_buffer("conn_mask", mask.clone())

    def forward(self, x):
        # pad with -1 (the background value) so 5x5/stride-2 convs map 16->8->4
        x = F.pad(x, (2, 2, 2, 2), value=-1.0)
        x = sigmoid(self.conv1(x))
        x = F.pad(x, (2, 2, 2, 2), value=-1.0)
        x = sigmoid(F.conv2d(x, self.conv2.weight * self.conn_mask,
                             self.conv2.bias, stride=2))
        x = x.flatten(1)
        x = sigmoid(self.fc1(x))
        x = sigmoid(self.fc2(x))
        return x


class CifarNet(nn.Module):
    """A conventional small VGG-style convnet for CIFAR-10 (~590k parameters).

    ReLU units and a cross-entropy head - the modern counterpart to the
    1989 network, for testing how OBD holds up beyond the paper's setup.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits; cross-entropy applies the softmax


MODELS = {"digits": MnistNet, "cifar": CifarNet}


def build(exp):
    if exp.name not in MODELS:
        raise ValueError(f"no model for experiment {exp.name!r}; "
                         f"choose from {sorted(MODELS)}")
    return MODELS[exp.name]()


def count_free_parameters(model) -> int:
    """Parameter count, excluding entries zeroed by a fixed connection table."""
    n = sum(p.numel() for p in model.parameters())
    if hasattr(model, "conn_mask"):
        n -= int((1 - model.conn_mask).sum().item())
    return n


LOSSES = {"mse": F.mse_loss, "ce": F.cross_entropy}


@torch.no_grad()
def evaluate(model, x, y, labels, loss="mse", batch_size=2048):
    """Return (loss, accuracy) on a dataset held in memory."""
    loss_fn = LOSSES[loss]
    model.eval()
    total_loss, correct = 0.0, 0
    for i in range(0, len(x), batch_size):
        out = model(x[i:i + batch_size])
        total_loss += loss_fn(out, y[i:i + batch_size], reduction="sum").item()
        correct += (out.argmax(dim=1) == labels[i:i + batch_size]).sum().item()
    return total_loss / y.numel(), correct / len(x)


def apply_masks(model, masks):
    """Zero out pruned weights in place. masks: {param_name: 0/1 tensor}."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name])


def train(model, x, y, exp, epochs=None, masks=None, verbose=False):
    """Minimize the experiment's loss with Adam.

    The weight decay is not cosmetic for OBD: without it hidden units
    saturate and redundant large weights sit in flat directions of the
    loss, where the quadratic saliency estimate breaks down.
    If masks are given, pruned weights are kept at zero.
    """
    loss_fn = LOSSES[exp.loss]
    optimizer = torch.optim.Adam(model.parameters(), lr=exp.lr,
                                 weight_decay=exp.weight_decay)
    model.train()
    for epoch in range(epochs if epochs is not None else exp.epochs):
        perm = torch.randperm(len(x), device=x.device)
        for i in range(0, len(x), exp.batch_size):
            idx = perm[i:i + exp.batch_size]
            optimizer.zero_grad()
            loss_fn(model(x[idx]), y[idx]).backward()
            optimizer.step()
            if masks is not None:
                apply_masks(model, masks)
        if verbose:
            with torch.no_grad():
                full = sum(loss_fn(model(x[i:i + 4096]), y[i:i + 4096],
                                   reduction="sum").item()
                           for i in range(0, len(x), 4096)) / y.numel()
            print(f"  epoch {epoch + 1:3d}  train loss {full:.4f}")
    return model


# --------------------------------------------------------------------------
# Diagonal Hessian and OBD saliencies
#
# OBD approximates the change in loss from deleting parameter w_k by a
# second-order Taylor expansion around the trained weights, keeping only
# the diagonal term:
#
#     delta_E ~= s_k = 1/2 * h_kk * w_k^2
#
# The paper computes h_kk with the Levenberg-Marquardt approximation,
# which is exactly the Gauss-Newton diagonal: with J the Jacobian of the
# network outputs and H_out the Hessian of the loss w.r.t. the outputs,
#
#     h_kk = sum over samples of  [J^T H_out J]_kk
#
# For MSE (mean reduction over N samples x O outputs), H_out is diagonal:
#     h_kk = 2/(N*O) * sum_po (d out_po / d w_k)^2
# For softmax cross-entropy, H_out = diag(p) - p p^T with p the softmax:
#     h_kk = 1/N * sum_p [ sum_o p_o J_ok^2 - (sum_o p_o J_ok)^2 ]
#
# Both make h_kk consistent with the mean-reduced loss, so predicted loss
# increases are directly comparable to measured ones. We get the per-sample
# Jacobians from torch.func instead of hand-rolling the paper's
# layer-by-layer second-order backprop.
# --------------------------------------------------------------------------

def _hessian_pass(model, x, params, buffers, loss, batch_size, n_outputs):
    """Accumulate the Gauss-Newton diagonal over x, differentiating only
    w.r.t. `params` (the rest of the model's parameters are held fixed)."""
    fixed = {name: p.detach() for name, p in model.named_parameters()
             if name not in params}

    def outputs(diff_params, x_single):
        merged = {**fixed, **diff_params}
        return functional_call(model, (merged, buffers),
                               (x_single.unsqueeze(0),)).squeeze(0)

    # per-sample Jacobian of the network outputs w.r.t. every diff parameter
    jac_fn = vmap(jacrev(outputs), in_dims=(None, 0))

    h = {name: torch.zeros_like(p) for name, p in params.items()}
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size]
        jac = jac_fn(params, batch)  # name -> (B, O, *param.shape)
        if loss == "mse":
            for name in h:
                h[name] += 2.0 * jac[name].pow(2).sum(dim=(0, 1))
        elif loss == "ce":
            with torch.no_grad():
                p = model(batch).softmax(dim=1)  # (B, O)
            for name in h:
                J = jac[name].flatten(2)                        # (B, O, K)
                mean = torch.einsum("bo,bok->bk", p, J)          # sum_o p_o J_ok
                hk = torch.einsum("bo,bok->k", p, J.pow(2)) - mean.pow(2).sum(0)
                h[name] += hk.view_as(h[name])
        else:
            raise ValueError(loss)

    scale = 1.0 / (len(x) * n_outputs) if loss == "mse" else 1.0 / len(x)
    return {name: value * scale for name, value in h.items()}


def diagonal_hessian(model, x, loss="mse", batch_size=None, max_samples=None,
                     names=None):
    """Gauss-Newton diagonal of the loss, as {param_name: tensor}.

    names restricts the result (and the Jacobian work) to those parameters;
    the default is every parameter. Only prunable weights are ever consumed
    downstream, so passing `prunable_names(model)` skips the bias Jacobians.
    """
    all_params = {name: p.detach() for name, p in model.named_parameters()}
    params = ({name: all_params[name] for name in names} if names is not None
              else all_params)
    buffers = {name: b.detach() for name, b in model.named_buffers()}

    with torch.no_grad():
        n_outputs = model(x[:1]).shape[-1]
    if batch_size is None:
        # keep the (batch, outputs, n_params) Jacobian around ~100M floats
        n_diff = sum(p.numel() for p in params.values())
        batch_size = max(1, min(256, 100_000_000 // (n_outputs * n_diff)))
    if max_samples is not None and max_samples < len(x):
        idx = torch.randperm(len(x), device=x.device)[:max_samples]
        x = x[idx]

    try:
        return _hessian_pass(model, x, params, buffers, loss, batch_size, n_outputs)
    except (RuntimeError, NotImplementedError) as err:
        # torch.func coverage on MPS (and some CUDA builds) is incomplete;
        # the diagonal is worth an expensive CPU pass rather than a crash.
        if x.device.type == "cpu":
            raise
        print(f"  warning: Hessian pass failed on {x.device.type} ({err}); "
              f"retrying on cpu")
        device = x.device
        cpu_model = model.to("cpu")
        try:
            h = _hessian_pass(
                cpu_model, x.to("cpu"),
                {name: p.to("cpu") for name, p in params.items()},
                {name: b.to("cpu") for name, b in buffers.items()},
                loss, batch_size, n_outputs)
        finally:
            model.to(device)
        return {name: value.to(device) for name, value in h.items()}


def saliencies(model, x, loss="mse", max_samples=None, names=None):
    """OBD saliency s_k = 1/2 * h_kk * w_k^2, for `names` (default: all)."""
    params = dict(model.named_parameters())
    names = list(params) if names is None else list(names)
    h = diagonal_hessian(model, x, loss=loss, max_samples=max_samples, names=names)
    return {name: 0.5 * h[name] * params[name].detach() ** 2 for name in names}


# --------------------------------------------------------------------------
# Pruning primitives
#
# Only weights are pruned (biases are left alone), and entries that are
# structurally absent under a fixed connection table are never counted.
# Pruning is enforced by 0/1 masks rather than by removing tensors, which
# keeps every layer's shape (and hence the Hessian code) uniform.
# --------------------------------------------------------------------------

def prunable_names(model):
    """Weight tensors of every conv/linear layer (biases excluded)."""
    return [f"{name}.weight" for name, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


def initial_masks(model):
    """All-ones masks, except a model's fixed connection table if it has one."""
    params = dict(model.named_parameters())
    masks = {name: torch.ones_like(params[name]) for name in prunable_names(model)}
    if hasattr(model, "conn_mask"):
        masks["conv2.weight"] = model.conn_mask.clone()
    return masks


def flatten(tensors, names):
    """Concatenate per-tensor dicts into one vector, for cross-layer ranking."""
    return torch.cat([tensors[name].flatten() for name in names])


def unflatten(vector, like, names):
    out, i = {}, 0
    for name in names:
        n = like[name].numel()
        out[name] = vector[i:i + n].view_as(like[name])
        i += n
    return out


# Deletion scores, in increasing order of what they assume about the curvature.
# All three are of the form (something) * w^2, so they differ only in how the
# curvature factor varies across weights:
#
#   magnitude           s = w^2               h treated as constant everywhere
#   saliency_layermean  s = 1/2 h_bar_L w^2   h treated as constant per layer
#   saliency            s = 1/2 h_kk w^2      the full Gauss-Newton diagonal
#
# The middle one is the control for "is OBD's advantage over magnitude just
# cross-layer budget allocation?" — see experiments.overlap_analysis.
RANKINGS = ("magnitude", "saliency_layermean", "saliency")


def scores_for(model, x_train, rank_by, exp):
    """Per-weight deletion scores. Higher score = more worth keeping."""
    names = prunable_names(model)
    params = dict(model.named_parameters())
    if rank_by == "magnitude":
        return {name: params[name].detach().abs() for name in names}
    if rank_by == "saliency":
        return saliencies(model, x_train, loss=exp.loss,
                          max_samples=exp.hessian_samples, names=names)
    if rank_by == "saliency_layermean":
        h = diagonal_hessian(model, x_train, loss=exp.loss,
                             max_samples=exp.hessian_samples, names=names)
        return {name: 0.5 * h[name].mean() * params[name].detach() ** 2
                for name in names}
    raise ValueError(f"unknown ranking {rank_by!r}; choose from {list(RANKINGS)}")


def prune_to(masks, scores, n_remaining, names):
    """Return new masks with only the n_remaining highest-scored weights kept."""
    mask_flat = flatten(masks, names).clone()
    score_flat = flatten(scores, names).clone()
    score_flat[mask_flat == 0] = float("inf")  # already-dead weights stay dead
    n_delete = int(mask_flat.sum().item()) - n_remaining
    if n_delete > 0:
        order = torch.argsort(score_flat)
        mask_flat[order[:n_delete]] = 0.0
    return unflatten(mask_flat, masks, names)


def keep_mask(scores_flat, n_remaining):
    """Boolean mask of the n_remaining highest entries of a flat score vector."""
    keep = torch.zeros_like(scores_flat, dtype=torch.bool)
    keep[torch.argsort(scores_flat, descending=True)[:n_remaining]] = True
    return keep
