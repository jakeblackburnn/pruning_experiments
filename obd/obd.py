"""Optimal Brain Damage (LeCun, Denker & Solla, NIPS 1989) at modern scales.

Core calculations only: the unit-of-work config, device selection, model
definitions, the training step, the diagonal-Hessian / OBD-saliency math, and
the pruning primitives. Experiments live in experiments.py, the factorial
design in design.py, the runner in main.py, analysis in aggregate.py.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacrev, vmap


@dataclass(frozen=True)
class Unit:
    """One unit of work: train a base network, then run every pruning
    experiment on it. A unit is the smallest thing the runner schedules,
    logs and resumes."""
    dataset: str            # mnist | fmnist | cifar10 (datasets.DATASETS)
    arch: str               # paper | mlp | vgg | resnet (ARCHS)
    width: float            # channel / hidden-unit multiplier
    data_frac: float        # fraction of the training pool used
    epochs: int             # base training epochs
    weight_decay: float
    retrain_epochs: int     # retraining between pruning steps
    seed: int
    retrain_criteria: tuple[str, ...] = ()   # prune-retrain loops to run (design.py)
    hessian_samples: int = 1024   # subsample for the diagonal Hessian
    n_cap: int | None = None      # cap on training samples (smoke tests)
    lr: float = 1e-3
    batch_size: int | None = None  # None: 32 for the paper net, else 128

    @property
    def loss(self) -> str:
        """The paper net trains tanh outputs to +-1 targets with MSE; the
        modern nets use cross-entropy."""
        return "mse" if self.arch == "paper" else "ce"

    @property
    def batch(self) -> int:
        return self.batch_size or (32 if self.arch == "paper" else 128)


def resolve_device(prefer: str = "auto") -> str:
    """CUDA first, then Metal, then CPU. An explicit name is honoured as-is."""
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
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


class MLP(nn.Module):
    """Two hidden ReLU layers of 256*width units."""

    def __init__(self, in_channels, size, classes, width):
        super().__init__()
        h = max(4, int(256 * width))
        self.fc1 = nn.Linear(in_channels * size * size, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, classes)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class VGG(nn.Module):
    """A small VGG-style convnet (four 3x3 convs, two pools, two fc layers),
    32*width / 64*width channels. ReLU units, logits out."""

    def __init__(self, in_channels, size, classes, width):
        super().__init__()
        c1, c2, h = (max(4, int(c * width)) for c in (32, 64, 128))
        self.conv1 = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c1, 3, padding=1)
        self.conv3 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv4 = nn.Conv2d(c2, c2, 3, padding=1)
        self.fc1 = nn.Linear(c2 * 4 * 4, h)
        self.fc2 = nn.Linear(h, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.adaptive_avg_pool2d(x, 4).flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(y + (x if self.short is None else self.short(x)))


class ResNet(nn.Module):
    """A small ResNet with BatchNorm: a stem and three residual stages of one
    block each, 32/64/128 * width channels. The BatchNorm arch is here to test
    OBD's curvature pass on a net with normalisation (BN in eval mode)."""

    def __init__(self, in_channels, size, classes, width):
        super().__init__()
        c1, c2, c3 = (max(4, int(c * width)) for c in (32, 64, 128))
        self.stem = nn.Conv2d(in_channels, c1, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.block1 = _Block(c1, c1, 1)
        self.block2 = _Block(c1, c2, 2)
        self.block3 = _Block(c2, c3, 2)
        self.fc = nn.Linear(c3, classes)

    def forward(self, x):
        x = F.relu(self.bn(self.stem(x)))
        x = self.block3(self.block2(self.block1(x)))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class _Paper(MnistNet):
    def __init__(self, in_channels, size, classes, width):
        assert (in_channels, size, classes) == (1, 16, 10) and width == 1.0, \
            "the paper net is 16x16 digits at width 1 only"
        super().__init__()


ARCHS = {"paper": _Paper, "mlp": MLP, "vgg": VGG, "resnet": ResNet}


def build(unit, info):
    """The unit's network for a dataset described by datasets.Info."""
    return ARCHS[unit.arch](info.channels, info.size, info.classes, unit.width)


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


def train(model, x, y, unit, epochs=None, masks=None, verbose=False):
    """Minimize the experiment's loss with Adam.

    The weight decay is not cosmetic for OBD: without it hidden units
    saturate and redundant large weights sit in flat directions of the
    loss, where the quadratic saliency estimate breaks down.
    If masks are given, pruned weights are kept at zero.
    """
    loss_fn = LOSSES[unit.loss]
    optimizer = torch.optim.Adam(model.parameters(), lr=unit.lr,
                                 weight_decay=unit.weight_decay)
    model.train()
    for epoch in range(epochs if epochs is not None else unit.epochs):
        perm = torch.randperm(len(x), device=x.device)
        for i in range(0, len(x), unit.batch):
            idx = perm[i:i + unit.batch]
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

    was_training = model.training
    model.eval()  # BatchNorm must use its running statistics for the Jacobians
    try:
        # vmap(jacrev) through BatchNorm aborts the whole process on MPS (a
        # Metal assertion, not a catchable error), so those models go to the CPU.
        if x.device.type == "mps" and any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                                          for m in model.modules()):
            return _cpu_hessian(model, x, params, buffers, loss, batch_size, n_outputs)
        return _hessian_pass(model, x, params, buffers, loss, batch_size, n_outputs)
    except (RuntimeError, NotImplementedError) as err:
        # torch.func coverage on MPS is incomplete; the diagonal is worth an
        # expensive CPU pass rather than a crash. Other devices fail loudly.
        if x.device.type != "mps":
            raise
        print(f"  warning: Hessian pass failed on mps ({err}); retrying on cpu")
        return _cpu_hessian(model, x, params, buffers, loss, batch_size, n_outputs)
    finally:
        model.train(was_training)


def _cpu_hessian(model, x, params, buffers, loss, batch_size, n_outputs):
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


# Deletion scores. The first three are of the form (something) * w^2, so they
# differ only in how the curvature factor varies across weights:
#
#   magnitude           s = w^2               h treated as constant everywhere
#   saliency_layermean  s = 1/2 h_bar_L w^2   h treated as constant per layer
#   saliency            s = 1/2 h_kk w^2      the full Gauss-Newton diagonal
#
# The middle one is the control for "is OBD's advantage over magnitude just
# cross-layer budget allocation?" (see experiments.overlap_analysis). Two more
# controls: `random` (the floor any criterion must beat) and `taylor`, the
# first-order estimate |g * w| of the loss change (a gradient-based criterion
# that ignores curvature).
RANKINGS = ("magnitude", "random", "taylor", "saliency_layermean", "saliency")


def taylor_scores(model, x, y, loss, names, max_samples=None, batch_size=512):
    """First-order Taylor score |w * dL/dw|, gradients averaged over x."""
    params = dict(model.named_parameters())
    loss_fn = LOSSES[loss]
    if max_samples is not None and max_samples < len(x):
        idx = torch.randperm(len(x), device=x.device)[:max_samples]
        x, y = x[idx], y[idx]
    was_training = model.training
    model.eval()
    grads = {n: torch.zeros_like(params[n]) for n in names}
    for i in range(0, len(x), batch_size):
        model.zero_grad()
        loss_fn(model(x[i:i + batch_size]), y[i:i + batch_size],
                reduction="sum").backward()
        for n in names:
            grads[n] += params[n].grad
    model.zero_grad()
    model.train(was_training)
    return {n: (params[n].detach() * grads[n] / len(x)).abs() for n in names}


def scores_for(model, data, rank_by, unit):
    """Per-weight deletion scores. Higher score = more worth keeping.
    data = (x_train, y_train)."""
    x_train, y_train = data
    names = prunable_names(model)
    params = dict(model.named_parameters())
    if rank_by == "magnitude":
        return {name: params[name].detach().abs() for name in names}
    if rank_by == "random":
        return {name: torch.rand_like(params[name]) for name in names}
    if rank_by == "taylor":
        return taylor_scores(model, x_train, y_train, unit.loss, names,
                             max_samples=unit.hessian_samples)
    if rank_by == "saliency":
        return saliencies(model, x_train, loss=unit.loss,
                          max_samples=unit.hessian_samples, names=names)
    if rank_by == "saliency_layermean":
        h = diagonal_hessian(model, x_train, loss=unit.loss,
                             max_samples=unit.hessian_samples, names=names)
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
