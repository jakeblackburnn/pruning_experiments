"""Five one-step forecasters over a window of shape (batch, seq_len, features):
RNN, LSTM, GRU, a 1D CNN and a small Transformer encoder. Each ends the same
way: a summary vector of `width` units, dropout, a linear head.

The single `dropout` slot serves the paper's dropout baselines: p=0 is "no
dropout", p>0 with the module in eval() is plain dropout, p>0 forced into
train() at eval time and averaged over K passes is MC dropout
(see train.py:predict_mc). Synaptic pruning uses p=0 and attaches a
SynapticPruner from outside (pruning.py).
"""

import math

import torch
import torch.nn as nn


def resolve_device(prefer: str = "auto") -> str:
    """CUDA first, then Metal, then CPU. An explicit name is honoured as-is."""
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _Recurrent(nn.Module):
    def __init__(self, rnn, width, dropout):
        super().__init__()
        self.rnn = rnn
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


def _rnn(cls, **kw):
    def make(n_features, width, seq_len, dropout):
        return _Recurrent(cls(n_features, width, batch_first=True, **kw), width, dropout)
    return make


class CNN(nn.Module):
    """Two 1D convs (kernel 3, same padding) over time, mean-pooled."""

    def __init__(self, n_features, width, seq_len, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, width, 3, padding=1)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv2(torch.relu(self.conv1(x))))
        return self.head(self.drop(x.mean(dim=2))).squeeze(-1)


class Transformer(nn.Module):
    """A linear embedding, sinusoidal positions, two pre-norm encoder layers
    (4 heads, feed-forward 2*width) and the last position as the summary. The
    dropout slot is the head's only; the encoder layers run without dropout,
    so `width` is the single size knob and every method sees the same body."""

    def __init__(self, n_features, width, seq_len, dropout):
        super().__init__()
        if width % 4:
            raise ValueError("transformer width must be a multiple of 4")
        self.embed = nn.Linear(n_features, width)
        pos = torch.arange(seq_len)[:, None]
        div = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        pe = torch.zeros(seq_len, width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.encoder(self.embed(x) + self.pe)
        return self.head(self.drop(h[:, -1])).squeeze(-1)


ARCHS = {
    "rnn": _rnn(nn.RNN, nonlinearity="tanh"),
    "lstm": _rnn(nn.LSTM),
    "gru": _rnn(nn.GRU),
    "cnn": CNN,
    "transformer": Transformer,
}


def build(arch, n_features, width, seq_len, dropout=0.0):
    return ARCHS[arch](n_features, width, seq_len, dropout)


def count_prunable_params(model):
    """Weight tensors (ndim >= 2): the parameters the pruner can zero."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad and p.dim() >= 2)
