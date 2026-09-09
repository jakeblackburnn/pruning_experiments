"""Two sequence forecasters, matching two of the paper's three architectures
(RNN, LSTM — PatchTST is a heavier transformer variant the paper also ran;
skipped here since the M4's Metal backend and this dataset's scale don't
need it to test the paper's claim, see README). Both: a recurrent encoder
over the input window, dropout on the last hidden state, a linear head
predicting the next value.

The same `dropout` slot serves double duty as the paper's baselines: p=0
is "no dropout", p>0 with the module in eval() is plain "Dropout", p>0 with
the module forced into train() at eval time and averaged over K stochastic
passes is "MC Dropout" (see train.py:predict_mc). "Synaptic Pruning" uses
p=0 and instead has a SynapticPruner attached externally (pruning.py).
"""

import torch
import torch.nn as nn


def resolve_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _Forecaster(nn.Module):
    def __init__(self, rnn: nn.Module, hidden_size: int, dropout: float):
        super().__init__()
        self.rnn = rnn
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(self.drop(last)).squeeze(-1)


class RNNForecaster(_Forecaster):
    def __init__(self, n_features, hidden_size=32, num_layers=1, dropout=0.0):
        rnn = nn.RNN(n_features, hidden_size, num_layers, batch_first=True,
                     nonlinearity="tanh",
                     dropout=dropout if num_layers > 1 else 0.0)
        super().__init__(rnn, hidden_size, dropout)


class LSTMForecaster(_Forecaster):
    def __init__(self, n_features, hidden_size=32, num_layers=1, dropout=0.0):
        rnn = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                      dropout=dropout if num_layers > 1 else 0.0)
        super().__init__(rnn, hidden_size, dropout)


MODELS = {"rnn": RNNForecaster, "lstm": LSTMForecaster}


def build(model_type, n_features, hidden_size=32, num_layers=1, dropout=0.0):
    return MODELS[model_type](n_features, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=dropout)


def count_prunable_params(model):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad and p.dim() >= 2)
