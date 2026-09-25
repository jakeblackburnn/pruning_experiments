# Pruning reproductions

Two independent paper reproductions that share one Python environment. Each
folder is self-contained: its own data loading, its own `main.py`, and its own
`checkpoints/`, `results/` and `figures/`.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The device is picked automatically (Metal, else CUDA, else CPU). Every
`main.py --help` lists its stages and flags.

## Optimal Brain Damage — [`obd/`](obd/)

**LeCun, Denker & Solla (NIPS 1989).** Rank each weight of a trained network
by its saliency `s = ½·h·w²` (h = the diagonal Hessian) instead of its
magnitude, and delete the least salient first. This repo runs the paper's setup
on 16×16 digits, a modern CIFAR-10 counterpart, and one experiment the paper
didn't run: how different the saliency and magnitude rankings really are.

```sh
cd obd && python3 main.py      # both datasets: train → prune → plot → report → notebook
```

**Results brief**

<!-- report:obd_headline -->
| dataset | prunable weights | base test acc | test acc at ~10% weights, OBD vs magnitude (retrained) | same, at the last step | Spearman(s, \|w\|) |
|---|---|---|---|---|---|
| digits | 8,760 | 95.1% | 93.5% vs 94.0% | 79.0% vs 73.6% (124 left) | 0.927 |
| cifar | 590,944 | 72.1% | 72.4% vs 72.6% | 68.5% vs 67.7% (4,006 left) | — |

_Generated from `results/digits.json` (2026-07-27, commit ?); `results/cifar.json` (no provenance — predates tracking)._
<!-- /report -->

<!-- fill: 2-3 bullets after each full run. What the reproduction confirms,
what it narrows, and the finding the paper didn't have. -->

Full results: [`obd/RESULTS.md`](obd/RESULTS.md). Code and argument:
[`obd/obd.ipynb`](obd/obd.ipynb). Also in the folder, unrelated to the
suite: `hoeffding.py` plots how the Hoeffding bound
`P(|ν − μ| > ε) ≤ 2·exp(−2ε²N)` shrinks with N and ε (`python3 hoeffding.py`
→ `figures/hoeffding.png`).

## Synaptic Pruning — [`synaptic_pruning/`](synaptic_pruning/)

**Vos, van Eijk, Sarnyai & Rahimi Azghadi (2025).** A dropout replacement that
permanently zeroes the globally smallest-magnitude weights on a sparsity
schedule that ramps up during training. This repo compares it with dropout and
no regularization on RNN/LSTM forecasters (UCI Air Quality), then runs a
bitter-lesson test: does its benefit survive when model, compute and data are
scaled up together?

```sh
cd synaptic_pruning && python3 main.py   # ~2 h on an M4: replicate → ladder → plot → report → notebook
```

**Results brief**

<!-- report:sp_headline -->
| experiment | result |
|---|---|
| replication | pruning has the lowest mean MAE in 4/12 cells; beats none in 7/12, beats dropout in 6/12, beats mc_dropout in 7/12; 1/12 cells significant at p<0.05 |
| bitter-lesson ladder | _not yet run — `python3 main.py --bitter-lesson`_ |

_Generated from `results/replication.json` (2026-09-07, commit ?); `results/bitter_lesson.json` (2026-09-07, commit ?)._
<!-- /report -->

<!-- fill: 2-3 bullets after each full run. Does the paper's claim hold on
this dataset, and does the benefit survive scale? -->

Full results: [`synaptic_pruning/RESULTS.md`](synaptic_pruning/RESULTS.md).
Code and argument:
[`synaptic_pruning/synaptic_pruning.ipynb`](synaptic_pruning/synaptic_pruning.ipynb).
