# Reproductions

Two independent paper reproductions, sharing one Python env.

- **[`obd/`](obd/README.md)** — Optimal Brain Damage (LeCun et al. 1989):
  prune by Hessian-weighted saliency instead of magnitude.
- **[`synaptic_pruning/`](synaptic_pruning/README.md)** — Synaptic Pruning
  (Vos et al. 2025): magnitude pruning as a dropout replacement, plus a
  bitter-lesson scaling test.

Each folder is self-contained (own data loading, own `main.py`, own
`checkpoints/` / `results/` / `figures/` caches) and has its own README with
the full story and options. This file is just setup + the one-liner to run
each.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```sh
cd obd && python3 main.py                  # OBD: digits (paper) + cifar variant
cd synaptic_pruning && python3 main.py     # synaptic pruning: replication + bitter-lesson ladder
```

Both CLIs support running stages individually (`--train`/`--prune`/`--plot`,
`--replicate`/`--bitter-lesson`/`--plot`), overriding config with
`--set field=value`, and picking the device (auto-detected: Metal, else CUDA,
else CPU). See each project's README for the flags and what each stage costs.
