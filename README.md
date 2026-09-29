# Pruning Experiments

**Pruning** is a class of regularization techniques that involve removing connections from ANNs.
This repo scrutinizes two pruning techniques: *Optimal Brain Damage* and *Synaptic Pruning*.

This repo is part of the **Neuro-AI lit review** on [My Weblog](https://jakeblackburn.dev/weblog/).

Two independent projects that share one Python environment. Each folder is
self-contained: its own data loading, its own `main.py`, and its own
`results/` and `figures/`.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For an RTX 50-series GPU install a CUDA 12.8+ build of torch first
(`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`).
The device is picked automatically (CUDA, else Metal, else CPU).

## Running

Each project is a factorial design of many small runs. `main.py` takes a time
budget, appends every finished run to `results/*.jsonl`, and resumes where it
stopped when run again, so a long study can be split over as many sessions as
you like. Runs go seed by seed, so stopping at any time leaves a balanced design.

```sh
cd obd                          # or synaptic_pruning
python3 main.py --smoke         # a tiny end-to-end check
python3 main.py --status        # progress and estimated time left
python3 main.py --budget 60     # run for about an hour, then stop
python3 main.py --tables --plot # summary CSVs and figures from what has finished
```

`python3 main.py --help` lists every flag.

### Recap: running the experiments

1. **Check it works** (a minute or two): `python3 main.py --smoke --seeds 1`. This writes to `results/smoke.jsonl`, which is git-ignored.
2. **Size the run:** do a short real run such as `python3 main.py --budget 5`, then `python3 main.py --status`. It prints done/total per block and seed, and the estimated time left, learned from the runs finished so far. If the total is too big, trim the blocks in `design.py`.
3. **Run in slices:** `python3 main.py --budget 600` (minutes), as many times as you like. A run that would overrun the remaining budget is not started. Ctrl-C loses only the run in flight. The log is `results/units.jsonl` (obd) or `results/runs.jsonl` (synaptic_pruning); a run is done once its line exists, so re-running the same command continues where it stopped.
4. **Look at partial results any time:** `python3 main.py --tables --plot` writes `results/tables/*.csv` and `figures/*.png` from whatever has finished.

Selecting part of the design (the same flags for `--budget`, `--status`, `--tables`):

```sh
python3 main.py --design core scale     # blocks (obd: core scale retrain; synaptic_pruning: core scale sweep)
python3 main.py --seeds 10              # seeds per configuration (default 5); more seeds append
python3 main.py --dataset cifar10 --arch vgg resnet     # restrict the design (synaptic_pruning also has --method)
python3 main.py --set hessian_samples=512               # override a Unit field (see OVERRIDABLE in main.py)
python3 main.py --device cpu            # pin the device
```

Runs are seed-major, so stopping at any point leaves complete seeds plus part of one, and the design stays balanced. Raising `--seeds` later only adds the new seeds' runs. Changing an override or a design constant gives runs a new key, so they are run again rather than mixed with old ones.

## Optimal Brain Damage — [`obd/`](obd/)

**LeCun, Denker & Solla (NIPS 1989).** Rank each weight of a trained network
by its saliency `s = ½·h·w²` (h = the diagonal Hessian) instead of its
magnitude, and delete the least salient first. `obd/` tests whether that holds up
at modern scales, and how different the saliency and magnitude rankings really are.
Also in the folder, unrelated to the suite: `hoeffding.py` plots how the Hoeffding
bound `P(|ν − μ| > ε) ≤ 2·exp(−2ε²N)` shrinks with N and ε.

## Synaptic Pruning — [`synaptic_pruning/`](synaptic_pruning/)

**Vos, van Eijk, Sarnyai & Rahimi Azghadi (2025).** A dropout replacement that
permanently zeroes the globally smallest-magnitude weights on a sparsity
schedule that ramps up during training, tested against dropout and other
regularizers on time-series forecasters.
