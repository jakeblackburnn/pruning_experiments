# Optimal Brain Damage — reproduction

Reproduces the pruning experiments and figures from **LeCun, Denker & Solla,
"Optimal Brain Damage" (NIPS 1989)**: rank each weight of a trained network by
its *saliency* s = ½·h·w² (h = diagonal Hessian, Gauss-Newton approximation)
and show that deleting low-saliency weights hurts far less than deleting
low-magnitude ones.

Two experiments:

- **digits** — the paper's setup. Its 16×16 zip code data isn't available, so
  MNIST is downsampled to 16×16 with the paper's dataset sizes (7,291 train /
  2,007 test). The network is the 1989 zip-code conv net (~8.8k free
  parameters, LeCun's scaled tanh, MSE to ±1 targets).
- **cifar** — a modern counterpart: conventional ReLU convnet (~590k
  parameters) on CIFAR-10 with cross-entropy.

Plus one experiment the paper didn't run: **how much does OBD's ranking
actually differ from plain magnitude?**

## Three lessons the reproduction teaches

1. **The saliency is a local estimate and goes stale.** A ranking computed
   once on the full network (one-shot) loses to plain magnitude pruning: the
   lowest-saliency set includes large weights sitting in flat, redundant
   directions, and deleting thousands of them together is far outside the
   quadratic approximation's validity. Recomputing saliencies after each
   deletion step (still with no retraining) restores the paper's result —
   Fig 2 shows every curve.
2. **OBD needs a well-conditioned minimum.** Trained without weight decay,
   hidden units saturate; curvature — and hence every saliency — collapses
   toward zero and the ranking becomes noise. (This is also why the model uses
   LeCun's scaled sigmoid `1.7159·tanh(2x/3)`: ±1 targets stay inside its
   linear range.)
3. **Magnitude pruning is OBD under an isotropic-curvature prior**, and the
   same weight decay that conditions the Hessian is what makes that prior a
   good one. Since s = ½·h·w², ranking by saliency and ranking by |w| are the
   *same ranking* whenever h is constant — magnitude is not an approximation
   to OBD, it is a special case of it. Figs 5–7 measure how far from constant
   h really is, and find that most of the variation is a per-layer offset: the
   two rankings agree closely *inside* a layer and differ mainly in how they
   split the pruning budget *across* layers. `EXPLANATION.md` §7 works through
   why.

## Run

From the repo root, with the shared virtualenv active
(`pip install -r requirements.txt` once, then `source .venv/bin/activate`):

```sh
cd obd
python3 main.py                    # digits: train -> prune -> plot
python3 main.py --dataset cifar    # the CIFAR-10 counterpart
```

The device is chosen automatically — Metal (MPS) if available, else CUDA, else
CPU — and printed at startup. Pin it with `--set device=cpu`, which is often
*faster* for digits: at 8.8k parameters and batch 32, kernel-launch overhead
dominates.

Stage by stage: `python3 main.py --train | --prune | --plot`. Each stage caches
its output (`checkpoints/`, `results/`), so later stages can be rerun alone, and
every results file records the config and checkpoint hash it came from so a
stale merge gets a warning rather than silence. The prune stage's eight
experiments can be rerun individually and merged into the cached results:

```sh
python3 main.py --only retrain_magnitude
python3 main.py --dataset cifar --set epochs=5 lr=3e-4 hessian_samples=1024
```

| `--only` key | what it does |
|---|---|
| `magnitude` | delete smallest-\|w\| first, no retraining |
| `saliency` / `saliency_recomputed` | same, ranked by OBD saliency — once, or re-ranked each step |
| `saliency_layermean` / `..._recomputed` | ranked by ½·h̄<sub>layer</sub>·w² — the control between the two |
| `retrain` / `retrain_magnitude` | the full prune-retrain loop, and its magnitude-ranked control |
| `overlap` | the ranking-agreement analysis behind Figs 5–7 |

`obd.ipynb` is the intended entry point: it reads the cached results and walks
through the whole story, figures and all.

## Code map

| file | contents |
|---|---|
| `obd.py` | configs, device selection, models, training, the diagonal Hessian and saliencies, the pruning primitives |
| `experiments.py` | the sweeps, the prune-retrain loop, the overlap analysis, the `--only` registry |
| `figures.py` | every figure, as a pure function from the results JSON |
| `datasets.py` | MNIST→16×16 and CIFAR-10 loading |
| `main.py` | the CLI and all disk I/O |

`EXPLANATION.md` is a walkthrough of each of them.
