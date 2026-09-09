# Synaptic Pruning — reproduction + a bitter-lesson scaling test

Reproduces the method from **Vos, van Eijk, Sarnyai & Rahimi Azghadi (2025),
["Synaptic Pruning: A Biological Inspiration for Deep Learning
Regularization"](https://arxiv.org/abs/2508.09330)** (`synaptic_pruning_review.pdf`
in this directory): a dropout replacement that, instead of stochastically
zeroing activations each forward pass, permanently zeros the
globally-smallest-magnitude weights on a schedule that ramps sparsity up
over training. Two experiments:

1. **replication** — the paper's four-way comparison (no regularization,
   dropout, MC dropout, synaptic pruning) across RNN and LSTM forecasters
   and the paper's own input-sequence-length sweep, on one of the paper's
   own datasets: UCI's **Air Quality** hourly gas-sensor series, downloaded
   fresh from `archive.ics.uci.edu` (see `datasets.py` — the paper's other
   three sources sit behind a Kaggle login or a JS proof-of-work wall and
   weren't reachable headlessly; Air Quality is the one the paper itself
   calls "high complexity", so it's also the more interesting one to keep).
2. **bitter_lesson** — not a grid: a small ladder of individually large
   runs. Each of 6 tiers scales network size (LSTM hidden width), compute
   (training epochs), and dataset size (training rows) *together*, roughly
   geometrically (32/20/4000 up to 1024/640/9000), comparing synaptic
   pruning against the unregularized baseline at each tier. The question:
   does the benefit of this technique grow, hold, or shrink as real compute
   scales up? A grid that varies one axis at a time can't cleanly answer
   that — it conflates "more compute" with "the same compute, spread over a
   bigger/smaller model or dataset." Scaling all three together is closer to
   how compute scaling actually happens in practice. See `REPORT.md` for the
   answer.

## Run

```sh
cd synaptic_pruning                    # from the repo root, with the shared
                                        # venv active (see top-level README)
python3 main.py --replicate            # ~25-30 min on an M4 (MPS)
python3 main.py --bitter-lesson        # ~90 min (6-tier compute ladder)
python3 main.py --plot                 # figures/*.png
python3 main.py                        # all three, in order
```

The device is chosen automatically (Metal, else CUDA, else CPU) and printed
at startup. The bitter-lesson ladder's largest tier (hidden_size=1024,
640 epochs) is real scale, not a toy — direct timing on an M4 showed the
cost growing roughly linearly with epoch count and only weakly with hidden
size (128→256 hidden added ~2x cost, matching just the epoch-doubling
factor alone), so the whole ladder still comfortably fits an M4 MacBook in a
single sitting; nothing here needed handing off to a bigger machine.

Override any config field with `--set`:

```sh
python3 main.py --replicate --set trials=5 epochs=10       # faster, noisier
python3 main.py --bitter-lesson --set trials=3              # faster, noisier
```

(`--set` doesn't support the bitter-lesson `tiers` field directly — it's a
tuple of tuples, and the CLI's override parser only coerces flat tuples of
scalars. To change tiers, edit `BitterLessonConfig.tiers` in
`experiments.py` directly, or construct a config in a small script.)

Each stage caches its output to `results/<stage>.json`, so `--plot` can be
rerun alone once both grids exist.

## What the paper's algorithm actually is

Three pieces (`pruning.py`, matching the paper's Algorithms 1–6):

- **Mask init** — a binary mask per weight tensor (every ≥2-D parameter:
  `Linear` weights, and RNN/LSTM `weight_ih_l*`/`weight_hh_l*`), all ones.
- **Cubic schedule** — target sparsity `s(t)` is 0 during a warmup of 2
  epochs, then ramps cubically from `smin=0.3` to `smax=0.7` by the final
  epoch: `s(t) = smin + (smax - smin) * ((t - twarmup)/(ttotal - twarmup))^3`.
  The paper's pseudocode is 1-indexed (`t = 1..ttotal`), which matters: it's
  what makes `s(ttotal) == smax` exactly (`figures/sparsity_schedule.png`).
- **Global magnitude selection** — every `prune_every=5` batches (after
  warmup), pool the absolute magnitudes of every *currently active* weight
  across the *entire* network, find the threshold that would bring total
  sparsity up to `s(t)`, and zero everything below it. Masks are reapplied
  after every optimizer step so a pruned weight can't drift back off zero
  via momentum.

**A subtlety that matters for the bitter-lesson experiment**: the schedule's
horizon (`schedule_epochs` in `pruning.py:PruneConfig`, i.e. the epoch at
which `s(t)` hits `smax`) is a separate knob from a run's actual training
length (`epochs` in `train.py:RunConfig`, via `prune_schedule_epochs`). The
original version of this code tied them together — every run's ramp was
rescaled to complete exactly on that run's last epoch, so a short 5-epoch
run got slammed from ~40% to 70% sparsity in its literal final epoch with
zero training afterward to adapt, while an 80-epoch run got the same ramp
spread smoothly with dozens of epochs to recover. That made the old
bitter-lesson "compute axis" result partly a code artifact (short runs
looked catastrophically bad in a way that had little to do with the method
itself) rather than a clean measurement of whether more compute helps.
`prune_schedule_epochs` is now fixed independently of `epochs`, so every
tier of the compute ladder below gets genuine post-ramp fine-tuning time.

This is deliberately different from Optimal Brain Damage
(`../obd.py` — same family of ideas, opposite mechanism): OBD ranks weights
by a second-order saliency computed after training converges and prunes
once (or iteratively, with retraining, between prunes); synaptic pruning
ranks by raw magnitude and prunes continuously *during* training, with no
separate pruning phase. It is, in the framing of `../README.md` §3,
literally magnitude pruning under an implicit constant-curvature prior —
except now the "training" and "pruning" loops are the same loop.

## Method comparison — four regularizers, one training loop

| method | dropout module | at eval time |
|---|---|---|
| `none` | off | — |
| `dropout` | `p=0.3` | disabled (standard) |
| `mc_dropout` | `p=0.3` | forced active, averaged over 20 stochastic passes |
| `pruning` | off | `SynapticPruner` attached; masks always applied |

`p=0.3` isn't published in the paper (its dropout baseline rate isn't
stated); it's chosen to match the pruning schedule's `smin`, so dropout and
pruning start from a comparable sparsity floor. `train.py:train_one` runs
one config to convergence and reports final-epoch test MAE (standardized
units — see `datasets.py:Standardizer` — since the point is comparing
*methods*, not recovering CO(GT)'s physical scale).

## Where this reproduction departs from the paper, and why

- **One dataset, not four.** Air Quality only; see above.
- **Two architectures, not three.** RNN and LSTM; PatchTST is a
  considerably heavier transformer variant, and the paper's own ablation
  (§2) found dynamic magnitude pruning gave "minimal efficacy" outside
  recurrent architectures on non-time-series tasks — the RNN/LSTM result is
  where the method's claim actually lives.
- **Trimmed to essentials, then re-weighted toward statistical power.** The
  replication grid dropped `mc_dropout` (a supplementary variant, not needed
  to test the central pruning-vs-dropout-vs-none claim) and cut the paper's
  6-point sequence-length sweep to 3 representative points
  (short/medium/long: 1, 14, 60). That freed enough compute to raise trials
  from 5 to 10, matching the paper — the original 5-trial count almost
  certainly left the Friedman significance test underpowered (only 1 of 12
  cells reached significance), which is itself a plausible reason the
  original run looked like a coin flip even if a real, smaller effect
  exists underneath the noise.
- **Feature engineering is ours**, not the paper's (undisclosed) 313-feature
  pipeline: we use the dataset's 11 native sensor/weather channels plus
  their own recent history (the sliding window), no derived features. This
  is the most likely dominant driver of the replication gap — it's the one
  difference that's both large and can't be closed (the paper's pipeline is
  undisclosed).
- **Dropout rate (`p=0.3`)** is an assumption, as noted above.

None of these change what's being tested: whether magnitude-pruning-as-you-
train beats stochastic dropout on real time-series forecasting, and whether
that edge is a small-model/small-data artifact. See `REPORT.md` for the
full causal breakdown of why the initial reproduction attempt fell short of
the paper's reported effect sizes.

## Code map

| file | contents |
|---|---|
| `pruning.py` | the algorithm: mask init, cubic schedule, global magnitude selection |
| `models.py` | RNN/LSTM forecasters, device selection |
| `train.py` | one training run under one of the four methods |
| `datasets.py` | Air Quality download, cleaning, windowing, standardization |
| `experiments.py` | the two grids (`replication`, `bitter_lesson`) |
| `stats.py` | trial summaries (mean/std/CI) and the Friedman test |
| `figures.py` | every figure, as a pure function of the cached results |
| `main.py` | the CLI and all disk I/O |

`synaptic_pruning.ipynb` reads the cached results and walks through the
whole story, figures included. `REPORT.md` is the write-up of what actually
came out.
