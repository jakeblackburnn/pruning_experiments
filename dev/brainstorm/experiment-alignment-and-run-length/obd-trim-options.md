# OBD: trim options, priced

Prices come from a cost model fitted to the 85 CUDA units (log duration per step, with a
per-combo constant plus width, data and epoch/retrain exponents; the fitted exponents are width
0.5 train / 0.8 retrain, data 0.9 / 0.7, epochs 1.0, retrain multiple 0.8). Treat each price as
±30%; it is an extrapolation from seed 0.

**The current grid costs ≈ 7.3 h per seed on CUDA by this model**, not the ≈ 17 h per seed that
`--status` implies (81 h for 4.7 seeds). The built-in `CostModel` pools the 14 Mac (MPS) units into
its medians and scales the whole unit linearly with epochs, although the retrain arm (70% of the
time) does not depend on training epochs. Either way the full 5-seed grid (≈ 36 h) is far over
budget.

## Levers, in the order they pay

| # | Lever | Per seed | Saves | Cost to the conclusions |
|---|---|---|---|---|
| — | current grid | 7.3 h, 366 units | — | — |
| 1 | retrain arm only at default weight decay in `core` (overlap and sweeps still on every cell) | 5.7 h | 22% | none for Q2; Q1 loses the weight-decay × retrain interaction |
| 2 | `scale` as a star: centre, width 0.5/2, data 0.1/0.3, epochs ⅓/3 (7 cells, not 27) | 3.3 h | a further 42% | loses the width × data × epochs interactions; keeps every main effect |
| 3 | weight decay 4 → 3 levels (0, dataset default, 1e-2) | 3.1 h | 6% | seed 0 shows agreement rising 0 → 1e-4 → 1e-3, then falling at 1e-2; three levels still show the rise and the fall |
| 4 | retrain criteria 5 → 3 (magnitude, saliency, saliency_layermean; taylor and random at the centre only) and retrain levels 12 → 7 (keep 50, 25, 12, 5, 2, 1, 0.5%) | 1.8 h | a further 42% | the coarser steps change the protocol above keep 8%, where Δ ≈ 0 on seed 0; check on the light combos that the tail Δ does not move |
| 5 | retrain multiple 4 → 3 levels (0.5, 1, 4) | 1.6 h | 9% | the dose response keeps its ends |
| 6 | drop `mnist/resnet` and `mnist/vgg` | 1.5 h | 6% | fmnist carries those architectures on a harder task |

After all six: **≈ 1.5 h per seed, 122 units**, of which cifar10/resnet is 0.65 h and cifar10/vgg
0.37 h. The centre point of every combo is 0.12 h per seed, so extra seeds there are cheap.

## Smaller, unpriced levers

- **Skip the train-split pass before retraining.** `iterative_prune_retrain` evaluates the full
  training set before every retrain step and then drops it (only `pre_val_*`, `pre_test_*` are
  kept). Stored numbers do not change, so it is safe mid-run. Worth an estimated 5–15% of retrain
  time on CIFAR; measure it.
- **Subsample the training split in sweep evaluation** (a fixed 10k): sweeps are mostly full
  60k-image evaluations. It changes the stored `train_*` numbers, so it needs a new key or a new
  log.
- **Run two processes at once.** The models are small and probably leave the 5080 idle between
  kernel launches. Two processes with disjoint `--dataset` filters need no design change and no
  key change. Unmeasured; the risk is two processes appending long JSON lines to one log at the
  same time (a lock around the append, or one log per process, removes it).
- **Fewer Hessian samples:** not worth it. The saliency arms cost the same as magnitude to within
  10% (see [obd-time-profile.md](obd-time-profile.md)).

## Keys and the 99 finished units

Levers 1 and 4 need something per unit that says which retrain arm to run, which changes keys; 4
also changes what a retrain record contains. Mixing old and new records under one key would be
wrong. The clean option is a new log (for example `results/units_v2.jsonl`), keeping the 99 seed-0
units as a pilot file: they are what the time profile and the signal analysis here came from.
