# OBD: where the time goes, and where the signal is

Measured 2026-09-30 from `obd/results/units.jsonl` (`_meta.step_durations_s`), CUDA units only
(RTX 5080), seed 0, medians over the mixed width/data/epoch cells each combo has so far.

## Time per unit, by step (seconds)

| combo | n | total | train | 6 sweeps | 5 retrains | overlap |
|---|---|---|---|---|---|---|
| cifar10/resnet | 11 | 164 | 20 | 26 | 119 | 0.8 |
| cifar10/vgg | 12 | 140 | 9 | 29 | 99 | 0.4 |
| fmnist/resnet | 11 | 114 | 15 | 14 | 81 | 0.7 |
| fmnist/vgg | 11 | 49 | 5 | 7 | 33 | 0.3 |
| mnist/paper | 3 | 34 | 7 | 0.2 | 27 | 0.0 |
| cifar10/mlp | 8 | 28 | 3 | 4 | 20 | 1.7 |
| mnist/resnet | 5 | 22 | 3 | 6 | 19 | 2.1 |
| mnist/vgg | 12 | 19 | 3 | 2 | 13 | 0.3 |
| mnist/mlp | 5 | 10 | 1 | 0.4 | 7 | 0.2 |
| fmnist/mlp | 7 | 6 | 1 | 0.2 | 6 | 0.1 |

- **Retraining is ≈ 70–80% of every unit.** Training the base net is 10–15%, sweeps 10–20%, the
  overlap analysis under 1%.
- **The Hessian is cheap:** the saliency sweep costs the same as the magnitude sweep (≈ 4 s on
  cifar10/vgg); saliency retraining costs ≈ 10% more than magnitude retraining. So fewer Hessian
  samples (item 6 of the "Budget" plan in `dev/agents/project/notes.md`) would save almost nothing.
- **Sweeps are mostly evaluation:** every pruning level evaluates train, val and test in full
  (60k images on CIFAR-10) for 6 criteria × 13 levels.
- **One waste found:** `iterative_prune_retrain` calls `_point` before retraining at each level,
  which evaluates the full training set too, but only `pre_val_*` and `pre_test_*` are kept.
  That is 60 full train-set passes per unit thrown away.

## Where saliency and magnitude differ (seed 0, test accuracy, saliency − magnitude)

Mean over the cells each combo has; keep = fraction of weights kept.

- **After iterative retraining:** |Δ| ≤ 0.02 at keep 70%–8% for every combo. The difference opens
  at keep 5% and grows: at keep 1% it is +0.4 to +11 pts, at 0.5% +4 to +25 pts. The spread
  across cells is as large as the mean in the tail (SD 0.04–0.23 at keep 0.5%).
- **Without retraining (sweep):** the difference lives at keep 50%–2%, and it is mostly negative:
  cifar10 and fmnist ResNets −12 to −16 pts at keep 35–25%; mnist/mlp −28 pts at 5%; the paper
  net −18 pts at 5%. Saliency is ahead only a little: cifar10/vgg (≤ 2 pts) and cifar10/mlp at
  keep 35–12% (≈ 2 pts, then behind below 5%).

Two consequences for the design:

1. The retrain arm spends 7 of its 12 steps (keep 70%–8%) where nothing differs. The protocol is
   iterative, so steps cannot simply be skipped, but they can be made coarser above 8%.
2. The seed-0 spread across cells is large, but it mixes factor effects with noise. Seed noise of
   the paired difference is still unmeasured: every finished unit is seed 0.
