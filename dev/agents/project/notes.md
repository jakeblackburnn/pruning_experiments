# Durable notes
- MPS (Mac): `vmap(jacrev)` through BatchNorm hard-aborts the process (Metal assertion, not catchable); `obd.diagonal_hessian` sends BN models to CPU on MPS (`_cpu_hessian`). VGG hits a catchable adaptive-pool error and falls back too. CUDA path untouched, unverified.
- `--budget` is checked only before a unit starts, so a run overruns by up to one unit (ResNet/CIFAR units take 6-21 min on the M4 Max).
- M4 Max (MPS) unit times: MLP/paper/VGG small 8-90 s; fmnist ResNet 6 min; cifar10 VGG w2 6 min; cifar10 ResNet w1 full 21 min. Estimated full OBD grid on the Mac ~276 h, so run the ResNet/CIFAR units on the RTX 5080.
- Backgrounded shell jobs ignore SIGINT (use `kill`); macOS has no `timeout`; zsh does not word-split `$VAR` (use `${=var}`).
- Log files hold per-device results: `obd/results/units.jsonl` mixes Mac (MPS) and later GPU runs by unit key.

## Budget: whole suite (OBD + synaptic) must run in < 8 h on the RTX 5080 (user, 2026-09-30)
- Now: OBD full grid ≈ 81 h remaining at CUDA speed (1731 of 1830 units, 5 seeds; ≈ 20 h per seed). Synaptic: 5760 runs, no timing yet (`--budget 5` first to fill the cost model). Aim ≈ 4 h each; OBD needs a ~15× cut per seed, so fewer seeds alone will not do it.
- Where OBD time goes (99 units, mixed Mac/GPU): cifar10/resnet, cifar10/vgg and fmnist/resnet are ~80% of unit time (median 2.5–3.3 min, up to 20 min); the six other combos are 8–50 s. Cut there first.
- Shrink without weakening conclusions:
  1. Pair, don't replicate: OBD compares saliency vs magnitude on the same trained net, so effects are paired within a unit and 3 seeds is enough. Keep 5 seeds only for the centre point and the headline (cifar10/resnet).
  2. Star design instead of full crosses in `scale`: centre + one-factor-at-a-time (data 0.1 / 1, epochs 1/3 / 3, width 0.5 / 2) = 5–7 cells instead of 27 (OBD) or 27×3 methods (synaptic). Keep the full cross only for the one interaction a research question needs (synaptic: width × method for "does benefit continue at scale").
  3. Tier the combos: light combos (mnist*, fmnist/mlp, fmnist/vgg, cifar10/mlp) keep the full `core` grid; heavy ones (cifar10/resnet, cifar10/vgg, fmnist/resnet) get centre + width + one weight-decay contrast only. Drop redundant combos (mnist/resnet, mnist/vgg) or keep them at 1 seed.
  4. Cut levels of factors with a flat response: weight decay 4 → 3 (0, 1e-4, 1e-2), retrain multiples 4 → 3 or run `retrain` on 3 combos only. Check seed-0 `--tables` (effects.csv) first and drop factors with negligible effect.
  5. Synaptic: `sweep` smin × smax × horizon (12 pruning variants) → 5–6 (fractional; keep the paper's setting and the corners); `core` seq_len 3 → 2 (14, 60); keep `none`, `pruning`, `dropout` as the scale methods, run `mc_dropout`, `oneshot`, `narrow` and `random_pruning` in `core` only (they are controls); keep the `random_pruning` control.
  6. Cost per unit: fewer Hessian samples for the big nets and shorter retrain, only after checking they do not move saliency-vs-magnitude on seed 0.
  7. Run order: seed 0 of the trimmed design first, look at the tables, then decide seeds 1–2; extra seeds only where the seed-0 interval straddles zero.
- Editing `design.py` blocks or constants changes unit keys only for units that change; already-finished units with identical keys stay done. Do not change unit fields.
