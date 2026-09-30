# Durable notes
- MPS (Mac): `vmap(jacrev)` through BatchNorm hard-aborts the process (Metal assertion, not catchable); `obd.diagonal_hessian` sends BN models to CPU on MPS (`_cpu_hessian`). VGG hits a catchable adaptive-pool error and falls back too. CUDA path untouched, unverified.
- `--budget` is a hard stop: a SIGALRM timer cuts off the unit in flight at the deadline (not logged, reruns next time); `--finish` lets it complete. No runtime estimates anywhere (user, 2026-09-30).
- M4 Max (MPS) unit times: MLP/paper/VGG small 8-90 s; fmnist ResNet 6 min; cifar10 VGG w2 6 min; cifar10 ResNet w1 full 21 min. Estimated full OBD grid on the Mac ~276 h, so run the ResNet/CIFAR units on the RTX 5080.
- Backgrounded shell jobs ignore SIGINT (use `kill`); macOS has no `timeout`; zsh does not word-split `$VAR` (use `${=var}`).
- The 99-unit OBD pilot log (Mac + CUDA, seed 0, 12-level retrain on every cell) was deleted 2026-09-30; it is in git at `8be767f:obd/results/units.jsonl`. `units.jsonl` starts fresh with the v2 design.

## Budget: whole suite (OBD + synaptic) must run in < 8 h on the RTX 5080 (user, 2026-09-30)
- Where OBD time goes (99 units, mixed Mac/GPU): cifar10/resnet, cifar10/vgg and fmnist/resnet are ~80% of unit time (median 2.5–3.3 min, up to 20 min); the six other combos are 8–50 s. Cut there first. Within a unit (CUDA, `step_durations_s`): the 5 retrain arms are 70–80%, base training 10–15%, sweeps 10–20%, overlap < 1%; the saliency arms cost within 10% of magnitude (the Hessian is cheap).
- v2 designs (2026-09-30) price at OBD ≈ 1.5 h/seed × 3 + centre 0.16 h × 2 extra seeds ≈ 4.8 h, synaptic ≈ 0.54 h/seed × 4 ≈ 2.2 h (+64 unpriced runs); ±30%, from pilot step times with the brainstorm's exponents.
- The shrink plan (star scale blocks, trimmed retrain arm, fewer weight-decay/retrain levels, 3/5 OBD seeds, 4 synaptic seeds) is implemented as the v2 designs; reasoning in `dev/brainstorm/experiment-alignment-and-run-length/`.
- Editing `design.py` blocks or constants changes unit keys only for units that change; already-finished units with identical keys stay done. Adding or removing a Unit field changes every key (v2 added OBD `retrain_criteria`, hence the fresh log).
