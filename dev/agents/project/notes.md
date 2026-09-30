# Durable notes
- MPS (Mac): `vmap(jacrev)` through BatchNorm hard-aborts the process (Metal assertion, not catchable); `obd.diagonal_hessian` sends BN models to CPU on MPS (`_cpu_hessian`). VGG hits a catchable adaptive-pool error and falls back too. CUDA path untouched, unverified.
- `--budget` is checked only before a unit starts, so a run overruns by up to one unit (ResNet/CIFAR units take 6-21 min on the M4 Max).
- M4 Max (MPS) unit times: MLP/paper/VGG small 8-90 s; fmnist ResNet 6 min; cifar10 VGG w2 6 min; cifar10 ResNet w1 full 21 min. Estimated full OBD grid on the Mac ~276 h, so run the ResNet/CIFAR units on the RTX 5080.
- Backgrounded shell jobs ignore SIGINT (use `kill`); macOS has no `timeout`; zsh does not word-split `$VAR` (use `${=var}`).
- Log files hold per-device results: `obd/results/units.jsonl` mixes Mac (MPS) and later GPU runs by unit key.
