# Journal · 2026-09-29 19:40 · main · 5c79a47
## Start
**State:** Two reproductions (`obd/`, `synaptic_pruning/`) were rebuilt as budgeted, resumable factorial designs (`060d1ff`): `design.py`, `main.py` (`--budget`, `--status`, `--smoke`, JSONL log), `aggregate.py`, `figures.py`. CPU smoke-tested only; CUDA and real-size runs are unverified, and cost estimates need a first real unit.
**Since:** Nothing after the 09-29 Close (`5c79a47` is that journal commit).
**Next:** The user's `dev/journal.md` ends with a `## Results` section (unrewritten): its open item is task 4, use previous results as a guide (partial, no new results yet). Carried forward: 1. GPU smoke, then `--budget 5` and `--status` in each project; trim `design.py` blocks if totals are too big. 2. Run the real grids in slices (`--budget 600`), check `--tables --plot` on partial logs. (Minor, low priority: OBD `aggregate.center()` draws no centre figures on `--set epochs=`/`--smoke` runs; default runs are fine.)
**Traps:** changing an override or a `design.py` constant changes unit keys, so those runs are redone; smoke runs use `results/smoke.jsonl`, so pass `--smoke` on the same flags for `--status/--tables`; old synaptic numbers are not comparable (target history is now an input, `<=` at the prune threshold); zsh does not word-split `$VAR`; `sed -i ''` on macOS; test Ctrl-C via Python `subprocess` + `send_signal`.
**Pointers:** `obd/design.py`, `synaptic_pruning/design.py` (blocks, `CostModel`); each `main.py` `run()` / `OVERRIDABLE`; each `aggregate.py`; `dev/agents/DECISIONS.md` D6, D7.

## Issue: OBD ResNet aborts on MPS · 20:20
- **cause:** `diagonal_hessian`'s `vmap(jacrev)` through BatchNorm (`obd/obd.py`) trips a Metal assertion (`MPSNDArray ... buffer is not large enough`) that kills the process, so the existing `except RuntimeError` CPU fallback never runs. Forward, train and eval are fine; the same net with BN removed is fine; MLP fine; VGG raises a catchable adaptive-pool error and falls back to CPU.
- **fix:** on MPS, models containing BatchNorm run the Hessian pass on the CPU directly (`_cpu_hessian`, shared with the old fallback). Verified: `--smoke --arch resnet` and `--arch vgg` on fmnist finish on MPS (18 and 9 units). Not verified: CUDA (untouched path); the speed cost of the CPU Hessian for real-size ResNets.
- **also:** the Mac 30 min run left one real unit in `obd/results/units.jsonl` (mlp, MPS, untracked).

## Close · 2026-09-30 00:40 · 5c79a47..HEAD
- **changed:** `obd/obd.py`: BatchNorm models on MPS run the Hessian pass on CPU (`_cpu_hessian`); ResNet no longer aborts on the Mac GPU
- **changed:** ran OBD on the M4 Max: 14 of 1830 units in `obd/results/units.jsonl` (untracked); new `dev/agents/project/notes.md`
- **why:** the first Mac run crashed on ResNet (Metal assertion in `vmap(jacrev)` through BatchNorm); the user wanted real results and Mac speed numbers
- **verified:** `--smoke --dataset fmnist --arch resnet` and `--arch vgg` on MPS finish; real `--budget 30` and `--budget 1` runs completed 13 units incl. cifar10/resnet. Not verified: CUDA; whether the CPU Hessian explains the ResNet slowness; tables/figures on this log
- **by:** pair
- `880f681` obd: run BatchNorm Hessian on CPU under MPS (Metal abort)
- 1 file, +20 −11
- ⚠ uncommitted: `obd/results/` (Mac run log)

### Next
1. Run the small units on the Mac or move all to the RTX 5080: `python3 main.py --smoke --seeds 1`, then `--budget 5` and `--status`; trim `design.py` blocks (est. 276 h on the Mac).
2. Check the cifar10/resnet saliency result (0.709 at 12% kept vs magnitude 0.796, below the 0.765 base) with more seeds and on CUDA; suspect BatchNorm handling in the curvature.
3. Then run the real grids in slices (`--budget 600`); `--tables --plot` on partial logs.
4. Optional: OBD `aggregate.center()` draws no centre figures under `--set epochs=`/`--smoke`.
### Traps
- `--budget` overruns by up to one unit; see `dev/agents/project/notes.md` for the MPS/BatchNorm and shell traps.
- Changing an override or a `design.py` constant changes unit keys; old synaptic numbers are not comparable.
### Pointers
- `obd/obd.py` `diagonal_hessian` / `_cpu_hessian`; `obd/results/units.jsonl`; `dev/agents/project/notes.md`
