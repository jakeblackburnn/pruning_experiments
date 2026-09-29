# Journal · 2026-09-29 18:25 · main · af34c67
## Start
**State:** Two reproductions (`obd/`, `synaptic_pruning/`) share one venv. Committed results, figures, briefs and notebooks are stale (synaptic ran the old 3×3×3 grid; OBD digits is from July; OBD cifar has no provenance). The user's `dev/journal.md` (2026-09-29) starts a rework: broader blocked factorial grids, old results only as a minor guide, drop the "bitter lesson" framing. `dev/main.md` holds the research questions.
**Since:** Only doc commits after the 09-25 close (`af34c67`). The 09-25 Close and Next list are superseded by this rework; D1/D2/D3/D5 in `dev/agents/DECISIONS.md` (RESULTS.md, generated blocks, notebooks, no resume) are overturned.
**Traps:** run outputs are per-cell JSONL from now on; old `results/*.json` are deleted. The user runs the full grids on an RTX 5080; I only smoke-test.

## Task: rework both experiments into budgetable, resumable factorial designs · 18:25
**Goal:** Two suites that answer `dev/main.md` (OBD at modern scales; OBD vs magnitude overlap; does synaptic pruning keep its benefit at scale), run by the user with `--budget MINUTES` and automatic resume.
**Now:** stale results; one seed, few criteria/archs; synaptic has one dataset, a confounded ladder, window leakage, no validation split; notebooks, RESULTS.md, `report.py` exist.
**Scope:** in: delete notebooks/RESULTS.md/report machinery/stale results; seeds as blocks; baselines; more archs/datasets; sweeps; validation split; budget + resume; CUDA. Out: writing results, running full grids, editing user's dev files, commits.
**Constraints:** append-only JSONL per (cell, seed); verify new dataset downloads headlessly (assumption).
**Approach:** shared runner (cells × seeds, seed-major, `--budget`, `--status`) + `aggregate.py`; OBD first, then port to synaptic. Full plan: `~/.claude/plans/lets-refactor-both-experimental-precious-quail.md`.
**Risks:** BatchNorm vs the Gauss-Newton `vmap(jacrev)` diagonal; download reachability; grid size.
**Done when:** `--help` works in both; no notebook/report references outside `dev/`; `--budget 1` stops in ~1 min, resumes without repeats, survives SIGINT; partial logs are balanced per seed; BN curvature matches an exact diagonal on a tiny net; split has no window overlap.
**Open:** 1. extra datasets/archs (defaults: Fashion-MNIST or CIFAR-100; Beijing PM2.5 + ETTh1; GRU/CNN/Transformer). 2. cell budgeting (default: core full cross + fractional add-ons).

## Progress · 2026-09-29 19:15 (folded into the Close below)
- **done:** removals (notebooks, RESULTS.md, report.py, stale results/figures, `synaptic_pruning.md`); README, `.gitignore`, `requirements.txt`; D6/D7 in `dev/agents/DECISIONS.md`.
- **done (obd):** `obd.py` (Unit, MLP/VGG/ResNet/paper archs, random and Taylor criteria, BN-safe curvature), `datasets.py` (mnist, fmnist, cifar10, train/val/test), `experiments.py` (shared fraction grid, any criterion in the retrain loop), `design.py`, `main.py`, `aggregate.py`, `figures.py`.
- **done (synaptic):** `datasets.py` (air_quality, beijing_pm25, etth1; causal fill, per-segment windows), `models.py` (rnn/lstm/gru/cnn/transformer), `pruning.py` (oneshot, random select), `train.py` (8 methods, val-selected test MAE), `design.py`, `main.py`, `aggregate.py`, `figures.py`; `stats.py` and `experiments.py` deleted.
- **verified:** `--help`; `--smoke` end to end for both (run, `--status`, `--tables`, `--plot`); `--budget` stops and resumes without repeats; SIGINT loses only the unit in flight; a truncated last log line is ignored; the diagonal-Hessian pass matches an exact one to 1e-16 on a BN ResNet (CE) and an MLP (MSE); data segments tile the rows and windows stay inside their segment; all three synaptic datasets download.
- **not verified:** anything on CUDA; real-size runs (the smoke tests used tiny caps and 1-2 epochs); the cost estimates, which need a first real unit.
- **changed behaviour:** synaptic `_global_prune` now uses `<=` at the threshold (prunes exactly the target count, the old `<` pruned one fewer); the target's own history is now a model input, with a persistence baseline recorded; OBD picks CUDA before Metal.
### Next
1. On the GPU: `python3 main.py --smoke` then `--budget 5` in each project, then `--status` for the full-run estimates; trim the blocks in `design.py` if the totals are too big.
2. OBD `aggregate.center()` assumes default epochs; a `--set epochs=` run has no centre rows for the curve figures.
3. The user writes up the results; nothing here generates prose.

## Close · 2026-09-29 19:22 · af34c67..7e9250a
- **changed:** both projects rebuilt as budgeted, resumable factorial designs: `design.py` (units, blocks, seed-major order, cost model), `main.py` (`--budget`, `--status`, `--smoke`, JSONL log), `aggregate.py` (CIs, paired contrasts, Holm, eta²), `figures.py`
- **changed:** obd: archs paper/mlp/vgg/resnet, datasets mnist/fmnist/cifar10 with val split, random + Taylor criteria, retrain loop for any criterion, BN-safe curvature. synaptic: datasets air_quality/beijing_pm25/etth1 (causal fill, per-segment windows), archs rnn/lstm/gru/cnn/transformer, 8 methods incl. random/one-shot pruning and a narrow control, val-selected test MAE, persistence baseline
- **changed:** removed notebooks, RESULTS.md, `report.py`, generated blocks, stale results/figures; README now has setup + a run recap (smoke → `--budget` → `--status` → resume → `--tables --plot`, and the selection flags)
- **why:** the user runs the full grids over days on an RTX 5080 and needs to split them across sessions; broader factorial design answers `dev/main.md`'s questions (see D6, D7)
- **verified:** `--help`, `--smoke` (run/status/tables/plot) in both; budget stop + resume with no repeats; SIGINT loses only the unit in flight; truncated log line ignored; Hessian matches exact to 1e-16 (BN ResNet CE, MLP MSE); data segments tile rows; all 3 synaptic datasets download. **Not verified:** CUDA, real-size runs, cost estimates
- **by:** pair
- `060d1ff` Rework both experiments as budgeted, resumable factorial designs
- `7e9250a` notes: start the rework, add main.md
- 43 files, +2183 −25166

### Next
1. On the GPU: `python3 main.py --smoke --seeds 1`, then `--budget 5` and `--status` in each project; trim blocks in `design.py` if the estimated totals are too large (per seed: obd 366 units, synaptic 1152 runs).
2. Run the real grids in slices (`--budget 600`, repeat), checking `--tables --plot` on partial logs.
3. Fix OBD `aggregate.center()`: it assumes default epochs, so `--set epochs=` runs draw no curve figures.

### Traps
- Changing an override or a `design.py` constant changes unit keys, so those runs are redone, not merged.
- Smoke runs use `results/smoke.jsonl` (git-ignored); use `--smoke` on the same flags for `--status/--tables`.
- The synaptic target's own history is now an input, and `_global_prune` uses `<=` at the threshold: old numbers are not comparable.
- zsh does not word-split `$VAR`; use a shell function to wrap repeated commands. `sed -i` needs `-i ''` on macOS.
- Backgrounded shell jobs ignore SIGINT; test Ctrl-C with Python's `subprocess` + `send_signal`.

### Pointers
- `obd/design.py`, `synaptic_pruning/design.py`: blocks, factor levels, `CostModel`
- `obd/main.py` `run()` / `synaptic_pruning/main.py` `run()`: budget loop; `OVERRIDABLE`
- `obd/aggregate.py`, `synaptic_pruning/aggregate.py`: tables; `dev/agents/DECISIONS.md` D6, D7
