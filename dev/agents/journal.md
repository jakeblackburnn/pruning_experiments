# Journal · 2026-09-30 · main · 8be767f
## Start
Previous session: `/brainstorm` priced trimmed designs (`dev/brainstorm/experiment-alignment-and-run-length/index.md`, "Recommended shape", ≈ 7 h). Its three open questions are now answered (see the task below). This session frames the rework of both designs to fit the 8 h budget.

Carried forward:
- Next: measure two concurrent GPU processes (disjoint `--dataset`) for throughput; decide keep/delete `backup/laptop-main`.
- Traps: SSH to GitHub fails here (`Permission denied (publickey)`); `obd/results/smoke.jsonl` is tracked though git-ignored and currently modified; `obd/results/tables/` is stale.
- Pointers: `obd/experiments.py` `iterative_prune_retrain`; `obd/design.py` `block_units`; `synaptic_pruning/design.py` `block_units`; `dev/agents/DECISIONS.md` D6, D7.

## Task: Rework OBD and synaptic designs for the 8 h budget · 14:01
**Goal:** both designs answer Q1–Q3 of `dev/main.md` within ≈ 8 h on the RTX 5080, with a simple budget runner that stops where told and resumes.
**Now:**
- OBD grid ≈ 7.3 h/seed on CUDA (fitted, ±30%), 5 seeds default; retrain arm (5 criteria × 12 levels, `obd/experiments.py:19,25`) is 70–80% of every unit and runs on every cell (`obd/main.py:102`).
- Synaptic grid ≈ 1.12 h/seed, 5 seeds (`synaptic_pruning/design.py:54-70`).
- `--budget` skips units whose `CostModel` estimate exceeds the time left (`obd/main.py:154-186`, same in synaptic); `--status` prints time estimates, ≈ 2× high for OBD.
- Both `aggregate.py` report a Wilcoxon/Holm p that can't reach 0.05 at ≤ 5 seeds (`obd/aggregate.py:126`, `synaptic_pruning/aggregate.py:109`).
**Scope:** in: the brainstorm's recommended shape for both designs; OBD per-unit retrain arm; drop the pre-retrain train-split pass; budget runner rewrite (no estimates) in both `main.py`; drop Wilcoxon; delete `obd/results/units.jsonl` (99 pilot units; git keeps them). out: a large-scale point; the two-process GPU test; the full 7–8 h run; `dev/main.md` wording (user's file).
**Constraints:**
- "Modern scale" is read as *larger than the papers tested* (user); no design change for it.
- No runtime estimates anywhere: remove both `CostModel`s and the estimate lines in `--status` (counts only).
- `--budget MIN` runs in seed-major order and resumes from the log at unit granularity; a switch decides whether the unit in flight is cut off at the deadline or finishes. Assumption: cut off by default (the budget is a hard limit), `--finish` lets it complete; a cut-off unit is not logged and reruns next time.
- Synaptic keeps its 84 runs (remaining keys unchanged). OBD starts a fresh `units.jsonl`.
- Keep stored record fields unchanged except where the retrain arm shrinks (fewer criteria/levels).
**Approach:**
- OBD `design.py`: weight decay (0, default, 1e-2); `scale` as a star (centre, width 0.5/2, data 0.1/0.3, epochs ⅓/3); retrain mults (0.5, 1, 4); drop mnist/resnet and mnist/vgg; new Unit field for the retrain criteria: all 5 at the centre point, (magnitude, saliency, saliency_layermean) on other default-weight-decay cells, none on non-default weight decay. Seeds per block: 3 for the grid, 5 for a `centre` block (centre points + paper net), run first.
- OBD `experiments.py`: retrain visits keep (0.5, 0.25, 0.12, 0.05, 0.02, 0.01, 0.005); sweeps keep the 12 `FRACTIONS`; `pre_*` point evaluates val and test only.
- Synaptic `design.py`: seq_len (14, 60); scale star per method with width × (none, dropout, pruning) crossed, data 0.1/0.3, epochs ×0.5/×3, plus `random_pruning` along width; sweep 6 pruning variants (paper, smin 0, smax 0.5/0.9, horizon 10, corner 0–0.9 h10); 4 seeds.
- Both `main.py`: budget loop = deadline check between units, plus a timer (`signal.alarm` → same path as Ctrl-C) when cutting off; `--status` counts per block and seed.
- Both `aggregate.py`: drop Wilcoxon/Holm; check `aggregate`/`figures` cope with retrain keep levels ≠ sweep levels and units with no retrain arm.
**Risks:**
- Aggregation assumes every unit has all 5 retrain criteria and the same keep grid → missing-key errors or silently empty tables; the smoke `--tables` run on the new design would show it.
- Coarser retrain steps change the protocol; the tail Δ (keep ≤ 5%) might move vs the pilot. Noticed only by comparing the centre-block results with the pilot (git history).
- Hard cut-off mid-CUDA-kernel could leave a partial JSON line if it hits during append; the alarm must be disarmed around `append_record`.
- Budget fit is unverified until run; with no estimates, an overrun just leaves complete seeds.
**Done when:**
- OBD `--status` lists per seed: the trimmed grid (≈ 122 units/seed, 3 seeds) and the `centre` block (5 seeds); synaptic ≈ 474 runs/seed, 4 seeds; no time estimates printed, no `CostModel` left.
- `--smoke --budget 0.2` in each project stops at ≈ 12 s, cutting the unit in flight (nothing partial in the log); with `--finish` the in-flight unit completes and is logged; a rerun skips logged units and continues.
- `--smoke` full runs complete on CUDA in both projects, and `--tables` / `--plot` run on the smoke logs without errors; tables have no p column.
- OBD records: non-default-weight-decay units have no retrain arm; other grid units have 3 criteria at 7 levels; centre units 5; `pre_*` present without a train-split pass.
- `obd/results/units.jsonl` deleted; `synaptic_pruning/results/runs.jsonl` keeps its 84 runs and `--status` counts the ones still in the design as done.
- A one-off pricing of the v2 design with the brainstorm's fitted model (scratchpad, not repo) shows ≤ 8 h total; reported in the Close.
**Open:**
1. Cut-off default: hard stop at the deadline with `--finish` to complete the unit (assumed), or the reverse?
2. D6 describes the estimate-based budget; after this lands, record the new budget rule with `/decide`?
Decided: cut off by default (user); Open 1 closed.

### Progress · 14:45
- Implemented as framed. OBD: `obd/design.py` (8 combos, 4 blocks incl. `centre`, `SEEDS` per block, `_retrain_criteria`), `obd/obd.py` (`Unit.retrain_criteria`), `obd/experiments.py` (`RETRAIN_FRACTIONS`, `_point(which=...)`), `obd/main.py` (timer budget, `--finish`, counts-only `--status`), `obd/aggregate.py`/`figures.py` (no Wilcoxon/Holm; heatmap `*` = CI excludes 0). Synaptic: `design.py` (seq 14/60, star scale + random_pruning on width, 6 sweep variants, `SEEDS = 4`), `main.py` (same runner), `aggregate.py` (no Wilcoxon; Friedman keeps Holm). README run section updated. Both `CostModel`s gone.
- Deleted `obd/results/units.jsonl` and the stale tracked `obd/results/tables/*.csv`.
- Verified (CUDA): `--status` OBD 358 units (114/seed × 3 + centre 8 × 2 extra seeds); synaptic 1896 runs (474/seed × 4), 36 of the 84 old runs still in the design and counted done. OBD `--smoke --budget 0.2` stopped at 13.9 s wall, 11 intact lines; 0.6 s budget on one cifar10/resnet unit: cut off (0 logged) vs `--finish` (1 logged); rerun skipped logged units (15 records, 15 keys). Full OBD smoke 164 units in 3.5 min; `--tables`/`--plot` ran; records have 0/3/5 retrain criteria at 7 levels as designed, no `pre_train_*`. Synaptic: 3000-epoch run cut at 3 s budget (0 logged) vs `--finish` (1 logged, 19 s); full smoke + `--tables`/`--plot` ran.
- Pricing (scratchpad, pilot step times, brainstorm exponents, ±30%): OBD 4.8 h, synaptic 2.2 h (+64 unpriced) ≈ 6.9 h total.
- Smoke outputs removed; the user's modified `obd/results/smoke.jsonl` restored as it was.
- Trap: three OBD centre-point figures skip on smoke logs (`SMOKE` forces epochs=1, so nothing sits at epoch_mult 1); pre-existing.

## Close · 2026-09-30 14:43 · 8be767f..c4db8c8
- **changed:** `obd/design.py`, `obd/experiments.py`, `obd/obd.py`: v2 design (8 combos; star `scale`; `centre` block at 5 seeds; `Unit.retrain_criteria` 5/3/0 by cell; 7 retrain levels; no pre-retrain train pass)
- **changed:** `synaptic_pruning/design.py`: seq 14/60, star `scale` + `random_pruning` along width, 6 sweep variants, `SEEDS = 4`
- **changed:** both `main.py`: `--budget` hard stop via SIGALRM, `--finish`, counts-only `--status`, `CostModel` removed; both `aggregate.py`: Wilcoxon/Holm column dropped; README run section
- **changed:** deleted `obd/results/units.jsonl` (99-unit pilot) and stale `obd/results/tables/*.csv`; D8 recorded
- **why:** whole suite must fit < 8 h on the RTX 5080 and still answer Q1–Q3; budgeting should be simple (user)
- **verified:** `main.py --status` → OBD 358 units, synaptic 1896 runs (36 old runs counted); `--smoke --budget` cut-off vs `--finish` on both projects → 0 vs 1 unit logged, no partial lines; full `--smoke` + `--tables --plot` both projects → ran clean on CUDA; scratch pricing → ≈ 6.9 h (±30%)
- **by:** claude
- `c4db8c8` Rework both designs to fit the 8 h budget; hard-stop budget runner
- 19 files, +185 −699
- ⚠ uncommitted: `obd/results/smoke.jsonl` (user's earlier CUDA smoke output; tracked though git-ignored; restored unchanged)

### Next
1. `cd obd && ../.venv/bin/python main.py --design centre --budget 60` (≈ 0.6 h): 5 seeds at every centre point; measures seed noise of saliency − magnitude.
2. From the centre results, per combo seeds needed ≈ (2.8 · SD / δ)²; confirm 3 OBD seeds suffice, and compare the retrain tail Δ (keep ≤ 5%) with the pilot (`8be767f:obd/results/units.jsonl`) to check the coarser retrain steps.
3. Run the rest in slices: `main.py --budget N` in `obd/` then `synaptic_pruning/`.
4. Optional: 10-min test of two concurrent processes (disjoint `--dataset`) for throughput; needs a lock or per-process log for the append.
5. Carried: decide keep/delete `backup/laptop-main`.
### Traps
- SSH to GitHub fails here (`Permission denied (publickey)`).
- Run from the project dir with `.venv/bin/python` by absolute path; `../.venv/bin/python` prints harmless `sys.prefix` RuntimeWarnings.
- OBD centre-point figures (`retrain_curves`, `sweep_curves`, `overlap_curves`) skip on smoke logs: `SMOKE` forces epochs=1, so no unit sits at epoch_mult 1.
- Synaptic smoke runs are ~10 ms; to test the budget cut-off use `--set epochs=3000`.
- Adding a Unit field changes every key (why OBD v2 has a fresh log).
### Pointers
- `obd/design.py` `_retrain_criteria`, `SEEDS`; `obd/main.py` `run` (timer), `OutOfBudget`
- `synaptic_pruning/design.py` `block_units`
- `dev/agents/DECISIONS.md` D8; `dev/agents/project/notes.md` "Budget"
- `dev/brainstorm/experiment-alignment-and-run-length/index.md`
