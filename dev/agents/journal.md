# Journal · 2026-09-30 · main · 98f9782
## Start
Previous session left 99/1830 OBD units and 84/5760 synaptic runs (all seed 0, CUDA) and a rough shrink plan in `dev/agents/project/notes.md` ("Budget": whole suite < 8 h). This session: `/brainstorm` on how the experiments align with the research questions in `dev/main.md` and how to trim run length while keeping meaningful numbers.

## Close · 2026-09-30 13:54 · 98f9782..721be62
- **changed:** `dev/brainstorm/experiment-alignment-and-run-length/`: 6 notes (question map, OBD time profile, OBD and synaptic trim options priced, seeds and inference, index)
- **changed:** `dev/agents/project/notes.md`: per-step OBD cost facts, `--status` overestimate, Budget item 6 corrected (retrain, not Hessian)
- **why:** the full grid (OBD ≈ 36 h, synaptic ≈ 5.6 h) must fit 8 h without losing the answers to Q1–Q3
- **verified:** analysis scripts on `obd/results/units.jsonl` and `synaptic_pruning/results/runs.jsonl` (seed 0 only); plan prices from a fitted cost model, ±30%, not from runs. No code changed, nothing run on the GPU.
- **by:** claude
- `a25758b` journal: snapshot before consolidate
- `721be62` consolidate: align seed recommendation with the brainstorm budget table
- 6 files, +322 −0
- ⚠ uncommitted: `obd/results/smoke.jsonl` (CUDA smoke output; tracked though git-ignored)

### Next
1. User answers the brainstorm's three open questions (`index.md` "Open for you"): Q1/Q3 wording (architectures vs scale), fresh OBD log vs keeping the 99 units, Wilcoxon column.
2. `/spec` the trimmed designs v2 from `index.md` "Recommended shape"; implement in `obd/design.py`, `obd/experiments.py` (retrain criteria/levels), `synaptic_pruning/design.py`.
3. Run OBD centre points + paper net at 5 seeds first (≈ 0.6 h) to measure seed noise; then decide the third OBD seed.
4. 10-min test of two concurrent processes on the GPU (disjoint `--dataset`) for throughput.
5. Carried: decide keep/delete `backup/laptop-main`.
### Traps
- SSH to GitHub fails here (`Permission denied (publickey)`); fetch/pull/push need a key or `gh` auth.
- `units.jsonl` mixes Mac (MPS) and CUDA runs; filter `_meta.device == "cuda"` for timing.
- OBD `--status` time estimates are ~2× high (see notes.md).
- With 5 seeds the Wilcoxon-across-seeds p can't go below 0.0625; with 2 seeds the t-interval is ±12.7 SE.
- `obd/results/tables/` is stale (not regenerated this session).
### Pointers
- `dev/brainstorm/experiment-alignment-and-run-length/index.md` (findings, budget table, open questions)
- `obd/experiments.py` `iterative_prune_retrain` (pre-retrain `_point` evaluates the full train split, then discards it)
- `obd/design.py` `block_units`, `CostModel`; `synaptic_pruning/design.py` `block_units`
- `dev/agents/project/notes.md` "Budget"; `dev/agents/DECISIONS.md` D6, D7
