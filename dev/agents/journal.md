# Journal · 2026-09-30 17:53 · main · 234016c
## Start
**State:** v2 designs for OBD and synaptic are in (`c4db8c8`, D8), and `--budget` is now a hard stop. The user ran synaptic with a 150 min budget and it finished: `main.py --status` → 1896/1896 runs (474/seed × 4 seeds; core 960, scale 864, sweep 240). There are 1860 new records in `synaptic_pruning/results/runs.jsonl` (1944 lines, 84 of them old), written 18:42–20:44 UTC on the RTX 5080. Summed `duration_s` is 2.03 h (max 46.6 s), about 0.5 h under the priced 2.2 h (±30%). OBD v2 has no real runs yet.
**Since:** no commits since the last journal. Uncommitted: `synaptic_pruning/results/runs.jsonl` (+1860, the user's run) and `obd/results/smoke.jsonl` (+141 lines, OBD smoke units at `n_cap` 256; tracked though `.gitignore:6` ignores `results/smoke*.jsonl`). Local `origin/main` = HEAD, but the fetch may be stale because SSH fails here.
**Next:** user: review the synaptic results, then push them to origin. Carried: OBD `--design centre --budget 60`; per-combo seed check against the pilot; run the rest of OBD in slices; two-process GPU test (optional); keep or delete `backup/laptop-main`.
**Traps:**
- SSH to GitHub fails here (`Permission denied (publickey)`), so the push may have to be run by the user.
- Run from the project dir with `.venv/bin/python` by absolute path.
- OBD centre-point figures skip on smoke logs (`SMOKE` forces epochs=1).
- Adding a Unit field changes every key.
**Pointers:**
- `synaptic_pruning/aggregate.py` (no Wilcoxon; Friedman keeps Holm), `synaptic_pruning/design.py` `block_units`
- `obd/design.py` `_retrain_criteria`, `SEEDS`; `obd/main.py` `run`, `OutOfBudget`
- `dev/agents/DECISIONS.md` D8; `dev/agents/project/notes.md` "Budget"

### Progress · 17:58
- Committed `b4647b0` (synaptic results only; `obd/results/smoke.jsonl` left out and still modified, per user). My push failed: SSH `Permission denied (publickey)`, so the user needs to push.
- Deferred by user: `--tables --plot` and the synaptic review. **Next session, first task:** `cd synaptic_pruning && /home/jake/Code/main/experiments/pruning/.venv/bin/python main.py --tables --plot`, then review against Q3.

## Close · 2026-09-30 17:57 · 234016c..b4647b0
- **changed:** `synaptic_pruning/results/runs.jsonl`: 1860 new v2 records from the user's 150 min budget run, committed and pushed
- **why:** synaptic v2 is complete (1896/1896 runs) and is the data for Q3; saved to origin before the review
- **verified:** `main.py --status` in `synaptic_pruning/` → 1896/1896 done (474/seed × 4); `origin/main` = `b4647b0` after the user's push
- **by:** me (run, push) | claude (commit)
- `b4647b0` synaptic: v2 results, 1896/1896 runs (4 seeds, RTX 5080, 2.03 h)
- 1 file, +1860 −0
- ⚠ uncommitted: `obd/results/smoke.jsonl` (+141 smoke lines; kept out on purpose, per user)

### Next
1. Synaptic tables and figures (put off by the user this session): `cd synaptic_pruning && /home/jake/Code/main/experiments/pruning/.venv/bin/python main.py --tables --plot`, then review against Q3 (`dev/main.md`): dropout vs pruning vs none by dataset/arch, the width/data/epoch trends, and the 6 sweep variants.
2. OBD centre block: `cd obd && /home/jake/Code/main/experiments/pruning/.venv/bin/python main.py --design centre --budget 60`.
3. From the centre results, seeds per combo ≈ (2.8 · SD / δ)². Check the retrain tail Δ (keep ≤ 5%) against the pilot (`8be767f:obd/results/units.jsonl`).
4. Run the rest of OBD in `--budget` slices (priced at ≈ 4.8 h).
5. Carried: keep or delete `backup/laptop-main`; optional two-process GPU throughput test.
### Traps
- SSH push fails from the agent shell (`Permission denied (publickey)`), so the user pushes with `! git push origin main`.
- `obd/results/smoke.jsonl` is tracked though `.gitignore:6` ignores it, and it has local smoke output. Don't commit it.
- The synaptic log still holds 48 old v1 runs that are no longer in the design (1944 lines vs 1896 runs). `--status` ignores them. Check that `aggregate.py` does too.
- OBD centre-point figures skip on smoke logs, because `SMOKE` forces epochs=1.
### Pointers
- `synaptic_pruning/aggregate.py` (no Wilcoxon; Friedman keeps Holm); `synaptic_pruning/design.py` `block_units`
- `obd/design.py` `_retrain_criteria`, `SEEDS`; `obd/main.py` `run`, `OutOfBudget`
- `dev/agents/DECISIONS.md` D8; `dev/agents/project/notes.md` "Budget"
