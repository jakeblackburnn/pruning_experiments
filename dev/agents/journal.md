# Journal · 2026-09-30 · main · 7ddb891
## Start
**State:** `main` was reset to `origin/main` (`7ddb891`) after the laptop's mistaken rebase; the four local commits are kept on branch `backup/laptop-main` (`4085ffe`). OBD (`obd/`) and synaptic pruning were rebuilt as budgeted, resumable factorial designs; `obd/results/units.jsonl` held 14 of 1830 units (Mac/MPS).
**Since:** `5c79a47` Close; Mac OBD run and BatchNorm-on-CPU fix (`880f681`), then `7ddb891` results.
**Next:** see the task below; carried forward: check cifar10/resnet saliency (0.709 vs magnitude 0.796, base 0.765) on CUDA with more seeds.
**Traps:** changing an override or a `design.py` constant changes unit keys; `--budget` overruns by up to one unit; SSH to GitHub is not authorised on this machine (`Permission denied (publickey)`), so `git fetch` fails here.
**Pointers:** `obd/design.py`, `obd/main.py`, `dev/agents/project/notes.md`.

## Task: sync to remote, then 2 h OBD run · 2026-09-30
**Goal:** local repo equals `origin/main`, then a 2-hour OBD slice on the RTX 5080 adds real CUDA results.
**Now:**
- Was mid-rebase (4 laptop commits onto `7ddb891`) with conflicts; aborted, backed up, hard-reset to `origin/main`.
- `origin/main` is as of the last successful pull (`7ddb891`); a re-fetch fails on SSH auth, so newer remote commits are unverified.
- CUDA smoke (`--smoke --seeds 1`, 141 units) passed in 4m48s. This was the first CUDA verification.
- `obd/results/units.jsonl`: 14/1830 done, est. 329 h remaining.
**Scope:** in: git sync, `python main.py --budget 120` in `obd/`. Out: synaptic_pruning, design changes, commits.
**Constraints:** venv `~/Code/python_venv_01_main`; one GPU; no design edits (unit keys). Assumption: "2 hour" = `--budget 120` (may overrun by one unit).
**Approach:**
- Abort rebase, branch `backup/laptop-main`, `git reset --hard origin/main`.
- Smoke on CUDA, then `--budget 120` in the background (log in the scratchpad `obd_run.log`).
- After: `--status`, `--tables`; compare CUDA vs Mac results by unit key.
**Risks:** ResNet/CIFAR units take 20+ min, so few units fit; the CPU Hessian (BatchNorm under CUDA) is untouched and unverified at real size; the laptop's discarded commits (rerun results and readings) exist only on the backup branch.
**Done when:** `git status` shows `main` equal to `origin/main`; the run exits with `units.jsonl` growing by complete units; `--status` reports the new count.
**Open:**
1. The backup branch holds unmerged rerun results and readings; keep or delete it? (Kept for now.)
2. Should the results from this run be committed? (Not committing unasked.)

## Close · 2026-09-30 12:45 · 22dfdb3..a6729f9
- **changed:** git: `main` reset to `origin/main` (`7ddb891`), laptop's 4 commits kept on `backup/laptop-main` (`4085ffe`); `obd/results/units.jsonl`: +85 CUDA units (99/1830, all seed 0); `dev/agents/project/notes.md`: 8 h whole-suite budget and how to shrink
- **why:** discard the accidental laptop rebase; get first real CUDA OBD results and a timing base; user needs the full suite (OBD + synaptic) under 8 h
- **verified:** `main.py --smoke --seeds 1` on CUDA → 141 units done, 4m48s; `main.py --budget 120` → "ran 85 unit(s) this session", then `--status` → 99/1830, est. 81.2 h left. Not verified: synaptic on CUDA (0/5760 runs, no timing); `--tables`/`--plot` on the new log (tables in `obd/results/tables/` are stale); the trimmed designs (nothing implemented yet)
- **by:** claude
- `a6729f9` Add CUDA results: 85 more OBD units (seed 0)
- 1 file, +85 −0 (results)
- ⚠ uncommitted: `obd/results/smoke.jsonl` (CUDA smoke output, overwrote the Mac one)

### Next
1. `python main.py --tables` in `obd/`, read `results/tables/effects.csv` and `paired_*.csv` to see which factors matter (input to the trim).
2. `cd synaptic_pruning && python main.py --budget 5`, then `--status` to get its cost model.
3. Trim both `design.py` files per `dev/agents/project/notes.md` ("Budget"): target ≈ 4 h each; check `--status` estimates.
4. Run seed 0 of the trimmed designs, review tables, then decide seeds 1–2.
5. Decide: keep or delete `backup/laptop-main`.
### Traps
- SSH to GitHub fails here (`Permission denied (publickey)`); `git fetch/pull/push` need a key or `gh` auth.
- `units.jsonl` mixes Mac (MPS) and CUDA runs: durations are not comparable across devices.
- Scratchpad `obd_run.log` holds the run log; the Bash background task for `nohup` reports "completed" when the launcher shell exits, not the run.
### Pointers
- `dev/agents/project/notes.md` (budget and shrink plan), `obd/design.py:34-62` (blocks), `synaptic_pruning/design.py` `block_units`, `dev/agents/DECISIONS.md`

## Close (addendum) · 2026-09-30 13:05 · a6729f9..HEAD
- **changed:** ran `synaptic_pruning/main.py --budget 5` on CUDA: 84 of 5760 runs in `synaptic_pruning/results/runs.jsonl` (seed 0); timing added to `dev/agents/project/notes.md`
- **why:** timing base for the 8 h budget; synaptic had no cost model
- **verified:** `main.py --budget 5` → "ran 84 run(s) this session"; `--status` → est. 5.5 h left (+120 runs with no estimate). Not verified: tables/figures for synaptic; the estimate is from only 84 shuffled runs (transformer/ep60 tail is thinly sampled)
- **by:** claude
- Synaptic full grid ≈ 5.5 h, OBD ≈ 81 h remaining: the budget cut is needed mostly on OBD; synaptic needs about a 2× trim (e.g. 3 seeds ≈ 3.3 h) to fit next to a ≈ 4 h OBD.
- ⚠ uncommitted: `obd/results/smoke.jsonl`
### Next
1. Unchanged from the Close above; step 2 (synaptic timing) is done.
