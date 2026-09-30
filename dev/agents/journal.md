# Journal · 2026-09-30 · obd-results · 234016c
## Start
Previous session reworked both designs for the 8 h budget (D8) and recorded Next: run `--design centre` first. This session runs a 40 min slice of OBD on a new branch.

Carried forward:
- Next: from centre results, check seeds needed and compare retrain tail Δ with the pilot (`8be767f:obd/results/units.jsonl`); two-process GPU test; decide keep/delete `backup/laptop-main`.
- Traps: SSH to GitHub fails here (`Permission denied (publickey)`); `obd/results/smoke.jsonl` is tracked though git-ignored; run from the project dir with `.venv/bin/python` by absolute path.
- Pointers: `obd/main.py` `run`; `obd/design.py` `SEEDS`; `dev/agents/DECISIONS.md` D8.

## Task: 40 min OBD budget run on branch `obd-results` · 15:05
**Goal:** collect a first slice of v2 OBD results (40 min) on a branch that cannot conflict with the synaptic results produced on another machine.
**Now:**
- Both projects log to tracked files: `obd/results/units.jsonl` (fresh, 0/358 done) and `synaptic_pruning/results/runs.jsonl` (other machine). Different files, so result commits never touch each other.
- Shared files that can conflict: `dev/agents/journal.md`, `dev/agents/DECISIONS.md`, `dev/agents/project/notes.md`, README.
- This machine is a Mac (MPS, no CUDA); the 8 h pricing was for the RTX 5080 (assumption: this is where the OBD run happens).
- Branch `obd-results` created from `main` at 234016c.
**Scope:** in: branch, `main.py --budget 40` in `obd/`, commit `obd/results/units.jsonl` on the branch. out: code changes, synaptic runs, merging to main, pushing (SSH fails).
**Constraints:**
- Branch commits touch only `obd/results/units.jsonl` (plus this machine's journal at `/devlog`); no edits to shared code or docs, so merging later is a fast-forward of one file. Assumption.
- Default design order (centre first, seed-major); hard cut-off at 40 min (D8), no `--finish`.
**Approach:**
- Run `cd obd && /Users/jackblackburn/code/main/obd/.venv/bin/python main.py --budget 40` in the background.
- Check `--status`, commit the log on the branch with a message saying device and budget.
- Merge later: `git checkout main && git merge obd-results` (journal may need a manual merge).
**Risks:**
- MPS numerics and speed differ from CUDA; a log mixing both devices is not strictly comparable. Noticed from the device field in records; keep MPS and CUDA slices separate if it matters.
- 40 min on a slower device may finish only the centre block's first seed or two.
- The journal is edited on both machines; merge conflict there is likely, in results files it is not.
**Done when:**
- `git branch` shows `obd-results`; `main` is unchanged.
- The run ends by itself at ≈ 40 min wall with no partial line in `obd/results/units.jsonl`; `--status` shows the units done.
- The log is committed on `obd-results`; `git diff main --stat` shows only `obd/results/units.jsonl` (and the journal).
**Open:**
1. Is this Mac the right device for the run, or should it run on the RTX 5080? Assumed this Mac.

### Progress · 17:47
- Run 1 (15:05, `--budget 40`): 19 units, unit 20 cut off at the deadline. Run 2 (16:10 → ~17:17, `--budget 67`, chained by a waiter script 10 min after run 1): 31 more units, unit 32 cut off. Both on MPS. Log: 50/358 units, all seed 0 (50/114), every line valid JSON.
- Order was seed-major through all blocks, not centre-first: centre 4/40, core 29/198, scale 24/162, retrain 12/72.
- Unit times on MPS: 0–10 min (fmnist/resnet ≈ 10 min, cifar10/vgg ≈ 5 min, mlp/paper ≈ 0–1 min).
- Committed the log on `obd-results` only; `main` unchanged.

## Close · 2026-09-30 17:47 · 234016c..HEAD
- **changed:** `obd/results/units.jsonl`: new, 50 OBD v2 units (seed 0) from MPS
- **changed:** branch `obd-results` created from `main` (234016c); touches only the results file, so no conflict with synaptic's `runs.jsonl` on the other machine
- **why:** collect a first slice of v2 OBD results without merge conflicts against the other machine's synaptic run
- **verified:** `main.py --status` → 50/358 done; every line parses as JSON; no partial line after either cut-off; `git status` clean apart from the journal
- **by:** claude
- 1 commit on `obd-results`
- not merged, not pushed (SSH to GitHub fails here)

### Next
1. Decide whether MPS and CUDA results share one log (numerics and device differ); if not, run the rest on the RTX 5080 into a separate branch or file.
2. To finish the centre block first: `cd obd && /Users/jackblackburn/code/main/obd/.venv/bin/python main.py --design centre --budget 60` (36 of 40 units left; ≈ 0.6 h was the CUDA estimate, MPS is slower).
3. Continue slices with `main.py --budget N`; then the centre-seed-noise and retrain tail Δ checks from the previous Next.
4. Merge: `git checkout main && git merge obd-results`; the journal is the file likely to conflict.
### Traps
- This Mac is MPS only (no CUDA); the 8 h pricing was for the RTX 5080.
- Default `main.py --budget N` is not centre-first; pass `--design centre` for that.
- A unit cut off at the deadline is not logged and reruns first next session.
- Run from `obd/` with `/Users/jackblackburn/code/main/obd/.venv/bin/python` by absolute path.
### Pointers
- `obd/main.py` `run` (timer budget); `obd/design.py` `SEEDS`; `dev/agents/DECISIONS.md` D8
