# Journal · 2026-09-25 12:39 · main · 93f7dc5
## Start
**State:** Two paper reproductions share one venv. `obd/` (LeCun 1989 OBD) looks finished: checkpoints, `results/{digits,cifar}.json`, and figs 2–7 plus Hoeffding are all in place, and its README/EXPLANATION give the three lessons. In `synaptic_pruning/` (Vos et al. 2025) the code was revised on Sep 9: 10 trials, mc_dropout dropped, a 3-point seq-len sweep, a 6-tier bitter-lesson *ladder* instead of the 3×3×3 grid, and a `prune_schedule_epochs` decoupled from `epochs`. The results, figures and `run.log` still come from the old Sep 7 run.
**Since:** First journal. The only commit is 93f7dc5, which split the repo into sibling folders.
**Next:** (1) Rerun synaptic_pruning `--replicate` and `--bitter-lesson`, then `--plot` (~2h on M4) · (2) Rewrite REPORT.md for the new results, with the missing §4 and updated caveats · (3) Check `synaptic_pruning.ipynb` against the new results · (4) Clean up stale artifacts (old heatmap/marginal figures, tracked `run.log`)
**Traps:** `results/` and `checkpoints/` are gitignored, so only figures and `run.log` reach git. REPORT.md §0 points to a "§4 rerun" that doesn't exist, and §3's verdict is based on the old grid. `--set` can't override the `tiers` tuple-of-tuples.
**Pointers:** synaptic_pruning/REPORT.md:11 (the fixes it describes), synaptic_pruning/experiments.py:47-56 and :112-124 (new configs), synaptic_pruning/figures.py:152 (ladder figure), synaptic_pruning/synaptic_pruning.md (the original task)

## Close · 2026-09-25 13:10 · 93f7dc5..36bfdbd
- **changed:** `obd/main.py`, `synaptic_pruning/main.py`: a bare run does experiments → figures → `--report` → `--notebook`; `--set` is validated before any stage runs; `_meta` records the git commit and durations
- **changed:** `*/report.py` (new): results JSON → `<!-- report:obd_*|sp_* -->` blocks in `*/RESULTS.md` and the root `README.md`; nbclient notebook execution
- **changed:** docs: minimal root README; per-project `RESULTS.md` templates (Scope, generated blocks, `<!-- fill -->` readings); notebooks rebuilt with live `show(fn)` code tours; EXPLANATION.md, per-project READMEs, run.log and the stale grid figures deleted
- **changed:** `.gitignore` no longer ignores `results/`; `requirements.txt` +scipy +pandas
- **why:** make both reproductions presentable: one command per project regenerates everything, and numbers in the docs can't go stale (workshop `dev/workshop/presentable-state.md`)
- **verified:** OBD `main.py --dataset digits --set epochs=1 sweep_points=3 retrain_min_remaining=4000 hessian_samples=256 device=cpu` → full pipeline OK in 1.5 min; synaptic `--replicate` (tiny) + a tiny 3-tier ladder + `--plot --report --notebook` → OK; `--set bogus=1` rejected up front. Afterwards the real results, checkpoint and figures were restored from backup and `--report --notebook` was rerun on them.
- **by:** pair (workshop decisions by the user, implementation by claude)
- `7e52791` Make each reproduction runnable in one go, with generated result briefs
- `dc3e139` journal: presentable-state workshop, runners and result briefs
- `36bfdbd` Ignore untracked files when marking a results commit dirty
- 23 files, +2923 −1258
- ⚠ uncommitted: `obd/results/`, `synaptic_pruning/results/` (stale: to be committed after the rerun, per D4)

### Next
1. `cd synaptic_pruning && python3 main.py` (~2 h on M4, from a tree with no modified tracked files), then `cd obd && python3 main.py` (digits + cifar; duration unknown). Each ends by regenerating its briefs and notebook.
2. Write the `<!-- fill -->` readings in `obd/RESULTS.md`, `synaptic_pruning/RESULTS.md` and the root README; revisit the notebooks' interpretive sentences against the new numbers.
3. Commit `*/results/*.json`, the figures, the briefs and the notebooks together.
4. Optional: move `synaptic_pruning/synaptic_pruning.md` (the original task prompt, which describes the old 3×3×3 grid) into `dev/`.

### Traps
- Any run overwrites `results/`, `checkpoints/` and figures, even a smoke run. Back them up first (smoke tests this session did this).
- The committed briefs and notebooks currently render **stale** data. Synaptic is the Sep 7 run (5 trials, mc_dropout, 6 seq_lens; the ladder shows "old grid — rerun"). OBD cifar has no `_meta` and no `overlap`.
- The existing digits data doesn't clearly support two drafted OBD claims: at 10% of weights left, magnitude has the lowest no-retrain loss, and in the retrain loop OBD and magnitude stay within ~0.5 pt until the last step. Check the readings against the rerun.
- Run with no modified tracked files, or `_meta.git_commit` gets a `-dirty` suffix (untracked files such as `results/` don't count, as of 36bfdbd).
- zsh: `echo =====` fails (`=cmd` expansion); use `echo "---"`.
- `--set` can't override the synaptic `tiers` tuple-of-tuples; build a `BitterLessonConfig` in a script.

### Pointers
- `dev/workshop/presentable-state.md`: threads, change list, deviations, follow-ups
- `dev/DECISIONS.md` D1–D5: doc layout, generated blocks, bare-run pipeline, commit results, live code tours
- `obd/report.py` `BLOCKS` / `write_all`; `synaptic_pruning/report.py` same
- `synaptic_pruning/main.py` `split_overrides`: how each `--set` field is routed to stages
