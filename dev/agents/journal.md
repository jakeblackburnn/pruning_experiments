# Journal · 2026-10-01 15:10 · main · 758d7a5
## Start
**State:** Both grids are finished: OBD 358/358 units (`obd/results/units.jsonl`, 308 CUDA + 50 MPS, 6.2 h), synaptic 1896/1896 runs (`synaptic_pruning/results/runs.jsonl`, 2.1 h). Tables and figures are not generated in the repo (`results/tables/` is git-ignored; no `figures/` for synaptic). Findings for the write-up are in `dev/agents/project/findings.md`.
**Since:** `758d7a5` deleted smoke results. Uncommitted: `obd/results/units.jsonl` +308 lines (the CUDA OBD run) and the user's `dev/journal.md`.
**Next:** User task (dev/journal.md, Oct 1): analysis, mainly collecting and summarizing findings for a write-up. Options: (1) commit the OBD results; (2) drop or filter the 48 off-design runs in `runs.jsonl`; (3) `--tables --plot` for both and pick write-up figures; (4) redo the synaptic summaries with and without the CNN forecaster.
**Traps:** `aggregate.load` reads every line of the log, including 48 seed-0 synaptic runs that are not in the current design; `--status` does not count them. The CNN forecaster is worse than persistence on two datasets and swamps arch-averaged synaptic numbers. No `.venv` on this machine: use `~/Code/python_venv_01_main/bin/python`.
**Pointers:** `dev/agents/project/findings.md` (numbers), `dev/agents/project/notes.md` (machine and budget facts), `dev/agents/DECISIONS.md` D6–D8, pilot OBD log at `8be767f:obd/results/units.jsonl`.

## Close · 2026-10-02 12:10 · 9529d8c..d1c035d
- **changed:** `dev/agents/project/findings.md`: numbers from both finished grids, by research question
- **changed:** `dev/agents/project/report.md`: write-up outline (claims, tables, verdicts, limits, figures) for a human writer
- **changed:** `obd/results/units.jsonl` (+308 CUDA units, grid complete); `obd/figures/`, `synaptic_pruning/figures/` (user-generated plots)
- **why:** runs are done; the user needs the findings collected to write the report. Answer: neither paper's headline grows with scale, and the cheap control catches up with width.
- **verified:** `main.py --status` in both projects → 358/358 and 1896/1896. Numbers in the two docs come from `aggregate.py` functions run in a scratch script this session; no tests exist or were run. The figures were not inspected.
- **by:** pair
- `758d7a5` Delete smoke test results... nothing happens.
- `442adf6` obd: v2 results, 358/358 units (308 CUDA + 50 MPS, 6.16 h)
- `6659176` figures: obd and synaptic plots from the finished grids
- `d1c035d` notes: Oct 1 task, analysis and findings for the write-up
- 14 files, +311 −42 (tracked text; plus 10 new PNGs)
- ⚠ uncommitted: `obd/results/tables/`, `synaptic_pruning/results/tables/` (generated CSVs, meant to be ignored; see Traps)

### Next
1. Fix `.gitignore`: `results/tables/` only matches at the repo root; `**/results/tables/` (and `**/results/smoke*.jsonl`) is what was meant. (S)
2. Decide what to do with the 48 off-design seed-0 runs in `synaptic_pruning/results/runs.jsonl`: delete the lines or filter in `aggregate.load`. Then regenerate synaptic tables and figures. (S)
3. Add the missing OBD control: unpruned net given the same retraining epochs, centre units only. Without it "pruned beats unpruned" on cifar10/mlp and cifar10/resnet is confounded with extra training. (M)
4. Check the committed figures against `report.md` §8 and add the missing ones (accuracy lost at keep 1% vs width; synaptic with and without CNN). (M)
5. Recompute the two "about" numbers in `report.md` §5.2 with a pooled interval. (S)

### Traps
- The analysis scripts for `findings.md`/`report.md` lived in the session scratchpad and are gone. To reproduce: `aggregate.load`, then `center`, `paired`, `summarize` (obd) or `improvements`, `contrast`, `summarize` (synaptic); pruned-vs-unpruned is retrain `test_acc` minus `units.base_test_acc`, merged on `CONFIG + seed`.
- "Without CNN" synaptic numbers exclude the 48 off-design runs; "all architectures" numbers include them.
- The paper headlines in `report.md` §1 are worded from the user's description and the README, not checked against the papers.

### Pointers
- `dev/agents/project/report.md` (outline), `dev/agents/project/findings.md` (fuller tables), `dev/agents/project/notes.md` (durable facts).
- `obd/aggregate.py:113` `paired`, `obd/aggregate.py:206` `center`; `synaptic_pruning/aggregate.py:109` `improvements`, `:130` `contrast`.
