# Workshop: presentable state (runners, result briefs, notebooks)
2026-09-25 · status: closed

## Context
The repo holds two paper reproductions: `obd/` (LeCun 1989) and `synaptic_pruning/` (Vos 2025). The user wants the repo in a
presentable state: (1) each project's whole experiment suite runs in one command; (2) template result briefs in md and in
the README; (3) notebooks that explain both the code and the results. Today the synaptic results, figures, REPORT.md and
notebook are all from an older experiment design. OBD's cifar results are partial. The docs are split differently in the two projects.

**Material:** `README.md`, `.gitignore`, and per project: `main.py`, `README.md`, the notebook and `figures/`.
Also `obd/EXPLANATION.md`, `obd/hoeffding.py`, `synaptic_pruning/REPORT.md` and `synaptic_pruning/run.log`.

## Tensions
| # | tension | depends on | status | outcome |
|---|---|---|---|---|
| 1 | The two projects split docs differently, and the roles overlap | — | decided | one root README; per project `RESULTS.md` + notebook; EXPLANATION.md folded into the notebook |
| 2 | Result numbers are typed by hand into prose, so they go stale | 1 | decided | `--report` regenerates marked number blocks; interpretation is written by hand |
| 3 | "Run everything" is incomplete or breaks in both runners | — | decided | a bare run does experiments → figures → report → notebook; no resume; `--set` checked up front; timings |
| 4 | Notebooks say little about the code; synaptic's is stale and half-executed | 1, 2 | decided | live `show(fn)` source, placed alongside each result; numbers from data, claims revisited after each run |
| 5 | Provenance: committed outputs come from results git doesn't hold | 3 | decided | commit `results/*.json` (not checkpoints); `_meta` gains the commit and duration |
| 6 | Orphan and stale artifacts | — | decided | stale files, references and cells removed; hoeffding kept in obd/, documented |
| 7 | With one README, where does the per-project detail go? | 1 | decided | minimal README; flags → `--help`; departures → RESULTS Scope; algorithm + code map → notebook |

## Threads
### 1. Doc roles
Options considered: README + RESULTS + notebook; the same plus EXPLANATION.md in both; the notebook as the only report. **Decided:** the first,
with the per-project READMEs merged into the **root README** at the user's request. Each project has `RESULTS.md` (the brief) plus its notebook
(the long-form walkthrough of code and results). `obd/EXPLANATION.md` goes into the OBD notebook and is then deleted.

### 2. Brief form
Options considered: generated tables with hand-written prose; a fully generated template with {slots}; a hand-filled skeleton. **Decided:** generated
tables with hand-written prose. `--report` rewrites only the `<!-- report:<name> -->…<!-- /report -->` blocks, in `RESULTS.md` and in the project's
README brief. Each block ends with a provenance line (date, results file, commit, duration). Interpretation goes in `<!-- fill: … -->`
sections written by hand. A fully generated template would fix the wording before anyone has seen the data; a hand-filled skeleton is what went stale.

### 3. Runner
**Decided:** a bare `python3 main.py` produces everything presentable: experiments → figures → `--report` → execute the notebook
in place. Each step still has its own flag. There is **no resume flag**: stages already cache one results file each, so after a crash
you rerun the failed stage by its flag, which avoids quietly reusing stale results. Decided on the spot: OBD runs all datasets by default,
synaptic checks `--set` before any stage starts, and both print a timing summary at the end.

### 4. Notebooks
**Decided:** a `show(fn)` helper displays the live source (`inspect.getsource`), so the code shown can't drift. Code sits alongside the results:
a short pipeline map first, then each results section opens with the function that produced it, then the figure, then the commentary.
Rule, the same as for RESULTS.md: every number comes through `note()` or computed output, and interpretive claims are revisited by hand after each
full run. Rejected: hand-copied excerpts (they drift), prose plus links (too much jumping on GitHub), a code-then-results split
(code read far from its result), and verdict words computed in code (stilted prose, arbitrary thresholds).

### 5. Provenance
**Decided:** commit `results/*.json` (about 450 KB). `checkpoints/` and `data/` stay ignored. `_meta` gains `git_commit` (with `-dirty`
if the tree has uncommitted changes) and `duration_s`, and the report's provenance line reads them.

### 6. Orphans and stale artifacts
Decided on the spot: delete the stale synaptic heatmap and marginal figures, untrack `run.log`, delete `obd.ipynb` cells 32-33 (copies of
20-21), and fix the stale references and four-method wording. **Hoeffding:** kept in `obd/` at the user's request, with one line in the README.

### 7. Per-project detail
**Decided:** minimal README. Flags go to each `main.py` docstring (`--help`), departures from the paper to `RESULTS.md §Scope`,
and the algorithm, design notes (for example `prune_schedule_epochs`) and code map to the notebook.

## Change list (as approved)

**Root**
- `README.md`: rewrite as minimal (thread 7). It covers setup, then per project: one paragraph, `cd <p> && python3 main.py` with a rough runtime,
  a **Results brief** containing a `<!-- report:<p>_headline -->` block plus 2-3 `<!-- fill -->` bullets, and links to `RESULTS.md` and the notebook.
  Hoeffding gets one line under OBD. (Threads 1, 2, 6, 7)
- `.gitignore`: drop `results/`. (Thread 5)
- Delete `obd/README.md` and `synaptic_pruning/README.md` once their content has been moved (flags → docstrings, departures → RESULTS,
  algorithm/code map/design notes → notebooks). (Thread 7)

**obd/**
- `main.py`: `--dataset` takes several values and defaults to all. New `--report` and `--notebook` stages. A bare run does, for each dataset,
  train → prune → plot, then report → notebook once at the end. `_meta` adds `git_commit` and `duration_s`, and records base-model
  train/test metrics so the report needs no torch. Timing summary at the end. The docstring absorbs the README's flag detail and the `--only` table. (Threads 3, 5, 7)
- `report.py` (new, pure: JSON → markdown, like `figures.py`): `render(results_by_dataset) -> {block: md}`, `write_blocks(path, blocks)`
  (it errors on a missing marker, so templates and code can't drift apart), and `execute_notebook(path)` via `nbclient` (already in the venv).
  Blocks: `obd_headline`, `base_models`, `sweep_summary` (loss increase at fixed remaining counts, per ranking), `retrain_summary`, `overlap_stats`. (Thread 2)
- `RESULTS.md` (new template): provenance line; **Scope** (16×16 MNIST stand-in, paper sizes, Gauss-Newton diagonal, CIFAR as the modern
  variant); §1 does saliency beat magnitude (Fig 2); §2 forecast accuracy (Fig 3); §3 prune-retrain (Fig 4); §4 CIFAR; §5 is magnitude OBD in
  disguise (Figs 5-7); **Verdict**; **Caveats**. Each section: a figure link, a report block, and a `<!-- fill -->` reading. The current "three lessons"
  text becomes the starting draft of the readings, marked for review after the rerun. (Threads 2, 7)
- `obd.ipynb`: add `show()` to the setup cell; add a pipeline map cell (condensed from EXPLANATION §1-4, 8-10); place code tours alongside the results
  (`sigmoid`/`MnistNet` → base net; `diagonal_hessian`, `saliencies` + EXPLANATION §5 → saliency; `sweep_no_retrain` → Fig 2;
  `iterative_prune_retrain` → Fig 4; `overlap_analysis` + the EXPLANATION §7 derivation → §6); end with an "Extending" appendix (EXPLANATION's last section)
  and the code map; delete cells 32-33. (Threads 1, 4, 6)
- `EXPLANATION.md`: delete after it has been moved into the notebook. (Thread 1)

**synaptic_pruning/**
- `main.py`: `--set` is parsed once; each field must exist in at least one stage config, otherwise error before anything runs; each stage gets
  only the fields it has. New `--report` and `--notebook`. Bare run: replicate → bitter-lesson → plot → report → notebook. `_meta` adds `git_commit`
  and `duration_s`. Timing summary. The docstring absorbs the README's runtime estimates and the note that `tiers` can't be set via `--set`. (Threads 3, 5, 7)
- `report.py` (new, same shape as OBD's): blocks `sp_headline`, `replication_table` (mean MAE by model×seq×method + Friedman p),
  `replication_winrate`, `ladder_table` (tier, h/e/n, none, pruning, Δ%, CI overlap), `ladder_trend` (slope per tier). (Thread 2)
- `REPORT.md` → `git mv` to `RESULTS.md` and rewrite as a template: provenance; **Scope** (README departures + REPORT §0 condensed into
  "changes since the first attempt"); §1 Replication; §2 Bitter-lesson ladder; **Verdict**; **Caveats**. Each has a block and a `<!-- fill -->`. The stale numbers are removed. (Threads 1, 2, 7)
- `synaptic_pruning.ipynb`: port `note`/`table`/`show` from OBD's setup cell; add a pipeline map and code map; place code tours alongside the results
  (`cubic_schedule`, `SynapticPruner` → §1 with the algorithm text from the README; `train_one` → §2; `run_bitter_lesson` + the
  `prune_schedule_epochs` design note → §3); make cell 8 say "three methods"; replace cell 14's hard-coded verdict with `note()`-driven numbers plus a
  hand-written reading to fill in after the run. (Threads 4, 6)
- `figures/bitter_lesson_heatmaps.png`, `figures/bitter_lesson_marginals.png`: delete. `run.log`: `git rm`. (Thread 6)

**dev/**
- `DECISIONS.md` (new) in `/decide` format: D1 doc layout (threads 1+7), D2 generated number blocks + hand-written readings (2, 4c),
  D3 a bare run produces everything presentable, with no resume (3), D4 commit results, not checkpoints (5), D5 live code tours placed alongside the results (4a, 4b).
- `workshop/presentable-state.md`: this record, set to `closed` with *Changes applied* filled in.

## Changes applied
All items in the change list were applied, with these deviations found while doing it:
- **Report markers are scoped by prefix** (`obd_*`, `sp_*`), because both projects write into the one root README. Each `report.py` touches only its own blocks.
- **The OBD retrain metric changed.** "Fewest weights within 0.5 pt of base" turned one 0.3-pt noise dip into a headline of "83% vs 20% removable". It was replaced with test accuracy side by side at fixed fractions (50/20/10/5% and the last step).
- **OBD `_meta` records per-experiment `durations_s`** (merged across `--only` reruns) rather than a single duration. Base-model metrics weren't added, because the first point of every series already is the base.
- **Guards for the old results format.** The ladder figure, the notebook and the report print "rerun `--bitter-lesson`" when `results/bitter_lesson.json` is from the 3×3×3 grid, instead of crashing.
- **`requirements.txt` gained `scipy` and `pandas`.** Both were imported but not listed.
- **Code comments that pointed at the README** (`obd.py`, `experiments.py`, `models.py`, `datasets.py`) now point at RESULTS.md or the notebook.
- **Notebook references.** The OBD notebook's "§8 below" pointed at a section that didn't exist; there's now a References appendix.

Verified: `--help` in both projects; `--set bogus=1` and `--bitter-lesson --set epochs=2` fail at parse time. A smoke run of the full bare OBD pipeline (tiny digits) took 1.5 min, and synaptic stages with tiny configs ran through plot → report → notebook. Blocks were replaced, markers are paired, `_meta` carries the commit and durations, and nbclient ran both notebooks with no cell errors on the venv kernel. Afterwards the real results, checkpoint and figures were restored from backup, and `--report --notebook` was rerun on them.

**State left behind:** the briefs and notebooks are rendered from the *existing* results. OBD digits is current; OBD cifar predates `_meta` and has no `overlap`. Synaptic is the Sep 7 run from the old design, so its tables show 5 trials, `mc_dropout`, six seq_lens, and "old grid — rerun" for the ladder. The full rerun (below) replaces all of it.

## Verification (as planned)
1. `python3 main.py --help` in both projects shows the full flag docs.
2. Synaptic: `--set bogus=1` errors immediately; `--set epochs=2` on a bare run doesn't raise at the ladder stage.
3. Smoke runs (fast configs): OBD `--dataset digits --set epochs=1 sweep_points=3 retrain_min_remaining=4000` through report + notebook;
   synaptic `--replicate --set trials=1 epochs=2 seq_lens=1 model_types=rnn`, a tiny-tier ladder from a `python -c` config, then
   `--plot --report --notebook`. Check that marker blocks in `RESULTS.md` and the README were replaced, the notebooks ran without errors (nbclient raises on
   a cell error), and `_meta` has the commit and duration. Confirm the notebook kernel is the venv's python (`sys.executable` printed in the setup cell).
4. `grep -rn 'report:' README.md */RESULTS.md` shows every block paired with a `/report`; `grep -rn 'fill:'` lists the readings still to write.
5. Re-read the edited docs once: does each summary still match its body, and are there any leftover links to deleted files (`EXPLANATION.md`, `REPORT.md`, per-project READMEs)?

## Follow-ups (outside the material, not applied)
1. **Commit the code changes** (runners, report.py, templates, notebooks, deletions) so the rerun's `_meta.git_commit` is clean.
2. **Full rerun of both suites** from the committed tree: `cd obd && python3 main.py` (digits + cifar; the duration is unknown, and `_meta` records it from now on) and `cd synaptic_pruning && python3 main.py` (about 2 h on M4). Each ends by regenerating its briefs and notebook.
3. **Write the `<!-- fill -->` readings** in both RESULTS.md files and the README bullets, and revisit the notebooks' interpretive sentences (OBD's §3–§6 notes and the CIFAR note) against the new numbers.
4. **Commit results, figures, briefs and notebooks** as the "presentable state" commit. The journal is closed with `/devlog`.
5. `synaptic_pruning/synaptic_pruning.md` (the original task prompt, which still describes the 3×3×3 grid): consider moving it to `dev/`.
