# Decisions

## D1 · 2026-09-25 · One README; per project a RESULTS.md and a notebook
**Decision:** The repo has a single minimal root README (setup, run commands, a results brief per project). Each project has `RESULTS.md` (the brief, with its Scope section) and a notebook (the long-form walkthrough of the code and the results). `EXPLANATION.md` and the per-project READMEs are folded into these.
**Why:** The two projects split docs differently (OBD had a code walkthrough but no results file; synaptic had the reverse), and their content overlapped. The user wanted one README for the whole repo. Flags live in each `main.py --help`, so the README doesn't need them.
**Rejected:** Also keeping EXPLANATION.md in both projects: four docs per project to keep in sync.
**Rejected:** The notebook as the only report: results can't be read on GitHub without rendering the notebook.
**Rejected:** Keeping all per-project detail in the root README: roughly 300 lines, and it duplicates the notebooks.
**Revisit if:** A third project is added and the root README stops being skimmable.

## D2 · 2026-09-25 · Generated number blocks, hand-written readings
**Decision:** `main.py --report` rewrites only the `<!-- report:<prefix>_<name> -->` blocks in `RESULTS.md` and the root README, from `results/*.json`, each ending with a provenance line. Interpretation is written by hand in `<!-- fill -->` sections after each full run. The notebooks follow the same rule: numbers come through `note()`, and claims are revisited after each run.
**Why:** The previous REPORT.md and notebook verdict quoted numbers from an experiment design the code no longer ran. Hand-typed numbers go stale silently, while a marker with no rendered block fails loudly.
**Rejected:** A fully generated template with {slots}: it fixes the wording ("shrinks", "coin flip") before anyone has seen the data.
**Rejected:** A hand-filled skeleton: this is how the staleness happened.
**Rejected:** Verdict words computed from thresholds: stilted prose and arbitrary cut-offs.
**Revisit if:** The readings keep drifting from the blocks after reruns. That would mean the hand-written half needs a check too.

## D3 · 2026-09-25 · A bare `main.py` produces everything presentable; no resume
**Decision:** In each project, `python3 main.py` with no flags runs every experiment (all OBD datasets), then figures, then the report, then executes the notebook in place. Every step keeps its own flag. `--set` is validated for every stage before the first one starts. There is no `--resume`.
**Why:** One command should leave the results, figures, briefs and notebook outputs all from the same run. Each stage already caches one results file, so after a crash you rerun just that stage's flag.
**Rejected:** Stopping at figures, or at the report: the notebook outputs, or the briefs, could drift from the results.
**Rejected:** `--resume` that skips stages whose `_meta` matches: more code, and a wrong match would quietly keep stale results.
**Revisit if:** A single stage grows long enough that losing it to a crash hurts (the synaptic ladder is about 90 min today).

## D4 · 2026-09-25 · Commit results, not checkpoints
**Decision:** `results/*.json` are committed; `checkpoints/` and `data/` stay ignored. `_meta` records the git commit (with `-dirty` if the tree has uncommitted changes) and durations, and the report's provenance line reads them.
**Why:** The figures, briefs and notebook outputs are committed, so the data they come from should be too. The results total about 450 KB; the checkpoints are 2.4 MB of CIFAR weights, and the datasets are downloaded.
**Rejected:** Keeping results ignored: the provenance line would be the only record of which run produced the numbers.
**Revisit if:** The results files grow into the tens of MB.

## D5 · 2026-09-25 · Notebooks show live source, placed alongside the results
**Decision:** The notebooks show code with a `show(fn)` helper (`inspect.getsource`). A short pipeline map comes first; then each results section opens with the function that produced it, followed by the figure and the commentary.
**Why:** The code shown can't drift from the code that ran, and each result is read next to its code.
**Rejected:** Hand-picked excerpts in markdown: they drift, as EXPLANATION.md could.
**Rejected:** Prose plus `file:line` links: too much jumping between files on GitHub.
**Rejected:** Part I code, Part II results: the code is read far from the result it produced.
**Revisit if:** A function worth showing grows too long to read inline even with `until=`.
