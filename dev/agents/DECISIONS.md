# Decisions

## D6 · 2026-09-29 · Budgeted, resumable factorial runs; no generated docs
**Decision:** Each project runs a factorial design of small units (`design.py`), with seeds as the blocking factor. `main.py --budget MIN` runs seed-major and stops starting units when the budget is spent. Every finished unit is appended to `results/*.jsonl`, and that log is the resume state. `aggregate.py` turns the log into CSV tables and figures. There are no notebooks, `RESULTS.md`, `report.py` or generated blocks, and the root README holds setup and run commands only.
**Why:** The user runs the full grids over days on an RTX 5080 and needs to split them across sessions; a balanced partial design must stay analysable. Results are written up by the user, so generated prose and number blocks were only more to keep in sync.
**Rejected:** `--resume` that skips stages by `_meta` match: a cell-level log needs no matching. One results JSON per stage: a crash lost the whole stage.
**Overturns:** D1 (RESULTS.md and notebooks), D2 (generated blocks), D3 (no resume), D5 (notebook code tours); D4's form (per-dataset JSON) but not its point: results are committed, checkpoints are not (OBD no longer saves checkpoints).
**Revisit if:** the log grows past tens of MB, or one unit outgrows a budget slice.

## D7 · 2026-09-29 · Scale is three independent axes; no "bitter lesson" ladder
**Decision:** Model size (width), data (training fraction) and compute (epochs) are separate crossed factors in both projects. The joint size+data+compute ladder and the "bitter lesson" framing are dropped.
**Why:** The ladder confounded the three, and the user's `dev/journal.md` asks to drop the framing.
**Revisit if:** a question needs the joint path (compute-optimal scaling).
