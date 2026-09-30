# Journal · 2026-09-30 18:00 · obd-results · 49e291a
## Start
**State:** v2 OBD log `obd/results/units.jsonl` has 50/358 units (seed 0, MPS), committed on `obd-results` (`59d9e2d`). `main` and `origin/main` (local ref) are both at `234016c`, so `obd-results` fast-forwards onto `main`. The synaptic log here has 84 runs (`a7ee240`); the newer synaptic results on the other machine's `main` are not in this clone.
**Since:** nothing after the last Close; the journal was committed at `49e291a`.
**Next:**
1. User's task: get all OBD results into `main` and bring the journal up to date so the OBD experiments continue on the other machine. Blocked here: `git fetch` and `ssh -T git@github.com` fail (`Permission denied (publickey)`), `gh` is not logged in. So the synaptic results on `origin/main` cannot be fetched and nothing can be pushed from this Mac.
2. Once GitHub works (fix the key, or `! gh auth login`): `git fetch`, `git checkout main`, merge `obd-results` (the journal is the file likely to conflict), push.
3. Carried forward: decide whether MPS and CUDA results share one log; finish the centre block with `--design centre`; centre seed-noise and retrain tail Δ checks against `8be767f:obd/results/units.jsonl`.
**Traps:**
- This Mac is MPS only; the 8 h pricing was for the RTX 5080.
- Default `main.py --budget N` is seed-major, not centre-first; pass `--design centre` for that.
- A unit cut off at the deadline is not logged and reruns first next time.
- Run from `obd/` with `/Users/jackblackburn/code/main/obd/.venv/bin/python` by absolute path.
- SSH to GitHub fails here, so local `origin/main` is stale and a push would need a fetch and merge first.
**Pointers:** `obd/main.py` `run`; `obd/design.py` `SEEDS`; `dev/agents/DECISIONS.md` D8; `dev/agents/project/notes.md`
