"""Synaptic pruning (Vos et al. 2025) against other regularizers: a budgetable,
resumable factorial run.

    python3 main.py --budget 60      run for about an hour, then stop
    python3 main.py --budget 600     ...run again to continue: finished runs are skipped
    python3 main.py                  run everything that is left
    python3 main.py --status         progress per block and seed, time left
    python3 main.py --tables         summary CSVs  -> results/tables/
    python3 main.py --plot           figures       -> figures/

The design (design.py) is a list of runs, each one model trained under one
method (train.py). Each finished run is appended as one JSON line to
results/runs.jsonl (the resume state: a run is done once its line exists).
Runs go seed-major, so any stop leaves complete seeds and a balanced design;
a run expected to overrun the remaining budget is not started. Ctrl-C loses
at most the run in flight.

Design selection:

    --design core scale sweep     blocks to run (default: all; see design.py)
    --seeds N                     seeds per configuration (default 5)
    --dataset air_quality ...     restrict the design
    --arch lstm cnn ...           restrict the design
    --method none pruning ...     restrict the design
    --set FIELD=VALUE ...         override Unit fields (lr, batch_size, epochs, ...)
    --smoke                       tiny fast version -> results/smoke.jsonl

Runs on CUDA if available, else Metal, else CPU (`--device` pins it).
"""

import argparse
import dataclasses
import datetime
import json
import subprocess
import time
from pathlib import Path

import torch

import design
from datasets import DATASETS
from models import ARCHS, resolve_device
from train import METHODS, Unit, train_one

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
OVERRIDABLE = ("lr", "batch_size", "epochs", "warmup", "prune_every", "n_cap")
SMOKE = {"epochs": 2, "n_cap": 200}


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

def git_commit(root=ROOT):
    """Short commit hash, with a -dirty suffix if tracked files are modified."""
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True,
                                        text=True, check=True).stdout.strip()
        dirty = run("status", "--porcelain", "--untracked-files=no")
        return run("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def run_unit(unit, device):
    """Train one model; returns the record appended to the log."""
    start = time.perf_counter()
    result = train_one(unit, device)
    return {"key": design.key(unit), "unit": dataclasses.asdict(unit), **result,
            "_meta": {"git_commit": git_commit(), "device": device,
                      "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                      "torch": torch.__version__,
                      "duration_s": round(time.perf_counter() - start, 2),
                      "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}}


# --------------------------------------------------------------------------
# The log: results/*.jsonl, one finished run per line
# --------------------------------------------------------------------------

def load_log(path):
    """Finished runs, in file order. A truncated last line (a run killed
    mid-write) is ignored."""
    records = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def append_record(path, record):
    RESULTS_DIR.mkdir(exist_ok=True)
    text = path.read_text() if path.exists() else ""
    prefix = "\n" if text and not text.endswith("\n") else ""
    with open(path, "a") as f:
        f.write(prefix + json.dumps(record, allow_nan=False) + "\n")
        f.flush()


def fmt_duration(seconds):
    hours = seconds / 3600
    return f"{hours:.1f} h" if hours >= 1 else f"{seconds / 60:.0f} min"


# --------------------------------------------------------------------------
# Running with a budget
# --------------------------------------------------------------------------

def run(units, path, device, budget_min=None):
    records = load_log(path)
    done_keys = {r["key"] for r in records}
    pending = [u for u in design.ordered(units) if design.key(u) not in done_keys]
    cost = design.CostModel(records)
    deadline = time.monotonic() + budget_min * 60 if budget_min else None
    print(f"{len(units) - len(pending)}/{len(units)} runs done; {len(pending)} to go"
          + (f"; budget {budget_min:g} min" if budget_min else "") + f"; device {device}")

    ran = 0
    try:
        while pending:
            left = deadline - time.monotonic() if deadline else float("inf")
            pick = next((u for u in pending
                         if (est := cost.estimate(u)) is None or est <= left), None)
            if left <= 0 or pick is None:
                print("budget used up; run again to continue")
                break
            pending.remove(pick)
            est = cost.estimate(pick)
            print(f"[{ran + 1}] {pick.dataset}/{pick.arch} w{pick.width} data{pick.train_frac:g} "
                  f"ep{pick.epochs} seq{pick.seq_len} {pick.method} "
                  f"seed{pick.seed}" + (f"  (~{est:.0f}s)" if est else ""), flush=True)
            record = run_unit(pick, device)
            append_record(path, record)
            cost.add(record)
            ran += 1
            print(f"    test MAE {record['test_mae']:.4f}  ({record['_meta']['duration_s']:.0f}s)",
                  flush=True)
        else:
            print("all runs done")
    except KeyboardInterrupt:
        print("\ninterrupted; the run in flight is lost, everything else is saved")
    print(f"ran {ran} run(s) this session")


def status(units, block_keys, path):
    records = load_log(path)
    done = {r["key"] for r in records}
    cost = design.CostModel(records)
    print(f"{path.name}: {sum(design.key(u) in done for u in units)}/{len(units)} runs done")
    print("\nblock      done / total  (blocks overlap where configurations coincide)")
    for block, keys in block_keys.items():
        print(f"  {block:<9}{len(keys & done):>6} / {len(keys)}")
    print("\nseed       done / total")
    for seed in sorted({u.seed for u in units}):
        keys = {design.key(u) for u in units if u.seed == seed}
        print(f"  {seed:<9}{len(keys & done):>6} / {len(keys)}")
    remaining = [u for u in units if design.key(u) not in done]
    ests = [cost.estimate(u) for u in remaining]
    known = [e for e in ests if e is not None]
    if remaining:
        print(f"\nremaining: {len(remaining)} runs, est. {fmt_duration(sum(known))}"
              + (f" (+ {len(ests) - len(known)} with no estimate yet)"
                 if len(known) < len(ests) else ""))


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def parse_overrides(items):
    template = Unit("air_quality", "lstm", 64, 1.0, 20, 14, "none", 0)
    out = {}
    for item in items:
        field, _, value = item.partition("=")
        if field not in OVERRIDABLE:
            raise ValueError(f"cannot override {field!r}; choose from {list(OVERRIDABLE)}")
        current = getattr(template, field)
        if value.lower() == "none":
            out[field] = None
        elif isinstance(current, float):
            out[field] = float(value)
        else:
            out[field] = int(float(value))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget", type=float, metavar="MIN", help="stop starting runs after MIN minutes")
    p.add_argument("--design", nargs="+", choices=design.BLOCKS, default=list(design.BLOCKS))
    p.add_argument("--seeds", type=int, help="seeds per configuration (default 5; 2 with --smoke)")
    p.add_argument("--dataset", nargs="+", choices=DATASETS)
    p.add_argument("--arch", nargs="+", choices=list(ARCHS))
    p.add_argument("--method", nargs="+", choices=METHODS)
    p.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true", help="tiny fast run into results/smoke.jsonl")
    p.add_argument("--run", action="store_true", help="run (the default unless a report flag is given)")
    p.add_argument("--status", action="store_true", help="show progress and estimated time left")
    p.add_argument("--tables", action="store_true", help="write summary CSVs to results/tables/")
    p.add_argument("--plot", action="store_true", help="draw the figures")
    args = p.parse_args()

    try:
        overrides = {**(SMOKE if args.smoke else {}), **parse_overrides(args.set)}
    except ValueError as err:
        p.error(str(err))
    seeds = args.seeds or (2 if args.smoke else 5)
    only = {"dataset": args.dataset, "arch": args.arch, "method": args.method}
    units = design.all_units(args.design, seeds, overrides, only)
    path = RESULTS_DIR / ("smoke.jsonl" if args.smoke else "runs.jsonl")

    reports = args.status or args.tables or args.plot
    if args.run or not reports:
        run(units, path, resolve_device(args.device), args.budget)
    if args.status:
        block_keys = {b: {design.key(u) for u in design.all_units([b], seeds, overrides, only)}
                      for b in args.design}
        status(units, block_keys, path)
    if args.tables or args.plot:
        import aggregate
        data = aggregate.load(path)
        if args.tables:
            aggregate.write_tables(data, RESULTS_DIR / "tables")
        if args.plot:
            import figures
            FIGURES_DIR.mkdir(exist_ok=True)
            figures.make_all(data, FIGURES_DIR)


if __name__ == "__main__":
    main()
