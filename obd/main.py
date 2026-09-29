"""Optimal Brain Damage at modern scales: a budgetable, resumable factorial run.

    python3 main.py --budget 60      run for about an hour, then stop
    python3 main.py --budget 600     ...run again to continue: finished units are skipped
    python3 main.py                  run everything that is left
    python3 main.py --status         progress per block and seed, time left
    python3 main.py --tables         summary CSVs  -> results/tables/
    python3 main.py --plot           figures       -> figures/

The design (design.py) is a list of units: one trained network plus every
pruning experiment on it. Each finished unit is appended as one JSON line to
results/units.jsonl (the resume state: a unit is done once its line exists).
Units run seed-major, so any stop leaves complete seeds and a balanced design;
a unit expected to overrun the remaining budget is not started. Ctrl-C loses
at most the unit in flight.

Design selection:

    --design core scale retrain   blocks to run (default: all; see design.py)
    --seeds N                     seeds per unit (default 5)
    --dataset mnist cifar10 ...   restrict the design
    --arch paper mlp vgg resnet   restrict the design
    --set FIELD=VALUE ...         override Unit fields (hessian_samples, epochs, ...)
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
from datasets import DATASETS, load_splits
from experiments import (RETRAIN_CRITERIA, SWEEP_CRITERIA, _json_safe, _point,
                         iterative_prune_retrain, overlap_analysis, sweep_no_retrain)
from obd import ARCHS, Unit, build, count_free_parameters, initial_masks, resolve_device, train

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
OVERRIDABLE = ("hessian_samples", "n_cap", "lr", "batch_size", "epochs",
               "retrain_epochs", "weight_decay")
SMOKE = {"hessian_samples": 64, "n_cap": 256, "epochs": 1, "retrain_epochs": 1}


# --------------------------------------------------------------------------
# One unit
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


def _columns(points):
    """A curve as {field: [values by level]}: the field names are stored once."""
    keys = list(dict.fromkeys(k for p in points for k in p))
    return {k: [p.get(k) for p in points] for k in keys}


def run_unit(unit, device):
    """Train the unit's base network and run every pruning experiment on it.
    Returns the record that is appended to the log."""
    info = DATASETS[unit.dataset]
    times = {}
    t0 = time.perf_counter()
    torch.manual_seed(unit.seed)
    splits = load_splits(unit, device)
    model = build(unit, info).to(device)
    x_tr, y_tr = splits["train"][0], splits["train"][1]
    train(model, x_tr, y_tr, unit)
    n_total = int(sum(m.sum() for m in initial_masks(model).values()).item())
    base = _point(model, splits, unit, n_total, n_total)
    times["train"] = time.perf_counter() - t0

    def timed(label, seed_offset, fn, *args, **kwargs):
        torch.manual_seed(unit.seed * 1000 + seed_offset)
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        times[label] = round(time.perf_counter() - start, 1)
        return out

    sweeps = {}
    for i, (criterion, recompute) in enumerate(SWEEP_CRITERIA):
        label = criterion + ("_recomputed" if recompute else "")
        sweeps[label] = _columns(timed(f"sweep_{label}", i, sweep_no_retrain, model,
                                       splits, criterion, unit, recompute=recompute))
    retrain = {}
    for i, criterion in enumerate(RETRAIN_CRITERIA):
        retrain[criterion] = _columns(timed(f"retrain_{criterion}", 100 + i,
                                            iterative_prune_retrain, model, splits, unit,
                                            criterion))
    overlap = timed("overlap", 200, overlap_analysis, model, splits, unit)

    return _json_safe({
        "key": design.key(unit), "unit": dataclasses.asdict(unit),
        "n_prunable": n_total, "n_params": count_free_parameters(model),
        "base": base, "sweeps": sweeps, "retrain": retrain, "overlap": overlap,
        "_meta": {"git_commit": git_commit(), "device": device,
                  "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                  "torch": torch.__version__, "duration_s": round(time.perf_counter() - t0, 1),
                  "step_durations_s": times,
                  "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}})


# --------------------------------------------------------------------------
# The log: results/*.jsonl, one finished unit per line
# --------------------------------------------------------------------------

def load_log(path):
    """Finished units, in file order. A truncated last line (a run killed
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
    print(f"{len(units) - len(pending)}/{len(units)} units done; {len(pending)} to go"
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
            label = (f"{pick.dataset}/{pick.arch} w{pick.width:g} data{pick.data_frac:g} "
                     f"ep{pick.epochs} wd{pick.weight_decay:g} rt{pick.retrain_epochs} "
                     f"seed{pick.seed}")
            est = cost.estimate(pick)
            print(f"[{ran + 1}] {label}" + (f"  (~{fmt_duration(est)})" if est else ""),
                  flush=True)
            record = run_unit(pick, device)
            append_record(path, record)
            cost.add(record)
            ran += 1
            print(f"    done in {fmt_duration(record['_meta']['duration_s'])}", flush=True)
        else:
            print("all units done")
    except KeyboardInterrupt:
        print("\ninterrupted; the unit in flight is lost, everything else is saved")
    print(f"ran {ran} unit(s) this session")


def status(units, block_keys, path):
    records = load_log(path)
    done = {r["key"] for r in records}
    cost = design.CostModel(records)
    print(f"{path.name}: {sum(design.key(u) in done for u in units)}/{len(units)} units done")
    print("\nblock      done / total  (blocks overlap at the centre point)")
    for block, keys in block_keys.items():
        print(f"  {block:<9}{len(keys & done):>5} / {len(keys)}")
    print("\nseed       done / total")
    for seed in sorted({u.seed for u in units}):
        keys = {design.key(u) for u in units if u.seed == seed}
        print(f"  {seed:<9}{len(keys & done):>5} / {len(keys)}")
    remaining = [u for u in units if design.key(u) not in done]
    ests = [cost.estimate(u) for u in remaining]
    known = [e for e in ests if e is not None]
    if remaining:
        print(f"\nremaining: {len(remaining)} units, est. {fmt_duration(sum(known))}"
              + (f" (+ {len(ests) - len(known)} with no estimate yet)"
                 if len(known) < len(ests) else ""))


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _coerce(value, current):
    if value.lower() == "none":
        return None
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(current, int) or current is None:
        try:
            return int(value)
        except ValueError:
            return int(float(value))
    if isinstance(current, float):
        return float(value)
    return value


def parse_overrides(items):
    template = Unit("mnist", "mlp", 1.0, 1.0, 1, 0.0, 1, 0)
    out = {}
    for item in items:
        field, _, value = item.partition("=")
        if field not in OVERRIDABLE:
            raise ValueError(f"cannot override {field!r}; choose from {list(OVERRIDABLE)}")
        out[field] = _coerce(value, getattr(template, field))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget", type=float, metavar="MIN", help="stop starting units after MIN minutes")
    p.add_argument("--design", nargs="+", choices=design.BLOCKS, default=list(design.BLOCKS))
    p.add_argument("--seeds", type=int, help="seeds per unit (default 5; 2 with --smoke)")
    p.add_argument("--dataset", nargs="+", choices=list(DATASETS))
    p.add_argument("--arch", nargs="+", choices=list(ARCHS))
    p.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true", help="tiny fast run into results/smoke.jsonl")
    p.add_argument("--run", action="store_true", help="run units (the default unless a report flag is given)")
    p.add_argument("--status", action="store_true", help="show progress and estimated time left")
    p.add_argument("--tables", action="store_true", help="write summary CSVs to results/tables/")
    p.add_argument("--plot", action="store_true", help="draw the figures")
    args = p.parse_args()

    try:
        overrides = {**(SMOKE if args.smoke else {}), **parse_overrides(args.set)}
    except ValueError as err:
        p.error(str(err))
    seeds = args.seeds or (2 if args.smoke else 5)
    only = {"dataset": args.dataset, "arch": args.arch}
    units = design.all_units(args.design, seeds, overrides, only)
    path = RESULTS_DIR / ("smoke.jsonl" if args.smoke else "units.jsonl")

    reports = args.status or args.tables or args.plot
    if args.run or not reports:
        run(units, path, resolve_device(args.device), args.budget)
    if args.status:
        block_keys = {b: {design.key(u) for u in design.all_units([b], seeds, overrides, only)}
                      for b in args.design}
        status(units, block_keys, path)
    if args.tables or args.plot:
        import aggregate
        units_df = aggregate.load(path)
        if args.tables:
            aggregate.write_tables(units_df, RESULTS_DIR / "tables")
        if args.plot:
            import figures
            FIGURES_DIR.mkdir(exist_ok=True)
            figures.make_all(units_df, FIGURES_DIR)


if __name__ == "__main__":
    main()
