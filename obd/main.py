"""Optimal Brain Damage — LeCun, Denker & Solla (NIPS 1989), reproduced,
plus a modern CIFAR-10 counterpart. Results: RESULTS.md; walkthrough: obd.ipynb.

    python3 main.py                  the whole suite, both datasets, in one go:
                                     train -> prune -> plot (per dataset), then
                                     report -> notebook

Stages (each caches its output, so any one can be rerun alone):

    --train      train the base network   -> checkpoints/<name>.pt
    --prune      run pruning experiments  -> results/<name>.json
    --plot       draw the figures         -> figures/*_<name>.png
    --report     rewrite the generated blocks in RESULTS.md and ../README.md
    --notebook   execute obd.ipynb in place

Datasets (default: both):

    --dataset digits    the paper: 1989 zip-code net, 16x16 digits, MSE
    --dataset cifar     VGG-style ReLU net (~590k params), CIFAR-10, cross-entropy

The prune stage's experiments can be rerun individually with --only (implies
--prune); they merge into the cached results file:

    magnitude                       delete smallest-|w| first, no retraining
    saliency, saliency_recomputed   same, ranked by OBD saliency, once / re-ranked each step
    saliency_layermean[_recomputed] ranked by 1/2 * mean_layer(h) * w^2, the control between the two
    retrain, retrain_magnitude      the prune-retrain loop, and its magnitude-ranked control
    overlap                         the ranking-agreement analysis behind Figs 5-7

Any Experiment field (obd.py) can be overridden, typed against its default:

    python3 main.py --dataset cifar --set epochs=5 lr=3e-4 hessian_samples=1024
    python3 main.py --dataset digits --only retrain_magnitude

The device is chosen automatically (Metal, else CUDA, else CPU). Pin it with
--set device=cpu, which is often *faster* for digits: at 8.8k parameters and
batch 32, kernel-launch overhead dominates. Each results file's _meta records
the config, checkpoint hash, git commit and per-experiment durations, and a
stale merge gets a warning rather than silence. A crashed run is resumed by
rerunning the stage that failed; nothing is skipped automatically.

Orchestration only: disk I/O, the experiment runner, and the CLI. The OBD math
lives in obd.py, the experiments in experiments.py, plotting in figures.py,
the result briefs in report.py.
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import time
from pathlib import Path

import torch

import report
from datasets import load
from experiments import PRUNE_EXPERIMENTS
from figures import FIGURES
from obd import EXPERIMENTS, SEED, build, evaluate, resolve_device, train

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
NOTEBOOK = ROOT / "obd.ipynb"
BRIEFS = (ROOT / "RESULTS.md", ROOT.parent / "README.md")


def checkpoint_path(exp):
    return CHECKPOINT_DIR / f"{exp.name}.pt"


def results_path(exp):
    return RESULTS_DIR / f"{exp.name}.json"


def load_splits(exp):
    """Both splits, as tensors already moved onto the experiment's device."""
    return [tuple(t.to(exp.device) for t in split) for split in load(exp)]


def train_base_model(exp):
    """Train the unpruned network and save it to checkpoints/<name>.pt."""
    torch.manual_seed(SEED)
    (x_tr, y_tr, l_tr), (x_te, y_te, l_te) = load_splits(exp)

    model = build(exp).to(exp.device)
    print(f"training base {exp.name} model on {exp.device}...")
    train(model, x_tr, y_tr, exp, verbose=True)

    loss_tr, acc_tr = evaluate(model, x_tr, y_tr, l_tr, exp.loss)
    loss_te, acc_te = evaluate(model, x_te, y_te, l_te, exp.loss)
    print(f"train: loss {loss_tr:.4f}, accuracy {acc_tr:.3f}")
    print(f"test:  loss {loss_te:.4f}, accuracy {acc_te:.3f}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path(exp))
    print(f"saved {checkpoint_path(exp)}")
    return model


def load_base_model(exp):
    model = build(exp)
    model.load_state_dict(torch.load(checkpoint_path(exp), map_location=exp.device))
    return model.to(exp.device)


def _meta(exp, durations):
    """Provenance for a results file: which config, checkpoint and commit,
    and how long each experiment took.

    Without this a partial rerun silently merges results from two different
    base models into one file, and nothing downstream can tell.
    """
    path = checkpoint_path(exp)
    digest = (hashlib.sha1(path.read_bytes()).hexdigest()[:12] if path.exists()
              else None)
    return {"config": dataclasses.asdict(exp), "checkpoint_sha1": digest,
            "git_commit": report.git_commit(ROOT), "durations_s": durations,
            "torch": torch.__version__,
            "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def check_meta(exp, results):
    """Warn when cached results were produced by a different configuration."""
    meta = results.get("_meta")
    if meta is None:
        print(f"  note: results/{exp.name}.json predates provenance tracking — "
              f"rerun the prune stage if in doubt")
        return
    current, cached = dataclasses.asdict(exp), meta["config"]
    changed = {k: (cached.get(k), v) for k, v in current.items()
               if k != "device" and cached.get(k) != v}
    if changed:
        print(f"  warning: results/{exp.name}.json was produced with a different "
              f"config: {changed}")
    if meta.get("checkpoint_sha1") and checkpoint_path(exp).exists():
        digest = hashlib.sha1(checkpoint_path(exp).read_bytes()).hexdigest()[:12]
        if digest != meta["checkpoint_sha1"]:
            print(f"  warning: checkpoints/{exp.name}.pt has changed since these "
                  f"results were written ({meta['checkpoint_sha1']} -> {digest})")


def run_experiments(exp, only=None):
    """Run pruning experiments on the saved base model.

    only: subset of PRUNE_EXPERIMENTS keys (default: all). Results merge
    into any existing results/<name>.json, so one experiment can be added
    to a cached run without redoing the others. Each experiment is seeded
    independently, so results don't depend on which subset ran.
    """
    keys = list(PRUNE_EXPERIMENTS) if only is None else list(only)
    unknown = [k for k in keys if k not in PRUNE_EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown experiment(s) {unknown}; "
                         f"choose from {list(PRUNE_EXPERIMENTS)}")

    splits = load_splits(exp)
    model = load_base_model(exp)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = results_path(exp)
    out = json.loads(path.read_text()) if path.exists() else {}
    if out:
        check_meta(exp, out)

    durations = dict(out.get("_meta", {}).get("durations_s") or {})
    for key in keys:
        print(f"[{exp.name}] {key}...")
        torch.manual_seed(SEED)
        start = time.perf_counter()
        out[key] = PRUNE_EXPERIMENTS[key](model, splits, exp)
        durations[key] = round(time.perf_counter() - start, 1)

    out["_meta"] = _meta(exp, durations)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")
    return out


def load_results(exp):
    path = results_path(exp)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python3 main.py --dataset {exp.name} --prune`")
    return json.loads(path.read_text())


def make_all_figures(exp, results=None, save=True):
    """Draw every figure whose data is present. Missing ones are skipped."""
    if results is None:
        results = load_results(exp)
    check_meta(exp, results)
    FIGURES_DIR.mkdir(exist_ok=True)
    drawn = {}
    for name, fn in FIGURES.items():
        fig = fn(results, exp)
        if fig is None:
            continue
        drawn[name] = fig
        if save:
            path = FIGURES_DIR / f"{name}_{exp.name}.png"
            fig.savefig(path, bbox_inches="tight")
            print(f"saved {path}")
    return drawn


def make_report(names):
    """Rewrite the generated blocks of every brief from the cached results.
    Datasets without results are listed as not yet run."""
    results = {}
    for name in names:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            results[name] = json.loads(path.read_text())
        else:
            print(f"  note: no {path} yet — its rows will be missing")
    report.write_all(BRIEFS, report.render(results))


# --------------------------------------------------------------------------
# Command line parsing
# --------------------------------------------------------------------------

def _coerce(value: str, current):
    """Parse a --set value against the field's current value."""
    if value.lower() == "none":
        return None
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(current, int) or current is None:  # None: int-or-None fields
        try:
            return int(value)
        except ValueError:
            return int(float(value))
    if isinstance(current, float):
        return float(value)
    return value


def configure(name, overrides=()):
    """The named preset with --set overrides applied and its device resolved."""
    exp = EXPERIMENTS[name]
    fields = {}
    for item in overrides:
        field, _, value = item.partition("=")
        if not hasattr(exp, field):
            raise ValueError(f"unknown field {field!r}; choose from "
                             f"{[f.name for f in dataclasses.fields(exp)]}")
        fields[field] = _coerce(value, getattr(exp, field))
    exp = dataclasses.replace(exp, **fields)
    return dataclasses.replace(exp, device=resolve_device(exp.device))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="train the base model")
    parser.add_argument("--prune", action="store_true", help="run the pruning experiments")
    parser.add_argument("--plot", action="store_true", help="generate the figures")
    parser.add_argument("--report", action="store_true",
                        help="rewrite the generated blocks in RESULTS.md and ../README.md")
    parser.add_argument("--notebook", action="store_true", help="execute obd.ipynb in place")
    parser.add_argument("--dataset", nargs="+", choices=list(EXPERIMENTS),
                        default=list(EXPERIMENTS), help="which experiments to run (default: all)")
    parser.add_argument("--only", nargs="+", choices=list(PRUNE_EXPERIMENTS),
                        help="prune stage: run only these experiments and merge "
                             "them into the existing results/<name>.json")
    parser.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                        help="override Experiment fields, e.g. --set epochs=5 lr=3e-4")
    args = parser.parse_args()

    if args.only:
        args.prune = True  # --only alone means: run just those prune experiments
    run_all = not (args.train or args.prune or args.plot or args.report or args.notebook)
    try:  # every override is parsed and type-checked before the first stage starts
        exps = [configure(name, args.set) for name in args.dataset]
    except ValueError as err:
        parser.error(str(err))
    timings = []

    def timed(label, fn, *fn_args, **fn_kwargs):
        start = time.perf_counter()
        fn(*fn_args, **fn_kwargs)
        timings.append((label, time.perf_counter() - start))

    for exp in exps:
        print(f"[{exp.name}] device: {exp.device}")
        if args.train or run_all:
            timed(f"{exp.name} train", train_base_model, exp)
        if args.prune or run_all:
            timed(f"{exp.name} prune", run_experiments, exp, only=args.only)
        if args.plot or run_all:
            timed(f"{exp.name} plot", make_all_figures, exp)
    if args.report or run_all:
        timed("report", make_report, list(EXPERIMENTS))
    if args.notebook or run_all:
        timed("notebook", report.execute_notebook, NOTEBOOK)

    if timings:
        print("\ntimings:")
        for label, seconds in timings:
            print(f"  {label:<16} {report.minutes(seconds)}")


if __name__ == "__main__":
    main()
