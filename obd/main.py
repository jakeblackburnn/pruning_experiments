"""Optimal Brain Damage — LeCun, Denker & Solla (NIPS 1989), reproduced,
plus a modern CIFAR-10 counterpart. See README.md for the story.

Stages (each caches its output, so later stages can be rerun alone):

    python3 main.py --train    train the base network -> checkpoints/<name>.pt
    python3 main.py --prune    run pruning experiments -> results/<name>.json
    python3 main.py --plot     draw the figures        -> figures/*.png
    python3 main.py            all of the above

Experiments (default: digits, the paper's setup):

    python3 main.py --dataset digits    1989 net, 16x16 digits, MSE
    python3 main.py --dataset cifar     VGG-style net, CIFAR-10, cross-entropy

The device is chosen automatically (Metal, else CUDA, else CPU) and can be
pinned with `--set device=cpu`. The prune stage can rerun a subset of its
experiments (merged into the existing results file), and any Experiment field
can be overridden:

    python3 main.py --only retrain_magnitude       just the magnitude control
    python3 main.py --dataset cifar --set epochs=5 lr=3e-4

Orchestration only: disk I/O, the experiment runner, and the CLI. The OBD math
lives in obd.py, the experiments in experiments.py, plotting in figures.py.
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
from pathlib import Path

import torch

from datasets import load
from experiments import PRUNE_EXPERIMENTS
from figures import FIGURES
from obd import EXPERIMENTS, SEED, build, evaluate, resolve_device, train

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


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


def _meta(exp):
    """Provenance for a results file: which config and which checkpoint.

    Without this a partial rerun silently merges results from two different
    base models into one file, and nothing downstream can tell.
    """
    path = checkpoint_path(exp)
    digest = (hashlib.sha1(path.read_bytes()).hexdigest()[:12] if path.exists()
              else None)
    return {"config": dataclasses.asdict(exp), "checkpoint_sha1": digest,
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

    for key in keys:
        print(f"[{exp.name}] {key}...")
        torch.manual_seed(SEED)
        out[key] = PRUNE_EXPERIMENTS[key](model, splits, exp)

    out["_meta"] = _meta(exp)
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
    parser.add_argument("--dataset", choices=sorted(EXPERIMENTS),
                        default="digits", help="which experiment to run")
    parser.add_argument("--only", nargs="+", choices=list(PRUNE_EXPERIMENTS),
                        help="prune stage: run only these experiments and merge "
                             "them into the existing results/<name>.json")
    parser.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                        help="override Experiment fields, e.g. --set epochs=5 lr=3e-4")
    args = parser.parse_args()

    exp = configure(args.dataset, args.set)
    print(f"[{exp.name}] device: {exp.device}")

    if args.only:
        args.prune = True  # --only alone means: run just those prune experiments
    run_all = not (args.train or args.prune or args.plot)
    if args.train or run_all:
        train_base_model(exp)
    if args.prune or run_all:
        run_experiments(exp, only=args.only)
    if args.plot or run_all:
        make_all_figures(exp)


if __name__ == "__main__":
    main()
