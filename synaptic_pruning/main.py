"""Synaptic Pruning — Vos et al. 2025, reproduced, plus a bitter-lesson
scaling test. See README.md for the story.

Stages (each caches its output, so later stages can be rerun alone):

    python3 main.py --replicate      run the method-comparison grid -> results/replication.json
    python3 main.py --bitter-lesson  run the compute-scaling ladder -> results/bitter_lesson.json
    python3 main.py --plot           draw the figures                -> figures/*.png
    python3 main.py                  all of the above

Override any config field with --set:

    python3 main.py --replicate --set trials=3 epochs=10
    python3 main.py --bitter-lesson --set trials=5

The device is chosen automatically (Metal, else CUDA, else CPU) and can be
pinned with `--set device=cpu`.

Orchestration only: disk I/O, config parsing, the CLI. The pruning algorithm
lives in pruning.py, models in models.py, one training run in train.py, the
two experiment grids in experiments.py, plotting in figures.py, data in
datasets.py.
"""

import argparse
import dataclasses
import datetime
import json
from pathlib import Path

import torch

from experiments import (BitterLessonConfig, ReplicationConfig,
                         run_bitter_lesson, run_replication)
from figures import FIGURES

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

STAGES = {
    "replication": (ReplicationConfig, run_replication),
    "bitter_lesson": (BitterLessonConfig, run_bitter_lesson),
}


def results_path(stage):
    return RESULTS_DIR / f"{stage}.json"


def _meta(cfg):
    return {"config": dataclasses.asdict(cfg), "torch": torch.__version__,
            "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def _coerce(value: str, current):
    if value.lower() == "none":
        return None
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(current, tuple):
        return tuple(_coerce(v, current[0]) for v in value.split(","))
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def configure(stage, overrides=()):
    config_cls, _ = STAGES[stage]
    cfg = config_cls()
    fields = {}
    for item in overrides:
        field, _, value = item.partition("=")
        if not hasattr(cfg, field):
            raise ValueError(f"unknown field {field!r}; choose from "
                             f"{[f.name for f in dataclasses.fields(cfg)]}")
        fields[field] = _coerce(value, getattr(cfg, field))
    return dataclasses.replace(cfg, **fields)


def run_stage(stage, overrides=()):
    cfg = configure(stage, overrides)
    _, run_fn = STAGES[stage]
    print(f"[{stage}] config: {cfg}")
    out = run_fn(cfg)
    out["_meta"] = _meta(cfg)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = results_path(stage)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")
    return out


def load_results(stage):
    path = results_path(stage)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python3 main.py --{stage.replace('_', '-')}`")
    return json.loads(path.read_text())


def make_all_figures(save=True):
    results = {}
    for stage in STAGES:
        try:
            results[stage] = load_results(stage)
        except FileNotFoundError as e:
            print(f"skipping figures that need {stage}: {e}")
    FIGURES_DIR.mkdir(exist_ok=True)
    drawn = {}
    for name, fn in FIGURES.items():
        fig = fn(results)
        if fig is None:
            continue
        drawn[name] = fig
        if save:
            path = FIGURES_DIR / f"{name}.png"
            fig.savefig(path, bbox_inches="tight")
            print(f"saved {path}")
    return drawn


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicate", action="store_true", help="run the replication grid")
    parser.add_argument("--bitter-lesson", action="store_true", help="run the compute-scaling ladder")
    parser.add_argument("--plot", action="store_true", help="generate the figures")
    parser.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                        help="override config fields, e.g. --set trials=3 epochs=10")
    args = parser.parse_args()

    stages = []
    if args.replicate:
        stages.append("replication")
    if args.bitter_lesson:
        stages.append("bitter_lesson")
    run_all = not (args.replicate or args.bitter_lesson or args.plot)

    if run_all or "replication" in stages:
        run_stage("replication", args.set)
    if run_all or "bitter_lesson" in stages:
        run_stage("bitter_lesson", args.set)
    if run_all or args.plot:
        make_all_figures()


if __name__ == "__main__":
    main()
