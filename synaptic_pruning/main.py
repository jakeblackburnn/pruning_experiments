"""Synaptic Pruning — Vos et al. 2025, reproduced, plus a bitter-lesson
scaling test. Results: RESULTS.md; walkthrough: synaptic_pruning.ipynb.

    python3 main.py                  the whole suite in one go (~2 h on an M4):
                                     replicate -> bitter-lesson -> plot ->
                                     report -> notebook

Stages (each caches its output, so any one can be rerun alone):

    --replicate      method comparison, RNN/LSTM x seq_len  -> results/replication.json   (~25-30 min)
    --bitter-lesson  the 6-tier compute-scaling ladder      -> results/bitter_lesson.json (~90 min)
    --plot           draw the figures                       -> figures/*.png
    --report         rewrite the generated blocks in RESULTS.md and ../README.md
    --notebook       execute synaptic_pruning.ipynb in place

Override config fields (experiments.py: ReplicationConfig, BitterLessonConfig)
with --set. Each override goes to every selected stage whose config has that
field, and a field no selected stage has is rejected before anything runs:

    python3 main.py --replicate --set trials=3 epochs=10     faster, noisier
    python3 main.py --set trials=3                           both stages
    python3 main.py --replicate --set seq_lens=1,14 model_types=lstm

`tiers` (a tuple of tuples) can't be set this way; edit
BitterLessonConfig.tiers or build a config in a script.

The device is chosen automatically (Metal, else CUDA, else CPU) and can be
pinned with --set device=cpu. Each results file's _meta records the config,
git commit and wall-clock duration. A crashed run is resumed by rerunning the
stage that failed; nothing is skipped automatically.

Orchestration only: disk I/O, config parsing, the CLI. The pruning algorithm
lives in pruning.py, models in models.py, one training run in train.py, the
two experiments in experiments.py, plotting in figures.py, the result briefs
in report.py, data in datasets.py.
"""

import argparse
import dataclasses
import datetime
import json
import time
from pathlib import Path

import torch

from experiments import (BitterLessonConfig, ReplicationConfig,
                         run_bitter_lesson, run_replication)
import report
from figures import FIGURES

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
NOTEBOOK = ROOT / "synaptic_pruning.ipynb"
BRIEFS = (ROOT / "RESULTS.md", ROOT.parent / "README.md")

STAGES = {
    "replication": (ReplicationConfig, run_replication),
    "bitter_lesson": (BitterLessonConfig, run_bitter_lesson),
}


def results_path(stage):
    return RESULTS_DIR / f"{stage}.json"


def _meta(cfg, duration_s):
    return {"config": dataclasses.asdict(cfg), "git_commit": report.git_commit(ROOT),
            "duration_s": round(duration_s, 1), "torch": torch.__version__,
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


def split_overrides(stages, overrides=()):
    """{stage: [FIELD=VALUE, ...]}: each override goes to every stage whose
    config has that field. Raises if a field belongs to none of `stages`,
    so a typo fails before a long run starts rather than an hour into it."""
    per_stage = {stage: [] for stage in stages}
    for item in overrides:
        field = item.partition("=")[0]
        owners = [stage for stage in stages
                  if field in {f.name for f in dataclasses.fields(STAGES[stage][0])}]
        if not owners:
            known = sorted({f.name for stage in stages
                            for f in dataclasses.fields(STAGES[stage][0])})
            raise ValueError(f"unknown field {field!r} for {stages}; choose from {known}")
        for stage in owners:
            per_stage[stage].append(item)
    return per_stage


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


def run_stage(stage, cfg):
    _, run_fn = STAGES[stage]
    print(f"[{stage}] config: {cfg}")
    start = time.perf_counter()
    out = run_fn(cfg)
    out["_meta"] = _meta(cfg, time.perf_counter() - start)
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


def load_all_results():
    """Every stage's cached results that exist, noting the ones that don't."""
    results = {}
    for stage in STAGES:
        try:
            results[stage] = load_results(stage)
        except FileNotFoundError as e:
            print(f"  note: {e}")
    return results


def make_all_figures(save=True):
    results = load_all_results()
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


def make_report():
    """Rewrite the generated blocks of every brief from the cached results."""
    report.write_all(BRIEFS, report.render(load_all_results()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicate", action="store_true", help="run the replication grid")
    parser.add_argument("--bitter-lesson", action="store_true", help="run the compute-scaling ladder")
    parser.add_argument("--plot", action="store_true", help="generate the figures")
    parser.add_argument("--report", action="store_true",
                        help="rewrite the generated blocks in RESULTS.md and ../README.md")
    parser.add_argument("--notebook", action="store_true",
                        help="execute synaptic_pruning.ipynb in place")
    parser.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                        help="override config fields, e.g. --set trials=3 epochs=10")
    args = parser.parse_args()

    run_all = not (args.replicate or args.bitter_lesson or args.plot
                   or args.report or args.notebook)
    stages = [stage for stage, flag in (("replication", args.replicate),
                                        ("bitter_lesson", args.bitter_lesson))
              if flag or run_all]
    if args.set and not stages:
        parser.error("--set only applies to --replicate / --bitter-lesson")
    # every override is parsed and type-checked before the first stage starts
    try:
        configs = {stage: configure(stage, items)
                   for stage, items in split_overrides(stages, args.set).items()}
    except ValueError as err:
        parser.error(str(err))
    timings = []

    def timed(label, fn, *fn_args):
        start = time.perf_counter()
        fn(*fn_args)
        timings.append((label, time.perf_counter() - start))

    for stage, cfg in configs.items():
        timed(stage, run_stage, stage, cfg)
    if args.plot or run_all:
        timed("plot", make_all_figures)
    if args.report or run_all:
        timed("report", make_report)
    if args.notebook or run_all:
        timed("notebook", report.execute_notebook, NOTEBOOK)

    print("\ntimings:")
    for label, seconds in timings:
        print(f"  {label:<16} {report.minutes(seconds)}")


if __name__ == "__main__":
    main()
