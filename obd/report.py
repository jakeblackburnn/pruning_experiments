"""Result briefs: the results JSON rendered as the markdown blocks that
RESULTS.md and the root README embed, plus running the notebook in place.

A block lives between `<!-- report:<name> -->` and `<!-- /report -->` in a
markdown file. `write_blocks` rewrites only those spans, so the hand-written
readings around them are never touched, and it refuses a marker it has no
block for, so a template and this module can't silently drift apart.

Like figures.py, a pure function of the results dicts (no torch).
"""

import datetime
import re
import statistics
import subprocess
from pathlib import Path

# Only this project's blocks: the root README carries both projects' markers.
PREFIX = "obd_"
MARKER = re.compile(r"(<!-- report:(obd_\w+) -->\n)(.*?)(<!-- /report -->)", re.S)

# fractions of the prunable weights at which the no-retrain sweeps are compared
SWEEP_LEVELS = (0.5, 0.1, 0.02)
RANKINGS = ("magnitude", "saliency", "saliency_recomputed",
            "saliency_layermean", "saliency_layermean_recomputed")
# fractions of the weights at which the prune-retrain loops are compared
RETRAIN_LEVELS = (0.5, 0.2, 0.1, 0.05, 0.0)  # 0.0 = the loop's last step


def git_commit(root):
    """Short HEAD hash, suffixed -dirty if the tree has uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True).stdout.strip()
    head = git("rev-parse", "--short", "HEAD")
    if not head:
        return None
    return head + ("-dirty" if git("status", "--porcelain") else "")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def minutes(seconds):
    return f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"


def _provenance(results):
    parts = []
    for name, res in results.items():
        meta = res.get("_meta")
        if not meta:
            parts.append(f"`results/{name}.json` (no provenance — predates tracking)")
            continue
        bits = [f"commit {meta.get('git_commit') or '?'}"]
        durations = meta.get("durations_s")
        if durations:
            bits.append(f"prune stage {minutes(sum(durations.values()))}")
        parts.append(f"`results/{name}.json` ({meta['written'][:10]}, {', '.join(bits)})")
    return f"\n_Generated from {'; '.join(parts)}._\n"


def _missing(name, keys):
    return (f"_`{name}`: not yet run — `python3 main.py --dataset {name} "
            f"--only {' '.join(keys)}`_")


def _at(series, target):
    return min(series, key=lambda p: abs(p["remaining"] - target))


def _base(res):
    for key in ("magnitude", "retrain"):
        if res.get(key):
            return res[key][0]
    return None


def _paired(res, frac):
    """OBD and magnitude prune-retrain points nearest `frac` of the weights."""
    n = res["retrain"][0]["remaining"]
    return _at(res["retrain"], frac * n), _at(res["retrain_magnitude"], frac * n)


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------

def block_headline(results):
    rows = []
    for name, res in results.items():
        base = _base(res)
        if base is None:
            continue
        cells = [name, f"{base['remaining']:,}", f"{base['test_acc']:.1%}"]
        if res.get("retrain") and res.get("retrain_magnitude"):
            obd, mag = _paired(res, 0.1)
            cells.append(f"{obd['test_acc']:.1%} vs {mag['test_acc']:.1%}")
            last_obd, last_mag = res["retrain"][-1], res["retrain_magnitude"][-1]
            cells.append(f"{last_obd['test_acc']:.1%} vs {last_mag['test_acc']:.1%} "
                         f"({last_obd['remaining']:,} left)")
        else:
            cells += ["—", "—"]
        ov = res.get("overlap", {}).get("global")
        cells.append(f"{ov['spearman']:.3f}" if ov else "—")
        rows.append(cells)
    return _table(("dataset", "prunable weights", "base test acc",
                   "test acc at ~10% weights, OBD vs magnitude (retrained)",
                   "same, at the last step", "Spearman(s, \\|w\\|)"), rows)


def block_base(results):
    rows = []
    for name, res in results.items():
        base = _base(res)
        cfg = res.get("_meta", {}).get("config", {})
        if base is None:
            continue
        rows.append((name, cfg.get("loss", "?"), f"{base['remaining']:,}",
                     cfg.get("epochs", "?"), f"{cfg.get('weight_decay', float('nan')):g}",
                     f"{base['train_loss']:.4f}", f"{base['test_loss']:.4f}",
                     f"{base['train_acc']:.1%}", f"{base['test_acc']:.1%}"))
    return _table(("dataset", "loss", "prunable weights", "epochs", "weight decay",
                   "train loss", "test loss", "train acc", "test acc"), rows)


def block_sweep(results):
    out = []
    for name, res in results.items():
        present = [k for k in RANKINGS if res.get(k)]
        if not present:
            out.append(_missing(name, RANKINGS))
            continue
        n = res[present[0]][0]["remaining"]
        rows = []
        for frac in SWEEP_LEVELS:
            pts = {k: _at(res[k], frac * n) for k in present}
            best = min(pts, key=lambda k: pts[k]["train_loss"])
            rows.append([f"{pts[present[0]]['remaining']:,} ({frac:.0%})"] +
                        [(f"**{pts[k]['train_loss']:.4f}**" if k == best
                          else f"{pts[k]['train_loss']:.4f}") for k in present])
        out.append(f"**{name}** — train loss after deleting down to each level, "
                   f"no retraining (bold = lowest):\n\n" +
                   _table(["weights left"] + present, rows))
    return "\n\n".join(out)


def block_forecast(results):
    rows = []
    for name, res in results.items():
        for key in ("saliency", "saliency_recomputed"):
            ratios = sorted(p["actual_increase"] / p["predicted_increase"]
                            for p in res.get(key, [])
                            if p.get("predicted_increase", 0) > 0)
            if ratios:
                rows.append((name, key, len(ratios), f"{statistics.median(ratios):.2f}×",
                             f"{ratios[0]:.2f}×–{ratios[-1]:.2f}×"))
    if not rows:
        return "_no saliency sweeps yet_"
    return _table(("dataset", "ranking", "points", "median actual / predicted",
                   "range"), rows)


def block_retrain(results):
    out = []
    for name, res in results.items():
        if not (res.get("retrain") and res.get("retrain_magnitude")):
            out.append(_missing(name, ["retrain", "retrain_magnitude"]))
            continue
        rows = []
        for frac in RETRAIN_LEVELS:
            obd, mag = _paired(res, frac)
            rows.append((f"{obd['remaining']:,} ({obd['remaining'] / res['retrain'][0]['remaining']:.0%})",
                         f"{obd['test_acc']:.1%}", f"{mag['test_acc']:.1%}",
                         f"{(obd['test_acc'] - mag['test_acc']) * 100:+.1f}"))
        out.append(f"**{name}** — test accuracy through the prune-retrain loop "
                   f"(base {res['retrain'][0]['test_acc']:.1%}):\n\n" +
                   _table(("weights left", "OBD saliency", "magnitude control",
                           "difference (pts)"), rows))
    return "\n\n".join(out)


def block_overlap(results):
    stats = [("spearman", "Spearman(s, \\|w\\|)", ".3f"),
             ("kendall_subsample", "Kendall τ (subsample)", ".3f"),
             ("pearson_log", "corr(log s, log\\|w\\|)", ".3f"),
             ("predicted_pearson", "— same, from the moments", ".3f"),
             ("var_log_h", "Var(log h)", ".3f"),
             ("var_log_w", "Var(log \\|w\\|)", ".3f"),
             ("cov_log_h_log_w", "Cov(log h, log \\|w\\|)", "+.3f"),
             ("between_layer_frac", "Var(log h) between layers", ".0%"),
             ("weight_decay", "weight decay", "g")]
    have = {name: res["overlap"] for name, res in results.items() if res.get("overlap")}
    if not have:
        return "_no overlap analysis yet_"
    rows = [[label] + [format(ov["global"][key], fmt) for ov in have.values()]
            for key, label, fmt in stats]
    tenth = []
    for ov in have.values():
        p = _at(ov["overlap_curve"], ov["n_prunable"] // 10)
        tenth.append(f"{p['overlap_frac']:.0%} (chance {p['chance']:.1%})")
    rows.append(["shared kept set at 10% remaining"] + tenth)
    table = _table(["statistic"] + list(have), rows)
    absent = [name for name in results if name not in have]
    return table + "".join(f"\n\n{_missing(name, ['overlap'])}" for name in absent)


BLOCKS = {
    "obd_headline": block_headline,
    "obd_base": block_base,
    "obd_sweep": block_sweep,
    "obd_forecast": block_forecast,
    "obd_retrain": block_retrain,
    "obd_overlap": block_overlap,
}


def render(results):
    """{block name: markdown} for every block, from {dataset: results}."""
    provenance = _provenance(results)
    return {name: fn(results).rstrip() + "\n" + provenance for name, fn in BLOCKS.items()}


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def write_blocks(path, blocks):
    """Rewrite every one of this project's report blocks in `path`; return
    the names written. A marker with no rendered block is an error, not a
    silent skip."""
    path = Path(path)
    text = path.read_text()
    unknown = [name for _, name, _, _ in MARKER.findall(text) if name not in blocks]
    if unknown:
        raise KeyError(f"{path}: no block renders {unknown}; choose from {list(blocks)}")
    written = []

    def replace(m):
        written.append(m.group(2))
        return m.group(1) + blocks[m.group(2)] + m.group(4)

    path.write_text(MARKER.sub(replace, text))
    return written


def write_all(paths, blocks):
    """Write blocks into every file, and fail if any block landed nowhere."""
    written = set()
    for path in paths:
        names = write_blocks(path, blocks)
        written.update(names)
        print(f"updated {path} ({', '.join(names) or 'no blocks'})")
    orphans = set(blocks) - written
    if orphans:
        raise KeyError(f"blocks with no marker in {[str(p) for p in paths]}: {sorted(orphans)}")


def execute_notebook(path, timeout=1800):
    """Run a notebook top to bottom in place, from its own directory.
    Raises on the first failing cell, so a broken notebook fails the run."""
    import nbformat
    from nbclient import NotebookClient

    path = Path(path)
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    print(f"executed {path} ({datetime.datetime.now():%H:%M})")
