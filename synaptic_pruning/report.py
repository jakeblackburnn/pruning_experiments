"""Result briefs: the results JSON rendered as the markdown blocks that
RESULTS.md and the root README embed, plus running the notebook in place.

A block lives between `<!-- report:<name> -->` and `<!-- /report -->` in a
markdown file. `write_blocks` rewrites only those spans, so the hand-written
readings around them are never touched, and it refuses a marker it has no
block for, so a template and this module can't silently drift apart.

Like figures.py, a pure function of the results dicts (no torch). The
writing half mirrors ../obd/report.py; each project stays self-contained.
"""

import datetime
import re
import subprocess
from pathlib import Path

import numpy as np

# Only this project's blocks: the root README carries both projects' markers.
PREFIX = "sp_"
MARKER = re.compile(r"(<!-- report:(sp_\w+) -->\n)(.*?)(<!-- /report -->)", re.S)
SIGNIFICANT = 0.05


def git_commit(root):
    """Short HEAD hash, suffixed -dirty if tracked files have uncommitted
    changes (untracked results/ being written by the run don't count)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True).stdout.strip()
    head = git("rev-parse", "--short", "HEAD")
    if not head:
        return None
    return head + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


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
    for stage, res in results.items():
        meta = res.get("_meta", {})
        bits = [f"commit {meta.get('git_commit') or '?'}"]
        if meta.get("duration_s"):
            bits.append(minutes(meta["duration_s"]))
        parts.append(f"`results/{stage}.json` ({meta.get('written', '?')[:10]}, "
                     f"{', '.join(bits)})")
    return f"\n_Generated from {'; '.join(parts) or 'no results yet'}._\n"


def _not_run(stage):
    return f"_not yet run — `python3 main.py --{stage.replace('_', '-')}`_"


def _ladder_cells(results):
    """The ladder's cells in tier order, or None if absent or from the old
    3x3x3 grid (which has no tiers)."""
    bl = results.get("bitter_lesson")
    if not bl or not bl["cells"] or "tier_index" not in bl["cells"][0]:
        return None
    return sorted(bl["cells"], key=lambda c: c["tier_index"])


def _replication_cells(results):
    rep = results.get("replication")
    if not rep:
        return None
    return sorted(rep["summary"].values(), key=lambda c: (c["model_type"], c["seq_len"]))


def _winrate(cells):
    """Counts of how pruning fared across the replication cells."""
    n = len(cells)
    best = sum(min(c["methods"], key=lambda m: c["methods"][m]["mean"]) == "pruning"
               for c in cells)
    beats = {other: sum(c["methods"]["pruning"]["mean"] < c["methods"][other]["mean"]
                        for c in cells)
             for other in cells[0]["methods"] if other != "pruning"}
    significant = [c for c in cells
                   if c["friedman"] and c["friedman"]["p_value"] < SIGNIFICANT]
    return n, best, beats, significant


def _trend(cells):
    x = [c["tier_index"] for c in cells]
    y = [c["improvement_pct"] for c in cells]
    return float(np.polyfit(x, y, 1)[0]) if len(cells) > 1 else float("nan")


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------

def block_headline(results):
    rows = []
    cells = _replication_cells(results)
    if cells:
        n, best, beats, significant = _winrate(cells)
        rows.append(("replication", f"pruning has the lowest mean MAE in {best}/{n} cells; "
                     + ", ".join(f"beats {m} in {k}/{n}" for m, k in beats.items())
                     + f"; {len(significant)}/{n} cells significant at p<{SIGNIFICANT}"))
    else:
        rows.append(("replication", _not_run("replication")))
    ladder = _ladder_cells(results)
    if ladder:
        first, last = ladder[0], ladder[-1]
        rows.append(("bitter-lesson ladder",
                     f"pruning vs baseline {first['improvement_pct']:+.1f}% at the smallest tier "
                     f"(h={first['hidden_size']}), {last['improvement_pct']:+.1f}% at the largest "
                     f"(h={last['hidden_size']}); trend {_trend(ladder):+.2f} pts/tier"))
    else:
        rows.append(("bitter-lesson ladder", _not_run("bitter_lesson")))
    return _table(("experiment", "result"), rows)


def block_replication_table(results):
    cells = _replication_cells(results)
    if not cells:
        return _not_run("replication")
    methods = list(cells[0]["methods"])
    rows = []
    for c in cells:
        means = {m: c["methods"][m]["mean"] for m in methods}
        best = min(means, key=means.get)
        fr = c["friedman"]
        p = (f"**{fr['p_value']:.3f}**" if fr["p_value"] < SIGNIFICANT
             else f"{fr['p_value']:.3f}") if fr else "—"
        rows.append([c["model_type"].upper(), c["seq_len"]] +
                    [f"**{means[m]:.4f}**" if m == best else f"{means[m]:.4f}"
                     for m in methods] + [p])
    n = cells[0]["methods"][methods[0]]["n"]
    return (_table(["model", "seq_len"] + methods + ["Friedman p"], rows) +
            f"\n\nMean test MAE over {n} trials (standardized units, lower is better); "
            f"bold = lowest mean in the row, and p < {SIGNIFICANT}.")


def block_replication_winrate(results):
    cells = _replication_cells(results)
    if not cells:
        return _not_run("replication")
    n, best, beats, significant = _winrate(cells)
    rows = [("pruning has the lowest mean MAE", f"{best}/{n}")]
    rows += [(f"pruning beats {m}", f"{k}/{n}") for m, k in beats.items()]
    rows.append((f"cells with Friedman p < {SIGNIFICANT}", f"{len(significant)}/{n}"))
    for c in significant:
        winner = min(c["methods"], key=lambda m: c["methods"][m]["mean"])
        rows.append((f"— {c['model_type'].upper()} seq_len={c['seq_len']}",
                     f"p={c['friedman']['p_value']:.3f}, lowest: {winner}"))
    for model in sorted({c["model_type"] for c in cells}):
        mine = [c for c in cells if c["model_type"] == model]
        avg = {m: np.mean([c["methods"][m]["mean"] for c in mine]) for m in mine[0]["methods"]}
        rows.append((f"{model.upper()} mean MAE across seq_lens",
                     ", ".join(f"{m} {v:.4f}" for m, v in avg.items())))
    return _table(("measure", "value"), rows)


def block_ladder_table(results):
    cells = _ladder_cells(results)
    if not cells:
        return (_not_run("bitter_lesson") if "bitter_lesson" not in results else
                "_results/bitter_lesson.json is from the old 3×3×3 grid — rerun "
                "`python3 main.py --bitter-lesson`_")
    rows = []
    for c in cells:
        none, prune = c["none"], c["pruning"]
        overlap = none["ci95"][0] <= prune["ci95"][1] and prune["ci95"][0] <= none["ci95"][1]
        rows.append((c["tier_index"], c["hidden_size"], c["epochs"], f"{c['n_rows']:,}",
                     f"{none['mean']:.4f} ± {(none['ci95'][1] - none['mean']):.4f}",
                     f"{prune['mean']:.4f} ± {(prune['ci95'][1] - prune['mean']):.4f}",
                     f"{c['improvement_pct']:+.1f}%", "yes" if overlap else "**no**"))
    n = cells[0]["none"]["n"]
    return (_table(("tier", "hidden", "epochs", "train rows", "none MAE", "pruning MAE",
                    "improvement", "95% CIs overlap"), rows) +
            f"\n\nMean ± 95% CI half-width over {n} trials. Improvement = "
            f"(none − pruning) / none; positive means pruning wins.")


def block_ladder_trend(results):
    cells = _ladder_cells(results)
    if not cells:
        return _not_run("bitter_lesson")
    wins = sum(c["improvement_pct"] > 0 for c in cells)
    rows = [("linear trend of improvement across tiers", f"{_trend(cells):+.2f} pts/tier"),
            ("tiers where pruning wins", f"{wins}/{len(cells)}"),
            ("mean improvement", f"{np.mean([c['improvement_pct'] for c in cells]):+.1f}%"),
            ("smallest → largest tier",
             f"{cells[0]['improvement_pct']:+.1f}% → {cells[-1]['improvement_pct']:+.1f}%")]
    return _table(("measure", "value"), rows)


BLOCKS = {
    "sp_headline": block_headline,
    "sp_replication_table": block_replication_table,
    "sp_replication_winrate": block_replication_winrate,
    "sp_ladder_table": block_ladder_table,
    "sp_ladder_trend": block_ladder_trend,
}


def render(results):
    """{block name: markdown} for every block, from {stage: results}."""
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
