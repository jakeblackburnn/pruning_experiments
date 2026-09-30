# Synaptic pruning: trim options, priced

Priced with the project's own `CostModel` on the 84 seed-0 CUDA runs (`results/runs.jsonl`):
**1.12 h per seed now** (core 0.39, scale 0.69, sweep 0.08; 24 core runs have no estimate yet), so
5 seeds ≈ 5.6 h. The grid is 5× cheaper than OBD per seed, so the trim here can be mild and the
seeds kept.

## A trimmed design (≈ 0.54 h per seed, 474 runs, 16 unestimated)

| Block | Now | Trimmed | Per seed | What changes |
|---|---|---|---|---|
| core | 360 | 240 | 0.26 h | seq_len 3 → 2 levels (14, 60) |
| scale | 729 | 216 | 0.28 h | star per method (width 16/64/256, data 0.1/0.3, epochs ×0.5/×3) instead of the 27-cell cross; + `random_pruning` along width |
| sweep | 96 | 60 | 0.05 h | pruning variants 12 → 6: the paper setting (0.3–0.7, h0), smin 0, smax 0.5 and 0.9, horizon 10, and one corner (0–0.9, h10) |

5 seeds ≈ 2.7 h.

- **Width × method stays crossed** (3 widths × none/dropout/pruning), because Q3 is about that
  interaction. Data and epochs become one-at-a-time.
- **`random_pruning` along width** (27 runs per seed) is the one addition: it tells whether a
  benefit that grows with width comes from magnitude selection or just from removing capacity.
- The epochs ×3 cells (60 epochs) are the expensive ones; the star keeps one per method per arch.

## Other options

- **Vary seq_len only on lstm** (all archs at 14): core 360 → 168. Only matters if seq length is
  not a replication target.
- **Drop `mc_dropout` from core:** no question asks about it; −45 runs per seed.
- **Early stopping on validation MAE:** would shorten runs, but the pruning schedule is tied to
  the run length (horizon 0 = the whole run) and epochs is one of the scale axes. Stopping early
  would change what `pruning` means and confound the compute axis. Not recommended.
- **Two processes at once:** runs are 1–6 s on tiny models, so the GPU mostly waits on Python.
  Two or three processes on disjoint `--dataset` filters could double throughput with no design
  change. Unmeasured; same log-append caveat as OBD.

## Keys

Trimming a block only removes units, and every unit that remains keeps its key, so the 84
finished runs still count. The one exception is `random_pruning` in scale: new keys, new runs,
nothing lost.
