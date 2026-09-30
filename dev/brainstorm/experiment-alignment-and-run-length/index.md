# Experiment alignment and run length · brainstorm 2026-09-30

Topic: do the OBD and synaptic designs answer the three questions in `dev/main.md`, and how can
the whole suite fit the 8 h budget (from `dev/agents/project/notes.md`) and still give numbers
worth reporting. Ideas and options only; nothing here is decided.

## Findings

1. **OBD time is retraining, not the Hessian.** The iterative prune-retrain arm is 70–80% of every
   unit; the saliency arms cost within 10% of magnitude. Cutting Hessian samples saves nothing.
2. **The OBD signal is concentrated in a narrow band.** With retraining, saliency − magnitude is
   ≈ 0 down to keep 8% and opens only from keep 5% down (up to +25 pts at 0.5%). Without
   retraining, saliency mostly *loses* to magnitude in the mid range, including on the paper net,
   which is the opposite of the 1989 result (seed 0 only).
3. **Blocks map onto questions:** OBD `core` is the Q2 (overlap) block, OBD `scale` the Q1 block,
   synaptic `scale` the Q3 block. The biggest misfit is `core` running the full retrain arm on
   every weight-decay cell, which Q2 does not use.
4. **`--status` overestimates OBD** (≈ 17 h per seed against ≈ 7.3 h from a per-step fit on CUDA
   units): its cost model mixes in the Mac units and scales the retrain arm with training epochs.
5. **The seed count caps inference.** With 5 seeds the Wilcoxon column can never reach p < 0.05
   (minimum 0.0625); with 2 seeds the t-interval is ±12.7 SE. Headline claims need ≥ 3 seeds and
   should be pooled over cells within a seed.
6. **"At scale" is small in both projects** (≈ 16× parameters). It tests modern architectures more
   than modern scale; a wording choice for you, or one large centre point.

## Recommended shape (≈ 7 h, with an 8 h ceiling)

| Part | Per seed | Seeds | Total |
|---|---|---|---|
| OBD trimmed (star scale; no retrain arm on non-default weight-decay cells; 3 retrain criteria, 7 levels; 8 combos) | 1.5 h | 3 | 4.6 h |
| OBD centre points + paper net, extra seeds | 0.12 h | +2 | 0.25 h |
| Synaptic trimmed (seq 2 levels; star scale + width × method cross + random_pruning control; 6 sweep variants) | 0.54 h | 4 | 2.2 h |

Estimates are ±30%. Seed-major order means an overrun still leaves complete seeds. If a
10-minute test shows two processes on the GPU raise throughput, running synaptic next to OBD
buys the margin back and synaptic can go to 5 seeds.

Order: first the centre points with 5 seeds (≈ 0.6 h OBD): they measure the seed noise, which
tells whether the third OBD seed is worth its 1.5 h, and they check the paper-net result.

## Files

- [question-map.md](question-map.md): each research question → its estimand → the blocks that
  serve it; gaps and what serves no question.
- [obd-time-profile.md](obd-time-profile.md): measured step times per combo; where the saliency vs
  magnitude difference lives.
- [obd-trim-options.md](obd-trim-options.md): six levers priced cumulatively (7.3 → 1.5 h per
  seed), unpriced levers, what to do with the 99 finished units.
- [synaptic-trim-options.md](synaptic-trim-options.md): trimmed design priced (1.12 → 0.54 h per
  seed), options not taken.
- [seeds-and-inference.md](seeds-and-inference.md): what 2, 3 and 5 seeds can support; pooling;
  measuring the noise first.

## Open for you

- Q1/Q3 wording: "modern architectures" vs "modern scale" (or one large point).
- OBD: start a fresh log and keep the 99 units as a pilot? (Trimming the retrain arm changes what
  a record contains.)
- Whether the Wilcoxon column stays in the tables.
