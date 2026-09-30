# Seeds and inference: how many seeds give a meaningful number

Both `aggregate.py` files treat seeds as the only replicates (right: cells of one seed share a
data split and code path, so they are not independent). That makes the seed count a hard cap on
what any table can say:

| seeds | 95% t multiplier | smallest two-sided Wilcoxon p possible |
|---|---|---|
| 2 | 12.7 | 0.50 |
| 3 | 4.30 | 0.25 |
| 5 | 2.78 | 0.0625 |
| 6 | 2.57 | 0.031 |
| 8 | 2.36 | 0.0078 |

Two consequences:

1. **The Wilcoxon p-values in the tables can never be significant at 0.05 with the default 5
   seeds.** Even five out of five seeds agreeing gives p = 0.0625, and Holm makes it worse. The
   t-interval is the usable output; the Wilcoxon column is decoration unless a cell gets 6+ seeds.
2. **2 seeds gives no usable interval at all** (± 12.7 SE). The "seed 0 first, then decide seeds
   1–2" plan in `dev/agents/project/notes.md` works for spotting direction, not for reporting.
   Anything quoted with an interval needs ≥ 3 seeds; 5 is where the t multiplier stops hurting.

## Getting more from few seeds

- **Pool cells within a seed before testing.** `aggregate.py` already does this: a summary over a
  factor averages the paired Δ across the cells of each seed first. So a claim like "saliency
  beats magnitude at keep 1% on cifar10/resnet, across widths" has n = seeds but a much smaller SD
  than any one cell. Frame the headline claims at this pooled level, not per cell.
- **Trends as per-seed slopes.** For "does the effect hold as width grows", fit the slope of Δ
  against log width within each seed, then a t-interval over seeds. One number per seed, the same
  replicate logic, and the star design supplies exactly those points.
- **Combos as replicates for the general claim.** "Does OBD beat magnitude in general" can use
  the combos (8–10 of them) as replicates of a pooled Δ. That answers a different question (across
  architectures and datasets) with far more replicates than seeds.
- **A mixed model** (seed and cell as random effects) would pool the most, but it is more
  machinery and more assumptions than the rest of the analysis; not recommended here.

## Measuring the noise before spending seeds

Every finished unit is seed 0, so the seed-to-seed SD of the paired Δ is unknown. The cheapest
way to get it: run the **centre point of every combo with 5 seeds** first (≈ 0.12 h per seed with
the trimmed retrain arm, so ≈ 0.6 h). Then, per combo, seeds needed for a 95% half-width δ are
roughly n ≈ (2.8 · SD / δ)². If the tail Δ is ≈ 10 pts with an SD of 3 pts, 3 seeds are plenty;
if the SD is 10 pts, no affordable seed count settles a single cell and only pooled claims will.

For synaptic pruning the paper-scale effect is a few % of MAE, and each run is 1–6 s, so seeds are
cheap there: keep 5.

## Recommendation

- OBD: 3 seeds for the trimmed grid if the budget allows, 2 otherwise; 5 seeds on the centre
  points and the paper net (both cheap), which also measures the noise.
- Synaptic: 5 seeds on the trimmed grid.
- Report t-intervals; drop the Wilcoxon column or keep it only for cells with 6+ seeds.
