# Findings · 2026-10-01 (for the write-up)

Computed from `obd/results/units.jsonl` (358/358 units; 308 CUDA + 50 MPS; 6.2 h) and
`synaptic_pruning/results/runs.jsonl` (1896/1896 runs; 2.1 h) with the functions in each
`aggregate.py` (`load`, `center`, `paired`, `summarize`, `improvements`, `contrast`).
± is the half-width of a 95% t-interval over seeds. Differences are paired by config and seed.

## OBD

Setup: 8 dataset/arch combos, 9k–855k params. Centre point has 5 seeds, everything else 3.
`keep` = fraction of weights kept. "Retrain" = prune then fine-tune; "sweep" = prune only.
Base test accuracy at centre: mnist/paper 0.942, mnist/mlp 0.934, fmnist mlp/vgg/resnet
0.887/0.902/0.899, cifar10 mlp/vgg/resnet 0.510/0.756/0.735.

1. **With retraining, OBD saliency beats magnitude only at extreme sparsity.** Saliency minus
   magnitude test accuracy at centre:

   | model | keep 0.5% | 1% | 2% | 5% | 12% | 25% |
   |---|---|---|---|---|---|---|
   | cifar10/resnet | +.209 ±.070 | +.094 ±.033 | +.063 ±.013 | +.017 ±.008 | −.010 ±.006 | −.006 ±.009 |
   | cifar10/vgg | +.069 ±.017 | +.051 ±.012 | +.033 ±.026 | +.009 ±.009 | +.002 | −.001 |
   | cifar10/mlp | +.024 ±.007 | +.023 ±.005 | +.012 ±.009 | +.006 | +.007 | +.003 |
   | fmnist/resnet | +.126 ±.021 | +.017 ±.006 | +.009 ±.005 | +.001 | −.005 ±.004 | −.004 ±.003 |
   | fmnist/vgg | +.025 ±.010 | +.012 ±.006 | +.004 | +.001 | −.002 | −.002 |
   | fmnist/mlp | +.023 ±.011 | +.015 ±.004 | +.009 ±.003 | +.003 | −.003 | −.002 |
   | mnist/mlp | +.074 ±.008 | +.035 ±.008 | +.019 ±.009 | .000 | −.002 | −.003 |
   | mnist/paper | +.031 ±.145 | +.076 ±.052 | +.010 ±.057 | +.002 | −.005 | +.001 |

   At keep ≥ 12% the two are within ~0.5 pt, and magnitude is slightly ahead in several cells.
   `keep` alone explains 39% of the variance of the gap; every other factor ≤ 3%.
2. **The gap shrinks with width.** Gap averaged over keep < 1, width 0.5 → 1 → 2:
   cifar10/resnet .095 → .052 → .007; fmnist/resnet .096 → .020 → −.001; cifar10/vgg
   .028 → .024 → .009; mnist/mlp .026 → .017 → .005. Bigger nets: OBD's edge goes to zero.
3. **The gap shrinks with more retraining** (0.5× → 1× → 4× retrain epochs): cifar10/vgg
   .032 → .024 → .010, cifar10/mlp .016 → .011 → .002, fmnist/vgg .012 → .006 → .002.
   cifar10/resnet is the exception (.059 → .052 → .050).
4. **Data and epochs:** less data widens the gap on most models (fmnist/vgg .096 at 10% data
   vs .006 at full; cifar10/vgg .072 vs .024), cifar10/resnet goes the other way
   (.025 → .052). Training length has no consistent effect (eta² 0.001).
5. **Other criteria (retrain, centre, averaged over keep):** layer-normalised saliency is no
   better than magnitude (−.033 ±.024 on mnist/mlp, +.017 ±.008 on cifar10/resnet, |diff| < .01
   elsewhere). Taylor is ≤ magnitude on 5 of 8 models (−.070 on mnist/paper). Every criterion
   beats random by 7–25 pt.
6. **Without retraining the picture is different.** One-shot saliency is *worse* than magnitude
   on the BatchNorm ResNets (fmnist/resnet keep 25%: 0.218 vs 0.695; cifar10/resnet keep 35%:
   0.105 vs 0.399) and better on the MLPs at moderate sparsity (cifar10/mlp keep 12%: 0.394 vs
   0.279). Saliency recomputed between pruning steps is the best no-retrain criterion on every
   model (fmnist/resnet keep 18%: 0.769 vs 0.311 magnitude vs 0.116 one-shot saliency).
7. **Overlap of the rankings is high globally, low where it matters.** Spearman(saliency,
   magnitude) at centre is 0.82–0.93 (lowest cifar10/resnet 0.816, highest mnist/paper 0.926).
   But the kept sets agree far less at high sparsity: fraction of magnitude's kept weights that
   saliency also keeps is 0.12–0.52 at keep 0.5%, 0.32–0.66 at 5%, 0.68–0.82 at 25%,
   0.77–0.93 at 50%. That is the same regime where the accuracy gap appears.
8. **Weight decay drives the overlap** (eta² 0.27, then width 0.20, arch 0.12). No weight decay
   lowers Spearman on every model (cifar10/resnet 0.82 → 0.65, mnist/mlp 0.91 → 0.61). Heavy
   decay (1e-2) collapses it on the fmnist/cifar conv nets (fmnist/vgg 0.34, fmnist/resnet
   0.50, cifar10/vgg 0.60) and raises it on MNIST (0.92–0.95).

## Synaptic pruning

Setup: 3 datasets (air_quality, beijing_pm25, etth1) × 5 archs (rnn, gru, lstm, cnn,
transformer), 4 seeds. Outcome: % improvement in test MAE over a reference, positive = better.

1. **No regulariser clearly beats no regularisation.** Core block, vs `none`: pruning
   −1.1 ±1.3, dropout −1.9 ±2.3, mc_dropout −4.6 ±2.8, l2 +0.8 ±1.6, oneshot +1.4 ±1.5,
   random_pruning −4.6 ±3.6, narrow −4.1 ±3.0. Friedman per cell: 0 of 84 cells significant
   after Holm (41 of 84 raw p < .05).
2. **The CNN forecaster dominates those averages and is broken as a baseline:** it is worse
   than last-value persistence on air_quality (skill −0.45) and etth1 (−1.57). Pruning on CNN is
   −9.7 ±4.2 vs none; on the others +0.6 (gru, lstm), +1.2 (rnn), +2.0 (transformer).
3. **Without CNN:** pruning +1.1 ±1.3 vs none, oneshot +1.9 ±0.8, l2 +0.6 ±0.7, dropout
   −0.9 ±1.3, mc_dropout −3.2 ±0.9, random_pruning −0.3 ±1.7, narrow −0.6 ±1.7.
4. **Pruning vs dropout (the paper's claim):** +0.2 ±1.4 over all archs; +1.8 ±0.9 without
   CNN, all 4 seeds positive. Largest on rnn (+4.5 ±0.8). Negative on air_quality (−2.3 ±4.6,
   driven by CNN −18.6).
5. **Controls:** pruning beats random pruning (+2.0 ±1.1, 4/4 seeds) and a narrow dense net of
   equal parameter count (+2.1 ±2.1), so which weights go matters. But one-shot pruning to the
   same 70% beats the gradual schedule (−2.7 ±2.1 overall; −0.9 ±0.6 without CNN, 0/4 seeds).
   The schedule, the paper's contribution, does not help here.
6. **Scale:** pruning vs none improves with width, 16 → 64 → 256: −3.0 → −0.6 → +0.5 (±5.6,
   4.7, 1.8); random pruning catches up at 256 (+1.6 ±0.6), so at large width any 70% sparsity
   is harmless. Pruning vs dropout: +1.0 ±1.9, +1.9 ±3.0, +2.4 ±1.6 (does not fade with
   width). Along training length it shrinks: +2.8 ±3.4, +1.9 ±3.0, +1.0 ±1.1 at 0.5×/1×/3×
   epochs. Along data it is not resolved (+0.3 ±4.0 at 10%, +1.2 ±4.1 at 30%).
7. **Schedule sweep (lstm, cnn):** starting from 0 sparsity (`0-0.7`) is the only variant that
   wins on all seeds (+1.2 ±1.6); the paper's `0.3-0.7` is −2.0 ±5.9. On lstm every variant
   is +1.4 to +2.5; on cnn all but `0-0.7` are −4 to −9.
8. Arch explains most of the spread in pruning's edge (eta² 0.14; width 0.02, data 0.01).

## Caveats

- OBD "scale" tops out at 855k params and width 2×; "modern scale" is not tested.
- 3–5 seeds: many intervals include zero; treat single-cell numbers as indicative.
- MNIST MLP base accuracy is only 0.934 test (0.998 train), an overfit baseline.
- `runs.jsonl` holds 48 seed-0 runs that are not in the current design (seq_len 1, stray
  scale cells, one `pruning:0-0.5:h10`). `--status` ignores them but `aggregate.load` reads
  them, so `--tables` includes them (e.g. a `seq_len=1` level, an n=1 sweep row). The
  numbers above marked "without CNN" exclude them; the others include them.
