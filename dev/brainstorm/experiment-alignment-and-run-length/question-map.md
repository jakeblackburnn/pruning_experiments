# Question map: which part of the design answers which research question

The three questions from `dev/main.md`, each with the number that would answer it (the
*estimand*: the quantity a table reports), the blocks that produce it, and what in the design
serves no question.

## Q1 · Does OBD work at modern scales?

- **Estimand:** paired difference in test accuracy, saliency − magnitude, same trained net, same
  seed. It is paired, so seed-to-seed noise in the base net cancels.
- **Where the signal is (seed 0, 99 units; see [obd-time-profile.md](obd-time-profile.md)):**
  - with iterative retraining the difference is ≈ 0 (within ±2 pts) down to keep 8%, and saliency
    pulls ahead from about keep 5%; at keep 0.5% it leads by +4 to +25 pts depending on the combo.
  - without retraining saliency *loses* to magnitude in the middle of the curve (ResNets at keep
    35–25%, MLPs and the paper net at 12–3%).
  - so "works" depends on the regime, and one scalar per unit should summarise the tail, e.g. the
    mean Δacc over keep ∈ {5, 2, 1, 0.5}% after retraining, or the smallest keep at which accuracy
    stays within 1 pt of the base net.
- **Blocks:** `scale` (width × data × epochs) is the Q1 block; `retrain` (retrain multiple) is a
  robustness check on Q1 ("does more retraining wash out the ranking?").
- **Gap:** "modern scales" here means widths 0.5–2× of small nets on CIFAR-10, about a 16×
  parameter range. That tests modern *architectures* (BatchNorm, residuals, cross-entropy) more
  than modern *scale*. Options: (a) reword the question to architectures; (b) add one large point
  (width 4, cifar10/resnet, centre only, 3 seeds), which costs roughly 4× a centre unit each;
  (c) leave it and say so in the write-up. Your call; (a) or (c) cost nothing.
- **Replication flag:** the paper net (`mnist/paper`) shows saliency 7–18 pts *worse* than
  magnitude at keep 12–3% without retraining, which is the opposite of the 1989 result. It is seed
  0 only and the paper-net unit is cheap (≈ 34 s), so 5 seeds there is the cheapest high-value
  check in the whole suite.

## Q2 · How much do magnitude and OBD pruning overlap?

- **Estimands (all in `overlap_analysis`):** Spearman(saliency, |w|); Jaccard of the kept sets at
  each keep level; `between_layer_frac` (share of the curvature variance that lies between layer
  means); and the `saliency_layermean` criterion, which tests "OBD = magnitude + a per-layer
  budget".
- **Seed 0:** Spearman 0.78–0.92; ResNets put 60–80% of curvature variance between layers (the
  curvature acts mostly as a per-layer offset), VGG/MLP under 20%; weight decay 0 gives the lowest
  agreement (0.69 against 0.80–0.86).
- **Blocks:** `core` (width × weight decay) is the Q2 block: weight decay moves the covariance term
  in the identity in `experiments.py`. Overlap costs < 2 s per unit, so it can run on every unit.
- **Misfit:** `core` also runs the full retrain arm on every width × weight-decay cell, which Q2
  does not need. That is most of core's cost.

## Q3 · Does synaptic pruning keep its benefit at scale?

- **Estimand:** paired % MAE improvement of `pruning` over `none` (and over `dropout`) along width,
  train fraction and epochs (`aggregate.py` already computes it).
- **Blocks:** `scale` (729 of 1152 runs per seed) is the Q3 block. `core` (5 archs × 3 seq lens ×
  8 methods) is replication plus controls; `sweep` is sensitivity to the method's own settings.
- **Gap 1:** the scale block has no control. If pruning's benefit grows with width, the question
  is whether it is magnitude selection or just losing capacity; `random_pruning` on the width axis
  only (3 widths × 3 archs × 3 datasets = 27 runs per seed) would answer that.
- **Gap 2:** "scale" is again small: widths 16–256, and data only goes *down* (the datasets are
  fixed and small). Same wording choice as Q1.
- **Unaligned:** `seq_len` (core) and `mc_dropout` answer no listed question; they are replication
  detail. Keep them only if replicating the paper's tables is itself a goal.

## What serves no question

| Piece | Cost | Serves | Option |
|---|---|---|---|
| retrain arm on every `core` cell | ≈ 70% of every core unit | nothing in Q2 | retrain only at the centre and along width |
| `taylor` in retrain | 1/5 of retrain | a modern baseline, not in a question | keep in sweeps (cheap), drop from retrain |
| `random` in retrain | 1/5 of retrain | the floor; sanity only | centre point only |
| retrain levels 70–8% | 7 of 12 retrain steps | Δ ≈ 0 there | fewer, larger steps above 8% |
| `mnist/resnet`, `mnist/vgg` | small (≈ 20 s units) | overlap with fmnist | drop or 1 seed; saves little time |
| synaptic `seq_len` × 8 methods × 5 archs | 360 runs per seed | replication | 2 seq lens, or seq on lstm only |
