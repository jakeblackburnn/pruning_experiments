# Report outline · 2026-10-01

Reference for the write-up. Each section gives the claim to make, the numbers that back it,
and what the writer must not overstate. Fuller tables are in `findings.md` next to this file.

Conventions: ± is the half-width of a 95% t-interval over seeds. Every comparison is paired
(same configuration, same seed). `keep` is the fraction of weights left after pruning.
Sources: `obd/results/units.jsonl` (358 units), `synaptic_pruning/results/runs.jsonl`
(1896 runs), analysed with each project's `aggregate.py`.

## 1. The question

Do Optimal Brain Damage (LeCun, Denker & Solla 1989) and synaptic pruning (Vos et al. 2025)
keep their benefit at scales larger than the papers showed?

Three sub-questions, asked of each technique:
1. Does the paper's headline hold at the paper's scale? (replication)
2. How does the technique do against control methods?
3. How do the technique and the controls move as model size, data and compute grow?

Headlines under test (writer: check the wording against the papers):
- OBD: a pruned and retrained network performs better than the unpruned one, and ranking
  weights by saliency `½·h·w²` is better than ranking by magnitude.
- Synaptic pruning: a gradual magnitude-pruning schedule during training is a better
  regulariser than dropout, shown on small forecasters trained for few epochs.

## 2. Bottom line (lead with this)

- Neither headline grows with scale. Pruning becomes close to free as networks get bigger,
  but it does not make them reliably better.
- OBD: the pruned net beats the unpruned net by more than 1 pt on 2 of 8 models. Magnitude
  pruning gives the same gain, so the gain is not OBD's. OBD beats magnitude only when under
  5% of weights are kept, and that edge shrinks toward zero as the net gets wider.
- Synaptic pruning: it beats dropout (+1.8 ±0.9% test error), mostly because dropout hurts.
  Against no regulariser it is +1.1 ±1.3%, not clearly above zero, and that does not grow
  with width, data or epochs. A plain one-shot prune does better than the paper's schedule.
- In both projects the cheap control catches up with the technique as width grows.

## 3. Method (short)

Both projects are factorial designs with seeds as the blocking factor. Scale is three
separate axes varied one at a time around a centre point: model size (width), data
(training fraction) and compute (epochs).

**OBD.** 8 dataset/architecture combos: MNIST (the paper-sized net, 8.8k params; MLP),
Fashion-MNIST and CIFAR-10 (MLP, VGG, ResNet). Width 0.5×/1×/2×, giving 50k to 1.8M params.
Data 10%/30%/100%. Epochs ⅓×/1×/3×. Retrain budget 0.5×/1×/4×. Weight decay 0/default/1e-2.
5 seeds at the centre, 3 elsewhere. Two measurements per unit: "sweep" (prune only) and
"retrain" (prune in steps, retraining after each). Criteria: saliency (OBD), magnitude,
layer-normalised saliency, saliency recomputed between steps (sweep only), and at the centre
only, Taylor and random.

**Synaptic pruning.** 3 datasets (air_quality, beijing_pm25, etth1) × 5 architectures (RNN,
GRU, LSTM, CNN, transformer), 4 seeds. Outcome: test MAE at the best-validation epoch,
reported as % improvement over a reference (positive = better). Methods: none, dropout 0.3,
MC dropout, L2, the paper's pruning schedule (30% → 70% sparsity), and three controls:
random pruning (same schedule, random weights), one-shot pruning to 70%, and a narrow dense
net with the same parameter count. Width 16/64/256 (up to 1M params), data 10%/30%/100%,
epochs 0.5×/1×/3×.

## 4. OBD results

### 4.1 Does a pruned network beat the unpruned one?

Test accuracy change against the unpruned net, OBD + retraining, centre point:

| model | keep 50% | keep 25% | keep 12% |
|---|---|---|---|
| mnist/paper (8.8k) | −.001 ±.005 | −.002 ±.003 | −.021 ±.005 |
| mnist/mlp | −.005 ±.005 | +.001 ±.008 | +.003 ±.008 |
| fmnist/mlp | +.003 ±.004 | +.003 ±.005 | −.001 ±.005 |
| fmnist/vgg | +.006 ±.003 | +.004 ±.002 | +.002 ±.002 |
| fmnist/resnet | +.004 ±.007 | +.005 ±.005 | .000 ±.007 |
| cifar10/vgg | +.002 ±.008 | +.003 ±.002 | −.005 ±.009 |
| cifar10/mlp | +.010 ±.005 | +.018 ±.003 | +.021 ±.006 |
| cifar10/resnet | +.037 ±.035 | +.047 ±.021 | +.035 ±.023 |

Claims:
- At the paper's scale: half to three-quarters of the weights can go at no cost, but the
  pruned net does not beat the unpruned one. "Free compression" replicates; "better" does not.
- At larger scale: a real gain on cifar10/mlp (~2 pt) and cifar10/resnet (~4 pt), under 1 pt
  elsewhere.
- Magnitude pruning gives the same gain (cifar10/resnet keep 25%: +.052 vs OBD +.047;
  cifar10/mlp: +.015 vs +.018). The benefit belongs to pruning-plus-retraining, not to OBD.

Do not overstate: the two models that gain are the most overfit (cifar10/mlp is 90% train,
51% test), and the unpruned baseline gets no extra training while the pruned net is retrained
at every step. The gain may be extra training rather than pruning (see §7).

### 4.2 Does that benefit grow with scale?

No. OBD, keep 25%, change against unpruned:

| axis | cifar10/resnet | cifar10/mlp | others |
|---|---|---|---|
| width 0.5× → 1× → 2× | +.013 → +.047 → +.023 (±.02–.03) | +.010 → +.018 → +.023 | within ±.01 |
| epochs ⅓× → 1× → 3× | +.028 → +.047 → +.011 (±.014 at 3×) | +.011 → +.018 → +.020 | within ±.015 |
| data 10% → 30% → 100% | −.041 → −.017 → +.047 | +.001 → +.021 → +.018 | within ±.015 |

Claims: only cifar10/mlp rises steadily with width. On cifar10/resnet the gain falls when
the base net is trained longer and turns negative with little data. No model shows the
benefit getting larger with scale in general.

### 4.3 OBD against the controls

Retrained, centre point, saliency minus magnitude test accuracy:

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

Claims:
- OBD beats magnitude only at extreme sparsity (under 5% kept). From 12% kept the two are
  within about 1 pt and magnitude is slightly ahead in several cells.
- Sparsity level explains 39% of the variance of the OBD-minus-magnitude gap; every other
  factor explains 3% or less.
- Other criteria, averaged over sparsity: both OBD and magnitude beat random by 7–25 pt.
  Layer-normalised saliency and Taylor are not reliably better than magnitude (Taylor is
  −.070 on mnist/paper).
- Without retraining the ranking changes: one-shot saliency is worse than magnitude on the
  BatchNorm ResNets (fmnist/resnet keep 25%: 0.218 vs 0.695) and better on MLPs at moderate
  sparsity (cifar10/mlp keep 12%: 0.394 vs 0.279). Recomputing the saliency between pruning
  steps is the best no-retrain criterion on every model (fmnist/resnet keep 18%: 0.769 vs
  0.311 magnitude vs 0.116 one-shot saliency).

### 4.4 How OBD and its control scale

Accuracy lost against unpruned at keep 1%, after retraining, width 0.5× → 1× → 2×:

| model (params at 2×) | magnitude | OBD |
|---|---|---|
| cifar10/resnet (1.2M) | −.505 → −.221 → −.070 | −.264 → −.127 → −.053 |
| cifar10/vgg (0.8M) | −.341 → −.199 → −.081 | −.303 → −.148 → −.056 |
| fmnist/resnet (1.2M) | −.252 → −.043 → −.003 | −.090 → −.026 → −.004 |
| fmnist/vgg (0.8M) | −.145 → −.042 → −.012 | −.092 → −.030 → −.009 |
| mnist/mlp (0.4M) | −.188 → −.087 → −.027 | −.133 → −.053 → −.009 |
| cifar10/mlp (1.8M) | −.053 → −.032 → −.010 | −.027 → −.008 → +.005 |

Claims:
- Both get better with width: wider nets tolerate extreme pruning.
- Magnitude improves faster, so OBD's edge closes: 24 pt → 9 pt → 2 pt on cifar10/resnet,
  16 pt → 2 pt → 0 on fmnist/resnet.
- At keep 5% and 2× width both criteria are within about 1 pt of unpruned on every model.
- Longer retraining also closes the gap (cifar10/vgg .032 → .024 → .010 at 0.5×/1×/4×);
  cifar10/resnet is the exception (.059 → .052 → .050).
- Less data widens the gap on most models (fmnist/vgg .096 at 10% data vs .006 at full).
- Random and Taylor ran at the centre only; their scaling is not measured.

### 4.5 Overlap between the two rankings (the secondary question)

- Rank correlation (Spearman) between saliency and magnitude is 0.82–0.93 at the centre.
- The kept sets agree much less where it matters. Share of magnitude's kept weights that
  saliency also keeps: 12–52% at keep 0.5%, 32–66% at 5%, 68–82% at 25%, 77–93% at 50%.
  This is the same regime where the accuracy gap appears.
- Weight decay moves the overlap most (27% of variance; width 20%, architecture 12%). No
  weight decay lowers it on every model (cifar10/resnet 0.82 → 0.65). Heavy decay (1e-2)
  collapses it on the conv nets (fmnist/vgg 0.34) and raises it on MNIST (0.92–0.95).

## 5. Synaptic pruning results

The CNN forecaster is worse than last-value persistence on two of three datasets (skill
−0.45 on air_quality, −1.57 on etth1), so its cells are noise that swamps averages. Report
both; the "no CNN" numbers are the ones to interpret.

### 5.1 Does it help at the paper's scale?

Base setting (width 64, full data, 20 epochs), % improvement in test MAE:

| pruning vs | all architectures | without CNN |
|---|---|---|
| no regulariser | −1.1 ±1.3 | +1.1 ±1.3 |
| dropout 0.3 | +0.2 ±1.4 | +1.8 ±0.9 (4/4 seeds) |

Other methods vs no regulariser, without CNN: one-shot +1.9 ±0.8, L2 +0.6 ±0.7, dropout
−0.9 ±1.3, MC dropout −3.2 ±0.9, random pruning −0.3 ±1.7, narrow −0.6 ±1.7.

Claims:
- The paper's claim holds in the narrow sense: pruning beats dropout. Most of that is
  dropout hurting these models rather than pruning helping.
- Against doing nothing the gain is about 1% and not clearly above zero. No method is
  significantly best in any of the 84 cells after Holm correction (Friedman test).
- Per cell it is small everywhere: every non-CNN dataset × architecture cell is between
  −0.4% and +3.5%. Largest on the transformer (+2.0 ±2.4) and RNN (+1.2 ±2.4).

### 5.2 Against the controls

Pruning minus control, % test MAE (all architectures | without CNN):

| control | all | without CNN | reading |
|---|---|---|---|
| random pruning | +2.0 ±1.1 (4/4 seeds) | about +1.3 (by arch +0.3 to +2.7) | which weights go matters |
| narrow dense net | +2.1 ±2.1 | about +1.6 (by arch +1.2 to +2.0) | not just fewer parameters |
| one-shot to 70% | −2.7 ±2.1 (0/4) | −0.9 ±0.6 (0/4) | the gradual schedule loses |

Claims:
- Magnitude selection is doing real work: it beats random selection and beats shrinking
  the net.
- The schedule, which is the paper's contribution, is not what helps. Pruning once to the
  same sparsity is better on every seed.
- Schedule sweep (LSTM, CNN): starting from zero sparsity (0 → 70%) is the only variant
  positive on all seeds (+1.2 ±1.6); the paper's 30% → 70% is −2.0 ±5.9.

("About" values are the mean of the four per-architecture means; they have no pooled
interval. Recompute before quoting.)

### 5.3 How it and the controls scale

% vs no regulariser, LSTM and transformer:

| | 16 | 64 | 256 units |
|---|---|---|---|
| synaptic pruning | +0.5 ±1.5 | +2.0 ±1.7 | +1.5 ±2.7 |
| dropout | −3.1 ±4.6 | +1.2 ±3.4 | −1.6 ±2.9 |
| random pruning | −2.1 ±1.5 | +0.1 ±3.1 | +2.1 ±0.8 |

| | 0.5× | 1× | 3× epochs |
|---|---|---|---|
| synaptic pruning | +0.3 ±4.3 | +2.0 ±1.7 | +1.0 ±1.0 |
| dropout | −3.8 ±3.6 | +1.2 ±3.4 | +1.8 ±0.8 |

| | 10% | 30% | 100% data |
|---|---|---|---|
| synaptic pruning | −0.3 ±0.9 | +0.5 ±3.7 | +2.0 ±1.7 |
| dropout | −1.4 ±6.0 | −3.6 ±4.7 | +1.2 ±3.4 |

Claims:
- No trend with width. On the LSTM the gain is 0.0 at width 256; the transformer rises
  (+0.9 → +2.3 → +3.0) but with intervals of ±3–5.
- The paper's regime (few epochs, little data) is where the benefit is smallest here.
- Random pruning catches up with width: pruning minus random is +2.5 ±1.6 → +1.8 ±2.3 →
  −0.6 ±2.0. At width 256 any 70% sparsity is harmless, so choosing by magnitude stops
  mattering.
- With 3× epochs dropout overtakes pruning (+1.8 vs +1.0), so pruning's edge over dropout
  is partly a short-training effect.
- One-shot and narrow ran at the base width only; their scaling is not measured.
- Architecture explains most of the spread in pruning's gain (14% of variance; width 2%,
  data 1%, epochs under 1%).

## 6. Verdict table

| headline | at paper scale | at larger scale |
|---|---|---|
| OBD: pruned beats unpruned | No. Free up to ~75% pruned, not better. | Gain on 2 of 8 models; magnitude matches it; does not grow with width, epochs or data. |
| OBD: saliency beats magnitude | Only under 5% kept. | Edge shrinks toward zero with width and with retraining. |
| Synaptic: beats dropout | Yes, +1.8 ±0.9% (CNN excluded). | Holds with width; dropout overtakes at 3× epochs. |
| Synaptic: helps over no regulariser | Not resolved, +1.1 ±1.3%. | No trend on any axis. |
| Synaptic: the schedule matters | No. One-shot is better. | Not measured off-centre. |

## 7. Limits the report must state

- "Larger" means 10–200× the papers' networks, up to 1.8M params (OBD) and 1M (synaptic).
  Modern scale is not tested.
- 3–5 seeds. Many intervals include zero; single-cell numbers are indicative only.
- OBD's "pruned beats unpruned" comparison is confounded: the unpruned net gets no extra
  epochs. A control that gives it the same retraining epochs is missing and is a small rerun
  (centre units only).
- Baselines are weak in places: MNIST MLP is 93.4% test (99.8% train); CIFAR-10 MLP is
  51%; inputs for MNIST and Fashion-MNIST are 16×16.
- The CNN forecaster does not beat persistence on two datasets; report with and without it.
- `runs.jsonl` contains 48 seed-0 runs outside the current design. `--tables` reads them.
  "Without CNN" numbers here exclude them; "all architectures" numbers include them, which
  shifts those by a small amount. Filter before producing final tables.
- Controls were not all scaled: OBD random and Taylor, and synaptic one-shot and narrow,
  exist at the centre only.

## 8. Figures to make

The stock set from `main.py --tables --plot` is committed (2026-10-02): `obd/figures/`
(retrain_curves, sweep_curves, paired_heatmap, scale_effects, overlap_curves,
overlap_vs_weight_decay) and `synaptic_pruning/figures/` (methods_core, controls,
scale_axes, sweep_heatmap). They were not checked against the list below, and the synaptic
ones include the CNN and the 48 off-design runs.
1. OBD: test accuracy vs keep, one panel per model, lines for OBD, magnitude, random, with
   the unpruned level as a horizontal line. Carries §4.1 and §4.3.
2. OBD: accuracy lost at keep 1% vs width, OBD and magnitude. Carries §4.4.
3. OBD: kept-set overlap vs keep, one line per model. Carries §4.5.
4. Synaptic: % improvement over no regulariser per method at the base setting, with and
   without CNN. Carries §5.1 and §5.2.
5. Synaptic: pruning, dropout, random pruning vs width. Carries §5.3.
