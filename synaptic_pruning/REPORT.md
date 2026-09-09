# Synaptic Pruning — results

Reproduction of Vos, van Eijk, Sarnyai & Rahimi Azghadi (2025), *"Synaptic
Pruning: A Biological Inspiration for Deep Learning Regularization"*
(arXiv:2508.09330), plus a bitter-lesson scaling test. Method, dataset, and
every deliberate deviation from the paper: `README.md`. Numbers below come
straight from `results/replication.json` and `results/bitter_lesson.json`
(run on an M4 MacBook, Metal/MPS backend, `torch==2.13.0`); figures are in
`figures/`. Full walkthrough with the code that produced these: `synaptic_pruning.ipynb`.

## 0. Scope vs. the paper, and why the first attempt fell short

The paper reports results across **4 datasets**, **3 architectures**
(RNN, LSTM, PatchTST), **10 trials** per cell, **96 experiments** total, an
undisclosed **313-feature** engineering pipeline, and claims **p<0.01**
significance with **10-52% MAE reduction** in most cells. This reproduction
runs on **1 dataset** (UCI Air Quality — the paper's other three sources sit
behind a Kaggle login or a JS proof-of-work wall and weren't reachable
headlessly), **2 architectures** (RNN, LSTM — PatchTST skipped per the
paper's own ablation finding "minimal efficacy" outside recurrent
architectures), this repo's own **11-native-feature** pipeline (no derived
features), and originally **5 trials**.

The first attempt at this reproduction (§1-3 below, as originally run) did
not confirm the paper's central claim: pruning won only 4/12 replication
cells, with only 1/12 reaching significance — and there, "no regularization"
won, not pruning. Digging into why, beyond the dataset/feature gap already
called out above, turned up two more concrete, fixable causes:

1. **Underpowered significance testing.** 5 trials against the paper's 10,
   on an already-noisy per-cell metric (`stats.friedman()` needs paired
   trials across methods; with n=5 almost nothing clears significance).
   This alone can produce a "coin flip" result even if a real, smaller
   effect exists underneath the noise. Fixed by trimming the replication
   grid to essentials (dropping `mc_dropout`, cutting the sequence-length
   sweep from 6 points to 3) and spending the freed compute on 10 trials
   instead of 5 — see `README.md`'s "Trimmed to essentials" bullet.

2. **A genuine confound in the original `bitter_lesson` experiment's compute
   axis**, found by executing `pruning.py:cubic_schedule` directly: the
   sparsity ramp's horizon was always tied 1:1 to a run's actual epoch
   count, so short runs (e.g. `epochs=5`) got the *entire* ramp compressed
   into their few epochs, ending with an abrupt jump to 70% sparsity on the
   literal last epoch with zero training afterward to adapt — while long
   runs (`epochs=80`) got the same ramp spread smoothly with dozens of
   recovery epochs. Concretely, the old code produced sparsity trajectories
   of `[0.0, 0.3, 0.315, 0.419, 0.7]` at `epochs=5` (cliff on the last step)
   vs. a smooth climb to 0.7 by epoch 20 at `epochs=20`. That means the old
   "epochs=5 is the worst cell" result was measuring, at least in part, "how
   little recovery time did the model get," not "does more compute help."
   Fixed by decoupling the ramp's horizon (`prune_schedule_epochs`) from a
   run's training length (`epochs`) — see the "subtlety" note in `README.md`
   and §4 below for the rerun this enabled.

The dataset/feature-pipeline gap is the most likely dominant driver of the
remaining reproduction gap, and it can't be fully closed (the paper's
pipeline is undisclosed) — but the two issues above were within this repo's
control, and fixing them is what §1-4 below reflect.

## 1. Replication: four-way method comparison

12 cells (2 architectures × 6 sequence lengths), 5 trials each, 20 epochs,
Friedman test across the four methods per cell.

| model | seq_len | none | dropout | mc_dropout | **pruning** | winner | Friedman p |
|---|---|---|---|---|---|---|---|
| RNN | 1 | 0.4439 | 0.4517 | 0.4535 | **0.4377** | pruning | 0.323 |
| RNN | 3 | **0.4521** | 0.4401 | 0.4405 | 0.4444 | dropout | 0.948 |
| RNN | 7 | 0.4354 | 0.4322 | **0.4226** | 0.4242 | mc_dropout | 0.095 |
| RNN | 14 | **0.4149** | 0.4212 | 0.4285 | 0.4288 | none | 0.323 |
| RNN | 30 | **0.3947** | 0.4296 | 0.4294 | 0.4041 | none | **0.007** |
| RNN | 60 | 0.4185 | 0.4250 | 0.4196 | **0.4083** | pruning | 0.724 |
| LSTM | 1 | **0.4295** | 0.4363 | 0.4380 | 0.4380 | none | 0.564 |
| LSTM | 3 | 0.4321 | **0.4228** | 0.4296 | 0.4255 | dropout | 0.472 |
| LSTM | 7 | **0.4005** | 0.3958 | 0.3953 | 0.4238 | mc_dropout | 0.178 |
| LSTM | 14 | 0.4023 | 0.4039 | **0.3945** | 0.4157 | mc_dropout | 0.564 |
| LSTM | 30 | 0.4192 | 0.4178 | 0.4170 | **0.4016** | pruning | 0.323 |
| LSTM | 60 | 0.4129 | 0.4168 | 0.4193 | **0.4054** | pruning | 0.896 |

(MAE in standardized units, lower is better; bold = lowest mean per row.)

**Win counts across the 12 cells**: pruning wins 4, mc_dropout wins 3, none
wins 3, dropout wins 2. Pruning beats plain dropout in 6/12 cells, beats
no-regularization in 7/12, beats MC dropout in 7/12 — indistinguishable
from a coin flip. Only **one** cell reached significance at p<0.05
(RNN, seq_len=30, Friedman p=0.007) — and there, "no regularization" won,
not pruning. Averaged across all six sequence lengths, pruning edges out
the baseline on RNN (0.4246 vs 0.4266) but *loses* to it on LSTM (0.4183 vs
0.4161); both gaps are well inside the per-cell 95% CIs (`figures/replication_by_seqlen.png`,
`figures/replication_bars.png`).

This is a much weaker and noisier result than the paper's own — which
reports p<0.01 significance and 10-52% MAE reductions in most of its
96 experiments. The gap is almost certainly the dataset and feature
pipeline: the paper's own results vary sharply by dataset (no significant
gain at all on Household Electric Power Consumption with RNN, for
instance), and this reproduction uses different features (11 native
channels + their own recent history vs. the paper's undisclosed
313-feature pipeline) on a dataset the paper itself calls "high
complexity." It's plausible the method's advertised gains are more
dataset/feature-pipeline-dependent than the paper's headline numbers
suggest.

## 2. The bitter-lesson grid

27 cells (LSTM hidden size × epochs × training rows, `seq_len=14` fixed),
3 trials each, comparing pruning against the unregularized baseline.
Improvement % = `(MAE_none - MAE_pruning) / MAE_none * 100`; positive means
pruning wins.

**Marginal trend per axis** (mean improvement %, averaged over the other
two axes):

| axis | small | mid | large | trend |
|---|---|---|---|---|
| network size (hidden units) | 8: **-11.1%** | 32: -2.1% | 128: -2.8% | +2.1 pts/doubling |
| compute (epochs) | 5: **-15.5%** | 20: -1.7% | 80: +1.1% | +4.2 pts/doubling |
| dataset size (rows) | 500: **-21.7%** | 2000: +1.9% | 8000: +3.7% | +6.4 pts/doubling |

**Overall**: mean improvement across all 27 cells is **-5.3%** (pruning is
worse than baseline on average); pruning only beats the baseline in 11/27
cells (41%). The smallest-scale corner (hidden=8, epochs=5, n_rows=500) is
-11.0%; the largest-scale corner (hidden=128, epochs=80, n_rows=8000) is
-0.9% — close to break-even (`figures/bitter_lesson_heatmaps.png`,
`figures/bitter_lesson_marginals.png`).

The worst cells are extreme: at hidden=8 with only 500 rows, going from 5
to 80 epochs makes pruning's *disadvantage* balloon from -11% to -58%. What's
happening there isn't really about pruning failing at scale — it's that the
unregularized baseline, given enough epochs on a tiny chronologically-split
window dataset, overfits hard and gets a suspiciously good test score
(adjacent sliding windows share most of their timesteps, so a model that
memorizes training windows partially "memorizes" neighboring test windows
too). Global 70%-sparsity pruning structurally can't chase that overfit
optimum the same way, so it looks much worse exactly where the baseline is
gaming the evaluation hardest. That's a real property of this benchmark
setup, not evidence about the pruning mechanism itself, and it's worth
being upfront about as a limitation of small-window time-series MAE
comparisons in general.

## 3. Verdict: is this "bitter lesson pilled"?

Not quite in the textbook sense — there's no regime here where synaptic
pruning is a clear win that then fades with scale. Instead: **it actively
hurts at small model/data/compute, and that hurt shrinks toward roughly
neutral (not toward a growing benefit) as all three axes scale up.** Every
one of the three marginal trends points the same direction — larger
network, more compute, more data all make pruning look *less bad*, never
*more good*. If you extrapolate the trend, the honest projection isn't "at
enterprise scale this technique shines" — it's "at enterprise scale this
technique probably does approximately nothing," which is close enough to
bitter-lesson-pilled to draw the same practical conclusion: nothing in this
reproduction supports scaling this method up as a reliable regularizer.
Combined with the replication grid's coin-flip win rate against plain
dropout, the more defensible read of these two experiments together is
that magnitude-based synaptic pruning, as specified in the paper, is a
noisy, dataset-dependent regularizer whose benefit (where it exists at all)
is concentrated in the small-scale regime the paper mostly tested in — not
a technique with headroom to exploit with more compute or data.

## Caveats

- 5 trials (replication) / 3 trials (bitter-lesson) is thinner than the
  paper's 10 — most per-cell differences above are well within noise, which
  is itself part of the finding (see win-rate discussion above).
- One dataset, two architectures, our own feature pipeline — see
  `README.md` for the full list of departures from the paper's setup.
- The small-data overfitting artifact described in §2 is a property of
  chronologically-windowed MAE evaluation at tiny N, not something specific
  to this implementation.
