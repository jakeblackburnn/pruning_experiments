# Code walkthrough

Five modules, in dependency order. Reading them top to bottom follows the data:

```
obd.py          config → device → models → training → Hessian / saliencies → pruning primitives
experiments.py  the three experiment families + the --only registry
figures.py      results JSON → matplotlib (no torch)
datasets.py     MNIST → 16×16, CIFAR-10
main.py         disk I/O + CLI
```

The three pipeline stages communicate only through files, so each can be rerun
alone:

```
--train   →  checkpoints/<name>.pt      (trained weights)
--prune   →  results/<name>.json        (every experiment's measurements)
--plot    →  figures/fig{2..7}_*.png
```

## 1. Configuration and device

`Experiment` is a frozen dataclass holding everything that differs between the
two setups: the loss (`"mse"` for the paper, `"ce"` for the modern net), the
device, training hyperparameters, and the pruning schedule (how many sweep
points, how far to prune, the shrink factor for the retrain loop). Two presets
exist — `DIGITS` (the paper) and `CIFAR` (the modern counterpart) — and
`EXPERIMENTS` maps names to them for the CLI. Because the dataclass is frozen,
`main.configure` applies `--set` overrides with `dataclasses.replace`
(`--set epochs=5 lr=3e-4`) rather than mutation, so a preset is never silently
changed.

`device` defaults to `"auto"`, and `resolve_device` turns that into **MPS, else
CUDA, else CPU** exactly once, at the CLI (or notebook) entry point — so every
downstream function sees a concrete device string and nothing has to re-check
availability. An explicit `--set device=cpu` is passed through untouched.

The one global that isn't in the dataclass: `SEED = 1989`, set before training
and before *each* pruning experiment, so results are reproducible and don't
depend on which subset of experiments you run.

## 2. Data (`datasets.py`)

Both datasets are small enough to hold fully in memory as single tensors — no
`DataLoader`, batching is manual slicing everywhere downstream.

Each loader returns `((x_train, y_train, labels_train), (x_test, y_test,
labels_test))`. The convention: **`y` is whatever the loss consumes, `labels`
are always integer classes** for computing accuracy. For digits, `y` is a ±1
one-hot matrix (the paper trains tanh outputs with MSE to ±1 targets); for
CIFAR, cross-entropy consumes the integer labels directly, so `y` *is*
`labels`.

Digits mimics the 1989 zip code data, which is not publicly available: MNIST
is bilinearly downsampled to 16×16, rescaled to [-1, 1], and truncated to the
paper's exact sizes (7,291 train / 2,007 test).

## 3. Models

`MnistNet` is the zip-code network from LeCun et al. 1989. Three details
matter:

- **Scaled sigmoid** `1.7159·tanh(2x/3)`: the ±1 targets sit inside its linear
  range, so the net can fit them without saturating. This is load-bearing for
  OBD — with a plain tanh, units saturate, curvature collapses to ~0, and the
  saliency ranking becomes noise.
- **Sparse connection table**: each H2 map sees only 8 of the 12 H1 maps. The
  paper's exact table was never published, so a rotating scheme is used (map
  *j* reads maps *j..j+7* mod 12). It's enforced by multiplying `conv2.weight`
  with a fixed binary mask *in the forward pass* — the underlying tensor keeps
  its full shape, which keeps the Hessian and pruning code uniform.
- **Padding with −1**, the background value, so the 5×5/stride-2 convs map
  16→8→4.

`count_free_parameters` subtracts the entries the connection table zeroes out
(8,824 free parameters, vs the paper's ~10k). `CifarNet` is a deliberately
conventional VGG-style ReLU convnet (~590k parameters) — the point is to test
OBD outside the paper's carefully conditioned setup.

## 4. Training and evaluation

`train()` is plain Adam over shuffled slices. Two non-obvious points:

- **Weight decay is not cosmetic.** Without it, redundant large weights sit in
  flat directions of the loss where the quadratic saliency estimate breaks
  down (lesson 2 in the README). §7 gives it a second job.
- **Pruning is enforced by masks, not by removing weights.** A mask is a
  `{param_name: 0/1 tensor}` dict; `apply_masks` multiplies parameters by it
  in place. During retraining the optimizer still updates every weight, so the
  masks are reapplied *after every optimizer step* to keep pruned weights at
  zero.

`evaluate` accumulates sum-reduced loss and divides by `y.numel()`, matching
the mean-reduced training loss exactly — this matters later, because OBD's
*predicted* loss increases must be comparable to these measured ones.

## 5. Diagonal Hessian and saliencies

OBD's saliency is `s_k = ½ h_kk w_k²`, the diagonal second-order Taylor
estimate of the loss increase from deleting weight `w_k`. The paper computes
`h_kk` with the Levenberg–Marquardt approximation, which is exactly the
**Gauss-Newton diagonal**: `h_kk = Σ_samples [Jᵀ H_out J]_kk`, where `J` is the
Jacobian of the network outputs and `H_out` the loss's Hessian w.r.t. the
outputs.

- For MSE, `H_out` is a constant diagonal, so `h_kk` reduces to a sum of
  squared output-Jacobian entries.
- For softmax cross-entropy, `H_out = diag(p) − ppᵀ`, giving the
  variance-like form implemented with the two `einsum`s. It is a per-sample
  variance, so it is non-negative by Jensen — the Gauss-Newton diagonal is
  PSD by construction, which is what lets §7 take its logarithm.

Instead of the paper's hand-rolled layer-by-layer second-order backprop, the
per-sample Jacobians come from `torch.func`: `vmap(jacrev(outputs))` gives a
`(batch, outputs, *param_shape)` Jacobian per parameter. That tensor is big, so
the batch size is chosen to keep it around 100M floats, and CIFAR subsamples
the training set (`hessian_samples=2048`). The `names` argument restricts the
differentiation to the tensors that can actually be pruned, skipping the bias
Jacobians entirely. Both loss branches scale `h` to be consistent with the
*mean*-reduced loss, so predicted and measured loss increases live on the same
scale (that's what makes Fig 3 an honest plot).

`torch.func` coverage on MPS varies by build, so `diagonal_hessian` catches a
failure on an accelerator and retries the pass on CPU with a warning rather
than aborting the run.

## 6. The pruning experiments (`experiments.py`)

Mask bookkeeping: only conv/linear *weights* are prunable (`prunable_names`,
biases excluded), and `initial_masks` starts from the connection table so
structurally absent weights are never counted. To rank globally across layers,
`flatten`/`unflatten` concatenate the per-tensor dicts into one long vector and
back. `prune_to` keeps the top-n by score; already-dead weights get score
`+inf` so they can never be resurrected. `sweep_targets` produces the log-spaced
deletion schedule — log-spaced because nothing happens between 100% and 80%
remaining and everything happens in the last decade.

`scores_for` implements three ranking rules, which differ *only* in how much
they let the curvature vary:

| `rank_by` | score | assumption about `h` |
|---|---|---|
| `magnitude` | `w²` | constant everywhere |
| `saliency_layermean` | `½·h̄_L·w²` | constant within each layer |
| `saliency` | `½·h_kk·w²` | the full diagonal |

`sweep_no_retrain` (Figs 2 & 3) deletes more and more weights with **no
retraining**, at those log-spaced targets, either ranking once on the trained
net or re-ranking on the pruned net before each further deletion. The
distinction reproduces the paper: the one-shot ranking goes stale because the
saliency is a *local* estimate. (Magnitude ranking is unaffected — deletion
doesn't change the surviving `|w|`.) For the saliency variants it also
accumulates `predicted_increase` (the sum of deleted saliencies — OBD's
forecast) alongside `actual_increase` (measured train-loss change), which is
exactly the data for Fig 3.

`iterative_prune_retrain` (Fig 4) is the full OBD loop: evaluate → recompute
scores → delete down to `shrink × n` weights → retrain briefly with masks
held → repeat until `retrain_min_remaining`. It takes a `rank_by` argument:
`"saliency"` is OBD proper, `"magnitude"` is the **control** — the identical
prune-retrain loop ranked by `|w|`, which isolates whether the second-order
ranking (and not just iterative retraining) deserves credit for Fig 4.

## 7. Is magnitude just OBD in disguise? (`overlap_analysis`)

The experiment the paper didn't run. Start from the definition:

```
s_k = ½ · h_kk · w_k²
```

**Claim 1 — the exact reduction.** Global magnitude pruning ranks by `w_k²`.
So ranking by saliency and ranking by `|w|` produce the *identical* ordering
whenever `h_kk` is constant across `k`. Magnitude pruning is not a cheap
approximation to OBD; it is OBD under an isotropic-curvature prior. The whole
question is how far from constant `h_kk` actually is.

**Claim 2 — the decomposition.** In logs the saliency separates additively:

```
log s_k = 2·log|w_k| + log(h_kk / 2)
```

so with variances `σ_w²`, `σ_h²` and covariance `σ_hw`,

```
corr(log s, log|w|) = (2σ_w² + σ_hw) / (σ_w · √(4σ_w² + σ_h² + 4σ_hw))
```

The rankings agree closely whenever `σ_h ≪ 2σ_w`, and positive covariance
raises the agreement further. `results["overlap"]["global"]` reports all three
moments plus `predicted_pearson` from that formula alongside the directly
measured `pearson_log` — they must match, which is the check that the
decomposition is being computed on what it claims. `spearman` is the exact rank
correlation over all live weights; `spearman_gaussian_estimate` is
`(6/π)·arcsin(ρ/2)`, what the rank correlation would be if `(log h, log|w|)`
were jointly Gaussian, so the gap between them measures how heavy the real
tails are.

**Claim 3 — why `σ_h` is small *within* a layer.** For a weight `W_ij` with
input activation `x_j` and pre-activation `a_i`,

```
∂out_o/∂W_ij = (∂out_o/∂a_i) · x_j
```

so the Gauss-Newton diagonal factorizes:

```
h_ij ≈ E[x_j²] · E[δ_i²] = u_i · v_j
```

— **rank-one** across the weight matrix (exact when input energy and output
sensitivity are uncorrelated across samples; for conv layers additionally
averaged over spatial positions). Two consequences. Within a fan-in group
(fixed output unit `i`), `h_ij ∝ v_j`, and normalized inputs make that nearly
flat, so the OBD ordering collapses onto the magnitude ordering *inside* a
layer. What survives is a per-layer offset in log space. `between_layer_frac`
is that claim as a single number: the fraction of `Var(log h)` explained by
differences between layer means. **Fig 6** is the same claim as a picture —
one log-log panel per layer, where a slope-2 line is forced by `s ∝ w²` and
only the *intercept* is free.

**Claim 4 — why weight decay aligns them.** At a stationary point of
`E + ½λ‖w‖²`, the quadratic model gives

```
w_k = h_kk/(h_kk + λ) · w*_k
```

(heuristic — it assumes a diagonal Hessian in the same basis). Low-curvature
coordinates are shrunk toward zero; high-curvature ones are left alone. Weight
decay therefore *induces* `σ_hw > 0`: it makes small magnitude a symptom of low
curvature. That closes a loop with lesson 2 in the README — the same
regularizer that gives OBD a well-conditioned Hessian is what makes `|w|`
informative about that Hessian. It also predicts a difference between the two
presets, since digits trains at `weight_decay=1e-3` and cifar at `1e-4`;
`cov_log_h_log_w` and `weight_decay` are both in the results so the comparison
is direct.

**Claim 5 — the ladder settles it.** `magnitude ⊂ saliency_layermean ⊂
saliency`: constant `h`, constant-per-layer `h`, full diagonal `h`. If the
layer-mean curve tracks full OBD on Fig 2, then within-layer curvature carries
no usable signal and OBD's entire contribution is *cross-layer budget
allocation* — which is also the standard explanation for why global magnitude
pruning beats layerwise magnitude pruning in the modern literature. **Fig 7**
shows that allocation directly, one panel per layer. Note where the argument
stops: methods that use off-diagonal curvature (OBS, WoodFisher, K-FAC-based
pruning) are outside this decomposition entirely.

Implementation notes: everything is computed once, at the trained weights, over
the *live* prunable entries (connection-table zeros excluded), in float64 on
CPU. Weights or curvatures that are exactly zero have no logarithm; they are
excluded from the log-space statistics and counted in `n_dropped_zero` rather
than silently kept. Rank correlations are done in torch (`argsort` twice, then
Pearson on the ranks) so there's no scipy dependency; Kendall's τ is O(n²) and
so runs on a fixed 2,048-weight subsample while Spearman is exact.

## 8. Results file

`run_experiments` runs the experiments in `PRUNE_EXPERIMENTS` and writes
`results/<name>.json`. It *merges* into an existing results file, and each
experiment is seeded independently, so a subset (`--only ...`) can be added to
a cached run without redoing the others. Because merging is exactly how a file
ends up holding results from two different base models, every write stamps a
`_meta` block — the full config, the checkpoint's SHA-1, the torch version, a
timestamp — and `check_meta` warns on the next read if either has moved.

```json
{
  "magnitude":           [ {"remaining", "train_loss", "test_loss", "train_acc", "test_acc"}, ... ],
  "saliency":            [ { ...same, plus "predicted_increase", "actual_increase" }, ... ],
  "saliency_recomputed":            [ ...same as saliency... ],
  "saliency_layermean":             [ ...same as saliency... ],
  "saliency_layermean_recomputed":  [ ...same as saliency... ],
  "retrain":             [ { ...same base fields... }, ... ],
  "retrain_magnitude":   [ ...same base fields... ],
  "overlap": {
    "n_prunable", "n_dropped_zero", "layers": [...],
    "global":     {"spearman", "spearman_gaussian_estimate", "kendall_subsample",
                   "pearson_log", "predicted_pearson", "var_log_h", "var_log_w",
                   "cov_log_h_log_w", "between_layer_frac", "weight_decay"},
    "per_layer":  [ {"name", "n", "spearman_within", "log_h_mean", "log_h_std",
                     "log_w_std", "h_median"}, ... ],
    "overlap_curve":    [ {"remaining", "overlap_frac", "jaccard_kept",
                           "jaccard_pruned", "chance"}, ... ],
    "layer_allocation": [ {"remaining", "kept_obd", "kept_mag", "layer_size"}, ... ],
    "scatter":    {"layer": [...], "log_w": [...], "log_s": [...]}
  },
  "_meta": {"config", "checkpoint_sha1", "torch", "written"}
}
```

## 9. Figures

Pure functions from that JSON to matplotlib — no torch involved, so plots can
be iterated on without rerunning anything. Each returns `None` (with a printed
note) when its data is missing, so a partial run still plots what it has.

- **Fig 2** — train loss vs weights remaining (log-x) for every no-retrain
  variant: the paper's core comparison, the one-shot-vs-re-ranked staleness
  lesson, and the layer-mean control (§7 claim 5) on one axis.
- **Fig 3** — predicted vs actual loss increase on log-log axes, against the
  `y = x` "perfect prediction" line, for both the full and layer-mean saliency.
- **Fig 4** — train/test loss vs weights remaining for the prune-retrain
  loop, OBD saliency against the magnitude control.
- **Fig 5** — the fraction of the kept set the two rankings share, at every
  pruning level, against the chance baseline `n/N` (what two independent
  rankings would share).
- **Fig 6** — saliency vs `|w|`, log-log, one panel per layer, with that
  layer's slope-2 line and the network-wide one for comparison. §7 claim 3.
- **Fig 7** — the fraction of each layer kept by each ranking, across pruning
  levels. §7 claim 5.

Three colours carry fixed meanings everywhere: blue = OBD saliency (or
"train"), red = magnitude (or "test"), violet = the layer-mean control. They
are three slots of a CVD-validated categorical palette, and every series also
carries its own marker and linestyle so identity never rests on colour alone.

## 10. CLI

`python3 main.py [--train] [--prune] [--plot] [--dataset digits|cifar] [--only
KEY ...] [--set FIELD=VALUE ...]`. No stage flags means run all three.
`--only` restricts the prune stage to the named experiments (and by itself
implies `--prune`), merging them into the cached results. `--set` parses each
value against the type of the field's current value (`_coerce`), so
`epochs=5`, `lr=3e-4`, `device=cpu`, and `hessian_samples=none` all do the
right thing, and an unknown field name is rejected rather than ignored.

## Extending

To add a third experiment: define a model class and register it in `MODELS`,
add an `Experiment` preset (it registers itself via `EXPERIMENTS`), and add a
loader to `LOADERS` in `datasets.py`. Everything downstream — training,
Hessian, pruning, the overlap analysis, figures — is already generic over the
`Experiment` spec and over the model's layer list and output count.
