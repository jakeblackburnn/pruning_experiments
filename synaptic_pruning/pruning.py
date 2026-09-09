"""Synaptic Pruning — Vos, van Eijk, Sarnyai & Rahimi Azghadi 2025
("Synaptic Pruning: A Biological Inspiration for Deep Learning
Regularization", arXiv:2508.09330). A dropout replacement: instead of
stochastically zeroing activations each forward pass, permanently zero the
globally-smallest-magnitude weights, on a schedule that ramps sparsity up
over training. Three pieces, matching the paper's Algorithms 1-6:

  mask init         one binary mask per weight tensor, all ones
  cubic schedule    target sparsity s(t), 0 during warmup then a cubic
                     ramp from smin to smax
  global selection  every `prune_every` batches, thread every unpruned
                     weight in the network into one pool, take the
                     magnitude threshold that hits s(t), zero what's below it

Masks are reapplied after every optimizer step (Algorithm 5) so a pruned
weight can't drift back off zero via momentum.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class PruneConfig:
    smin: float = 0.3
    smax: float = 0.7
    warmup_epochs: int = 2
    schedule_epochs: int = 20  # ramp horizon; independent of a run's actual
                                # total epoch count (see train.py:RunConfig.epochs)
    prune_every: int = 5  # batches


def cubic_schedule(epoch, cfg: PruneConfig):
    """Target sparsity s(t) — Algorithm 2. `epoch` is 1-indexed (1..schedule_epochs),
    matching the paper's pseudocode, so that s(schedule_epochs) == smax exactly,
    regardless of how many epochs the run actually trains for. If the run trains
    longer than schedule_epochs, sparsity holds at smax for the remaining epochs
    (progress is clamped to 1.0) so the extra epochs are genuine fine-tuning at
    fixed sparsity, not more ramping."""
    if epoch < cfg.warmup_epochs:
        return 0.0
    span = max(1, cfg.schedule_epochs - cfg.warmup_epochs)
    progress = (epoch - cfg.warmup_epochs) / span
    progress = min(1.0, max(0.0, progress))
    return cfg.smin + (cfg.smax - cfg.smin) * progress ** 3


class SynapticPruner:
    """Attach to a model, call `step(epoch)` after `loss.backward()` and
    `apply_masks()` after every `optimizer.step()` (Algorithm 4's ordering).

    Prunable parameters: every weight tensor with ndim >= 2 (Linear weights,
    and RNN/LSTM weight_ih_l*/weight_hh_l*) — biases and norm parameters are
    1-D and excluded, matching the paper's "weight tensor" scope (Algorithm 1).
    """

    def __init__(self, model: torch.nn.Module, cfg: PruneConfig = None):
        self.model = model
        self.cfg = cfg or PruneConfig()
        self.params = {name: p for name, p in model.named_parameters()
                        if p.requires_grad and p.dim() >= 2}
        self.masks = {name: torch.ones_like(p, dtype=torch.bool)
                       for name, p in self.params.items()}
        self.batch_count = 0
        self.last_target_sparsity = 0.0

    @torch.no_grad()
    def apply_masks(self):
        """Algorithm 5."""
        for name, p in self.params.items():
            p.mul_(self.masks[name])

    @torch.no_grad()
    def _global_magnitude_prune(self, target_sparsity):
        """Algorithm 3."""
        active_chunks = [p[self.masks[name]].abs()
                          for name, p in self.params.items()
                          if self.masks[name].any()]
        if not active_chunks:
            return
        all_active = torch.cat(active_chunks)
        n_total = sum(p.numel() for p in self.params.values())
        n_active = all_active.numel()
        n_target_pruned = int(target_sparsity * n_total)
        n_currently_pruned = n_total - n_active
        n_additional = max(0, n_target_pruned - n_currently_pruned)
        if n_additional == 0 or n_additional >= n_active:
            return
        threshold = torch.kthvalue(all_active, n_additional).values
        for name, p in self.params.items():
            mask = self.masks[name]
            self.masks[name] = mask & ~((p.abs() < threshold) & mask)
        self.apply_masks()

    def step(self, epoch):
        """Call once per training batch, after backward() and before
        optimizer.step() (Algorithm 4). `epoch` is 1-indexed (1..the run's
        total epochs, which may exceed cfg.schedule_epochs).
        Returns the target sparsity if a prune update ran this batch, else None."""
        self.batch_count += 1
        if epoch < self.cfg.warmup_epochs or self.batch_count % self.cfg.prune_every != 0:
            return None
        target = cubic_schedule(epoch, self.cfg)
        self._global_magnitude_prune(target)
        self.last_target_sparsity = target
        return target

    def sparsity_stats(self):
        """Algorithm 6, plus an overall figure."""
        stats, total, pruned = {}, 0, 0
        for name, mask in self.masks.items():
            n = mask.numel()
            npruned = int((~mask).sum().item())
            stats[name] = {"total": n, "pruned": npruned, "sparsity": npruned / n}
            total += n
            pruned += npruned
        stats["_overall"] = {"total": total, "pruned": pruned,
                              "sparsity": pruned / total if total else 0.0}
        return stats
