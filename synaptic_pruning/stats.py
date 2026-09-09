"""Summary statistics across trials: mean, std, a 95% CI, and (when there
are enough paired trials) a Friedman test across methods — the same
significance test the paper reports its p-values with."""

import numpy as np
from scipy import stats as _stats


def summarize(values):
    """mean, std, 95% CI (normal approximation on the trial mean, matching
    the paper's reported CIs) for a 1-D list of trial values."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    z = 1.959963985
    return {"mean": mean, "std": std, "n": n,
            "ci95": [mean - z * se, mean + z * se]}


def friedman(method_to_trials):
    """Friedman chi-square test across methods, each with the same number
    of paired trials (same seeds). method_to_trials: {method: [mae_trial0,
    mae_trial1, ...]}. Returns {statistic, p_value} or None if there are
    fewer than 3 methods or fewer than 3 trials."""
    arrays = list(method_to_trials.values())
    if len(arrays) < 3 or min(len(a) for a in arrays) < 3:
        return None
    stat, p = _stats.friedmanchisquare(*arrays)
    return {"statistic": float(stat), "p_value": float(p)}
