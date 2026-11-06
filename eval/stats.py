"""Statistics for the report: bootstrap CIs, paired bootstrap, exact McNemar, Cohen's kappa.

All resampling is over *questions* (the unit of evaluation), seeded, with 10,000 resamples. A
difference whose 95% CI contains zero is reported as "no detectable difference" (`verdict`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import comb

import numpy as np

N_BOOT = 10_000
SEED = 42
ALPHA = 0.05
_CHUNK = 1000


def _rng(seed: int, salt: str = "") -> np.random.Generator:
    return np.random.default_rng([seed, *[ord(c) for c in salt]])


def bootstrap_ci(
    values: Sequence[float], n_boot: int = N_BOOT, seed: int = SEED, alpha: float = ALPHA
) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap CI of the mean. Empty input -> (nan, nan, nan)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = _rng(seed)
    means = np.concatenate([
        x[rng.integers(0, x.size, size=(min(_CHUNK, n_boot - i), x.size))].mean(axis=1)
        for i in range(0, n_boot, _CHUNK)
    ])  # fmt: skip
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def bootstrap_stat(
    n: int,
    stat: Callable[[np.ndarray], float],
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> tuple[float, float, float]:
    """CI for any statistic of the question indices (ratios such as abstention precision)."""
    if n == 0:
        return (float("nan"),) * 3
    rng = _rng(seed, "stat")
    draws = np.array([stat(rng.integers(0, n, size=n)) for _ in range(n_boot)], dtype=float)
    draws = draws[~np.isnan(draws)]
    point = stat(np.arange(n))
    if draws.size == 0:
        return float(point), float("nan"), float("nan")
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> dict[str, float | int | str]:
    """Mean of (a - b) over the same questions, with a paired percentile-bootstrap CI."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    d = x - y
    mean, lo, hi = bootstrap_ci(d, n_boot, seed, alpha)
    return {"n": int(d.size), "mean_diff": mean, "lo": lo, "hi": hi, "verdict": verdict(lo, hi)}


def verdict(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "not measurable"
    if lo > 0:
        return "first better"
    if hi < 0:
        return "second better"
    return "no detectable difference"


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, float | int]:
    """Exact two-sided McNemar test on paired correct/incorrect outcomes.

    b = questions only A got right, c = only B got right; under H0 b ~ Binomial(b + c, 1/2).
    """
    a, b = np.asarray(a_correct, dtype=bool), np.asarray(b_correct, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "discordant": 0, "p": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_a": only_a, "only_b": only_b, "discordant": n, "p": p}


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    x, y = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if x.size == 0:
        return float("nan")
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)
