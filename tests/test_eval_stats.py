import math

import numpy as np
import pytest

from eval.stats import (
    bootstrap_ci,
    bootstrap_stat,
    cohens_kappa,
    mcnemar_exact,
    paired_bootstrap,
    verdict,
)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    x = [1.0] * 60 + [0.0] * 40
    m, lo, hi = bootstrap_ci(x)
    assert m == pytest.approx(0.6) and lo < 0.6 < hi
    assert hi - lo == pytest.approx(
        2 * 1.96 * math.sqrt(0.6 * 0.4 / 100), abs=0.015
    )  # normal approximation
    assert bootstrap_ci(x) == (m, lo, hi)  # seeded
    assert all(math.isnan(v) for v in bootstrap_ci([]))
    assert bootstrap_ci([1.0] * 20) == (1.0, 1.0, 1.0)


def test_bootstrap_uses_ten_thousand_resamples_by_default():
    from eval import stats

    assert stats.N_BOOT == 10_000


def test_bootstrap_stat_for_a_ratio():
    hit = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=float)
    seen = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=float)  # precision = hits among "seen"
    stat = lambda idx: hit[idx][seen[idx] == 1].sum() / max(1, seen[idx].sum())  # noqa: E731
    p, lo, hi = bootstrap_stat(8, stat, n_boot=500)
    assert p == pytest.approx(5 / 7) and lo <= p <= hi


def test_paired_bootstrap_detects_a_clear_difference_and_declines_a_null_one():
    rng = np.random.default_rng(0)
    base = (rng.random(200) < 0.5).astype(float)
    better = np.maximum(base, (rng.random(200) < 0.3).astype(float))
    clear = paired_bootstrap(better, base)
    assert clear["mean_diff"] > 0.1 and clear["lo"] > 0 and clear["verdict"] == "first better"
    assert paired_bootstrap(base, base)["verdict"] == "no detectable difference"
    flip = paired_bootstrap(base, better)
    assert flip["verdict"] == "second better" and flip["mean_diff"] == pytest.approx(
        -clear["mean_diff"]
    )
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1])


def test_verdict_wording():
    assert verdict(-0.1, 0.1) == "no detectable difference"
    assert verdict(0.0, 0.2) == "no detectable difference"  # a CI touching zero does not exclude it
    assert verdict(float("nan"), 1) == "not measurable"


@pytest.mark.parametrize(
    "a, b, only_a, only_b, p",
    [
        ([1] * 5 + [0] * 5, [0] * 5 + [0] * 5, 5, 0, 2 * 0.5**5),  # exact: 2 * P(X <= 0), n = 5
        ([1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], 3, 3, 1.0),
        ([1, 0, 1], [1, 0, 1], 0, 0, 1.0),
        ([1] * 8 + [0] * 2, [0] * 8 + [1] * 2, 8, 2, 2 * (1 + 10 + 45) / 2**10),
    ],
)
def test_mcnemar_exact(a, b, only_a, only_b, p):
    r = mcnemar_exact(np.array(a, bool), np.array(b, bool))
    assert (r["only_a"], r["only_b"]) == (only_a, only_b) and r["p"] == pytest.approx(p)


def test_cohens_kappa_known_values():
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0
    # 2x2: both yes 20, A yes/B no 5, A no/B yes 10, both no 15 -> po = 0.7, pe = 0.5, kappa = 0.4
    a = [True] * 25 + [False] * 25
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4)
    assert math.isnan(cohens_kappa([], []))
