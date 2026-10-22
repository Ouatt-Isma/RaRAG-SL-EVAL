"""Accuracy from outcome strings, paired comparisons on the same questions, and aggregation over runs.

A unit is one dataset and seed. Within a unit two rows answer the same questions, so their difference is
tested question by question (McNemar's exact test; a paired bootstrap for its interval). Across units the
mean difference gets a t interval, and we count the units where the expected direction holds."""
import math
import random

from scipy import stats as sps


def accuracy(codes):
    return sum(c == "c" for c in codes) / len(codes) if codes else None


def rate(codes, mask, symbol):
    sel = [c for c, m in zip(codes, mask) if m]
    return sum(c == symbol for c in sel) / len(sel) if sel else None


def mcnemar(a, b, symbol="c"):
    """Exact McNemar test that rows a and b are right (symbol) equally often on the same questions."""
    only_a = sum(x == symbol and y != symbol for x, y in zip(a, b))
    only_b = sum(y == symbol and x != symbol for x, y in zip(a, b))
    n = only_a + only_b
    return 1.0 if n == 0 else min(1.0, sps.binomtest(only_a, n, 0.5).pvalue)


def paired_bootstrap(a, b, symbol="c", reps=1000, seed=0):
    """95% interval of share(b == symbol) - share(a == symbol), resampling questions."""
    n = len(a)
    if n == 0:
        return None, None
    rng = random.Random(seed)
    d = [(y == symbol) - (x == symbol) for x, y in zip(a, b)]
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def aggregate(diffs, direction):
    """Mean difference over units, its 95% t interval (when there are at least 2), and how many units go
    the expected way: direction +1 (higher is the claim), -1 (lower), 0 (no difference: |diff| < 1 point)."""
    xs = [d for d in diffs if d is not None and not math.isnan(d)]
    n = len(xs)
    if not n:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "holds": 0}
    mean = sum(xs) / n
    lo = hi = None
    if n >= 2:
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
        half = sps.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        lo, hi = mean - half, mean + half
    if direction > 0:
        holds = sum(x > 0 for x in xs)
    elif direction < 0:
        holds = sum(x < 0 for x in xs)
    else:
        holds = sum(abs(x) < 0.01 for x in xs)
    return {"n": n, "mean": mean, "lo": lo, "hi": hi, "holds": holds}
