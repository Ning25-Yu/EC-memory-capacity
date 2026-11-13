"""Statistics for the published p16 data (analysis script 21): descriptive summaries, Holm adjustment, exact
Jonckheere-Terpstra trend tests and log-ratio analyses."""

import itertools
import math

import numpy as np
from scipy import stats


def holm(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    prev = 0.0
    m = len(values)
    for k, i in enumerate(order):
        prev = max(prev, min(1.0, (m - k) * values[i]))
        out[i] = prev
    return out


def describe(values):
    y = np.asarray(values, dtype=float)
    n = len(y)
    assert n > 1 and np.isfinite(y).all()
    sd = float(np.std(y, ddof=1))
    m = float(np.mean(y))
    margin = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return dict(
        n=n,
        mean=m,
        SD=sd,
        minimum=float(min(y)),
        maximum=float(max(y)),
        pointwise_t95_lower=m - margin,
        pointwise_t95_upper=m + margin,
    )


def jt_score(groups):
    score = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = np.asarray(groups[i])[:, None]
            b = np.asarray(groups[j])[None, :]
            score += float(np.sum(a > b) + 0.5 * np.sum(a == b))
    return score


def exact_jt_independent(groups):
    """One-sided JT for decreasing outcome; enumerate all distinct label allocations."""
    assert len(groups) == 3
    y = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    ids = tuple(range(len(y)))
    observed = jt_score(groups)
    # Scores depend on ordinal comparisons, including half weight for ties.
    relation = (y[:, None] > y[None, :]).astype(float) + 0.5 * (y[:, None] == y[None, :])
    total = extreme = 0
    for a in itertools.combinations(ids, sizes[0]):
        aset = set(a)
        rest = tuple(i for i in ids if i not in aset)
        for b in itertools.combinations(rest, sizes[1]):
            bset = set(b)
            c = tuple(i for i in rest if i not in bset)
            score = relation[np.ix_(a, b)].sum() + relation[np.ix_(a, c)].sum() + relation[np.ix_(b, c)].sum()
            total += 1
            extreme += score >= observed - 1e-10
    assert total == math.factorial(len(y)) // math.prod(math.factorial(n) for n in sizes)
    return dict(
        JT_score=observed,
        permutations=total,
        extreme=int(extreme),
        p_one_sided=float(extreme / total),
        assumption="Independent groups and exchangeable condition labels under the null",
    )


def exact_jt_blocked(groups):
    """Alternative layout only; rows are matched blocks IF source Rep rows are paired."""
    matrix = np.column_stack(groups)
    observed = jt_score(groups)
    permutations = list(itertools.permutations(range(3)))
    total = extreme = 0
    for choices in itertools.product(permutations, repeat=len(matrix)):
        changed = np.array([row[list(c)] for row, c in zip(matrix, choices)])
        score = jt_score(changed.T)
        total += 1
        extreme += score >= observed - 1e-10
    return dict(
        JT_score=observed,
        permutations=total,
        extreme=extreme,
        p_one_sided=extreme / total,
        assumption="Source Rep rows identify matched experiments and condition labels are exchangeable within rows",
    )


def exact_signflip(log_values):
    y = np.asarray(log_values)
    observed = abs(float(np.mean(y)))
    all_scores = [abs(float(np.mean(y * np.asarray(s)))) for s in itertools.product([-1, 1], repeat=len(y))]
    k = sum(x >= observed - 1e-14 for x in all_scores)
    return dict(
        p_two_sided=k / len(all_scores),
        permutations=len(all_scores),
        extreme=k,
        assumption="Independent experimental log ratios have sign-symmetric errors under the null",
    )


def log_ratio_analysis(values):
    y = np.asarray(values, dtype=float)
    assert (y > 0).all()
    z = np.log(y)
    d = describe(z)
    t = stats.ttest_1samp(z, 0)
    flip = exact_signflip(z)
    return dict(
        n=len(y),
        arithmetic_mean=float(np.mean(y)),
        arithmetic_SD=float(np.std(y, ddof=1)),
        geometric_mean_ratio=math.exp(d["mean"]),
        log_t95_lower=math.exp(d["pointwise_t95_lower"]),
        log_t95_upper=math.exp(d["pointwise_t95_upper"]),
        log_t_statistic=float(t.statistic),
        log_t_df=len(y) - 1,
        log_t_p_two_sided=float(t.pvalue),
        exact_signflip_p_two_sided=flip["p_two_sided"],
        exact_signflip_permutations=flip["permutations"],
        t_assumption="Independent approximately normal log ratios across biological experiments; pointwise interval",
        signflip_assumption=flip["assumption"],
    )
