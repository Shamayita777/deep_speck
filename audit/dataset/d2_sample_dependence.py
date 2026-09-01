"""
D2 — Sample Dependence Audit.

Generic Dataset Integrity implementation.

D2 is distinct from D1. D1 owns exact row duplication and exact partition
overlap. D2 evaluates non-exact repetition and dependence structures that can
remain when exact-equality tests pass.

The confirmatory dependence tests are conditional on the supplied finite
dataset instance.  Wherever possible, the null preserves the observed
marginal feature distribution rather than assuming perfectly balanced bits.
The Gohr adapter may additionally provide a nominal Binomial(64, 0.5)
reference as a case-study diagnostic; that nominal reference is not used as a
universal theorem of independence.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from audit.dataset.common.certificate import make_certificate, write_certificate
from audit.dataset.common.provenance import array_sha256, build_provenance

AUDIT_ID = "D2"
AUDIT_NAME = "Sample Dependence Audit"
SCHEMA_VERSION = "4.0"

FAMILYWISE_ALPHA = 0.01
TVD_THRESHOLD = 0.05
TVD_CONFIDENCE_LEVEL = 0.95
NEAR_DUPLICATE_RADII = (1, 2, 4, 8)
DEFAULT_LAGS = tuple(list(range(1, 17)) + [32, 64, 128, 256, 512, 1024])
DEFAULT_PAIRS_PER_TEST = 100_000
DEFAULT_AUDIT_REPLICATES = 10
DEFAULT_BOOTSTRAP_REPLICATES = 1_000
DEFAULT_MULTIVARIATE_PAIRS = 50_000
DEFAULT_MULTIVARIATE_PERMUTATIONS = 1_000
DEFAULT_DETECTOR_C = 1.0
MULTIVARIATE_AUC_TOLERANCE = 0.05
CALIBRATION_MIN_REPLICATES = 10
CALIBRATION_DETECTION_TARGET = 0.95


def _binary(x: np.ndarray, name: str, n_bits: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != n_bits:
        raise ValueError(f"{name} must have shape (n, {n_bits}); got {x.shape}.")
    if len(x) < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError(f"{name} must contain only binary values 0/1.")
    return x.astype(np.uint8, copy=False)


def validate_reference_pmf(pmf: np.ndarray, n_bits: int) -> np.ndarray:
    p = np.asarray(pmf, dtype=np.float64)
    if p.ndim != 1 or len(p) != n_bits + 1:
        raise ValueError("reference_pmf must have length n_bits + 1.")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or not math.isclose(float(p.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("reference_pmf must be finite, non-negative, and sum to one.")
    return p


def binomial_reference_distribution(n_bits: int, p: float = 0.5) -> np.ndarray:
    if n_bits < 1 or not 0 <= p <= 1:
        raise ValueError("Invalid binomial parameters.")
    pmf = np.array(
        [math.comb(n_bits, k) * p**k * (1 - p) ** (n_bits - k) for k in range(n_bits + 1)],
        dtype=float,
    )
    return pmf / pmf.sum()


def independent_hamming_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Plug-in Hamming null preserving empirical per-bit marginals.

    Under this diagnostic null, rows are independent and each bit uses its
    observed marginal probability.  The resulting Poisson-binomial model is
    therefore a *model-based approximation*: it also assumes independence of
    mismatch indicators across bit positions. It is not a universal null for
    arbitrary within-row bit dependence.
    """
    a = _binary(a, "a", a.shape[1])
    b = _binary(b, "b", b.shape[1])
    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must have the same feature width.")
    pa = a.mean(axis=0, dtype=np.float64)
    pb = b.mean(axis=0, dtype=np.float64)
    mismatch = pa * (1.0 - pb) + (1.0 - pa) * pb
    pmf = np.array([1.0], dtype=np.float64)
    for q in mismatch:
        pmf = np.convolve(pmf, np.array([1.0 - q, q], dtype=np.float64))
    return pmf / pmf.sum()


def _sample_disjoint_pairs(n: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample unordered pairs with no repeated endpoint in one replicate."""
    if n < 2 or count < 1:
        raise ValueError("Need n >= 2 and count >= 1.")
    if 2 * count > n:
        raise ValueError(f"Cannot sample {count:,} disjoint pairs from {n:,} samples.")
    indices = rng.choice(n, size=2 * count, replace=False)
    return indices[:count].astype(np.int64), indices[count:].astype(np.int64)


def _sample_cross_pairs(n_a: int, n_b: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample cross-partition pairs without endpoint reuse within a replicate."""
    if n_a < 1 or n_b < 1 or count < 1:
        raise ValueError("Invalid cross-partition sampling request.")
    if count > min(n_a, n_b):
        raise ValueError(
            f"Cannot sample {count:,} unique cross pairs from partitions of sizes {n_a:,} and {n_b:,}."
        )
    return (
        rng.choice(n_a, size=count, replace=False).astype(np.int64),
        rng.choice(n_b, size=count, replace=False).astype(np.int64),
    )


def _sample_disjoint_lag_pairs(n: int, lag: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample vertex-disjoint (i, i+lag) pairs efficiently and exactly within the matching bound."""
    if lag < 1 or lag >= n:
        raise ValueError("lag must satisfy 1 <= lag < number of samples.")
    candidate_blocks: list[np.ndarray] = []
    max_pairs = 0
    for r in range(min(lag, n)):
        length = 1 + (n - 1 - r) // lag
        block_count = length // 2
        if block_count:
            # Path vertices are r, r+lag, ...; taking every other vertex gives
            # a maximum vertex-disjoint matching on that path.
            candidate_blocks.append(r + 2 * lag * np.arange(block_count, dtype=np.int64))
            max_pairs += block_count
    if count > max_pairs:
        raise ValueError(
            f"Cannot sample {count:,} disjoint lag-{lag} pairs from {n:,} samples; maximum is {max_pairs:,}."
        )
    candidates = np.concatenate(candidate_blocks) if candidate_blocks else np.empty(0, dtype=np.int64)
    rng.shuffle(candidates)
    chosen = candidates[:count]
    return chosen, chosen + lag


def hamming_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) != len(b):
        raise ValueError("Pair arrays must have equal length.")
    return np.count_nonzero(a != b, axis=1).astype(np.int16, copy=False)


def sample_within_distances(x: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    i, j = _sample_disjoint_pairs(len(x), count, rng)
    return hamming_distances(x[i], x[j])


def sample_cross_distances(a: np.ndarray, b: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    i, j = _sample_cross_pairs(len(a), len(b), count, rng)
    return hamming_distances(a[i], b[j])


def sample_lagged_distances(x: np.ndarray, lag: int, count: int, rng: np.random.Generator) -> np.ndarray:
    i, j = _sample_disjoint_lag_pairs(len(x), lag, count, rng)
    return hamming_distances(x[i], x[j])


def _merge_expected_bins(obs: np.ndarray, exp: np.ndarray, minimum: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    mo: list[float] = []
    me: list[float] = []
    ro = re = 0.0
    for o, e in zip(obs, exp):
        ro += float(o)
        re += float(e)
        if re >= minimum:
            mo.append(ro)
            me.append(re)
            ro = re = 0.0
    if re:
        if me:
            mo[-1] += ro
            me[-1] += re
        else:
            mo.append(ro)
            me.append(re)
    return np.asarray(mo), np.asarray(me)


def _chi_square_gof(distances: np.ndarray, pmf: np.ndarray, alpha: float) -> dict[str, Any]:
    from scipy.stats import chi2
    obs = np.bincount(distances.astype(np.int64), minlength=len(pmf)).astype(float)
    exp = pmf * len(distances)
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]
    obs, exp = _merge_expected_bins(obs, exp)
    if len(exp) < 2:
        raise ValueError("Reference distribution produced fewer than two usable chi-square bins.")
    stat = float(np.sum((obs - exp) ** 2 / exp))
    dof = len(exp) - 1
    p = float(chi2.sf(stat, dof))
    return {
        "statistic": stat,
        "degrees_of_freedom": dof,
        "p_value": p,
        "alpha": alpha,
        "reject": p < alpha,
        "minimum_expected_count": float(exp.min()),
        "merged_bins": len(exp),
    }


def tvd(distances: np.ndarray, pmf: np.ndarray) -> float:
    counts = np.bincount(distances.astype(np.int64), minlength=len(pmf)).astype(float)
    empirical = counts / len(distances)
    return float(0.5 * np.abs(empirical - pmf).sum())


def bootstrap_tvd(distances: np.ndarray, pmf: np.ndarray, rng: np.random.Generator, reps: int) -> dict[str, Any]:
    """Percentile bootstrap for the categorical TVD statistic.

    This is a pair-level Monte Carlo uncertainty approximation.  Pair
    endpoints are disjoint within each sampling replicate; the bootstrap is
    therefore an approximation rather than an exact finite-population CI.
    """
    if reps < 100:
        raise ValueError("bootstrap_replicates must be >= 100.")
    empirical_counts = np.bincount(distances.astype(np.int64), minlength=len(pmf)).astype(float)
    empirical = empirical_counts / len(distances)
    draws = rng.multinomial(len(distances), empirical, size=reps) / len(distances)
    values = 0.5 * np.abs(draws - pmf).sum(axis=1)
    lo = float(np.quantile(values, 0.025))
    hi = float(np.quantile(values, 0.975))
    return {
        "confidence_level": TVD_CONFIDENCE_LEVEL,
        "replicates": reps,
        "lower": lo,
        "upper": hi,
        "within_threshold": hi <= TVD_THRESHOLD,
        "interpretation": "percentile pair-level bootstrap approximation for observed TVD",
    }


def summarize_distances(
    distances: np.ndarray,
    pmf: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
    bootstrap_reps: int,
) -> dict[str, Any]:
    return {
        "sampled_pairs": int(len(distances)),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "min": int(np.min(distances)),
        "max": int(np.max(distances)),
        "tvd": tvd(distances, pmf),
        "tvd_threshold": TVD_THRESHOLD,
        "chi_square": _chi_square_gof(distances, pmf, alpha),
        "tvd_uncertainty": bootstrap_tvd(distances, pmf, rng, bootstrap_reps),
    }


def _fisher_combine(p_values: Sequence[float]) -> dict[str, Any]:
    from scipy.stats import chi2
    p = np.clip(np.asarray(p_values, dtype=float), np.finfo(float).tiny, 1.0)
    stat = float(-2.0 * np.sum(np.log(p)))
    dof = int(2 * len(p))
    combined = float(chi2.sf(stat, dof))
    return {"method": "Fisher combination across independent Monte Carlo replicates", "statistic": stat, "degrees_of_freedom": dof, "p_value": combined}


def _aggregate(results: Sequence[Mapping[str, Any]], *, alpha: float) -> dict[str, Any]:
    if not results:
        raise ValueError("At least one replicate is required.")
    combined = _fisher_combine([float(r["chi_square"]["p_value"]) for r in results])
    return {
        "replicate_count": len(results),
        "replicate_type": "independent Monte Carlo sampling replicates conditional on one audited dataset instance",
        "replicates": [dict(r) for r in results],
        "max_tvd": max(float(r["tvd"]) for r in results),
        "max_tvd_ci_upper": max(float(r["tvd_uncertainty"]["upper"]) for r in results),
        "combined_p_value": combined["p_value"],
        "combined_p_value_method": combined["method"],
        "practical_pass": all(bool(r["tvd_uncertainty"]["within_threshold"]) for r in results),
        "statistical_warning": combined["p_value"] < alpha,
        "alpha": alpha,
    }


def near_duplicate_summary(
    distance_replicates: Sequence[np.ndarray],
    radii: Sequence[int],
    reference_pmf: np.ndarray,
    bootstrap_replicates: int,
    rng: np.random.Generator,
    practical_relative_excess: float = 0.25,
    alpha: float = FAMILYWISE_ALPHA,
    independent_reference_pmf: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize near-duplicate rates against an explicit independent-row null."""
    from scipy.stats import binom
    if not distance_replicates:
        raise ValueError("At least one distance replicate is required.")
    distances = np.concatenate(distance_replicates)
    total = len(distances)
    pmf = validate_reference_pmf(independent_reference_pmf if independent_reference_pmf is not None else reference_pmf, len(reference_pmf) - 1)
    out: dict[str, Any] = {}
    for r in radii:
        if r < 0 or r >= len(pmf):
            raise ValueError("Near-duplicate radius outside feature space.")
        observed = int(np.count_nonzero(distances <= r))
        q = float(pmf[: r + 1].sum())
        expected = total * q
        observed_rate = observed / total
        excess_ratio = float(observed / expected) if expected > 0 else (1.0 if observed == 0 else float("inf"))
        p = float(binom.sf(observed - 1, total, q)) if q > 0 else (1.0 if observed == 0 else 0.0)
        replicate_rates = np.asarray([np.mean(d <= r) for d in distance_replicates], dtype=float)
        # Bootstrap the pooled Bernoulli event count rather than resampling
        # only the small number of audit replicates.  This uses the same
        # pair-level approximation as bootstrap_tvd and preserves the actual
        # number of observed pair trials.
        boot = rng.binomial(total, observed_rate, size=bootstrap_replicates) / total
        rate_lo, rate_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        practical = excess_ratio > (1.0 + practical_relative_excess)
        out[str(r)] = {
            "observed_pairs": observed,
            "sampled_pairs": total,
            "observed_rate": observed_rate,
            "null_probability": q,
            "expected_pairs": expected,
            "excess_ratio": excess_ratio,
            "practical_relative_excess_threshold": practical_relative_excess,
            "one_sided_excess_p_value": p,
            "bootstrap_rate_uncertainty": {"confidence_level": TVD_CONFIDENCE_LEVEL, "lower": rate_lo, "upper": rate_hi, "replicates": bootstrap_replicates, "unit": "audit-replicate rate"},
            "statistically_excessive": p < alpha,
            "practically_excessive": practical,
            "reference_basis": "empirical independent-row Hamming null preserving observed per-bit marginals",
        }
    return out


def analyze_structured_views(structured_views: Mapping[str, Mapping[str, Any]], alpha: float, practical_relative_excess: float = 0.25) -> dict[str, Any]:
    from scipy.stats import poisson
    results = {}
    for name, spec in structured_views.items():
        values = np.asarray(spec["values"]).reshape(-1)
        if len(values) < 2:
            raise ValueError(f"{name}: at least two values required.")
        domain = int(spec["domain_size"])
        if domain < 2:
            raise ValueError(f"{name}: domain_size must be >= 2.")
        _, counts = np.unique(values, return_counts=True)
        repeated = counts[counts > 1]
        observed = int(np.sum(repeated * (repeated - 1) // 2))
        expected = len(values) * (len(values) - 1) / (2 * domain)
        p = float(poisson.sf(observed - 1, expected)) if observed else 1.0
        ratio = float(observed / expected) if expected else (1.0 if observed == 0 else float("inf"))
        results[name] = {
            "description": str(spec.get("description", "")),
            "sample_count": len(values),
            "unique_count": int(len(np.unique(values))),
            "collision_pairs": observed,
            "maximum_multiplicity": int(repeated.max()) if len(repeated) else 1,
            "reference": {
                "model": "Poisson approximation to pair-collision count under independent uniform sampling",
                "domain_size": domain,
                "expected_collision_pairs": expected,
                "observed_collision_pairs": observed,
                "excess_ratio": ratio,
                "practical_relative_excess_threshold": practical_relative_excess,
                "one_sided_excess_p_value": p,
                "alpha": alpha,
                "statistically_excessive": p < alpha,
                "practically_excessive": ratio > (1.0 + practical_relative_excess),
            },
        }
    return results


def _pair_matrix(x: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    a = x[i].astype(np.float32)
    b = x[j].astype(np.float32)
    xor = np.abs(a - b)
    return np.concatenate([a, b, xor], axis=1)


def _roc_auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def _deranged_partner(j: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(j) < 2:
        raise ValueError("At least two partners are required for a negative pairing.")
    for _ in range(32):
        candidate = rng.permutation(j)
        if not np.any(candidate == j):
            return candidate
    # Deterministic cyclic shift is a guaranteed derangement for length >= 2.
    return np.roll(j, 1)


def _auc_permutation_normal_p_value(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Normal approximation to the fixed-score label-permutation null for AUC.

    With fixed scores and a fixed number of positive labels, AUC is a scaled
    Wilcoxon rank-sum statistic.  The mean is 0.5.  The variance below includes
    the standard tie correction, making the calculation conditional on the
    observed score ranks and label count.
    """
    from scipy.stats import norm, rankdata
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)
    n1 = int(labels.sum())
    n = len(labels)
    n0 = n - n1
    if n1 < 1 or n0 < 1:
        raise ValueError("AUC permutation null requires both classes.")
    ranks = rankdata(scores, method="average")
    rsum = float(ranks[labels == 1].sum())
    u = rsum - n1 * (n1 + 1) / 2.0
    auc = u / (n1 * n0)
    _, tie_counts = np.unique(scores, return_counts=True)
    tie_term = float(np.sum(tie_counts**3 - tie_counts))
    variance_u = n1 * n0 / 12.0 * (n + 1.0 - tie_term / (n * (n - 1.0)))
    sd_auc = math.sqrt(variance_u) / (n1 * n0)
    if sd_auc == 0:
        p = 1.0 if math.isclose(auc, 0.5) else 0.0
        z = 0.0 if p == 1.0 else math.copysign(float("inf"), auc - 0.5)
    else:
        z = (auc - 0.5) / sd_auc
        p = float(2.0 * norm.sf(abs(z)))
    return {"auc": float(auc), "null_mean": 0.5, "null_sd": sd_auc, "z": float(z), "p_value": p, "method": "tie-corrected normal approximation to fixed-score label-permutation AUC null"}

def multivariate_pair_discrimination(
    x: np.ndarray,
    *,
    pairs: int,
    permutations: int,
    rng: np.random.Generator,
    c: float = DEFAULT_DETECTOR_C,
    alpha: float = FAMILYWISE_ALPHA,
    lag: int = 1,
) -> dict[str, Any]:
    """Detect multivariate generation-order dependence at a specified lag.

    The detector is trained on one set of disjoint lag pairs.  It is evaluated
    on a separate set of disjoint lag pairs against a negative set created by
    deranging the second endpoints.  The randomization test then regenerates
    the negative partner assignment on the untouched test endpoints and
    recomputes the held-out AUC.  This avoids treating the two examples derived
    from one endpoint pair as exchangeable labels after the classifier has
    already seen that pairing.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    if pairs < 4 or permutations < 1:
        raise ValueError("pairs must be >= 4 and permutations >= 1.")
    # Generate 2*pairs natural lag pairs, with no endpoint reuse, then split
    # the pair groups into detector-training and held-out groups.
    i, j = _sample_disjoint_lag_pairs(len(x), lag, 2 * pairs, rng)
    pair_ids = np.arange(2 * pairs, dtype=np.int64)
    train_groups, test_groups = train_test_split(
        pair_ids, test_size=0.30, random_state=int(rng.integers(0, 2**31 - 1))
    )
    train_i, train_j = i[train_groups], j[train_groups]
    test_i, test_j = i[test_groups], j[test_groups]

    train_neg_j = _deranged_partner(train_j, rng)
    test_neg_j = _deranged_partner(test_j, rng)
    X_train = np.vstack([_pair_matrix(x, train_i, train_j), _pair_matrix(x, train_i, train_neg_j)])
    y_train = np.concatenate([np.ones(len(train_groups), dtype=np.int8), np.zeros(len(train_groups), dtype=np.int8)])
    model = LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=int(rng.integers(0, 2**31 - 1)))
    model.fit(X_train, y_train)

    def score_for_partner(partners: np.ndarray) -> tuple[float, np.ndarray]:
        positive = _pair_matrix(x, test_i, test_j)
        negative = _pair_matrix(x, test_i, partners)
        X_test = np.vstack([positive, negative])
        y_test = np.concatenate([np.ones(len(test_groups), dtype=np.int8), np.zeros(len(test_groups), dtype=np.int8)])
        scores = model.predict_proba(X_test)[:, 1]
        return _roc_auc_from_scores(scores, y_test), scores

    observed_auc, observed_scores = score_for_partner(test_neg_j)
    observed_excess = abs(observed_auc - 0.5)

    null = np.empty(permutations, dtype=float)
    # The null randomizes the partner assignment on the held-out endpoints,
    # while the trained detector remains fixed and the test set is untouched.
    for k in range(permutations):
        null_partner = _deranged_partner(test_j, rng)
        # Randomly designate which of the two partnerings is treated as the
        # observed/natural side.  This creates the conditional null directly.
        if rng.integers(0, 2):
            positive_partner, negative_partner = null_partner, test_j
        else:
            positive_partner, negative_partner = test_j, null_partner
        positive = _pair_matrix(x, test_i, positive_partner)
        negative = _pair_matrix(x, test_i, negative_partner)
        X_test = np.vstack([positive, negative])
        y_test = np.concatenate([np.ones(len(test_groups), dtype=np.int8), np.zeros(len(test_groups), dtype=np.int8)])
        scores = model.predict_proba(X_test)[:, 1]
        null[k] = _roc_auc_from_scores(scores, y_test)
    null_excess = np.abs(null - 0.5)
    mc_p = float((1 + np.count_nonzero(null_excess >= observed_excess)) / (permutations + 1))
    analytic = _auc_permutation_normal_p_value(observed_scores, np.concatenate([np.ones(len(test_groups), dtype=np.int8), np.zeros(len(test_groups), dtype=np.int8)]))
    return {
        "pairs": int(2 * pairs),
        "lag": lag,
        "train_pair_groups": int(len(train_groups)),
        "test_pair_groups": int(len(test_groups)),
        "detector": "logistic_regression_on_[x_i, x_i+lag, abs(x_i-x_i+lag)]",
        "detector_C": c,
        "auc": observed_auc,
        "auc_excess_over_chance": observed_excess,
        "practical_tolerance": MULTIVARIATE_AUC_TOLERANCE,
        "permutations": permutations,
        "permutation_null": "held-out partner-assignment randomization with fixed detector; natural and randomized partnerings are reconstituted on untouched test endpoints",
        "null_auc_mean": float(null.mean()),
        "null_auc_95_upper": float(np.quantile(null, 0.95)),
        "null_excess_95_upper": float(np.quantile(null_excess, 0.95)),
        "permutation_p_value": mc_p,
        "permutation_p_value_resolution": 1.0 / (permutations + 1),
        "analytic_permutation_p_value": analytic["p_value"],
        "analytic_permutation_null_sd": analytic["null_sd"],
        "analytic_permutation_z": analytic["z"],
        "analytic_permutation_method": analytic["method"],
        "statistically_excessive": analytic["p_value"] < alpha,
        "practically_excessive": observed_excess > MULTIVARIATE_AUC_TOLERANCE,
    }


def inject_duplicates(x: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1].")
    y = np.array(x, copy=True)
    count = int(round(len(y) * fraction))
    if count:
        dst = rng.choice(len(y), size=count, replace=False)
        src = rng.integers(0, len(y), size=count, dtype=np.int64)
        same = src == dst
        if np.any(same):
            src[same] = (src[same] + 1) % len(y)
        y[dst] = y[src]
    return y


def inject_lag_copy(x: np.ndarray, fraction: float, lag: int, rng: np.random.Generator) -> np.ndarray:
    if not 0 <= fraction <= 1 or lag < 1 or lag >= len(x):
        raise ValueError("Invalid lag-copy parameters.")
    y = np.array(x, copy=True)
    count = int(round((len(y) - lag) * fraction))
    if count:
        starts = rng.choice(np.arange(lag, len(y)), size=count, replace=False)
        y[starts] = y[starts - lag]
    return y


def calibrate_fault(*, clean_features: np.ndarray, injector: Callable[[np.ndarray, np.random.Generator], np.ndarray], detector: Callable[[np.ndarray, np.random.Generator], bool], replicates: int, seed: int) -> dict[str, Any]:
    if replicates < CALIBRATION_MIN_REPLICATES:
        raise ValueError(f"Calibration requires at least {CALIBRATION_MIN_REPLICATES} replicates.")
    ss = np.random.SeedSequence(seed)
    detected = []
    for child in ss.spawn(replicates):
        rng = np.random.default_rng(child)
        detected.append(bool(detector(injector(clean_features, rng), rng)))
    rate = float(np.mean(detected))
    return {"replicates": replicates, "detections": int(sum(detected)), "detection_rate": rate, "target": CALIBRATION_DETECTION_TARGET, "target_met": rate >= CALIBRATION_DETECTION_TARGET}


def _count_hypotheses(*, comparisons: int, lag_tests: int, near_radii: int, structured: int, multivariate: int) -> int:
    # One p-value per comparison after replicate aggregation; replicates are
    # not counted as separate hypotheses.
    return max(1, comparisons + lag_tests + comparisons * near_radii + structured + multivariate)


def run_d2(
    *,
    partitions: Mapping[str, np.ndarray],
    reference_pmf: np.ndarray,
    feature_bits: int,
    structured_views: Mapping[str, Mapping[str, Any]] | None = None,
    pairs_per_test: int = DEFAULT_PAIRS_PER_TEST,
    audit_replicates: int = DEFAULT_AUDIT_REPLICATES,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    lags: Sequence[int] = DEFAULT_LAGS,
    near_duplicate_radii: Sequence[int] = NEAR_DUPLICATE_RADII,
    multivariate_partitions: Sequence[str] | None = None,
    multivariate_pairs: int = DEFAULT_MULTIVARIATE_PAIRS,
    multivariate_permutations: int = DEFAULT_MULTIVARIATE_PERMUTATIONS,
    audit_seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not partitions or pairs_per_test < 1 or audit_replicates < 1 or bootstrap_replicates < 100:
        raise ValueError("Invalid D2 configuration.")
    nominal_pmf = validate_reference_pmf(reference_pmf, feature_bits)
    arrays = {name: _binary(x, name, feature_bits) for name, x in partitions.items()}
    structured_views = structured_views or {}
    lags = tuple(int(k) for k in lags)
    near_duplicate_radii = tuple(int(r) for r in near_duplicate_radii)
    if any(k < 1 for k in lags):
        raise ValueError("All lags must be positive.")
    if any(r < 0 or r >= feature_bits for r in near_duplicate_radii):
        raise ValueError("Near-duplicate radii must lie in [0, feature_bits-1].")

    comparisons = [(f"within:{n}", x, x) for n, x in arrays.items()]
    names = list(arrays)
    comparisons += [(f"cross:{names[i]}:{names[j]}", arrays[names[i]], arrays[names[j]]) for i in range(len(names)) for j in range(i + 1, len(names))]
    lag_tests = [(f"{n}:lag{k}", x, k) for n, x in arrays.items() for k in lags if k < len(x)]
    selected_mv = tuple(multivariate_partitions or arrays.keys())
    for name in selected_mv:
        if name not in arrays:
            raise ValueError(f"Unknown multivariate partition: {name}")
        if 2 * multivariate_pairs > len(arrays[name]):
            raise ValueError(f"Multivariate pair count exceeds half the size of partition {name!r}.")

    total_tests = _count_hypotheses(
        comparisons=len(comparisons),
        lag_tests=len(lag_tests),
        near_radii=len(near_duplicate_radii),
        structured=len(structured_views),
        multivariate=len(selected_mv),
    )
    alpha = FAMILYWISE_ALPHA / total_tests
    ss = np.random.SeedSequence(audit_seed)

    hamming: dict[str, Any] = {}
    near_dups: dict[str, Any] = {}
    for name, a, b in comparisons:
        # Primary Hamming reference preserves the empirical bit marginals.
        empirical_null = independent_hamming_reference(a, b)
        reps = []
        distance_replicates = []
        for child in ss.spawn(audit_replicates):
            rng = np.random.default_rng(child)
            d = sample_within_distances(a, pairs_per_test, rng) if a is b else sample_cross_distances(a, b, pairs_per_test, rng)
            distance_replicates.append(d)
            reps.append(summarize_distances(d, empirical_null, rng, alpha, bootstrap_replicates))
        hamming[name] = _aggregate(reps, alpha=alpha)
        hamming[name]["reference_basis"] = "empirical independent-row Hamming null preserving observed per-bit marginals"
        hamming[name]["nominal_gohr_reference_tvd"] = [tvd(d, nominal_pmf) for d in distance_replicates]
        near_dups[name] = near_duplicate_summary(
            distance_replicates,
            near_duplicate_radii,
            nominal_pmf,
            bootstrap_replicates,
            np.random.default_rng(ss.spawn(1)[0]),
            independent_reference_pmf=empirical_null,
            alpha=alpha,
        )

    lagged: dict[str, Any] = {}
    for name, x, lag in lag_tests:
        empirical_null = independent_hamming_reference(x, x)
        reps = []
        for child in ss.spawn(audit_replicates):
            rng = np.random.default_rng(child)
            reps.append(summarize_distances(sample_lagged_distances(x, lag, pairs_per_test, rng), empirical_null, rng, alpha, bootstrap_replicates))
        lagged[name] = _aggregate(reps, alpha=alpha)
        lagged[name]["lag"] = lag
        lagged[name]["reference_basis"] = "empirical independent-row Hamming null preserving observed per-bit marginals"
        lagged[name]["nominal_gohr_reference"] = "Binomial(64, 0.5) retained as a case-study diagnostic only"

    structured = analyze_structured_views(structured_views, alpha)

    mv: dict[str, Any] = {}
    for name in selected_mv:
        rng = np.random.default_rng(ss.spawn(1)[0])
        mv[name] = multivariate_pair_discrimination(
            arrays[name], pairs=multivariate_pairs, permutations=multivariate_permutations, rng=rng, alpha=alpha, lag=1
        )

    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for name, result in {**hamming, **lagged}.items():
        if not result["practical_pass"]:
            failures.append({"component": name, "type": "distributional", "reason": "Upper bootstrap TVD uncertainty bound exceeded the pre-specified practical tolerance.", "max_tvd_ci_upper": result["max_tvd_ci_upper"]})
        elif result["statistical_warning"]:
            warnings.append({"component": name, "type": "statistical", "reason": "Replicate-level evidence rejected the empirical independent-row reference after multiplicity correction while remaining practically below the TVD tolerance.", "combined_p_value": result["combined_p_value"]})

    for comparison, radii_results in near_dups.items():
        for radius, result in radii_results.items():
            if result["practically_excessive"]:
                failures.append({"component": f"{comparison}:radius{radius}", "type": "near_duplicate_structure", "reason": "Observed near-duplicate rate exceeded the pre-specified practical excess threshold relative to the empirical independent-row null.", "excess_ratio": result["excess_ratio"]})
            elif result["statistically_excessive"]:
                warnings.append({"component": f"{comparison}:radius{radius}", "type": "near_duplicate_structure", "reason": "Near-duplicate rate was statistically excessive under the empirical independent-row null but not practically excessive.", "p_value": result["one_sided_excess_p_value"]})

    for name, result in structured.items():
        ref = result["reference"]
        if ref["practically_excessive"]:
            failures.append({"component": name, "type": "structured_collision", "reason": "Structured-value collision excess exceeded the practical tolerance."})
        elif ref["statistically_excessive"]:
            warnings.append({"component": name, "type": "structured_collision", "reason": "Structured-value collision count was statistically excessive but not practically excessive.", "p_value": ref["one_sided_excess_p_value"]})

    for name, result in mv.items():
        if result["practically_excessive"]:
            failures.append({"component": name, "type": "multivariate_dependence", "reason": "Lagged pairs were distinguishable from independently permuted partners beyond the practical AUC tolerance.", "auc": result["auc"], "auc_excess_over_chance": result["auc_excess_over_chance"]})
        elif result["statistically_excessive"]:
            warnings.append({"component": name, "type": "multivariate_dependence", "reason": "The permutation test detected statistical evidence of lagged pair dependence without a practically material AUC effect.", "p_value": result["permutation_p_value"]})

    outcome = "FAIL" if failures else ("CONDITIONAL_PASS" if warnings else "PASS")
    results = {
        "d2_1_near_duplicate_structure": near_dups,
        "d2_2_serial_dependence": lagged,
        "d2_2_pairwise_structure": hamming,
        "d2_3_multivariate_dependence": mv,
        "d2_4_structured_repetition": structured,
        "configuration": {
            "schema_version": SCHEMA_VERSION,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "effective_test_alpha": alpha,
            "familywise_test_count": total_tests,
            "pairs_per_test": pairs_per_test,
            "audit_replicates": audit_replicates,
            "bootstrap_replicates": bootstrap_replicates,
            "lags": list(lags),
            "near_duplicate_radii": list(near_duplicate_radii),
            "multivariate_pairs": multivariate_pairs,
            "multivariate_permutations": multivariate_permutations,
            "multivariate_auc_tolerance": MULTIVARIATE_AUC_TOLERANCE,
            "tvd_threshold": TVD_THRESHOLD,
            "audit_seed": audit_seed,
            "pair_sampling": "within and lagged pairs use disjoint endpoints within each replicate; cross-partition pairs sample without replacement within each partition",
            "replication_semantics": "replicates are Monte Carlo sampling replicates from one fixed audited dataset instance, not independent dataset generations",
            "confirmatory_hamming_null": "per-bit-marginal Poisson-binomial Hamming null under independent rows; mismatch-indicator independence across bit positions is an explicit model assumption",
            "nominal_case_study_hamming_null": "adapter-supplied Binomial(64, 0.5) diagnostic; not used as the universal independence null",
            "multivariate_positive_pairs": "generation-order lag-1 pairs",
            "multivariate_statistical_null": "fixed-score test-label permutation null; analytic tie-corrected normal approximation is primary because finite Monte Carlo permutation resolution can be coarser than the Bonferroni-adjusted alpha",
            "hypothesis_counting": "one hypothesis per comparison after replicate-level p-value aggregation; replicate observations are not counted as separate familywise hypotheses",
        },
    }
    decision = {
        "outcome": outcome,
        "failures": failures,
        "warnings": warnings,
        "interpretation": "No practically material dependence detected within the tested classes, declared null models, practical thresholds, and measured sensitivity envelope." if outcome == "PASS" else "The decision is limited to the explicitly tested dependence classes, null models, practical thresholds, and measured sensitivity envelope.",
        "not_proof_of_independence": True,
    }
    return results, decision


def build_d2_certificate(*, results: Mapping[str, Any], decision: Mapping[str, Any], partitions: Mapping[str, np.ndarray], dataset_id: str, dataset_version: str | None, generation_procedure: str | None, generation_parameters: Mapping[str, Any] | None, generation_random_seed: int | None, reference_description: str, reference_model_description: str, audit_seed: int, output_path: str) -> dict[str, Any]:
    provenance = build_provenance(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        generation_procedure=generation_procedure,
        generation_parameters=generation_parameters,
        random_seed=generation_random_seed,
        partitions={name: {"sample_count": int(x.shape[0]), "feature_count": int(x.shape[1]), "dtype": str(x.dtype), "shape": list(x.shape), "sha256": array_sha256(x)} for name, x in partitions.items()},
        audit_configuration={**dict(results["configuration"]), "reference_distribution": reference_description, "reference_model": reference_model_description},
    )
    certificate = make_certificate(
        audit_id=AUDIT_ID,
        audit_name=AUDIT_NAME,
        claim="The supplied dataset instance was audited for detectable near-duplicate structure, generation-order dependence, pairwise Hamming structure, multivariate lag dependence, and adapter-exposed structured repetition.",
        outcome=str(decision["outcome"]),
        findings={"results": dict(results), "decision": dict(decision)},
        methodology={
            "scope": "Dataset Integrity D2 Sample Dependence Audit",
            "d1_boundary": "D1 owns exact row duplication and exact partition overlap; D2 does not repeat those exact-equality claims.",
            "d2_1": "Near-duplicate rates are evaluated at pre-specified Hamming radii against the declared per-bit-marginal independent-row null.",
            "d2_2": "Generation-order lagged Hamming distributions are tested against the declared per-bit-marginal independent-row null. Within/cross pairwise comparisons are retained as feature-space controls; the lag tests carry the primary generation-order dependence interpretation.",
            "d2_3": "A fixed logistic detector distinguishes natural lag-1 pairs from independently permuted partners on untouched endpoints. The primary p-value uses the tie-corrected normal approximation to the fixed-score test-label permutation null; finite Monte Carlo permutations are retained as a direct randomization diagnostic and resolution check.",
            "d2_4": "Adapter-supplied semantic views are evaluated for excess finite-domain collisions under the stated independent-uniform reference model.",
            "d2_5": "Controlled duplicate and lag-copy injectors are available for empirical sensitivity calibration; calibration results are recorded separately and are not clean-data findings.",
            "multiple_comparison_control": "Bonferroni familywise control at 0.01 across hypothesis-level tests after replicate-level aggregation; audit replicates are not themselves counted as separate hypotheses.",
            "replication": "Repeated sampling replicates quantify Monte Carlo variability conditional on the same audited dataset instance; they are not independent dataset generations.",
            "decision_semantics": "PASS is evidence against the tested dependence hypotheses within the declared scope, not proof of universal independence.",
            "nominal_reference": reference_description,
            "confirmatory_reference": "Empirical independent-row Poisson-binomial Hamming null preserving observed per-bit marginals.",
            "reference_model": reference_model_description,
            "audit_seed": audit_seed,
        },
        provenance=provenance,
        limitations=[
            "Finite statistical testing cannot establish universal mutual independence.",
            "The Hamming null conditions on observed per-bit marginals and assumes row independence plus independence of mismatch indicators across bit positions; it is therefore a model-based diagnostic, not a universal independence null.",
            "Pearson chi-square p-values are model-based diagnostics; the primary practical criterion also requires the pre-specified TVD tolerance and bootstrap uncertainty.",
            "The pair-level bootstrap is an approximation because disjoint endpoint sampling creates finite-population dependence among pair statistics.",
            "Near-duplicate and structured-collision p-values rely on their stated finite-domain or empirical null approximations.",
            "Multivariate discrimination is specifically a lag-1 detector and therefore has sensitivity only to dependence represented by that detector and feature construction. The finite Monte Carlo permutation p-value is diagnostic; the primary hypothesis p-value uses the stated fixed-score permutation-null normal approximation so that familywise alpha is not below the Monte Carlo resolution.",
            "Audit replicates reuse the same dataset instance; they quantify sampling variability rather than dataset-to-dataset variability.",
            "Calibration must be run and reported before claiming a corresponding detection sensitivity level.",
            "The nominal Binomial(64, 0.5) Gohr reference is retained as a diagnostic and is not treated as a universal independence theorem.",
        ],
        evidence_level="DATASET_INTEGRITY_D2_SAMPLE_DEPENDENCE",
        certificate_version=SCHEMA_VERSION,
    )
    write_certificate(certificate, output_path)
    return certificate


def print_report(results: Mapping[str, Any], certificate: Mapping[str, Any]) -> None:
    print("=" * 78)
    print("Dataset Integrity — D2 Sample Dependence Audit")
    print("=" * 78)
    print(f"Pairwise comparisons : {len(results['d2_2_pairwise_structure'])}")
    print(f"Lagged comparisons   : {len(results['d2_2_serial_dependence'])}")
    print(f"Multivariate tests   : {len(results['d2_3_multivariate_dependence'])}")
    print(f"Structured views     : {len(results['d2_4_structured_repetition'])}")
    print(f"Near-duplicate views : {len(results['d2_1_near_duplicate_structure'])}")
    print(f"Outcome              : {certificate['decision']['outcome']}")
    print(f"FWER alpha           : {results['configuration']['familywise_alpha']}")
    print(f"Effective test alpha : {results['configuration']['effective_test_alpha']:.3e}")
    print(f"Familywise tests     : {results['configuration']['familywise_test_count']}")
    print("=" * 78)
