"""
D2 — Sample Dependence Audit.

Generic Dataset Integrity implementation.

D2 is intentionally distinct from D1. D1 owns exact row duplication and
exact partition overlap. D2 evaluates dependence structures that can remain
when exact equality tests pass.

Layers:
    D2.1  Near-duplicate structure.
    D2.2  Pairwise and serial/lagged dependence.
    D2.3  Multivariate pair dependence using a fixed classifier and an
          exact test-set label-permutation null.
    D2.4  Adapter-supplied structured/semantic repetition.
    D2.5  Controlled fault-injection sensitivity calibration.

The implementation is evidence-bounded: PASS is not a proof of mutual
independence. It means that no practically material violation was detected
within the declared null models, tests, thresholds, and measured sensitivity.
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


def empirical_independent_hamming_pmf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Construct the independent-row Hamming null from observed bit marginals.

    For each bit position k, let p_a[k] and p_b[k] be the observed marginal
    probabilities of one in the two endpoint populations. Under independent
    rows, the mismatch indicator at bit k has probability

        q_k = p_a(1-p_b) + (1-p_a)p_b.

    The Hamming distance is therefore modeled as a Poisson-binomial sum of
    independent, non-identically distributed Bernoulli(q_k) variables.

    This is the confirmatory D2 null. The Binomial(64, 0.5) reference remains
    available separately as a case-study diagnostic only.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("a and b must be 2-D arrays with the same feature count.")
    if len(a) < 1 or len(b) < 1:
        raise ValueError("a and b must each contain at least one sample.")
    pa = a.mean(axis=0)
    pb = b.mean(axis=0)
    q = pa * (1.0 - pb) + (1.0 - pa) * pb
    pmf = np.array([1.0], dtype=np.float64)
    for prob in q:
        pmf = np.convolve(pmf, np.array([1.0 - prob, prob], dtype=np.float64))
    pmf = np.maximum(pmf, 0.0)
    return pmf / pmf.sum()


def fisher_combine_pvalues(p_values: Sequence[float]) -> float:
    """Combine independent replicate p-values with Fisher's method."""
    from scipy.stats import chi2

    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("At least one p-value is required.")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and lie in [0, 1].")
    clipped = np.clip(p, np.finfo(float).tiny, 1.0)
    statistic = float(-2.0 * np.sum(np.log(clipped)))
    return float(chi2.sf(statistic, 2 * len(clipped)))


def _sample_disjoint_pairs(n: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample pairs with no repeated endpoint within one replicate."""
    if n < 2 or count < 1:
        raise ValueError("Need n >= 2 and count >= 1.")
    if 2 * count > n:
        raise ValueError(f"Cannot sample {count:,} disjoint pairs from {n:,} samples.")
    indices = rng.choice(n, size=2 * count, replace=False)
    return indices[:count].astype(np.int64), indices[count:].astype(np.int64)


def _sample_cross_pairs(n_a: int, n_b: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample cross-partition pairs without reusing an index within a replicate."""
    if n_a < 1 or n_b < 1 or count < 1:
        raise ValueError("Invalid cross-partition sampling request.")
    if count > min(n_a, n_b):
        raise ValueError(f"Cannot sample {count:,} unique cross pairs from partitions of sizes {n_a:,} and {n_b:,}.")
    return (
        rng.choice(n_a, size=count, replace=False).astype(np.int64),
        rng.choice(n_b, size=count, replace=False).astype(np.int64),
    )


def _sample_disjoint_lag_pairs(n: int, lag: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample lagged pairs with no repeated endpoint within one replicate."""
    if lag < 1 or lag >= n:
        raise ValueError("lag must satisfy 1 <= lag < number of samples.")
    starts = rng.permutation(n - lag)
    chosen: list[int] = []
    used = np.zeros(n, dtype=np.bool_)
    for start in starts:
        end = int(start + lag)
        if not used[start] and not used[end]:
            chosen.append(int(start))
            used[start] = True
            used[end] = True
            if len(chosen) == count:
                break
    if len(chosen) < count:
        raise ValueError(f"Could not construct {count:,} disjoint lag-{lag} pairs from {n:,} samples.")
    starts = np.asarray(chosen, dtype=np.int64)
    return starts, starts + lag


def hamming_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def _chi_square_gof(distances: np.ndarray, pmf: np.ndarray, alpha: float) -> dict[str, Any]:
    from scipy.stats import chi2

    obs = np.bincount(distances.astype(np.int64), minlength=len(pmf)).astype(float)
    exp = pmf * len(distances)
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]

    # Pearson's approximation requires adequate expected counts. Merge
    # adjacent categories until every reported bin has expected count >= 5.
    mo, me, ro, re = [], [], 0.0, 0.0
    for o, e in zip(obs, exp):
        ro += o
        re += e
        if re >= 5:
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

    if len(me) < 2:
        raise ValueError("Reference distribution produced fewer than two usable chi-square bins.")
    stat = float(np.sum((np.asarray(mo) - np.asarray(me)) ** 2 / np.asarray(me)))
    dof = len(me) - 1
    p = float(chi2.sf(stat, dof))
    return {
        "statistic": stat,
        "degrees_of_freedom": dof,
        "p_value": p,
        "alpha": alpha,
        "reject": p < alpha,
        "minimum_expected_count": float(min(me)),
        "merged_bins": len(me),
    }


def tvd(distances: np.ndarray, pmf: np.ndarray) -> float:
    counts = np.bincount(distances.astype(np.int64), minlength=len(pmf)).astype(float)
    empirical = counts / len(distances)
    return float(0.5 * np.abs(empirical - pmf).sum())


def bootstrap_tvd(distances: np.ndarray, pmf: np.ndarray, rng: np.random.Generator, reps: int) -> dict[str, Any]:
    """Percentile bootstrap uncertainty for the empirical TVD statistic."""
    empirical_counts = np.bincount(distances.astype(np.int64), minlength=len(pmf)).astype(float)
    empirical = empirical_counts / len(distances)
    draws = rng.multinomial(len(distances), empirical, size=reps) / len(distances)
    values = 0.5 * np.abs(draws - pmf).sum(axis=1)
    lo = float(np.quantile(values, (1 - TVD_CONFIDENCE_LEVEL) / 2))
    hi = float(np.quantile(values, 1 - (1 - TVD_CONFIDENCE_LEVEL) / 2))
    return {
        "confidence_level": TVD_CONFIDENCE_LEVEL,
        "replicates": reps,
        "lower": lo,
        "upper": hi,
        "within_threshold": hi <= TVD_THRESHOLD,
        "interpretation": "percentile bootstrap uncertainty interval for the observed TVD statistic",
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


def _aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("At least one replicate is required.")
    p_values = [float(r["chi_square"]["p_value"]) for r in results]
    combined_p = fisher_combine_pvalues(p_values)
    return {
        "replicate_count": len(results),
        "replicate_type": "independent Monte Carlo sampling replicates conditional on one audited dataset instance",
        "replicates": [dict(r) for r in results],
        "max_tvd": max(float(r["tvd"]) for r in results),
        "max_tvd_ci_upper": max(float(r["tvd_uncertainty"]["upper"]) for r in results),
        "min_p_value": min(p_values),
        "combined_p_value": combined_p,
        "combined_p_value_method": "Fisher combination across independent Monte Carlo sampling replicates",
        "practical_pass": all(bool(r["tvd_uncertainty"]["within_threshold"]) for r in results),
        "statistical_warning": combined_p < float(results[0]["chi_square"]["alpha"]),
    }


def near_duplicate_summary(
    distance_replicates: Sequence[np.ndarray],
    radii: Sequence[int],
    reference_pmf: np.ndarray,
    bootstrap_replicates: int,
    rng: np.random.Generator,
    practical_relative_excess: float = 0.25,
    alpha: float = FAMILYWISE_ALPHA,
) -> dict[str, Any]:
    """Summarize near-duplicate rates, preserving replicate structure."""
    from scipy.stats import binom

    if not distance_replicates:
        raise ValueError("At least one distance replicate is required.")
    distances = np.concatenate(distance_replicates)
    total = len(distances)
    out: dict[str, Any] = {}

    for r in radii:
        if r < 0 or r >= len(reference_pmf):
            raise ValueError("Near-duplicate radius outside feature space.")
        observed = int(np.count_nonzero(distances <= r))
        q = float(reference_pmf[: r + 1].sum())
        expected = total * q
        observed_rate = observed / total
        excess_ratio = float(observed / expected) if expected > 0 else (1.0 if observed == 0 else float("inf"))

        # Model-based one-sided excess diagnostic. It is reported transparently
        # but the primary practical decision is based on the observed excess.
        p = float(binom.sf(observed - 1, total, q)) if q > 0 else (1.0 if observed == 0 else 0.0)

        # Bootstrap the empirical near-duplicate rate. This quantifies sampling
        # uncertainty without treating the repeated audit replicates as new data.
        bootstrap_rates = np.empty(bootstrap_replicates, dtype=float)
        replicate_rates = np.asarray([np.mean(d <= r) for d in distance_replicates], dtype=float)
        for k in range(bootstrap_replicates):
            resampled = rng.choice(replicate_rates, size=len(replicate_rates), replace=True)
            bootstrap_rates[k] = float(np.mean(resampled))
        rate_lo = float(np.quantile(bootstrap_rates, (1 - TVD_CONFIDENCE_LEVEL) / 2))
        rate_hi = float(np.quantile(bootstrap_rates, 1 - (1 - TVD_CONFIDENCE_LEVEL) / 2))
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
            "bootstrap_rate_uncertainty": {
                "confidence_level": TVD_CONFIDENCE_LEVEL,
                "lower": rate_lo,
                "upper": rate_hi,
                "replicates": bootstrap_replicates,
                "unit": "audit replicate rate",
            },
            "statistically_excessive": p < alpha,
            "practically_excessive": practical,
        }
    return out


def analyze_structured_views(
    structured_views: Mapping[str, Mapping[str, Any]],
    alpha: float,
    practical_relative_excess: float = 0.25,
) -> dict[str, Any]:
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


def multivariate_pair_discrimination(
    x: np.ndarray,
    *,
    pairs: int,
    permutations: int,
    rng: np.random.Generator,
    c: float = DEFAULT_DETECTOR_C,
    alpha: float = FAMILYWISE_ALPHA,
) -> dict[str, Any]:
    """Test whether real pairs are distinguishable from permuted partners.

    Positive and negative pair construction uses the same endpoints. The
    classifier is fit once. The null distribution is generated by permuting
    labels on the untouched test set, which is an exact randomization test
    conditional on the fitted predictor and test scores. This avoids the
    invalid practice of refitting the detector against a null training set
    while leaving test labels fixed.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    if pairs < 2 or permutations < 1:
        raise ValueError("pairs must be >= 2 and permutations >= 1.")
    i, j = _sample_disjoint_pairs(len(x), pairs, rng)
    negative_j = rng.permutation(j)
    bad = negative_j == i
    if np.any(bad):
        # A cyclic shift within the negative partner set avoids self-pairs
        # without changing endpoint multiplicities.
        negative_j[bad] = np.roll(negative_j, 1)[bad]
        if np.any(negative_j == i):
            raise RuntimeError("Unable to construct self-pair-free negative controls.")

    positive = _pair_matrix(x, i, j)
    negative = _pair_matrix(x, i, negative_j)
    X = np.vstack([positive, negative])
    y = np.concatenate([np.ones(pairs, dtype=np.int8), np.zeros(pairs, dtype=np.int8)])

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=y,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    model = LogisticRegression(
        C=c,
        solver="liblinear",
        max_iter=2000,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    model.fit(X[train_idx], y[train_idx])
    test_scores = model.predict_proba(X[test_idx])[:, 1]
    y_test = y[test_idx]
    auc = _roc_auc_from_scores(test_scores, y_test)
    observed_excess = abs(auc - 0.5)

    null = np.empty(permutations, dtype=float)
    for k in range(permutations):
        permuted_labels = rng.permutation(y_test)
        null[k] = _roc_auc_from_scores(test_scores, permuted_labels)
    null_excess = np.abs(null - 0.5)
    p = float((1 + np.count_nonzero(null_excess >= observed_excess)) / (permutations + 1))

    return {
        "pairs": pairs,
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "detector": "logistic_regression_on_[x_i, x_j, xor(x_i,x_j)]",
        "detector_C": c,
        "auc": auc,
        "auc_excess_over_chance": observed_excess,
        "practical_tolerance": MULTIVARIATE_AUC_TOLERANCE,
        "permutations": permutations,
        "permutation_null": "test-set label permutation conditional on fixed fitted detector and test scores",
        "null_auc_mean": float(null.mean()),
        "null_auc_95_upper": float(np.quantile(null, 0.95)),
        "null_excess_95_upper": float(np.quantile(null_excess, 0.95)),
        "permutation_p_value": p,
        "statistically_excessive": p < alpha,
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


def calibrate_fault(
    *,
    clean_features: np.ndarray,
    injector: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    detector: Callable[[np.ndarray, np.random.Generator], bool | Mapping[str, bool]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate empirical fault-detection sensitivity over independent trials."""
    if replicates < CALIBRATION_MIN_REPLICATES:
        raise ValueError(f"Calibration requires at least {CALIBRATION_MIN_REPLICATES} replicates.")
    ss = np.random.SeedSequence(seed)
    outcomes: list[bool] = []
    for child in ss.spawn(replicates):
        rng = np.random.default_rng(child)
        outcome = detector(injector(clean_features, rng), rng)
        if isinstance(outcome, Mapping):
            outcomes.append(bool(outcome.get("detected", False)))
        else:
            outcomes.append(bool(outcome))
    rate = float(np.mean(outcomes))
    # Wilson interval gives a stable finite-sample uncertainty interval for the
    # detection probability, including when the observed count is 0 or n.
    z = 1.959963984540054
    n = len(outcomes)
    phat = rate
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
    return {
        "replicates": replicates,
        "detections": int(sum(outcomes)),
        "detection_rate": rate,
        "detection_rate_95_ci": {"lower": max(0.0, center - half), "upper": min(1.0, center + half)},
        "target": CALIBRATION_DETECTION_TARGET,
        "target_met": rate >= CALIBRATION_DETECTION_TARGET,
    }


def familywise_test_count(
    *,
    partition_count: int,
    lag_count: int,
    near_duplicate_radius_count: int,
    structured_view_count: int,
    multivariate_count: int,
) -> int:
    """Return the number of hypothesis-level tests in one D2 execution.

    Replicates are sampling repetitions for the same hypothesis and therefore
    are combined before multiplicity correction; they are not separate
    familywise hypotheses.
    """
    if partition_count < 1:
        raise ValueError("partition_count must be >= 1.")
    comparisons = partition_count + partition_count * (partition_count - 1) // 2
    return max(1, comparisons + lag_count + near_duplicate_radius_count * comparisons
               + structured_view_count + multivariate_count)


def run_d2(
    *,
    partitions: Mapping[str, np.ndarray],
    feature_bits: int,
    reference_pmf: np.ndarray | None = None,
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
    nominal_pmf = validate_reference_pmf(reference_pmf, feature_bits) if reference_pmf is not None else None
    arrays = {name: _binary(x, name, feature_bits) for name, x in partitions.items()}
    structured_views = structured_views or {}
    lags = tuple(int(k) for k in lags)
    near_duplicate_radii = tuple(int(r) for r in near_duplicate_radii)
    if any(k < 1 for k in lags):
        raise ValueError("All lags must be positive.")

    comparisons = [(f"within:{n}", x, x) for n, x in arrays.items()]
    names = list(arrays)
    comparisons += [
        (f"cross:{names[i]}:{names[j]}", arrays[names[i]], arrays[names[j]])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    lag_tests = [(f"{n}:lag{k}", x, k) for n, x in arrays.items() for k in lags if k < len(x)]

    selected_mv = tuple(multivariate_partitions or arrays.keys())
    for name in selected_mv:
        if name not in arrays:
            raise ValueError(f"Unknown multivariate partition: {name}")
        if 2 * multivariate_pairs > len(arrays[name]):
            raise ValueError(f"Multivariate pair count exceeds half the size of partition {name!r}.")

    # The family includes every hypothesis-level p-value reported by D2.
    # Bonferroni therefore also covers near-duplicate, structured-collision,
    # and multivariate tests, which the previous implementation omitted.
    total_tests = familywise_test_count(
        partition_count=len(arrays),
        lag_count=len(lag_tests),
        near_duplicate_radius_count=len(near_duplicate_radii),
        structured_view_count=len(structured_views),
        multivariate_count=len(selected_mv),
    )
    alpha = FAMILYWISE_ALPHA / total_tests
    ss = np.random.SeedSequence(audit_seed)

    hamming: dict[str, Any] = {}
    near_dups: dict[str, Any] = {}
    for name, a, b in comparisons:
        reps = []
        distance_replicates = []
        for child in ss.spawn(audit_replicates):
            rng = np.random.default_rng(child)
            if a is b:
                d = sample_within_distances(a, pairs_per_test, rng)
            else:
                d = sample_cross_distances(a, b, pairs_per_test, rng)
            distance_replicates.append(d)
            pmf = empirical_independent_hamming_pmf(a, b)
            reps.append(summarize_distances(d, pmf, rng, alpha, bootstrap_replicates))
        hamming[name] = _aggregate(reps)
        near_dups[name] = near_duplicate_summary(
            distance_replicates,
            near_duplicate_radii,
            empirical_independent_hamming_pmf(a, b),
            bootstrap_replicates,
            np.random.default_rng(ss.spawn(1)[0]),
            alpha=alpha,
        )

    lagged: dict[str, Any] = {}
    for name, x, lag in lag_tests:
        reps = []
        for child in ss.spawn(audit_replicates):
            rng = np.random.default_rng(child)
            reps.append(
                summarize_distances(
                    sample_lagged_distances(x, lag, pairs_per_test, rng),
                    empirical_independent_hamming_pmf(x, x),
                    rng,
                    alpha,
                    bootstrap_replicates,
                )
            )
        lagged[name] = _aggregate(reps)

    structured = analyze_structured_views(structured_views, alpha)

    mv: dict[str, Any] = {}
    for name in selected_mv:
        rng = np.random.default_rng(ss.spawn(1)[0])
        mv[name] = multivariate_pair_discrimination(
            arrays[name],
            pairs=multivariate_pairs,
            permutations=multivariate_permutations,
            rng=rng,
            alpha=alpha,
        )

    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for name, result in {**hamming, **lagged}.items():
        if not result["practical_pass"]:
            failures.append({
                "component": name,
                "type": "distributional",
                "reason": "Upper bootstrap TVD uncertainty bound exceeded the pre-specified practical tolerance.",
            })
        elif result["statistical_warning"]:
            warnings.append({
                "component": name,
                "type": "statistical",
                "reason": "The replicate-level p-value combination rejected the declared independent-row reference after Bonferroni correction; the observed TVD remained practically below tolerance.",
                "minimum_p_value": result["min_p_value"],
                "combined_p_value": result["combined_p_value"],
            })

    for comparison, radii_results in near_dups.items():
        for radius, result in radii_results.items():
            if result["practically_excessive"]:
                failures.append({
                    "component": f"{comparison}:radius{radius}",
                    "type": "near_duplicate_structure",
                    "reason": "Observed near-duplicate rate exceeded the pre-specified practical excess threshold relative to the supplied null model.",
                    "excess_ratio": result["excess_ratio"],
                })
            elif result["statistically_excessive"]:
                warnings.append({
                    "component": f"{comparison}:radius{radius}",
                    "type": "near_duplicate_structure",
                    "reason": "Near-duplicate rate was statistically excessive under the model-based diagnostic but not practically excessive.",
                    "p_value": result["one_sided_excess_p_value"],
                })

    for name, result in structured.items():
        ref = result["reference"]
        if ref["practically_excessive"]:
            failures.append({
                "component": name,
                "type": "structured_collision",
                "reason": "Structured-value collision excess exceeded the practical tolerance.",
            })
        elif ref["statistically_excessive"]:
            warnings.append({
                "component": name,
                "type": "structured_collision",
                "reason": "Structured-value collision count was statistically excessive but not practically excessive.",
            })

    for name, result in mv.items():
        if result["practically_excessive"]:
            failures.append({
                "component": name,
                "type": "multivariate_dependence",
                "reason": "Real pairs were distinguishable from permuted partners beyond the practical AUC tolerance.",
                "auc": result["auc"],
                "auc_excess_over_chance": result["auc_excess_over_chance"],
            })
        elif result["statistically_excessive"]:
            warnings.append({
                "component": name,
                "type": "multivariate_dependence",
                "reason": "The permutation test detected statistical evidence of pair dependence without a practically material AUC effect.",
                "p_value": result["permutation_p_value"],
            })

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
            "hypothesis_counting": "one hypothesis per comparison after replicate-level p-value aggregation; audit replicates are not counted as separate familywise hypotheses",
            "confirmatory_hamming_null": "empirical independent-row Poisson-binomial Hamming null preserving observed endpoint bit marginals; independence of mismatch indicators across bit positions is an explicit model assumption",
            "nominal_hamming_reference": "optional Binomial(feature_bits, 0.5) case-study diagnostic; not used as the universal independence null",
            "nominal_reference_supplied": nominal_pmf is not None,
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
            "pair_sampling": "within and lagged pairs use disjoint endpoints within each sampling replicate; cross-partition pairs sample without replacement within each partition",
            "replication_semantics": "replicates are Monte Carlo sampling replicates from one fixed audited dataset instance, not independent dataset generations",
        },
    }
    decision = {
        "outcome": outcome,
        "failures": failures,
        "warnings": warnings,
        "interpretation": (
            "No practically material dependence detected within the tested classes, declared null models, and measured sensitivity envelope."
            if outcome == "PASS"
            else "The decision is limited to the explicitly tested dependence classes, null models, practical thresholds, and sensitivity demonstrated by the recorded calibration experiments."
        ),
        "not_proof_of_independence": True,
    }
    return results, decision


def build_d2_certificate(
    *,
    results: Mapping[str, Any],
    decision: Mapping[str, Any],
    partitions: Mapping[str, np.ndarray],
    dataset_id: str,
    dataset_version: str | None,
    generation_procedure: str | None,
    generation_parameters: Mapping[str, Any] | None,
    generation_random_seed: int | None,
    reference_description: str,
    reference_model_description: str,
    audit_seed: int,
    output_path: str,
) -> dict[str, Any]:
    provenance = build_provenance(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        generation_procedure=generation_procedure,
        generation_parameters=generation_parameters,
        random_seed=generation_random_seed,
        partitions={
            name: {
                "sample_count": int(x.shape[0]),
                "feature_count": int(x.shape[1]),
                "dtype": str(x.dtype),
                "shape": list(x.shape),
                "sha256": array_sha256(x),
            }
            for name, x in partitions.items()
        },
        audit_configuration={
            **dict(results["configuration"]),
            "reference_distribution": reference_description,
            "reference_model": reference_model_description,
        },
    )
    certificate = make_certificate(
        audit_id=AUDIT_ID,
        audit_name=AUDIT_NAME,
        claim=(
            "The supplied dataset instance was audited for detectable near-duplicate structure, "
            "pairwise and serial dependence, multivariate pair dependence, and adapter-exposed structured repetition."
        ),
        outcome=str(decision["outcome"]),
        findings={"results": dict(results), "decision": dict(decision)},
        methodology={
            "scope": "Dataset Integrity D2 Sample Dependence Audit",
            "d1_boundary": "D1 owns exact row duplication and exact partition overlap; D2 does not repeat those exact-equality claims.",
            "d2_1": "Near-duplicate rates are evaluated at pre-specified Hamming radii against the empirical independent-row Poisson-binomial null derived from observed endpoint marginals, with model-based excess tests and replicate-level uncertainty.",
            "d2_2": "Generation-order lagged Hamming structure is tested over the pre-specified lag family against the empirical independent-row Poisson-binomial null. Within/cross pairwise comparisons are retained as feature-space controls; lag tests carry the primary serial-dependence interpretation.",
            "d2_3": "A fixed logistic detector attempts to distinguish real pairs from permuted partners. The classifier is fitted once and significance is evaluated by permuting held-out test labels conditional on the fixed test scores.",
            "d2_4": "Adapter-supplied semantic views are evaluated for excess finite-domain collisions under the stated independent-uniform reference model.",
            "d2_5": "Controlled duplicate and lag-copy injectors are available for empirical sensitivity calibration; calibration results are recorded separately and are not treated as clean-data findings.",
            "multiple_comparison_control": "Bonferroni familywise control at 0.01 across one hypothesis per reported comparison after replicate-level p-value aggregation; audit replicates are not counted as separate hypotheses.",
            "replication": "Repeated sampling replicates quantify Monte Carlo variability conditional on the same audited dataset instance; they are not treated as independent dataset generations.",
            "decision_semantics": "PASS is evidence against the tested dependence hypotheses within the declared scope, not proof of universal independence.",
            "reference_distribution": reference_description,
            "reference_model": reference_model_description,
            "audit_seed": audit_seed,
        },
        provenance=provenance,
        limitations=[
            "Finite statistical testing cannot establish universal mutual independence.",
            "D2 is a falsification battery for explicitly tested dependence mechanisms, not a proof of the full joint independence property.",
            "Sensitivity is specific to the declared tests, sample sizes, detector, null models, practical thresholds, and completed calibration experiments.",
            "The Binomial(64, 0.5) Hamming reference is retained only as a case-study diagnostic; confirmatory Hamming inference uses an empirical independent-row Poisson-binomial null.",
            "Pearson chi-square p-values are diagnostic model-based evidence; practical decisions also require the pre-specified TVD tolerance and bootstrap uncertainty.",
            "Near-duplicate and structured-collision p-values rely on their stated finite-domain null approximations and should not be interpreted as formal proofs.",
            "Structured repetition tests are conditional on adapter-exposed representations.",
            "Multivariate discrimination detects dependence only to the extent represented by the fixed detector and feature construction.",
            "Audit replicates reuse the same dataset instance; they quantify sampling variability rather than independent dataset-to-dataset variability.",
            "Calibration must be run and reported before claiming a corresponding detection sensitivity level.",
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
    print("=" * 78)
