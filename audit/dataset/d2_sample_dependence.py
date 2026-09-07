"""Generic D2 sample-dependence audit engine.

D2 is a falsification battery.  It does not prove universal mutual
independence.  The implementation deliberately separates:

* fixed-dataset randomization inference (pairwise/serial structure),
* practical-effect thresholds,
* the probabilistic near-neighbour detector, and
* controlled fault-injection calibration.

No case-study-specific generator assumptions live in this module.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from audit.dataset.common.certificate import make_certificate, write_certificate
from audit.dataset.common.provenance import array_sha256, build_provenance

AUDIT_ID = "D2"
AUDIT_NAME = "Sample Dependence Audit"
SCHEMA_VERSION = "6.4"
FAMILYWISE_ALPHA = 0.01
PRACTICAL_EXCESS_TVD_THRESHOLD = 0.01
MULTIVARIATE_AUC_TOLERANCE = 0.05
NEAR_DUPLICATE_RADII = (1, 2, 4, 8)
DEFAULT_LAGS = tuple(list(range(1, 17)) + [32, 64, 128, 256, 512, 1024])
DEFAULT_PAIRS_PER_TEST = 20_000
DEFAULT_AUDIT_REPLICATES = 10
DEFAULT_NULL_PERMUTATIONS = 120_000
DEFAULT_NULL_REFERENCE_PERMUTATIONS = 50
DEFAULT_NULL_BATCH_SIZE = 100
DEFAULT_NULL_STOPPING_ERROR = 0.01
DEFAULT_BOOTSTRAP_REPLICATES = 1_000
DEFAULT_MULTIVARIATE_PAIRS = 50_000
DEFAULT_MULTIVARIATE_PERMUTATIONS = 1_000
DEFAULT_DETECTOR_C = 1.0
CALIBRATION_MIN_REPLICATES = 100
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
    if p.ndim != 1 or len(p) != n_bits + 1 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("reference_pmf has invalid shape or values.")
    if not math.isclose(float(p.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("reference_pmf must sum to one.")
    return p


def binomial_reference_distribution(n_bits: int, p: float = 0.5) -> np.ndarray:
    """Nominal Binomial reference retained for diagnostic reporting only."""
    if n_bits < 1 or not 0 <= p <= 1:
        raise ValueError("Invalid binomial parameters.")
    pmf = np.array(
        [math.comb(n_bits, k) * p**k * (1 - p) ** (n_bits - k) for k in range(n_bits + 1)],
        dtype=np.float64,
    )
    return pmf / pmf.sum()


def empirical_independent_hamming_pmf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Legacy diagnostic: bit-marginal Poisson-binomial reference.

    This is *not* used for confirmatory D2 inference because it assumes
    independence of mismatch indicators across bit positions.  Cryptographic
    rows can have strong within-row bit structure even when rows are independent.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or len(a) < 1 or len(b) < 1:
        raise ValueError("a and b must be non-empty 2-D arrays with equal feature counts.")
    pa, pb = a.mean(axis=0), b.mean(axis=0)
    q = pa * (1 - pb) + (1 - pa) * pb
    pmf = np.array([1.0], dtype=np.float64)
    for prob in q:
        pmf = np.convolve(pmf, np.array([1 - prob, prob], dtype=np.float64))
    return pmf / pmf.sum()


def _sample_disjoint_pairs(n: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n < 2 or count < 1 or 2 * count > n:
        raise ValueError(f"Cannot sample {count:,} disjoint pairs from {n:,} samples.")
    z = rng.choice(n, size=2 * count, replace=False)
    return z[:count].astype(np.int64), z[count:].astype(np.int64)


def _sample_cross_pairs(n_a: int, n_b: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if min(n_a, n_b) < count or count < 1:
        raise ValueError("Invalid cross-partition sampling request.")
    return (
        rng.choice(n_a, count, replace=False).astype(np.int64),
        rng.choice(n_b, count, replace=False).astype(np.int64),
    )


def _sample_disjoint_lag_pairs(n: int, lag: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= lag < n or count < 1:
        raise ValueError("Invalid lag.")
    residues = np.arange(min(lag, n), dtype=np.int64)
    starts_parts = []
    for r in residues:
        # Each residue class is a path whose vertices are r, r+lag, ... .
        # A path with L vertices has floor(L/2) disjoint lag-edges.  For an
        # even-length path only phase 0 attains that maximum; for an odd-length
        # path either phase is maximum.  Choosing only maximum matchings avoids
        # an avoidable off-by-one loss at lag values such as lag=1.
        length = ((n - 1 - int(r)) // lag) + 1
        max_edges = length // 2
        if max_edges == 0:
            continue
        if length % 2 == 0:
            phases = (0,)
        else:
            phases = (0, 1)
        ph = phases[int(rng.integers(0, len(phases)))]
        first = int(r + ph * lag)
        starts = np.arange(first, first + 2 * max_edges * lag, 2 * lag, dtype=np.int64)
        starts_parts.append(starts)
    candidates = np.concatenate(starts_parts) if starts_parts else np.empty(0, dtype=np.int64)
    if len(candidates) < count:
        raise ValueError(
            f"Could not construct {count:,} disjoint lag-{lag} pairs from {n:,} samples; "
            f"maximum available is {len(candidates):,}."
        )
    chosen = rng.choice(candidates, size=count, replace=False)
    return chosen.astype(np.int64), (chosen + lag).astype(np.int64)


_POPCOUNT8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)

def _pack_binary_rows(x: np.ndarray) -> np.ndarray:
    """Pack binary rows once for repeated Hamming-distance evaluation.

    This is a representation-only optimization. It preserves the exact bit
    values and therefore cannot alter the Hamming-distance statistic or its
    null distribution.
    """
    x = np.asarray(x, dtype=np.uint8)
    if x.ndim != 2:
        raise ValueError("x must be a 2-D binary row matrix.")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("x must contain only binary values 0/1.")
    return np.packbits(x, axis=1, bitorder="little")

def _hamming_distances_packed(packed_a: np.ndarray, packed_b: np.ndarray) -> np.ndarray:
    """Exact row-wise Hamming distance for already-packed binary rows."""
    packed_a = np.asarray(packed_a, dtype=np.uint8)
    packed_b = np.asarray(packed_b, dtype=np.uint8)
    if packed_a.ndim != 2 or packed_b.ndim != 2 or packed_a.shape != packed_b.shape:
        raise ValueError("packed_a and packed_b must have identical 2-D shapes.")
    return _POPCOUNT8[np.bitwise_xor(packed_a, packed_b)].sum(axis=1, dtype=np.uint16).astype(np.int16)

def hamming_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact row-wise Hamming distance using packed-byte XOR/popcount.

    The calculation is algebraically identical to counting differing bits in
    the original binary rows. Packing is performed here for ordinary callers;
    repeated Monte Carlo callers reuse packed arrays through
    ``_hamming_distances_packed`` to avoid repeated representation work.
    """
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("a and b must have identical 2-D shapes.")
    return _hamming_distances_packed(_pack_binary_rows(a), _pack_binary_rows(b))


def sample_within_distances(x: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    i, j = _sample_disjoint_pairs(len(x), count, rng)
    return hamming_distances(x[i], x[j])


def sample_cross_distances(a: np.ndarray, b: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    i, j = _sample_cross_pairs(len(a), len(b), count, rng)
    return hamming_distances(a[i], b[j])


def sample_lagged_distances(x: np.ndarray, lag: int, count: int, rng: np.random.Generator) -> np.ndarray:
    i, j = _sample_disjoint_lag_pairs(len(x), lag, count, rng)
    return hamming_distances(x[i], x[j])


def _histogram(distances: np.ndarray, n_bits: int) -> np.ndarray:
    return np.bincount(np.asarray(distances, dtype=np.int64), minlength=n_bits + 1).astype(np.float64)


def _tvd_from_histogram(hist: np.ndarray, reference_pmf: np.ndarray) -> float:
    return float(0.5 * np.abs(hist / hist.sum() - reference_pmf).sum())


def _bootstrap_observed_tvd(
    distances: np.ndarray,
    reference_pmf: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    """Bootstrap uncertainty for the observed histogram/TVD statistic.

    This is an uncertainty interval for the empirical TVD, not a null
    hypothesis test and not a confidence interval for dependence itself.
    """
    hist = _histogram(distances, len(reference_pmf) - 1)
    p = hist / hist.sum()
    draws = rng.multinomial(len(distances), p, size=reps) / len(distances)
    vals = 0.5 * np.abs(draws - reference_pmf).sum(axis=1)
    return {
        "confidence_level": 0.95,
        "replicates": int(reps),
        "lower": float(np.quantile(vals, 0.025)),
        "upper": float(np.quantile(vals, 0.975)),
        "interpretation": "bootstrap uncertainty for the observed TVD; not a null p-value",
    }


def _randomize_pairing(
    a: np.ndarray,
    b: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    within: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample endpoint rows once, then randomize their pairing.

    The complete binary rows are preserved.  This is the key change from the
    earlier bit-marginal null: within-row cryptographic structure is retained.
    """
    if within:
        i, j = _sample_disjoint_pairs(len(a), count, rng)
        left = a[i]
        right = a[j]
        return left, right, i, j
    i, j = _sample_cross_pairs(len(a), len(b), count, rng)
    return a[i], b[j], i, j


def _null_pair_histogram(
    left: np.ndarray,
    right: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
    within: bool,
    reference_permutations: int = DEFAULT_NULL_REFERENCE_PERMUTATIONS,
    batch_size: int = DEFAULT_NULL_BATCH_SIZE,
    stopping_error: float = DEFAULT_NULL_STOPPING_ERROR,
    alpha: float | None = None,
    observed_distances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate a fixed-dataset randomization null with adaptive tail sampling.

    The first ``reference_permutations`` draws estimate the fixed-dataset
    reference PMF.  Reference draws are kept separate from the confirmatory
    tail sample so their Monte Carlo noise is visible in the certificate.  Independent subsequent draws estimate the randomization
    tail.  Tail sampling proceeds in fixed batches and may stop early only when
    a Clopper--Pearson bound, with a Bonferroni correction over the predeclared
    batch looks, places the randomization p-value wholly above or below the
    decision threshold.  The maximum draw count remains large enough to resolve
    the corrected alpha when a test is genuinely close to the boundary.
    """
    from scipy.stats import beta as beta_dist

    n, n_bits = len(left), left.shape[1]
    if len(right) != n:
        raise ValueError("left/right pair pools must have equal size.")
    if permutations < 2 or reference_permutations < 1 or reference_permutations >= permutations:
        raise ValueError("permutations must exceed reference_permutations >= 1.")
    if batch_size < 1 or not 0 < stopping_error < 1:
        raise ValueError("Invalid batch_size or stopping_error.")
    if alpha is not None and not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")

    # Pack the fixed row pools once. The randomization below changes only row
    # indices, so re-packing each draw would be pure overhead and cannot add
    # statistical information. This preserves the exact statistic/null model.
    packed_left = _pack_binary_rows(left)
    packed_right = _pack_binary_rows(right)
    packed_pool = np.concatenate((packed_left, packed_right), axis=0) if within else None

    def one_hist() -> np.ndarray:
        if within:
            z = rng.permutation(2 * n)
            x1 = packed_pool[z[:n]]
            x2 = packed_pool[z[n:]]
        else:
            perm = rng.permutation(n)
            x1, x2 = packed_left, packed_right[perm]
        return _histogram(_hamming_distances_packed(x1, x2), n_bits)

    ref_hists = np.empty((reference_permutations, n_bits + 1), dtype=np.float64)
    for k in range(reference_permutations):
        ref_hists[k] = one_hist()
    reference = ref_hists.mean(axis=0)
    reference /= reference.sum()
    reference_component_sd = ref_hists.std(axis=0, ddof=1) if reference_permutations > 1 else np.zeros(n_bits + 1)

    observed_tvd = None
    if observed_distances is not None:
        observed_tvd = _tvd_from_histogram(_histogram(observed_distances, n_bits), reference)

    max_tail = permutations - reference_permutations
    max_looks = int(math.ceil(max_tail / batch_size))
    look_error = stopping_error / max_looks
    null_stats_list: list[float] = []
    exceedances = 0
    stop_reason = "maximum_permutations_reached"
    decision = "undetermined"
    lower_p = None
    upper_p = None

    for _ in range(max_looks):
        batch_n = min(batch_size, max_tail - len(null_stats_list))
        if batch_n <= 0:
            break
        batch_stats = np.empty(batch_n, dtype=np.float64)
        for k in range(batch_n):
            h = one_hist()
            batch_stats[k] = _tvd_from_histogram(h, reference)
        null_stats_list.extend(batch_stats.tolist())

        if observed_tvd is not None and alpha is not None:
            b = len(null_stats_list)
            x = exceedances + int(np.count_nonzero(batch_stats >= observed_tvd))
            exceedances = x
            if x == 0:
                lower_q = 0.0
            else:
                lower_q = float(beta_dist.ppf(look_error / 2.0, x, b - x + 1))
            if x == b:
                upper_q = 1.0
            else:
                upper_q = float(beta_dist.ppf(1.0 - look_error / 2.0, x + 1, b - x))
            # The +1 correction is appropriate for the reported Monte Carlo
            # p-value, but must not be used to manufacture a non-significance
            # conclusion when the observed exceedance count is zero.  The
            # lower bound below is a bound on the underlying randomization-tail
            # probability q; if it is above alpha, then the true p-value is
            # necessarily above alpha.
            upper_p = float((1.0 + b * upper_q) / (b + 1.0))
            # The inferential decision is based on simultaneous confidence
            # bounds, not on an adaptively stopped plug-in p-value.  This
            # avoids optional-stopping bias in the reported decision.
            lower_p = float((1.0 + b * lower_q) / (b + 1.0))
            if upper_p < alpha:
                stop_reason = "upper_p_below_alpha"
                decision = "reject"
                break
            if lower_q > alpha:
                stop_reason = "lower_tail_above_alpha"
                decision = "do_not_reject"
                break

    null_stats = np.asarray(null_stats_list, dtype=np.float64)
    meta = {
        "reference_permutations": int(reference_permutations),
        "reference_pmf_estimator": "mean of independent fixed-dataset randomization histograms",
        "reference_pmf_component_sd_max": float(np.max(reference_component_sd)),
        "reference_pmf_component_sd_median": float(np.median(reference_component_sd)),
        "tail_permutations_used": int(len(null_stats)),
        "requested_permutations": int(permutations),
        "batch_size": int(batch_size),
        "adaptive_stopping": bool(alpha is not None and observed_tvd is not None),
        "stopping_error": float(stopping_error),
        "stopping_look_error": float(look_error),
        "stopping_max_looks": int(max_looks),
        "stopping_reason": stop_reason,
        "alpha_for_stopping": alpha,
        "minimum_attainable_p_at_used_tail": float(1.0 / (len(null_stats) + 1)),
        "sequential_decision": decision,
        "sequential_lower_p_bound": lower_p,
        "sequential_upper_p_bound": upper_p,
        "decision_interpretation": (
            "reject only when the simultaneous upper confidence bound for the Monte Carlo p-value is below alpha; "
            "do not reject only when the simultaneous lower bound is above alpha; otherwise the result is statistically undetermined"
        ),
    }
    return reference, null_stats, meta

def summarize_randomization_test(
    observed_distances: np.ndarray,
    null_pmf: np.ndarray,
    null_stats: np.ndarray,
    *,
    alpha: float,
    rng: np.random.Generator,
    bootstrap_replicates: int,
    practical_excess_tvd_threshold: float = PRACTICAL_EXCESS_TVD_THRESHOLD,
    null_sampling_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not 0 <= practical_excess_tvd_threshold <= 1:
        raise ValueError("practical_excess_tvd_threshold must be in [0, 1].")
    n_bits = len(null_pmf) - 1
    observed_hist = _histogram(observed_distances, n_bits)
    observed_tvd = _tvd_from_histogram(observed_hist, null_pmf)
    null_mean_tvd = float(np.mean(null_stats))
    null_sd_tvd = float(np.std(null_stats, ddof=1)) if len(null_stats) > 1 else 0.0
    excess_tvd = float(observed_tvd - null_mean_tvd)
    standardized_excess = (excess_tvd / null_sd_tvd) if null_sd_tvd > 0 else (float("inf") if excess_tvd > 0 else 0.0)
    p_value = float((1 + np.count_nonzero(null_stats >= observed_tvd)) / (len(null_stats) + 1))
    uncertainty = _bootstrap_observed_tvd(observed_distances, null_pmf, rng, bootstrap_replicates)
    meta = dict(null_sampling_metadata or {})
    sequential_decision = meta.get("sequential_decision", "undetermined")
    statistically_excessive = sequential_decision == "reject"
    return {
        "sampled_pairs": int(len(observed_distances)),
        "mean_hamming_distance": float(np.mean(observed_distances)),
        "std_hamming_distance": float(np.std(observed_distances)),
        "min_hamming_distance": int(np.min(observed_distances)),
        "max_hamming_distance": int(np.max(observed_distances)),
        "tvd": observed_tvd,
        "null_mean_tvd": null_mean_tvd,
        "null_sd_tvd": null_sd_tvd,
        "excess_tvd": excess_tvd,
        "standardized_excess_tvd": standardized_excess,
        "practical_effect_measure": "excess_TVD = observed_TVD - mean_TVD_under_fixed_dataset_randomization_null",
        "practical_effect_threshold": float(practical_excess_tvd_threshold),
        "practical_effect_interpretation": (
            f"The practical threshold is {practical_excess_tvd_threshold:.6g} in TVD units; "
            "because total variation distance is the maximum event-probability discrepancy, "
            "this is the corresponding maximum discrepancy in probability mass. The raw observed TVD "
            "is not used as the practical gate because finite samples have nonzero TVD under the null."
        ),
        "tvd_bootstrap_uncertainty": uncertainty,
        "randomization_test": {
            "p_value": p_value,
            "alpha": alpha,
            "null_permutations": int(len(null_stats)),
            "null_permutations_requested": int((null_sampling_metadata or {}).get("requested_permutations", len(null_stats))),
            "null_reference_permutations": int((null_sampling_metadata or {}).get("reference_permutations", 0)),
            "null_tvd_mean": null_mean_tvd,
            "null_tvd_sd": null_sd_tvd,
            "null_tvd_95_quantile": float(np.quantile(null_stats, 0.95)),
            "statistically_excessive": statistically_excessive,
            "decision_basis": "simultaneous sequential confidence bounds",
            "sequential_decision": sequential_decision,
            "sequential_lower_p_bound": meta.get("sequential_lower_p_bound"),
            "sequential_upper_p_bound": meta.get("sequential_upper_p_bound"),
            "mc_p_value_estimate": p_value,
            "mc_p_value_interpretation": "Monte Carlo tail estimate; not the sole inferential basis when adaptive stopping is used",
            "null_model": "fixed-dataset randomization of complete rows; all within-row bit structure is preserved",
            "sampling_metadata": meta,
        },
        "practically_excessive": excess_tvd > practical_excess_tvd_threshold,
    }


def _aggregate_randomization_results(
    results: Sequence[Mapping[str, Any]],
    *,
    practical_excess_tvd_threshold: float,
) -> dict[str, Any]:
    """Summarize repeated sampling without treating batches as independent studies."""
    tvds = np.asarray([float(r["tvd"]) for r in results])
    pvals = np.asarray([float(r["randomization_test"]["p_value"]) for r in results])
    statistical_decisions = [r["randomization_test"].get("sequential_decision", "undetermined") for r in results]
    excess = np.asarray([float(r["excess_tvd"]) for r in results])
    return {
        "sampling_replicates": int(len(results)),
        "replication_semantics": "Monte Carlo sampling replicates conditional on one fixed dataset instance; no p-value combination is performed",
        "replicates": [dict(r) for r in results],
        "tvd_median": float(np.median(tvds)),
        "tvd_max": float(np.max(tvds)),
        "tvd_95_percentile": float(np.quantile(tvds, 0.95)),
        "null_centered_excess_tvd_median": float(np.median(excess)),
        "null_centered_excess_tvd_max": float(np.max(excess)),
        "null_centered_excess_tvd_95_percentile": float(np.quantile(excess, 0.95)),
        "practical_effect_threshold": float(practical_excess_tvd_threshold),
        "minimum_randomization_p_value": float(np.min(pvals)),
        "practical_pass": bool(np.max(excess) <= practical_excess_tvd_threshold),
        "statistical_warning": bool(any(d == "reject" for d in statistical_decisions)),
        "statistical_decision_status": (
            "reject" if any(d == "reject" for d in statistical_decisions)
            else "do_not_reject" if all(d == "do_not_reject" for d in statistical_decisions)
            else "undetermined"
        ),
    }


# ---------- D2.1 scalable near-neighbour screen ----------


def _uint64_rows(x: np.ndarray) -> np.ndarray:
    if x.shape[1] != 64:
        raise ValueError("Near-neighbour index currently requires exactly 64 binary features.")
    packed = np.ascontiguousarray(np.packbits(x, axis=1, bitorder="little"))
    return packed.view("<u8").reshape(-1)


def _projection_signature(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    sig = np.zeros(len(values), dtype=np.uint16)
    for k, p in enumerate(positions):
        sig |= (((values >> np.uint64(int(p))) & 1).astype(np.uint16) << np.uint16(k))
    return sig


def _empirical_pair_distance_probability(
    x: np.ndarray,
    radius: int,
    *,
    pairs: int,
    rng: np.random.Generator,
) -> float:
    """Estimate P[d(X,X') <= radius] for two distinct empirical rows.

    Sampling the second endpoint conditional on the first prevents accidental
    self-pairs, which otherwise create artificial distance-zero mass.
    """
    n = len(x)
    if n < 2:
        raise ValueError("At least two rows are required.")
    i = rng.integers(0, n, size=pairs, dtype=np.int64)
    j = rng.integers(0, n - 1, size=pairs, dtype=np.int64)
    j += (j >= i)
    d = hamming_distances(x[i], x[j])
    return float(np.mean(d <= radius))


def near_neighbor_summary(
    x: np.ndarray,
    radii: Sequence[int],
    *,
    rng: np.random.Generator,
    query_count: int = 2048,
    projection_tables: int = 8,
    max_candidates_per_bucket: int = 10_000,
    null_pairs: int = 100_000,
    alpha: float = FAMILYWISE_ALPHA,
    practical_relative_excess: float = 0.25,
) -> dict[str, Any]:
    """Screen sampled rows for near-neighbours using exact verification.

    The null is empirical row resampling, not a bitwise/binomial model.  It
    preserves the complete observed row distribution, including arbitrary
    within-row dependencies.  Candidate retrieval remains probabilistic and
    therefore must be calibrated separately.
    """
    x = _binary(x, "x", x.shape[1])
    n = len(x)
    qn = min(int(query_count), n)
    radii = tuple(sorted(set(int(r) for r in radii)))
    if qn < 1 or projection_tables < 1 or null_pairs < 1:
        raise ValueError("Invalid near-neighbour configuration.")
    if any(r < 1 or r >= x.shape[1] for r in radii):
        raise ValueError("Invalid near-neighbour radius.")

    qidx = rng.choice(n, qn, replace=False).astype(np.int64)
    values = _uint64_rows(x) if x.shape[1] == 64 else None
    tables = []
    for _ in range(projection_tables):
        pos = np.sort(rng.choice(x.shape[1], min(16, x.shape[1]), replace=False).astype(np.uint8))
        if values is None:
            sig = np.zeros(n, dtype=np.uint16)
            for k, p in enumerate(pos):
                sig |= x[:, int(p)].astype(np.uint16) << np.uint16(k)
        else:
            sig = _projection_signature(values, pos)
        order = np.argsort(sig, kind="stable")
        if n <= np.iinfo(np.int32).max:
            order = order.astype(np.int32, copy=False)
        sorted_sig = sig[order]
        query_sig = sig[qidx].copy()
        del sig
        tables.append((order, sorted_sig, query_sig))

    table_bounds = []
    for order, ss, query_sig in tables:
        left = np.searchsorted(ss, query_sig, side="left")
        right = np.searchsorted(ss, query_sig, side="right")
        table_bounds.append((order, left, right))

    hits = np.zeros(len(radii), dtype=np.int64)
    candidate_checks = 0
    for qpos, qi in enumerate(qidx):
        chunks = []
        for order, left, right in table_bounds:
            lo, hi = int(left[qpos]), int(right[qpos])
            if hi > lo:
                if hi - lo > max_candidates_per_bucket:
                    sel = np.linspace(lo, hi - 1, max_candidates_per_bucket, dtype=np.int64)
                    chunks.append(order[sel])
                else:
                    chunks.append(order[lo:hi])
        if not chunks:
            continue
        candidates = np.unique(np.concatenate(chunks))
        candidates = candidates[candidates != qi]
        candidate_checks += len(candidates)
        if len(candidates) == 0:
            continue
        d = np.count_nonzero(x[candidates] != x[qi], axis=1)
        min_distance = int(d.min())
        hits += np.asarray([min_distance <= r for r in radii], dtype=np.int64)

    out = {}
    for radius, hit_count in zip(radii, hits):
        pair_q = _empirical_pair_distance_probability(x, radius, pairs=null_pairs, rng=rng)
        query_q = float(-math.expm1((n - 1) * math.log1p(-pair_q))) if pair_q < 1 else 1.0
        expected = qn * query_q
        # The query-hit indicators share a fixed empirical reference set and
        # are therefore not independent Bernoulli trials.  A binomial tail
        # would be anti-conservative.  We intentionally do not report a
        # formal hypothesis-test p-value here; this component is a calibrated
        # screening statistic, with D2.2 providing the confirmatory
        # fixed-dataset randomization inference.
        p_value = None
        ratio = float(hit_count / expected) if expected > 0 else (1.0 if hit_count == 0 else float("inf"))
        out[str(radius)] = {
            "queried_rows": int(qn),
            "rows_with_verified_neighbor_within_radius": int(hit_count),
            "observed_query_rate": float(hit_count / qn),
            "candidate_checks": int(candidate_checks),
            "projection_tables": int(projection_tables),
            "projection_bits": int(min(16, x.shape[1])),
            "candidate_retrieval": "random projection buckets followed by exact Hamming verification",
            "exhaustive": False,
            "null_model": "independent complete rows sampled from the empirical row population",
            "null_pair_probability": pair_q,
            "null_query_probability": query_q,
            "null_pair_simulation_pairs": int(null_pairs),
            "expected_query_hits": float(expected),
            "excess_ratio": ratio,
            "one_sided_excess_p_value": p_value,
            "statistical_inference": "not_claimed; shared reference set induces dependence among query-hit indicators",
            "statistically_excessive": False,
            "practically_excessive": bool(ratio > 1 + practical_relative_excess),
            "practical_relative_excess_threshold": float(practical_relative_excess),
        }
    return out


# Backward-compatible helper retained for callers that already use it.
def near_duplicate_summary(
    distance_replicates,
    radii,
    reference_pmf,
    bootstrap_replicates,
    rng,
    practical_relative_excess=0.25,
    alpha=FAMILYWISE_ALPHA,
):
    from scipy.stats import binom
    d = np.concatenate(distance_replicates)
    out = {}
    for r in radii:
        observed = int(np.count_nonzero(d <= r))
        q = float(reference_pmf[: r + 1].sum())
        expected = len(d) * q
        ratio = observed / expected if expected else (1.0 if observed == 0 else float("inf"))
        p = float(binom.sf(observed - 1, len(d), q)) if q > 0 else (1.0 if observed == 0 else 0.0)
        out[str(r)] = {
            "observed_pairs": observed,
            "sampled_pairs": len(d),
            "observed_rate": observed / len(d),
            "null_probability": q,
            "expected_pairs": expected,
            "excess_ratio": ratio,
            "practical_relative_excess_threshold": practical_relative_excess,
            "one_sided_excess_p_value": p,
            "statistically_excessive": p < alpha,
            "practically_excessive": ratio > 1 + practical_relative_excess,
            "bootstrap_rate_uncertainty": {
                "status": "legacy helper; uncertainty is over supplied replicate rates and is not used by run_d2"
            },
        }
    return out


def analyze_structured_views(
    structured_views: Mapping[str, Mapping[str, Any]],
    alpha: float,
    practical_relative_excess: float = 0.25,
) -> dict[str, Any]:
    """Analyze explicit finite-domain collision models supplied by the adapter.

    Generic D2 refuses to silently assume uniformity.  A structured view must
    explicitly declare ``collision_null='uniform'`` (or provide a future
    reference model) before a collision p-value is produced.
    """
    from scipy.stats import poisson

    out = {}
    for name, spec in structured_views.items():
        v = np.asarray(spec["values"]).reshape(-1)
        domain = int(spec.get("domain_size", 0))
        null_name = str(spec.get("collision_null", ""))
        if len(v) < 2 or domain < 2:
            raise ValueError(f"Invalid structured view {name}.")
        if null_name != "uniform":
            out[name] = {
                "description": str(spec.get("description", "")),
                "sample_count": int(len(v)),
                "unique_count": int(len(np.unique(v))),
                "status": "NOT_TESTED",
                "reason": "No explicit finite-domain collision null was supplied; D2 will not assume uniformity.",
            }
            continue
        _, counts = np.unique(v, return_counts=True)
        rep = counts[counts > 1]
        observed = int(np.sum(rep * (rep - 1) // 2))
        expected = len(v) * (len(v) - 1) / (2 * domain)
        p = float(poisson.sf(observed - 1, expected)) if observed else 1.0
        ratio = observed / expected if expected else (1.0 if observed == 0 else float("inf"))
        out[name] = {
            "description": str(spec.get("description", "")),
            "sample_count": int(len(v)),
            "unique_count": int(len(np.unique(v))),
            "collision_pairs": observed,
            "maximum_multiplicity": int(rep.max()) if len(rep) else 1,
            "reference": {
                "model": "Poisson approximation to pair-collision count under independent uniform sampling",
                "domain_size": domain,
                "expected_collision_pairs": float(expected),
                "observed_collision_pairs": observed,
                "excess_ratio": float(ratio),
                "practical_relative_excess_threshold": float(practical_relative_excess),
                "one_sided_p_value": p,
                "statistically_excessive": bool(p < alpha),
                "practically_excessive": bool(ratio > 1 + practical_relative_excess),
            },
        }
    return out


# ---------- D2.3 multivariate detector ----------


def _pair_matrix(x: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    a = x[i].astype(np.float32)
    b = x[j].astype(np.float32)
    return np.concatenate([a, b, np.abs(a - b)], axis=1)


def multivariate_pair_discrimination(
    x: np.ndarray,
    *,
    pairs: int,
    permutations: int,
    rng: np.random.Generator,
    c: float = DEFAULT_DETECTOR_C,
    alpha: float = FAMILYWISE_ALPHA,
) -> dict[str, Any]:
    """Test pair structure with row-disjoint train/test pools.

    The previous implementation reused the same endpoint rows across the
    detector's train and test split.  This version allocates disjoint row pools
    before constructing any pairs, preventing endpoint leakage.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if pairs < 2 or permutations < 1:
        raise ValueError("pairs and permutations must be positive.")
    # Four row pools are required: positive-train, negative-train,
    # positive-test, negative-test.  We use 2*pairs rows for each side and
    # create negatives by deranging the second endpoint within that pool.
    required = 4 * pairs
    if required > len(x):
        raise ValueError(f"Need at least {required:,} rows for row-disjoint detector; got {len(x):,}.")
    rows = rng.choice(len(x), required, replace=False)
    train_a, train_b, test_a, test_b = np.split(rows, 4)

    def derange(base: np.ndarray) -> np.ndarray:
        perm = rng.permutation(len(base))
        for _ in range(20):
            if not np.any(perm == np.arange(len(base))):
                return base[perm]
            perm = rng.permutation(len(base))
        # Deterministic cyclic derangement is guaranteed for n >= 2.
        return base[np.roll(np.arange(len(base)), 1)]

    X_train = np.vstack([
        _pair_matrix(x, train_a[:pairs], train_b[:pairs]),
        _pair_matrix(x, train_a[:pairs], derange(train_b[:pairs])),
    ])
    y_train = np.r_[np.ones(pairs, dtype=np.int8), np.zeros(pairs, dtype=np.int8)]

    X_test = np.vstack([
        _pair_matrix(x, test_a[:pairs], test_b[:pairs]),
        _pair_matrix(x, test_a[:pairs], derange(test_b[:pairs])),
    ])
    y_test = np.r_[np.ones(pairs, dtype=np.int8), np.zeros(pairs, dtype=np.int8)]

    model = LogisticRegression(
        C=c,
        solver="liblinear",
        max_iter=2000,
        random_state=int(rng.integers(2**31 - 1)),
    ).fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, scores))
    excess = abs(auc - 0.5)
    null = np.empty(permutations, dtype=np.float64)
    for k in range(permutations):
        null[k] = float(roc_auc_score(rng.permutation(y_test), scores))
    p = float((1 + np.count_nonzero(np.abs(null - 0.5) >= excess)) / (permutations + 1))
    return {
        "pairs": int(pairs),
        "train_examples": int(len(y_train)),
        "test_examples": int(len(y_test)),
        "row_disjoint_train_test": True,
        "detector": "logistic_regression_on_[x_i,x_j,abs(x_i-x_j)]",
        "detector_C": float(c),
        "auc": auc,
        "auc_excess_over_chance": float(excess),
        "practical_tolerance": MULTIVARIATE_AUC_TOLERANCE,
        "permutations": int(permutations),
        "permutation_p_value": p,
        "null_auc_mean": float(null.mean()),
        "null_auc_95_upper": float(np.quantile(null, 0.95)),
        "statistically_excessive": bool(p < alpha),
        "practically_excessive": bool(excess > MULTIVARIATE_AUC_TOLERANCE),
    }


# ---------- Controlled fault injectors ----------


def inject_near_duplicates(x: np.ndarray, fraction: float, radius: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    x = _binary(x, "x", x.shape[1])
    n = len(x)
    if not 0 <= fraction <= 1 or radius < 1 or radius > x.shape[1]:
        raise ValueError("Invalid near-duplicate parameters.")
    y = x.copy()
    count = int(round(n * fraction))
    if count == 0:
        return y, {"requested_fraction": fraction, "requested_rows": 0, "actual_modified_rows": 0, "radius": radius}
    if 2 * count > n:
        raise ValueError("Near-duplicate injection requires disjoint source and destination rows.")
    selected = rng.choice(n, 2 * count, replace=False)
    src, dst = selected[:count], selected[count:]
    scores = rng.integers(0, 256, size=(count, x.shape[1]), dtype=np.uint8)
    masks = np.argpartition(scores, radius - 1, axis=1)[:, :radius]
    y[dst] = y[src]
    y[dst[:, None], masks] ^= 1
    realized = np.mean(np.count_nonzero(y[dst] != x[dst], axis=1) == radius)
    return y, {
        "requested_fraction": float(fraction),
        "requested_rows": int(count),
        "actual_modified_rows": int(count),
        "realized_fraction_of_rows": float(count / n),
        "radius": int(radius),
        "realized_exact_radius_fraction": float(realized),
    }


def inject_lag_copy(x: np.ndarray, fraction: float, lag: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    x = _binary(x, "x", x.shape[1])
    n = len(x)
    if not 0 <= fraction <= 1 or not 1 <= lag < n:
        raise ValueError("Invalid lag-copy parameters.")
    y = x.copy()
    count = int(round((n - lag) * fraction))
    starts = rng.choice(np.arange(lag, n), count, replace=False) if count else np.empty(0, dtype=np.int64)
    if count:
        y[starts] = x[starts - lag]
    return y, {
        "requested_fraction": float(fraction),
        "requested_rows": int(count),
        "actual_modified_rows": int(count),
        "realized_fraction_of_eligible_rows": float(count / max(1, n - lag)),
        "lag": int(lag),
    }


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n < 1 or not 0 <= k <= n:
        raise ValueError("Invalid binomial count.")
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + confidence / 2))
    phat = k / n
    den = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def calibrate_fault(*, clean_features, injector, detector, replicates, seed):
    if replicates < CALIBRATION_MIN_REPLICATES:
        raise ValueError(f"Calibration requires at least {CALIBRATION_MIN_REPLICATES} independent trials.")
    outcomes = []
    metadata = []
    ss = np.random.SeedSequence(seed)
    for child in ss.spawn(replicates):
        rng = np.random.default_rng(child)
        features, meta = injector(clean_features, rng)
        result = detector(features, rng)
        outcomes.append(bool(result.get("detected", False) if isinstance(result, Mapping) else result))
        metadata.append(meta)
    k = int(sum(outcomes))
    lo, hi = wilson_interval(k, replicates)
    return {
        "replicates": int(replicates),
        "detections": k,
        "detection_rate": float(k / replicates),
        "detection_rate_95_ci": {"lower": lo, "upper": hi},
        "target": CALIBRATION_DETECTION_TARGET,
        "point_estimate_target_met": bool(k / replicates >= CALIBRATION_DETECTION_TARGET),
        "confidence_lower_bound_target_met": bool(lo >= CALIBRATION_DETECTION_TARGET),
        "trial_semantics": "independent fault-injection trials conditional on one fixed clean dataset instance",
        "injected_fault_metadata_summary": {
            "first_trial": _sanitize_metadata(metadata[0]) if metadata else {}
        },
    }


def _sanitize_metadata(meta):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in meta.items()}


def familywise_test_count(
    *,
    partition_count: int,
    lag_count: int,
    near_duplicate_radius_count: int,
    structured_view_count: int,
    multivariate_count: int,
) -> int:
    if partition_count < 1:
        raise ValueError("partition_count must be >=1")
    comparisons = partition_count + partition_count * (partition_count - 1) // 2
    return max(1, comparisons + lag_count + near_duplicate_radius_count * partition_count + structured_view_count + multivariate_count)


def run_d2(
    *,
    partitions,
    feature_bits,
    reference_pmf=None,
    structured_views=None,
    pairs_per_test=DEFAULT_PAIRS_PER_TEST,
    audit_replicates=DEFAULT_AUDIT_REPLICATES,
    null_permutations=DEFAULT_NULL_PERMUTATIONS,
    bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
    lags=DEFAULT_LAGS,
    near_duplicate_radii=NEAR_DUPLICATE_RADII,
    multivariate_partitions=None,
    multivariate_pairs=DEFAULT_MULTIVARIATE_PAIRS,
    multivariate_permutations=DEFAULT_MULTIVARIATE_PERMUTATIONS,
    audit_seed=0,
    near_duplicate_query_count=2048,
    near_duplicate_projection_tables=8,
    near_duplicate_null_pairs=100_000,
    practical_excess_tvd_threshold=PRACTICAL_EXCESS_TVD_THRESHOLD,
):
    if not partitions or audit_replicates < 1 or null_permutations < 100 or bootstrap_replicates < 100 or pairs_per_test < 1:
        raise ValueError("Invalid D2 configuration.")
    if practical_excess_tvd_threshold < 0 or practical_excess_tvd_threshold > 1:
        raise ValueError("practical_excess_tvd_threshold must be in [0, 1].")
    nominal = validate_reference_pmf(reference_pmf, feature_bits) if reference_pmf is not None else None
    arrays = {name: _binary(x, name, feature_bits) for name, x in partitions.items()}
    structured_views = structured_views or {}
    lags = tuple(int(k) for k in lags)
    radii = tuple(int(r) for r in near_duplicate_radii)
    if any(k < 1 for k in lags) or any(r < 1 or r >= feature_bits for r in radii):
        raise ValueError("Invalid lags or radii.")

    names = list(arrays)
    comparisons = [(f"within:{n}", arrays[n], arrays[n], True) for n in names]
    comparisons += [
        (f"cross:{names[i]}:{names[j]}", arrays[names[i]], arrays[names[j]], False)
        for i in range(len(names)) for j in range(i + 1, len(names))
    ]
    lag_tests = [(f"{n}:lag{k}", x, k) for n, x in arrays.items() for k in lags if k < len(x)]
    selected = tuple(multivariate_partitions or names)
    total = familywise_test_count(
        partition_count=len(arrays),
        lag_count=len(lag_tests),
        near_duplicate_radius_count=len(radii),
        structured_view_count=len(structured_views),
        multivariate_count=len(selected),
    )
    alpha = FAMILYWISE_ALPHA / total
    reference_permutations = min(DEFAULT_NULL_REFERENCE_PERMUTATIONS, max(1, null_permutations - 1))
    max_tail_permutations = null_permutations - reference_permutations
    if 1.0 / (max_tail_permutations + 1) >= alpha:
        raise ValueError(
            "--null-permutations is too small to resolve the corrected per-test alpha: "
            f"effective alpha={alpha:.6g}, maximum tail draws={max_tail_permutations:,}, "
            f"minimum attainable Monte Carlo p={1.0/(max_tail_permutations+1):.6g}. "
            f"Use more than {math.ceil(1.0/alpha) - 1 + reference_permutations:,} total permutations."
        )
    ss = np.random.SeedSequence(audit_seed)
    hamming, lagged, near, mv = {}, {}, {}, {}

    # D2.2 pairwise structure: observed pairs and randomization null are based
    # on the same endpoint pool. Complete rows are permuted; no bitwise null.
    for name, a, b, within in comparisons:
        reps = []
        for child in ss.spawn(audit_replicates):
            rng = np.random.default_rng(child)
            left, right, _, _ = _randomize_pairing(a, b, pairs_per_test, rng, within=within)
            observed = hamming_distances(left, right)
            null_pmf, null_stats, null_meta = _null_pair_histogram(
                left,
                right,
                permutations=null_permutations,
                rng=rng,
                within=within,
                reference_permutations=reference_permutations,
                batch_size=DEFAULT_NULL_BATCH_SIZE,
                stopping_error=DEFAULT_NULL_STOPPING_ERROR,
                alpha=alpha,
                observed_distances=observed,
            )
            reps.append(
                summarize_randomization_test(
                    observed,
                    null_pmf,
                    null_stats,
                    alpha=alpha,
                    rng=rng,
                    bootstrap_replicates=bootstrap_replicates,
                    practical_excess_tvd_threshold=practical_excess_tvd_threshold,
                    null_sampling_metadata=null_meta,
                )
            )
        hamming[name] = _aggregate_randomization_results(
            reps, practical_excess_tvd_threshold=practical_excess_tvd_threshold
        )

    # D2.1: probabilistic near-neighbour detector.  Query batches are pooled
    # only as a sampling budget; they are not treated as independent studies.
    for name, x in arrays.items():
        qbudget = near_duplicate_query_count * audit_replicates
        near[f"within:{name}"] = near_neighbor_summary(
            x,
            radii,
            rng=np.random.default_rng(ss.spawn(1)[0]),
            query_count=qbudget,
            projection_tables=near_duplicate_projection_tables,
            null_pairs=near_duplicate_null_pairs,
            alpha=alpha,
        )

    # Serial structure uses exactly the same complete-row randomization logic.
    for name, x, lag in lag_tests:
        reps = []
        for child in ss.spawn(audit_replicates):
            rng = np.random.default_rng(child)
            i, j = _sample_disjoint_lag_pairs(len(x), lag, pairs_per_test, rng)
            left, right = x[i], x[j]
            observed = hamming_distances(left, right)
            # A lag test has an ordered endpoint set.  Randomly re-pair the two
            # endpoint pools; this preserves every complete row and tests only
            # whether the generation-order pairing carries excess structure.
            null_pmf, null_stats, null_meta = _null_pair_histogram(
                left,
                right,
                permutations=null_permutations,
                rng=rng,
                within=False,
                reference_permutations=reference_permutations,
                batch_size=DEFAULT_NULL_BATCH_SIZE,
                stopping_error=DEFAULT_NULL_STOPPING_ERROR,
                alpha=alpha,
                observed_distances=observed,
            )
            reps.append(
                summarize_randomization_test(
                    observed,
                    null_pmf,
                    null_stats,
                    alpha=alpha,
                    rng=rng,
                    bootstrap_replicates=bootstrap_replicates,
                    practical_excess_tvd_threshold=practical_excess_tvd_threshold,
                    null_sampling_metadata=null_meta,
                )
            )
        lagged[name] = _aggregate_randomization_results(
            reps, practical_excess_tvd_threshold=practical_excess_tvd_threshold
        )

    structured = analyze_structured_views(structured_views, alpha)
    for name in selected:
        mv[name] = multivariate_pair_discrimination(
            arrays[name],
            pairs=multivariate_pairs,
            permutations=multivariate_permutations,
            rng=np.random.default_rng(ss.spawn(1)[0]),
            alpha=alpha,
        )

    failures, warnings = [], []
    for name, result in {**hamming, **lagged}.items():
        practical = not result["practical_pass"]
        statistical = bool(result["statistical_warning"])
        if practical and statistical:
            failures.append({
                "component": name,
                "type": "distributional",
                "reason": "Both the pre-specified practical-effect criterion and the corrected fixed-dataset randomization test indicate excess structure.",
                "excess_tvd_max": result["null_centered_excess_tvd_max"],
                "practical_effect_threshold": result["practical_effect_threshold"],
                "minimum_randomization_p_value": result["minimum_randomization_p_value"],
            })
        elif practical or statistical:
            warnings.append({
                "component": name,
                "type": "distributional",
                "reason": (
                    "The component crossed only one of the two evidence gates (statistical or practical); "
                    "this is treated as inconclusive rather than as evidence of dependence."
                ),
                "practical_effect_excessive": practical,
                "statistically_excessive": statistical,
                "excess_tvd_max": result["null_centered_excess_tvd_max"],
                "practical_effect_threshold": result["practical_effect_threshold"],
                "minimum_randomization_p_value": result["minimum_randomization_p_value"],
            })

    for name, radius_results in near.items():
        for radius, r in radius_results.items():
            practical = bool(r.get("practically_excessive", False))
            statistical = bool(r.get("statistically_excessive", False))
            if practical and statistical:
                failures.append({
                    "component": f"{name}:radius{radius}",
                    "type": "near_duplicate_structure",
                    "reason": "Both practical and statistical gates indicate excess verified near-neighbour structure.",
                    "excess_ratio": r["excess_ratio"],
                    "p_value": r["one_sided_excess_p_value"],
                    "statistical_inference": r.get("statistical_inference"),
                })
            elif practical or statistical:
                warnings.append({
                    "component": f"{name}:radius{radius}",
                    "type": "near_duplicate_structure",
                    "reason": "Only one evidence gate was crossed; result is treated as inconclusive rather than as a confirmed dependence finding.",
                    "practically_excessive": practical,
                    "statistically_excessive": statistical,
                    "excess_ratio": r["excess_ratio"],
                    "p_value": r["one_sided_excess_p_value"],
                    "statistical_inference": r.get("statistical_inference"),
                })

    for name, r in structured.items():
        ref = r.get("reference")
        if not ref:
            warnings.append({"component": name, "type": "structured_collision", "reason": r.get("reason", "Structured view was not tested.")})
            continue
        practical = bool(ref["practically_excessive"])
        statistical = bool(ref["statistically_excessive"])
        if practical and statistical:
            failures.append({"component": name, "type": "structured_collision", "reason": "Both practical and statistical gates indicate excess structured collisions."})
        elif practical or statistical:
            warnings.append({"component": name, "type": "structured_collision", "reason": "Only one evidence gate was crossed; result is treated as inconclusive.", "practically_excessive": practical, "statistically_excessive": statistical})

    for name, r in mv.items():
        practical = bool(r["practically_excessive"])
        statistical = bool(r["statistically_excessive"])
        if practical and statistical:
            failures.append({"component": name, "type": "multivariate_dependence", "reason": "Both practical and permutation-statistical gates indicate pair discrimination beyond the declared tolerance.", "p_value": r["permutation_p_value"], "auc_excess": r["auc_excess_over_chance"]})
        elif practical or statistical:
            warnings.append({"component": name, "type": "multivariate_dependence", "reason": "Only one evidence gate was crossed; result is treated as inconclusive.", "practically_excessive": practical, "statistically_excessive": statistical, "p_value": r["permutation_p_value"], "auc_excess": r["auc_excess_over_chance"]})

    outcome = "FAIL" if failures else ("INCONCLUSIVE" if warnings else "PASS")
    results = {
        "d2_1_near_duplicate_structure": near,
        "d2_2_serial_dependence": lagged,
        "d2_2_pairwise_structure": hamming,
        "d2_3_multivariate_dependence": mv,
        "d2_4_structured_repetition": structured,
        "configuration": {
            "schema_version": SCHEMA_VERSION,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "effective_test_alpha": alpha,
            "familywise_test_count": total,
            "hypothesis_counting": "one hypothesis per reported component-level comparison; sampling replicates are not independent studies and no Fisher combination is used",
            "nominal_hamming_reference": "Binomial(feature_bits, 0.5) may be reported diagnostically only",
            "practical_effect_measure": "null-centered excess TVD = observed TVD - mean TVD under the fixed-dataset randomization null",
            "practical_effect_threshold": float(practical_excess_tvd_threshold),
            "confirmatory_pairwise_null": "fixed-dataset randomization of complete rows; preserves within-row dependencies",
            "confirmatory_serial_null": "fixed-dataset randomization of complete endpoint rows across the observed lag pairing",
            "randomization_tail_sampling": "adaptive fixed-batch Monte Carlo; inferential decisions use simultaneous Clopper-Pearson confidence bounds over predeclared looks; the reported Monte Carlo p-value is descriptive under adaptive stopping",
            "near_duplicate_null": "empirical complete-row resampling; no bit-independence assumption",
            "near_duplicate_inference": "screening-only; no formal p-value because query-hit indicators share a fixed reference set",
            "near_duplicate_method": "random projection candidate retrieval followed by exact Hamming verification; probabilistic and non-exhaustive",
            "pairs_per_test": pairs_per_test,
            "audit_replicates": audit_replicates,
            "null_permutations": null_permutations,
            "null_permutations_semantics": "maximum total randomization draws; reference draws are separate from the confirmatory tail sample",
            "null_reference_permutations": reference_permutations,
            "null_batch_size": DEFAULT_NULL_BATCH_SIZE,
            "null_adaptive_stopping": True,
            "null_stopping_error": DEFAULT_NULL_STOPPING_ERROR,
            "bootstrap_replicates": bootstrap_replicates,
            "lags": list(lags),
            "near_duplicate_radii": list(radii),
            "near_duplicate_query_count": near_duplicate_query_count,
            "near_duplicate_projection_tables": near_duplicate_projection_tables,
            "near_duplicate_null_pairs": near_duplicate_null_pairs,
            "multivariate_pairs": multivariate_pairs,
            "multivariate_permutations": multivariate_permutations,
            "practical_effect_threshold": float(practical_excess_tvd_threshold),
            "audit_seed": audit_seed,
            "reference_pmf_supplied": nominal is not None,
        },
    }
    decision = {
        "outcome": outcome,
        "failures": failures,
        "warnings": warnings,
        "interpretation": (
            "No practically material violation was detected within the declared tests and scope."
            if outcome == "PASS"
            else "Decision is limited to tested mechanisms, null models, thresholds, detector class, and demonstrated calibration envelope."
        ),
        "not_proof_of_independence": True,
    }
    return results, decision


def build_d2_certificate(
    *,
    results,
    decision,
    partitions,
    dataset_id,
    dataset_version,
    generation_procedure,
    generation_parameters,
    generation_random_seed,
    reference_description,
    reference_model_description,
    audit_seed,
    output_path,
):
    provenance = build_provenance(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        generation_procedure=generation_procedure,
        generation_parameters=generation_parameters,
        random_seed=generation_random_seed,
        partitions={
            n: {
                "sample_count": int(x.shape[0]),
                "feature_count": int(x.shape[1]),
                "dtype": str(x.dtype),
                "shape": list(x.shape),
                "sha256": array_sha256(x),
            }
            for n, x in partitions.items()
        },
        audit_configuration={
            **dict(results["configuration"]),
            "nominal_reference_distribution": reference_description,
            "structured_reference_model": reference_model_description,
        },
    )
    cert = make_certificate(
        audit_id=AUDIT_ID,
        audit_name=AUDIT_NAME,
        claim="The supplied dataset instance was audited for specified near-duplicate, pairwise, serial, multivariate, and structured repetition mechanisms.",
        outcome=str(decision["outcome"]),
        findings={"results": dict(results), "decision": dict(decision)},
        methodology={
            "scope": "Dataset Integrity D2 Sample Dependence Audit",
            "d1_boundary": "D1 owns exact duplication and exact partition overlap.",
            "d2_1": "Sampled-query near-neighbour screen with exact Hamming verification; the null is empirical complete-row resampling, and candidate retrieval sensitivity is not assumed exhaustive.",
            "d2_2_pairwise": "Complete-row fixed-dataset randomization test; no bitwise independence assumption and no p-value combination across sampling replicates. Practical magnitude is assessed using null-centered excess TVD rather than raw finite-sample TVD.",
            "d2_2_serial": "Generation-order lag pairs are compared with random re-pairings of the same complete endpoint rows. Practical magnitude is assessed using null-centered excess TVD rather than raw finite-sample TVD.",
            "d2_3": "Logistic detector with row-disjoint train/test pools and held-out label permutation.",
            "d2_4": "Structured collision tests require an explicit adapter-supplied finite-domain null; generic D2 does not assume uniformity.",
            "d2_5": "Controlled fault injection with Wilson intervals; calibration is sensitivity evidence, not clean-data evidence.",
            "multiple_comparison_control": "Bonferroni FWER 0.01 across declared hypothesis-level components; sampling replicates are not counted as separate hypotheses.",
            "decision_semantics": "Not a proof of universal mutual independence. A component is a D2 failure only when both statistical and practical evidence gates are crossed; a one-gate result is INCONCLUSIVE.",
            "audit_seed_semantics": "Controls audit sampling/randomization only unless the adapter explicitly supplies deterministic generation.",
            "audit_seed": audit_seed,
        },
        limitations=[
            "Finite testing cannot establish universal mutual independence.",
            "D2.1 near-neighbour retrieval is probabilistic and non-exhaustive; exact verification prevents approximate-distance false positives but candidate retrieval can miss neighbours.",
            "D2.1 sensitivity is established only for the explicitly calibrated fault classes and strengths.",
            "Fixed-dataset randomization tests are conditional on the audited dataset instance and test pairing/ordering structure, not a universal population theorem.",
            "D2.1 uses an empirical complete-row null estimated by Monte Carlo; the near-neighbour query-probability transform is an approximation for the finite population.",
            "Structured collision tests are valid only where the adapter's declared finite-domain null is justified.",
            "Calibration trials are independent fault-injection trials on one fixed clean dataset unless the driver explicitly regenerates datasets.",
            "The audit does not establish reproducible dataset generation when the underlying generator is OS-randomized unless raw randomness or an exact replay mechanism is archived.",
        ],
        evidence_level="DATASET_INTEGRITY_D2_SAMPLE_DEPENDENCE",
        provenance=provenance,
        certificate_version=SCHEMA_VERSION,
    )
    write_certificate(cert, output_path)
    return cert


def print_report(results, certificate):
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
    print("Null architecture    : complete-row randomization/resampling; no bit-independence assumption")
    print("=" * 78)
