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
SCHEMA_VERSION = "7.0"
FAMILYWISE_ALPHA = 0.01
PRACTICAL_EXCESS_TVD_THRESHOLD = 0.01
MULTIVARIATE_AUC_TOLERANCE = 0.05
NEAR_DUPLICATE_RADII = (1, 2, 4, 8)
DEFAULT_LAGS = tuple(list(range(1, 17)) + [32, 64, 128, 256, 512, 1024])
DEFAULT_PAIRS_PER_TEST = 20_000
DEFAULT_AUDIT_REPLICATES = 10
#: Raised from 120_000 at schema 6.5. The statistical gate fires on a
#: simultaneous Clopper-Pearson upper bound, whose attainable minimum at
#: 120,000 draws is 1.1161e-04. Both production configurations run at a
#: corrected alpha below that (Gohr 9.804e-05 at a family of 102, AutoND
#: analysis A 9.524e-05 at 105), so at the old default neither could reject a
#: distributional null however large the effect. The worst case across the
#: declared configurations requires 142,419; this clears it with margin.
#: Use required_permutations_for_alpha() to check any other configuration.
DEFAULT_NULL_PERMUTATIONS = 150_000
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


def minimum_attainable_decision_bound(
    total_permutations: int,
    *,
    reference_permutations: int | None = None,
    batch_size: int = DEFAULT_NULL_BATCH_SIZE,
    stopping_error: float = DEFAULT_NULL_STOPPING_ERROR,
) -> dict[str, Any]:
    """Smallest ``upper_p`` the statistical gate can ever attain.

    The statistical gate in ``_null_pair_histogram`` does NOT fire on the
    Monte Carlo point estimate. It fires when a simultaneous
    Clopper--Pearson upper bound, Bonferroni-corrected over the predeclared
    batch looks, falls below alpha. With zero exceedances after ``b`` tail
    draws that bound is far above ``1/(b+1)``, so a budget that resolves the
    point estimate can still leave the gate unreachable: the component then
    cannot reject however large the effect, and a maximal violation is
    silently downgraded to INCONCLUSIVE.

    Measured consequence of getting this wrong: with every second row an
    exact copy of its predecessor, the lag-1 test produced all-zero observed
    distances, TVD 0.996 and excess TVD 0.895 against a 0.01 threshold, and
    still reported ``statistically_excessive: False`` because the upper bound
    stalled at 3.3e-3 against alpha 7.7e-4.

    Returns the bound and its inputs so a preflight can report them.
    """
    from scipy.stats import beta as beta_dist

    total = int(total_permutations)
    ref = (min(DEFAULT_NULL_REFERENCE_PERMUTATIONS, max(1, total - 1))
           if reference_permutations is None else int(reference_permutations))
    tail = total - ref
    if tail < 1:
        raise ValueError("total_permutations must exceed reference_permutations.")
    max_looks = int(math.ceil(tail / batch_size))
    look_error = stopping_error / max_looks
    # x = 0 exceedances, b = the full tail budget: the most favourable case.
    upper_q = float(beta_dist.ppf(1.0 - look_error / 2.0, 1, tail))
    upper_p = float((1.0 + tail * upper_q) / (tail + 1.0))
    return {
        "total_permutations": total,
        "reference_permutations": ref,
        "max_tail_draws": tail,
        "batch_size": int(batch_size),
        "max_looks": max_looks,
        "per_look_error": look_error,
        "minimum_attainable_upper_p": upper_p,
        "minimum_attainable_monte_carlo_p": 1.0 / (tail + 1.0),
        "gate_semantics": (
            "the statistical gate requires minimum_attainable_upper_p < alpha; "
            "minimum_attainable_monte_carlo_p is NOT the operative quantity"),
    }


def required_permutations_for_alpha(
    alpha: float,
    *,
    batch_size: int = DEFAULT_NULL_BATCH_SIZE,
    stopping_error: float = DEFAULT_NULL_STOPPING_ERROR,
    reference_permutations: int = DEFAULT_NULL_REFERENCE_PERMUTATIONS,
    maximum: int = 100_000_000,
) -> int:
    """Smallest total permutation budget whose statistical gate can fire."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")
    lo, hi = 2, maximum
    def ok(tail: int) -> bool:
        b = minimum_attainable_decision_bound(
            tail + reference_permutations,
            reference_permutations=reference_permutations,
            batch_size=batch_size, stopping_error=stopping_error)
        return b["minimum_attainable_upper_p"] < alpha
    if not ok(hi):
        raise ValueError(
            f"alpha={alpha:.6g} is not attainable below {maximum:,} permutations.")
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo + reference_permutations


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

    The complete binary rows are preserved, so within-row cryptographic
    structure is retained.

    CRITICAL SEMANTICS — read before using this for inference.
    ----------------------------------------------------------
    For ``within=True`` the observed endpoints are drawn as one random
    ``2*count`` subset split into two halves, so the observed pairing is an
    ARBITRARY matching with no relation to generation order. The matching
    randomization null in ``_null_pair_histogram`` re-splits that same row
    pool, so the observed pairing is itself one draw from the null
    distribution. The null is therefore true by construction: the p-value is
    uniform under EVERY alternative and the test has no power against row
    dependence. ``within=False`` has the same defect, because ``i`` and ``j``
    are sampled independently and then paired in arbitrary order.

    Measured on 40 trials at alpha=0.05 with half of the dataset an exact copy
    of its adjacent row: within rejected 1/40 and cross 2/40, against 40/40
    for the lag-1 test on the same data.

    A valid randomization test of row dependence needs an observed pairing
    defined by the SOURCE. The only such pairing available here is generation
    adjacency, which is what ``_sample_disjoint_lag_pairs`` supplies and what
    the serial-dependence tests use. Results from this function for
    ``within:*`` and ``cross:*`` components are consequently recorded as
    DESCRIPTIVE ONLY and excluded from the D2 decision gates by ``run_d2``.
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


# ---------------------------------------------------------------------------
# D2.1 near-duplicate screening (schema 7.0)
#
# WHY THIS CHANGED (schema 7.0).  The schema 6.5 implementation
# ``near_neighbor_summary`` is retained below for version-aware reading of
# existing certificates but is NO LONGER USED FOR INFERENCE.  Four defects
# were diagnosed in it; see ``D21_SUPERSEDED_METHOD_DEFECTS``.  The dominant
# one: its reference quantity was a Monte Carlo estimate of a probability
# between 1e-18 and 1e-10, drawn with 1e5 samples, so it evaluated to exactly
# 0.0 across the whole operating regime.  The divisor was therefore always
# zero and the reported outcome was decided solely by whether the numerator
# happened to be zero -- the zero case being hardcoded to ``ratio = 1.0`` with
# status ``FINITE``, which serialises a vacuous result as a favourable one.
#
# The replacement below removes the defects at the source rather than
# calibrating around them:
#   * retrieval is EXHAUSTIVE (exact distance to every other row), so recall
#     is 1.0 by construction and there is no approximate-retrieval stage to
#     calibrate;
#   * the null is a DECLARED model evaluated in CLOSED FORM, so no Monte
#     Carlo resolution limit exists and the divisor is provably positive;
#   * the statistical gate is a Clopper--Pearson bound that can actually
#     fire, and the attainable sensitivity is reported so an underpowered
#     configuration is visible before the run rather than after.
# ---------------------------------------------------------------------------

#: Row chunk for the exact scan, in rows.  Chosen by measurement at n = 2e6,
#: w = 64: 32,768-row chunks with a 64-query block run 10.3 ms/query against
#: 25.0 ms/query at 1e6-row chunks, because a chunk this size stays
#: cache-resident while it is reused across the query block.
DEFAULT_EXACT_SCAN_CHUNK = 32_768
#: Queries processed against each resident row chunk before moving on.
DEFAULT_EXACT_SCAN_QUERY_BLOCK = 64
DEFAULT_NEAR_DUPLICATE_QUERIES = 2048

D21_SUPERSEDED_METHOD_DEFECTS = (
    "B: the declared reference probability was estimated by Monte Carlo with "
    "null_pairs draws of a probability between 1e-18 and 1e-10, so it "
    "evaluated to exactly 0.0; expected became 0 and the zero-hit branch "
    "hardcoded ratio=1.0 with status FINITE, serialising a vacuous result as "
    "a favourable one, while any non-zero count yielded +inf and fired the "
    "practical gate unconditionally.",
    "C: statistically_excessive was the literal False, so under the two-gate "
    "rule the component could never contribute a FAIL at any effect size.",
    "D: the reference distribution was resampled from the dataset under "
    "audit, so it preserved the very near-duplicates the component exists to "
    "detect; increasing the sample count cannot repair this.",
    "A: the numerator counted neighbours among LSH-retrieved candidates while "
    "the denominator reasoned over all n-1 rows, bounding the ratio above by "
    "retrieval recall.  Measured recall was 0.97-1.00 at the tested settings, "
    "so this defect was real in kind but minor in magnitude.",
    "E: buckets above max_candidates_per_bucket were reduced by np.linspace "
    "over the signature-sorted order -- a deterministic, position-dependent "
    "subsample that varied recall with sort position.",
)

_SWAR_M1 = np.uint64(0x5555555555555555)
_SWAR_M2 = np.uint64(0x3333333333333333)
_SWAR_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_SWAR_H = np.uint64(0x0101010101010101)


def _popcount_u64(x: np.ndarray) -> np.ndarray:
    """SWAR population count.  numpy.bitwise_count requires NumPy >= 2.0 and
    this engine is pinned to 1.26.4, so the standard 5-step reduction is used.
    """
    x = x - ((x >> np.uint64(1)) & _SWAR_M1)
    x = (x & _SWAR_M2) + ((x >> np.uint64(2)) & _SWAR_M2)
    x = (x + (x >> np.uint64(4))) & _SWAR_M4
    return (x * _SWAR_H) >> np.uint64(56)


def exact_min_distances(
    x: np.ndarray,
    query_indices: np.ndarray,
    *,
    chunk_rows: int = DEFAULT_EXACT_SCAN_CHUNK,
    query_block: int = DEFAULT_EXACT_SCAN_QUERY_BLOCK,
) -> np.ndarray:
    """Exact minimum Hamming distance from each query row to every other row.

    Exhaustive: for query ``q`` the returned value is
    ``min over i != q of d(x[q], x[i])`` computed over the complete row
    population, with no retrieval, filtering or sampling stage.  Recall is
    therefore 1.0 by construction.

    The loops are ordered rows-outermost so that a row chunk small enough to
    stay in cache is reused across a block of queries.  That ordering is a
    performance property only and cannot change the result: each (query, row)
    distance is computed exactly once either way.  Measured at n = 2e6,
    w = 64: 10.3 ms/query at ``chunk_rows=32768, query_block=64`` against
    25.0 ms/query at ``chunk_rows=1000000``.

    Returns an ``int16`` array of length ``len(query_indices)``.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must be a 2-D binary row matrix.")
    n = len(x)
    qidx = np.asarray(query_indices, dtype=np.int64)
    if qidx.size == 0:
        return np.empty(0, dtype=np.int16)
    if qidx.min() < 0 or qidx.max() >= n:
        raise ValueError("query_indices out of range.")
    if n < 2:
        raise ValueError("At least two rows are required.")
    chunk = max(1, int(chunk_rows))
    qblock = max(1, int(query_block))
    sentinel = np.int64(np.iinfo(np.int16).max)

    packed = _pack_binary_rows(x)
    nbytes = packed.shape[1]
    # The uint64 view is a representation change only; it requires a byte
    # width that is a multiple of 8 and a C-contiguous buffer.
    use_u64 = (nbytes % 8 == 0) and packed.flags["C_CONTIGUOUS"]
    words = packed.view(np.uint64).reshape(n, nbytes // 8) if use_u64 else None

    best = np.full(len(qidx), sentinel, dtype=np.int64)
    for qs in range(0, len(qidx), qblock):
        qe = min(qs + qblock, len(qidx))
        block_q = qidx[qs:qe]
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            blk_u64 = words[start:stop] if use_u64 else None
            blk_u8 = packed[start:stop]
            for t in range(qe - qs):
                if best[qs + t] == 0:
                    continue          # 0 is the attainable minimum
                qi = int(block_q[t])
                if use_u64:
                    acc = _popcount_u64(blk_u64[:, 0] ^ words[qi, 0])
                    for c in range(1, words.shape[1]):
                        acc = acc + _popcount_u64(blk_u64[:, c] ^ words[qi, c])
                    d = acc.astype(np.int64, copy=False)
                else:
                    d = _POPCOUNT8[blk_u8 ^ packed[qi]].sum(axis=1, dtype=np.int64)
                if start <= qi < stop:
                    # Exclude the query's own row; every other entry is
                    # untouched, and d is a freshly allocated array.
                    d[qi - start] = sentinel
                m = np.int64(d.min())
                if m < best[qs + t]:
                    best[qs + t] = m
    return best.astype(np.int16)


def exact_duplicate_pair_count(x: np.ndarray) -> dict[str, Any]:
    """Exhaustive count of exactly-identical row pairs, by value sorting.

    Cheap (one sort) and exact, so it is always computed.  It cross-checks the
    D1 duplicate count without re-implementing D1's algorithm.
    """
    packed = _pack_binary_rows(x)
    view = np.ascontiguousarray(packed).view(
        np.dtype((np.void, packed.shape[1] * packed.dtype.itemsize))
    ).ravel()
    _, counts = np.unique(view, return_counts=True)
    multiplicities = counts[counts > 1]
    pairs = int((multiplicities * (multiplicities - 1) // 2).sum())
    return {
        "rows": int(len(x)),
        "distinct_rows": int(len(counts)),
        "duplicated_row_values": int(len(multiplicities)),
        "exact_duplicate_pairs": pairs,
        "largest_multiplicity": int(multiplicities.max()) if len(multiplicities) else 1,
        "method": "exhaustive value sort; exact, not sampled",
    }


def declared_pair_probability(
    n_bits: int,
    radius: int,
    *,
    bit_probabilities: Sequence[float] | None = None,
    query_row: np.ndarray | None = None,
) -> float:
    """P(d(X, X') <= radius) under a DECLARED null, in closed form.

    No Monte Carlo is used anywhere in this function, so there is no
    resolution limit and the result is provably positive for radius >= 0.
    This is the fix for the dominant defect of schema 6.5, whose Monte Carlo
    estimate of this quantity evaluated to exactly 0.0.

    ``bit_probabilities is None`` selects the uniform model, for which the
    probability is the exact rational ``sum_{k<=r} C(w,k) / 2^w`` and does not
    depend on the query row.

    Otherwise a per-bit Bernoulli product model is used.  There the mismatch
    probability depends on the query row's bits, so ``query_row`` is required;
    the exact Poisson-binomial CDF over the per-position mismatch
    probabilities is evaluated by dynamic programming.
    """
    w = int(n_bits)
    r = int(radius)
    if w < 1:
        raise ValueError("n_bits must be positive.")
    if not 0 <= r <= w:
        raise ValueError(f"radius {r} outside [0, {w}].")

    if bit_probabilities is None:
        # Integer numerator divided once: full double precision, no
        # accumulation of rounding across terms.
        num = sum(math.comb(w, k) for k in range(r + 1))
        return num / float(2 ** w)

    p = np.asarray(bit_probabilities, dtype=float)
    if p.shape != (w,):
        raise ValueError(f"bit_probabilities must have shape ({w},).")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("bit_probabilities must lie in [0,1].")
    if query_row is None:
        raise ValueError(
            "query_row is required for a bernoulli_product null: the mismatch "
            "probability depends on the query row's bit values.")
    q = np.asarray(query_row, dtype=np.uint8).ravel()
    if q.shape != (w,):
        raise ValueError(f"query_row must have shape ({w},).")
    if not np.all((q == 0) | (q == 1)):
        raise ValueError("query_row must be binary.")
    mismatch = np.where(q == 0, p, 1.0 - p)

    # Exact Poisson-binomial DP over w positions.
    pmf = np.zeros(w + 1, dtype=float)
    pmf[0] = 1.0
    for m in mismatch:
        pmf[1:] = pmf[1:] * (1.0 - m) + pmf[:-1] * m
        pmf[0] = pmf[0] * (1.0 - m)
    return float(pmf[: r + 1].sum())


def _clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper--Pearson lower confidence bound on a proportion."""
    if k <= 0:
        return 0.0
    from scipy.stats import beta as beta_dist
    return float(beta_dist.ppf(alpha, k, n - k + 1))


def minimum_detectable_near_duplicate_ratio(
    *,
    queries: int,
    null_query_probability: float,
    alpha: float,
) -> dict[str, Any]:
    """Smallest excess ratio at which the statistical gate can fire.

    The analogue of ``minimum_attainable_decision_bound`` for D2.1.  The
    statistical gate requires the Clopper--Pearson LOWER bound on the observed
    query rate to exceed the declared null probability.  With a small query
    budget that can be unreachable at any effect size, which is exactly the
    pathology that made three other D2 components vacuous at schema 6.5.  This
    is reported alongside every result so an underpowered configuration is
    visible before the run.
    """
    m = int(queries)
    q = float(null_query_probability)
    if m < 1:
        raise ValueError("queries must be positive.")
    if not 0.0 < q < 1.0:
        raise ValueError("null_query_probability must lie in (0,1).")
    for k in range(1, m + 1):
        if _clopper_pearson_lower(k, m, alpha) > q:
            return {
                "queries": m,
                "alpha": float(alpha),
                "null_query_probability": q,
                "minimum_rejecting_hits": k,
                "minimum_detectable_observed_rate": k / m,
                "minimum_detectable_excess_ratio": (k / m) / q,
                "attainable": True,
            }
    return {
        "queries": m,
        "alpha": float(alpha),
        "null_query_probability": q,
        "minimum_rejecting_hits": None,
        "minimum_detectable_observed_rate": None,
        "minimum_detectable_excess_ratio": float("inf"),
        "attainable": False,
    }


#: Below this expected null hit count a radius is in the DETECTION regime:
#: an excess RATIO is not a stable summary there, because the reference count
#: is far below one and a single observed hit produces an arbitrarily large
#: ratio.  The meaningful question becomes presence, not proportional excess.
DETECTION_REGIME_EXPECTED_HITS = 1.0


def near_duplicate_regime(*, queries: int, null_query_probability: float) -> str:
    """DETECTION or ESTIMATION, from the expected null hit count."""
    expected = int(queries) * float(null_query_probability)
    return "DETECTION" if expected < DETECTION_REGIME_EXPECTED_HITS else "ESTIMATION"


def required_queries_for_power(
    target_ratio: float,
    *,
    null_query_probability: float,
    alpha: float,
    power: float = 0.80,
    maximum: int = 1_000_000_000,
) -> dict[str, Any]:
    """Query budget giving ``power`` against a ``target_ratio`` excess.

    REPLACES an earlier helper of mine that asked the wrong question.  That
    one tested whether the gate would reject IF the observed count equalled
    the alternative's expected count, which is vacuous when that expected
    count is far below one.  Measured consequence: at n = 1e7, r = 4,
    m = 6,000 it reported the 1.25x target "attainable" while the true power
    against a 1.25x alternative was 0.276 per cent -- because rejection
    needed one hit and the alternative produced one hit only 0.28 per cent of
    the time.  Attainability is not power, and only power belongs in a
    preflight.

    Here the critical count is the smallest ``k`` with
    ``P(Bin(m, q) >= k) < alpha``, and the power is ``P(Bin(m, t*q) >= k)``
    evaluated exactly.  Both are monotone in ``m``, so a ladder scan with a
    refinement finds the smallest adequate budget.

    A budget of ``None`` means the target is not reachable below ``maximum``:
    at radii where ``q`` is minute, a proportional excess over an already
    negligible baseline needs more queries than the dataset has rows, and the
    radius should be read as a DETECTION test instead.
    """
    from scipy.stats import binom as _binom

    q = float(null_query_probability)
    t = float(target_ratio)
    pw = float(power)
    if not 0.0 < q < 1.0:
        raise ValueError("null_query_probability must lie in (0,1).")
    if t <= 1.0:
        raise ValueError("target_ratio must exceed 1.")
    if not 0.0 < pw < 1.0:
        raise ValueError("power must lie in (0,1).")
    if t * q >= 1.0:
        raise ValueError("target_ratio implies a rate at or above 1.")

    def critical_k(m: int) -> int | None:
        # Smallest k whose upper-tail null probability falls below alpha.
        hi = max(2, int(m * q + 10.0 * math.sqrt(max(m * q, 1.0)) + 20))
        for k in range(1, min(m, hi) + 1):
            if float(_binom.sf(k - 1, m, q)) < alpha:
                return k
        return None

    def achieved_power(m: int) -> float:
        k = critical_k(m)
        if k is None:
            return 0.0
        return float(_binom.sf(k - 1, m, t * q))

    ladder, m = [], 1
    cap = int(maximum)
    while m <= cap:
        ladder.append(m)
        m = max(m + 1, int(m * 1.3))
    if ladder[-1] != cap:
        ladder.append(cap)

    found = None
    for i, cand in enumerate(ladder):
        if achieved_power(cand) >= pw:
            lo = ladder[i - 1] if i else 1
            hi = cand
            while lo < hi:                       # power is monotone in m
                mid = (lo + hi) // 2
                if achieved_power(mid) >= pw:
                    hi = mid
                else:
                    lo = mid + 1
            found = lo
            break

    base = {
        "target_ratio": t,
        "null_query_probability": q,
        "alpha": float(alpha),
        "target_power": pw,
        "maximum_considered": cap,
        "criterion": (
            "smallest m such that P(Bin(m, target_ratio*q) >= k_crit) >= "
            "target_power, where k_crit is the smallest k with "
            "P(Bin(m, q) >= k) < alpha"),
    }
    if found is None:
        return dict(base, required_queries=None, attainable_below_maximum=False,
                    critical_hits=None, achieved_power=achieved_power(cap),
                    note=("not reachable below maximum; at this radius a "
                          "proportional excess over a negligible baseline is "
                          "not estimable and the radius should be read as a "
                          "DETECTION test"))
    return dict(base, required_queries=int(found), attainable_below_maximum=True,
                critical_hits=critical_k(found),
                achieved_power=achieved_power(found),
                expected_null_hits=float(found) * q,
                regime=near_duplicate_regime(
                    queries=found, null_query_probability=q))


def reevaluate_legacy_d21_record(
    record: Mapping[str, Any],
    *,
    rows: int,
    bits: int,
    radius: int,
    alpha: float = FAMILYWISE_ALPHA,
    radius_family_size: int = 1,
    practical_relative_excess: float = 0.25,
    bit_probabilities: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Re-evaluate one schema <= 6.5 D2.1 record against the declared null.

    WHY THIS EXISTS.  The superseded method's defect was in its REFERENCE
    quantity, not in its observed count: hit counts were produced by exact
    Hamming verification of retrieved candidates, so a recorded hit is a real
    verified neighbour.  The corrected denominator is a closed-form function
    of ``(rows, bits, radius)`` alone.  An existing certificate can therefore
    be re-evaluated from its recorded counts WITHOUT rerunning the audit.

    WHAT CANNOT BE RECOVERED.  Retrieval was not exhaustive, so the recorded
    count is a LOWER bound on the exhaustive count: ``observed_exact >=
    observed_legacy``.  Consequently the recomputed ratio is a lower bound on
    the corrected ratio, and the re-evaluation is one-sided:

      * if the lower bound already crosses both gates, EXCESS_DETECTED is
        sound, because the exact value can only be larger;
      * if it does not, the result is INCONCLUSIVE_LOWER_BOUND.  It is NOT
        evidence of absence, because the missed neighbours are unquantified
        in the existing record.

    Establishing absence requires rerunning that partition with
    ``near_neighbor_exact_screen``.  This function never reports a pass.
    """
    n, w, r = int(rows), int(bits), int(radius)
    if n < 2 or w < 1 or not 0 <= r < w:
        raise ValueError("Invalid rows/bits/radius for re-evaluation.")

    queried = record.get("queried_rows")
    observed = record.get("rows_with_verified_neighbor_within_radius")
    missing = [k for k, v in (("queried_rows", queried),
                              ("rows_with_verified_neighbor_within_radius", observed))
               if v is None]
    if missing:
        return {
            "radius": r,
            "status": "UNVERIFIED",
            "reason": f"record lacks required field(s): {', '.join(missing)}",
            "recomputed": None,
        }
    m, k = int(queried), int(observed)
    if not 0 <= k <= m:
        return {"radius": r, "status": "UNVERIFIED",
                "reason": f"implausible counts: {k} hits of {m} queries",
                "recomputed": None}

    p_r = declared_pair_probability(
        w, r, bit_probabilities=bit_probabilities,
        query_row=None if bit_probabilities is None else np.zeros(w, dtype=np.uint8))
    q_r = float(-math.expm1((n - 1) * math.log1p(-p_r)))
    per_radius_alpha = float(alpha) / max(1, int(radius_family_size))

    if not q_r > 0.0:
        return {"radius": r, "status": "NULL_PROBABILITY_UNDERFLOW",
                "declared_pair_probability": p_r, "recomputed": None}

    ratio_lb = (k / m) / q_r
    cp_lower = _clopper_pearson_lower(k, m, per_radius_alpha)
    power = minimum_detectable_near_duplicate_ratio(
        queries=m, null_query_probability=q_r, alpha=per_radius_alpha)
    stat = bool(cp_lower > q_r)
    prac = bool(ratio_lb > 1.0 + practical_relative_excess)
    status = "EXCESS_DETECTED" if (stat and prac) else "INCONCLUSIVE_LOWER_BOUND"

    return {
        "radius": r,
        "status": status,
        "one_sided": True,
        "recomputed": {
            "rows": n,
            "bits": w,
            "queried_rows": m,
            "observed_hits_recorded": k,
            "observed_hits_semantics": (
                "lower bound on the exhaustive count; retrieval in the "
                "superseded method was not exhaustive"),
            "declared_pair_probability": p_r,
            "declared_query_probability": q_r,
            "expected_query_hits": m * q_r,
            "excess_ratio_lower_bound": ratio_lb,
            "clopper_pearson_lower_bound": cp_lower,
            "per_radius_alpha": per_radius_alpha,
            "statistically_excessive": stat,
            "practically_excessive": prac,
            "power": power,
        },
        "legacy_as_recorded": {
            "excess_ratio": record.get("excess_ratio"),
            "excess_ratio_status": record.get("excess_ratio_status"),
            "null_pair_probability": record.get("null_pair_probability"),
            "null_pair_simulation_pairs": record.get("null_pair_simulation_pairs"),
            "practically_excessive": record.get("practically_excessive"),
            "statistically_excessive": record.get("statistically_excessive"),
        },
        "interpretation": (
            "EXCESS_DETECTED is sound here because the recomputed ratio is a "
            "lower bound and both gates already cross. "
            if status == "EXCESS_DETECTED" else
            "NOT evidence of absence: the recorded count is a lower bound on "
            "the exhaustive count, so a true excess may be present but "
            "unrecorded. Rerun this partition with near_neighbor_exact_screen "
            "to establish absence."),
    }


def near_neighbor_exact_screen(
    x: np.ndarray,
    radii: Sequence[int],
    *,
    rng: np.random.Generator,
    null_model: Mapping[str, Any],
    query_count: int | Mapping[int, int] = DEFAULT_NEAR_DUPLICATE_QUERIES,
    alpha: float = FAMILYWISE_ALPHA,
    practical_relative_excess: float = 0.25,
    chunk_rows: int = DEFAULT_EXACT_SCAN_CHUNK,
    query_block: int = DEFAULT_EXACT_SCAN_QUERY_BLOCK,
) -> dict[str, Any]:
    """Exact near-duplicate screen against a declared closed-form null.

    ESTIMAND.  For each radius r: the probability that a row of the dataset
    has at least one other row within Hamming distance r.  Estimated by
    sampling ``query_count`` rows without replacement and computing, for each,
    the EXACT minimum distance to every other row in the dataset.  Numerator
    and denominator therefore refer to the same population -- all n-1 other
    rows -- which the schema 6.5 implementation did not.

    NULL.  ``null_model`` is required and has no default: the reference
    distribution must be declared, never fitted to the data under audit.
      ``{"kind": "uniform"}``
      ``{"kind": "bernoulli_product", "bit_probabilities": [...]}``
    Under a declared i.i.d. row model, conditioning on a query row leaves the
    other n-1 rows independent, so
    ``q_r = 1 - (1 - p_r)**(n-1)``
    is EXACT rather than an approximation.

    DEPENDENCE ACROSS QUERIES -- DOCUMENTED APPROXIMATION.  Distinct queries
    share the same reference rows, so their hit indicators are weakly
    positively associated and the Clopper--Pearson bound is not exactly
    conservative.  This is an approximation, stated here rather than assumed
    away, and the realized false-positive rate is measured empirically on null
    controls instead of being asserted.  Consult the calibration record before
    relying on the statistical gate.

    PER-RADIUS QUERY BUDGETS.  ``query_count`` may be a single integer or a
    mapping from radius to budget, because NO SINGLE POOLED BUDGET IS
    ADEQUATE ACROSS A RADIUS FAMILY.  The statistical gate is the exact
    binomial tail test (the Clopper--Pearson bound is its inversion), and its
    attainability is not monotone in the budget: where the declared null
    probability is tiny the gate rejects on one observed hit, and one hit in
    many queries implies a lower rate than one hit in few.  Measured at
    n = 1e7, alpha = 0.0025, target 1.25x excess, the budgets that can reject
    are r=1 any, r=2 up to 2,180,065, r=4 only up to 6,799, and r=8 at least
    47,238.  The windows for r=4 and r=8 are DISJOINT, so a pooled budget
    such as 20,480 sits above r=4's ceiling and below r=8's floor and can
    reject at neither.

    A mapping is served by one scan, not several: ``max(budgets)`` query rows
    are drawn once without replacement and their exact minimum distances are
    computed once, then radius r is evaluated on the FIRST ``m_r`` of them.
    A prefix of a without-replacement sample drawn in random order is itself
    a uniform without-replacement sample, so each radius gets a valid sample
    of its own size at no additional scanning cost.
    """
    x = _binary(x, "x", x.shape[1])
    n, w = x.shape
    radii = tuple(sorted(set(int(r) for r in radii)))
    if not radii:
        raise ValueError("At least one radius is required.")
    if any(r < 0 or r >= w for r in radii):
        raise ValueError(f"Radii must lie in [0, {w}).")
    if n < 2:
        raise ValueError("At least two rows are required.")
    if isinstance(query_count, Mapping):
        missing = [r for r in radii if r not in query_count]
        if missing:
            raise ValueError(
                f"query_count mapping lacks budgets for radii {missing}.")
        budgets = {r: min(int(query_count[r]), n) for r in radii}
    else:
        budgets = {r: min(int(query_count), n) for r in radii}
    if any(v < 1 for v in budgets.values()):
        raise ValueError("every query budget must be positive.")
    m = max(budgets.values())

    kind = str(null_model.get("kind", "")).lower()
    if kind not in {"uniform", "bernoulli_product"}:
        raise ValueError(
            "null_model['kind'] must be 'uniform' or 'bernoulli_product'; the "
            "null must be declared, not inferred from the data under audit.")
    bit_probabilities = None
    if kind == "bernoulli_product":
        if "bit_probabilities" not in null_model:
            raise ValueError("bernoulli_product null requires bit_probabilities.")
        bit_probabilities = np.asarray(null_model["bit_probabilities"], dtype=float)
        if bit_probabilities.shape != (w,):
            raise ValueError(f"bit_probabilities must have shape ({w},).")

    qidx = rng.choice(n, m, replace=False).astype(np.int64)
    min_d = exact_min_distances(
        x, qidx, chunk_rows=chunk_rows, query_block=query_block)

    family = len(radii)
    per_radius_alpha = float(alpha) / family

    out: dict[str, Any] = {}
    for r in radii:
        # A prefix of a without-replacement sample drawn in random order is
        # itself a uniform without-replacement sample of that size.
        m_r = budgets[r]
        prefix = min_d[:m_r]
        qidx_r = qidx[:m_r]
        hits = int(np.count_nonzero(prefix <= r))
        if bit_probabilities is None:
            p_r = declared_pair_probability(w, r)
            q_r = float(-math.expm1((n - 1) * math.log1p(-p_r)))
            p_report: Any = p_r
            q_report: Any = q_r
            expected = m_r * q_r
            per_query_q = None
        else:
            # The mismatch probability depends on the query row, so the null
            # is evaluated per query and summed.  This conditions on the
            # observed query rows; it does not fit the null to the data.
            qs = np.array([
                float(-math.expm1((n - 1) * math.log1p(
                    -declared_pair_probability(
                        w, r, bit_probabilities=bit_probabilities,
                        query_row=x[qi]))))
                for qi in qidx_r
            ], dtype=float)
            per_query_q = qs
            q_r = float(qs.mean())
            p_report = None
            q_report = q_r
            expected = float(qs.sum())

        if not q_r > 0.0:
            out[str(r)] = {
                "queried_rows": m_r,
                "radius": int(r),
                "rows_with_neighbor_within_radius": hits,
                "exhaustive": True,
                "declared_null_kind": kind,
                "declared_pair_probability": p_report,
                "declared_query_probability": q_report,
                "resolution_status": "NULL_PROBABILITY_UNDERFLOW",
                "decision": "INCONCLUSIVE",
                "statistically_excessive": False,
                "practically_excessive": False,
                "note": (
                    "the declared query probability underflowed to zero in "
                    "double precision; no ratio is defined and none is "
                    "reported"),
            }
            continue

        observed_rate = hits / m_r
        ratio = observed_rate / q_r
        cp_lower = _clopper_pearson_lower(hits, m_r, per_radius_alpha)
        power = minimum_detectable_near_duplicate_ratio(
            queries=m_r, null_query_probability=q_r, alpha=per_radius_alpha)
        # The Clopper--Pearson gate is the inversion of the exact binomial
        # tail test, so this p-value and the bound comparison are the same
        # decision.  It is reported because a tail probability is far easier
        # to read than a bound-versus-null comparison.
        from scipy.stats import binom as _binom
        binomial_p = (1.0 if hits <= 0
                      else float(_binom.sf(hits - 1, m_r, q_r)))
        stat = bool(cp_lower > q_r)
        prac = bool(ratio > 1.0 + practical_relative_excess)
        # Power status.  The excess-ratio arithmetic alone is a misleading
        # summary at small radii: when the declared null probability is tiny,
        # the minimum detectable RATIO is enormous while the test in fact
        # rejects on a single observed occurrence, which is the greatest
        # sensitivity any occurrence-counting test can have.  So a budget that
        # rejects at one hit is reported as sufficient on its own terms, and
        # the ratio comparison is applied only when two or more hits are
        # needed.
        regime = near_duplicate_regime(queries=m_r, null_query_probability=q_r)
        if not power["attainable"]:
            power_status = "STATISTICAL_GATE_UNATTAINABLE"
        elif regime == "DETECTION":
            # Expected null hits < 1.  One verified occurrence is already
            # significant and no stronger sensitivity is possible, but an
            # excess RATIO is not a stable summary here: with an expected
            # count far below one, a single hit yields an arbitrarily large
            # ratio.  Read this radius as presence/absence.
            power_status = "DETECTION_SINGLE_OCCURRENCE"
        elif power["minimum_detectable_excess_ratio"] <= 1.0 + practical_relative_excess:
            power_status = "SUFFICIENT"
        else:
            power_status = "PRACTICAL_GATE_EXCEEDS_STATISTICAL_SENSITIVITY"
        if stat and prac:
            decision = "EXCESS_DETECTED"
        elif stat or prac:
            decision = "INCONCLUSIVE"
        else:
            decision = "NO_EXCESS_DETECTED"

        out[str(r)] = {
            "queried_rows": m_r,
            "radius": int(r),
            "rows_with_neighbor_within_radius": hits,
            "observed_query_rate": float(observed_rate),
            "exhaustive": True,
            "retrieval": "exhaustive exact Hamming scan over all n-1 rows",
            "declared_null_kind": kind,
            "declared_pair_probability": p_report,
            "declared_query_probability": q_report,
            "declared_null_method": "closed form; no Monte Carlo resolution limit",
            "mean_per_query_null_probability": (
                None if per_query_q is None else float(per_query_q.mean())),
            "expected_query_hits": float(expected),
            "expected_pair_count": (
                None if p_report is None else float(n * (n - 1) / 2.0 * p_report)),
            "excess_ratio": float(ratio),
            "excess_ratio_status": "FINITE",
            "familywise_alpha": float(alpha),
            "per_radius_alpha": per_radius_alpha,
            "clopper_pearson_lower_bound": float(cp_lower),
            "exact_binomial_upper_tail_p_value": binomial_p,
            "gate_equivalence": (
                "statistically_excessive is equivalent to "
                "exact_binomial_upper_tail_p_value < per_radius_alpha"),
            "statistically_excessive": stat,
            "practically_excessive": prac,
            "practical_relative_excess_threshold": float(practical_relative_excess),
            "power": power,
            "power_status": power_status,
            "regime": regime,
            "excess_ratio_interpretation": (
                "UNSTABLE SUMMARY -- expected null hits "
                f"({expected:.3g}) is below 1, so the ratio is dominated by "
                "whether a single occurrence was seen. Read "
                "rows_with_neighbor_within_radius against expected_query_hits "
                "and exact_binomial_upper_tail_p_value instead."
                if regime == "DETECTION" else
                "proportional excess over the declared null rate; the "
                "expected null hit count is at least 1, so the ratio is a "
                "stable summary"),
            "decision": decision,
            "inference_caveat": (
                "query hit indicators share a common reference population and "
                "are weakly positively associated; the Clopper--Pearson bound "
                "is an approximation whose realized false-positive rate is "
                "measured on null controls, not asserted"),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "method": "D2.1-exact",
        "rows": int(n),
        "bits": int(w),
        "radii": [int(r) for r in radii],
        "queries_scanned": m,
        "queries_per_radius": {str(r): int(v) for r, v in budgets.items()},
        "query_budget_note": (
            "one exact scan of max(queries_per_radius) rows serves every "
            "radius; radius r is evaluated on the first queries_per_radius[r] "
            "of them, which is itself a uniform without-replacement sample"),
        "radius_family_size": family,
        "exact_duplicates": exact_duplicate_pair_count(x),
        "by_radius": out,
    }


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
        if math.isinf(ratio):
            serialized_ratio = None
            ratio_status = "UNBOUNDED"
        else:
            serialized_ratio = ratio
            ratio_status = "FINITE"
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
            "excess_ratio": serialized_ratio,
            "excess_ratio_status": ratio_status,
            "one_sided_excess_p_value": p_value,
            "statistical_inference": "not_claimed; shared reference set induces dependence among query-hit indicators",
            # Containment (schema 7.0).  These two fields are added so that a
            # record produced by this superseded method cannot be read as a
            # valid D2.1 result, whether it is generated now or recovered from
            # an existing certificate.  No computed value above or below is
            # altered: reproducing an old certificate byte-for-byte remains
            # possible, and the numbers are preserved exactly as recorded.
            "method_status": "SUPERSEDED_METHOD",
            "method_status_detail": (
                "produced by near_neighbor_summary (schema <= 6.5), whose "
                "reference probability was a Monte Carlo estimate that "
                "evaluated to exactly 0.0 across the operating regime; "
                "excess_ratio and practically_excessive from this method do "
                "not support an inference in either direction.  Use "
                "near_neighbor_exact_screen (schema >= 7.0)."),
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
    lag: int = 1,
) -> dict[str, Any]:
    """Can a learned detector tell a GENERATION-ORDER pair from a re-pairing?

    Positive examples are source-order lag-``lag`` pairs ``(row k, row k+lag)``.
    Negative examples reuse the same first endpoints with a deranged second
    endpoint drawn from the same pool, so the two classes differ only in
    whether the pairing is the generator's. Train and test pools are
    row-disjoint.

    WHY THIS CHANGED (schema 6.5)
    -----------------------------
    The previous implementation drew ``4*pairs`` rows, split them in quarters
    with ``np.split``, and paired ``train_a[k]`` with ``train_b[k]``. Those
    positives were an ARBITRARY matching, not generation-order pairs, so the
    positive and negative classes were both arbitrary matchings of the same
    row pool and differed in nothing. The detector had no power:

        data                    shipped        corrected
        independent rows        AUC 0.5033     AUC 0.4981, 1/20 at a=0.05
        exact copy at lag 1     AUC 0.4937     AUC 1.0000, 20/20, p=0.0005
        2-bit copy at lag 1     AUC 0.4899     AUC 1.0000, 20/20

    The null also changed. Each first endpoint appears in exactly one positive
    and one negative example, so the test examples are dependent in matched
    couples and free label permutation is not valid for them. The null now
    swaps labels WITHIN each matched couple, which is the exchangeability the
    design actually supplies.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if pairs < 2 or permutations < 1:
        raise ValueError("pairs and permutations must be positive.")
    if lag < 1:
        raise ValueError("lag must be >= 1.")
    # 2*pairs disjoint source-order edges: `pairs` for training, `pairs` for
    # testing, with no row shared between the two pools.
    i, j = _sample_disjoint_lag_pairs(len(x), lag, 2 * pairs, rng)
    order = rng.permutation(2 * pairs)
    i, j = i[order], j[order]
    train, test = slice(0, pairs), slice(pairs, 2 * pairs)

    def build(sl: slice) -> tuple[np.ndarray, np.ndarray]:
        pi, pj = i[sl], j[sl]
        # Cyclic derangement: guaranteed fixed-point-free for n >= 2, so no
        # negative example is accidentally a true source-order pair.
        neg_j = pj[np.roll(np.arange(len(pj)), 1)]
        return (
            np.vstack([_pair_matrix(x, pi, pj), _pair_matrix(x, pi, neg_j)]),
            np.r_[np.ones(len(pi), dtype=np.int8),
                  np.zeros(len(pi), dtype=np.int8)],
        )

    X_train, y_train = build(train)
    X_test, y_test = build(test)

    model = LogisticRegression(
        C=c,
        solver="liblinear",
        max_iter=2000,
        random_state=int(rng.integers(2**31 - 1)),
    ).fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, scores))
    excess = abs(auc - 0.5)

    # Matched-couple permutation. Example k and example k+pairs share their
    # first endpoint; the null swaps their labels together or not at all.
    k = len(y_test) // 2
    null = np.empty(permutations, dtype=np.float64)
    for t in range(permutations):
        flip = rng.integers(0, 2, size=k).astype(bool)
        y_perm = y_test.copy()
        y_perm[:k][flip] = 0
        y_perm[k:][flip] = 1
        null[t] = roc_auc_score(y_perm, scores)
    p = float((1 + np.count_nonzero(np.abs(null - 0.5) >= excess)) / (permutations + 1))
    return {
        "pairs": int(pairs),
        "lag": int(lag),
        "pair_definition": (
            f"positives are generation-order lag-{lag} pairs (row k, row "
            f"k+{lag}); negatives reuse the same first endpoints with a "
            f"cyclically deranged second endpoint from the same pool"),
        "train_examples": int(len(y_train)),
        "test_examples": int(len(y_test)),
        "row_disjoint_train_test": True,
        "null_model": "matched-couple label permutation within shared-endpoint couples",
        "null_model_reason": (
            "each first endpoint appears in one positive and one negative "
            "example, so test examples are dependent in matched couples and "
            "free label permutation is not valid for them"),
        "detector": "logistic_regression_on_[x_i,x_j,abs(x_i-x_j)]",
        "detector_C": float(c),
        "auc": auc,
        "auc_excess_over_chance": float(excess),
        "practical_tolerance": MULTIVARIATE_AUC_TOLERANCE,
        "permutations": int(permutations),
        "minimum_attainable_permutation_p": 1.0 / (permutations + 1),
        "permutation_p_value": p,
        "null_auc_mean": float(null.mean()),
        "null_auc_95_upper": float(np.quantile(null, 0.95)),
        "statistically_excessive": bool(p < alpha),
        "practically_excessive": bool(excess > MULTIVARIATE_AUC_TOLERANCE),
        "alpha": float(alpha),
        "permutation_budget_resolves_alpha": bool(1.0 / (permutations + 1) < alpha),
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
    # Schema 7.0.  D2.1's reference distribution is DECLARED, never fitted to
    # the data under audit.  Left as None it defaults to the uniform model and
    # the certificate records that the caller did not declare one, so the
    # provenance distinguishes a deliberate declaration from a default.
    near_duplicate_null_model=None,
    near_duplicate_chunk_rows=DEFAULT_EXACT_SCAN_CHUNK,
    near_duplicate_query_block=DEFAULT_EXACT_SCAN_QUERY_BLOCK,
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

    # D2.1's null must be declared.  Record whether the caller declared it or
    # whether the uniform default was taken, so a reader can tell the two
    # apart in the certificate.
    null_model_declared_by_caller = near_duplicate_null_model is not None
    declared_null = (
        {"kind": "uniform"} if near_duplicate_null_model is None
        else dict(near_duplicate_null_model))

    names = list(arrays)
    comparisons = [(f"within:{n}", arrays[n], arrays[n], True) for n in names]
    comparisons += [
        (f"cross:{names[i]}:{names[j]}", arrays[names[i]], arrays[names[j]], False)
        for i in range(len(names)) for j in range(i + 1, len(names))
    ]
    lag_tests = [(f"{n}:lag{k}", x, k) for n, x in arrays.items() for k in lags if k < len(x)]
    # None means "default to every partition"; an EMPTY tuple means "run no
    # multivariate test at all". `multivariate_partitions or names` collapsed
    # those two cases, so a caller could not switch the detector off. That
    # became necessary at schema 6.5: the detector's positives are now
    # generation-order lag pairs, so it must be disable-able wherever lag
    # semantics do not hold -- a label-filtered subset, for instance, whose
    # consecutive rows were not consecutive when generated.
    selected = (tuple(names) if multivariate_partitions is None
                else tuple(multivariate_partitions))
    unknown = [n for n in selected if n not in arrays]
    if unknown:
        raise ValueError(
            f"multivariate_partitions names unknown partitions: {unknown}")
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
    # The statistical gate fires on a simultaneous Clopper-Pearson UPPER BOUND,
    # not on the Monte Carlo point estimate. An earlier guard here compared
    # alpha with 1/(max_tail+1), which is the point-estimate resolution and is
    # roughly an order of magnitude smaller than the bound the gate actually
    # uses. A budget could therefore pass that guard while leaving the gate
    # unreachable, so a component could never reject however large the effect
    # and a maximal violation was silently downgraded to INCONCLUSIVE.
    gate = minimum_attainable_decision_bound(
        null_permutations, reference_permutations=reference_permutations,
        batch_size=DEFAULT_NULL_BATCH_SIZE,
        stopping_error=DEFAULT_NULL_STOPPING_ERROR)
    if gate["minimum_attainable_upper_p"] >= alpha:
        raise ValueError(
            "REFUSED: the permutation budget cannot reach the statistical gate, "
            "so no distributional component could ever reject its null.\n"
            f"  corrected per-test alpha        {alpha:.6g}\n"
            f"  total permutations configured   {null_permutations:,}\n"
            f"  maximum tail draws              {gate['max_tail_draws']:,}\n"
            f"  minimum attainable upper bound  {gate['minimum_attainable_upper_p']:.6g}  "
            f"<-- must be BELOW alpha\n"
            f"  (minimum attainable Monte Carlo p is {gate['minimum_attainable_monte_carlo_p']:.6g}, "
            f"which is NOT the operative quantity)\n"
            f"  required total permutations     "
            f"{required_permutations_for_alpha(alpha, batch_size=DEFAULT_NULL_BATCH_SIZE, stopping_error=DEFAULT_NULL_STOPPING_ERROR, reference_permutations=DEFAULT_NULL_REFERENCE_PERMUTATIONS):,}\n"
            "Raise --null-permutations; do not lower the familywise alpha to pass this check."
        )
    ss = np.random.SeedSequence(audit_seed)
    hamming, lagged, near, mv = {}, {}, {}, {}

    # D2.2 pairwise structure: DESCRIPTIVE ONLY.
    #
    # The observed pairing here is an arbitrary matching of randomly chosen
    # rows and the randomization null re-pairs that same row pool, so the
    # observed statistic is one draw from its own null. These components
    # cannot evidence row dependence and are excluded from the decision gates
    # below; see _randomize_pairing for the construction and the measured
    # power. Serial (lag) tests carry the pairing-structure inference.
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
        hamming[name]["inference"] = {
            "status": "DESCRIPTIVE_ONLY",
            "excluded_from_decision": True,
            "reason": (
                "The observed pairing is an arbitrary matching of randomly "
                "sampled rows and the randomization null re-pairs the same "
                "row pool, so the observed statistic is one draw from its own "
                "null. The null is true by construction, the p-value is "
                "uniform under every alternative, and the component has no "
                "power against row dependence."),
            "valid_instrument_for_this_question": (
                "d2_2_serial_dependence (generation-order lag pairs), "
                "d2_1_near_duplicate_structure, and d2_4_structured_repetition"),
        }

    # D2.1 (schema 7.0): EXACT near-neighbour screen against a declared
    # closed-form null.  Query batches are pooled only as a sampling budget;
    # they are not treated as independent studies.  The superseded
    # near_neighbor_summary is no longer called from any inference path.
    for name, x in arrays.items():
        qbudget = near_duplicate_query_count * audit_replicates
        near[f"within:{name}"] = near_neighbor_exact_screen(
            x,
            radii,
            rng=np.random.default_rng(ss.spawn(1)[0]),
            null_model=declared_null,
            query_count=qbudget,
            alpha=alpha,
            chunk_rows=near_duplicate_chunk_rows,
            query_block=near_duplicate_query_block,
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

    # Pairwise within/cross components are DESCRIPTIVE ONLY (see
    # _randomize_pairing): their null is true by construction, so neither a
    # "statistical" nor a "practical" flag from them carries information about
    # row dependence. They are recorded as excluded rather than dropped, so a
    # reader can see that they ran and why their verdict is not used. No gate
    # is relaxed here: a vacuous test is removed, not a real one.
    excluded_from_decision = []
    for name, result in hamming.items():
        excluded_from_decision.append({
            "component": name,
            "type": "distributional",
            "inference_status": "DESCRIPTIVE_ONLY",
            "reason": result["inference"]["reason"],
            "would_have_flagged_statistical": bool(result["statistical_warning"]),
            "would_have_flagged_practical": not result["practical_pass"],
            "excess_tvd_max": result["null_centered_excess_tvd_max"],
            "minimum_randomization_p_value": result["minimum_randomization_p_value"],
        })

    for name, result in lagged.items():
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

    # D2.1 (schema 7.0).  Records are nested under "by_radius" and carry an
    # explicit per-radius decision plus an attainable-sensitivity status.  A
    # power-limited radius is surfaced as its own warning, because at such a
    # radius NO_EXCESS_DETECTED is weak evidence rather than a clean pass and
    # must not be read as one.
    for name, screen in near.items():
        if "by_radius" not in screen:
            warnings.append({
                "component": name,
                "type": "near_duplicate_method",
                "reason": (
                    "D2.1 record was produced by the superseded schema <= 6.5 "
                    "method; it supports no inference in either direction."),
                "method_status": "SUPERSEDED_METHOD",
                "defects": list(D21_SUPERSEDED_METHOD_DEFECTS),
            })
            continue
        for radius, r in screen["by_radius"].items():
            practical = bool(r.get("practically_excessive", False))
            statistical = bool(r.get("statistically_excessive", False))
            decision = r.get("decision")
            common = {
                "component": f"{name}:radius{radius}",
                "type": "near_duplicate_structure",
                "practically_excessive": practical,
                "statistically_excessive": statistical,
                "excess_ratio": r.get("excess_ratio"),
                "observed_hits": r.get("rows_with_neighbor_within_radius"),
                "expected_query_hits": r.get("expected_query_hits"),
                "declared_null_kind": r.get("declared_null_kind"),
                "clopper_pearson_lower_bound": r.get("clopper_pearson_lower_bound"),
                "power_status": r.get("power_status"),
            }
            if decision == "EXCESS_DETECTED":
                failures.append(dict(
                    common,
                    reason="Both practical and statistical gates indicate excess exactly verified near-neighbour structure."))
            elif decision == "INCONCLUSIVE":
                warnings.append(dict(
                    common,
                    reason="Only one evidence gate was crossed; result is treated as inconclusive rather than as a confirmed dependence finding."))
            elif decision == "NO_EXCESS_DETECTED" and r.get("power_status") != "SUFFICIENT_SINGLE_OCCURRENCE":
                if r.get("power_status") != "SUFFICIENT":
                    warnings.append(dict(
                        common,
                        reason=(
                            "No excess was detected, but at this radius and query "
                            "budget the statistical gate cannot reject an excess as "
                            "small as the practical threshold; this is not a clean "
                            "pass. See the power block for the attainable "
                            "sensitivity and the required query budget."),
                        power=r.get("power")))
            elif decision == "INCONCLUSIVE" or r.get("resolution_status"):
                warnings.append(dict(common, reason=r.get("note", "D2.1 radius was not resolvable.")))

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
    # An explicit inventory: what actually carried inference, what merely
    # occupied a denominator slot. Section 7 of the audit protocol forbids
    # calling every slot an executed test.
    structured_tested = sum(1 for r in structured.values() if "reference" in r)
    structured_skipped = len(structured) - structured_tested
    inferential_inventory = {
        "inferential_tests": {
            "serial_dependence_lag": len(lagged),
            # Count RADII, not the record's top-level keys.  At schema 7.0 a
            # D2.1 record is nested under "by_radius"; counting len(v) here
            # would silently count schema metadata as hypotheses.
            "near_duplicate_radii": sum(
                len(v.get("by_radius", v)) for v in near.values()),
            "structured_views_tested": structured_tested,
            "multivariate": len(mv),
        },
        "non_inferential_slots": {
            "pairwise_within_cross_descriptive_only": len(hamming),
            "structured_views_not_tested": structured_skipped,
        },
        "familywise_denominator": total,
        "statement": (
            "The Bonferroni denominator is deliberately conservative and "
            "counts declared slots, including non-inferential ones. The "
            "inferential_tests counts above are what produced a verdict."),
    }
    inferential_inventory["inferential_tests"]["total"] = sum(
        inferential_inventory["inferential_tests"].values())
    inferential_inventory["non_inferential_slots"]["total"] = sum(
        inferential_inventory["non_inferential_slots"].values())

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
            "familywise_denominator_semantics": (
                "familywise_test_count is a CONSERVATIVE DENOMINATOR over "
                "declared component slots. It is not a count of executed "
                "inferential tests: it includes structured-view slots the "
                "adapter left NOT_TESTED and pairwise slots that are "
                "descriptive only. See inferential_test_inventory."),
            "pairwise_inference_status": (
                "DESCRIPTIVE_ONLY — d2_2_pairwise_structure (within:* and "
                "cross:*) has an arbitrary observed pairing and a null that "
                "re-pairs the same row pool, so its null is true by "
                "construction and it has no power against row dependence. "
                "Excluded from the decision gates. Row-dependence inference "
                "rests on d2_2_serial_dependence, "
                "d2_1_near_duplicate_structure and "
                "d2_4_structured_repetition."),
            "confirmatory_pairwise_components": "d2_2_serial_dependence only",
            "nominal_hamming_reference": "Binomial(feature_bits, 0.5) may be reported diagnostically only",
            "practical_effect_measure": "null-centered excess TVD = observed TVD - mean TVD under the fixed-dataset randomization null",
            "practical_effect_threshold": float(practical_excess_tvd_threshold),
            "confirmatory_pairwise_null": "NOT CONFIRMATORY — see pairwise_inference_status",
            "confirmatory_serial_null": "fixed-dataset randomization of complete endpoint rows across the observed lag pairing",
            "randomization_tail_sampling": "adaptive fixed-batch Monte Carlo; inferential decisions use simultaneous Clopper-Pearson confidence bounds over predeclared looks; the reported Monte Carlo p-value is descriptive under adaptive stopping",
            "near_duplicate_null": (
                "DECLARED closed-form model, evaluated exactly; never fitted "
                "or resampled from the data under audit. Under a declared "
                "i.i.d. row model, conditioning on a query row leaves the "
                "other n-1 rows independent, so the query-level null "
                "probability q = 1-(1-p)**(n-1) is exact rather than an "
                "approximation."),
            "near_duplicate_null_model": declared_null,
            "near_duplicate_null_model_declared_by_caller": null_model_declared_by_caller,
            "near_duplicate_inference": (
                "two gates. practical: observed query rate / declared null "
                "query probability exceeds the relative-excess threshold. "
                "statistical: a Clopper-Pearson LOWER bound on the observed "
                "rate, Bonferroni-corrected across the radius family, exceeds "
                "the declared null probability. Hit indicators across queries "
                "share a common reference population and are weakly "
                "positively associated, so the bound is an approximation "
                "whose realized false-positive rate is MEASURED on null "
                "controls rather than asserted."),
            "near_duplicate_method": (
                "EXHAUSTIVE exact Hamming scan: for each sampled query row "
                "the minimum distance to every other row is computed, so "
                "retrieval recall is 1.0 by construction and there is no "
                "approximate-retrieval stage to calibrate."),
            "near_duplicate_attainable_sensitivity": (
                "every radius reports a power block giving the minimum "
                "rejecting hit count, the minimum detectable excess ratio, "
                "and a power_status. A radius whose statistical gate cannot "
                "reject an excess as small as the practical threshold is "
                "surfaced as a warning, because NO_EXCESS_DETECTED there is "
                "weak evidence and not a clean pass."),
            "near_duplicate_superseded_method": (
                "near_neighbor_summary (schema <= 6.5) is retained for "
                "version-aware reading of existing certificates and is not "
                "called from any inference path. Its defects are enumerated "
                "in D21_SUPERSEDED_METHOD_DEFECTS."),
            "pairs_per_test": pairs_per_test,
            "audit_replicates": audit_replicates,
            "null_permutations": null_permutations,
            "null_permutations_semantics": "maximum total randomization draws; reference draws are separate from the confirmatory tail sample",
            "null_reference_permutations": reference_permutations,
            "null_batch_size": DEFAULT_NULL_BATCH_SIZE,
            "null_adaptive_stopping": True,
            "statistical_gate_reachability": gate,
            "statistical_gate_semantics": (
                "a distributional component rejects only when the simultaneous "
                "Clopper-Pearson upper bound falls below the corrected alpha; "
                "the Monte Carlo point estimate is descriptive. run_d2 refuses "
                "a budget whose attainable bound cannot cross alpha."),
            "null_stopping_error": DEFAULT_NULL_STOPPING_ERROR,
            "bootstrap_replicates": bootstrap_replicates,
            "lags": list(lags),
            "near_duplicate_radii": list(radii),
            "near_duplicate_query_count": near_duplicate_query_count,
            "near_duplicate_effective_queries": near_duplicate_query_count * audit_replicates,
            "near_duplicate_chunk_rows": near_duplicate_chunk_rows,
            "near_duplicate_query_block": near_duplicate_query_block,
            "near_duplicate_superseded_parameters": {
                "note": (
                    "accepted for signature compatibility with schema <= 6.5 "
                    "callers and recorded for provenance; INERT at schema 7.0 "
                    "because the exact screen has no retrieval stage and no "
                    "Monte Carlo null"),
                "projection_tables": near_duplicate_projection_tables,
                "null_pairs": near_duplicate_null_pairs,
            },
            "multivariate_pairs": multivariate_pairs,
            "multivariate_permutations": multivariate_permutations,
            "practical_effect_threshold": float(practical_excess_tvd_threshold),
            "audit_seed": audit_seed,
            "reference_pmf_supplied": nominal is not None,
            "inferential_test_inventory": inferential_inventory,
        },
    }
    decision = {
        "outcome": outcome,
        "failures": failures,
        "warnings": warnings,
        "excluded_from_decision": excluded_from_decision,
        "excluded_from_decision_note": (
            "These components executed and are reported, but their null is "
            "true by construction, so their verdicts are not inferential and "
            "do not contribute to the outcome. See "
            "configuration.pairwise_inference_status."),
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
            "d2_1": "Sampled-query near-neighbour screen with EXHAUSTIVE exact Hamming verification against every other row, compared to a declared closed-form null evaluated exactly. Retrieval recall is 1.0 by construction. Attainable sensitivity is reported per radius.",
            "d2_2_pairwise": "DESCRIPTIVE ONLY, EXCLUDED FROM THE DECISION. The observed within/cross pairing is an arbitrary matching of randomly sampled rows and the randomization null re-pairs that same row pool, so the null is true by construction, the p-value is uniform under every alternative, and the component has no power against row dependence. Reported for description only; it evidences nothing about sample dependence.",
            "d2_2_serial": "Generation-order lag pairs are compared with random re-pairings of the same complete endpoint rows. Practical magnitude is assessed using null-centered excess TVD rather than raw finite-sample TVD.",
            "d2_3": "Logistic detector on generation-order lag-1 pairs versus deranged re-pairings of the same endpoints, with row-disjoint train/test pools and a matched-couple label permutation null. Schema 6.4 and earlier used an arbitrary matching for the positive class and had no power; see multivariate_pair_discrimination.",
            "d2_4": "Structured collision tests require an explicit adapter-supplied finite-domain null; generic D2 does not assume uniformity.",
            "d2_5": "Controlled fault injection with Wilson intervals; calibration is sensitivity evidence, not clean-data evidence.",
            "multiple_comparison_control": "Bonferroni FWER 0.01 across declared hypothesis-level components; sampling replicates are not counted as separate hypotheses.",
            "decision_semantics": "Not a proof of universal mutual independence. A component is a D2 failure only when both statistical and practical evidence gates are crossed; a one-gate result is INCONCLUSIVE.",
            "audit_seed_semantics": "Controls audit sampling/randomization only unless the adapter explicitly supplies deterministic generation.",
            "audit_seed": audit_seed,
        },
        limitations=[
            "Finite testing cannot establish universal mutual independence.",
            "D2.1 verification is exhaustive over the full row population, so no neighbour within a tested radius can be missed; the sampled quantity is which ROWS are queried, not which neighbours are found.",
            "D2.1 sensitivity is established only for the explicitly calibrated fault classes and strengths.",
            "Fixed-dataset randomization tests are conditional on the audited dataset instance and test pairing/ordering structure, not a universal population theorem. This applies to the SERIAL (lag) tests, whose observed pairing is generation order. The within/cross pairwise components have no source-defined pairing, are descriptive only, and are excluded from the decision.",
            "No statement about row dependence rests on d2_2_pairwise_structure.",
            "D2.1 compares against a DECLARED null evaluated in closed form, so the comparison is only as good as that declaration: a dataset whose true row marginals depart from the declared model will shift the reference probability. Marginal conformance is D3's question, not D2.1's, and the declared model is recorded in the configuration.",
            "D2.1 hit indicators across queries share a common reference population and are weakly positively associated, so the Clopper-Pearson gate is an approximation; its realized false-positive rate is measured on null controls and reported, not assumed.",
            "D2.1 statistical sensitivity is bounded by the query budget. At radii where the declared null probability is tiny the gate rejects on a single observed occurrence, which is maximal sensitivity; at larger radii it may be unable to reject an excess as small as the practical threshold, and that case is reported per radius as a warning rather than a pass.",
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
    print(f"Pairwise comparisons : {len(results['d2_2_pairwise_structure'])}"
          f"  (DESCRIPTIVE ONLY — excluded from the decision)")
    print(f"Lagged comparisons   : {len(results['d2_2_serial_dependence'])}")
    print(f"Multivariate tests   : {len(results['d2_3_multivariate_dependence'])}")
    print(f"Structured views     : {len(results['d2_4_structured_repetition'])}")
    print(f"Near-duplicate views : {len(results['d2_1_near_duplicate_structure'])}")
    print(f"Outcome              : {certificate['decision']['outcome']}")
    print(f"FWER alpha           : {results['configuration']['familywise_alpha']}")
    print(f"Effective test alpha : {results['configuration']['effective_test_alpha']:.3e}")
    inv = results["configuration"].get("inferential_test_inventory", {})
    if inv:
        print(f"Inferential tests    : {inv['inferential_tests']['total']}"
              f"   (denominator slots: {inv['familywise_denominator']})")
    print("Null architecture    : complete-row randomization/resampling; no bit-independence assumption")
    print("Row-dependence rests on: serial lag tests, near-duplicate screen, structured views")
    print("=" * 78)
