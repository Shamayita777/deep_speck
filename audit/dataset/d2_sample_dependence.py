"""
D2 — Sample Dependence Audit.

Generic Dataset Integrity implementation.

This module is dataset-agnostic. It knows nothing about Gohr, Speck,
cryptographic rounds, plaintexts, ciphertexts, keys, or any particular
model. Dataset-specific semantics are supplied by an adapter/experiment.

D2 has five layers:
    D2.1  Exact/near-duplicate structure (near-duplicate only; D1 owns
          exact row duplication and exact partition overlap).
    D2.2  Serial/lagged dependence.
    D2.3  Multivariate dependence via real-vs-permuted pair discrimination.
    D2.4  Adapter-supplied structured/semantic repetition.
    D2.5  Controlled fault-injection calibration and sensitivity.

The decision is evidence-bounded: PASS never means mathematical proof of
independence. It means that no practically material violation was detected
within the pre-specified audit scope and validated sensitivity envelope.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from audit.dataset.common.certificate import make_certificate, write_certificate
from audit.dataset.common.provenance import array_sha256, build_provenance

AUDIT_ID = "D2"
AUDIT_NAME = "Sample Dependence Audit"
SCHEMA_VERSION = "2.0"

# Locked protocol defaults.
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
    pmf = np.array([math.comb(n_bits, k) * p**k * (1-p)**(n_bits-k) for k in range(n_bits+1)], dtype=float)
    return pmf / pmf.sum()


def _sample_pairs(n: int, count: int, rng: np.random.Generator, distinct: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if n < 2 or count < 1:
        raise ValueError("Need n >= 2 and count >= 1.")
    a = rng.integers(0, n, size=count, dtype=np.int64)
    if not distinct:
        return a, rng.integers(0, n, size=count, dtype=np.int64)
    b = rng.integers(0, n-1, size=count, dtype=np.int64)
    b = np.where(b >= a, b + 1, b)
    return a, b


def hamming_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.count_nonzero(a != b, axis=1).astype(np.int16, copy=False)


def sample_within_distances(x: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    i, j = _sample_pairs(len(x), count, rng)
    return hamming_distances(x[i], x[j])


def sample_cross_distances(a: np.ndarray, b: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    i = rng.integers(0, len(a), size=count, dtype=np.int64)
    j = rng.integers(0, len(b), size=count, dtype=np.int64)
    return hamming_distances(a[i], b[j])


def sample_lagged_distances(x: np.ndarray, lag: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if lag < 1 or lag >= len(x):
        raise ValueError("lag must satisfy 1 <= lag < number of samples.")
    starts = rng.integers(0, len(x)-lag, size=count, dtype=np.int64)
    return hamming_distances(x[starts], x[starts+lag])


def _chi_square_gof(distances: np.ndarray, pmf: np.ndarray, alpha: float) -> dict[str, Any]:
    from scipy.stats import chi2
    obs = np.bincount(distances.astype(np.int64), minlength=len(pmf)).astype(float)
    exp = pmf * len(distances)
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]
    # Merge sparse tails conservatively.
    mo, me, ro, re = [], [], 0.0, 0.0
    for o, e in zip(obs, exp):
        ro += o; re += e
        if re >= 5:
            mo.append(ro); me.append(re); ro = re = 0.0
    if re:
        if me: mo[-1] += ro; me[-1] += re
        else: mo.append(ro); me.append(re)
    stat = float(np.sum((np.asarray(mo)-np.asarray(me))**2 / np.asarray(me)))
    dof = max(1, len(me)-1)
    p = float(chi2.sf(stat, dof))
    return {"statistic": stat, "degrees_of_freedom": dof, "p_value": p, "alpha": alpha, "reject": p < alpha}


def tvd(distances: np.ndarray, pmf: np.ndarray) -> float:
    counts = np.bincount(distances.astype(np.int64), minlength=len(pmf)).astype(float)
    empirical = counts / len(distances)
    return float(0.5 * np.abs(empirical-pmf).sum())


def bootstrap_tvd(distances: np.ndarray, pmf: np.ndarray, rng: np.random.Generator, reps: int) -> dict[str, Any]:
    empirical_counts = np.bincount(distances.astype(np.int64), minlength=len(pmf)).astype(float)
    empirical = empirical_counts / len(distances)
    draws = rng.multinomial(len(distances), empirical, size=reps) / len(distances)
    values = 0.5 * np.abs(draws-pmf).sum(axis=1)
    lo = float(np.quantile(values, (1-TVD_CONFIDENCE_LEVEL)/2))
    hi = float(np.quantile(values, 1-(1-TVD_CONFIDENCE_LEVEL)/2))
    return {"confidence_level": TVD_CONFIDENCE_LEVEL, "replicates": reps, "lower": lo, "upper": hi, "within_threshold": hi <= TVD_THRESHOLD}


def summarize_distances(distances: np.ndarray, pmf: np.ndarray, rng: np.random.Generator, alpha: float, bootstrap_reps: int) -> dict[str, Any]:
    value = tvd(distances, pmf)
    return {
        "sampled_pairs": int(len(distances)),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "min": int(np.min(distances)),
        "max": int(np.max(distances)),
        "tvd": value,
        "tvd_threshold": TVD_THRESHOLD,
        "chi_square": _chi_square_gof(distances, pmf, alpha),
        "tvd_uncertainty": bootstrap_tvd(distances, pmf, rng, bootstrap_reps),
    }


def _aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("At least one replicate is required.")
    return {
        "replicate_count": len(results),
        "replicates": [dict(r) for r in results],
        "max_tvd": max(float(r["tvd"]) for r in results),
        "max_tvd_ci_upper": max(float(r["tvd_uncertainty"]["upper"]) for r in results),
        "min_p_value": min(float(r["chi_square"]["p_value"]) for r in results),
        "practical_pass": all(bool(r["tvd_uncertainty"]["within_threshold"]) for r in results),
        "statistical_warning": any(bool(r["chi_square"]["reject"]) for r in results),
    }


def near_duplicate_summary(distances: np.ndarray, radii: Sequence[int], reference_pmf: np.ndarray, sample_pairs: int) -> dict[str, Any]:
    from scipy.stats import binom
    out: dict[str, Any] = {}
    for r in radii:
        if r < 0 or r >= len(reference_pmf):
            raise ValueError("Near-duplicate radius outside feature space.")
        observed = int(np.count_nonzero(distances <= r))
        q = float(reference_pmf[:r+1].sum())
        expected = sample_pairs * q
        p = float(binom.sf(observed-1, sample_pairs, q)) if q > 0 else (1.0 if observed == 0 else 0.0)
        relative = float(observed / expected) if expected > 0 else (1.0 if observed == 0 else float("inf"))
        out[str(r)] = {"observed_pairs": observed, "null_probability": q, "expected_pairs": expected, "excess_ratio": relative, "one_sided_excess_p_value": p}
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
        observed = int(np.sum(repeated*(repeated-1)//2))
        expected = len(values)*(len(values)-1)/(2*domain)
        p = float(poisson.sf(observed-1, expected)) if observed else 1.0
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
                "excess_ratio": float(observed/expected) if expected else (1.0 if observed == 0 else float("inf")),
                "one_sided_excess_p_value": p,
                "alpha": alpha,
                "statistically_excessive": p < alpha,
                "practically_excessive": observed > expected*(1+practical_relative_excess),
            },
        }
    return results


def _pair_matrix(x: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    a, b = x[i].astype(np.float32), x[j].astype(np.float32)
    # Include XOR so the fixed linear detector can see pairwise bit relations.
    xor = np.abs(a-b)
    return np.concatenate([a, b, xor], axis=1)


def multivariate_pair_discrimination(x: np.ndarray, *, pairs: int, permutations: int, rng: np.random.Generator, c: float = DEFAULT_DETECTOR_C) -> dict[str, Any]:
    """Test whether ordered real pairs are distinguishable from permuted pairs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    i, j = _sample_pairs(len(x), pairs, rng, distinct=True)
    negative_j = rng.permutation(j)
    # Prevent accidental self-pairing in the negative set where possible.
    bad = negative_j == i
    if np.any(bad):
        negative_j[bad] = (negative_j[bad] + 1) % len(x)

    X = np.vstack([_pair_matrix(x, i, j), _pair_matrix(x, i, negative_j)])
    y = np.concatenate([np.ones(pairs, dtype=np.int8), np.zeros(pairs, dtype=np.int8)])
    train, test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=int(rng.integers(0, 2**31-1)))
    model = LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=int(rng.integers(0, 2**31-1)))
    model.fit(train, y_train)
    auc = float(roc_auc_score(y_test, model.predict_proba(test)[:, 1]))

    # Permutation null for the observed AUC. Refit the same fixed detector.
    null = np.empty(permutations, dtype=float)
    for k in range(permutations):
        yp = rng.permutation(y_train)
        m = LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=0)
        m.fit(train, yp)
        null[k] = roc_auc_score(y_test, m.predict_proba(test)[:, 1])
    p = float((1 + np.count_nonzero(null >= auc)) / (permutations + 1))
    return {
        "pairs": pairs,
        "detector": "logistic_regression_on_[x_i, x_j, xor(x_i,x_j)]",
        "auc": auc,
        "null_auc_mean": float(null.mean()),
        "null_auc_95_upper": float(np.quantile(null, 0.95)),
        "auc_excess_over_chance": abs(auc-0.5),
        "practical_tolerance": MULTIVARIATE_AUC_TOLERANCE,
        "permutations": permutations,
        "permutation_p_value": p,
        "statistically_excessive": p < FAMILYWISE_ALPHA,
        "practically_excessive": abs(auc-0.5) > MULTIVARIATE_AUC_TOLERANCE,
    }


def inject_duplicates(x: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1].")
    y = np.array(x, copy=True)
    count = int(round(len(y)*fraction))
    if count:
        src = rng.integers(0, len(y), size=count)
        dst = rng.choice(len(y), size=count, replace=False)
        y[dst] = y[src]
    return y


def inject_lag_copy(x: np.ndarray, fraction: float, lag: int, rng: np.random.Generator) -> np.ndarray:
    if not 0 <= fraction <= 1 or lag < 1 or lag >= len(x):
        raise ValueError("Invalid lag-copy parameters.")
    y = np.array(x, copy=True)
    count = int(round((len(y)-lag)*fraction))
    if count:
        starts = rng.choice(np.arange(lag, len(y)), size=count, replace=False)
        y[starts] = y[starts-lag]
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


def run_d2(*, partitions: Mapping[str, np.ndarray], reference_pmf: np.ndarray, feature_bits: int, structured_views: Mapping[str, Mapping[str, Any]] | None = None, pairs_per_test: int = DEFAULT_PAIRS_PER_TEST, audit_replicates: int = DEFAULT_AUDIT_REPLICATES, bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES, lags: Sequence[int] = DEFAULT_LAGS, near_duplicate_radii: Sequence[int] = NEAR_DUPLICATE_RADII, multivariate_partitions: Sequence[str] | None = None, multivariate_pairs: int = DEFAULT_MULTIVARIATE_PAIRS, multivariate_permutations: int = DEFAULT_MULTIVARIATE_PERMUTATIONS, audit_seed: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    if not partitions or pairs_per_test < 1 or audit_replicates < 1 or bootstrap_replicates < 100:
        raise ValueError("Invalid D2 configuration.")
    pmf = validate_reference_pmf(reference_pmf, feature_bits)
    arrays = {name: _binary(x, name, feature_bits) for name, x in partitions.items()}
    structured_views = structured_views or {}
    lags = tuple(int(k) for k in lags)
    near_duplicate_radii = tuple(int(r) for r in near_duplicate_radii)
    comparisons = [(f"within:{n}", x, x) for n, x in arrays.items()]
    names = list(arrays)
    comparisons += [(f"cross:{names[i]}:{names[j]}", arrays[names[i]], arrays[names[j]]) for i in range(len(names)) for j in range(i+1, len(names))]
    lag_tests = [(f"{n}:lag{k}", x, k) for n, x in arrays.items() for k in lags if k < len(x)]
    total_gof = (len(comparisons) + len(lag_tests))*audit_replicates
    total_scalar = total_gof + len(structured_views) + len(near_duplicate_radii)*len(comparisons)
    alpha = FAMILYWISE_ALPHA / max(1, total_scalar)
    ss = np.random.SeedSequence(audit_seed)

    hamming = {}
    near_dups = {}
    for name, a, b in comparisons:
        reps = []
        all_distances = []
        for child in ss.spawn(audit_replicates):
            rng = np.random.default_rng(child)
            d = sample_within_distances(a, pairs_per_test, rng) if a is b else sample_cross_distances(a, b, pairs_per_test, rng)
            all_distances.append(d)
            reps.append(summarize_distances(d, pmf, rng, alpha, bootstrap_replicates))
        hamming[name] = _aggregate(reps)
        near_dups[name] = near_duplicate_summary(np.concatenate(all_distances), near_duplicate_radii, pmf, pairs_per_test*audit_replicates)

    lagged = {}
    for name, x, lag in lag_tests:
        reps = []
        for child in ss.spawn(audit_replicates):
            rng = np.random.default_rng(child)
            reps.append(summarize_distances(sample_lagged_distances(x, lag, pairs_per_test, rng), pmf, rng, alpha, bootstrap_replicates))
        lagged[name] = _aggregate(reps)

    structured = analyze_structured_views(structured_views, alpha)

    mv = {}
    selected = tuple(multivariate_partitions or arrays.keys())
    for name in selected:
        if name not in arrays:
            raise ValueError(f"Unknown multivariate partition: {name}")
        rng = np.random.default_rng(ss.spawn(1)[0])
        mv[name] = multivariate_pair_discrimination(arrays[name], pairs=multivariate_pairs, permutations=multivariate_permutations, rng=rng)

    failures, warnings = [], []
    for name, result in {**hamming, **lagged}.items():
        if not result["practical_pass"]:
            failures.append({"component": name, "type": "distributional", "reason": "Upper TVD uncertainty bound exceeded the pre-specified practical tolerance."})
        elif result["statistical_warning"]:
            warnings.append({"component": name, "type": "statistical", "reason": "At least one replicate rejected the supplied reference after multiplicity correction.", "minimum_p_value": result["min_p_value"]})
    for name, result in structured.items():
        ref = result["reference"]
        if ref["practically_excessive"]:
            failures.append({"component": name, "type": "structured_collision", "reason": "Structured-value collision excess exceeded the practical tolerance."})
        elif ref["statistically_excessive"]:
            warnings.append({"component": name, "type": "structured_collision", "reason": "Structured-value collision count was statistically excessive but not practically excessive."})
    for name, result in mv.items():
        if result["practically_excessive"]:
            failures.append({"component": name, "type": "multivariate_dependence", "reason": "Real sequential pairs were distinguishable from permuted pairs beyond the practical AUC tolerance."})
        elif result["statistically_excessive"]:
            warnings.append({"component": name, "type": "multivariate_dependence", "reason": "Permutation test detected statistical evidence of pair dependence without a practically material AUC effect."})

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
        },
    }
    decision = {
        "outcome": outcome,
        "failures": failures,
        "warnings": warnings,
        "interpretation": "No practically material dependence detected within the tested classes and validated sensitivity envelope." if outcome == "PASS" else "The decision is limited to the explicitly tested dependence classes and practical thresholds.",
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
        claim="The supplied dataset instance was audited for detectable near-duplicate structure, serial dependence, pairwise structure, multivariate dependence, and adapter-exposed structured repetition.",
        outcome=str(decision["outcome"]),
        findings={"results": dict(results), "decision": dict(decision)},
        methodology={
            "scope": "Dataset Integrity D2 Sample Dependence Audit",
            "d1_boundary": "D1 owns exact row duplication and exact partition overlap; D2 tests near-duplicate and dependence structure without repeating D1's exact census.",
            "d2_1": "Near-duplicate rates are measured at pre-specified Hamming radii against the supplied feature-space reference.",
            "d2_2": "Pairwise and generation-order lagged Hamming structure are tested over the pre-specified lag family.",
            "d2_3": "A fixed logistic detector attempts to distinguish real sequential pairs from independently permuted pairs; significance uses a permutation null and practical significance uses an AUC tolerance.",
            "d2_4": "Adapter-supplied semantic views are evaluated for excess finite-domain collisions.",
            "d2_5": "Controlled fault injectors can be evaluated separately through calibrate_fault() to establish empirical sensitivity.",
            "multiple_comparison_control": "Bonferroni familywise control across the declared scalar test family.",
            "decision_semantics": "PASS is evidence against the tested dependence hypotheses, not proof of universal independence.",
            "reference_distribution": reference_description,
            "reference_model": reference_model_description,
            "audit_seed": audit_seed,
        },
        provenance=provenance,
        limitations=[
            "Finite statistical testing cannot establish universal mutual independence.",
            "Power is specific to the declared tests, sample sizes, detector, null models, and practical thresholds.",
            "The supplied reference distribution/model must be scientifically justified by the dataset-specific experiment.",
            "Structured repetition tests are conditional on adapter-exposed representations.",
            "Calibration must be run and reported before claiming the corresponding sensitivity level.",
        ],
        evidence_level="DATASET_INTEGRITY_D2_SAMPLE_DEPENDENCE",
        certificate_version=SCHEMA_VERSION,
    )
    write_certificate(certificate, output_path)
    return certificate


def print_report(results: Mapping[str, Any], certificate: Mapping[str, Any]) -> None:
    print("="*78)
    print("Dataset Integrity — D2 Sample Dependence Audit")
    print("="*78)
    print(f"Pairwise comparisons : {len(results['d2_2_pairwise_structure'])}")
    print(f"Lagged comparisons   : {len(results['d2_2_serial_dependence'])}")
    print(f"Multivariate tests   : {len(results['d2_3_multivariate_dependence'])}")
    print(f"Structured views     : {len(results['d2_4_structured_repetition'])}")
    print(f"Near-duplicate views : {len(results['d2_1_near_duplicate_structure'])}")
    print(f"Outcome              : {certificate['decision']['outcome']}")
    print("="*78)
