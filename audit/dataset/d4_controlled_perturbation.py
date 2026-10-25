"""
D4 - Controlled Perturbation Audit.

Generic framework for repeated controlled perturbation experiments.

Scientific design
-----------------
D4 estimates whether a controlled perturbation changes predictive
performance relative to an unperturbed baseline.

For binary classification, inference is performed on predictions
from the same held-out test examples, giving a paired comparison.
McNemar's exact test is used for the paired binary correctness
outcomes, while replicate-level performance differences quantify
training-run variability.

D4 deliberately does NOT interpret a label-shuffle effect as proof
of cryptographic dependence. A label shuffle tests whether the
original feature/label correspondence is necessary for predictive
performance; the observed effect may arise from cryptographic
structure, implementation artifacts, dataset-construction artifacts,
or other label-correlated structure.

The framework therefore reports:

- replicate-level baseline accuracy;
- replicate-level perturbed accuracy;
- paired accuracy difference;
- replicate mean and standard deviation;
- bootstrap confidence interval for the replicate-level effect;
- exact McNemar p-value for each paired test-set comparison;
- multiplicity-adjusted McNemar inference across replicates;
- effect-detection decision based on a pre-specified effect size
  and inferential criterion.

Under rule ``near_chance_tost_v3`` the primary endpoint is different: it
asks whether each label-shuffled fit's test accuracy is practically
equivalent to chance (exact binomial TOST against a registered band
[0.5 - margin, 0.5 + margin]).  The paired baseline-vs-perturbed McNemar
comparison is then a separate supporting (or co-primary) endpoint.  A
reduction relative to baseline never substitutes for near-chance evidence:
a shuffled model at 0.70 is far worse than a 0.93 baseline yet far from
chance.

No result is labelled "statistically significant" merely because
an effect exceeds a percentage threshold.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import os

import numpy as np
import json
from pathlib import Path
from scipy.stats import beta, binom, binomtest

from audit.dataset.integrity import (
    IntegrityError,
    labelled_partition_digest,
    partition_digest,
    verify_permutation_indices,
    read_json,
    sha256_file,
    write_bytes_exclusive,
    write_json_exclusive,
)

# ============================================================
# Perturbation interface
# ============================================================

class Perturbation:
    """
    Abstract interface for a controlled dataset perturbation.

    A perturbation receives a feature matrix and label array and
    returns the perturbed feature/label pair.

    Perturbation randomness must be supplied explicitly through
    the NumPy Generator argument. This prevents accidental use
    of global RNG state.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
    ) -> None:
        if not name:
            raise ValueError(
                "Perturbation name must be non-empty."
            )

        if not description:
            raise ValueError(
                "Perturbation description must be non-empty."
            )

        self.name = name
        self.description = description

    def apply(
        self,
        features: Any,
        labels: Any,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the perturbation.

        Subclasses must implement this method.
        """

        raise NotImplementedError(
            "Perturbation subclasses must implement apply()."
        )

    #: True if apply() returns the input features unchanged (only labels
    #: are perturbed); the engine then persists labels only.
    label_only: bool = False

    #: True if the perturbation is a permutation of label positions given by
    #: ``permutation_indices(n, rng)`` and applied by ``apply_indices``; the
    #: engine then persists and verifies the indices themselves.
    index_based: bool = False

    def permutation_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError("index_based perturbations must implement permutation_indices().")

    def apply_indices(self, features: Any, labels: Any, indices: Any) -> tuple[Any, Any]:
        raise NotImplementedError("index_based perturbations must implement apply_indices().")

    def verify(
        self,
        features: Any,
        labels: Any,
        perturbed_features: Any,
        perturbed_labels: Any,
    ) -> None:
        """Raise if (perturbed_features, perturbed_labels) is not a valid
        output of this perturbation for (features, labels).

        The engine refuses perturbations that do not override this.
        """
        raise NotImplementedError("Perturbation subclasses must implement verify().")

# ============================================================
# Result structures
# ============================================================

@dataclass
class ReplicateResult:
    """One independent baseline/perturbation training replicate."""

    replicate: int
    seed: int

    baseline_score: float
    perturbed_score: float

    absolute_difference: float
    relative_difference_percent: float

    mcnemar_b: int
    mcnemar_c: int
    mcnemar_pvalue: float

    # Selection record of the checkpoint each condition was evaluated with
    # (adapter-supplied; empty for adapters that do not select).
    baseline_selection: dict = field(default_factory=dict)
    perturbed_selection: dict = field(default_factory=dict)
    perturbation_record: dict = field(default_factory=dict)
    test_examples: int = 0
    # Sufficient statistics: correct-prediction counts on the test rows.
    baseline_correct: int = 0
    perturbed_correct: int = 0
    # Persisted raw test predictions of each selected model (file, sha256, dtype, shape).
    baseline_predictions: dict = field(default_factory=dict)
    perturbed_predictions: dict = field(default_factory=dict)


@dataclass
class D4Result:
    """Aggregated repeated D4 experiment."""

    perturbation: str

    replicates: list[ReplicateResult]

    mean_baseline_score: float
    mean_perturbed_score: float

    mean_absolute_difference: float
    sd_absolute_difference: float

    mean_relative_difference_percent: float

    bootstrap_ci_low: float
    bootstrap_ci_high: float

    min_mcnemar_pvalue: float
    max_mcnemar_pvalue: float

    adjusted_alpha: float

    effect_threshold: float

    effect_detected: bool

    inference_supported: bool

    interpretation: str

    decision: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)


# ============================================================
# Paired binary inference
# ============================================================

def mcnemar_exact_pvalue(
    baseline_correct: np.ndarray,
    perturbed_correct: np.ndarray,
) -> tuple[int, int, float]:
    """
    Perform exact McNemar inference on paired binary outcomes.

    Parameters
    ----------
    baseline_correct:
        Boolean correctness indicators for the baseline model.

    perturbed_correct:
        Boolean correctness indicators for the perturbed model.

    Returns
    -------
    (b, c, p_value)

    b:
        Baseline incorrect / perturbed correct.

    c:
        Baseline correct / perturbed incorrect.

    Notes
    -----
    Only discordant pairs contribute to the McNemar test.
    """

    baseline_correct = np.asarray(
        baseline_correct,
        dtype=bool,
    )

    perturbed_correct = np.asarray(
        perturbed_correct,
        dtype=bool,
    )

    if baseline_correct.shape != perturbed_correct.shape:
        raise ValueError(
            "Paired correctness arrays must have identical shapes."
        )

    b = int(
        np.sum(
            (~baseline_correct)
            & perturbed_correct
        )
    )

    c = int(
        np.sum(
            baseline_correct
            & (~perturbed_correct)
        )
    )

    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    else:
        # Exact two-sided McNemar test.
        #
        # Under H0, conditional on the discordant total,
        # b ~ Binomial(discordant, 0.5).
        p_value = float(
            binomtest(
                b,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return b, c, p_value


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    bootstrap_replicates: int = 5000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap percentile confidence interval for the mean.

    The bootstrap resamples independent training replicates,
    not individual test examples.

    This distinction is important: test examples are paired
    observations within a replicate, whereas independent model
    fits are the experimental replicates.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1 or len(values) < 2:
        raise ValueError(
            "At least two replicate-level observations are required."
        )

    if bootstrap_replicates < 1000:
        raise ValueError(
            "bootstrap_replicates must be >= 1000."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie in (0, 1)."
        )

    indices = rng.integers(
        0,
        len(values),
        size=(bootstrap_replicates, len(values)),
    )

    bootstrap_means = np.mean(
        values[indices],
        axis=1,
    )

    alpha = 1.0 - confidence_level

    low = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    high = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    return low, high


# ============================================================
# Prediction handling
# ============================================================

def binary_predictions_to_correctness(
    predictions: Any,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Convert binary model predictions into correctness indicators.

    Predictions may be shaped:

        (n,)
        (n, 1)

    and are interpreted as probabilities/logits thresholded at 0.5.

    This function is intentionally explicit because D4's paired
    inference is defined for binary classification.
    """

    labels = np.asarray(labels).reshape(-1)

    predictions = np.asarray(predictions)

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]

    if predictions.ndim != 1:
        raise ValueError(
            "Binary predictions must have shape (n,) or (n, 1)."
        )

    if len(predictions) != len(labels):
        raise ValueError(
            "Prediction and label counts must match."
        )

    predicted_labels = (
        predictions >= 0.5
    ).astype(np.int64)

    return predicted_labels == labels.astype(np.int64)

def checkpoint_paths(
    checkpoint_root: Path,
    replicate: int,
    condition: str,
) -> tuple[Path, Path]:
    """Return the persistent condition directory and state path.

    The first return value is intentionally a directory rather than a
    single rolling checkpoint: the adapter owns its checkpoint layout
    inside it.  Baseline and perturbed fits of a replicate get disjoint
    directories.
    """
    directory = (
        checkpoint_root
        / f"replicate_{replicate:02d}"
        / condition
    )
    return directory, directory / "state.json"


# ============================================================
# Adapter contract
# ============================================================


class D4Adapter(Protocol):
    """What the shared engine requires of a target adapter.

    ``train`` fits one model on exactly the arrays given, performs the
    adapter's predeclared checkpoint selection, and leaves the selected
    model loaded.  Afterwards ``selection`` is a mapping containing at
    least REQUIRED_SELECTION_FIELDS, where ``train_digest`` is
    ``integrity.labelled_partition_digest(features, labels)`` of the
    arrays passed to ``train``.  ``predict`` returns one score per row
    (probability of class 1, or a hard 0/1 label) from the selected model.

    The engine never passes test data to ``train``; adapters must not be
    constructed with test labels either.
    """

    selection: Mapping[str, Any] | None

    def train(self, features: Any, labels: Any, *, checkpoint_path: Path, state_path: Path,
              replicate: int, condition: str, run_config_hash: str) -> Any: ...

    def predict(self, features: Any) -> Any: ...


REQUIRED_SELECTION_FIELDS = frozenset({
    "condition", "replicate", "seed", "config_hash", "train_digest",
    "rule", "selected_epoch", "checkpoint_sha256",
})


def verify_adapter_selection(adapter: Any, *, condition: str, replicate: int, seed: int,
                             train_digest: str, run_config_hash: str) -> dict:
    """Refuse a fit whose selection record is missing or attributes it elsewhere."""
    selection = getattr(adapter, "selection", None)
    if not isinstance(selection, Mapping):
        raise IntegrityError(
            f"{type(adapter).__name__} did not expose a selection record after train(); "
            "it does not satisfy the D4 adapter contract."
        )
    missing = REQUIRED_SELECTION_FIELDS - set(selection)
    if missing:
        raise IntegrityError(f"Selection record lacks required fields {sorted(missing)}.")
    expected = {"condition": condition, "replicate": replicate, "seed": seed,
                "train_digest": train_digest, "config_hash": run_config_hash}
    for key, value in expected.items():
        if selection[key] != value:
            raise IntegrityError(
                f"Selection record for {condition} replicate {replicate} has {key}="
                f"{selection[key]!r}, expected {value!r}. Refusing to attribute this model."
            )
    return dict(selection)


# ============================================================
# Persisted perturbations
# ============================================================


PERTURBATION_SCHEMA = "d4-perturbation-3"


def _perturbation_binding(*, replicate, seed, perturbation, audit_seed, source_train_digest,
                          source_label_digest) -> dict:
    return {
        "schema_version": PERTURBATION_SCHEMA,
        "perturbation": perturbation.name,
        "kind": "permutation_indices" if perturbation.index_based else
                ("labels" if perturbation.label_only else "features_and_labels"),
        "condition": "perturbed",
        "replicate": int(replicate),
        "seed": int(seed),
        "audit_seed": int(audit_seed),
        "source_train_digest": source_train_digest,
        "source_label_digest": source_label_digest,
    }


def save_perturbation(directory: Path, *, replicate: int, seed: int, perturbation: Perturbation,
                      source_train_digest: str, source_label_digest: str, audit_seed: int,
                      features: Any, labels: Any, indices: Any = None) -> dict:
    """Persist one replicate's perturbation with provenance; never replace anything.

    Index-based perturbations (label shuffle) persist only the permutation
    indices (``replicate_NN_indices.npy``, little-endian int64, a verified
    bijection of range(n)); the perturbed labels are re-derived from them.
    Other perturbations persist their labels, and features unless label-only.
    The sidecar binds the files' SHA-256 to the perturbation, condition,
    replicate, model seed, audit seed and the source training-data and
    training-label digests.
    """
    directory = Path(directory)
    stem = f"replicate_{replicate:02d}"
    sidecar_path = directory / f"{stem}.perturbation.json"
    if perturbation.index_based:
        arrays = {"indices": verify_permutation_indices(indices, len(np.asarray(labels).reshape(-1)))}
    elif perturbation.label_only:
        arrays = {"labels": labels}
    else:
        arrays = {"features": features, "labels": labels}
    paths = {name: directory / f"{stem}_{name}.npy" for name in arrays}
    for path in [sidecar_path, *paths.values()]:
        if path.exists():
            raise IntegrityError(f"Refusing to overwrite persisted perturbation file {path}.")
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, array in arrays.items():
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(np.asarray(array)))
        write_bytes_exclusive(paths[name], buffer.getvalue())
        files[name] = {"file": paths[name].name, "sha256": sha256_file(paths[name]),
                       "dtype": np.asarray(array).dtype.str, "shape": list(np.asarray(array).shape)}
    sidecar = dict(
        _perturbation_binding(replicate=replicate, seed=seed, perturbation=perturbation,
                              audit_seed=audit_seed, source_train_digest=source_train_digest,
                              source_label_digest=source_label_digest),
        files=files,
        perturbed_train_digest=labelled_partition_digest(features, labels),
    )
    write_json_exclusive(sidecar_path, sidecar)
    return sidecar


def load_verified_perturbation(directory: Path, *, replicate: int, seed: int, perturbation: Perturbation,
                               source_train_digest: str, source_label_digest: str, audit_seed: int,
                               train_features: Any, train_labels: Any,
                               expected_indices: Any = None) -> tuple[Any, Any, dict]:
    """Reload a persisted perturbation only if it provably belongs here.

    The sidecar must carry exactly this perturbation, condition, replicate,
    seed, audit seed and source digests; every file must match its digest,
    dtype and shape; permutation indices must be a bijection of range(n)
    and, when ``expected_indices`` is given, equal the permutation the audit
    seed regenerates; and the perturbation's ``verify`` must accept the
    re-derived data.
    """
    directory = Path(directory)
    stem = f"replicate_{replicate:02d}"
    sidecar = read_json(directory / f"{stem}.perturbation.json", f"replicate {replicate} perturbation record")
    expected = _perturbation_binding(replicate=replicate, seed=seed, perturbation=perturbation,
                                     audit_seed=audit_seed, source_train_digest=source_train_digest,
                                     source_label_digest=source_label_digest)
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise IntegrityError(
                f"Persisted perturbation for replicate {replicate} has {key}={sidecar.get(key)!r}, "
                f"expected {value!r}. Refusing to reuse it across conditions, replicates, seeds or datasets."
            )
    names = {"permutation_indices": ("indices",), "labels": ("labels",),
             "features_and_labels": ("features", "labels")}[expected["kind"]]
    if set(sidecar.get("files", {})) != set(names):
        raise IntegrityError(f"Persisted perturbation for replicate {replicate} lists unexpected files.")
    arrays = {}
    for name in names:
        entry = sidecar["files"][name]
        path = directory / str(entry.get("file", ""))
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise IntegrityError(f"Persisted perturbation {name} for replicate {replicate} is missing "
                                 "or does not match its recorded digest.")
        array = np.load(path, mmap_mode="r")
        if array.dtype.str != entry.get("dtype") or list(array.shape) != entry.get("shape"):
            raise IntegrityError(f"Persisted perturbation {name} for replicate {replicate} has the wrong "
                                 "dtype or shape.")
        arrays[name] = array
    if perturbation.index_based:
        indices = verify_permutation_indices(arrays["indices"], len(np.asarray(train_labels).reshape(-1)))
        if expected_indices is not None and not np.array_equal(indices, expected_indices):
            raise IntegrityError(f"Persisted permutation for replicate {replicate} differs from the one "
                                 "the audit seed regenerates. Refusing to resume.")
        features, labels = perturbation.apply_indices(train_features, train_labels, indices)
    else:
        features = train_features if perturbation.label_only else arrays["features"]
        labels = arrays["labels"]
    if labelled_partition_digest(features, labels) != sidecar.get("perturbed_train_digest"):
        raise IntegrityError(f"Persisted perturbation for replicate {replicate} does not reproduce "
                             "its recorded perturbed-data digest.")
    perturbation.verify(train_features, train_labels, features, labels)
    return features, labels, sidecar


def prediction_path(checkpoint_root: Path, replicate: int, condition: str) -> Path:
    return Path(checkpoint_root) / "predictions" / f"replicate_{replicate:02d}_{condition}.npy"


def persist_predictions(path: Path, predictions: Any) -> tuple[np.ndarray, dict]:
    """Write one fit's raw test predictions once; on resume, reuse only an agreeing file.

    A resumed fit is re-predicted.  An existing file is kept (never replaced)
    only if it yields the same thresholded labels as the fresh predictions;
    the kept bytes are then the evidence the counts are computed from.
    """
    fresh = np.ascontiguousarray(np.asarray(predictions).reshape(-1))
    path = Path(path)
    if path.exists():
        stored = np.load(path)
        if stored.shape != fresh.shape or not np.array_equal(stored >= 0.5, fresh >= 0.5):
            raise IntegrityError(f"Persisted predictions {path} disagree with the reloaded model's "
                                 "predictions. Refusing to resume.")
        predictions = stored
    else:
        buffer = io.BytesIO()
        np.save(buffer, fresh)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_bytes_exclusive(path, buffer.getvalue())
        predictions = fresh
    return predictions, {"file": path.name, "sha256": sha256_file(path),
                         "dtype": predictions.dtype.str, "shape": list(predictions.shape)}


def load_completed_state(state_path: Path) -> dict[str, Any] | None:
    if not Path(state_path).exists():
        return None
    return read_json(state_path, "training state")


# ============================================================
# Decision
# ============================================================

#: ``conservative_two_sided_v1`` is the rule the September 2026 run used.
#: ``directional_iut_v2`` tests paired degradation only; it never establishes
#: near-chance accuracy.  ``near_chance_tost_v3`` (below) makes near-chance
#: equivalence the primary endpoint.  Which applies is a preregistration
#: decision, never chosen by this module.
DECISION_RULES = ("conservative_two_sided_v1", "directional_iut_v2", "near_chance_tost_v3")


def mcnemar_one_sided_pvalue(b: int, c: int) -> float:
    """Exact one-sided McNemar p-value for H1: the perturbed model is worse.

    Under H0 the discordant pairs split evenly; a perturbed-worse effect
    makes c (baseline correct, perturbed wrong) large, i.e. b small.
    """
    if b + c == 0:
        return 1.0
    return float(binomtest(int(b), n=int(b + c), p=0.5, alternative="less").pvalue)


def evaluate_decision(rows: Sequence[Mapping[str, Any]], *, rule: str, alpha: float,
                      effect_threshold: float, audit_seed: int, bootstrap_replicates: int,
                      confidence_level: float) -> dict:
    """Pure decision function over replicate rows; used at run and verify time.

    Each row needs baseline_score, perturbed_score, mcnemar_b, mcnemar_c.
    Every p-value is recomputed here from (b, c); nothing self-reported
    is trusted.  ``effect_threshold`` is an absolute accuracy difference
    expressed as a fraction (0.01 = one percentage point).
    """
    if rule not in DECISION_RULES:
        raise ValueError(f"Unknown decision rule {rule!r}; expected one of {DECISION_RULES}.")
    if rule == NEAR_CHANCE_RULE:
        raise ValueError(f"{rule} is evaluated by evaluate_near_chance_decision, not evaluate_decision.")
    k = len(rows)
    if k < 2:
        raise ValueError("At least two independent training replicates are required.")
    diffs = np.asarray([float(r["perturbed_score"]) - float(r["baseline_score"]) for r in rows])
    two_sided = [mcnemar_exact_pvalue_from_counts(int(r["mcnemar_b"]), int(r["mcnemar_c"])) for r in rows]
    one_sided = [mcnemar_one_sided_pvalue(int(r["mcnemar_b"]), int(r["mcnemar_c"])) for r in rows]
    ci_low, ci_high = bootstrap_mean_ci(
        diffs, rng=np.random.default_rng(np.random.SeedSequence([audit_seed, 0xD4])),
        bootstrap_replicates=bootstrap_replicates, confidence_level=confidence_level,
    )
    mean = float(np.mean(diffs))
    if rule == "conservative_two_sided_v1":
        per_test_alpha = alpha / k
        inference = all(p < per_test_alpha for p in two_sided) and (ci_low > 0.0 or ci_high < 0.0)
        practical = abs(mean) >= effect_threshold
        criteria = {"all_two_sided_mcnemar_below_bonferroni_alpha": all(p < per_test_alpha for p in two_sided),
                    "bootstrap_ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                    "abs_mean_difference_at_least_threshold": bool(practical)}
    else:
        # Intersection-union test: H1 (every replicate degrades) is accepted
        # only if each replicate's one-sided test rejects at alpha; the IUT
        # has level alpha without a multiplicity correction.
        per_test_alpha = alpha
        each_significant = all(p < alpha for p in one_sided)
        each_practical = bool(np.all(diffs <= -effect_threshold))
        inference = each_significant
        practical = each_practical
        criteria = {"every_one_sided_mcnemar_below_alpha": each_significant,
                    "every_replicate_degrades_by_at_least_threshold": each_practical}
    detected = bool(inference and practical)
    return {
        "rule": rule,
        "outcome": "EFFECT_DETECTED" if detected else "NO_REPRODUCIBLE_EFFECT_DETECTED",
        "effect_detected": detected,
        "inference_supported": bool(inference),
        "practical_effect": bool(practical),
        "criteria": criteria,
        "per_test_alpha": per_test_alpha,
        "two_sided_pvalues": two_sided,
        "one_sided_pvalues": one_sided,
        "mean_difference": mean,
        "sd_difference": float(np.std(diffs, ddof=1)),
        "bootstrap_ci": [ci_low, ci_high],
    }


def mcnemar_exact_pvalue_from_counts(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return float(binomtest(int(b), n=int(b + c), p=0.5, alternative="two-sided").pvalue)


# ============================================================
# Near-chance primary endpoint (rule near_chance_tost_v3)
# ============================================================
#
# Per perturbed fit r, X_r = number of correct test predictions out of n.
# Conditional on the fitted model and given test rows drawn i.i.d. from the
# generator independently of training, X_r ~ Binomial(n, p_r), where p_r is
# that model's accuracy over the test-generating distribution.  Nothing is
# pooled across fits: all fits share one test sample, so their X_r are
# dependent, and every multiplicity argument below holds without independence.
#
#   Equivalence (fit r):  H0: p_r <= L or p_r >= U   vs   H1: L < p_r < U,
#     L = chance - margin, U = chance + margin.  Two exact one-sided tests
#     (TOST), each at alpha:  P(X >= x | L) < alpha  and  P(X <= x | U) < alpha.
#     Equivalent to the two-sided 1-2*alpha Clopper-Pearson interval lying
#     strictly inside (L, U).  Level alpha (Berger & Hsu 1996).
#   Departure (fit r):  H0: L <= p_r <= U  vs  H1: p_r > U or p_r < L.
#     Exact one-sided tests, each side at alpha / (2k):
#     P(X >= x | U) < alpha/(2k)  (above band)  or  P(X <= x | L) < alpha/(2k).
#   Joint:  SUPPORTED     iff every fit is equivalent (intersection-union
#                         test: level alpha with no adjustment, Berger 1982);
#           NOT_SUPPORTED iff some fit departs (Bonferroni over the 2k
#                         one-sided tests: familywise level alpha);
#           INCONCLUSIVE  otherwise;  NOT_COMPLETED if any fit is missing.
#   Equivalence and departure of one fit are mutually exclusive (asserted).
#
# Failing to reject p = chance is never used: it is not evidence of
# equivalence (methodology 3.9.8, 3.9.9 Rule III).

NEAR_CHANCE_RULE = "near_chance_tost_v3"
PAIRED_ENDPOINT_ROLES = ("secondary", "co_primary")
PRIMARY_OUTCOMES = ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "NOT_COMPLETED")

#: The registered procedure.  A preregistration's fixed design must equal
#: these exactly, and certificates record them; any change is a new procedure.
NEAR_CHANCE_PROCEDURE = {
    "estimand": "per perturbed fit: test accuracy p_r of the reloaded selected model, conditional on that model",
    "chance_reference": 0.5,
    "equivalence_test": "exact_binomial_tost_each_one_sided_test_at_alpha",
    "equivalence_interval": "clopper_pearson_two_sided_1_minus_2alpha_strictly_inside_band",
    "equivalence_aggregation": "intersection_union_every_replicate_equivalent",
    "departure_test": "exact_one_sided_binomial_each_side_at_alpha_over_2k",
    "departure_aggregation": "bonferroni_any_replicate_departs",
    "rejection": "p_value_strictly_less_than_level",
    "outcomes": list(PRIMARY_OUTCOMES),
}
PAIRED_PROCEDURE = {
    "estimand": "per replicate: delta_r = acc(perturbed_r) - acc(baseline_r) = (b - c) / n on the same test rows",
    "discordance": "b = baseline wrong & perturbed right; c = baseline right & perturbed wrong",
    "test": "exact_mcnemar_one_sided_h1_perturbed_worse_p_equals_P(B<=b|b+c,0.5)",
    "zero_discordant": "p_value_1",
    "practical": "delta_r_at_most_minus_effect_threshold",
    "aggregation": "intersection_union_every_replicate_degrades",
    "rejection": "p_value_strictly_less_than_level",
}


def binomial_upper_tail(x: int, n: int, p: float) -> float:
    """P(X >= x) for X ~ Binomial(n, p)."""
    return 1.0 if x <= 0 else float(binom.sf(int(x) - 1, int(n), float(p)))


def binomial_lower_tail(x: int, n: int, p: float) -> float:
    """P(X <= x) for X ~ Binomial(n, p)."""
    return 1.0 if x >= n else float(binom.cdf(int(x), int(n), float(p)))


def clopper_pearson(x: int, n: int, tail: float) -> tuple[float, float]:
    """Exact interval with probability ``tail`` excluded on each side."""
    lower = 0.0 if x == 0 else float(beta.ppf(tail, x, n - x + 1))
    upper = 1.0 if x == n else float(beta.ppf(1.0 - tail, x + 1, n - x))
    return lower, upper


def near_chance_replicate(x: int, n: int, *, chance: float, margin: float, alpha: float,
                          replicates: int) -> dict:
    """Classify one perturbed fit: EQUIVALENT, ABOVE_BAND, BELOW_BAND or INDETERMINATE."""
    if not (isinstance(n, int) and n >= 1 and isinstance(x, int) and 0 <= x <= n):
        raise ValueError(f"Invalid binomial counts x={x!r}, n={n!r}.")
    low, high = chance - margin, chance + margin
    departure_level = alpha / (2 * replicates)
    p_not_below = binomial_upper_tail(x, n, low)    # TOST H01: p <= L
    p_not_above = binomial_lower_tail(x, n, high)   # TOST H02: p >= U
    p_above = binomial_upper_tail(x, n, high)       # departure H0: p <= U
    p_below = binomial_lower_tail(x, n, low)        # departure H0: p >= L
    equivalent = p_not_below < alpha and p_not_above < alpha
    above, below = p_above < departure_level, p_below < departure_level
    if equivalent + above + below > 1:
        raise AssertionError("Equivalence and departure are mutually exclusive; numerical defect.")
    state = ("EQUIVALENT" if equivalent else "ABOVE_BAND" if above else
             "BELOW_BAND" if below else "INDETERMINATE")
    return {
        "test_examples": n, "correct": x, "accuracy": x / n, "band": [low, high],
        "tost_pvalue_h0_at_or_below_lower": p_not_below,
        "tost_pvalue_h0_at_or_above_upper": p_not_above,
        "equivalence_level": alpha,
        "equivalence_interval": list(clopper_pearson(x, n, alpha)),
        "departure_pvalue_above_band": p_above,
        "departure_pvalue_below_band": p_below,
        "departure_level_per_side": departure_level,
        "departure_interval": list(clopper_pearson(x, n, departure_level)),
        "state": state,
    }


def evaluate_near_chance(rows: Sequence[Mapping[str, Any]], *, planned_replicates: int, chance: float,
                         margin: float, alpha: float) -> dict:
    """Primary endpoint over all planned fits; rows need test_examples, perturbed_correct."""
    if not 0.0 < margin < min(chance, 1.0 - chance):
        raise ValueError(f"chance margin {margin!r} must lie strictly inside (0, {min(chance, 1 - chance)}).")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5) for the equivalence test.")
    if len(rows) != planned_replicates:
        return {"outcome": "NOT_COMPLETED", "replicates": [],
                "reason": f"{len(rows)} of {planned_replicates} planned fits are present"}
    per = [near_chance_replicate(int(r["perturbed_correct"]), int(r["test_examples"]), chance=chance,
                                 margin=margin, alpha=alpha, replicates=planned_replicates) for r in rows]
    states = [p["state"] for p in per]
    if all(s == "EQUIVALENT" for s in states):
        outcome = "SUPPORTED"
    elif any(s in ("ABOVE_BAND", "BELOW_BAND") for s in states):
        outcome = "NOT_SUPPORTED"
    else:
        outcome = "INCONCLUSIVE"
    return {"outcome": outcome, "replicates": per, "chance": chance, "margin": margin, "alpha": alpha}


def evaluate_paired_degradation(rows: Sequence[Mapping[str, Any]], *, alpha: float,
                                effect_threshold: float) -> dict:
    """Supporting endpoint: every replicate's perturbed model is worse than its baseline."""
    per = []
    for r in rows:
        b, c, n = int(r["mcnemar_b"]), int(r["mcnemar_c"]), int(r["test_examples"])
        delta = (b - c) / n
        m = b + c
        theta = clopper_pearson(b, m, alpha / 2) if m else (0.5, 0.5)
        p = mcnemar_one_sided_pvalue(b, c)
        per.append({
            "mcnemar_b": b, "mcnemar_c": c, "discordant": m, "delta": delta,
            "one_sided_pvalue": p, "two_sided_pvalue": mcnemar_exact_pvalue_from_counts(b, c),
            "conditional_delta_interval": [m * (2 * theta[0] - 1) / n, m * (2 * theta[1] - 1) / n],
            "significant": p < alpha, "practical": delta <= -effect_threshold,
        })
    established = bool(per) and all(p["significant"] and p["practical"] for p in per)
    return {"established": established, "alpha": alpha, "effect_threshold": effect_threshold,
            "replicates": per}


def evaluate_near_chance_decision(rows: Sequence[Mapping[str, Any]], *, planned_replicates: int,
                                  chance: float, margin: float, alpha: float, effect_threshold: float,
                                  paired_endpoint_role: str) -> dict:
    """Decision of rule near_chance_tost_v3; pure, used at run and verify time.

    secondary:  outcome = primary outcome; the paired endpoint is reported
                beside it at its own level and never changes the outcome.
    co_primary: SUPPORTED needs primary SUPPORTED and paired degradation
                established (intersection-union over all 2k component tests,
                level alpha).  Primary NOT_SUPPORTED gives NOT_SUPPORTED;
                paired degradation that is not established gives
                INCONCLUSIVE, because non-significance is not evidence of
                no degradation.
    """
    if paired_endpoint_role not in PAIRED_ENDPOINT_ROLES:
        raise ValueError(f"paired_endpoint_role must be one of {PAIRED_ENDPOINT_ROLES}.")
    primary = evaluate_near_chance(rows, planned_replicates=planned_replicates, chance=chance,
                                   margin=margin, alpha=alpha)
    if primary["outcome"] == "NOT_COMPLETED":
        return {"rule": NEAR_CHANCE_RULE, "outcome": "NOT_COMPLETED", "primary_claim_supported": False,
                "paired_endpoint_role": paired_endpoint_role, "primary": primary, "supporting": None}
    supporting = evaluate_paired_degradation(rows, alpha=alpha, effect_threshold=effect_threshold)
    outcome = primary["outcome"]
    if paired_endpoint_role == "co_primary" and outcome == "SUPPORTED" and not supporting["established"]:
        outcome = "INCONCLUSIVE"
    return {"rule": NEAR_CHANCE_RULE, "outcome": outcome, "primary_claim_supported": outcome == "SUPPORTED",
            "paired_endpoint_role": paired_endpoint_role, "primary": primary, "supporting": supporting}


# ============================================================
# One repeated experiment
# ============================================================


def run_d4(
    *,
    perturbation: Perturbation,
    train_features: Any,
    train_labels: Any,
    test_features: Any,
    test_labels: Any,
    adapter_factory: Callable[[int], D4Adapter],
    checkpoint_root: Path,
    replicates: int,
    audit_seed: int,
    effect_threshold: float,
    bootstrap_replicates: int = 5000,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
    run_config_hash: str | None = None,
    decision_rule: str = "conservative_two_sided_v1",
    chance_margin: float | None = None,
    paired_endpoint_role: str | None = None,
    chance_level: float = 0.5,
) -> D4Result:
    """Execute repeated, paired controlled-perturbation inference.

    For each replicate r (seed spawned from ``audit_seed``): fit a fresh
    adapter on the original training data ("baseline") and a fresh adapter
    with the same seed on the perturbed training data ("perturbed"); the
    two fits differ only in the perturbation.  Both selected models are
    evaluated on the same fixed test partition, which is never given to
    ``train``.  Each fit's selection record is checked against the
    condition, replicate, seed, configuration and the digest of exactly
    the training data it was given.

    No failed replicate is retried or excluded here: any exception aborts
    the run, and a resumed run must reproduce the identical replicate
    plan.  ``effect_threshold`` is an accuracy difference as a fraction.

    Under ``near_chance_tost_v3``, ``chance_margin`` (the registered
    near-chance equivalence margin, a fraction of accuracy, separate from
    ``effect_threshold``) and ``paired_endpoint_role`` are required.
    """
    if decision_rule == NEAR_CHANCE_RULE:
        if not (isinstance(chance_margin, float) and 0.0 < chance_margin < min(chance_level, 1 - chance_level)):
            raise ValueError("near_chance_tost_v3 needs a registered chance_margin in (0, 0.5); "
                             "it is never defaulted or taken from effect_threshold.")
        if paired_endpoint_role not in PAIRED_ENDPOINT_ROLES:
            raise ValueError(f"near_chance_tost_v3 needs paired_endpoint_role in {PAIRED_ENDPOINT_ROLES}.")
    elif chance_margin is not None or paired_endpoint_role is not None:
        raise ValueError("chance_margin and paired_endpoint_role apply only to near_chance_tost_v3.")
    if replicates < 2:
        raise ValueError("At least two independent training replicates are required.")
    if effect_threshold < 0:
        raise ValueError("effect_threshold must be non-negative.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    if decision_rule not in DECISION_RULES:
        raise ValueError(f"Unknown decision rule {decision_rule!r}.")
    if not run_config_hash:
        raise ValueError("run_config_hash is required: every fit must be bound to a configuration.")
    if type(perturbation).verify is Perturbation.verify:
        raise TypeError(f"Perturbation {perturbation.name!r} does not implement verify(); "
                        "the engine cannot check its definition and refuses to run it.")

    checkpoint_root = Path(checkpoint_root)
    source_digest = labelled_partition_digest(train_features, train_labels)
    label_digest = partition_digest({"labels": train_labels})
    child_sequences = np.random.SeedSequence(audit_seed).spawn(replicates)
    replicate_results: list[ReplicateResult] = []

    for replicate_index, child_sequence in enumerate(child_sequences, start=1):
        rng = np.random.default_rng(child_sequence)
        seed_value = int(child_sequence.generate_state(1, dtype=np.uint32)[0])
        print(f"\nRunning D4 replicate {replicate_index}/{replicates} (seed={seed_value})...")

        def fit(features, labels, condition):
            adapter = adapter_factory(seed_value)
            checkpoint, state = checkpoint_paths(checkpoint_root, replicate_index, condition)
            adapter.train(features, labels, checkpoint_path=checkpoint, state_path=state,
                          replicate=replicate_index, condition=condition,
                          run_config_hash=run_config_hash)
            selection = verify_adapter_selection(
                adapter, condition=condition, replicate=replicate_index, seed=seed_value,
                train_digest=labelled_partition_digest(features, labels),
                run_config_hash=run_config_hash,
            )
            predictions, record = persist_predictions(
                prediction_path(checkpoint_root, replicate_index, condition), adapter.predict(test_features))
            correct = binary_predictions_to_correctness(predictions, test_labels)
            return correct, selection, record

        baseline_correct, baseline_selection, baseline_predictions = fit(train_features, train_labels, "baseline")

        perturbation_directory = checkpoint_root / "perturbations"
        sidecar_path = perturbation_directory / f"replicate_{replicate_index:02d}.perturbation.json"
        # The perturbation RNG is always drawn, so the stream does not depend
        # on whether this is a fresh run or a resume.
        if perturbation.index_based:
            fresh_indices = perturbation.permutation_indices(len(np.asarray(train_labels).reshape(-1)), rng)
            fresh_features, fresh_labels = perturbation.apply_indices(train_features, train_labels, fresh_indices)
        else:
            fresh_indices = None
            fresh_features, fresh_labels = perturbation.apply(train_features, train_labels, rng=rng)
        binding = dict(replicate=replicate_index, seed=seed_value, perturbation=perturbation,
                       source_train_digest=source_digest, source_label_digest=label_digest,
                       audit_seed=audit_seed)
        if sidecar_path.exists():
            perturbed_features, perturbed_labels, perturbation_record = load_verified_perturbation(
                perturbation_directory, train_features=train_features, train_labels=train_labels,
                expected_indices=fresh_indices, **binding,
            )
            if labelled_partition_digest(fresh_features, fresh_labels) != perturbation_record["perturbed_train_digest"]:
                raise IntegrityError(
                    f"Persisted perturbation for replicate {replicate_index} differs from the one "
                    "the audit seed regenerates. Refusing to resume."
                )
        else:
            perturbation.verify(train_features, train_labels, fresh_features, fresh_labels)
            perturbation_record = save_perturbation(
                perturbation_directory, features=fresh_features, labels=fresh_labels,
                indices=fresh_indices, **binding,
            )
            perturbed_features, perturbed_labels = fresh_features, fresh_labels

        perturbed_correct, perturbed_selection, perturbed_predictions = fit(
            perturbed_features, perturbed_labels, "perturbed")

        b, c, p_value = mcnemar_exact_pvalue(baseline_correct, perturbed_correct)
        baseline_score = float(np.mean(baseline_correct))
        perturbed_score = float(np.mean(perturbed_correct))
        difference = perturbed_score - baseline_score
        replicate_results.append(ReplicateResult(
            replicate=replicate_index,
            seed=seed_value,
            baseline_score=baseline_score,
            perturbed_score=perturbed_score,
            absolute_difference=difference,
            relative_difference_percent=(float("nan") if baseline_score == 0.0
                                         else difference / baseline_score * 100.0),
            mcnemar_b=b,
            mcnemar_c=c,
            mcnemar_pvalue=p_value,
            baseline_selection=baseline_selection,
            perturbed_selection=perturbed_selection,
            perturbation_record=perturbation_record,
            test_examples=int(len(baseline_correct)),
            baseline_correct=int(np.sum(baseline_correct)),
            perturbed_correct=int(np.sum(perturbed_correct)),
            baseline_predictions=baseline_predictions,
            perturbed_predictions=perturbed_predictions,
        ))

    rows = [{"baseline_score": r.baseline_score, "perturbed_score": r.perturbed_score,
             "mcnemar_b": r.mcnemar_b, "mcnemar_c": r.mcnemar_c, "test_examples": r.test_examples,
             "perturbed_correct": r.perturbed_correct} for r in replicate_results]
    if decision_rule == NEAR_CHANCE_RULE:
        decision = evaluate_near_chance_decision(
            rows, planned_replicates=replicates, chance=chance_level, margin=chance_margin, alpha=alpha,
            effect_threshold=effect_threshold, paired_endpoint_role=paired_endpoint_role)
        diffs = [r["perturbed_score"] - r["baseline_score"] for r in rows]
        ci_low, ci_high = bootstrap_mean_ci(
            diffs, rng=np.random.default_rng(np.random.SeedSequence([audit_seed, 0xD4])),
            bootstrap_replicates=bootstrap_replicates, confidence_level=confidence_level)
        decision.update(mean_difference=float(np.mean(diffs)), sd_difference=float(np.std(diffs, ddof=1)),
                        bootstrap_ci=[ci_low, ci_high], per_test_alpha=alpha,
                        effect_detected=decision["supporting"]["established"],
                        inference_supported=decision["primary_claim_supported"])
    else:
        decision = evaluate_decision(rows, rule=decision_rule, alpha=alpha,
                                     effect_threshold=effect_threshold, audit_seed=audit_seed,
                                     bootstrap_replicates=bootstrap_replicates,
                                     confidence_level=confidence_level)
    pvalues = [r.mcnemar_pvalue for r in replicate_results]
    return D4Result(
        perturbation=perturbation.name,
        replicates=replicate_results,
        mean_baseline_score=float(np.mean([r.baseline_score for r in replicate_results])),
        mean_perturbed_score=float(np.mean([r.perturbed_score for r in replicate_results])),
        mean_absolute_difference=decision["mean_difference"],
        sd_absolute_difference=decision["sd_difference"],
        mean_relative_difference_percent=float(np.nanmean(
            [r.relative_difference_percent for r in replicate_results])),
        bootstrap_ci_low=decision["bootstrap_ci"][0],
        bootstrap_ci_high=decision["bootstrap_ci"][1],
        min_mcnemar_pvalue=float(np.min(pvalues)),
        max_mcnemar_pvalue=float(np.max(pvalues)),
        adjusted_alpha=decision["per_test_alpha"],
        effect_threshold=effect_threshold,
        effect_detected=decision["effect_detected"],
        inference_supported=decision["inference_supported"],
        interpretation=(
            NEAR_CHANCE_INTERPRETATION[decision["outcome"]] if decision_rule == NEAR_CHANCE_RULE else
            "A reproducible performance effect was detected under the specified perturbation; "
            "every pre-specified criterion of the decision rule was satisfied."
            if decision["effect_detected"] else
            "The experiment did not satisfy all pre-specified criteria for a reproducible "
            "performance effect."
        ),
        decision=decision,
        settings={"alpha": alpha, "audit_seed": audit_seed, "bootstrap_replicates": bootstrap_replicates,
                  "confidence_level": confidence_level, "decision_rule": decision_rule,
                  "replicates": replicates, "run_config_hash": run_config_hash,
                  "source_train_digest": source_digest,
                  **({"chance_level": chance_level, "chance_margin": chance_margin,
                      "paired_endpoint_role": paired_endpoint_role, "effect_threshold": effect_threshold,
                      "test_examples": int(len(np.asarray(test_labels).reshape(-1)))}
                     if decision_rule == NEAR_CHANCE_RULE else {})},
    )


NEAR_CHANCE_INTERPRETATION = {
    "SUPPORTED": "Every label-shuffled fit's test accuracy is established, at the registered level, to lie "
                 "inside the registered near-chance band, conditional on each fitted model and the shared "
                 "test sample. This is a blunt control: it does not show that no leakage of any form exists.",
    "NOT_SUPPORTED": "At least one label-shuffled fit's test accuracy is established to lie outside the "
                     "registered near-chance band (familywise level alpha). The pipeline produced non-chance "
                     "performance without the original feature/label correspondence; this calls for "
                     "targeted leakage investigation.",
    "INCONCLUSIVE": "Neither near-chance equivalence of every label-shuffled fit nor a departure of any fit "
                    "from the band was established (or, with a co-primary paired endpoint, paired degradation "
                    "was not established). This is not evidence of near-chance behaviour.",
    "NOT_COMPLETED": "Not every planned fit completed; no scientific conclusion is drawn.",
}


# ============================================================
# Certificate
# ============================================================


def generate_certificate(
    result: D4Result,
    *,
    dataset_id: str,
    dataset_version: str,
    generation_procedure: str,
    generation_parameters: Mapping[str, Any],
    audit_seed: int,
    confidence_level: float,
    alpha: float,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a machine-readable D4 certificate.

    ``provenance`` (preregistration binding, source identity, partition
    digests, overlap audit, run identity) is copied verbatim; the
    post-run verifier checks it against the files on disk.
    """
    decision = result.decision or {}
    if decision.get("rule") == NEAR_CHANCE_RULE:
        return _near_chance_certificate(result, dataset_id=dataset_id, dataset_version=dataset_version,
                                        generation_procedure=generation_procedure,
                                        generation_parameters=generation_parameters, audit_seed=audit_seed,
                                        confidence_level=confidence_level, provenance=provenance)
    return {
        "schema_version": "d4-certificate-3",
        "audit": {
            "id": "D4",
            "name": "Controlled Perturbation Audit",
            "claim": (
                "Whether the specified controlled perturbation produces a reproducible change in "
                "predictive performance under the stated experimental and inferential design."
            ),
        },
        "decision": {
            "outcome": "EFFECT_DETECTED" if result.effect_detected else "NO_REPRODUCIBLE_EFFECT_DETECTED",
            "effect_detected": result.effect_detected,
            "inference_supported": result.inference_supported,
            "rule": decision.get("rule"),
            "criteria": decision.get("criteria"),
        },
        "settings": dict(result.settings or {}),
        "findings": {
            "perturbation": result.perturbation,
            "replicates": [
                {
                    "replicate": r.replicate,
                    "seed": r.seed,
                    "test_examples": r.test_examples,
                    "baseline_accuracy": r.baseline_score,
                    "perturbed_accuracy": r.perturbed_score,
                    "accuracy_difference": r.absolute_difference,
                    "relative_difference_percent": r.relative_difference_percent,
                    "mcnemar_b": r.mcnemar_b,
                    "mcnemar_c": r.mcnemar_c,
                    "mcnemar_pvalue": r.mcnemar_pvalue,
                    "mcnemar_one_sided_pvalue": mcnemar_one_sided_pvalue(r.mcnemar_b, r.mcnemar_c),
                    "baseline_selection": r.baseline_selection,
                    "perturbed_selection": r.perturbed_selection,
                    "perturbation_record": r.perturbation_record,
                }
                for r in result.replicates
            ],
            "mean_baseline_accuracy": result.mean_baseline_score,
            "mean_perturbed_accuracy": result.mean_perturbed_score,
            "mean_accuracy_difference": result.mean_absolute_difference,
            "sd_accuracy_difference": result.sd_absolute_difference,
            "mean_relative_difference_percent": result.mean_relative_difference_percent,
            "bootstrap_ci": {"confidence_level": confidence_level,
                             "low": result.bootstrap_ci_low, "high": result.bootstrap_ci_high},
            "mcnemar": {"minimum_pvalue": result.min_mcnemar_pvalue,
                        "maximum_pvalue": result.max_mcnemar_pvalue,
                        "familywise_alpha": alpha, "per_test_alpha": result.adjusted_alpha},
        },
        "methodology": {
            "unit_of_independent_replication": "independent model training replicate",
            "evaluation_design": "same fixed held-out test partition for baseline and perturbed models",
            "paired_test": "exact McNemar (binomial on discordant pairs); sidedness set by the rule",
            "effect_size": "perturbed minus baseline accuracy, as a fraction",
            "practical_effect_threshold": result.effect_threshold,
            "confidence_interval": "percentile bootstrap over independent training replicates",
            "decision_rule": decision.get("rule"),
        },
        "dataset": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "generation_procedure": generation_procedure,
            "generation_parameters": dict(generation_parameters),
        },
        "reproducibility": {"audit_seed": audit_seed},
        "provenance": dict(provenance or {}),
        "interpretation": result.interpretation,
        "limitations": [
            "A label-shuffle effect shows dependence on the feature/label correspondence; it is "
            "not by itself evidence of cryptographic structure.",
            "McNemar inference is conditional on each fitted model pair; replicate-to-replicate "
            "training variability enters only through the replicate-level criteria.",
            "With few replicates the percentile bootstrap interval has poor coverage and is "
            "descriptive unless the decision rule uses it.",
        ],
    }


NEAR_CHANCE_CERTIFICATE_SCHEMA = "d4-certificate-4"

#: What a v4 certificate lets a third party check, and what it does not.
VERIFIABILITY = {
    "recomputed_from_persisted_evidence": [
        "per-fit correct counts and McNemar (b, c), from the persisted raw predictions and the test labels",
        "every p-value, interval, per-fit state, endpoint result and the outcome, from those counts",
        "the test partition, by its SHA-256 and canonical digest recorded at preflight",
        "the identity of each selected checkpoint, by SHA-256, and its validation-loss selection history",
    ],
    "not_recomputed": [
        "that the persisted predictions are what the selected checkpoint outputs: verify does not rerun "
        "the network (that needs TensorFlow and the run machine); it checks the checkpoint digests and the "
        "prediction digests only",
        "the data generator's assumptions (i.i.d. rows, fair labels), which are D3's subject",
    ],
}


def _near_chance_certificate(result: D4Result, *, dataset_id, dataset_version, generation_procedure,
                             generation_parameters, audit_seed, confidence_level, provenance) -> dict:
    decision = result.decision
    settings = dict(result.settings)
    return {
        "schema_version": NEAR_CHANCE_CERTIFICATE_SCHEMA,
        "audit": {
            "id": "D4", "name": "Controlled Perturbation Audit",
            "claim": ("Primary: whether every label-shuffled fit's test accuracy is practically equivalent to "
                      "chance (inside the registered band), conditional on each fitted model and the shared "
                      "test sample. Supporting: whether every shuffled fit is less accurate than its paired "
                      "baseline on the same test rows."),
        },
        "decision": {
            "rule": decision["rule"],
            "outcome": decision["outcome"],
            "primary_claim_supported": decision["primary_claim_supported"],
            "paired_endpoint_role": decision["paired_endpoint_role"],
            "primary_outcome": decision["primary"]["outcome"],
            "supporting_degradation_established": decision["supporting"]["established"],
        },
        "settings": settings,
        "findings": {
            "perturbation": result.perturbation,
            "replicates": [
                {
                    "replicate": r.replicate, "seed": r.seed, "test_examples": r.test_examples,
                    "baseline_correct": r.baseline_correct, "perturbed_correct": r.perturbed_correct,
                    "baseline_accuracy": r.baseline_score, "perturbed_accuracy": r.perturbed_score,
                    "accuracy_difference": r.absolute_difference,
                    "mcnemar_b": r.mcnemar_b, "mcnemar_c": r.mcnemar_c, "mcnemar_pvalue": r.mcnemar_pvalue,
                    "mcnemar_one_sided_pvalue": mcnemar_one_sided_pvalue(r.mcnemar_b, r.mcnemar_c),
                    "baseline_selection": r.baseline_selection, "perturbed_selection": r.perturbed_selection,
                    "perturbation_record": r.perturbation_record,
                    "baseline_predictions": r.baseline_predictions,
                    "perturbed_predictions": r.perturbed_predictions,
                }
                for r in result.replicates
            ],
            "primary_endpoint": decision["primary"],
            "supporting_endpoint": decision["supporting"],
            "descriptive": {
                "mean_baseline_accuracy": result.mean_baseline_score,
                "mean_perturbed_accuracy": result.mean_perturbed_score,
                "mean_accuracy_difference": decision["mean_difference"],
                "sd_accuracy_difference": decision["sd_difference"],
                "bootstrap_ci_mean_difference": {"confidence_level": confidence_level,
                                                 "low": decision["bootstrap_ci"][0],
                                                 "high": decision["bootstrap_ci"][1],
                                                 "role": "descriptive only; not used by the decision"},
            },
        },
        "methodology": {
            "primary_procedure": NEAR_CHANCE_PROCEDURE,
            "supporting_procedure": PAIRED_PROCEDURE,
            "chance_margin": settings["chance_margin"],
            "chance_margin_units": "accuracy_fraction",
            "effect_threshold": settings["effect_threshold"],
            "alpha": settings["alpha"],
            "paired_endpoint_role": settings["paired_endpoint_role"],
            "unit_of_inference": "each fitted model, conditional on it; the k fits share one test sample and "
                                 "are never pooled or treated as independent test sets",
            "verifiability": VERIFIABILITY,
        },
        "dataset": {"dataset_id": dataset_id, "dataset_version": dataset_version,
                    "generation_procedure": generation_procedure,
                    "generation_parameters": dict(generation_parameters)},
        "reproducibility": {"audit_seed": audit_seed},
        "provenance": dict(provenance or {}),
        "interpretation": result.interpretation,
        "limitations": [
            "Label shuffling is a blunt control: it removes genuine signal and label-correlated artifacts "
            "together. Near-chance accuracy does not show that no leakage of any form exists; targeted "
            "generation-order and label-leakage checks are a separate, complementary audit not performed here.",
            "Inference is conditional on each fitted model and assumes test rows i.i.d. from the generator, "
            "independent of training. It is a statement about these k fits, not a population estimate over "
            "training seeds.",
            "The chance reference 0.5 assumes labels are fair coin flips independent of the features under "
            "the generator; the observed test class balance is recorded but not tested here.",
            "A near-chance result is informative only if the baseline pipeline produces signal; the paired "
            "endpoint reports this, and gates the outcome only when registered as co-primary.",
        ],
    }


def _recompute_near_chance(certificate: Mapping[str, Any]) -> dict:
    settings = certificate.get("settings") or {}
    methodology = certificate.get("methodology") or {}
    findings = certificate.get("findings") or {}
    if methodology.get("primary_procedure") != NEAR_CHANCE_PROCEDURE or \
            methodology.get("supporting_procedure") != PAIRED_PROCEDURE:
        raise IntegrityError("Certificate records a procedure other than the implemented near_chance_tost_v3.")
    for key in ("chance_margin", "effect_threshold", "alpha", "paired_endpoint_role"):
        if methodology.get(key) != settings.get(key):
            raise IntegrityError(f"Certificate methodology {key} differs from its settings.")
    rows = findings.get("replicates") or []
    k = settings.get("replicates")
    if len(rows) != k or [r.get("replicate") for r in rows] != list(range(1, len(rows) + 1)):
        raise IntegrityError(f"Certificate has {len(rows)} replicate rows; settings declare {k}. "
                             "A missing replicate is not allowed.")
    n = settings.get("test_examples")
    for r in rows:
        counts = [r.get(f) for f in ("test_examples", "baseline_correct", "perturbed_correct", "mcnemar_b",
                                     "mcnemar_c")]
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts):
            raise IntegrityError(f"Replicate {r.get('replicate')}: counts must be non-negative integers.")
        total, base, pert, b, c = counts
        if total != n or total < 1:
            raise IntegrityError(f"Replicate {r['replicate']}: test_examples {total} differs from settings {n}.")
        both_correct = base - c
        if not (0 <= both_correct and both_correct + b + c <= total and pert == both_correct + b):
            raise IntegrityError(f"Replicate {r['replicate']}: counts are not a consistent paired 2x2 table.")
        if r["baseline_accuracy"] != base / total or r["perturbed_accuracy"] != pert / total:
            raise IntegrityError(f"Replicate {r['replicate']}: accuracies do not equal counts / n.")
        if r["accuracy_difference"] != r["perturbed_accuracy"] - r["baseline_accuracy"]:
            raise IntegrityError(f"Replicate {r['replicate']}: accuracy difference is inconsistent.")
        if mcnemar_exact_pvalue_from_counts(b, c) != r["mcnemar_pvalue"] or \
                mcnemar_one_sided_pvalue(b, c) != r["mcnemar_one_sided_pvalue"]:
            raise IntegrityError(f"Replicate {r['replicate']}: McNemar p-value does not match (b, c).")
    decision = evaluate_near_chance_decision(
        [{"test_examples": r["test_examples"], "perturbed_correct": r["perturbed_correct"],
          "mcnemar_b": r["mcnemar_b"], "mcnemar_c": r["mcnemar_c"]} for r in rows],
        planned_replicates=k, chance=settings.get("chance_level"), margin=settings.get("chance_margin"),
        alpha=settings.get("alpha"), effect_threshold=settings.get("effect_threshold"),
        paired_endpoint_role=settings.get("paired_endpoint_role"))
    if findings.get("primary_endpoint") != decision["primary"] or \
            findings.get("supporting_endpoint") != decision["supporting"]:
        raise IntegrityError("Certificate endpoint statistics do not match the values recomputed from its counts.")
    recorded = certificate.get("decision") or {}
    expected = {"rule": decision["rule"], "outcome": decision["outcome"],
                "primary_claim_supported": decision["primary_claim_supported"],
                "paired_endpoint_role": decision["paired_endpoint_role"],
                "primary_outcome": decision["primary"]["outcome"],
                "supporting_degradation_established": decision["supporting"]["established"]}
    if recorded != expected:
        raise IntegrityError(f"Certificate decision {recorded.get('outcome')!r} does not match the "
                             f"recomputed decision {decision['outcome']!r}.")
    if certificate.get("interpretation") != NEAR_CHANCE_INTERPRETATION[decision["outcome"]]:
        raise IntegrityError("Certificate interpretation does not match the recomputed outcome.")
    return decision


def recompute_certificate_decision(certificate: Mapping[str, Any]) -> dict:
    """Independently re-derive the decision from a certificate's replicate rows.

    Recomputes every p-value from (b, c) and every difference from the
    accuracies, then applies the recorded rule and settings.  Raises
    IntegrityError if the certificate's decision or statistics disagree.
    A near_chance_tost_v3 certificate is recomputed from its integer counts.
    """
    if certificate.get("schema_version") == NEAR_CHANCE_CERTIFICATE_SCHEMA or \
            (certificate.get("settings") or {}).get("decision_rule") == NEAR_CHANCE_RULE:
        if certificate.get("schema_version") != NEAR_CHANCE_CERTIFICATE_SCHEMA or \
                (certificate.get("settings") or {}).get("decision_rule") != NEAR_CHANCE_RULE:
            raise IntegrityError("Certificate schema and decision rule disagree.")
        return _recompute_near_chance(certificate)
    settings = certificate.get("settings") or {}
    rows = certificate.get("findings", {}).get("replicates") or []
    if len(rows) != settings.get("replicates"):
        raise IntegrityError(f"Certificate has {len(rows)} replicate rows; settings declare "
                             f"{settings.get('replicates')}. A missing replicate is not allowed.")
    if [r.get("replicate") for r in rows] != list(range(1, len(rows) + 1)):
        raise IntegrityError("Certificate replicate rows are not exactly 1..k in order.")
    for r in rows:
        if r.get("test_examples", 0) < 1:
            raise IntegrityError(f"Replicate {r.get('replicate')} has no test examples recorded.")
        if not abs(r["accuracy_difference"] - (r["perturbed_accuracy"] - r["baseline_accuracy"])) <= 1e-12:
            raise IntegrityError(f"Replicate {r['replicate']}: accuracy difference is inconsistent.")
        if mcnemar_exact_pvalue_from_counts(r["mcnemar_b"], r["mcnemar_c"]) != r["mcnemar_pvalue"]:
            raise IntegrityError(f"Replicate {r['replicate']}: McNemar p-value does not match (b, c).")
    decision = evaluate_decision(
        [{"baseline_score": r["baseline_accuracy"], "perturbed_score": r["perturbed_accuracy"],
          "mcnemar_b": r["mcnemar_b"], "mcnemar_c": r["mcnemar_c"]} for r in rows],
        rule=settings.get("decision_rule"), alpha=settings.get("alpha"),
        effect_threshold=certificate["methodology"]["practical_effect_threshold"],
        audit_seed=settings.get("audit_seed"), bootstrap_replicates=settings.get("bootstrap_replicates"),
        confidence_level=settings.get("confidence_level"),
    )
    recorded = certificate.get("decision", {})
    if (recorded.get("outcome") != decision["outcome"] or recorded.get("rule") != decision["rule"]
            or recorded.get("criteria") != decision["criteria"]):
        raise IntegrityError(f"Certificate decision {recorded.get('outcome')!r} does not match the "
                             f"recomputed decision {decision['outcome']!r}.")
    return decision


# ============================================================
# Report
# ============================================================

def print_report(
    result: D4Result,
) -> None:
    """Print the D4 statistical report."""

    print("=" * 72)
    print("Dataset Integrity Audit")
    print("D4 - Controlled Perturbation Audit")
    print("=" * 72)
    print()

    print(
        f"Perturbation                 : "
        f"{result.perturbation}"
    )

    print(
        f"Independent replicates      : "
        f"{len(result.replicates)}"
    )

    print()

    print("Aggregate performance")
    print("-" * 72)

    print(
        f"Mean baseline accuracy      : "
        f"{result.mean_baseline_score:.8f}"
    )

    print(
        f"Mean perturbed accuracy     : "
        f"{result.mean_perturbed_score:.8f}"
    )

    print(
        f"Mean accuracy difference    : "
        f"{result.mean_absolute_difference:.8f}"
    )

    print(
        f"SD accuracy difference      : "
        f"{result.sd_absolute_difference:.8f}"
    )

    print(
        f"Mean relative difference    : "
        f"{result.mean_relative_difference_percent:.4f}%"
    )

    print()

    print("Replicate-level paired inference")
    print("-" * 72)

    for replicate in result.replicates:
        print(
            f"Replicate {replicate.replicate:2d} | "
            f"baseline={replicate.baseline_score:.6f} | "
            f"perturbed={replicate.perturbed_score:.6f} | "
            f"diff={replicate.absolute_difference:+.6f} | "
            f"McNemar p={replicate.mcnemar_pvalue:.6g}"
        )

    print()

    print("Replicate-level bootstrap")
    print("-" * 72)

    print(
        f"95% CI for mean accuracy difference: "
        f"[{result.bootstrap_ci_low:.8f}, "
        f"{result.bootstrap_ci_high:.8f}]"
    )

    print()

    print("Decision")
    print("-" * 72)

    decision = result.decision or {}
    if decision.get("rule") == NEAR_CHANCE_RULE:
        primary = decision["primary"]
        print(f"Rule                          : {NEAR_CHANCE_RULE} "
              f"(paired endpoint {decision['paired_endpoint_role']})")
        print(f"Near-chance band              : [{primary['chance'] - primary['margin']}, {primary['chance'] + primary['margin']}]")
        for r, row in enumerate(primary["replicates"], start=1):
            print(f"Shuffled fit {r:2d}               : acc={row['accuracy']:.6f} "
                  f"{100 * (1 - 2 * row['equivalence_level']):.0f}% CP={row['equivalence_interval']} "
                  f"-> {row['state']}")
        print(f"Primary endpoint              : {primary['outcome']}")
        print(f"Paired degradation established: {decision['supporting']['established']}")
        print(f"OUTCOME                       : {decision['outcome']}")
        print(f"Interpretation                : {result.interpretation}")
        print("=" * 72)
        return

    print(
        f"Practical-effect threshold    : "
        f"{result.effect_threshold:.8f}"
    )

    print(
        f"Bonferroni-adjusted alpha     : "
        f"{result.adjusted_alpha:.8g}"
    )

    print(
        f"Paired inference supported    : "
        f"{result.inference_supported}"
    )

    print(
        f"Effect detected               : "
        f"{result.effect_detected}"
    )

    print(
        f"Interpretation                : "
        f"{result.interpretation}"
    )

    print("=" * 72)

# ============================================================
# Preregistration content (target-agnostic part)
# ============================================================

PENDING = "PENDING_HUMAN_APPROVAL"
PREREGISTRATION_SCHEMA = "d4-preregistration-1"
REQUIRED_PREREGISTRATION_SECTIONS = (
    "protocol_id", "question", "hypothesis", "estimand", "conditions", "perturbation",
    "fixed_design", "decisions", "outcomes", "provenance_requirements", "scope_and_limitations",
)
#: Decisions every D4 protocol must settle; targets may add more.
REQUIRED_DECISIONS = (
    "decision_rule", "alpha", "effect_threshold", "replicates", "partition_overlap_policy",
    "within_partition_duplicates_policy", "failure_handling", "calibration",
)
#: Further decisions a rule needs whenever it is the chosen rule or one of the options.
RULE_REQUIRED_DECISIONS = {"near_chance_tost_v3": ("chance_margin", "paired_endpoint_role")}


def review_preregistration(record: Mapping[str, Any]) -> dict:
    """Structural review of a D4 preregistration; never raises on content.

    Returns {"problems": [...], "pending": [...], "values": {...}}.
    ``problems`` are structural defects (the draft itself is wrong);
    ``pending`` lists every field still awaiting a human decision.
    """
    problems, pending, values = [], [], {}
    if record.get("schema_version") != PREREGISTRATION_SCHEMA:
        problems.append(f"schema_version is {record.get('schema_version')!r}, expected {PREREGISTRATION_SCHEMA!r}")
    for section in REQUIRED_PREREGISTRATION_SECTIONS:
        if not record.get(section):
            problems.append(f"section {section!r} is missing or empty")
    decisions = record.get("decisions") or {}
    for name in REQUIRED_DECISIONS:
        if name not in decisions:
            problems.append(f"decision {name!r} is missing")
    for name, entry in decisions.items():
        if not isinstance(entry, Mapping) or "status" not in entry or "value" not in entry:
            problems.append(f"decision {name!r} lacks status/value")
            continue
        if not entry.get("rationale"):
            problems.append(f"decision {name!r} has no rationale")
        if entry["status"] == PENDING or entry["value"] is None:
            pending.append(f"decisions.{name}")
            continue
        if entry["status"] != "APPROVED":
            problems.append(f"decision {name!r} has status {entry['status']!r}")
        options = entry.get("options")
        if options is not None and entry["value"] not in options:
            problems.append(f"decision {name!r} value {entry['value']!r} is not one of {options}")
        values[name] = entry["value"]
    if "decision_rule" in values and values["decision_rule"] not in DECISION_RULES:
        problems.append(f"decision_rule {values['decision_rule']!r} is not implemented")
    rule_entry = decisions.get("decision_rule") if isinstance(decisions.get("decision_rule"), Mapping) else {}
    candidate_rules = set(rule_entry.get("options") or []) | {rule_entry.get("value")}
    for rule, names in RULE_REQUIRED_DECISIONS.items():
        if rule in candidate_rules:
            problems += [f"decision {name!r} is missing (required by {rule})" for name in names
                         if name not in decisions]
    margin_entry = decisions.get("chance_margin")
    if isinstance(margin_entry, Mapping):
        if margin_entry.get("units") != "accuracy_fraction":
            problems.append("decision 'chance_margin' must declare units 'accuracy_fraction'")
        if "chance_margin" in values:
            margin = values["chance_margin"]
            if not (isinstance(margin, float) and 0.0 < margin < 0.5):
                problems.append(f"decisions.chance_margin {margin!r} must be a float in (0, 0.5)")
    if values.get("decision_rule") == NEAR_CHANCE_RULE and "paired_endpoint_role" in values and \
            values["paired_endpoint_role"] not in PAIRED_ENDPOINT_ROLES:
        problems.append(f"decisions.paired_endpoint_role must be one of {PAIRED_ENDPOINT_ROLES}")
    for field_name in ("approval_statement", "data_examined_before_approval", "approved_by", "approved_utc"):
        value = record.get(field_name)
        if value in (None, "", PENDING):
            pending.append(field_name)
    if record.get("approved") is not True:
        pending.append("approved")
    return {"problems": problems, "pending": pending, "values": values}


def require_approved_preregistration(record: Mapping[str, Any]) -> dict:
    """Return the approved decision values, or raise IntegrityError."""
    review = review_preregistration(record)
    if review["problems"]:
        raise IntegrityError(f"Preregistration is malformed: {review['problems']}")
    if review["pending"]:
        raise IntegrityError(f"Preregistration is not approved; pending: {review['pending']}")
    return review["values"]
