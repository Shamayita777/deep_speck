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
from scipy.stats import binomtest

from audit.dataset.integrity import (
    IntegrityError,
    labelled_partition_digest,
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


def save_perturbation(directory: Path, *, replicate: int, perturbation: Perturbation,
                      source_train_digest: str, audit_seed: int, features: Any, labels: Any) -> dict:
    """Persist one replicate's perturbed training data with provenance.

    Writes ``replicate_NN_labels.npy`` (and ``replicate_NN_features.npy``
    unless the perturbation is label-only) and a sidecar binding their
    SHA-256 to the source training data, replicate, audit seed and
    perturbation.  Nothing existing is replaced.
    """
    directory = Path(directory)
    stem = f"replicate_{replicate:02d}"
    sidecar_path = directory / f"{stem}.perturbation.json"
    arrays = {"labels": labels} if perturbation.label_only else {"features": features, "labels": labels}
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
        files[name] = {"file": paths[name].name, "sha256": sha256_file(paths[name])}
    sidecar = {
        "schema_version": "d4-perturbation-2",
        "perturbation": perturbation.name,
        "label_only": bool(perturbation.label_only),
        "replicate": int(replicate),
        "audit_seed": int(audit_seed),
        "source_train_digest": source_train_digest,
        "perturbed_train_digest": labelled_partition_digest(features, labels),
        "files": files,
    }
    write_json_exclusive(sidecar_path, sidecar)
    return sidecar


def load_verified_perturbation(directory: Path, *, replicate: int, perturbation: Perturbation,
                               source_train_digest: str, audit_seed: int,
                               train_features: Any, train_labels: Any) -> tuple[Any, Any, dict]:
    """Reload a persisted perturbation only if it provably belongs here.

    The sidecar must name this perturbation, replicate, audit seed and
    source training data; every file must match its digest; and the
    perturbation's own ``verify`` must accept the result.
    """
    directory = Path(directory)
    stem = f"replicate_{replicate:02d}"
    sidecar = read_json(directory / f"{stem}.perturbation.json", f"replicate {replicate} perturbation record")
    expected = {"schema_version": "d4-perturbation-2", "perturbation": perturbation.name,
                "label_only": bool(perturbation.label_only), "replicate": int(replicate),
                "audit_seed": int(audit_seed), "source_train_digest": source_train_digest}
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise IntegrityError(
                f"Persisted perturbation for replicate {replicate} has {key}={sidecar.get(key)!r}, "
                f"expected {value!r}. Refusing to reuse it across conditions, replicates or datasets."
            )
    arrays = {}
    for name in (("labels",) if perturbation.label_only else ("features", "labels")):
        entry = sidecar.get("files", {}).get(name) or {}
        path = directory / str(entry.get("file", ""))
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise IntegrityError(f"Persisted perturbation {name} for replicate {replicate} is missing "
                                 "or does not match its recorded digest.")
        arrays[name] = np.load(path, mmap_mode="r")
    features = train_features if perturbation.label_only else arrays["features"]
    labels = arrays["labels"]
    if labelled_partition_digest(features, labels) != sidecar.get("perturbed_train_digest"):
        raise IntegrityError(f"Persisted perturbation for replicate {replicate} does not reproduce "
                             "its recorded perturbed-data digest.")
    perturbation.verify(train_features, train_labels, features, labels)
    return features, labels, sidecar


def load_completed_state(state_path: Path) -> dict[str, Any] | None:
    if not Path(state_path).exists():
        return None
    return read_json(state_path, "training state")


# ============================================================
# Decision
# ============================================================

#: ``conservative_two_sided_v1`` is the rule the September 2026 run used.
#: ``directional_iut_v2`` is a proposed alternative; which applies is a
#: preregistration decision, never chosen by this module.
DECISION_RULES = ("conservative_two_sided_v1", "directional_iut_v2")


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
    """
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
            correct = binary_predictions_to_correctness(adapter.predict(test_features), test_labels)
            return correct, selection

        baseline_correct, baseline_selection = fit(train_features, train_labels, "baseline")

        perturbation_directory = checkpoint_root / "perturbations"
        sidecar_path = perturbation_directory / f"replicate_{replicate_index:02d}.perturbation.json"
        # The perturbation RNG is always drawn, so the stream does not depend
        # on whether this is a fresh run or a resume.
        fresh_features, fresh_labels = perturbation.apply(train_features, train_labels, rng=rng)
        if sidecar_path.exists():
            perturbed_features, perturbed_labels, perturbation_record = load_verified_perturbation(
                perturbation_directory, replicate=replicate_index, perturbation=perturbation,
                source_train_digest=source_digest, audit_seed=audit_seed,
                train_features=train_features, train_labels=train_labels,
            )
            if labelled_partition_digest(fresh_features, fresh_labels) != perturbation_record["perturbed_train_digest"]:
                raise IntegrityError(
                    f"Persisted perturbation for replicate {replicate_index} differs from the one "
                    "the audit seed regenerates. Refusing to resume."
                )
        else:
            perturbation.verify(train_features, train_labels, fresh_features, fresh_labels)
            perturbation_record = save_perturbation(
                perturbation_directory, replicate=replicate_index, perturbation=perturbation,
                source_train_digest=source_digest, audit_seed=audit_seed,
                features=fresh_features, labels=fresh_labels,
            )
            perturbed_features, perturbed_labels = fresh_features, fresh_labels

        perturbed_correct, perturbed_selection = fit(perturbed_features, perturbed_labels, "perturbed")

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
        ))

    rows = [{"baseline_score": r.baseline_score, "perturbed_score": r.perturbed_score,
             "mcnemar_b": r.mcnemar_b, "mcnemar_c": r.mcnemar_c} for r in replicate_results]
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
                  "source_train_digest": source_digest},
    )


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


def recompute_certificate_decision(certificate: Mapping[str, Any]) -> dict:
    """Independently re-derive the decision from a certificate's replicate rows.

    Recomputes every p-value from (b, c) and every difference from the
    accuracies, then applies the recorded rule and settings.  Raises
    IntegrityError if the certificate's decision or statistics disagree.
    """
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
