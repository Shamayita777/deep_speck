"""
CE1: distinguishing-signal destruction, paired-block design.

SCIENTIFIC FIX over historical CE1
----------------------------------
Historical CE1 (a) generated DIFFERENT datasets for the baseline and
signal-destroyed arms, (b) shuffled the *validation* labels too, so the
destroyed model was scored against a different prediction target, and
(c) ran once per arm, giving no run-to-run uncertainty.

CE1 fixes all three:
  * one dataset per block, shared by both arms;
  * ONLY the TRAINING labels are permuted;
  * the evaluation set and its labels are byte-identical for both arms;
  * multiple independent blocks; the paired per-block difference is the
    statistical unit.

Interpretation is deliberately narrow: evidence that destroying the
differential training signal causes loss of distinguishing performance.
It does NOT establish that no non-cryptographic shortcut exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import numpy as np


class BlockConstructionError(RuntimeError):
    pass


@dataclass
class CE1Block:
    """One independent block: a shared dataset plus its two training arms."""
    block_id: str
    X_train: np.ndarray
    Y_train_baseline: np.ndarray
    Y_train_destroyed: np.ndarray
    X_eval: np.ndarray
    Y_eval: np.ndarray
    permutation_seed: int
    metadata: dict = field(default_factory=dict)

    def verify_invariants(self) -> dict:
        """
        Assert the properties the design depends on. Called before any
        training so a malformed block cannot silently produce evidence.
        """
        problems = []
        if self.Y_train_baseline.shape != self.Y_train_destroyed.shape:
            problems.append("training label arrays differ in shape")
        # the destroyed arm must be a PERMUTATION: same multiset of labels
        if not np.array_equal(np.sort(self.Y_train_baseline),
                              np.sort(self.Y_train_destroyed)):
            problems.append("destroyed training labels are not a permutation of the baseline")
        if np.array_equal(self.Y_train_baseline, self.Y_train_destroyed) \
                and self.Y_train_baseline.size > 8:
            problems.append("permutation left the training labels unchanged")
        if self.X_train is None or self.X_eval is None:
            problems.append("missing inputs")
        # evaluation data must be untouched and shared
        if self.Y_eval.size == 0:
            problems.append("empty evaluation set")
        if problems:
            raise BlockConstructionError(f"{self.block_id}: " + "; ".join(problems))
        return {
            "block_id": self.block_id,
            "n_train": int(self.X_train.shape[0]),
            "n_eval": int(self.X_eval.shape[0]),
            "training_labels_permuted": True,
            "evaluation_labels_intact": True,
            "evaluation_shared_between_arms": True,
            "permutation_seed": self.permutation_seed,
        }


def build_ce1_block(X_train, Y_train, X_eval, Y_eval, *, block_id: str,
                    permutation_seed: int) -> CE1Block:
    """
    Construct a paired block from ONE generated dataset.

    The destroyed arm is produced by permuting the training labels only.
    Evaluation arrays are passed through unchanged and shared by both
    arms, so the two models are scored on an identical prediction target.
    """
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_eval = np.asarray(X_eval)
    Y_eval = np.asarray(Y_eval)
    if X_train.shape[0] != Y_train.shape[0]:
        raise BlockConstructionError("training inputs and labels differ in length")
    if X_eval.shape[0] != Y_eval.shape[0]:
        raise BlockConstructionError("evaluation inputs and labels differ in length")
    rng = np.random.default_rng(permutation_seed)
    destroyed = Y_train[rng.permutation(Y_train.shape[0])].copy()
    block = CE1Block(
        block_id=block_id, X_train=X_train, Y_train_baseline=Y_train.copy(),
        Y_train_destroyed=destroyed, X_eval=X_eval, Y_eval=Y_eval,
        permutation_seed=permutation_seed,
    )
    block.verify_invariants()
    return block


def exact_sign_flip_test(diffs) -> dict:
    """
    Exact paired sign-flip (permutation) test on the block differences.

    PRIMARY CE1 TEST. Chosen over the paired t because at K<=12 normality
    is assumed but untestable, and over Wilcoxon because sign-flip
    preserves the magnitudes that Wilcoxon discards for ranks. Under the
    sharp null the sign of each Delta_k is exchangeable, so enumerating
    all 2^K sign assignments gives an EXACT level.

    Resolution limit: the smallest attainable two-sided p is 2/2^K
    (0.0625 at K=5, 0.03125 at K=6). Below 6 blocks the test cannot reach
    alpha=0.05 at any effect size, which is why min_valid_blocks >= 6.
    """
    from itertools import product

    d = np.asarray(list(diffs), dtype=float)
    k = d.size
    if k < 2:
        raise ValueError("the sign-flip test needs at least 2 blocks.")
    observed = abs(float(d.mean()))
    count = sum(1 for signs in product((1.0, -1.0), repeat=k)
                if abs(float((d * np.array(signs)).mean())) >= observed - 1e-15)
    min_p = 2.0 / (2 ** k)
    return {
        "test": "exact_paired_sign_flip",
        "n_blocks": int(k),
        "p_value": count / (2 ** k),
        "min_attainable_two_sided_p": min_p,
        "can_reach_alpha_005": bool(min_p < 0.05),
        "enumerated_assignments": 2 ** k,
    }


def paired_block_analysis(baseline_acc, destroyed_acc, *, chance: float = 0.5,
                          alpha: float = 0.05) -> dict:
    """
    Primary CE1 analysis over INDEPENDENT BLOCKS (frozen design).

    Primary test   exact paired sign-flip on Delta_k
    Interval       t-based 95% CI (normality stated, not assumed silently)
    Effect size    mean Delta (raw accuracy) + Cohen's d_z
    Reported       every individual Delta_k

    NO bootstrap CI: with 2^8 distinct resamples at this K it would imply
    precision the data cannot support.

    EQUIVALENCE IS DISABLED by the frozen design. The destroyed arm is
    reported as a mean with a 95% CI plus the distinguishing advantage
    that interval EXCLUDES. A non-significant difference is never reported
    as equivalence to chance.
    """
    from scipy import stats

    from audit.cryptography.frozen_design import CE1 as CE1_SPEC
    from audit.cryptography.statistics import p_value_report

    b = np.asarray(list(baseline_acc), dtype=float)
    d = np.asarray(list(destroyed_acc), dtype=float)
    if b.shape != d.shape:
        raise ValueError("baseline and destroyed arms must have one value per block")
    diff = b - d
    n = diff.size
    if n < 2:
        raise ValueError("CE1 requires at least 2 valid blocks")

    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    sign_flip = exact_sign_flip_test(diff)
    sign_flip["null_hypothesis"] = CE1_SPEC.primary_hypothesis
    sign_flip["exactness_assumption"] = CE1_SPEC.exactness_assumption
    sign_flip["rejection_interpretation"] = CE1_SPEC.rejection_interpretation
    sign_flip["randomization_basis"] = (
        "randomized arm assignment: per block a prospectively recorded fair coin flip "
        "assigns which of the block's two model seeds goes to the intact arm. The 2^K "
        "enumerated sign patterns ARE that randomization distribution.")
    sign_flip["arm_assignment_randomized"] = CE1_SPEC.arm_assignment_randomized

    # destroyed arm vs chance: interval only, no verdict
    dev = d - chance
    d_sd = float(d.std(ddof=1))
    d_se = d_sd / math.sqrt(n)
    d_tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    d_lo, d_hi = float(d.mean() - d_tcrit * d_se), float(d.mean() + d_tcrit * d_se)
    # |2p-1| is the distinguishing advantage; the CI excludes anything beyond
    # the furthest bound from chance.
    excluded_adv = 2 * max(abs(d_lo - chance), abs(d_hi - chance))

    return {
        "design_specification": CE1_SPEC.to_dict(),
        "statistical_unit": "independent training block (paired difference)",
        "n_blocks": int(n),
        "baseline_mean": float(b.mean()),
        "destroyed_mean": float(d.mean()),
        "paired_difference": {
            "mean": mean, "sd": sd, "se": se,
            "ci95_t": [mean - tcrit * se, mean + tcrit * se],
            "effect_size_cohens_dz": (mean / sd) if sd > 0 else None,
            "raw_block_differences": [float(x) for x in diff],
            "primary_test": {**sign_flip, **p_value_report(sign_flip["p_value"])},
        },
        "destroyed_vs_chance": {
            "chance": chance,
            "mean_accuracy": float(d.mean()),
            "ci95_t": [d_lo, d_hi],
            "mean_deviation": float(dev.mean()),
            "advantage_excluded_by_ci": excluded_adv,
            "equivalence": {
                "enabled": False,
                "verdict": "NOT_ASSESSED",
                "reason": (
                    "Equivalence-to-chance is DISABLED by the frozen design. No "
                    "defensible margin exists: a cryptanalytically meaningful bound "
                    "(advantage <= 2^-10, eps ~ 4.9e-4) lies at or below the 95% CI "
                    "half-width of the sealed 10^6 test set (9.8e-4). A non-significant "
                    "difference is NOT evidence of equivalence."),
            },
        },
        "interpretation_scope": (
            "Evidence that destroying the differential training signal causes loss of "
            "distinguishing performance, for this architecture and training protocol, "
            "conditional on the fixed sealed test set. It does NOT establish that no "
            "non-cryptographic shortcut exists, and it makes no equivalence-to-chance "
            "claim."),
    }
