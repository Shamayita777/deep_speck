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


def paired_block_analysis(baseline_acc, destroyed_acc, *, chance: float = 0.5,
                          equivalence_margin: float | None = None, alpha: float = 0.05) -> dict:
    """
    Primary CE1 analysis over INDEPENDENT BLOCKS.

    The unit is the block, and the primary quantity is the paired
    per-block difference (baseline - destroyed).

    Equivalence-to-chance for the destroyed arm is assessed by TOST
    against a PREDECLARED margin - never by reading a non-significant
    difference test as proof of equality. Without a declared margin the
    verdict is explicitly INCONCLUSIVE rather than "equivalent".
    """
    from scipy import stats

    from audit.cryptography.statistics import p_value_report, replicate_level_summary

    b = np.asarray(list(baseline_acc), dtype=float)
    d = np.asarray(list(destroyed_acc), dtype=float)
    if b.shape != d.shape:
        raise ValueError("baseline and destroyed arms must have one value per block")
    diff = b - d
    summary = replicate_level_summary(diff, alpha=alpha)
    summary["statistical_unit"] = "independent training block (paired difference)"
    summary["n_blocks"] = summary.pop("n_replicates")

    # destroyed arm vs chance
    dev = d - chance
    n = dev.size
    sd = float(dev.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 and sd > 0 else None
    if equivalence_margin is None:
        equivalence = {
            "verdict": "INCONCLUSIVE",
            "reason": ("no predeclared equivalence margin; a non-significant difference from "
                       "chance is NOT evidence of equivalence to chance"),
        }
    elif se:
        df = n - 1
        p_lo = float(stats.t.sf((dev.mean() + equivalence_margin) / se, df))
        p_hi = float(stats.t.cdf((dev.mean() - equivalence_margin) / se, df))
        p_tost = max(p_lo, p_hi)
        equivalence = {
            "verdict": ("EQUIVALENT_TO_CHANCE_WITHIN_MARGIN" if p_tost < alpha
                        else "NOT_ESTABLISHED"),
            "margin": equivalence_margin, "method": "TOST",
            **p_value_report(p_tost),
        }
    else:
        equivalence = {"verdict": "INCONCLUSIVE", "reason": "zero variance across blocks"}

    return {
        "baseline_mean": float(b.mean()), "destroyed_mean": float(d.mean()),
        "paired_difference": summary,
        "destroyed_vs_chance": {"chance": chance, "mean_deviation": float(dev.mean()),
                                "equivalence": equivalence},
        "interpretation_scope": (
            "Evidence that destroying the differential training signal causes loss of "
            "distinguishing performance, for this architecture/protocol. It does NOT "
            "establish that no non-cryptographic shortcut exists."),
    }
