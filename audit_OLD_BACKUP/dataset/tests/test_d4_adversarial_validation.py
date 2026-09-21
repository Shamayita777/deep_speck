"""
D4 adversarial validation tests.

Purpose
-------
Deliberately construct known situations and verify that D4 responds according
to its declared methodology.

These tests do NOT run the production Gohr experiment and do NOT retrain the
production 10M/1M/1M models.

They validate:

1. clean/no-op control;
2. known predictive relationship destroyed by label shuffle;
3. preservation of the label multiset;
4. destruction of feature/label correspondence;
5. held-out test isolation;
6. paired-prediction dependence;
7. no-op perturbation;
8. practical-effect boundary;
9. statistical-only evidence;
10. practical-only evidence;
11. all-gates positive control;
12. one failed replicate;
13. deterministic adversarial fixtures;
14. wrong-target perturbation negative control.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import audit.dataset.d4_controlled_perturbation as d4

from audit.dataset.d4_controlled_perturbation import (
    Perturbation,
    ReplicateResult,
    binary_predictions_to_correctness,
    mcnemar_exact_pvalue,
    run_d4,
)

from audit.dataset.perturbations.label_shuffle import (
    LabelShufflePerturbation,
)


# ============================================================================
# Deterministic fixtures
# ============================================================================


def _signal_fixture(n=40):
    """
    Deterministic balanced task:

        X[:, 0] == Y
    """

    x = np.zeros(
        (n, 2),
        dtype=np.uint8,
    )

    x[:, 0] = np.arange(n) % 2
    x[:, 1] = (np.arange(n) // 2) % 2

    y = x[:, 0].copy()

    return x, y


def _independent_feature_label_fixture(n=40):
    """
    Deterministic fixture where X[:,0] and Y are both balanced but not
    deterministically identical.
    """

    x = np.zeros(
        (n, 2),
        dtype=np.uint8,
    )

    x[:, 0] = np.arange(n) % 2
    x[:, 1] = (np.arange(n) // 2) % 2

    y = (np.arange(n) // 4) % 2

    return x, y


# ============================================================================
# Test-only no-op perturbation
# ============================================================================


class NoOpPerturbation(Perturbation):

    def __init__(self):
        super().__init__(
            name="noop_test",
            description="Test-only identity perturbation.",
        )

    def apply(
        self,
        features,
        labels,
        *,
        rng,
    ):
        return (
            np.asarray(features),
            np.asarray(labels).copy(),
        )


# ============================================================================
# Test-only deterministic adapter
# ============================================================================


class KnownSignalAdapter:
    """
    Baseline detects X[:,0] == Y.

    After label shuffle, that exact relationship disappears and the fake
    learner predicts class 0.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.signal = False

    def train(
        self,
        train_features,
        train_labels,
        *,
        checkpoint_path,
        state_path,
        replicate,
        condition,
        run_config_hash,
    ):
        x = np.asarray(train_features)
        y = np.asarray(train_labels).reshape(-1)

        self.signal = np.array_equal(
            x[:, 0].astype(int),
            y.astype(int),
        )

        checkpoint_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            checkpoint_path
            / "checkpoint_epoch_001.keras"
        ).write_text(
            "adversarial-test checkpoint\n",
            encoding="utf-8",
        )

        state_path.write_text(
            json.dumps(
                {
                    "schema_version": "test",
                    "replicate": replicate,
                    "condition": condition,
                    "seed": self.seed,
                    "completed_epochs": 1,
                    "total_epochs": 1,
                    "status": "complete",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return self

    def predict(self, test_features):
        x = np.asarray(test_features)

        if self.signal:
            return x[:, 0].astype(float)

        return np.zeros(
            len(x),
            dtype=float,
        )


# ============================================================================
# A1 — clean control
# ============================================================================


def test_clean_control_has_no_confirmed_d4_effect(
    tmp_path,
):
    train_x, train_y = _signal_fixture()
    test_x, test_y = _signal_fixture()

    result = run_d4(
        perturbation=NoOpPerturbation(),
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        adapter_factory=KnownSignalAdapter,
        checkpoint_root=tmp_path / "clean_control",
        replicates=3,
        audit_seed=0,
        effect_threshold=0.01,
        bootstrap_replicates=1000,
        alpha=0.05,
    )

    assert result.effect_detected is False
    assert result.mean_absolute_difference == pytest.approx(
        0.0,
    )


# ============================================================================
# A2 — known predictive relationship destroyed
# ============================================================================


def test_label_shuffle_destroys_known_predictive_signal(
    tmp_path,
):
    train_x, train_y = _signal_fixture()
    test_x, test_y = _signal_fixture()

    result = run_d4(
        perturbation=LabelShufflePerturbation(),
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        adapter_factory=KnownSignalAdapter,
        checkpoint_root=tmp_path / "known_effect",
        replicates=3,
        audit_seed=0,
        effect_threshold=0.01,
        bootstrap_replicates=1000,
        alpha=0.05,
    )

    assert result.mean_baseline_score == pytest.approx(
        1.0,
    )

    assert result.mean_perturbed_score == pytest.approx(
        0.5,
    )

    assert result.mean_absolute_difference == pytest.approx(
        -0.5,
    )

    assert result.effect_detected is True


# ============================================================================
# A3 — label multiset preservation
# ============================================================================


def test_adversarial_label_shuffle_preserves_class_counts():
    x, y = _signal_fixture(100)

    _, shuffled = LabelShufflePerturbation().apply(
        x,
        y,
        rng=np.random.default_rng(1234),
    )

    original_counts = np.bincount(
        y,
        minlength=2,
    )

    shuffled_counts = np.bincount(
        shuffled,
        minlength=2,
    )

    assert original_counts.tolist() == (
        shuffled_counts.tolist()
    )


# ============================================================================
# A4 — correspondence destruction
# ============================================================================


def test_adversarial_shuffle_breaks_feature_label_correspondence():
    x, y = _signal_fixture(200)

    _, shuffled = LabelShufflePerturbation().apply(
        x,
        y,
        rng=np.random.default_rng(1234),
    )

    original_match_rate = np.mean(
        x[:, 0] == y
    )

    shuffled_match_rate = np.mean(
        x[:, 0] == shuffled
    )

    assert original_match_rate == pytest.approx(
        1.0,
    )

    # A random permutation should substantially reduce the deterministic
    # correspondence. The test deliberately uses a broad bound rather than
    # assuming an exact 0.5 outcome.
    assert shuffled_match_rate < 0.75


# ============================================================================
# A5 — held-out test isolation
# ============================================================================


def test_adversarial_perturbation_does_not_modify_heldout_targets():
    train_x, train_y = _signal_fixture(100)
    test_x, test_y = _signal_fixture(100)

    test_x_before = test_x.copy()
    test_y_before = test_y.copy()

    LabelShufflePerturbation().apply(
        train_x,
        train_y,
        rng=np.random.default_rng(77),
    )

    assert np.array_equal(
        test_x,
        test_x_before,
    )

    assert np.array_equal(
        test_y,
        test_y_before,
    )


# ============================================================================
# A6 — paired prediction-order attack
# ============================================================================
def test_adversarial_prediction_reordering_changes_mcnemar_pairing():
    """
    Adversarial test for McNemar pairing integrity.

    McNemar's test is paired: baseline_correct[i] must correspond to
    perturbed_correct[i] for the same held-out observation.

    This test deliberately reorders the perturbed correctness vector and
    verifies that the resulting discordance counts change.

    The test validates the mathematical consequence of breaking observation
    identity. It does not claim that the production D4 pipeline independently
    detects prediction reordering.
    """

    # The production helper thresholds predictions at 0.5:
    #
    #     prediction >= 0.5 -> class 1
    #     prediction <  0.5 -> class 0
    #
    # Therefore the following fixture produces:
    #
    # labels:
    #     [1, 1, 0, 0]
    #
    # baseline predictions:
    #     [0.9, 0.1, 0.9, 0.1]
    # baseline classes:
    #     [1, 0, 1, 0]
    # baseline correctness:
    #     [True, False, False, True]
    #
    # perturbed predictions:
    #     [0.9, 0.9, 0.9, 0.1]
    # perturbed classes:
    #     [1, 1, 1, 0]
    # perturbed correctness:
    #     [True, True, False, True]

    labels = np.array(
        [1, 1, 0, 0],
    )

    baseline_predictions = np.array(
        [0.9, 0.1, 0.9, 0.1],
    )

    perturbed_predictions = np.array(
        [0.9, 0.9, 0.9, 0.1],
    )

    baseline_correct = binary_predictions_to_correctness(
        baseline_predictions,
        labels,
    )

    perturbed_correct = binary_predictions_to_correctness(
        perturbed_predictions,
        labels,
    )

    # Verify that the fixture produces exactly the intended correctness
    # vectors under the actual production helper.
    assert baseline_correct.tolist() == [
        True,
        False,
        False,
        True,
    ]

    assert perturbed_correct.tolist() == [
        True,
        True,
        False,
        True,
    ]

    # Correct pairing:
    #
    # baseline    perturbed
    #    T           T
    #    F           T   -> b = 1
    #    F           F
    #    T           T
    #
    # Therefore:
    #     b1 = 1
    #     c1 = 0
    b1, c1, _ = mcnemar_exact_pvalue(
        baseline_correct,
        perturbed_correct,
    )

    assert (b1, c1) == (1, 0)

    # Deliberately break observation identity.
    #
    # Reordering:
    #
    #     original indices: [0, 1, 2, 3]
    #     reordered indices: [2, 1, 0, 3]
    #
    # perturbed_correct:
    #     [T, T, F, T]
    #
    # reordered:
    #     [F, T, T, T]
    #
    # Now the pairing becomes:
    #
    # baseline    reordered
    #    T           F   -> c = 1
    #    F           T   -> b = 1
    #    F           T   -> b = 1
    #    T           T
    #
    # Therefore:
    #     b2 = 2
    #     c2 = 1
    reordered = perturbed_correct[
        [2, 1, 0, 3]
    ]

    b2, c2, _ = mcnemar_exact_pvalue(
        baseline_correct,
        reordered,
    )

    assert (b2, c2) == (2, 1)

    # Breaking observation identity changes the McNemar contingency table.
    assert (b1, c1) != (b2, c2)
# ============================================================================
# A7 — no-op perturbation
# ============================================================================

def test_noop_perturbation_has_no_detected_effect(
    tmp_path,
):
    train_x, train_y = _signal_fixture()
    test_x, test_y = _signal_fixture()

    result = run_d4(
        perturbation=NoOpPerturbation(),
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        adapter_factory=KnownSignalAdapter,
        checkpoint_root=tmp_path / "noop",
        replicates=3,
        audit_seed=0,
        effect_threshold=0.01,
        bootstrap_replicates=1000,
        alpha=0.05,
    )

    assert result.mean_absolute_difference == pytest.approx(
        0.0,
    )

    assert result.effect_detected is False


# ============================================================================
# A8 — practical effect boundary
# ============================================================================


@pytest.mark.parametrize(
    "difference,expected",
    [
        (0.009, False),
        (0.010, True),
        (0.011, True),
    ],
)
def test_practical_effect_threshold_boundary(
    difference,
    expected,
):
    threshold = 0.01

    practical_effect = (
        abs(difference)
        >= threshold
    )

    assert practical_effect is expected


# ============================================================================
# A9 — statistical evidence without practical effect
# ============================================================================


def test_statistical_evidence_without_practical_effect_does_not_detect(
    monkeypatch,
    tmp_path,
):
    train_x, train_y = _signal_fixture()
    test_x, test_y = _signal_fixture()

    # Force very strong inference while making the practical threshold
    # intentionally impossible for the observed effect.
    monkeypatch.setattr(
        d4,
        "bootstrap_mean_ci",
        lambda *args, **kwargs: (
            -0.51,
            -0.49,
        ),
    )

    monkeypatch.setattr(
        d4,
        "mcnemar_exact_pvalue",
        lambda *args, **kwargs: (
            0,
            20,
            1e-12,
        ),
    )

    result = run_d4(
        perturbation=LabelShufflePerturbation(),
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        adapter_factory=KnownSignalAdapter,
        checkpoint_root=tmp_path / "statistical_only",
        replicates=3,
        audit_seed=0,
        effect_threshold=1.0,
        bootstrap_replicates=1000,
        alpha=0.05,
    )

    assert abs(
        result.mean_absolute_difference
    ) < result.effect_threshold

    assert result.effect_detected is False


# ============================================================================
# A10 — practical effect without inferential support
# ============================================================================


def test_practical_effect_without_inferential_support_does_not_detect(
    monkeypatch,
    tmp_path,
):
    train_x, train_y = _signal_fixture()
    test_x, test_y = _signal_fixture()

    monkeypatch.setattr(
        d4,
        "bootstrap_mean_ci",
        lambda *args, **kwargs: (
            -0.60,
            0.10,
        ),
    )

    monkeypatch.setattr(
        d4,
        "mcnemar_exact_pvalue",
        lambda *args, **kwargs: (
            0,
            20,
            0.5,
        ),
    )

    result = run_d4(
        perturbation=LabelShufflePerturbation(),
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        adapter_factory=KnownSignalAdapter,
        checkpoint_root=tmp_path / "practical_only",
        replicates=3,
        audit_seed=0,
        effect_threshold=0.01,
        bootstrap_replicates=1000,
        alpha=0.05,
    )

    assert abs(
        result.mean_absolute_difference
    ) >= result.effect_threshold

    assert result.inference_supported is False
    assert result.effect_detected is False


# ============================================================================
# A11 — all gates satisfied
# ============================================================================


def test_all_required_gates_detect_effect(
    monkeypatch,
    tmp_path,
):
    train_x, train_y = _signal_fixture()
    test_x, test_y = _signal_fixture()

    monkeypatch.setattr(
        d4,
        "bootstrap_mean_ci",
        lambda *args, **kwargs: (
            -0.51,
            -0.49,
        ),
    )

    monkeypatch.setattr(
        d4,
        "mcnemar_exact_pvalue",
        lambda *args, **kwargs: (
            0,
            20,
            1e-12,
        ),
    )

    result = run_d4(
        perturbation=LabelShufflePerturbation(),
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        adapter_factory=KnownSignalAdapter,
        checkpoint_root=tmp_path / "all_gates",
        replicates=3,
        audit_seed=0,
        effect_threshold=0.01,
        bootstrap_replicates=1000,
        alpha=0.05,
    )

    assert result.effect_detected is True
    assert result.inference_supported is True


# ============================================================================
# A12 — one failed replicate blocks confirmation
# ============================================================================


def test_one_failed_replicate_blocks_confirmed_effect(
    monkeypatch,
    tmp_path,
):
    train_x, train_y = _signal_fixture()
    test_x, test_y = _signal_fixture()

    monkeypatch.setattr(
        d4,
        "bootstrap_mean_ci",
        lambda *args, **kwargs: (
            -0.51,
            -0.49,
        ),
    )

    calls = {
        "count": 0,
    }

    def alternating_mcnemar(
        *args,
        **kwargs,
    ):
        calls["count"] += 1

        if calls["count"] == 3:
            return (
                0,
                20,
                0.5,
            )

        return (
            0,
            20,
            1e-12,
        )

    monkeypatch.setattr(
        d4,
        "mcnemar_exact_pvalue",
        alternating_mcnemar,
    )

    result = run_d4(
        perturbation=LabelShufflePerturbation(),
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        adapter_factory=KnownSignalAdapter,
        checkpoint_root=tmp_path / "failed_replicate",
        replicates=3,
        audit_seed=0,
        effect_threshold=0.01,
        bootstrap_replicates=1000,
        alpha=0.05,
    )

    assert result.mean_absolute_difference == pytest.approx(
        -0.5,
    )

    assert result.inference_supported is False
    assert result.effect_detected is False


# ============================================================================
# A13 — deterministic adversarial fixture
# ============================================================================


def test_d4_adversarial_fixture_is_deterministic():
    x1, y1 = _signal_fixture(200)
    x2, y2 = _signal_fixture(200)

    assert np.array_equal(
        x1,
        x2,
    )

    assert np.array_equal(
        y1,
        y2,
    )

    _, shuffled1 = LabelShufflePerturbation().apply(
        x1,
        y1,
        rng=np.random.default_rng(2026),
    )

    _, shuffled2 = LabelShufflePerturbation().apply(
        x2,
        y2,
        rng=np.random.default_rng(2026),
    )

    assert np.array_equal(
        shuffled1,
        shuffled2,
    )


# ============================================================================
# A14 — wrong-target perturbation negative control
# ============================================================================


def test_wrong_target_perturbation_does_not_change_training_labels():
    x, y = _signal_fixture(100)

    # Deliberately mutate an unrelated feature rather than the target.
    mutated_x = x.copy()

    mutated_x[:, 1] ^= 1

    # Feature 0 and labels remain unchanged.
    assert np.array_equal(
        mutated_x[:, 0],
        x[:, 0],
    )

    assert np.array_equal(
        y,
        x[:, 0],
    )

    assert np.array_equal(
        y,
        x[:, 0],
    )