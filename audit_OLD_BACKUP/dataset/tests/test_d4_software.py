"""
D4 software/regression tests.

These tests validate the generic D4 machinery without running the production
10M/1M/1M Gohr experiment.

Scientific status
-----------------
These are implementation/regression tests. Passing them does NOT establish
that the production Gohr D4 experiment is scientifically valid, nor does it
establish the production result.

Coverage
--------
1. Label-shuffle perturbation contract
2. RNG reproducibility
3. Input immutability
4. Binary prediction conversion
5. Exact McNemar inference
6. Replicate-level bootstrap
7. Checkpoint/state helpers
8. Certificate structure
9. Tiny end-to-end run_d4 integration
10. Basic argument validation
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from audit.dataset.d4_controlled_perturbation import (
    D4Result,
    Perturbation,
    ReplicateResult,
    binary_predictions_to_correctness,
    bootstrap_mean_ci,
    checkpoint_paths,
    generate_certificate,
    load_completed_state,
    mcnemar_exact_pvalue,
    run_d4,
)
from audit.dataset.perturbations.label_shuffle import (
    LabelShufflePerturbation,
)


# ============================================================================
# Test-only perturbation
# ============================================================================


class IdentityPerturbation(Perturbation):
    """
    Test-only no-op perturbation.

    It is intentionally not a production perturbation. It exists to test
    that the generic framework does not manufacture an effect when the
    perturbation leaves the data unchanged.
    """

    def __init__(self) -> None:
        super().__init__(
            name="identity_test",
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
# Deterministic fake adapter
# ============================================================================


class DeterministicFakeAdapter:
    """
    Tiny adapter for generic run_d4 integration testing.

    The synthetic training data encode:

        X[:, 0] == Y

    The baseline condition therefore learns a deterministic feature rule.

    A label-shuffled training set no longer satisfies that relationship, so
    the fake adapter falls back to predicting class 0.

    This is NOT a machine-learning experiment. It is a deterministic test
    double for the D4 framework.
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.use_feature_rule = False

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

        self.use_feature_rule = bool(
            np.array_equal(
                x[:, 0].astype(int),
                y.astype(int),
            )
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
            "test-only fake checkpoint\n",
            encoding="utf-8",
        )

        state_path.write_text(
            json.dumps(
                {
                    "schema_version": "test",
                    "replicate": replicate,
                    "condition": condition,
                    "seed": self.seed,
                    "total_epochs": 1,
                    "completed_epochs": 1,
                    "status": "complete",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return self

    def predict(self, test_features):
        x = np.asarray(test_features)

        if self.use_feature_rule:
            return x[:, 0].astype(float)

        return np.zeros(
            len(x),
            dtype=float,
        )


# ============================================================================
# Synthetic fixture
# ============================================================================


def make_balanced_signal_fixture(
    n: int = 40,
):
    """
    Deterministic balanced binary task.

    X[:, 0] is exactly equal to Y.
    """

    x = np.zeros(
        (n, 2),
        dtype=np.uint8,
    )

    x[:, 0] = np.arange(n) % 2
    x[:, 1] = (np.arange(n) // 2) % 2

    y = x[:, 0].copy()

    return x, y


# ============================================================================
# Perturbation contract
# ============================================================================


def test_label_shuffle_preserves_length_and_label_multiset():
    x, y = make_balanced_signal_fixture(100)

    out_x, out_y = LabelShufflePerturbation().apply(
        x,
        y,
        rng=np.random.default_rng(123),
    )

    assert out_x.shape == x.shape
    assert out_y.shape == y.shape

    assert np.array_equal(
        out_x,
        x,
    )

    assert np.array_equal(
        np.sort(out_y),
        np.sort(y),
    )


def test_label_shuffle_is_reproducible_for_fixed_seed():
    x, y = make_balanced_signal_fixture(100)

    perturbation = LabelShufflePerturbation()

    _, y1 = perturbation.apply(
        x,
        y,
        rng=np.random.default_rng(12345),
    )

    _, y2 = perturbation.apply(
        x,
        y,
        rng=np.random.default_rng(12345),
    )

    assert np.array_equal(
        y1,
        y2,
    )


def test_label_shuffle_does_not_mutate_input():
    x, y = make_balanced_signal_fixture(100)

    x_before = x.copy()
    y_before = y.copy()

    LabelShufflePerturbation().apply(
        x,
        y,
        rng=np.random.default_rng(9),
    )

    assert np.array_equal(
        x,
        x_before,
    )

    assert np.array_equal(
        y,
        y_before,
    )


def test_label_shuffle_requires_numpy_generator():
    x, y = make_balanced_signal_fixture()

    with pytest.raises(TypeError):
        LabelShufflePerturbation().apply(
            x,
            y,
            rng=np.random.RandomState(1),
        )


def test_identity_perturbation_preserves_inputs():
    x, y = make_balanced_signal_fixture()

    x_before = x.copy()
    y_before = y.copy()

    out_x, out_y = IdentityPerturbation().apply(
        x,
        y,
        rng=np.random.default_rng(1),
    )

    assert np.array_equal(
        out_x,
        x_before,
    )

    assert np.array_equal(
        out_y,
        y_before,
    )

    assert np.array_equal(
        x,
        x_before,
    )

    assert np.array_equal(
        y,
        y_before,
    )


# ============================================================================
# Binary prediction handling
# ============================================================================


def test_binary_predictions_to_correctness_probability_vector():
    labels = np.array(
        [0, 1, 1, 0],
    )

    predictions = np.array(
        [0.1, 0.8, 0.7, 0.2],
    )

    correct = binary_predictions_to_correctness(
        predictions,
        labels,
    )

    assert np.array_equal(
        correct,
        np.array(
            [True, True, True, True],
        ),
    )


def test_binary_predictions_to_correctness_accepts_n_by_one():
    labels = np.array(
        [0, 1, 0],
    )

    predictions = np.array(
        [
            [0.1],
            [0.9],
            [0.4],
        ],
    )

    correct = binary_predictions_to_correctness(
        predictions,
        labels,
    )

    assert np.array_equal(
        correct,
        np.array(
            [True, True, True],
        ),
    )


def test_binary_predictions_to_correctness_rejects_wrong_shape():
    with pytest.raises(ValueError):
        binary_predictions_to_correctness(
            np.zeros(
                (3, 2),
            ),
            np.array(
                [0, 1, 0],
            ),
        )


def test_binary_predictions_to_correctness_rejects_length_mismatch():
    with pytest.raises(ValueError):
        binary_predictions_to_correctness(
            np.zeros(2),
            np.array(
                [0, 1, 0],
            ),
        )


# ============================================================================
# Exact McNemar inference
# ============================================================================


def test_mcnemar_zero_discordance_returns_one():
    baseline = np.array(
        [True, False, True, False],
    )

    perturbed = baseline.copy()

    b, c, p_value = mcnemar_exact_pvalue(
        baseline,
        perturbed,
    )

    assert b == 0
    assert c == 0
    assert p_value == pytest.approx(1.0)


def test_mcnemar_symmetric_discordance_has_no_directional_signal():
    baseline = np.array(
        [True, True, False, False],
    )

    perturbed = np.array(
        [True, False, True, False],
    )

    b, c, p_value = mcnemar_exact_pvalue(
        baseline,
        perturbed,
    )

    assert b == 1
    assert c == 1
    assert p_value == pytest.approx(1.0)


def test_mcnemar_known_asymmetric_case():
    baseline = np.ones(
        10,
        dtype=bool,
    )

    perturbed = np.zeros(
        10,
        dtype=bool,
    )

    b, c, p_value = mcnemar_exact_pvalue(
        baseline,
        perturbed,
    )

    assert b == 0
    assert c == 10

    expected = 2.0 / (2.0 ** 10)

    assert p_value == pytest.approx(
        expected,
    )


def test_mcnemar_rejects_unpaired_shapes():
    with pytest.raises(ValueError):
        mcnemar_exact_pvalue(
            np.array(
                [True, False],
            ),
            np.array(
                [True],
            ),
        )


# ============================================================================
# Bootstrap
# ============================================================================


def test_bootstrap_is_reproducible():
    values = np.array(
        [
            -0.40,
            -0.42,
            -0.43,
            -0.41,
            -0.44,
        ],
    )

    result_a = bootstrap_mean_ci(
        values,
        rng=np.random.default_rng(123),
        bootstrap_replicates=1000,
        confidence_level=0.95,
    )

    result_b = bootstrap_mean_ci(
        values,
        rng=np.random.default_rng(123),
        bootstrap_replicates=1000,
        confidence_level=0.95,
    )

    assert result_a == pytest.approx(
        result_b,
    )

    assert result_a[0] <= result_a[1]


def test_bootstrap_rejects_too_few_replicate_observations():
    with pytest.raises(ValueError):
        bootstrap_mean_ci(
            [0.1],
            rng=np.random.default_rng(1),
            bootstrap_replicates=1000,
        )


def test_bootstrap_rejects_fewer_than_1000_resamples():
    with pytest.raises(ValueError):
        bootstrap_mean_ci(
            [0.1, 0.2],
            rng=np.random.default_rng(1),
            bootstrap_replicates=999,
        )


# ============================================================================
# Checkpoint/state helpers
# ============================================================================


def test_checkpoint_paths_are_partitioned_by_condition(
    tmp_path,
):
    root = tmp_path / "checkpoints"

    baseline_dir, baseline_state = checkpoint_paths(
        root,
        1,
        "baseline",
    )

    perturbed_dir, perturbed_state = checkpoint_paths(
        root,
        1,
        "perturbed",
    )

    assert baseline_dir == (
        root / "replicate_01" / "baseline"
    )

    assert perturbed_dir == (
        root / "replicate_01" / "perturbed"
    )

    assert baseline_dir != perturbed_dir
    assert baseline_state != perturbed_state


def test_load_completed_state_returns_none_when_missing(
    tmp_path,
):
    assert (
        load_completed_state(
            tmp_path / "missing.json"
        )
        is None
    )


def test_load_completed_state_reads_valid_state(
    tmp_path,
):
    path = tmp_path / "state.json"

    expected = {
        "status": "complete",
        "completed_epochs": 3,
    }

    path.write_text(
        json.dumps(expected),
        encoding="utf-8",
    )

    assert load_completed_state(path) == expected


# ============================================================================
# Certificate
# ============================================================================


def make_synthetic_d4_result():
    replicates = [
        ReplicateResult(
            replicate=i,
            seed=100 + i,
            baseline_score=0.90,
            perturbed_score=0.50,
            absolute_difference=-0.40,
            relative_difference_percent=-44.4444,
            mcnemar_b=5,
            mcnemar_c=40,
            mcnemar_pvalue=0.001,
        )
        for i in range(1, 3)
    ]

    return D4Result(
        perturbation="label_shuffle",
        replicates=replicates,
        mean_baseline_score=0.90,
        mean_perturbed_score=0.50,
        mean_absolute_difference=-0.40,
        sd_absolute_difference=0.0,
        mean_relative_difference_percent=-44.4444,
        bootstrap_ci_low=-0.41,
        bootstrap_ci_high=-0.39,
        min_mcnemar_pvalue=0.001,
        max_mcnemar_pvalue=0.001,
        adjusted_alpha=0.025,
        effect_threshold=0.01,
        effect_detected=True,
        inference_supported=True,
        interpretation="synthetic test result",
    )


def test_certificate_contains_required_sections():
    certificate = generate_certificate(
        make_synthetic_d4_result(),
        dataset_id="test-dataset",
        dataset_version="test-v1",
        generation_procedure="synthetic",
        generation_parameters={
            "samples": 40,
        },
        audit_seed=0,
        confidence_level=0.95,
        alpha=0.05,
    )

    required = {
        "audit",
        "decision",
        "findings",
        "methodology",
        "provenance",
        "limitations",
        "interpretation",
    }

    assert required.issubset(
        certificate.keys(),
    )

    assert certificate["audit"]["id"] == "D4"

    assert (
        certificate["findings"]["perturbation"]
        == "label_shuffle"
    )


# ============================================================================
# Tiny end-to-end generic D4 integration
# ============================================================================


def test_run_d4_detects_known_synthetic_label_shuffle_effect(
    tmp_path,
):
    train_x, train_y = (
        make_balanced_signal_fixture(40)
    )

    test_x, test_y = (
        make_balanced_signal_fixture(40)
    )

    result = run_d4(
        perturbation=LabelShufflePerturbation(),
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        adapter_factory=DeterministicFakeAdapter,
        checkpoint_root=tmp_path / "checkpoints",
        replicates=3,
        audit_seed=0,
        effect_threshold=0.01,
        bootstrap_replicates=1000,
        confidence_level=0.95,
        alpha=0.05,
        run_config_hash="test-config-hash",
    )

    assert len(result.replicates) == 3

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
    assert result.inference_supported is True

    for replicate in result.replicates:
        assert replicate.mcnemar_b == 0
        assert replicate.mcnemar_c == 20
        assert (
            replicate.mcnemar_pvalue
            < result.adjusted_alpha
        )


# ============================================================================
# Argument validation
# ============================================================================


def test_run_d4_requires_at_least_two_replicates(
    tmp_path,
):
    train_x, train_y = (
        make_balanced_signal_fixture(20)
    )

    with pytest.raises(ValueError):
        run_d4(
            perturbation=IdentityPerturbation(),
            train_features=train_x,
            train_labels=train_y,
            test_features=train_x,
            test_labels=train_y,
            adapter_factory=DeterministicFakeAdapter,
            checkpoint_root=tmp_path,
            replicates=1,
            audit_seed=0,
            effect_threshold=0.01,
        )


def test_run_d4_rejects_negative_effect_threshold(
    tmp_path,
):
    train_x, train_y = (
        make_balanced_signal_fixture(20)
    )

    with pytest.raises(ValueError):
        run_d4(
            perturbation=IdentityPerturbation(),
            train_features=train_x,
            train_labels=train_y,
            test_features=train_x,
            test_labels=train_y,
            adapter_factory=DeterministicFakeAdapter,
            checkpoint_root=tmp_path,
            replicates=2,
            audit_seed=0,
            effect_threshold=-0.01,
        )