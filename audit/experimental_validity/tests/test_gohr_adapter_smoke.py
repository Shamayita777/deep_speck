"""
Integration test for the GohrAdapter end-to-end pipeline, using the
drastically reduced smoke-scale configuration. This exercises real
TensorFlow/Keras training and evaluation - it is a software-correctness
test, and its numeric results are NON_EVIDENTIARY (see
gohr.experiments._baseline_overrides_for_run_mode's smoke branch).
"""

import uuid
from pathlib import Path

from gohr.adapter import GohrAdapter, ReplicateConfig, generate_matched_datasets
from gohr.baseline import BASELINE
from gohr.representation import identity_permutation


def test_single_replicate_runs_end_to_end(tmp_path):
    adapter = GohrAdapter()
    cfg = ReplicateConfig(
        experiment_id="TEST", hypothesis_id=None, condition_id="smoke",
        replicate_id="r0", run_id=str(uuid.uuid4()), run_mode="smoke",
        rounds=3, differential=BASELINE.differential, depth=1,
        epochs=1, batch_size=64, shuffle=True, optimizer="adam",
        lr_high=0.002, lr_low=0.0001, lr_period=10, model_seed=42,
        representation=identity_permutation(), output_dir=Path(tmp_path),
        matched_datasets=None,
        independent_train_size=256, independent_val_size=64, independent_confirmatory_test_size=64,
        same_dataset=False, same_model_initialization=False,
        same_training_shuffle_stream=False, same_evaluation_data=False,
    )
    result = adapter.run_replicate(cfg)
    assert result.outcome.status.value in ("valid", "failed")
    if result.outcome.status.value == "valid":
        assert 0.0 <= result.outcome.metric_value <= 1.0
        assert result.manifest_fields["checkpoint_hash"] is not None
        assert result.manifest_fields["dataset_hash"]["train"] is not None
        assert result.manifest_fields["exact_replay_available"] is False


def test_matched_pair_shares_dataset_and_seed(tmp_path):
    """
    Confirms the matched-pair mechanism: two replicates given the same
    matched_datasets object and the same model_seed must be trained on
    byte-identical inputs (verified via dataset hash equality in the
    resulting manifests).
    """
    adapter = GohrAdapter()
    datasets = generate_matched_datasets(
        rounds=3, differential=BASELINE.differential,
        train_size=256, val_size=64, confirmatory_test_size=64,
    )

    def make_cfg(condition_id, shuffle):
        return ReplicateConfig(
            experiment_id="TEST", hypothesis_id=None, condition_id=condition_id,
            replicate_id="pair0", run_id=str(uuid.uuid4()), run_mode="smoke",
            rounds=3, differential=BASELINE.differential, depth=1,
            epochs=1, batch_size=64, shuffle=shuffle, optimizer="adam",
            lr_high=0.002, lr_low=0.0001, lr_period=10, model_seed=7,
            representation=identity_permutation(), output_dir=Path(tmp_path),
            matched_datasets=datasets,
            same_dataset=True, same_model_initialization=True,
            same_training_shuffle_stream=(shuffle == shuffle), same_evaluation_data=True,
        )

    result_a = adapter.run_replicate(make_cfg("a", True))
    result_b = adapter.run_replicate(make_cfg("b", False))

    assert (
        result_a.manifest_fields["dataset_hash"]["train"]
        == result_b.manifest_fields["dataset_hash"]["train"]
    )
    assert (
        result_a.manifest_fields["dataset_hash"]["confirmatory_test"]
        == result_b.manifest_fields["dataset_hash"]["confirmatory_test"]
    )


def test_make_resnet_requires_explicit_depth():
    """Root-cause guard for II-FINDING-DEPTH-V1: no silent depth default."""
    import pytest
    from gohr.model import make_resnet
    with pytest.raises(TypeError):
        make_resnet()                       # depth omitted -> must fail loudly
    with pytest.raises(ValueError):
        make_resnet(depth=0)


def test_make_resnet_explicit_depths_realize_declared_architecture():
    """depth=10 -> 10 Add merges; depth=5 -> 5. Verifies the arms are what they claim."""
    from gohr.model import make_resnet
    for depth in (5, 10):
        m = make_resnet(depth=depth)
        adds = sum(1 for l in m.layers if l.__class__.__name__ == "Add")
        convs = sum(1 for l in m.layers if l.__class__.__name__ == "Conv1D")
        assert adds == depth
        assert convs == 1 + 2 * depth


def test_ev_baseline_declares_reference_reg_param():
    """EV previously never declared reg_param and silently used the 1e-4 default."""
    from gohr.baseline import BASELINE
    assert BASELINE.reg_param == 1e-5
    assert BASELINE.to_dict()["reg_param"] == 1e-5


def test_ev_adapter_builds_models_with_reference_l2():
    """Read the realized regularizer off the built layers, not the config value."""
    import inspect
    from gohr import adapter
    assert "reg_param=BASELINE.reg_param" in inspect.getsource(adapter)
    from gohr.model import make_resnet
    from gohr.baseline import BASELINE
    m = make_resnet(depth=BASELINE.depth, reg_param=BASELINE.reg_param)
    l2 = {round(float(l.kernel_regularizer.l2), 12) for l in m.layers
          if getattr(l, "kernel_regularizer", None) is not None}
    assert l2 == {1e-5}
