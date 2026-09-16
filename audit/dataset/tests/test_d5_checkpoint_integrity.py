"""Synthetic tests for D5 checkpoint/state persistence integrity.

These tests validate checkpoint plumbing only.
They are NOT scientific evidence.
"""

from pathlib import Path

import importlib.util

def load_d5_module():
    """Load the D5 module from the repository source."""
    import sys
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "d5_training_data_scaling.py"
    )

    module_name = "audit.dataset.d5_training_data_scaling"

    if not path.exists():
        raise FileNotFoundError(
            f"D5 source file not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load D5 module from {path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Required for Python 3.12 dataclasses with postponed annotations.
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module

def test_neither_checkpoint_nor_state_exists(tmp_path):
    """No persisted state means the condition may start fresh."""
    d5 = load_d5_module()

    paths = d5.condition_paths(
        tmp_path,
        replicate=1,
        size=100,
    )

    result = d5._validate_checkpoint_state_pair(
        paths,
        "replicate_01/n_100",
    )

    assert result is False


def test_both_checkpoint_and_state_exist(tmp_path):
    """A complete checkpoint/state pair is valid."""
    d5 = load_d5_module()

    condition_dir = (
        tmp_path
        / "runs"
        / "replicate_01"
        / "n_100"
    )
    condition_dir.mkdir(parents=True)

    (condition_dir / "latest.keras").write_bytes(
        b"synthetic checkpoint"
    )

    (condition_dir / "state.json").write_text(
        "{}",
        encoding="utf-8",
    )

    paths = d5.condition_paths(
        tmp_path,
        replicate=1,
        size=100,
    )

    result = d5._validate_checkpoint_state_pair(
        paths,
        "replicate_01/n_100",
    )

    assert result is True


def test_checkpoint_without_state_hard_fails(tmp_path):
    """An orphaned checkpoint must never be silently overwritten."""
    d5 = load_d5_module()

    condition_dir = (
        tmp_path
        / "runs"
        / "replicate_01"
        / "n_100"
    )
    condition_dir.mkdir(parents=True)

    (condition_dir / "latest.keras").write_bytes(
        b"synthetic checkpoint"
    )

    paths = d5.condition_paths(
        tmp_path,
        replicate=1,
        size=100,
    )

    try:
        d5._validate_checkpoint_state_pair(
            paths,
            "replicate_01/n_100",
        )
    except RuntimeError as exc:
        assert "state.json is missing" in str(exc)
    else:
        raise AssertionError(
            "Orphaned checkpoint did not hard-fail."
        )


def test_state_without_checkpoint_hard_fails(tmp_path):
    """An orphaned state file must never be silently accepted."""
    d5 = load_d5_module()

    condition_dir = (
        tmp_path
        / "runs"
        / "replicate_01"
        / "n_100"
    )
    condition_dir.mkdir(parents=True)

    (condition_dir / "state.json").write_text(
        "{}",
        encoding="utf-8",
    )

    paths = d5.condition_paths(
        tmp_path,
        replicate=1,
        size=100,
    )

    try:
        d5._validate_checkpoint_state_pair(
            paths,
            "replicate_01/n_100",
        )
    except RuntimeError as exc:
        assert "latest.keras is missing" in str(exc)
    else:
        raise AssertionError(
            "Orphaned state did not hard-fail."
        )