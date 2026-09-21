"""
Gohr/Speck32/64-specific Implementation Integrity adapter.

ALL Gohr-specific knowledge for Pillar 1 lives here. The generic
framework (audit/common, audit/implementation/stages.py,
audit/implementation/equivalence.py) contains none of it.

REFERENCE PROTOCOL
------------------
The authoritative reference is the original Gohr source supplied with
this audit and stored read-only under audit/implementation/reference/:

    speck.py          sha256 59296f7b7990ec3886fc524ff49256005d360eb3748d3b1570351d669acbb8ce
    train_nets.py     sha256 b0585971da829f8cd89e2d1e3f85eed600048fcd04bc7d0d32f33ee1cd03c669
    train_5_rounds.py sha256 63b8bec9779a6b9a95cda4abdbd763b2b8aef384df45a40109c926ff8864ff24

Those files are never modified. The declared baseline below is read
FROM them, not from documentation or recollection:

    train_5_rounds.py : train_speck_distinguisher(200, num_rounds=5, depth=10)
    train_nets.py     : bs = 5000; optimizer 'adam'; loss 'mse';
                        cyclic_lr(10, 0.002, 0.0001);
                        make_checkpoint monitors 'val_loss', save_best_only=True;
                        make_train_data(10**7) train / make_train_data(10**6) eval
    speck.py          : diff default (0x0040, 0), urandom-based generation

test_key_recovery.py concerns the separate 11/12-round key-recovery
endpoint and is deliberately NOT part of this 5-round distinguisher
audit's declared baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REFERENCE_FILE_HASHES = {
    "speck.py": "59296f7b7990ec3886fc524ff49256005d360eb3748d3b1570351d669acbb8ce",
    "train_nets.py": "b0585971da829f8cd89e2d1e3f85eed600048fcd04bc7d0d32f33ee1cd03c669",
    "train_5_rounds.py": "63b8bec9779a6b9a95cda4abdbd763b2b8aef384df45a40109c926ff8864ff24",
}

# Declared reference protocol, transcribed from the reference source.
DECLARED_BASELINE: dict[str, Any] = {
    "cipher": "speck32_64",
    "rounds": 5,
    "differential": (0x0040, 0x0000),
    "depth": 10,
    "num_filters": 32,
    "kernel_size": 3,
    "word_size": 16,
    "num_blocks": 2,
    "epochs": 200,
    "batch_size": 5000,
    "optimizer": "adam",
    "loss": "mse",
    "lr_schedule": "cyclic(period=10, high=0.002, low=0.0001)",
    "checkpoint_monitor": "val_loss",
    "checkpoint_save_best_only": True,
    "train_samples": 10 ** 7,
    "validation_samples": 10 ** 6,
    "reported_best_validation_accuracy": 0.9291,
}

# Conv1D layer count implied by the reference architecture:
# one initial bit-sliced 1x1 conv, then 2 convs per residual block.
def implied_conv1d_count(depth: int) -> int:
    return 1 + 2 * depth


def depth_from_conv1d_count(n_conv1d: int) -> Optional[int]:
    """Inverse of implied_conv1d_count; None if the count is not of the expected form."""
    if n_conv1d < 1 or (n_conv1d - 1) % 2 != 0:
        return None
    return (n_conv1d - 1) // 2


@dataclass(frozen=True)
class CheckpointArchitectureProbe:
    """
    Result of reading an actual Keras .h5 checkpoint and deriving the
    architecture it REALLY contains - as opposed to what its filename
    or the surrounding documentation asserts.
    """
    path: str
    sha256: str
    n_conv1d_layers: int
    n_dense_layers: int
    n_batchnorm_layers: int
    derived_depth: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "sha256": self.sha256,
            "n_conv1d_layers": self.n_conv1d_layers, "n_dense_layers": self.n_dense_layers,
            "n_batchnorm_layers": self.n_batchnorm_layers, "derived_depth": self.derived_depth,
        }


def probe_checkpoint_architecture(path: str | Path) -> CheckpointArchitectureProbe:
    """
    Read a Keras HDF5 checkpoint and count its layer groups to derive
    the realized residual depth. Uses h5py only - does not require
    TensorFlow, and does not execute or deserialize the model.

    This is the II-3 Controlled Verification primitive that established
    the finding recorded in audit/implementation/findings.py: the
    artifact's realized architecture is read from the artifact itself,
    never inferred from its filename.
    """
    import re

    import h5py

    from audit.common.provenance import sha256_file

    path = Path(path)
    with h5py.File(path, "r") as handle:
        if "model_weights" in handle:
            keys = list(handle["model_weights"].keys())
        else:
            keys = list(handle.keys())

    conv = [k for k in keys if re.match(r"^conv1d(_\d+)?$", k)]
    dense = [k for k in keys if re.match(r"^dense(_\d+)?$", k)]
    bn = [k for k in keys if re.match(r"^batch_normalization(_\d+)?$", k)]

    return CheckpointArchitectureProbe(
        path=str(path), sha256=sha256_file(path),
        n_conv1d_layers=len(conv), n_dense_layers=len(dense), n_batchnorm_layers=len(bn),
        derived_depth=depth_from_conv1d_count(len(conv)),
    )
