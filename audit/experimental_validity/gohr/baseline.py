"""
Frozen Gohr/Speck32/64 5-round baseline configuration.

Every value here is a repository-derived fact, verified directly
against archive/train_5_rounds.py, archive/train_nets.py, and
archive/notebooks/gohr_reproduction_tf2.ipynb (see the project's Round-4
design document for the exact citations). This module is the single
source of truth for the baseline; no other module may hard-code a
competing value.

Do not silently change any of these values. If a repository fact
conflicts with a value here, that is a scientific discrepancy to
surface (see docs/discrepancies.md), not something to resolve by
quietly editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrozenBaseline:
    cipher: str = "speck3264"
    differential: tuple[int, int] = (0x0040, 0x0000)
    rounds: int = 5
    depth: int = 10
    # Reference: train_nets.py::train_speck_distinguisher builds
    # make_resnet(depth=depth, reg_param=10**-5). Previously undeclared here,
    # so the EV adapter silently used make_resnet's 1e-4 default - ten times
    # the reference L2 weight. Declared explicitly (Implementation Integrity
    # finding: undeclared parameter taking a function default).
    reg_param: float = 1e-5
    architecture: str = "gohr_resnet"
    optimizer: str = "adam"
    lr_schedule_high: float = 0.002
    lr_schedule_low: float = 0.0001
    lr_schedule_period: int = 10
    batch_size: int = 5000
    epochs: int = 200
    shuffle: bool = True
    checkpoint_monitor: str = "val_loss"
    checkpoint_save_best_only: bool = True
    accuracy_reporting_convention: str = "max_val_acc_over_history"
    historical_reference_seed: int = 0
    train_size: int = 10_000_000
    val_size: int = 1_000_000
    test_size: int = 1_000_000
    dataset_generation_method: str = "os.urandom"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cipher": self.cipher,
            "differential": list(self.differential),
            "rounds": self.rounds,
            "depth": self.depth,
            "reg_param": self.reg_param,
            "architecture": self.architecture,
            "optimizer": self.optimizer,
            "lr_schedule": {
                "high": self.lr_schedule_high,
                "low": self.lr_schedule_low,
                "period": self.lr_schedule_period,
            },
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "shuffle": self.shuffle,
            "checkpoint_monitor": self.checkpoint_monitor,
            "checkpoint_save_best_only": self.checkpoint_save_best_only,
            "accuracy_reporting_convention": self.accuracy_reporting_convention,
            "historical_reference_seed": self.historical_reference_seed,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "dataset_generation_method": self.dataset_generation_method,
        }


BASELINE = FrozenBaseline()

# EXPLICIT PROTOCOL DECISION (Issue 5, Round 6): confirmatory evaluation
# uses the FINAL-EPOCH in-memory model, not a reload of the best-val-loss
# checkpoint. This was already the actual behavior (gohr/adapter.py never
# reloads from checkpoint_path before evaluating) but was not previously
# recorded as an explicit, named protocol choice anywhere. Per this
# decision: ModelCheckpoint's saved artifact serves ONLY as provenance
# and failure/corruption protection (its hash is recorded; it is not
# read back for the confirmatory metric). This is "Option B" from the
# Round-6 audit: the smallest-change, already-implemented behavior,
# now made explicit rather than ambiguous.
CONFIRMATORY_EVALUATION_PROTOCOL = "final_epoch_weights"

# Software versions as literally observed in
# archive/notebooks/gohr_reproduction_tf2.ipynb (cells 8-10), which
# produced the historical 0.9291 reference figure.
DOCUMENTED_SOFTWARE = {
    "tensorflow_version": "2.20.0",
    "keras_version": "3.13.2",
    "python_version": "3.12.13",
}


def compare_software_environment(actual: dict[str, Any]) -> dict[str, Any]:
    """
    Compare the actual, installed software versions (from
    framework.provenance.software_provenance) against the documented
    reference environment. Never silently assumes a match - every
    comparison is recorded explicitly, including mismatches.
    """
    comparison = {}
    for key, documented_value in DOCUMENTED_SOFTWARE.items():
        actual_value = actual.get(key)
        comparison[key] = {
            "documented": documented_value,
            "actual": actual_value,
            "match": actual_value == documented_value,
        }
    return comparison
