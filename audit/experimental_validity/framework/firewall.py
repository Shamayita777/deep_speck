"""
Test-set firewall for Experimental Validity.

Enforces the four-tier data lifecycle:

    development -> calibration -> training/validation -> freeze ->
    seal confirmatory test(s) -> single-use confirmatory evaluation

REVISION NOTE (structural redesign): the previous version sealed a
SINGLE global confirmatory-dataset hash per experiment and rejected any
reseal with a different hash. That does not match the actual EV design:
H-EV-SHUFFLE and H-EV-REPRESENTATION generate a FRESH, independent
confirmatory-test dataset for EACH replicate pair (correctly, per the
independent-replication requirement), so a single global hash could
never accommodate more than one pair. This version seals ONE hash PER
replicate_id, still refusing to reseal a given replicate_id with a
different hash (protecting against substitution/regeneration of that
specific replicate's data) while allowing many replicate_ids, each with
their own sealed hash, within one experiment.

This is a technical enforcement, not a policy note: the confirmatory
tier cannot be sealed before a freeze record exists (which must itself
already contain the frozen Candidate-1 permutation hash, where
applicable - see FreezeRecord), and a sealed confirmatory dataset for a
given replicate can be evaluated against exactly once per declared
(experiment_id, condition_id, replicate_id) key - a second attempt
raises, it does not silently proceed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class DataTier(str, Enum):
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    TRAINING_VALIDATION = "training_validation"
    CONFIRMATORY = "confirmatory"


class FirewallViolation(RuntimeError):
    pass


@dataclass
class FreezeRecord:
    """
    Records that all confirmatory-affecting choices were frozen before
    any confirmatory test set was generated/sealed. `frozen_permutation_hash`
    must be populated (not None) for any experiment that manipulates
    representation (H-EV-REPRESENTATION) - the permutation must be
    generated, validated, and persisted BEFORE this record is created.
    """
    frozen_config_hash: str
    frozen_permutation_hash: Optional[str]
    frozen_replicate_plan_hash: str
    frozen_statistical_plan_hash: str
    frozen_at_utc: str


class ConfirmatoryDataMode(str, Enum):
    """
    How confirmatory test data relate across replicates.

    PER_REPLICATE - each replicate has its own confirmatory dataset
                    (historical default; unchanged behaviour).
    SHARED        - one confirmatory dataset is used by every
                    replicate/condition, BY DESIGN.

    WHY THIS EXISTS: the firewall was ALREADY capable of enforcing a
    shared confirmatory set - sealing one hash under many replicate_ids
    is accepted, and exactly-once consumption per (condition, replicate)
    still applies. This mode is a PROVENANCE / DESIGN-SEMANTICS
    enhancement, NOT a capability fix. It records that sharing was
    intentional, so a reader need not infer design intent from the
    coincidence that several per-replicate seals carry identical hashes.
    When SHARED is declared the firewall additionally checks that the
    declared intent is honoured.
    """
    PER_REPLICATE = "PER_REPLICATE"
    SHARED = "SHARED"


@dataclass
class TestSetFirewall:
    """
    One instance per experiment. Tracks the freeze record and, per
    replicate_id, the sealed confirmatory-dataset hash and the set of
    (condition_id, replicate_id) evaluation keys already consumed.
    """
    experiment_id: str
    freeze_record: Optional[FreezeRecord] = None
    sealed_dataset_hashes: dict[str, str] = field(default_factory=dict)  # replicate_id -> hash
    confirmatory_data_mode: ConfirmatoryDataMode = ConfirmatoryDataMode.PER_REPLICATE
    _consumed_keys: set[str] = field(default_factory=set)

    def freeze(self, record: FreezeRecord) -> None:
        if self.freeze_record is not None:
            raise FirewallViolation(
                f"Experiment {self.experiment_id} is already frozen; "
                "refusing to re-freeze (this would allow post-hoc tuning)."
            )
        self.freeze_record = record

    def is_frozen(self) -> bool:
        return self.freeze_record is not None

    def seal_confirmatory_dataset(self, replicate_id: str, dataset_hash: str) -> None:
        """
        Seal the confirmatory-test dataset hash for one replicate_id.
        Must be called after freeze() and before any
        consume_confirmatory_evaluation() call for that replicate_id.
        Refuses to reseal the same replicate_id with a different hash
        (protects against accidental regeneration/substitution of that
        specific replicate's confirmatory data).
        """
        if self.freeze_record is None:
            raise FirewallViolation(
                f"Cannot seal a confirmatory dataset for {self.experiment_id}/{replicate_id} "
                "before the experiment is frozen (freeze() was not called)."
            )
        existing = self.sealed_dataset_hashes.get(replicate_id)
        if existing is not None and existing != dataset_hash:
            raise FirewallViolation(
                f"Experiment {self.experiment_id} replicate {replicate_id!r} already has a "
                f"sealed confirmatory dataset ({existing}); refusing to reseal with a "
                f"different dataset ({dataset_hash}). Regenerating/reselecting the "
                "confirmatory set after sealing is prohibited."
            )
        if self.confirmatory_data_mode is ConfirmatoryDataMode.SHARED:
            distinct = {h for rid, h in self.sealed_dataset_hashes.items()
                        if rid != replicate_id}
            if distinct and dataset_hash not in distinct:
                raise FirewallViolation(
                    f"Experiment {self.experiment_id} declares confirmatory_data_mode=SHARED, "
                    f"so every replicate must seal the SAME confirmatory dataset. Replicate "
                    f"{replicate_id!r} seals {dataset_hash}, but {sorted(distinct)} are "
                    "already sealed. Either the declaration or the data is wrong."
                )
        self.sealed_dataset_hashes[replicate_id] = dataset_hash

    def consume_confirmatory_evaluation(
        self,
        *,
        condition_id: str,
        replicate_id: str,
        observed_dataset_hash: str,
    ) -> None:
        """
        Must be called exactly once per (condition_id, replicate_id)
        before that replicate/condition's confirmatory metric is
        allowed to be recorded. Raises if called twice for the same
        key, or if the dataset hash does not match the hash sealed for
        that replicate_id (guards against accidental substitution).
        """
        sealed_hash = self.sealed_dataset_hashes.get(replicate_id)
        if sealed_hash is None:
            raise FirewallViolation(
                f"No confirmatory dataset has been sealed for "
                f"{self.experiment_id}/{replicate_id}."
            )
        if observed_dataset_hash != sealed_hash:
            raise FirewallViolation(
                f"Confirmatory dataset hash mismatch for {self.experiment_id}/{replicate_id}: "
                f"expected {sealed_hash}, got {observed_dataset_hash}. Refusing to evaluate "
                "against an unsealed or substituted dataset."
            )
        key = f"{condition_id}:{replicate_id}"
        if key in self._consumed_keys:
            raise FirewallViolation(
                f"Confirmatory test set already evaluated for {key} in "
                f"experiment {self.experiment_id}. The confirmatory set is "
                "single-use per (condition, replicate); repeated evaluation "
                "would constitute implicit tuning."
            )
        self._consumed_keys.add(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "frozen": self.freeze_record is not None,
            "freeze_record": self.freeze_record.__dict__ if self.freeze_record else None,
            "confirmatory_data_mode": self.confirmatory_data_mode.value,
            "sealed_dataset_hashes": dict(self.sealed_dataset_hashes),
            "consumed_evaluation_keys": sorted(self._consumed_keys),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "TestSetFirewall":
        data = json.loads(Path(path).read_text())
        fw = cls(experiment_id=data["experiment_id"])
        if data["freeze_record"]:
            fw.freeze_record = FreezeRecord(**data["freeze_record"])
        fw.confirmatory_data_mode = ConfirmatoryDataMode(
            data.get("confirmatory_data_mode", ConfirmatoryDataMode.PER_REPLICATE.value))
        fw.sealed_dataset_hashes = dict(data["sealed_dataset_hashes"])
        fw._consumed_keys = set(data["consumed_evaluation_keys"])
        return fw
