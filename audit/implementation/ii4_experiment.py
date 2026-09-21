"""
II-4 orchestration: dataset-blocked conformance-impact experiment.

Executes K blocks; within each block trains the DECLARED-value arm and
the REALIZED-value arm on the SAME persisted dataset, with INDEPENDENT
initialization and shuffle, then evaluates both terminal-epoch models on
the block's sealed confirmatory test set.

This module builds and validates the plan without training. Training is
performed only when `execute=True` is passed explicitly, so importing or
dry-running this module cannot start GPU work.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from audit.common.outcomes import ExecutionMode
from audit.common.provenance import sha256_bytes, utc_timestamp
from audit.common.strict_json import dumps_strict
from audit.implementation.conformance import BlockOutcome, ConformanceFactor
from audit.implementation.ii4_design import (
    CONDITIONAL_INFERENCE_STATEMENT,
    ESTIMAND_STATEMENT,
    EVALUATION_RULE_RATIONALE,
    FROZEN_BASE_SEED,
    GOHR_DEPTH_FACTOR,
    GOHR_FIXED_PROTOCOL,
    II4_DESIGN,
    INTERPRETATION_RULE,
    STATISTICAL_UNIT_STATEMENT,
    II4Design,
)


@dataclass
class II4SeedManifest:
    """
    Predetermined, recorded seeds.

    One seed PER ARM PER BLOCK - never one seed shared by both arms.
    The arms differ in an architectural parameter, so a shared integer
    seed would produce non-corresponding initial weights while creating
    the false appearance of paired initialization.
    """
    base_seed: int
    n_blocks: int

    def seed_for(self, block_index: int, arm: str) -> int:
        if arm not in ("declared", "realized"):
            raise ValueError(f"unknown arm {arm!r}")
        offset = 0 if arm == "declared" else 1
        return self.base_seed + 2 * block_index + offset

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_seed": self.base_seed,
            "n_blocks": self.n_blocks,
            "policy": ("one distinct seed per (block, arm); arms are NEVER given the same "
                       "seed - see II4Design.same_model_initialization=False"),
            "seeds": {
                f"block{i}": {arm: self.seed_for(i, arm) for arm in ("declared", "realized")}
                for i in range(self.n_blocks)
            },
        }


@dataclass
class II4Plan:
    """A fully specified, hash-stable II-4 execution plan."""
    experiment_id: str
    design: II4Design
    factor: ConformanceFactor
    fixed_protocol: dict[str, Any]
    seeds: II4SeedManifest
    output_dir: Path
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION
    created_at_utc: str = field(default_factory=utc_timestamp)

    def preregistration(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "created_at_utc": self.created_at_utc,
            "execution_mode": self.execution_mode.value,
            "design": self.design.to_dict(),
            "conformance_factor": self.factor.to_dict(),
            "fixed_protocol": {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in self.fixed_protocol.items()},
            "seed_manifest": self.seeds.to_dict(),
            "estimand": ESTIMAND_STATEMENT,
            "conditional_inference": CONDITIONAL_INFERENCE_STATEMENT,
            "interpretation_rule": INTERPRETATION_RULE,
            "confirmatory_test_set": {
                "scope": "GLOBAL_SHARED",
                "firewall_mode": "SHARED",
                "lifecycle": ("generated ONCE before block 0 -> persisted -> sha256 -> "
                              "sealed under every block id -> reused for all blocks and "
                              "both arms -> consumed exactly once per (arm, block)"),
                "per_block_test_sets": False,
            },
            "statistical_unit": STATISTICAL_UNIT_STATEMENT,
            "evaluation_rule_rationale": EVALUATION_RULE_RATIONALE,
            "confirmatory_endpoint": self.design.primary_endpoint,
            "secondary_descriptive": [
                "max_val_acc (historical comparability only; optimistically biased as a "
                "generalisation estimate and NOT the confirmatory endpoint)",
                "final_val_acc",
                "best_val_loss_checkpoint_hash (provenance/debugging only; does NOT "
                "determine the primary endpoint)",
            ],
        }

    def preregistration_hash(self) -> str:
        return sha256_bytes(
            dumps_strict(self.preregistration(), sort_keys=True, default=str).encode("utf-8"))

    def config_fingerprint(self) -> str:
        """
        Hash of everything that defines the SCIENTIFIC experiment, excluding
        wall-clock creation time. Used on resume: a resumed run whose
        fingerprint differs from the recorded one is a different experiment
        and must fail loudly rather than continue.
        """
        body = {k: v for k, v in self.preregistration().items() if k != "created_at_utc"}
        return sha256_bytes(dumps_strict(body, sort_keys=True, default=str).encode("utf-8"))

    def validate_production(self) -> list[str]:
        """
        Additional checks that apply ONLY to a production run: the plan must
        be exactly the frozen specification, so that no smoke setting can
        leak into evidence.
        """
        problems = list(self.validate())
        if self.execution_mode is not ExecutionMode.PRODUCTION:
            problems.append("execution_mode is not PRODUCTION.")
        if "SMOKE" in self.experiment_id.upper():
            problems.append("a production experiment_id must not contain 'SMOKE'.")
        if self.design.to_dict() != II4_DESIGN.to_dict():
            problems.append("design differs from the frozen II4_DESIGN.")
        if self.fixed_protocol != dict(GOHR_FIXED_PROTOCOL):
            problems.append("fixed_protocol differs from the frozen GOHR_FIXED_PROTOCOL.")
        if self.seeds.base_seed != FROZEN_BASE_SEED:
            problems.append(f"base_seed {self.seeds.base_seed} != frozen {FROZEN_BASE_SEED}.")
        return problems

    def validate(self) -> list[str]:
        """Pre-flight checks. Must be empty before execution is permitted."""
        problems: list[str] = []
        d = self.design
        if d.evaluation_rule != "terminal_epoch":
            problems.append(
                f"evaluation_rule is {d.evaluation_rule!r}; the frozen II-4 primary rule is "
                "'terminal_epoch'.")
        if d.primary_endpoint != "sealed_confirmatory_test_accuracy":
            problems.append(
                f"primary_endpoint is {d.primary_endpoint!r}; validation-derived metrics must "
                "not be the confirmatory endpoint.")
        if d.same_model_initialization:
            problems.append("same_model_initialization must be False for a conformance comparison.")
        if not d.same_dataset:
            problems.append("same_dataset must be True: the block design depends on it.")
        if self.fixed_protocol.get("epochs") != d.epochs:
            problems.append("fixed_protocol epochs disagree with the design epochs.")
        if self.fixed_protocol.get("shuffle") is not True:
            problems.append("shuffle must be True to match the reference training behaviour.")
        # arms must be distinct and explicit
        arms = self.factor.arm_values
        if arms["declared"] == arms["realized"]:
            problems.append("declared and realized arm values are identical.")
        # seeds must be pairwise distinct
        seen = set()
        for i in range(d.n_blocks):
            for arm in ("declared", "realized"):
                s = self.seeds.seed_for(i, arm)
                if s in seen:
                    problems.append(f"duplicate seed {s} at block{i}/{arm}.")
                seen.add(s)
        return problems


def build_gohr_ii4_plan(
    *, output_dir: Path, base_seed: int = FROZEN_BASE_SEED,
    experiment_id: str = "II-4-GOHR-DEPTH-V1",
) -> II4Plan:
    """Construct (not execute) the frozen Gohr II-4 plan."""
    return II4Plan(
        experiment_id=experiment_id,
        design=II4_DESIGN,
        factor=GOHR_DEPTH_FACTOR,
        fixed_protocol=dict(GOHR_FIXED_PROTOCOL),
        seeds=II4SeedManifest(base_seed=base_seed, n_blocks=II4_DESIGN.n_blocks),
        output_dir=Path(output_dir),
    )


def new_block(block_index: int, dataset_hash: str, plan: II4Plan) -> BlockOutcome:
    """Create an empty block outcome with its predetermined seeds attached."""
    return BlockOutcome(
        block_id=f"block{block_index}",
        dataset_hash=dataset_hash,
        declared_seed=plan.seeds.seed_for(block_index, "declared"),
        realized_seed=plan.seeds.seed_for(block_index, "realized"),
    )


def collect_valid_differences(blocks: list[BlockOutcome]) -> list[float]:
    """
    Signed differences from VALID blocks only.

    A block with a failed arm contributes nothing - not even its
    surviving arm. Retaining a half-block would silently convert a
    paired observation into an unpaired one and destroy the blocking the
    design depends on.
    """
    return [b.signed_difference for b in blocks if b.is_valid]


def build_gohr_ii4_smoke_plan(
    *, output_dir: Path, n_blocks: int = 10, epochs: int = 2,
    train_samples: int = 2_000, validation_samples: int = 500, test_samples: int = 2_000,
) -> II4Plan:
    """
    NON-EVIDENTIARY smoke plan: tiny data and epochs, execution_mode SMOKE,
    experiment_id containing 'SMOKE-NONEVIDENTIARY'. It exercises the full
    runner path but can never pass validate_production(), so it cannot be
    mistaken for II-4 evidence.
    """
    design = II4Design(
        n_blocks=n_blocks, min_valid_blocks=max(2, n_blocks - 1),
        primary_endpoint=II4_DESIGN.primary_endpoint,
        evaluation_rule=II4_DESIGN.evaluation_rule, epochs=epochs,
        same_dataset=True, same_model_initialization=False,
        same_training_shuffle_stream=False, same_evaluation_data=True,
        alpha=II4_DESIGN.alpha, delta_materiality=II4_DESIGN.delta_materiality,
        delta_is_equivalence_margin=True,
    )
    protocol = dict(GOHR_FIXED_PROTOCOL)
    protocol.update(epochs=epochs, train_samples=train_samples,
                    validation_samples=validation_samples,
                    sealed_test_samples=test_samples, batch_size=250)
    return II4Plan(
        experiment_id="II-4-SMOKE-NONEVIDENTIARY",
        design=design, factor=GOHR_DEPTH_FACTOR, fixed_protocol=protocol,
        seeds=II4SeedManifest(base_seed=FROZEN_BASE_SEED, n_blocks=n_blocks),
        output_dir=Path(output_dir), execution_mode=ExecutionMode.SMOKE,
    )
