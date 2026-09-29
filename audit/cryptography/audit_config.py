"""
Frozen scientific reference configuration.

Every scientifically important parameter is declared HERE and passed
explicitly. Nothing relies on a constructor default: the historical
depth-5 finding arose precisely because `GohrModel()`'s default depth was
never overridden, and a silent default is indistinguishable from a
deliberate choice once the run is over.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ReferenceConfig:
    cipher: str = "speck32_64"
    rounds: int = 5
    differential: tuple = (0x0040, 0x0000)
    depth: int = 10
    l2_reg: float = 1e-5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["differential"] = list(self.differential)
        return d


#: The single declared reference configuration for all current CE experiments.
REFERENCE = ReferenceConfig()

#: Independently verified depth-10 checkpoint (Archive/best5depth10.h5).
#: Verified by h5py inspection, NOT by filename: 10 residual merges,
#: 21 Conv1D layers, L2 = 1e-5.
REFERENCE_CHECKPOINT_SHA256 = (
    "256eb4a5ba93414f0a46ffd498d6121a03705698ac2a81764c2375207c5d6737"
)

#: Historical depth-5 checkpoints. Retained as HISTORICAL EVIDENCE ONLY;
#: the current audit must never load these. The first has a misleading filename.
HISTORICAL_DEPTH5_CHECKPOINTS = {
    "de630afcf95e7d0d40c33ff70623a7dda9290c6ae7e80133323ceb80997ad2ea":
        "evidence/ce1/best5depth10 (10).h5  (filename says depth10; actually depth 5)",
    "a110fad1e5b3adc3600acf9bbfd26cd0decdc563f6c5f08d965a8dd3c50a8308":
        "evidence/ce1/signal_destroyed.h5  (depth 5)",
}


class ConfigurationError(RuntimeError):
    """Raised when a audit run is not at the declared reference configuration."""


def require_reference_config(*, rounds: int, differential, depth: int,
                             l2_reg: float | None = None) -> None:
    """Fail loudly if any declared parameter deviates from the reference."""
    problems = []
    if rounds != REFERENCE.rounds:
        problems.append(f"rounds={rounds!r} (reference {REFERENCE.rounds})")
    if tuple(differential) != REFERENCE.differential:
        problems.append(
            f"differential={tuple(differential)!r} (reference {REFERENCE.differential})")
    if depth != REFERENCE.depth:
        problems.append(f"depth={depth!r} (reference {REFERENCE.depth})")
    if l2_reg is not None and l2_reg != REFERENCE.l2_reg:
        problems.append(f"l2_reg={l2_reg!r} (reference {REFERENCE.l2_reg})")
    if problems:
        raise ConfigurationError(
            "The audit requires the declared reference configuration; deviations: "
            + "; ".join(problems))
