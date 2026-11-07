"""
Factories for the II-4 parallel-scheduler tests. A worker subprocess imports one of
these by name, so it must rebuild the identical plan and adapter from plain
arguments. Accuracies are a deterministic function of (arm, seed) only, never of
Python's per-process string hash, so serial and parallel runs are exactly comparable.
"""
from __future__ import annotations

import time
from pathlib import Path

from audit.tests.test_ii4_comparison import ARM_A, ARM_B, FakeAdapter, design, plan


def accuracy(arm, block_id, seed):
    return 0.9290 + (0.0004 if arm == ARM_A else 0.0) + 0.00003 * ((seed * 7) % 11)


class PacedFakeAdapter(FakeAdapter):
    """The frozen test adapter, with each arm's training held open for `arm_seconds`.

    Real training takes hours, so two workers always overlap; the unpaced fake trains in
    milliseconds and could finish before the second worker has started. The pause runs
    inside the training window, so it only makes the concurrency observable.
    """

    def __init__(self, *, arm_seconds=0.0, **kw):
        super().__init__(**kw)
        self.arm_seconds = arm_seconds

    def train_arm(self, *args, **kw):
        if self.arm_seconds:
            time.sleep(self.arm_seconds)
        return super().train_arm(*args, **kw)


def make(root, n_blocks=4, min_valid=3, replacements=1, fail_train=(), fail_build=(), fail_at=(),
         arm_seconds=0.6):
    """fail_at: [[block_id, arm, epoch], ...] raises an injected interruption after that epoch."""
    root = Path(root)
    d = design(n_blocks=n_blocks, min_valid_blocks=min_valid, max_replacement_blocks=replacements)
    p = plan(root.parent, d=d, output_dir=root, created_at_utc="FROZEN")
    ad = PacedFakeAdapter(arm_ids=(ARM_A, ARM_B), accuracy_fn=accuracy, arm_seconds=arm_seconds,
                          fail_train=[tuple(x) for x in fail_train], fail_build=list(fail_build),
                          fail_at_epoch={(b, a): int(e) for b, a, e in fail_at})
    return p, ad
