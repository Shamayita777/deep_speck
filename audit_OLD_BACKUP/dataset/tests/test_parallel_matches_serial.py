"""
D2 serial/parallel/checkpoint equivalence test — v6.4 API.

NON-EVIDENTIARY. Validates execution equivalence and checkpoint plumbing only.
It does not establish detector sensitivity or a scientific calibration result.

The frozen v6.4 calibrate_fault() API exposes aggregate results rather than
per-replicate records. Accordingly, equivalence is tested on the aggregate
fields actually returned by the production runner.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from audit.dataset.d2_sample_dependence import (  # noqa: E402
    DEFAULT_LAGS,
    FAMILYWISE_ALPHA,
    NEAR_DUPLICATE_RADII,
    PRACTICAL_EXCESS_TVD_THRESHOLD,
    familywise_test_count,
)
from audit.dataset.experiments.d2_gohr_sample_dependence import (  # noqa: E402
    DEFAULT_CALIBRATION_FRACTIONS,
    run_calibration,
)
from audit.dataset.experiments.d2_calibration_parallel_runner import (  # noqa: E402
    run_calibration_checkpointed,
)

total_tests = familywise_test_count(
    partition_count=3,
    lag_count=3 * len(DEFAULT_LAGS),
    near_duplicate_radius_count=len(NEAR_DUPLICATE_RADII),
    structured_view_count=15,
    multivariate_count=3,
)
ALPHA = FAMILYWISE_ALPHA / total_tests

results: list[tuple[str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    results.append((name, status))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")


def main() -> None:
    # Small fixed dataset: this is a plumbing/equivalence workload, not calibration.
    rng = np.random.default_rng(2024)
    clean = rng.integers(
        0,
        2,
        size=(20_000, 64),
        dtype=np.uint8,
    )
    SEED = 4242

    # Reduced workload is intentional: this test checks equality of execution paths,
    # not detector sensitivity at the frozen production calibration workload.
    QUERY_COUNT = 256
    TABLES = 4
    NEAR_NULL_PAIRS = 2_000


    def stable_result_tree(tree: Any) -> Any:
        """
        Remove runtime/checkpoint metadata and retain the scientific result tree.

        The function is deliberately recursive so it remains robust to harmless
        metadata additions in the runner.
        """
        if isinstance(tree, dict):
            volatile = {
                "_checkpoint_meta",
                "wall_clock_seconds",
                "elapsed_seconds",
                "duration_seconds",
                "timestamp",
                "timestamps",
            }
            return {
                k: stable_result_tree(v)
                for k, v in sorted(tree.items())
                if k not in volatile
            }
        if isinstance(tree, list):
            return [stable_result_tree(v) for v in tree]
        if isinstance(tree, tuple):
            return tuple(stable_result_tree(v) for v in tree)
        return tree


    def comparable_near_duplicate_tree(out: dict[str, Any]) -> dict[str, Any]:
        """
        Extract only the aggregate near-duplicate calibration fields exposed by
        the frozen v6.4 API.

        Per-replicate `detected` sequences and `seed_spawn_key` sequences are not
        asserted because v6.4 does not expose them through calibrate_fault().
        """
        tree = out["near_duplicate_injection"]
        result: dict[str, Any] = {}

        for radius_str, by_fraction in tree.items():
            result[str(radius_str)] = {}
            for fraction_str, record in by_fraction.items():
                keep = {}
                for key in (
                    "detection_rate",
                    "detections",
                    "detection_rate_95_ci",
                    "point_estimate_target_met",
                    "confidence_lower_bound_target_met",
                    "target",
                    "replicates",
                    "trial_semantics",
                ):
                    if key in record:
                        keep[key] = record[key]
                result[str(radius_str)][str(fraction_str)] = keep

        return result


    def timed_run(checkpoint_dir: str, max_workers: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        out = run_calibration_checkpointed(
            clean,
            replicates=100,
            seed=SEED,
            query_count=QUERY_COUNT,
            tables=TABLES,
            null_pairs=NEAR_NULL_PAIRS,
            pairs=200,
            null_permutations=500,
            alpha=ALPHA,
            practical_excess_tvd_threshold=PRACTICAL_EXCESS_TVD_THRESHOLD,
            checkpoint_dir=checkpoint_dir,
            max_workers=max_workers,
            progress_cb=lambda message: print("  ", message),
        )
        print(f"  total wall clock: {time.perf_counter() - t0:.1f}s")
        return out


    # ---------------------------------------------------------------------------
    # 1. Canonical serial reference using the production run_calibration path.
    # ---------------------------------------------------------------------------
    print("=== Building canonical serial reference ===")

    serial_out = {}
    for radius in NEAR_DUPLICATE_RADII:
        serial_out[str(radius)] = {}
        for fraction in DEFAULT_CALIBRATION_FRACTIONS:
            seed_here = SEED + 10_000 + radius * 100 + int(fraction * 1_000_000)

            from audit.dataset.d2_sample_dependence import (  # noqa: E402
                calibrate_fault,
                inject_near_duplicates,
            )
            import audit.dataset.experiments.d2_gohr_sample_dependence as drv  # noqa: E402

            detector = drv._near_detector(
                radius,
                QUERY_COUNT,
                TABLES,
                NEAR_NULL_PAIRS,
                ALPHA,
            )

            result = calibrate_fault(
                clean_features=clean,
                injector=lambda x, rr, f=fraction, rad=radius:
                    inject_near_duplicates(x, f, rad, rr),
                detector=detector,
                replicates=100,
                seed=seed_here,
            )
            serial_out[str(radius)][f"{fraction:.6g}"] = result


    reference = {
        "near_duplicate_injection": serial_out,
    }


    # ---------------------------------------------------------------------------
    # 2. Checkpointed serial execution matches canonical execution.
    # ---------------------------------------------------------------------------
    serial_ckpt = "/tmp/d2_v64_ckpt_serial"
    parallel_ckpt = "/tmp/d2_v64_ckpt_parallel"

    shutil.rmtree(serial_ckpt, ignore_errors=True)
    shutil.rmtree(parallel_ckpt, ignore_errors=True)

    print()
    print("=== Checkpointed runner, max_workers=1 ===")
    ckpt_serial_out = timed_run(serial_ckpt, max_workers=1)

    serial_reference_comparable = comparable_near_duplicate_tree(reference)
    checkpointed_serial_comparable = comparable_near_duplicate_tree(ckpt_serial_out)

    check(
        "checkpointed runner (serial) matches canonical v6.4 aggregate reference",
        checkpointed_serial_comparable == serial_reference_comparable,
    )


    # ---------------------------------------------------------------------------
    # 3. Checkpointed parallel execution matches checkpointed serial execution.
    # ---------------------------------------------------------------------------
    print()
    print("=== Checkpointed runner, max_workers=4 ===")
    ckpt_parallel_out = timed_run(parallel_ckpt, max_workers=4)

    checkpointed_parallel_comparable = comparable_near_duplicate_tree(
        ckpt_parallel_out
    )

    check(
        "checkpointed runner (parallel) matches canonical v6.4 aggregate reference",
        checkpointed_parallel_comparable == serial_reference_comparable,
    )
    check(
        "serial and parallel checkpointed aggregate results are identical",
        checkpointed_serial_comparable == checkpointed_parallel_comparable,
    )


    # ---------------------------------------------------------------------------
    # 4. Checkpoint/resume: remove five condition checkpoints and verify that
    # exactly those missing conditions are recomputed and final result is unchanged.
    # ---------------------------------------------------------------------------
    print()
    print(
        "=== Checkpoint/resume: remove five checkpoint files and rerun ==="
    )

    checkpoint_files = sorted(
        p for p in Path(parallel_ckpt).iterdir() if p.is_file()
    )
    print(f"  {len(checkpoint_files)} checkpoint files present after full run")

    if len(checkpoint_files) < 5:
        raise AssertionError(
            f"Expected at least 5 checkpoint files, found {len(checkpoint_files)}"
        )

    for path in checkpoint_files[:5]:
        path.unlink()

    print(
        f"  removed 5 checkpoint files; "
        f"{len(list(Path(parallel_ckpt).iterdir()))} remain"
    )

    resumed_out = timed_run(parallel_ckpt, max_workers=4)

    meta = resumed_out.get("_checkpoint_meta", {})
    computed_this_invocation = meta.get("conditions_computed_this_invocation")

    check(
        "resumed run recomputes exactly the five missing conditions",
        computed_this_invocation == 5,
        f"got {computed_this_invocation}",
    )

    resumed_comparable = comparable_near_duplicate_tree(resumed_out)

    check(
        "resumed final aggregate result matches uninterrupted reference",
        resumed_comparable == serial_reference_comparable,
    )

    print()
    passed = sum(1 for _, status in results if status == "PASS")
    print(f"=== {passed}/{len(results)} equivalence checks passed ===")
    print(
        "NON-EVIDENTIARY: validates serial/parallel/checkpoint plumbing only; "
        "does not establish detector sensitivity or a scientific calibration result."
    )


if __name__ == "__main__":
    main()
