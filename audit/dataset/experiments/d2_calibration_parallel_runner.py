"""Checkpointed / parallel orchestration for the 25 independent calibration
conditions (4 near-duplicate radii x 5 fractions, plus 5 lag-copy fractions).

This module does NOT reimplement any statistics. It calls exactly the same
functions the canonical serial `run_calibration()` in
`audit.dataset.experiments.d2_gohr_sample_dependence` calls
(`calibrate_fault`, `_near_detector`, `_lag_detector`,
`inject_near_duplicates`, `inject_lag_copy`), using the IDENTICAL
per-condition seed formulas copied verbatim from that function. It exists
only to decompose those 25 independent calibrate_fault(...) calls into
separately-checkpointable, optionally-parallel units of work, and to
reassemble their outputs into the exact same nested dict shape
run_calibration() returns.

Equivalence with the serial implementation is not assumed -- it is tested.
See test_parallel_matches_serial.py, which asserts this module produces
value-identical output to run_calibration() for a shared set of conditions
under a shared seed.

Safety properties relied on:
  - Each condition's seed is a pure, deterministic function of
    (outer seed, radius or None, fraction) via the formulas below -- copied
    from run_calibration() -- so which process/order a condition executes in
    cannot change its result.
  - calibrate_fault() internally uses np.random.SeedSequence(seed_here).spawn(
    replicates), which is itself deterministic and process-independent.
  - Conditions share no mutable state (each receives a read-only `clean`
    array).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from audit.dataset.d2_sample_dependence import (
    calibrate_fault, inject_near_duplicates, inject_lag_copy, CALIBRATION_DETECTION_TARGET,
)
from audit.dataset.experiments.d2_gohr_sample_dependence import (
    _near_detector,
    _lag_detector,
    NEAR_DUPLICATE_RADII,
    DEFAULT_CALIBRATION_FRACTIONS,
)


def _condition_id(kind: str, fraction: float, radius: int | None = None) -> str:
    if kind == "near_duplicate":
        return f"near_duplicate:radius={radius}:fraction={fraction:.6g}"
    return f"lag_copy:fraction={fraction:.6g}"


def enumerate_conditions(seed: int):
    """Yield (condition_id, kind, radius_or_None, fraction, seed_here) for all
    25 conditions, using the seed formulas copied verbatim from
    run_calibration() in d2_gohr_sample_dependence.py. THESE FORMULAS MUST BE
    KEPT IN SYNC WITH run_calibration(); test_parallel_matches_serial.py
    checks this by comparing actual output, not by comparing source text.
    """
    for radius in NEAR_DUPLICATE_RADII:
        for fraction in DEFAULT_CALIBRATION_FRACTIONS:
            seed_here = seed + 10_000 + radius * 100 + int(fraction * 1_000_000)  # noqa: E501 -- copied from run_calibration()
            yield _condition_id("near_duplicate", fraction, radius), "near_duplicate", radius, fraction, seed_here
    for fraction in DEFAULT_CALIBRATION_FRACTIONS:
        seed_here = seed + 50_000 + int(fraction * 1_000_000)  # noqa: E501 -- copied from run_calibration()
        yield _condition_id("lag_copy", fraction), "lag_copy", None, fraction, seed_here


def _run_one_condition(args):
    """Runs in a worker process (or the main process for serial mode).
    Pure function of its arguments -- no shared mutable state.
    """
    (kind, radius, fraction, seed_here, replicates,
     query_count, tables, null_pairs, pairs, null_permutations, alpha,
     practical_excess_tvd_threshold, clean) = args
    if kind == "near_duplicate":
        detector = _near_detector(radius, query_count, tables, null_pairs, alpha)
        result = calibrate_fault(
            clean_features=clean,
            injector=lambda x, r, f=fraction, rad=radius: inject_near_duplicates(x, f, rad, r),
            detector=detector,
            replicates=replicates,
            seed=seed_here,
        )
        result["target_component"] = "d2_1_near_duplicate_structure"
    else:
        detector = _lag_detector(pairs, null_permutations, alpha, practical_excess_tvd_threshold)
        result = calibrate_fault(
            clean_features=clean,
            injector=lambda x, r, f=fraction: inject_lag_copy(x, f, 1, r),
            detector=detector,
            replicates=replicates,
            seed=seed_here,
        )
        result["target_component"] = "d2_2_serial_dependence"
    return result


def run_calibration_checkpointed(
    clean,
    *,
    replicates,
    seed,
    query_count,
    tables,
    null_pairs,
    pairs,
    null_permutations,
    alpha,
    practical_excess_tvd_threshold,
    checkpoint_dir: str,
    max_workers: int = 1,
    progress_cb=None,
):
    """Same predeclared 25-condition calibration design as run_calibration(),
    decomposed into independently-checkpointed units and optionally run
    across `max_workers` processes. Restarting with the same checkpoint_dir
    skips conditions already completed. The combined return value has the
    exact same shape as run_calibration()'s return value (before the
    final driver's monotonicity diagnostic is attached).
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    conditions = list(enumerate_conditions(seed))
    todo = []
    done_results = {}
    for cid, kind, radius, fraction, seed_here in conditions:
        ckpt_file = ckpt_dir / (cid.replace(":", "__").replace("=", "-") + ".json")
        if ckpt_file.exists():
            with open(ckpt_file) as f:
                done_results[cid] = json.load(f)
        else:
            todo.append((cid, kind, radius, fraction, seed_here, ckpt_file))

    def _submit_args(kind, radius, fraction, seed_here):
        return (kind, radius, fraction, seed_here, replicates, query_count, tables,
                null_pairs, pairs, null_permutations, alpha, practical_excess_tvd_threshold, clean)

    t0 = time.perf_counter()
    n_total = len(conditions)
    n_already_done = len(done_results)
    if progress_cb:
        progress_cb(f"{n_already_done}/{n_total} conditions already checkpointed; {len(todo)} to run "
                    f"(max_workers={max_workers})")

    if max_workers <= 1:
        for cid, kind, radius, fraction, seed_here, ckpt_file in todo:
            t_c0 = time.perf_counter()
            result = _run_one_condition(_submit_args(kind, radius, fraction, seed_here))
            with open(ckpt_file, "w") as f:
                json.dump(result, f)
            done_results[cid] = result
            if progress_cb:
                progress_cb(f"[serial] {cid} done in {time.perf_counter()-t_c0:.1f}s")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_one_condition, _submit_args(kind, radius, fraction, seed_here)): (cid, ckpt_file)
                for cid, kind, radius, fraction, seed_here, ckpt_file in todo
            }
            for fut in as_completed(futures):
                cid, ckpt_file = futures[fut]
                t_c0 = time.perf_counter()
                result = fut.result()
                with open(ckpt_file, "w") as f:
                    json.dump(result, f)
                done_results[cid] = result
                if progress_cb:
                    progress_cb(f"[parallel] {cid} collected ({time.perf_counter()-t0:.1f}s elapsed total)")

    # Reassemble into the exact shape run_calibration() returns.
    out = {
        "protocol": "component-specific controlled fault-injection sensitivity calibration",
        "replicates": int(replicates),
        "detection_target": CALIBRATION_DETECTION_TARGET,
        "target_semantics": "Wilson lower bound is the only confidence-supported target criterion; point estimate is reported separately",
        "calibration_is_not_clean_evidence": True,
        "near_duplicate_injection": {},
        "lag_copy_injection": {},
    }
    for cid, kind, radius, fraction, seed_here in conditions:
        r = done_results[cid]
        if kind == "near_duplicate":
            out["near_duplicate_injection"].setdefault(str(radius), {})[f"{fraction:.6g}"] = r
        else:
            out["lag_copy_injection"][f"{fraction:.6g}"] = r

    out["_checkpoint_meta"] = {
        "wall_clock_seconds_this_invocation": time.perf_counter() - t0,
        "conditions_already_checkpointed_at_start": n_already_done,
        "conditions_computed_this_invocation": len(todo),
        "max_workers": max_workers,
    }
    return out
