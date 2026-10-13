#!/usr/bin/env python3
"""
Is the declared-architecture difference a Keras SERIALIZATION detail, or
a real change of architecture?

READ-ONLY. Builds both arms, compares every field of the committed
signature against the one this stack produces, and truncates NOTHING.

A difference is BENIGN only if it is a key that exists on one side and
not the other AND whose value is null - that is, a newer Keras writing
out a default that an older one omitted. Anything else is SUBSTANTIVE:
a different filter count, a different activation, different wiring, a
different parameter total, or a key whose value actually differs.

    python3 verify_architecture_drift.py

Exit 0 = every difference is benign serialization; the architecture is
unchanged and a re-capture is safe.
Exit 1 = at least one substantive difference; do NOT re-capture, the
comparison premise must be re-established first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from audit.implementation.adapters.gohr_impl_comparison import (
    load_independent, load_reference,
)
from audit.implementation.architecture_signature import architecture_signature

DECLARED = Path("audit/implementation/adapters/gohr_declared_architecture.json")


def walk(a, b, path=""):
    """Every leaf difference between two nested structures."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else key
            if key not in a:
                yield (sub, "<absent>", b[key])
            elif key not in b:
                yield (sub, a[key], "<absent>")
            else:
                yield from walk(a[key], b[key], sub)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield (f"{path}[len]", len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk(x, y, f"{path}[{i}]")
    elif a != b:
        yield (path, a, b)


def benign(committed, current) -> bool:
    """A default that a newer Keras serialises and an older one omits."""
    return ((committed == "<absent>" and current is None)
            or (current == "<absent>" and committed is None))


def main() -> int:
    record = json.loads(DECLARED.read_text())
    committed = record["signature"]

    _, train_nets = load_reference()
    ref = architecture_signature(train_nets.make_resnet(depth=10, reg_param=1e-5))
    build, _ = load_independent()
    ind = architecture_signature(build(depth=10, filters=32, dense_1=64, dense_2=64,
                                       kernel_size=3, word_size=16, input_words=4,
                                       l2_strength=1e-5))

    print("COMMITTED captured with", record.get("captured_with"))
    try:
        import keras
        import tensorflow as tf
        print("CURRENT  stack          "
              f"{{'keras': '{keras.__version__}', 'tensorflow': '{tf.__version__}'}}")
    except Exception:                                             # noqa: BLE001
        pass
    print()

    # 1. the premise: both arms must agree with EACH OTHER on this stack
    arm_diffs = list(walk(ref, ind))
    arm_diffs = [d for d in arm_diffs if not d[0].endswith("digest")]
    print(f"reference vs independent, on THIS stack: {len(arm_diffs)} difference(s)")
    if arm_diffs:
        print("  THE COMPARISON PREMISE DOES NOT HOLD. These are not the same")
        print("  architecture, so a measured difference would confound")
        print("  implementation with architecture:")
        for p, a, b in arm_diffs[:40]:
            print(f"    {p}: reference={a!r} independent={b!r}")
        return 1
    print("  identical - the premise holds\n")

    # 2. invariants that no serialisation change may touch
    print("structural invariants, committed -> current")
    hard = False
    for field in ("layer_count", "total_parameters", "trainable_parameters",
                  "non_trainable_parameters", "input_shape", "output_shape",
                  "connectivity"):
        same = committed.get(field) == ref.get(field)
        label = field if field != "connectivity" else "connectivity (wiring)"
        shown = "identical" if same else f"{committed.get(field)!r} -> {ref.get(field)!r}"
        if field == "connectivity" and not same:
            shown = "DIFFERENT"
        print(f"  {'OK  ' if same else 'DIFF'}  {label:32} {shown}")
        hard = hard or not same
    print()

    # 3. every remaining difference, classified, nothing suppressed
    diffs = [d for d in walk(committed, ref) if not d[0].endswith("digest")]
    benign_diffs = [d for d in diffs if benign(d[1], d[2])]
    substantive = [d for d in diffs if not benign(d[1], d[2])]

    print(f"field differences: {len(diffs)} total  "
          f"({len(benign_diffs)} benign, {len(substantive)} substantive)\n")

    if benign_diffs:
        keys = sorted({d[0].rsplit(".", 1)[-1] for d in benign_diffs})
        print(f"BENIGN - a default serialised by one Keras and omitted by the other")
        print(f"  affected keys: {keys}")
        print(f"  affected fields: {len(benign_diffs)}")
        for p, a, b in benign_diffs[:5]:
            print(f"    {p}: committed={a!r} current={b!r}")
        if len(benign_diffs) > 5:
            print(f"    ... and {len(benign_diffs) - 5} more of the same shape "
                  f"(all listed in the JSON report)")
        print()

    if substantive:
        print("SUBSTANTIVE - NOT a serialisation detail:")
        for p, a, b in substantive:
            print(f"    {p}: committed={a!r} current={b!r}")
        print()

    out = Path("architecture_drift_report.json")
    out.write_text(json.dumps({
        "committed_captured_with": record.get("captured_with"),
        "committed_digest": committed.get("digest"),
        "current_digest": ref.get("digest"),
        "arms_agree_on_this_stack": not arm_diffs,
        "structural_invariants_identical": not hard,
        "benign_differences": [{"path": p, "committed": a, "current": b}
                               for p, a, b in benign_diffs],
        "substantive_differences": [{"path": p, "committed": a, "current": b}
                                    for p, a, b in substantive],
    }, indent=2, default=str))

    print("VERDICT")
    if substantive or hard:
        print("  SUBSTANTIVE DIFFERENCE. Do NOT delete and re-capture: the")
        print("  architecture itself has changed and the comparison premise must")
        print("  be re-established before anything is frozen.")
        print(f"  full report: {out}")
        return 1
    print("  SERIALISATION ONLY. Every difference is a null-valued default that")
    print("  one Keras version writes and the other omits. Layer count, parameter")
    print("  counts, wiring, shapes and every configured value are identical, and")
    print("  both arms agree on this stack. Deleting the committed file and")
    print("  re-capturing records the same architecture under this stack's")
    print("  serialisation. Keep this report with the run's evidence.")
    print(f"  full report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
