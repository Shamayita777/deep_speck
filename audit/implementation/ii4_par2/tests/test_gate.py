"""
The PAR2 gate combines the frozen gate's output with the corrected measurement.

The frozen gate is supplied as recorded dictionaries shaped exactly as
run_ii4_impl.readiness_gate returns them, so no frozen module is patched.
"""
from __future__ import annotations

import copy

from audit.implementation.ii4_par2 import gate

MEM_PROBLEM = ("0.24 GB of memory available against an estimated requirement of 8.0 GB. The training "
               "arrays are cast to float32 for training, which is four times their on-disk uint8 size.")


def frozen(*, mem_problem=True, other=(), dir_problems=(), suites_clear=True, unack=()):
    mem = {"total_gb": 33.6, "available_gb": 0.24, "required_gb": 8.0,
           "problems": [MEM_PROBLEM] if mem_problem else []}
    pre = {"memory": mem, "problems": [*([MEM_PROBLEM] if mem_problem else []), *other],
           "unacknowledged": list(unack), "warnings": [], "clear_to_launch": False}
    g = {"output_dir": "x", "dir_problems": list(dir_problems), "suites": {"clear": suites_clear},
         "preflight": pre}
    return (not mem_problem and not other and not dir_problems and suites_clear and not unack), g


def corrected(clear=True):
    return {"clear": clear, "required_bytes": 8_000_000_000, "blocks": [] if clear else ["below"],
            "usable_bytes": {"0.0": 1, "0.5": 2, "1.0": 3}}


SUITES = [{"suite": "par1", "clear": True, "tail": "40 passed"}, {"suite": "par2", "clear": True, "tail": "ok"}]


def test_a_refusal_caused_only_by_the_frozen_memory_measure_is_superseded():
    ok, g = frozen()
    e = gate.combine(ok, g, corrected(), SUITES)
    assert not ok and e["clear"]
    assert e["frozen_memory_superseded"]["problems_removed"] == [MEM_PROBLEM]
    assert e["frozen_memory_report"]["available_gb"] == 0.24                   # kept verbatim


def test_every_other_frozen_refusal_stands():
    for kw in ({"other": ["disk: 3 GB free"]}, {"dir_problems": ["exists"]}, {"suites_clear": False},
               {"unack": ["idle_gpus"]}):
        ok, g = frozen(**kw)
        assert not gate.combine(ok, g, corrected(), SUITES)["clear"], kw


def test_a_corrected_memory_failure_refuses_even_if_the_frozen_measure_passed():
    ok, g = frozen(mem_problem=False)
    assert ok and not gate.combine(ok, g, corrected(clear=False), SUITES)["clear"]


def test_a_failing_or_skipping_suite_refuses():
    ok, g = frozen()
    bad = [SUITES[0], {"suite": "par2", "clear": False, "tail": "1 skipped"}]
    assert not gate.combine(ok, g, corrected(), bad)["clear"]


def test_the_corrected_measure_must_use_the_frozen_requirement():
    ok, g = frozen()
    c = corrected()
    c["required_bytes"] = 4_000_000_000
    assert not gate.combine(ok, g, c, SUITES)["clear"]
    g2 = copy.deepcopy(g)
    g2["preflight"]["memory"]["required_gb"] = None
    assert not gate.combine(ok, g2, corrected(), SUITES)["clear"]


def test_an_inconsistent_frozen_report_refuses():
    ok, g = frozen()
    g["preflight"]["problems"] = []                    # memory problem not in the combined list
    assert not gate.combine(ok, g, corrected(), SUITES)["clear"]


def test_par2_gate_passes_the_frozen_requirement_to_the_measurement():
    seen = {}

    def measure(*, required_bytes):
        seen["req"] = required_bytes
        return corrected()

    e = gate.par2_gate("out", frozen=lambda out: frozen(), measure=measure, suites=lambda: SUITES)
    assert seen["req"] == 8_000_000_000 and e["clear"] and e["frozen_gate"]["preflight"]["memory"]


def test_the_threshold_is_the_frozen_preregistration_value():
    from audit.implementation.ii4_par2.amendment import frozen_threshold_gb
    assert frozen_threshold_gb() == 8.0
