"""The memory probe's PASS rule, applied to recorded observations."""
from __future__ import annotations

from audit.implementation.ii4_par2 import memprobe

REQ = 8_000_000_000


def m(oom=0, kill=0, peak=None):
    return {"boot_id": "b", "cgroup_levels": [{"path": "/", "memory_peak": peak,
                                               "events": {"oom": oom, "oom_kill": kill, "max": 0},
                                               "pressure": {"some": {"avg60": 0.0}}}]}


def child(**kw):
    c = {"returncode": 0, "ok": True, "arms_completed": ["reference", "independent"], "scale": 1.0,
         "final_VmHWM": 3_500_000_000}
    c.update(kw)
    return c


def summarise(children, before=None, after=None):
    return memprobe.summarise(before or m(peak=10), after or m(peak=20), children,
                              {"peak_rss": {}, "peak_current": None, "n": 3}, REQ)


def test_two_full_size_children_without_oom_pass_and_their_peaks_are_summed():
    e = summarise({"0": child(), "1": child(final_VmHWM=3_600_000_000)})
    assert e["status"] == "PASS" and e["full_size"]
    assert e["sum_child_vmhwm_bytes"] == 7_100_000_000
    assert e["against_frozen_estimates"]["two_processes_estimate_gb"] == 8.0
    assert e["against_frozen_estimates"]["requirement_changed"] is False
    assert e["memory_peak_before"] == {"/": 10} and e["memory_peak_after"] == {"/": 20}


def test_an_oom_during_the_probe_fails_it():
    assert summarise({"0": child(), "1": child()}, after=m(kill=1))["status"] == "FAIL"
    assert summarise({"0": child(), "1": child()}, after=m(oom=1))["status"] == "FAIL"
    assert summarise({"0": child(), "1": child()}, before=m(kill=1), after=m(kill=1))["status"] == "PASS"


def test_a_failed_partial_or_reduced_child_fails_the_probe():
    for bad in (child(returncode=1, ok=False), child(arms_completed=["reference"]), child(scale=0.05)):
        assert summarise({"0": child(), "1": bad})["status"] == "FAIL"
    assert summarise({"0": child()})["status"] == "FAIL"


def test_probe_seeds_are_outside_the_experiment():
    assert memprobe.PROBE_SEED_BASE == 20261106
