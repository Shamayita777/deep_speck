"""
The PAR2 memory measurement on constructed /proc and cgroup-v2 trees.

Each test writes the files a Linux host would expose into a temporary
directory and reads them through `read_sources`, so the reader and the
assessment are tested together. Values marked "as reported" reproduce the
Kaggle figures the operator reported; they were not captured from Kaggle.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from audit.implementation.ii4_par2 import memory

GiB, MiB = 2 ** 30, 2 ** 20
REQ = 8_000_000_000                                      # the frozen 8.0 GB, decimal as in preflight.py
QUIET_EVENTS = "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n"
QUIET_PSI = ("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
             "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")


def stat(inactive_file=0, file_dirty=0, file_writeback=0, file_mapped=0, **extra) -> str:
    fields = {"anon": 3 * GiB, "file": inactive_file + 2 * GiB, "kernel": 200 * MiB, "shmem": 0,
              "file_mapped": file_mapped, "file_dirty": file_dirty, "file_writeback": file_writeback,
              "inactive_anon": 0, "active_anon": 3 * GiB, "inactive_file": inactive_file,
              "active_file": 2 * GiB, "unevictable": 0, "slab_reclaimable": 100 * MiB, **extra}
    return "".join(f"{k} {v}\n" for k, v in fields.items())


def host(tmp: Path, *, mem_free=237 * MiB, mem_available=29 * GiB, cgroup="0::/\n",
         levels=None, v2_mounted=True, meminfo=None) -> dict:
    proc, cg = tmp / "proc", tmp / "cgroup"
    (proc / "self").mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot-1\n")
    if meminfo is None:
        lines = [f"MemTotal: {31 * GiB // 1024} kB", f"MemFree: {mem_free // 1024} kB"]
        if mem_available is not None:
            lines.append(f"MemAvailable: {mem_available // 1024} kB")
        meminfo = "\n".join(lines) + "\n"
    if meminfo is not False:
        (proc / "meminfo").write_text(meminfo)
    if cgroup is not None:
        (proc / "self/cgroup").write_text(cgroup)
    cg.mkdir()
    if v2_mounted:
        (cg / "cgroup.controllers").write_text("cpu io memory pids\n")
    for rel, files in (levels or {}).items():
        d = cg.joinpath(*[p for p in rel.split("/") if p])
        d.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (d / name).write_text(text)
    return memory.read_sources(str(proc), str(cg))


def level(maximum, current, *, high="max", st=None, events=QUIET_EVENTS, psi=QUIET_PSI, peak=None) -> dict:
    files = {"memory.max": f"{maximum}\n", "memory.high": f"{high}\n", "memory.current": f"{current}\n",
             "memory.stat": st if st is not None else stat(), "memory.events": events, "memory.pressure": psi}
    if peak is not None:
        files["memory.peak"] = f"{peak}\n"
    return {k: v for k, v in files.items() if v is not False}


def assess(raw):
    return memory.assess(raw, required_bytes=REQ)


# --- 1. low MemFree, high MemAvailable -------------------------------------------------------

def test_low_memfree_high_memavailable_clears_and_never_uses_memfree(tmp_path):
    r = assess(host(tmp_path, levels={"/": level("max", 2 * GiB)}))
    assert r["clear"], r["blocks"]
    assert r["usable_bytes"]["0.5"] == 29 * GiB
    assert r["meminfo"]["MemFree"] == 237 * MiB               # recorded, not used
    assert r["usable_bytes"]["0.5"] != r["meminfo"]["MemFree"]


def test_the_frozen_measure_on_the_same_numbers_would_refuse():
    """What the frozen memory_report compares: MemFree in decimal GB, rounded to 2 places."""
    assert round(237 * MiB / 1e9, 2) < 8.0


# --- 2. cgroup near its limit with reclaimable cache (Kaggle, as reported) ----------------------

def kaggle(tmp_path, **kw):
    cur, inact = int(25.88 * GiB), int(19.08 * GiB)
    st = stat(inactive_file=inact, **kw.pop("stat_kw", {}))
    st = kw.pop("st", st)
    return host(tmp_path, levels={"/": level(30 * GiB, cur, st=st, peak=cur, **kw)}), cur, inact


def test_kaggle_as_reported_clears_through_the_halved_inactive_cache(tmp_path):
    raw, cur, inact = kaggle(tmp_path)
    r = assess(raw)
    free = 30 * GiB - cur
    assert r["clear"], r["blocks"]
    assert r["usable_bytes"]["0.0"] == free
    assert r["usable_bytes"]["0.5"] == free + 0.5 * inact
    assert r["usable_bytes"]["1.0"] == free + inact
    assert r["sensitivity_clear"] == {"0.0": False, "0.5": True, "1.0": True}
    assert abs(r["usable_gb"]["0.5"] - 14.667) < 0.01
    lv = r["cgroup_levels"][0]
    assert lv["memory_peak"] == cur and lv["stat"]["inactive_file"] == inact     # all stat fields kept


def test_dirty_writeback_and_mapped_pages_are_not_credited(tmp_path):
    raw, cur, inact = kaggle(tmp_path, stat_kw={"file_dirty": 5 * GiB, "file_writeback": 1 * GiB,
                                                "file_mapped": 2 * GiB})
    r = assess(raw)
    assert r["cgroup_levels"][0]["reclaimable"] == inact - 8 * GiB
    assert r["usable_bytes"]["0.5"] == 30 * GiB - cur + 0.5 * (inact - 8 * GiB)


# --- 3. genuinely insufficient -------------------------------------------------------------------

def test_cgroup_at_its_limit_with_little_cache_blocks(tmp_path):
    raw = host(tmp_path, levels={"/": level(30 * GiB, int(29.5 * GiB), st=stat(inactive_file=1 * GiB))})
    r = assess(raw)
    assert not r["clear"]
    assert r["usable_bytes"]["0.5"] == int(0.5 * GiB) + 0.5 * GiB
    assert any("below the requirement" in b for b in r["blocks"])


def test_cache_that_is_mostly_dirty_blocks(tmp_path):
    raw, _, _ = kaggle(tmp_path, stat_kw={"file_dirty": 15 * GiB})
    assert not assess(raw)["clear"]


def test_low_host_memavailable_governs_an_unlimited_cgroup(tmp_path):
    r = assess(host(tmp_path, mem_available=4 * GiB, levels={"/": level("max", 1 * GiB)}))
    assert not r["clear"] and r["usable_bytes"]["0.5"] == 4 * GiB


def test_the_requirement_boundary_is_exact(tmp_path):
    a = host(tmp_path / "a", mem_available=None, meminfo=f"MemAvailable: {REQ // 1024} kB\n", cgroup=None)
    assert memory.assess(a, required_bytes=REQ // 1024 * 1024)["clear"]
    assert not memory.assess(a, required_bytes=REQ // 1024 * 1024 + 1)["clear"]


# --- 4. missing cgroup files and unlimited cgroups ---------------------------------------------------

def test_unlimited_cgroup_uses_the_host_term(tmp_path):
    r = assess(host(tmp_path, levels={"/": level("max", 25 * GiB)}))
    assert r["clear"] and r["cgroup_levels"] == []
    assert any("no finite memory.max" in n for n in r["notes"])


def test_no_cgroup_information_is_disclosed_and_host_governs(tmp_path):
    r = assess(host(tmp_path, cgroup=None))
    assert r["clear"] and r["cgroup_mode"] == "none"
    assert any("no cgroup is visible" in n for n in r["notes"])


def test_cgroup_v1_blocks(tmp_path):
    r = assess(host(tmp_path, cgroup="12:memory:/docker/x\n11:cpu,cpuacct:/docker/x\n0::/\n"))
    assert not r["clear"] and r["cgroup_mode"] == "v1"


def test_v2_named_but_not_mounted_blocks(tmp_path):
    r = assess(host(tmp_path, v2_mounted=False))
    assert not r["clear"] and r["cgroup_mode"] == "v2-unmounted"


def test_finite_limit_without_memory_current_blocks(tmp_path):
    lv = level(30 * GiB, 0)
    del lv["memory.current"]
    r = assess(host(tmp_path, levels={"/": lv}))
    assert not r["clear"] and any("memory.current" in b for b in r["blocks"])


def test_missing_memory_stat_gives_no_cache_credit(tmp_path):
    raw, cur, _ = kaggle(tmp_path, st=False)
    r = assess(raw)
    assert r["usable_bytes"]["0.5"] == 30 * GiB - cur and not r["clear"]
    assert "memory.stat unreadable" in r["cgroup_levels"][0]["no_credit_because"]


def test_missing_memavailable_blocks_rather_than_falling_back(tmp_path):
    r = assess(host(tmp_path, mem_available=None))
    assert not r["clear"] and any("MemAvailable" in b for b in r["blocks"])


def test_unparseable_files_block(tmp_path):
    lv = level(30 * GiB, 5 * GiB)
    lv["memory.max"] = "thirty\n"
    assert not assess(host(tmp_path / "a", levels={"/": lv}))["clear"]
    assert not assess(host(tmp_path / "b", meminfo="MemAvailable: lots\n"))["clear"]


# --- 5. pressure and OOM events ------------------------------------------------------------------------

def test_oom_kill_blocks(tmp_path):
    raw, _, _ = kaggle(tmp_path, events=QUIET_EVENTS.replace("oom_kill 0", "oom_kill 1"))
    r = assess(raw)
    assert not r["clear"] and any("oom_kill=1" in b for b in r["blocks"])


def test_oom_blocks(tmp_path):
    raw, _, _ = kaggle(tmp_path, events=QUIET_EVENTS.replace("oom 0", "oom 2", 1))
    assert not assess(raw)["clear"]


def test_limit_events_withdraw_the_cache_credit(tmp_path):
    raw, cur, _ = kaggle(tmp_path, events=QUIET_EVENTS.replace("max 0", "max 3"))
    r = assess(raw)
    assert r["usable_bytes"]["0.5"] == 30 * GiB - cur and not r["clear"]
    assert r["usable_bytes_formula_only"]["0.5"] > r["usable_bytes"]["0.5"]


def test_full_pressure_blocks(tmp_path):
    raw, _, _ = kaggle(tmp_path, psi=QUIET_PSI.replace("full avg10=0.00", "full avg10=0.40"))
    assert not assess(raw)["clear"]


def test_some_pressure_withdraws_the_cache_credit(tmp_path):
    raw, cur, _ = kaggle(tmp_path, psi=QUIET_PSI.replace("some avg10=0.00 avg60=0.00",
                                                         "some avg10=2.00 avg60=1.50"))
    r = assess(raw)
    assert r["usable_bytes"]["0.5"] == 30 * GiB - cur


# --- hierarchy and memory.high -------------------------------------------------------------------------

def test_a_tighter_parent_limit_governs(tmp_path):
    raw = host(tmp_path, cgroup="0::/kaggle/session\n",
               levels={"/kaggle/session": level(30 * GiB, 2 * GiB),
                       "/kaggle": level(6 * GiB, 2 * GiB), "/": level("max", 2 * GiB)})
    r = assess(raw)
    assert [lv["path"] for lv in r["cgroup_levels"]] == ["/kaggle/session", "/kaggle"]
    assert r["usable_bytes"]["0.5"] == 4 * GiB and not r["clear"]


def test_memory_high_below_max_is_the_limit(tmp_path):
    raw = host(tmp_path, levels={"/": level(30 * GiB, 1 * GiB, high=str(6 * GiB))})
    r = assess(raw)
    assert r["cgroup_levels"][0]["limit"] == 6 * GiB and not r["clear"]


def test_phi_must_be_one_of_the_recorded_values(tmp_path):
    with pytest.raises(ValueError):
        memory.assess(host(tmp_path), required_bytes=REQ, phi=0.7)


def test_the_gating_phi_is_the_documented_policy():
    assert memory.PHI == 0.5 and memory.SENSITIVITY_PHIS == (0.0, 0.5, 1.0)
