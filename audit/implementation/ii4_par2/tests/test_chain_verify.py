"""The hash-chained gate record, the PAR2 verifier's checks, and the PAR2 record's binding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from audit.implementation.ii4_par2 import amendment, chain, verify_par2

REC = "f" * 64


def entries(tmp: Path, kinds) -> Path:
    p = tmp / "run.par2_gate.jsonl"
    gate_sha = None
    for k in kinds:
        e = {"par2_record_sha256": REC, "kind": k}
        if k == "gate":
            e.update({"clear": True, "corrected_memory": {"clear": True, "boot_id": "b1"}})
        elif k == "gate_refused":
            e.update({"kind": "gate", "clear": False, "corrected_memory": {"clear": False, "boot_id": "b1"}})
        elif k == "probe":
            e.update({"kind": "memprobe", "status": "PASS", "full_size": True, "boot_id": "b1"})
        elif k == "probe_fail":
            e.update({"kind": "memprobe", "status": "FAIL", "full_size": True, "boot_id": "b1"})
        elif k == "start":
            e.update({"kind": "execute_start", "gate_entry_sha256": gate_sha, "boot_id": "b1"})
        w = chain.append(p, e)
        if w["kind"] == "gate":
            gate_sha = w["entry_sha256"]
    return p


def test_append_links_and_hashes_every_entry(tmp_path):
    p = entries(tmp_path, ["gate", "probe", "start"])
    es = chain.read_chain(p)
    assert [e["seq"] for e in es] == [0, 1, 2] and es[0]["prev_sha256"] == chain.GENESIS
    assert es[2]["prev_sha256"] == es[1]["entry_sha256"]
    assert chain.check_chain(es) == [] and verify_par2.check_entries(es, REC) == []


def test_any_edit_breaks_the_chain_and_appending_to_it_is_refused(tmp_path):
    p = entries(tmp_path, ["gate", "probe"])
    lines = p.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["clear"] = False
    p.write_text(json.dumps(tampered) + "\n" + lines[1] + "\n")
    es = chain.read_chain(p)
    assert chain.check_chain(es) and verify_par2.check_entries(es, REC)
    with pytest.raises(RuntimeError):
        chain.append(p, {"kind": "gate"})


def test_execution_needs_a_clear_gate_and_a_passing_probe_in_the_same_boot(tmp_path):
    assert verify_par2.check_entries(chain.read_chain(entries(tmp_path / "a", ["gate", "probe", "start"])), REC) == []
    for name, kinds in (("b", ["gate", "start"]), ("c", ["probe_fail", "gate", "start"]),
                        ("d", ["probe", "gate_refused", "start"])):
        assert verify_par2.check_entries(chain.read_chain(entries(tmp_path / name, kinds)), REC), name


def test_entries_must_name_the_record_on_disk(tmp_path):
    es = chain.read_chain(entries(tmp_path, ["gate"]))
    assert verify_par2.check_entries(es, "0" * 64)


def test_each_session_needs_its_own_earlier_execute_start(tmp_path):
    es = chain.read_chain(entries(tmp_path, ["probe", "gate", "start"]))
    t = es[2]["recorded_at_utc"]
    later = "2999-01-01T00:00:00+00:00"
    assert verify_par2.check_sessions(es, {"parallel_sessions": [{"session_id": "s1", "started_at_utc": later}]}) == []
    two = {"parallel_sessions": [{"session_id": "s1", "started_at_utc": later},
                                 {"session_id": "s2", "started_at_utc": later}]}
    assert verify_par2.check_sessions(es, two)
    early = {"parallel_sessions": [{"session_id": "s0", "started_at_utc": "2000-01-01T00:00:00+00:00"}]}
    assert verify_par2.check_sessions(es, early) and t


def test_verify_end_to_end_on_a_constructed_repository(tmp_path):
    repo, run = tmp_path / "repo", tmp_path / "repo" / "runs" / "r"
    run.mkdir(parents=True)
    src = repo / "audit/implementation/ii4_par2/memory.py"
    src.parent.mkdir(parents=True)
    src.write_text("x = 1\n")
    rec = {"par2_sources": {"par2_memory": {"path": str(src.relative_to(repo)),
                                            "sha256": hashlib.sha256(b"x = 1\n").hexdigest()}}, "amends": []}
    rp = repo / verify_par2.PAR2_RECORD
    rp.parent.mkdir(parents=True)
    rp.write_text(json.dumps(rec))
    sha = hashlib.sha256(rp.read_bytes()).hexdigest()
    p = chain.chain_path(run)
    g = chain.append(p, {"kind": "gate", "clear": True, "par2_record_sha256": sha,
                         "corrected_memory": {"clear": True, "boot_id": "b"}})
    chain.append(p, {"kind": "memprobe", "status": "PASS", "full_size": True, "boot_id": "b",
                     "par2_record_sha256": sha})
    chain.append(p, {"kind": "execute_start", "gate_entry_sha256": g["entry_sha256"], "boot_id": "b",
                     "par2_record_sha256": sha})
    (run / "state.json").write_text(json.dumps({"parallel_sessions": [
        {"session_id": "s", "started_at_utc": "2999-01-01T00:00:00+00:00"}]}))
    ok = lambda *a: {"status": "VERIFIED", "problems": []}
    assert verify_par2.verify(run, repo_root=repo, frozen_verifier=ok)["status"] == "VERIFIED"
    bad = lambda *a: {"status": "NOT_VERIFIED", "problems": ["x"]}
    assert verify_par2.verify(run, repo_root=repo, frozen_verifier=bad)["status"] == "NOT_VERIFIED"
    src.write_text("x = 2\n")
    assert verify_par2.verify(run, repo_root=repo, frozen_verifier=ok)["status"] == "NOT_VERIFIED"


def test_the_par2_verifier_imports_nothing_from_the_project():
    text = Path(verify_par2.__file__).read_text()
    assert "from audit" not in text and "import audit" not in text


def test_the_par2_record_binds_the_sources_and_the_amended_records():
    problems, rec = amendment.check_record()
    assert problems == [], problems
    assert rec["threshold"]["recommended_free_host_gb"] == 8.0 and rec["threshold"]["changed"] is False
    assert set(rec["par2_sources"]) == set(amendment.PAR2_SOURCES)
    assert {a["record"] for a in rec["amends"]} == {str(amendment.ORIGINAL_FROZEN), str(amendment.PAR1_FROZEN)}


def test_par1_files_are_not_par2_sources():
    assert not any("ii4_parallel" in p or p.endswith(("preflight.py", "run_ii4_impl.py"))
                   for p in amendment.PAR2_SOURCES.values())
