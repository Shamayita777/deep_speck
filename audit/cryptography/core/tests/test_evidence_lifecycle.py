"""core.evidence: commit protocol, fault injection at every boundary, journal/record consistency."""

from __future__ import annotations

import json

import pytest

from audit.cryptography.core import evidence as EV

MACHINE = EV.StateMachine(initial="NOT_STARTED",
                          transitions={"NOT_STARTED": ("DATA_GENERATED", "FAILED"),
                                       "DATA_GENERATED": ("VALID", "FAILED")},
                          terminal=("VALID", "FAILED"))
PAYLOAD = b"x" * 4096


@pytest.fixture(autouse=True)
def _disarm():
    EV.disarm_faults()
    yield
    EV.disarm_faults()


def store(tmp_path) -> EV.EvidenceStore:
    s = EV.EvidenceStore(tmp_path / "out", "EXP-TEST", MACHINE)
    s.root.mkdir(parents=True, exist_ok=True)
    return s


def generate(s: EV.EvidenceStore, rid="run_00", payload=PAYLOAD) -> dict:
    """One full step: recover, commit data.bin, transition."""
    s.recover_run(rid, ("data.bin",))
    info = s.commit_artifact(rid, "data.bin", s.writer_with_fault(lambda p: p.write_bytes(payload)))
    return s.transition(rid, "DATA_GENERATED", **info)


def events(s, name):
    return [e for e in s.journal.entries() if e["event"] == name]


# ------------------------------------------------------- fault injection: data commit


@pytest.mark.parametrize("point", ["before_write", "during_write", "after_write", "after_hash", "after_promote"])
def test_interruption_before_commit_is_recovered_deterministically(tmp_path, point):
    s = store(tmp_path)
    EV.arm_fault(point)
    with pytest.raises(EV.InjectedFault):
        generate(s)
    assert s.record("run_00")["state"] == "NOT_STARTED"          # nothing committed
    rec = generate(s)                                           # resume
    assert rec["state"] == "DATA_GENERATED"
    assert (s.run_dir("run_00") / "data.bin").read_bytes() == PAYLOAD
    q = events(s, "artifact_quarantined")
    expect = {"before_write": 0, "during_write": 1, "after_write": 1, "after_hash": 1, "after_promote": 1}[point]
    assert len(q) == expect
    for e in q:                                                 # retained, hashed, never deleted
        assert (s.root / e["destination"]).is_file()
        assert EV.sha256_file(s.root / e["destination"]) == e["sha256"]
    if point == "after_promote":
        assert q[0]["reason"].startswith("complete artifact whose transition was never committed")
    if point == "during_write":
        assert q[0]["reason"].startswith("partial")
    assert not list((s.run_dir("run_00") / EV.STAGING).iterdir())
    assert not EV.chain_problems(s.journal.raw_lines(), "EXP-TEST")


def test_interruption_after_commit_before_record_cache(tmp_path):
    s = store(tmp_path)
    EV.arm_fault("after_commit")
    with pytest.raises(EV.InjectedFault):
        generate(s)
    cache = s.run_dir("run_00") / EV.RECORD
    assert not cache.exists()                                  # journal committed, cache never written
    rec = s.record("run_00")
    assert rec["state"] == "DATA_GENERATED" and cache.exists()
    assert events(s, "record_cache_rematerialised")[0]["reason"] == "record cache absent"


def test_stale_record_cache_is_rematerialised_and_journalled(tmp_path):
    s = store(tmp_path)
    generate(s)
    cache = s.run_dir("run_00") / EV.RECORD
    stale = cache.read_text()
    EV.arm_fault("after_commit")
    with pytest.raises(EV.InjectedFault):
        s.transition("run_00", "VALID")
    assert cache.read_text() == stale
    assert s.record("run_00")["state"] == "VALID"
    assert events(s, "record_cache_rematerialised")[-1]["reason"].startswith("stale record cache")


def test_interruption_during_quarantine_is_recovered(tmp_path):
    s = store(tmp_path)
    EV.arm_fault("after_promote")
    with pytest.raises(EV.InjectedFault):
        generate(s)
    EV.arm_fault("during_quarantine")
    with pytest.raises(EV.InjectedFault):
        generate(s)                                             # crashes after the quarantine intent
    assert (s.run_dir("run_00") / "data.bin").exists()          # not yet moved
    rec = generate(s)
    assert rec["state"] == "DATA_GENERATED"
    q = events(s, "artifact_quarantined")
    assert len(q) == 2 and not (s.root / q[0]["destination"]).exists() and (s.root / q[1]["destination"]).exists()


def test_torn_journal_tail_is_set_aside_not_deleted(tmp_path):
    s = store(tmp_path)
    generate(s)
    with open(s.journal.path, "ab") as f:
        f.write(b'{"partial": tru')
    s2 = EV.EvidenceStore(s.root, "EXP-TEST", MACHINE)
    notes = s2.open()
    assert notes and "torn" in notes[0]
    torn = events(s2, "journal_torn_tail_recovered")[0]
    assert (s2.root / torn["file"]).read_bytes() == b'{"partial": tru'
    assert s2.record("run_00")["state"] == "DATA_GENERATED"


def test_committed_artifact_damage_is_irrecoverable(tmp_path):
    s = store(tmp_path)
    generate(s)
    (s.run_dir("run_00") / "data.bin").write_bytes(b"tampered")
    with pytest.raises(EV.InconsistentEvidence, match="SHA-256"):
        s.recover_run("run_00", ("data.bin",))
    (s.run_dir("run_00") / "data.bin").unlink()
    with pytest.raises(EV.InconsistentEvidence, match="missing"):
        s.recover_run("run_00", ("data.bin",))


def test_promotion_never_overwrites(tmp_path):
    s = store(tmp_path)
    s.run_dir("run_00").mkdir(parents=True)
    (s.run_dir("run_00") / "data.bin").write_bytes(b"orphan")
    with pytest.raises(EV.EvidenceError, match="recover_run"):
        s.commit_artifact("run_00", "data.bin", lambda p: p.write_bytes(PAYLOAD))
    assert (s.run_dir("run_00") / "data.bin").read_bytes() == b"orphan"


# ----------------------------------------------- journal / record consistency


def test_terminal_state_cannot_be_left(tmp_path):
    s = store(tmp_path)
    generate(s)
    s.transition("run_00", "VALID")
    with pytest.raises(EV.EvidenceError, match="terminal"):
        s.transition("run_00", "FAILED")


def test_illegal_transition_refused(tmp_path):
    s = store(tmp_path)
    with pytest.raises(EV.EvidenceError, match="not permitted"):
        s.transition("run_00", "VALID")


def test_record_updated_without_journal_event_is_rejected(tmp_path):
    s = store(tmp_path)
    generate(s)
    cache = s.run_dir("run_00") / EV.RECORD
    rec = json.loads(cache.read_text())
    rec.update(state="VALID", journal_seq=rec["journal_seq"] + 5)
    cache.write_text(json.dumps(rec))
    with pytest.raises(EV.InconsistentEvidence):
        s.record("run_00")


def test_modified_record_state_is_rejected(tmp_path):
    s = store(tmp_path)
    generate(s)
    cache = s.run_dir("run_00") / EV.RECORD
    rec = json.loads(cache.read_text())
    rec["data_sha256"] = "0" * 64
    cache.write_text(json.dumps(rec))
    with pytest.raises(EV.InconsistentEvidence):
        s.record("run_00")


def _rewrite(path, lines):
    path.write_bytes(b"".join(l + b"\n" for l in lines))


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "reordered", "payload", "prev_hash", "seq",
                                      "experiment"])
def test_journal_mutations_are_detected(tmp_path, mutation):
    s = store(tmp_path)
    generate(s)
    s.transition("run_00", "VALID")
    lines = s.journal.raw_lines()
    if mutation == "duplicate":
        lines = lines + [lines[-1]]
    elif mutation == "missing":
        lines = lines[:1] + lines[2:]
    elif mutation == "reordered":
        lines[1], lines[2] = lines[2], lines[1]
    elif mutation == "payload":
        e = json.loads(lines[-1]); e["to"] = "FAILED"; lines[-1] = EV.canonical(e)
    elif mutation == "prev_hash":
        e = json.loads(lines[-1]); e["prev_sha256"] = "f" * 64; lines[-1] = EV.canonical(e)
    elif mutation == "seq":
        e = json.loads(lines[-1]); e["seq"] += 7; lines[-1] = EV.canonical(e)
    elif mutation == "experiment":
        e = json.loads(lines[-1]); e["experiment_id"] = "OTHER"; lines[-1] = EV.canonical(e)
    _rewrite(s.journal.path, lines)
    s2 = EV.EvidenceStore(s.root, "EXP-TEST", MACHINE)
    with pytest.raises(EV.InconsistentEvidence):
        s2.open()
        s2.record("run_00")


def test_journal_event_without_record_is_rematerialised(tmp_path):
    """Journal updated, record absent: the record is a cache and is rebuilt (and that is journalled)."""
    s = store(tmp_path)
    generate(s)
    (s.run_dir("run_00") / EV.RECORD).unlink()
    assert s.record("run_00")["state"] == "DATA_GENERATED"
    assert events(s, "record_cache_rematerialised")


def test_replay_rejects_transition_from_wrong_state(tmp_path):
    s = store(tmp_path)
    s.journal.append(EV.TRANSITION, run_id="run_00", **{"from": "DATA_GENERATED", "to": "VALID"}, fields={})
    with pytest.raises(EV.InconsistentEvidence, match="starts from"):
        s.record("run_00")
