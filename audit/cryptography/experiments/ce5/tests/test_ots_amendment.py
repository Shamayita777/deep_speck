"""Amendment 01 (OTS verification method): the amended gate accepts a Java
OpenTimestamps run corroborated by a second endpoint's raw headers only when
everything recomputes against the real approved preregistration and proof, and
refuses endpoint disagreement, malformed proofs, wrong digests, changed proof
bytes, missing evidence and unapproved amendments. The headers are the public
Bitcoin headers of blocks 974769 and 974770; Java output and endpoint replies
are test doubles built from them."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from audit.cryptography.experiments.ce5 import design as DS
from audit.cryptography.experiments.ce5 import runner as RU

CE5 = Path(__file__).resolve().parents[1]
PREREG = CE5 / "preregistration" / "preregistration_409e7dcb_APPROVED.json"
AMENDMENT = CE5 / "preregistration" / "amendment_01_ots_verification_method.json"
PREREG_SHA = "0aa8fd61e38a52c6d207c5b1e5e426ae358f3214df684458cc29d97a8942f26d"
PROOF_SHA = "852fe259c326ed725018582f8e3fe979b94de66ec59463f012758500b711e986"
BLOCKS = {  # height: (block hash, raw 80-byte header)
    974769: ("000000000000000000013965d58e0f9e4ec9f76a4ecf05d29d451221e8aa16ae",
             "0000fc24646c6255b166fa283a7545f8440d4efb8482ce917f4301000000000000000000ab522eb1741c43b488ecca7d2be1a6cd"
             "0e9f1574d2447838a53d9cae6af1c7b8a370ed6aace50117403167a8"),
    974770: ("00000000000000000000e9d5b31e790333bc0b40af1d6bd7ec220d6d71371dbe",
             "00e04424ae16aae82112459dd205cf4e6af7c94e9e0f8ed5653901000000000000000000ea0430687d3c4498476a2ac8d09a36c5"
             "733439b7cce838b360a8faa405bb99cf4175ed6aace5011778a8270c")}
ROOTS = {974769: "b8c7f16aae9c3da5387844d274159f0ecda6e12b7dcaec88b4431c74b12e52ab",
         974770: "cf99bb05a4faa860b338e8ccb7393473c5369ad0c82a6a4798443c7d683004ea"}
TIMES = {974769: 1793945763, 974770: 1793946945}
pytestmark = pytest.mark.skipif(not (PREREG.exists() and Path(str(PREREG) + ".ots").exists()),
                                reason="approved CE5 preregistration and proof not present")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def java_output(p, roots=ROOTS, times=TIMES):
    out = f"Assuming target filename is '{p}'\nSuccess! Bitcoin block {min(roots)} attests data existed as of 2026-11-06 IST\n"
    err = ""
    for h, (bh, _) in BLOCKS.items():
        err += (f"{DS.JAVA_ENDPOINT}/block-height/{h} {bh}\n"
                f"{DS.JAVA_ENDPOINT}/block/{bh} BlockHeader{{merkleroot='{roots[h]}', blockHash='{bh}', "
                f"time='{times[h]}'}}\nLite-client verification, assuming block {bh} is valid\n")
    return out, err


def responses(blocks=BLOCKS):
    r = []
    for h, (bh, hdr) in blocks.items():
        r += [{"path": f"/block-height/{h}", "requested_utc": "t", "body": bh},
              {"path": f"/block/{bh}/header", "requested_utc": "t", "body": hdr}]
    return r


def write_amendment(path, prereg, **approval):
    a = {"schema": DS.AMENDMENT_SCHEMA, "amendment_id": DS.AMENDMENT_ID,
         "amends": {"preregistration_sha256": _sha(prereg), "design_sha256": DS.design_sha256(),
                    "proof_sha256": _sha(str(prereg) + ".ots")},
         "method": {"name": DS.JAVA_METHOD_NAME, "java_repository": DS.JAVA_OTS_REPOSITORY,
                    "java_commit": DS.JAVA_OTS_COMMIT, "java_jar_sha256": DS.JAVA_OTS_JAR_SHA256,
                    "java_runtime": DS.JAVA_RUNTIME, "java_endpoint": DS.JAVA_ENDPOINT,
                    "second_endpoint": DS.SECOND_ENDPOINT, "evidence_schema": DS.JAVA_EVIDENCE_SCHEMA},
         "registered_identities": {"code_sha256": {"x.py": "0" * 64}, "frozen_dependencies_sha256": {}}}
    a["approval"] = {"status": "APPROVED", "approved_by": "x", "approved_utc": "2099-01-01T00:00:00Z",
                     "amendment_body_sha256": DS.amendment_body_sha256(a), **approval}
    path.write_text(json.dumps(a))
    return a, _sha(path)


def write_record(p, amendment_sha, *, out=None, err=None, resp=None, exit_code=0, **over):
    o, e = java_output(p)
    proof = Path(str(p) + ".ots")
    rec = {"schema": DS.JAVA_EVIDENCE_SCHEMA, "method": DS.JAVA_METHOD_NAME, "amendment_sha256": amendment_sha,
           "java": {"source_commit": DS.JAVA_OTS_COMMIT, "jar_sha256": DS.JAVA_OTS_JAR_SHA256,
                    "runtime_version_output": f"OpenJDK Runtime Environment {DS.JAVA_RUNTIME} (build 11.0.33+10)\n",
                    "command": ["java", "-jar", "OtsCli.jar", "-v", "verify", "-f", str(p), str(proof)],
                    "exit_code": exit_code, "stdout": o if out is None else out, "stderr": e if err is None else err},
           "second_endpoint": {"base": DS.SECOND_ENDPOINT, "responses": responses() if resp is None else resp},
           "file_sha256": _sha(p), "proof_sha256": _sha(proof), **over}
    record = Path(str(proof) + DS.JAVA_EVIDENCE_SUFFIX)
    record.write_text(json.dumps(rec))
    return record


@pytest.fixture
def real(tmp_path):
    """Byte-identical copies of the approved preregistration and its upgraded proof."""
    p = tmp_path / PREREG.name
    shutil.copyfile(PREREG, p)
    shutil.copyfile(str(PREREG) + ".ots", str(p) + ".ots")
    assert _sha(p) == PREREG_SHA and _sha(str(p) + ".ots") == PROOF_SHA
    a, asha = write_amendment(tmp_path / "amendment.json", p)
    return p, a, asha


def gate(p, a, asha):
    pre = json.loads(p.read_text())
    return DS.production_authorisation(pre, DS.CONFIRM_TOKEN, prereg_path=p, amendment=a, amendment_sha256=asha)


def test_proof_commitments_are_recomputed_from_the_operation_tree():
    assert DS.ots_bitcoin_commitments(Path(str(PREREG) + ".ots")) == sorted(ROOTS.items())
    for h, (bh, hdr) in BLOCKS.items():
        b = bytes.fromhex(hdr)
        assert hashlib.sha256(hashlib.sha256(b).digest()).digest()[::-1].hex() == bh
        assert b[36:68][::-1].hex() == ROOTS[h] and int.from_bytes(b[68:72], "little") == TIMES[h]


def test_success(real):
    p, a, asha = real
    write_record(p, asha)
    assert gate(p, a, asha) == (True, "approved under amendment 01")


def test_without_the_amendment_the_registered_gate_still_refuses(real):
    p, a, asha = real
    write_record(p, asha)
    ok, why = DS.production_authorisation(json.loads(p.read_text()), DS.CONFIRM_TOKEN, prereg_path=p)
    assert not ok and "no recorded 'ots verify' run" in why


@pytest.mark.parametrize("case,expect", [
    ("height_hash", "endpoints disagree on the hash of block 974770"),
    ("header_bytes", "does not hash to the block hash"),
    ("header_time", "endpoints disagree on the time"),
    ("java_root", "Java endpoint Merkle root for block 974769 differs"),
    ("short_header", "no 80-byte header"),
    ("missing_block", "endpoints disagree on the hash of block 974769"),
    ("other_endpoint", "second endpoint is not the registered one")])
def test_endpoint_disagreement_refuses(real, case, expect):
    p, a, asha = real
    resp = responses()
    if case == "height_hash":
        resp[2]["body"] = BLOCKS[974769][0]
    elif case == "header_bytes":
        hdr = bytearray.fromhex(resp[1]["body"])
        hdr[76] ^= 1                                      # nonce: same Merkle root, different hash
        resp[1]["body"] = hdr.hex()
    elif case == "short_header":
        resp[3]["body"] = resp[3]["body"][:-2]
    elif case == "missing_block":
        resp = resp[2:]
    if case in ("java_root", "header_time"):
        o, e = java_output(p, {**ROOTS, 974769: "0" * 64}) if case == "java_root" else \
            java_output(p, times={**TIMES, 974769: TIMES[974769] + 1})
        write_record(p, asha, out=o, err=e)
    elif case == "other_endpoint":
        write_record(p, asha, second_endpoint={"base": DS.JAVA_ENDPOINT, "responses": resp})
    else:
        write_record(p, asha, resp=resp)
    ok, why = gate(p, a, asha)
    assert not ok and expect in why, why


def test_malformed_proof_refuses(real):
    p, a, asha = real
    proof = Path(str(p) + ".ots")
    good = proof.read_bytes()
    first_op = len(DS.OTS_MAGIC) + 34
    for bad in (good[:-3], good + b"\x00", good[:first_op] + b"\x0a" + good[first_op + 1:]):
        proof.write_bytes(bad)
        with pytest.raises(ValueError):
            DS.ots_bitcoin_commitments(proof)
        rec = write_record(p, asha)                      # a record made on the malformed bytes
        ok, why = DS.java_ots_verification_ok(rec, p, asha)
        assert not ok and "proof cannot be evaluated" in why, why
        assert not gate(p, a, asha)[0]                   # and the amendment does not bind them


def test_wrong_digest_refuses(real):
    p, a, asha = real
    rec = write_record(p, asha)
    p.write_bytes(p.read_bytes() + b" ")
    ok, why = DS.java_ots_verification_ok(rec, p, asha)
    assert not ok and "different file or proof bytes" in why
    rec.unlink()
    rec = write_record(p, asha)                          # even a record made on the new bytes
    ok, why = DS.java_ots_verification_ok(rec, p, asha)
    assert not ok and "does not commit to the preregistration file" in why
    assert not gate(p, a, asha)[0]


def test_real_java_wrong_file_output_refuses_although_it_exits_0(real):
    p, a, asha = real
    err = ("Expected digest " + PREREG_SHA + "\nFile does not match original!\njava.lang.Exception: File does not "
           "match original!\n\tat com.eternitywall.ots.OpenTimestamps.verify(OpenTimestamps.java:319)\n")
    write_record(p, asha, out=f"Assuming target filename is '{p}'\n", err=err, exit_code=0)
    ok, why = gate(p, a, asha)
    assert not ok and "File does not match original" in why


def test_changed_proof_bytes_refuse(real):
    p, a, asha = real
    rec = write_record(p, asha)
    proof = Path(str(p) + ".ots")
    proof.write_bytes(proof.read_bytes()[:-1] + bytes([proof.read_bytes()[-1] ^ 1]))
    ok, why = DS.java_ots_verification_ok(rec, p, asha)
    assert not ok and "different file or proof bytes" in why
    ok, why = gate(p, a, asha)
    assert not ok and "does not bind these proof bytes" in why


def test_missing_evidence_refuses(real):
    p, a, asha = real
    ok, why = gate(p, a, asha)
    assert not ok and "no recorded Java OTS verification" in why


@pytest.mark.parametrize("case,expect", [
    ("exit", "exited 1"), ("no_success", "no success line"), ("jar", "pinned Java OpenTimestamps build"),
    ("runtime", "pinned Java runtime"), ("commit", "pinned Java OpenTimestamps build"),
    ("command", "is not 'verify -f"), ("amendment_sha", "different amendment"),
    ("schema", "not an amendment-01"), ("not_json", "not JSON")])
def test_record_identity_rules(real, case, expect):
    p, a, asha = real
    o, _ = java_output(p)
    kw = {"exit": dict(exit_code=1), "no_success": dict(out=o.split("\n")[0] + "\n")}.get(case, {})
    rec = write_record(p, asha, **kw)
    r = json.loads(rec.read_text())
    if case == "jar":
        r["java"]["jar_sha256"] = "0" * 64
    elif case == "runtime":
        r["java"]["runtime_version_output"] = "openjdk 17"
    elif case == "commit":
        r["java"]["source_commit"] = "0" * 40
    elif case == "command":
        r["java"]["command"][4] = "info"
    elif case == "amendment_sha":
        r["amendment_sha256"] = "0" * 64
    elif case == "schema":
        r["schema"] = "ots-verify"
    rec.write_text("{" if case == "not_json" else json.dumps(r))
    ok, why = gate(p, a, asha)
    assert not ok and expect in why, why


@pytest.mark.parametrize("case,expect", [
    ("pending", "not approved"), ("tampered_body", "approval hash does not match"),
    ("other_prereg", "does not amend this preregistration"), ("method_name", "method differs"),
    ("one_endpoint", "method differs")])
def test_amendment_rules(real, tmp_path, case, expect):
    p, _, _ = real
    a, asha = write_amendment(tmp_path / "a.json", p, **({"status": "PENDING_APPROVAL"} if case == "pending" else {}))
    if case == "other_prereg":
        a["amends"]["preregistration_sha256"] = "0" * 64
    elif case == "method_name":
        a["method"]["name"] = "two-explorer Java verification"
    elif case in ("one_endpoint", "tampered_body"):
        a["method"]["second_endpoint"] = DS.JAVA_ENDPOINT
    if case not in ("pending", "tampered_body"):           # re-approve the changed body
        a["approval"]["amendment_body_sha256"] = DS.amendment_body_sha256(a)
    write_record(p, asha)
    ok, why = gate(p, a, asha)
    assert not ok and expect in why, why


def test_load_amendment_needs_path_and_hash(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("{}")
    assert RU.load_amendment(None, None) is None
    for args in ((f, None), (None, _sha(f)), (f, "0" * 64), (tmp_path / "none.json", _sha(f))):
        with pytest.raises(SystemExit):
            RU.load_amendment(*args)
    assert RU.load_amendment(f, _sha(f)) == {}


def test_source_integrity_uses_the_amendment_manifest(tmp_path):
    src, crypto = tmp_path / "src", tmp_path / "crypto"
    src.mkdir()
    (src / "m.py").write_text("x = 1\n")
    deps = {}
    for f in RU.FROZEN_DEPENDENCIES:
        (crypto / f).parent.mkdir(parents=True, exist_ok=True)
        (crypto / f).write_text("")
        deps[f] = _sha(crypto / f)
    old = {"registered_identities": {"code_sha256": {"m.py": "0" * 64}, "frozen_dependencies_sha256": deps}}
    new = {"registered_identities": {"code_sha256": {"m.py": _sha(src / "m.py")}, "frozen_dependencies_sha256": deps}}
    with pytest.raises(SystemExit, match="changed"):
        RU.source_integrity(old, source_root=src, crypto_root=crypto)
    assert RU.source_integrity(old, source_root=src, crypto_root=crypto, amendment=new)["decision"] == "PASS"


def test_recorder_records_raw_output_and_never_overwrites(real, tmp_path):
    p, a, asha = real
    o, e = java_output(p)
    java = tmp_path / "java"
    (tmp_path / "o.txt").write_text(o)
    (tmp_path / "e.txt").write_text(e)
    java.write_text("#!/bin/sh\nif [ \"$1\" = -version ]; then echo 'OpenJDK Runtime Environment "
                    f"{DS.JAVA_RUNTIME} (build 11.0.33+10)' >&2; exit 0; fi\n"
                    "if [ \"$3\" = -V ]; then echo 'Version: OtsCli v.1.0'; exit 0; fi\n"
                    f"cat '{tmp_path / 'o.txt'}'; cat '{tmp_path / 'e.txt'}' >&2; exit 0\n")
    java.chmod(0o755)
    jar = tmp_path / "OtsCli.jar"
    jar.write_bytes(b"jar")
    replies = {DS.SECOND_ENDPOINT + q["path"]: q["body"] for q in responses()}
    rec = RU.record_java_ots_verification(p, asha, java=str(java), jar=jar, fetch=replies.__getitem__)
    assert rec["java"]["command"][3:6] == ["-v", "verify", "-f"] and rec["java"]["exit_code"] == 0
    assert [q["path"] for q in rec["second_endpoint"]["responses"]] == [q["path"] for q in responses()]
    assert rec["file_sha256"] == PREREG_SHA and rec["proof_sha256"] == PROOF_SHA
    record = Path(str(p) + ".ots" + DS.JAVA_EVIDENCE_SUFFIX)
    ok, why = DS.java_ots_verification_ok(record, p, asha)
    assert not ok and "pinned Java OpenTimestamps build" in why     # a fake jar is not the pinned build
    with pytest.raises(SystemExit, match="already exists"):
        RU.record_java_ots_verification(p, asha, java=str(java), jar=jar, fetch=replies.__getitem__)


def test_committed_amendment_and_evidence_pass_the_amended_gate():
    record = Path(str(PREREG) + ".ots" + DS.JAVA_EVIDENCE_SUFFIX)
    if not (AMENDMENT.exists() and record.exists()):
        pytest.skip("amendment 01 and its evidence record not present")
    a = RU.load_amendment(AMENDMENT, _sha(AMENDMENT))
    pre = RU.load_approved_preregistration(PREREG, PREREG_SHA)
    assert DS.production_authorisation(pre, DS.CONFIRM_TOKEN, prereg_path=PREREG, amendment=a,
                                       amendment_sha256=_sha(AMENDMENT)) == (True, "approved under amendment 01")
    assert RU.source_integrity(pre, amendment=a)["decision"] == "PASS"
    assert "Python" not in json.loads(record.read_text())["method"]
