"""D0 build attestation for the canonical Gohr instance (audit/dataset/tools/d0_gohr_attested_build.py).

Every instance here is a SMALL, clearly synthetic-scale build of the real Gohr
generator into a temporary directory (sizes far below the canonical
specification), made through the unmodified canonical.build(). No production
instance is generated or read.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from audit.dataset import canonical as C
from audit.dataset.tools import d0_gohr_attested_build as ATT

#: Test-only reduced sizes; production uses canonical.CANONICAL_SPEC unchanged.
SMALL_SPEC = dict(C.CANONICAL_SPEC, train_samples=6000, validation_samples=3000, test_samples=3000)


def _quiet(*a, **k):
    return None


@pytest.fixture(scope="module")
def attested(tmp_path_factory):
    out = tmp_path_factory.mktemp("d0g") / "instance"
    att = ATT.attested_build(out, spec=SMALL_SPEC, command=["test-fixture"], log=_quiet)
    return out, att


def _copy(attested, tmp_path) -> Path:
    dst = tmp_path / "copy"
    shutil.copytree(attested[0], dst)
    return dst


def test_attested_build_binds_instance_generator_and_sources(attested) -> None:
    out, att = attested
    man = C.load_manifest(out)
    assert att["status"] == ATT.ATTESTED and att["schema_version"] == ATT.ATTESTATION_SCHEMA_VERSION
    assert att["canonical_instance_id"] == man["canonical_instance_id"]
    assert att["manifest_sha256"] == ATT._sha256_file(out / C.MANIFEST_NAME)
    for n in C.PARTITIONS:
        for k in ("file_sha256", "features_sha256", "labels_sha256", "samples"):
            assert att["partitions"][n][k] == man["partitions"][n][k]
    assert att["generator_pin"] == ATT.GENERATOR_PIN and att["build_code_pins"] == ATT.BUILD_CODE_PINS
    imp = att["imported_generator"]
    assert imp["resolved_file_is_pinned_path"] and imp["code_object_file_is_pinned_path"]
    assert imp["file_sha256"] == ATT.GENERATOR_PIN["sha256"] and imp["relative_path"] == "speck.py"
    assert att["sources_before"]["sources"] == att["sources_after"]["sources"]
    assert att["generator_path_recorded_by_canonical"] == man["generator_path"]
    assert att["specification"] == SMALL_SPEC and att["specification_is_canonical"] is False
    assert att["canonical_verify"] == {"verified": True}
    assert att["build_log"] and "not prove" not in att["proves"]
    assert "i.i.d. sampling or the quality of os.urandom" in att["does_not_prove"]
    assert "/Users/" not in json.dumps(att), "the attestation records no absolute home path"


def test_new_manifest_records_no_absolute_repository_path(attested) -> None:
    """canonical.build() writes str(repo_root) into the manifest; the wrapper passes a relative root."""
    man = C.load_manifest(attested[0])
    assert not Path(man["git"]["repo_root"]).is_absolute()
    assert str(Path.home()) not in json.dumps(man)


def test_attestation_verifies_and_instance_verifies(attested) -> None:
    out, _ = attested
    res = ATT.verify_build_attestation(out, expected_spec=SMALL_SPEC)
    assert res["sha256"] == ATT._sha256_file(out / ATT.ATTESTATION_NAME)
    assert C.verify(out, log=_quiet)["verified"] is True


def test_expected_spec_mismatch_is_refused(attested) -> None:
    with pytest.raises(ATT.AttestationRejected, match="not the expected specification"):
        ATT.verify_build_attestation(attested[0], expected_spec=dict(C.CANONICAL_SPEC))


def test_existing_output_directory_is_refused_and_untouched(tmp_path) -> None:
    out = tmp_path / "exists"
    out.mkdir()
    (out / "keep.txt").write_text("user file")
    with pytest.raises(SystemExit, match="exists"):
        ATT.attested_build(out, spec=SMALL_SPEC, log=_quiet)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_wrong_generator_pin_refuses_before_generation(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(ATT.GENERATOR_PIN, "sha256", "0" * 64)
    out = tmp_path / "never"
    with pytest.raises(SystemExit, match="before generation"):
        ATT.attested_build(out, spec=SMALL_SPEC, log=_quiet)
    assert not out.exists()


def test_wrong_build_code_pin_refuses_before_generation(tmp_path, monkeypatch) -> None:
    pins = copy.deepcopy(ATT.BUILD_CODE_PINS)
    pins["audit/dataset/canonical.py"]["sha256"] = "f" * 64
    monkeypatch.setattr(ATT, "BUILD_CODE_PINS", pins)
    out = tmp_path / "never"
    with pytest.raises(SystemExit, match="canonical.py differs from its pin"):
        ATT.attested_build(out, spec=SMALL_SPEC, log=_quiet)
    assert not out.exists()


def test_generator_imported_from_elsewhere_refuses(tmp_path, monkeypatch) -> None:
    real = ATT.generator_module_record

    def elsewhere(repo):
        return dict(real(repo), resolved_file_is_pinned_path=False)
    monkeypatch.setattr(ATT, "generator_module_record", elsewhere)
    out = tmp_path / "never"
    with pytest.raises(SystemExit, match="not the repository-root speck.py"):
        ATT.attested_build(out, spec=SMALL_SPEC, log=_quiet)
    assert not out.exists()


def test_non_gohr_target_is_refused(tmp_path) -> None:
    with pytest.raises(SystemExit, match="Gohr target only"):
        ATT.attested_build(tmp_path / "x", spec=dict(SMALL_SPEC, target="autond"), log=_quiet)


def test_missing_attestation_is_refused(attested, tmp_path) -> None:
    d = _copy(attested, tmp_path)
    (d / ATT.ATTESTATION_NAME).unlink()
    with pytest.raises(ATT.AttestationRejected, match="no build attestation"):
        ATT.verify_build_attestation(d)


@pytest.mark.parametrize("mutate, message", [
    (lambda a: a.update(manifest_sha256="0" * 64), "manifest SHA-256"),
    (lambda a: a.update(canonical_instance_id="0" * 64), "instance id"),
    (lambda a: a["partitions"]["test"].update(features_sha256="0" * 64), "partition test"),
    (lambda a: a["generator_pin"].update(sha256="0" * 64), "generator pin"),
    (lambda a: a["imported_generator"].update(code_object_file_is_pinned_path=False), "imported generator"),
    (lambda a: a["sources_after"].update(git_head="0" * 40), "changed during"),
    (lambda a: a["sources_before"]["sources"][ATT.THIS_FILE].update(sha256="0" * 64), "changed during"),
    (lambda a: a.update(status="FAILED"), "ATTESTED"),
    (lambda a: a.update(generator_path_recorded_by_canonical="other.generator"), "generator path"),
])
def test_mismatched_attestation_is_refused(attested, tmp_path, mutate, message) -> None:
    d = _copy(attested, tmp_path)
    att = json.loads((d / ATT.ATTESTATION_NAME).read_text())
    mutate(att)
    (d / ATT.ATTESTATION_NAME).write_text(json.dumps(att))
    with pytest.raises(ATT.AttestationRejected, match=message):
        ATT.verify_build_attestation(d)


def test_attestation_from_a_different_tool_version_is_refused(attested, tmp_path) -> None:
    d = _copy(attested, tmp_path)
    att = json.loads((d / ATT.ATTESTATION_NAME).read_text())
    for side in ("sources_before", "sources_after"):
        att[side]["sources"][ATT.THIS_FILE]["sha256"] = "1" * 64
    (d / ATT.ATTESTATION_NAME).write_text(json.dumps(att))
    with pytest.raises(ATT.AttestationRejected, match="different version of the attestation tool"):
        ATT.verify_build_attestation(d)


def test_manifest_edited_after_attestation_is_refused(attested, tmp_path) -> None:
    d = _copy(attested, tmp_path)
    man = json.loads((d / C.MANIFEST_NAME).read_text())
    man["created_utc"] = "edited"
    (d / C.MANIFEST_NAME).write_text(json.dumps(man, indent=2, sort_keys=True))
    with pytest.raises(ATT.AttestationRejected, match="manifest SHA-256"):
        ATT.verify_build_attestation(d)


def test_attestation_is_never_overwritten(tmp_path) -> None:
    p = tmp_path / "a.json"
    ATT._write_exclusive(p, b"first")
    with pytest.raises(FileExistsError):
        ATT._write_exclusive(p, b"second")
    assert p.read_bytes() == b"first" and [x.name for x in tmp_path.iterdir()] == ["a.json"]


def test_cli_verify_reports_refusal_without_writing(tmp_path, capsys) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    (d / C.MANIFEST_NAME).write_text("{}")
    assert ATT.main(["verify", "--out-dir", str(d)]) == 1
    assert "REFUSED" in capsys.readouterr().out and sorted(p.name for p in d.iterdir()) == [C.MANIFEST_NAME]
