"""Instance-bound Gohr D1 runner (audit/dataset/experiments/d1_gohr_canonical_census.py).

Fixtures are SYNTHETIC-SCALE: a small real Gohr instance built through the
attested builder into a temporary directory. The runner's PRODUCTION_SIZES is
reduced to the fixture's only in tests that need it. No production data is
generated or read.
"""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from audit.dataset import canonical as C
from audit.dataset.experiments import d1_gohr_canonical_census as R
from audit.dataset.tools import d0_gohr_attested_build as ATT

SMALL = {"train": 6000, "validation": 3000, "test": 3000}
SMALL_SPEC = dict(C.CANONICAL_SPEC, **{f"{n}_samples": v for n, v in SMALL.items()})
LEGACY = {"audit/dataset/experiments/d1_gohr_exact_census.py":
          "136ee37266135289254d482a3a1c896c2d33a724d34895eadb689930bb3220c9",
          "audit/dataset/d1_duplicate_detection.py":
          "c62603281a746e85f0a08def299ea4cfc12cec8725e4e897e343447ebb211c14",
          "audit/dataset/evidence/d1/d1_gohr_10000000_1000000_1000000_5r.json":
          "cffc1d6416b5bd62fb0a4d19358b87ea297afb1cfdecb8925b5c35bc72c0bdca"}


@pytest.fixture(scope="module")
def instance(tmp_path_factory):
    out = tmp_path_factory.mktemp("gohr_d0_d1") / "instance"
    ATT.attested_build(out, spec=SMALL_SPEC, command=["test-fixture"], log=lambda *a, **k: None)
    return out


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(R, "PRODUCTION_SIZES", dict(SMALL))


@pytest.fixture
def blocking_engine(monkeypatch):
    calls = []

    def spy(*a, **k):
        calls.append(1)
        raise AssertionError("the D1 engine was invoked")
    monkeypatch.setattr(R, "D1_ENGINE", spy)
    return calls


def _forbidden(*a, **k):
    raise AssertionError("data generation attempted")


@pytest.fixture
def no_generation(monkeypatch):
    """Every Gohr generation entry point raises. The pinned `speck` module itself is left
    untouched: patching it is detected and refused by the runner (see the next test)."""
    from audit.dataset.adapters import gohr
    monkeypatch.setattr(C, "build", _forbidden)
    monkeypatch.setattr(C, "_resolve_generator", _forbidden)
    monkeypatch.setattr(gohr.GohrAdapter, "generate_partition", _forbidden)


def test_a_patched_generator_function_is_detected_and_refused(instance, tmp_path, small, blocking_engine,
                                                              monkeypatch) -> None:
    import speck
    monkeypatch.setattr(speck, "make_train_data", _forbidden)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="not compiled from the repository-root speck.py"):
        R.main(["--instance-dir", str(instance), "--out-dir", str(out)])
    _none_written(out)
    assert not blocking_engine


def _copy(instance, tmp_path) -> Path:
    d = tmp_path / "copy"
    shutil.copytree(instance, d)
    return d


def _none_written(out: Path) -> None:
    assert not out.exists() and not list(out.parent.glob(".d1_gohr_staging_*"))


def test_runner_uses_the_generic_engine_and_not_the_legacy_runner() -> None:
    tree = ast.parse(Path(R.__file__).read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any("d1_gohr_exact_census" in m for m in mods)
    assert R.D1_ENGINE is R.D1.audit_duplicates
    assert R.EXPECTED_SPEC == C.CANONICAL_SPEC and R.PRODUCTION_SIZES == {"train": 10_000_000,
                                                                         "validation": 1_000_000, "test": 1_000_000}


def test_census_end_to_end_binds_instance_without_regenerating(instance, tmp_path, small, no_generation) -> None:
    out = tmp_path / "out"
    assert R.main(["--instance-dir", str(instance), "--out-dir", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted([R.CERTIFICATE_NAME, R.RUN_MANIFEST_NAME])
    cert = json.loads((out / R.CERTIFICATE_NAME).read_text())
    man = C.load_manifest(instance)
    ci = cert["canonical_instance"]
    assert ci["canonical_instance_id"] == man["canonical_instance_id"]
    assert ci["manifest_sha256"] == hashlib.sha256((instance / C.MANIFEST_NAME).read_bytes()).hexdigest()
    for n in C.PARTITIONS:
        assert ci["partition_hashes"][n]["features_sha256"] == man["partitions"][n]["features_sha256"]
    assert cert["build_attestation"]["sha256"] == hashlib.sha256((instance / ATT.ATTESTATION_NAME).read_bytes()).hexdigest()
    assert cert["generator_pin"] == ATT.GENERATOR_PIN
    assert cert["verdict"]["outcome"] in {"PASS", "FAIL"}
    assert any("does NOT establish i.i.d. sampling" in s for s in cert["scope_limitations"])
    rm = json.loads((out / R.RUN_MANIFEST_NAME).read_text())
    assert rm["files"][R.CERTIFICATE_NAME] == hashlib.sha256((out / R.CERTIFICATE_NAME).read_bytes()).hexdigest()
    assert rm["d0_manifest_sha256"] == ci["manifest_sha256"] and rm["canonical_instance_id"] == ci["canonical_instance_id"]
    assert str(Path.home()) not in json.dumps(cert)
    assert not list(tmp_path.glob(".d1_gohr_staging_*"))


def test_existing_output_directory_is_refused(instance, tmp_path, small, blocking_engine) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep").write_text("x")
    with pytest.raises(SystemExit, match="exists"):
        R.main(["--instance-dir", str(instance), "--out-dir", str(out)])
    assert [p.name for p in out.iterdir()] == ["keep"] and not blocking_engine


def test_production_sizes_are_required(instance, tmp_path, blocking_engine) -> None:
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="not the frozen Gohr D0 configuration"):
        R.main(["--instance-dir", str(instance), "--out-dir", str(out)])
    _none_written(out)


def _reseal(d: Path, mutate) -> None:
    man = json.loads((d / C.MANIFEST_NAME).read_text())
    mutate(man)
    man["canonical_instance_id"] = C.compute_instance_id(man["partitions"])
    man["specification_sha256"] = C.compute_specification_digest(man["specification"])
    (d / C.MANIFEST_NAME).write_text(json.dumps(man, indent=2, sort_keys=True))


@pytest.mark.parametrize("damage, message", [
    (lambda d: _reseal(d, lambda m: m["specification"].update(num_rounds=6)), "num_rounds"),
    (lambda d: (d / ATT.ATTESTATION_NAME).unlink(), "no build attestation"),
    (lambda d: (d / ATT.ATTESTATION_NAME).write_text(
        json.dumps(dict(json.loads((d / ATT.ATTESTATION_NAME).read_text()), manifest_sha256="0" * 64))),
     "manifest SHA-256"),
    (lambda d: (d / "test.npz").write_bytes(bytes(b ^ 0xFF if i == 200 else b for i, b in
                                                  enumerate((d / "test.npz").read_bytes()))), "did not verify"),
])
def test_mismatched_instances_are_refused(instance, tmp_path, small, blocking_engine, damage, message) -> None:
    d = _copy(instance, tmp_path)
    damage(d)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=message):
        R.main(["--instance-dir", str(d), "--out-dir", str(out)])
    _none_written(out)
    assert not blocking_engine


def test_wrong_generator_pin_is_refused(instance, tmp_path, small, blocking_engine, monkeypatch) -> None:
    monkeypatch.setitem(ATT.GENERATOR_PIN, "sha256", "0" * 64)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="not the pinned source"):
        R.main(["--instance-dir", str(instance), "--out-dir", str(out)])
    _none_written(out)


@pytest.mark.parametrize("rel", sorted(LEGACY))
def test_legacy_d1_runner_engine_and_evidence_unchanged(rel) -> None:
    p = R.REPO_ROOT / rel
    if not p.is_file():
        pytest.skip(f"{rel} not present (SKIPPED IS NOT PASSED)")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == LEGACY[rel]
