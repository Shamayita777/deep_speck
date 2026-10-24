"""Focused fail-closed regression tests for the D3 Gohr D1 evidence gate.

Run in the repository environment with:
  python -m pytest -q audit/dataset/tests/test_d3_gohr_d1_gate.py

The tests synthesize small evidence and source files in a temporary directory;
no production dataset or sealed evidence is read or modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from audit.dataset.experiments import d3_gohr_confirmatory as D3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


@pytest.fixture

def evidence_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Install minimal, synthetic repo modules and one internally consistent D1 evidence set."""
    repo = (tmp_path / "repo").resolve()
    monkeypatch.setattr(D3, "REPO_ROOT", repo)

    relpaths = {
        "runner": "audit/dataset/experiments/d1_gohr_canonical_census.py",
        "engine": "audit/dataset/d1_duplicate_detection.py",
        "canonical": "audit/dataset/canonical.py",
        "attestation_tool": "audit/dataset/tools/d0_gohr_attested_build.py",
        "generator": "speck.py",
    }
    source_files = {}
    pins = {}
    for name, rel in relpaths.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic frozen source for {name}\n")
        source_files[name] = path
        pins[name] = {"path": rel, "sha256": _sha256(path)}
    monkeypatch.setattr(D3, "D1_CODE_IDENTITY_PINS", pins)

    # Stub only the module metadata the gate must bind to the frozen files.
    d1_engine = SimpleNamespace(__file__=str(source_files["engine"]))
    d1_runner = SimpleNamespace(
        __file__=str(source_files["runner"]),
        D1=d1_engine,
        CERTIFICATE_NAME="d1_gohr_certificate.json",
        RUN_MANIFEST_NAME="d1_gohr_run_manifest.json",
        RUNNER_SCHEMA_VERSION="d1-gohr-canonical-certificate-1",
    )
    canonical = SimpleNamespace(
        __file__=str(source_files["canonical"]),
        MANIFEST_NAME="d0_canonical_manifest.json",
        load_manifest=lambda directory: json.loads(
            (Path(directory) / "d0_canonical_manifest.json").read_text()
        ),
    )
    attestation = SimpleNamespace(
        __file__=str(source_files["attestation_tool"]),
        ATTESTATION_NAME="d0_gohr_build_attestation.json",
    )
    monkeypatch.setattr(D3, "D1R", d1_runner)
    monkeypatch.setattr(D3, "C", canonical)
    monkeypatch.setattr(D3, "ATT", attestation)

    instance_dir = repo / "audit/dataset/evidence/d0_fixture"
    instance_dir.mkdir(parents=True)
    d0_manifest = {
        "canonical_instance_id": "fixture-instance-001",
        "specification_sha256": "fixture-spec-sha256",
    }
    _write_json(instance_dir / canonical.MANIFEST_NAME, d0_manifest)
    _write_json(instance_dir / attestation.ATTESTATION_NAME, {"fixture": "attestation"})
    d0_manifest_sha = _sha256(instance_dir / canonical.MANIFEST_NAME)
    attestation_sha = _sha256(instance_dir / attestation.ATTESTATION_NAME)

    d1_dir = repo / "audit/dataset/evidence/d1_fixture"
    d1_dir.mkdir(parents=True)
    run_id = "fixture-run-001"
    cert = {
        "certificate_schema_version": "1.0",
        "runner": {"schema_version": d1_runner.RUNNER_SCHEMA_VERSION, "run_id": run_id},
        "canonical_instance": {
            "canonical_instance_id": d0_manifest["canonical_instance_id"],
            "manifest_path": D3._rel(instance_dir / canonical.MANIFEST_NAME),
            "manifest_sha256": d0_manifest_sha,
            "specification_sha256": d0_manifest["specification_sha256"],
        },
        "build_attestation": {
            "path": D3._rel(instance_dir / attestation.ATTESTATION_NAME),
            "sha256": attestation_sha,
        },
        "code_identity": pins,
        "verdict": {"outcome": "PASS"},
    }
    cert_path = d1_dir / d1_runner.CERTIFICATE_NAME
    _write_json(cert_path, cert)
    run_manifest = {
        "run_status": "COMPLETE",
        "run_id": run_id,
        "canonical_instance_id": d0_manifest["canonical_instance_id"],
        "d0_manifest_sha256": d0_manifest_sha,
        "build_attestation_sha256": attestation_sha,
        "files": {d1_runner.CERTIFICATE_NAME: _sha256(cert_path)},
    }
    _write_json(d1_dir / d1_runner.RUN_MANIFEST_NAME, run_manifest)
    return {
        "repo": repo,
        "instance_dir": instance_dir,
        "d1_dir": d1_dir,
        "cert_path": cert_path,
        "manifest_path": d1_dir / d1_runner.RUN_MANIFEST_NAME,
        "source_files": source_files,
    }


def _rewrite_cert(env: dict, mutate) -> None:
    cert = json.loads(env["cert_path"].read_text())
    mutate(cert)
    _write_json(env["cert_path"], cert)
    manifest = json.loads(env["manifest_path"].read_text())
    manifest["files"]["d1_gohr_certificate.json"] = _sha256(env["cert_path"])
    _write_json(env["manifest_path"], manifest)


def test_accepts_consistent_instance_bound_d1_evidence(evidence_env: dict) -> None:
    result = D3.verify_d1_evidence(evidence_env["d1_dir"], evidence_env["instance_dir"])
    assert result["outcome"] == "PASS"
    assert result["run_id"] == "fixture-run-001"


def test_rejects_certificate_manifest_run_id_mismatch(evidence_env: dict) -> None:
    manifest = json.loads(evidence_env["manifest_path"].read_text())
    manifest["run_id"] = "different-run"
    _write_json(evidence_env["manifest_path"], manifest)
    with pytest.raises(SystemExit, match="run_id does not exactly match"):
        D3.verify_d1_evidence(evidence_env["d1_dir"], evidence_env["instance_dir"])


def test_rejects_missing_code_identity_entry(evidence_env: dict) -> None:
    _rewrite_cert(evidence_env, lambda cert: cert["code_identity"].pop("engine"))
    with pytest.raises(SystemExit, match="code_identity"):
        D3.verify_d1_evidence(evidence_env["d1_dir"], evidence_env["instance_dir"])


def test_rejects_unexpected_or_changed_code_identity_hash(evidence_env: dict) -> None:
    _rewrite_cert(
        evidence_env,
        lambda cert: cert["code_identity"]["engine"].update(sha256="0" * 64),
    )
    with pytest.raises(SystemExit, match=r"code_identity\['engine'\]"):
        D3.verify_d1_evidence(evidence_env["d1_dir"], evidence_env["instance_dir"])


def test_rejects_source_bytes_changed_after_d1_run(evidence_env: dict) -> None:
    evidence_env["source_files"]["engine"].write_text("changed source after D1\n")
    with pytest.raises(SystemExit, match="frozen D1 source 'engine' has SHA-256"):
        D3.verify_d1_evidence(evidence_env["d1_dir"], evidence_env["instance_dir"])


def test_rejects_missing_run_id(evidence_env: dict) -> None:
    _rewrite_cert(evidence_env, lambda cert: cert["runner"].update(run_id=""))
    with pytest.raises(SystemExit, match="non-empty run-manifest run_id"):
        D3.verify_d1_evidence(evidence_env["d1_dir"], evidence_env["instance_dir"])


def test_rejects_mismatched_d0_specification_identity(evidence_env: dict) -> None:
    _rewrite_cert(evidence_env, lambda cert: cert["canonical_instance"].update(specification_sha256="wrong"))
    with pytest.raises(SystemExit, match="specification hash"):
        D3.verify_d1_evidence(evidence_env["d1_dir"], evidence_env["instance_dir"])


def test_rejects_malformed_json_without_traceback_dependency(evidence_env: dict) -> None:
    evidence_env["cert_path"].write_text("{ not json")
    with pytest.raises(SystemExit, match="unreadable or malformed"):
        D3.verify_d1_evidence(evidence_env["d1_dir"], evidence_env["instance_dir"])
