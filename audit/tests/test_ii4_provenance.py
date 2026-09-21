"""
Final pre-execution blockers: source manifest, environment capture,
experiment fingerprint, output-directory guard, preregistered
interpretation rule, and the II-4 -> certificate converter.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("framework.firewall")

from audit.implementation.ii4_certificate import II4CertificateError, convert_ii4_run
from audit.implementation.ii4_design import INTERPRETATION_RULE, apply_interpretation_rule
from audit.implementation.ii4_experiment import build_gohr_ii4_plan, build_gohr_ii4_smoke_plan
from audit.implementation.ii4_runner import (
    GENERIC_II4_SOURCES,
    REPO_ROOT,
    II4ExecutionError,
    II4Runner,
    build_source_manifest,
    check_output_dir,
)
from audit.tests.test_ii4_runner import FakeAdapter, SimulatedCrash


def _plan(tmp_path, **kw):
    return build_gohr_ii4_smoke_plan(output_dir=tmp_path / "run", n_blocks=10, epochs=2, **kw)


def _complete_run(tmp_path, adapter):
    II4Runner(plan=_plan(tmp_path), adapter=adapter).run(execute=True)
    return tmp_path / "run"


# --- output-directory guard ---------------------------------------------

@pytest.mark.parametrize("sub", [
    "audit/evidence_bundle/x", "audit/integration/evidence/x", "audit/integration/frozen/x",
    "audit/implementation/evidence/x", "audit/dataset/evidence/x", "audit/cryptography/evidence/x",
    "audit/implementation/reference/x",
])
def test_guard_refuses_protected_repository_locations(sub):
    assert check_output_dir(REPO_ROOT / sub)


def test_guard_refuses_any_path_containing_a_protected_component(tmp_path):
    assert check_output_dir(tmp_path / "evidence" / "ii4")
    assert check_output_dir(tmp_path / "frozen")
    assert check_output_dir(tmp_path / "a" / "evidence_bundle" / "b")


def test_guard_refuses_repository_root():
    assert any("repository root" in p for p in check_output_dir(REPO_ROOT))


def test_guard_refuses_unrelated_non_empty_directory(tmp_path):
    d = tmp_path / "busy"; d.mkdir(); (d / "something.txt").write_text("x")
    assert any("not an II-4 run directory" in p for p in check_output_dir(d))


def test_guard_accepts_fresh_and_existing_ii4_directories(tmp_path):
    assert check_output_dir(tmp_path / "fresh_run") == []
    d = tmp_path / "resume"; d.mkdir(); (d / "ii4_state.json").write_text("{}")
    assert check_output_dir(d) == []


def test_runner_refuses_protected_output_before_touching_disk(tmp_path):
    plan = build_gohr_ii4_smoke_plan(output_dir=tmp_path / "evidence" / "run", n_blocks=10, epochs=2)
    with pytest.raises(II4ExecutionError, match="Output directory refused"):
        II4Runner(plan=plan, adapter=FakeAdapter(2)).run(execute=True)
    assert not (tmp_path / "evidence").exists()


def test_production_plan_default_run_dir_guarded(tmp_path):
    plan = build_gohr_ii4_plan(output_dir=REPO_ROOT / "audit/implementation/evidence/ii4")
    with pytest.raises(II4ExecutionError, match="Output directory refused"):
        II4Runner(plan=plan, adapter=FakeAdapter(200)).run(execute=False)


# --- source manifest ---------------------------------------------------------

def test_source_manifest_covers_generic_and_reference_files():
    from audit.implementation.adapters.gohr_ii4 import GOHR_II4_PROVENANCE_SOURCES

    class _A:
        def provenance_sources(self):
            return GOHR_II4_PROVENANCE_SOURCES

    m = build_source_manifest(_A())
    paths = {e["path"] for e in m.values()}
    for rel in list(GENERIC_II4_SOURCES.values()) + list(GOHR_II4_PROVENANCE_SOURCES.values()):
        assert rel in paths
    for ref in ("speck.py", "train_nets.py", "train_5_rounds.py"):
        assert ref in paths
    assert "audit/implementation/adapters/gohr_ii4.py" in paths
    assert all(len(e["sha256"]) == 64 for e in m.values())
    assert not any(p.startswith("/") for p in paths)      # repository-relative only


def test_source_manifest_refuses_missing_file():
    class _A:
        def provenance_sources(self):
            return {"ghost": "does/not/exist.py"}
    with pytest.raises(II4ExecutionError, match="missing"):
        build_source_manifest(_A())


def test_reference_files_hash_to_recorded_values():
    from audit.common.provenance import sha256_file
    assert sha256_file(REPO_ROOT / "speck.py") == \
        "59296f7b7990ec3886fc524ff49256005d360eb3748d3b1570351d669acbb8ce"
    assert sha256_file(REPO_ROOT / "train_nets.py") == \
        "b0585971da829f8cd89e2d1e3f85eed600048fcd04bc7d0d32f33ee1cd03c669"


# --- environment + fingerprint -----------------------------------------------

def test_preregistration_records_sources_environment_and_rule(tmp_path):
    run = _complete_run(tmp_path, FakeAdapter(2))
    prereg = json.loads((run / "ii4_preregistration.json").read_text())
    assert prereg["source_manifest"] and prereg["source_manifest_sha256"]
    env = prereg["environment"]
    for key in ("python_version", "numpy_version", "tensorflow_version", "keras_version",
                "git_commit", "adapter_gpus"):
        assert key in env
    assert set(prereg["software_stack"]) == {"python", "numpy", "scipy", "tensorflow", "keras"}
    assert prereg["interpretation_rule"]["version"] == "II4-INTERPRETATION-V1"
    assert prereg["experiment_fingerprint"] != prereg["config_fingerprint"]


def test_final_report_carries_provenance(tmp_path):
    run = _complete_run(tmp_path, FakeAdapter(2))
    rep = json.loads((run / "ii4_analysis.json").read_text())
    assert rep["source_manifest"] and rep["software_stack"] and rep["session_environments"]


def test_resume_refused_when_source_code_changes(tmp_path):
    plan = _plan(tmp_path)
    with pytest.raises(SimulatedCrash):
        II4Runner(plan=plan, adapter=FakeAdapter(2, crash_at=("block2", "declared"))).run(execute=True)
    changed = FakeAdapter(2, sources={"fake_adapter": "audit/tests/test_ii4.py"})  # different file
    with pytest.raises(II4ExecutionError, match="experiment fingerprint"):
        II4Runner(plan=_plan(tmp_path), adapter=changed).run(execute=True)


# --- interpretation rule ---------------------------------------------------------

@pytest.mark.parametrize("mat,eq,outcome", [
    ("MATERIALLY_HIGHER", "NOT_EQUIVALENT", "FAIL"),
    ("MATERIALLY_LOWER", "NOT_EQUIVALENT", "FAIL"),
    ("NO_DIRECTIONAL_CLAIM", "EQUIVALENT_WITHIN_MARGIN", "CONDITIONAL_PASS"),
    ("NO_DIRECTIONAL_CLAIM", "INCONCLUSIVE", "INCONCLUSIVE"),
    ("INSUFFICIENT_BLOCKS", "NOT_ASSESSED", "INCONCLUSIVE"),
])
def test_interpretation_rule_mapping(mat, eq, outcome):
    assert apply_interpretation_rule(INTERPRETATION_RULE, mat, eq)["ii_outcome"] == outcome


def test_equivalence_clause_disclaims_ce_transferability():
    c = apply_interpretation_rule(INTERPRETATION_RULE, "NO_DIRECTIONAL_CLAIM", "EQUIVALENT_WITHIN_MARGIN")
    assert "ACCURACY CONFORMANCE ONLY" in c["meaning"] and "NOT validate" in c["meaning"]


def test_unmatched_verdict_combination_is_an_error():
    with pytest.raises(ValueError):
        apply_interpretation_rule(INTERPRETATION_RULE, "UNKNOWN", "UNKNOWN")


# --- certificate converter ----------------------------------------------------------

def test_converter_material_difference_gives_fail(tmp_path):
    run = _complete_run(tmp_path, FakeAdapter(2))
    out = convert_ii4_run(run, tmp_path / "certs", allow_non_evidentiary=True)
    assert out["decision"] == "FAIL" and out["clause"]["scientific_impact"] == "DEMONSTRATED"
    assert out["identifier"] == "implementation-certificate-V1"      # fresh temp registry
    assert out["tag"] == "CONFIRMATORY"                               # genuine pre-execution prereg


def test_converter_equivalence_gives_conditional_pass(tmp_path):
    run = _complete_run(tmp_path, FakeAdapter(2, base=(0.90, 0.90), noise=1e-4))
    out = convert_ii4_run(run, tmp_path / "certs", allow_non_evidentiary=True)
    assert out["decision"] == "CONDITIONAL_PASS"
    cert = json.loads(Path(out["path"]).read_text())
    assert any("does NOT validate transferability" in l for l in cert["limitations"])


def test_converter_wide_noise_gives_inconclusive(tmp_path):
    run = _complete_run(tmp_path, FakeAdapter(2, base=(0.90, 0.888), noise=0.02))
    out = convert_ii4_run(run, tmp_path / "certs", allow_non_evidentiary=True)
    assert out["decision"] == "INCONCLUSIVE"


def test_converter_insufficient_blocks_gives_inconclusive(tmp_path):
    adapter = FakeAdapter(2, short_epochs=True)          # every arm fails -> 0 valid blocks
    run = _complete_run(tmp_path, adapter)
    out = convert_ii4_run(run, tmp_path / "certs", allow_non_evidentiary=True)
    assert out["decision"] == "INCONCLUSIVE"


def test_converter_refuses_smoke_without_flag(tmp_path):
    run = _complete_run(tmp_path, FakeAdapter(2))
    with pytest.raises(II4CertificateError, match="NON-EVIDENTIARY"):
        convert_ii4_run(run, tmp_path / "certs")


def test_converter_never_writes_smoke_into_an_evidence_directory(tmp_path):
    run = _complete_run(tmp_path, FakeAdapter(2))
    with pytest.raises(II4CertificateError, match="evidence"):
        convert_ii4_run(run, tmp_path / "x" / "evidence", allow_non_evidentiary=True)


def test_converter_uses_rule_from_preregistration_not_code(tmp_path):
    run = _complete_run(tmp_path, FakeAdapter(2))
    prereg = json.loads((run / "ii4_preregistration.json").read_text())
    prereg["interpretation_rule"]["rules"] = []            # tamper: rule removed
    (run / "ii4_preregistration.json").write_text(json.dumps(prereg))
    with pytest.raises(ValueError, match="matched 0"):
        convert_ii4_run(run, tmp_path / "certs", allow_non_evidentiary=True)


def test_converter_appends_next_version_and_preserves_prior(tmp_path):
    certs = tmp_path / "certs"
    run = _complete_run(tmp_path, FakeAdapter(2))
    first = convert_ii4_run(run, certs, allow_non_evidentiary=True)
    before = Path(first["path"]).read_bytes()
    second = convert_ii4_run(run, certs, allow_non_evidentiary=True)
    assert second["identifier"] == "implementation-certificate-V2"
    assert Path(first["path"]).read_bytes() == before                # V1 untouched


def test_converter_refuses_incomplete_run(tmp_path):
    with pytest.raises(SimulatedCrash):
        II4Runner(plan=_plan(tmp_path),
                  adapter=FakeAdapter(2, crash_at=("block3", "declared"))).run(execute=True)
    with pytest.raises(II4CertificateError):
        convert_ii4_run(tmp_path / "run", tmp_path / "certs", allow_non_evidentiary=True)


# --- II-4 data/model path is proven equivalent to the reference ------------------

def _load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_ev_make_train_data_byte_identical_to_reference_under_identical_randomness():
    """
    II-4 generates data through experimental_validity/gohr/speck.py, whose
    file hash differs from the reference (documentation/formatting). This
    locks FUNCTIONAL identity: fed the same byte stream in place of
    os.urandom, both produce byte-identical X and Y.
    """
    import hashlib
    import numpy as np
    ref = _load("ref_speck_t", REPO_ROOT / "speck.py")
    ev = _load("ev_speck_t", REPO_ROOT / "audit/experimental_validity/gohr/speck.py")

    def feeder(tag):
        ctr = [0]
        def urandom(k):
            ctr[0] += 1
            return hashlib.shake_256(f"{tag}:{ctr[0]}".encode()).digest(k)
        return urandom

    for tag, n in (("a", 5000), ("b", 3001)):
        ref.urandom, ev.urandom = feeder(tag), feeder(tag)
        xa, ya = ref.make_train_data(n, 5); xb, yb = ev.make_train_data(n, 5)
        assert np.array_equal(xa, xb) and np.array_equal(ya, yb)
    assert ref.check_testvector() and ev.check_testvector()


def test_ev_model_layer_identical_to_reference_at_both_arms():
    pytest.importorskip("tensorflow")
    import json as _json
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT))
    import train_nets as ref
    from gohr import model as ev

    def sig(m):
        return [(l.__class__.__name__, _json.dumps(
            {k: v for k, v in l.get_config().items() if k != "name"}, sort_keys=True, default=str))
            for l in m.layers]

    for depth in (5, 10):
        a = ref.make_resnet(depth=depth, reg_param=1e-5)
        b = ev.make_resnet(depth=depth, reg_param=1e-5)
        assert sig(a) == sig(b) and a.count_params() == b.count_params()
