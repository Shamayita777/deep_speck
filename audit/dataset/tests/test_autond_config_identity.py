"""
Configuration-and-identity tests for the AutoND D0 experiment.

Scientific status
-----------------
NON-EVIDENTIARY.

What this suite exists to prevent
---------------------------------
The reviewed experiment accepted a `--cipher` override while continuing to
emit the fixed identifier `autond-speck3264`. Generating SIMON32/64 would
therefore have produced a canonical instance recorded as SPECK32/64 data, with
SPECK's verified input difference (0x400000) silently reused for a cipher it
was never chosen for. Every downstream certificate would have inherited that
identifier. A mislabelled instance is worse than a missing one: it is cited.

Two independent defences are tested here:

  * the experiment REFUSES a cipher outside its verified set, and
  * the adapter DERIVES the identifier from the cipher actually configured,
    so even a direct caller cannot obtain a mismatched label.

For the verified speck3264 configuration the derived identifier is exactly the
module constant, so no existing instance's identity changes.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from audit.dataset.adapters import autond as A
from audit.dataset.experiments import d0_autond_canonical as E

_CANDIDATE_ROOTS = (os.environ.get("AUTOND_ROOT"),
                    "~/Documents/AutoND-upstream")


def _pinned_root() -> Path:
    for cand in _CANDIDATE_ROOTS:
        if not cand:
            continue
        p = Path(cand).expanduser()
        if (p / "main.py").is_file():
            return p.resolve()
    pytest.skip("pinned AutoND checkout not found; set AUTOND_ROOT. "
                "SKIPPED IS NOT PASSED.")


@pytest.fixture(scope="module")
def root() -> Path:
    return _pinned_root()


def _args(root: Path, **over) -> argparse.Namespace:
    ns = argparse.Namespace(
        autond_root=str(root), num_rounds=8, cipher=None,
        delta_plain_hex=None, train_samples=100,
        validation_samples=10, test_samples=10)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------
# 1. The experiment restricts itself to the verified configuration
# ---------------------------------------------------------------------

def test_supported_cipher_is_accepted(root):
    spec = E.build_spec(_args(root, cipher="speck3264"))
    assert spec["cipher"] == "speck3264"
    assert spec["dataset_id"] == "autond-speck3264"
    assert spec["verified_configuration"] is True


def test_unsupported_cipher_is_refused(root):
    with pytest.raises(SystemExit) as exc:
        E.build_spec(_args(root, cipher="simon3264"))
    msg = str(exc.value)
    assert "REFUSED" in msg
    assert "simon3264" in msg
    # The refusal must say what is missing, not merely that it is unsupported.
    assert "verified" in msg


def test_unknown_cipher_is_refused(root):
    with pytest.raises(SystemExit, match="REFUSED"):
        E.build_spec(_args(root, cipher="not_a_cipher"))


def test_supported_set_is_exactly_the_verified_set():
    """The experiment must not claim support beyond what was verified."""
    assert set(E.SUPPORTED_CIPHERS) <= set(A.VERIFIED_CIPHERS)
    assert set(E.SUPPORTED_CIPHERS) == {"speck3264"}
    assert set(E.VERIFIED_DELTA_PLAIN_HEX) == set(E.SUPPORTED_CIPHERS), (
        "every supported cipher needs a verified input difference of its own; "
        "reusing another cipher's difference is the defect this prevents")


# ---------------------------------------------------------------------
# 2. The identifier is derived from the cipher, not fixed
# ---------------------------------------------------------------------

def test_adapter_dataset_id_is_derived_from_the_cipher(root):
    """Even a direct caller cannot get a mismatched identifier.

    Constructed with a cipher outside the verified set, the adapter must label
    it for what it is. This is the second defence: the experiment refuses such
    a configuration, but the adapter is importable on its own.
    """
    a = A.AutoNDDatasetAdapter(
        autond_root=root, cipher_name="simon3264",
        delta_plain_int=0x400000, num_rounds=8)
    assert a.dataset_id == "autond-simon3264"
    assert a.dataset_id != A.DATASET_ID, (
        "a SIMON adapter reported the SPECK identifier")
    assert a.is_verified_configuration is False


def test_derived_id_equals_the_module_constant_for_the_case_study(root):
    """Backward compatibility: no existing instance's identity changes."""
    a = A.AutoNDDatasetAdapter(
        autond_root=root, cipher_name="speck3264",
        delta_plain_int=0x400000, num_rounds=8)
    assert a.dataset_id == A.DATASET_ID == "autond-speck3264"


def test_provenance_carries_the_derived_id_and_verification_flag(root):
    a = A.AutoNDDatasetAdapter(
        autond_root=root, cipher_name="simon3264",
        delta_plain_int=0x400000, num_rounds=8)
    prov = a.dataset_provenance(train_samples=1, validation_samples=1,
                                test_samples=1)
    assert prov["dataset_id"] == "autond-simon3264"
    assert prov["verified_configuration"] is False
    assert "simon3264" in prov["dataset_version"]


def test_dataset_version_names_the_cipher(root):
    """Two ciphers at the same round count must not share a version string."""
    speck = A.AutoNDDatasetAdapter(
        autond_root=root, cipher_name="speck3264",
        delta_plain_int=0x400000, num_rounds=8).dataset_provenance(
            train_samples=1, validation_samples=1, test_samples=1)
    simon = A.AutoNDDatasetAdapter(
        autond_root=root, cipher_name="simon3264",
        delta_plain_int=0x400000, num_rounds=8).dataset_provenance(
            train_samples=1, validation_samples=1, test_samples=1)
    assert speck["dataset_version"] != simon["dataset_version"]


# ---------------------------------------------------------------------
# 3. The spec, the identity and the artifacts agree
# ---------------------------------------------------------------------

def test_spec_identity_agrees_with_the_adapter(root):
    spec = E.build_spec(_args(root))
    assert spec["dataset_id"] == f"autond-{spec['cipher']}"
    assert (spec["adapter_provenance"]["dataset_id"] == spec["dataset_id"])
    assert (spec["adapter_provenance"]["generation_parameters"]["cipher"]
            == spec["cipher"])


def test_declared_constraints_are_preserved(root):
    """The round count, differences, sample counts and representation.

    All five identify the instance, so all five must appear in the spec with
    the values actually used.
    """
    spec = E.build_spec(_args(root, num_rounds=6, train_samples=321,
                              validation_samples=21, test_samples=11))
    assert spec["num_rounds"] == 6
    assert "r6" in spec["dataset_version"]
    assert spec["delta_plain_hex"] == "0x400000"
    assert spec["delta_key_hex"] == "0x0"
    assert spec["scenario"] == "single-key"
    assert spec["train_samples"] == 321
    assert spec["validation_samples"] == 21
    assert spec["test_samples"] == 11
    rep = spec["representation"]
    assert rep["feature_bits"] == 64
    assert rep["plain_bits"] == 32
    assert rep["key_bits"] == 64


def test_reduced_scale_is_announced_in_the_spec(root):
    spec = E.build_spec(_args(root, train_samples=100))
    assert spec["reduced_scale"] is True
    assert "NOT production evidence" in spec["reduced_scale_note"]


def test_production_sizes_are_not_marked_reduced(root):
    spec = E.build_spec(_args(
        root, train_samples=10_000_000,
        validation_samples=1_000_000, test_samples=1_000_000))
    assert spec["reduced_scale"] is False


# ---------------------------------------------------------------------
# 4. A non-verified input difference is recorded as such
# ---------------------------------------------------------------------

def test_non_verified_delta_is_flagged_not_silently_accepted(root):
    spec = E.build_spec(_args(root, delta_plain_hex="0x11"))
    assert spec["delta_plain_hex"] == "0x11"
    assert spec["delta_plain_verified"] is False
    assert spec["verified_configuration"] is False
    assert "has not been checked" in spec["delta_plain_verified_note"]


def test_verified_delta_sets_the_flag(root):
    spec = E.build_spec(_args(root, delta_plain_hex="0x400000"))
    assert spec["delta_plain_verified"] is True
    assert spec["verified_configuration"] is True


def test_equivalent_hex_forms_of_the_verified_delta_are_accepted(root):
    """Compared numerically, so 0x0400000 is the same difference."""
    spec = E.build_spec(_args(root, delta_plain_hex="0x0400000"))
    assert spec["delta_plain_verified"] is True


# ---------------------------------------------------------------------
# 5. The source pin is ENFORCED by the production experiment
# ---------------------------------------------------------------------
# The adapter's `expected_source_digests` is optional so the adapter stays
# reusable. The production experiment does not inherit that latitude: it
# always supplies and enforces a reviewed pin, because an instance generated
# from an unapproved checkout would otherwise be described as the approved
# target. Enforcement happens in the adapter CONSTRUCTOR, i.e. before any data
# is generated.

def test_the_reviewed_pin_is_enforced_by_default(root):
    spec = E.build_spec(_args(root))
    assert spec["source_pin"]["enforced"] is True
    assert spec["source_pin"]["upstream_commit"] == (
        "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779")
    assert set(spec["source_digests"]) >= {
        "main.py", "ciphers/speck3264.py", "optimizer.py", "train_nets.py",
        "dbitnet.py", "gohrnet.py"}


def test_recorded_digests_match_the_enforced_pin(root):
    spec = E.build_spec(_args(root))
    for rel, pinned in spec["source_pin"]["source_digests"].items():
        assert spec["source_digests"][rel] == pinned


def test_recorded_source_paths_are_inside_the_checkout(root):
    spec = E.build_spec(_args(root))
    for rel, path in spec["source_paths"].items():
        assert Path(path).is_relative_to(root), (
            f"{rel} was loaded from {path}, outside the checkout")


def test_the_pin_records_how_its_digests_were_established(root):
    """A pin copied from an unverified report is not a pin.

    The provenance list is asserted so the values cannot later be replaced
    with unattributed ones without this failing.
    """
    prov = E.AUTOND_SOURCE_PIN["digest_provenance"]
    assert len(prov) >= 2
    joined = " ".join(prov).lower()
    assert "checkout" in joined
    assert "corroborated" in joined or "independent" in joined


def test_production_experiment_refuses_an_omitted_pin(root):
    with pytest.raises(SystemExit) as exc:
        E.build_spec(_args(root), source_pin={})
    msg = str(exc.value)
    assert "REFUSED" in msg
    assert "unverified checkout" in msg


def test_production_experiment_refuses_a_pin_missing_a_required_file(root):
    partial = {"upstream_commit": E.AUTOND_SOURCE_PIN["upstream_commit"],
               "source_digests": {
                   "main.py":
                       E.AUTOND_SOURCE_PIN["source_digests"]["main.py"]}}
    with pytest.raises(SystemExit) as exc:
        E.build_spec(_args(root), source_pin=partial)
    assert "ciphers/speck3264.py" in str(exc.value)


def test_mismatching_source_digest_is_refused(root):
    from audit.dataset.adapters.autond import SourceBindingError
    bad = {"upstream_commit": E.AUTOND_SOURCE_PIN["upstream_commit"],
           "source_digests": dict(E.AUTOND_SOURCE_PIN["source_digests"],
                                  **{"main.py": "0" * 64})}
    with pytest.raises(SourceBindingError, match="pin mismatch"):
        E.build_spec(_args(root), source_pin=bad)


def test_mismatching_revision_is_refused(root):
    with pytest.raises(SystemExit, match="never approved"):
        E.build_spec(_args(root), source_pin={
            "upstream_commit": "d" * 40,
            "source_digests": dict(E.AUTOND_SOURCE_PIN["source_digests"])})


def test_observed_commit_matches_the_pin(root):
    spec = E.build_spec(_args(root))
    assert spec["source_commit_observed"] == (
        "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779")


def _run_isolated(body: str, tmp_path: Path) -> dict:
    """Run a scenario in a fresh interpreter and return its JSON verdict.

    Needed because the two tests below point the experiment at a SECOND
    checkout. By then this process has already imported the real one, so the
    adapter's two-checkouts-in-one-process guard fires first and masks the
    digest check these tests are about. That guard is correct; the tests just
    have to respect it.
    """
    import json as _json, os as _os, subprocess as _sp, sys as _sys, textwrap
    script = tmp_path / "scenario.py"
    script.write_text(textwrap.dedent(body))
    env = dict(_os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = _sp.run([_sys.executable, str(script)], capture_output=True,
                   text=True, env=env, cwd=str(tmp_path), timeout=300)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("@@")]
    if not lines:
        raise AssertionError(
            f"scenario produced no verdict.\nstdout:\n{proc.stdout[-3000:]}\n"
            f"stderr:\n{proc.stderr[-3000:]}")
    return _json.loads(lines[-1][2:])


def _copy_checkout(root: Path, dest: Path) -> Path:
    import shutil
    shutil.copytree(root, dest,
                    ignore=shutil.ignore_patterns(".git", "__pycache__",
                                                  "model_weights"))
    return dest


def test_changing_the_generator_source_invalidates_the_pin(root, tmp_path):
    """Modify a COPY of the checkout; the pinned original is never touched.

    The property that matters most: if the generator's source changes by one
    byte, the case study must refuse to generate rather than produce a dataset
    it would then describe as the approved target. Run isolated, so the digest
    check is what fires.
    """
    import hashlib
    copy = _copy_checkout(root, tmp_path / "modified_checkout")
    main_py = copy / "main.py"
    main_py.write_text(main_py.read_text()
                       + "\n# appended by a test; one byte is enough\n")
    verdict = _run_isolated(f"""
        import argparse, json
        from audit.dataset.experiments import d0_autond_canonical as E
        from audit.dataset.adapters.autond import SourceBindingError
        ns = argparse.Namespace(
            autond_root={str(copy)!r}, num_rounds=8, cipher=None,
            delta_plain_hex=None, train_samples=100, validation_samples=10,
            test_samples=10)
        pin = {{"upstream_commit": None,
                "source_digests": dict(E.AUTOND_SOURCE_PIN["source_digests"])}}
        try:
            E.build_spec(ns, source_pin=pin)
            print("@@" + json.dumps({{"refused": False, "message": ""}}))
        except (SourceBindingError, SystemExit) as exc:
            print("@@" + json.dumps({{"refused": True, "message": str(exc)}}))
    """, tmp_path)
    assert verdict["refused"] is True, (
        "a modified generator source was accepted; the instance would have "
        "been recorded as the approved target")
    assert "mismatch" in verdict["message"].lower()
    # The real checkout is untouched.
    assert hashlib.sha256((root / "main.py").read_bytes()).hexdigest() == (
        E.AUTOND_SOURCE_PIN["source_digests"]["main.py"])


def test_changing_the_cipher_source_invalidates_the_pin(root, tmp_path):
    import hashlib
    copy = _copy_checkout(root, tmp_path / "modified_cipher_checkout")
    cipher = copy / "ciphers" / "speck3264.py"
    cipher.write_text(cipher.read_text() + "\n# appended by a test\n")
    verdict = _run_isolated(f"""
        import argparse, json
        from audit.dataset.experiments import d0_autond_canonical as E
        from audit.dataset.adapters.autond import SourceBindingError
        ns = argparse.Namespace(
            autond_root={str(copy)!r}, num_rounds=8, cipher=None,
            delta_plain_hex=None, train_samples=100, validation_samples=10,
            test_samples=10)
        pin = {{"upstream_commit": None,
                "source_digests": dict(E.AUTOND_SOURCE_PIN["source_digests"])}}
        try:
            E.build_spec(ns, source_pin=pin)
            print("@@" + json.dumps({{"refused": False, "message": ""}}))
        except (SourceBindingError, SystemExit) as exc:
            print("@@" + json.dumps({{"refused": True, "message": str(exc)}}))
    """, tmp_path)
    assert verdict["refused"] is True
    assert "mismatch" in verdict["message"].lower()
    assert hashlib.sha256(
        (root / "ciphers" / "speck3264.py").read_bytes()).hexdigest() == (
        E.AUTOND_SOURCE_PIN["source_digests"]["ciphers/speck3264.py"])


def test_the_enforced_pin_is_recorded_in_the_specification(root):
    """D1's preflight reads these fields, so they must be present."""
    spec = E.build_spec(_args(root))
    assert spec["source_pin"]["source_digests"]
    assert spec["source_digests"]
    assert spec["adapter_provenance"]["source_digests"] == spec[
        "source_digests"]
