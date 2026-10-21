"""
Fail-closed tests for the AutoND D1 experiment runner.

Scientific status
-----------------
NON-EVIDENTIARY.

What this suite exists to prevent
---------------------------------
The earlier runner loaded the canonical arrays and ran duplicate detection
BEFORE the manifest had been fully verified. Expensive audit work could
therefore execute against an instance whose metadata did not match its data,
and emit a certificate for it.

Every test here replaces the engine with a SPY and asserts the spy was never
called. Asserting only that an exception was raised would not establish the
ordering: the engine could have run first and the failure surfaced afterwards.
The spy is the thing that proves nothing expensive started.

A second property is pinned here: `production_status` is derived from the
VERIFIED row counts, never from `specification.reduced_scale`. Editing a flag,
or hand-editing a manifest, must not promote a smoke test to production
evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audit.dataset import canonical as C
from audit.dataset.experiments import d1_autond_exact_census as D1

TEST_TARGET = "autond"          # the identity the runner expects

#: The frozen pin, imported rather than copied, so the fixture cannot drift
#: away from the value D1 checks against.
from audit.dataset.experiments.d0_autond_canonical import (      # noqa: E402
    AUTOND_SOURCE_PIN as _FROZEN_PIN)


def _resolver(spec: dict):
    """A deterministic stand-in for the AutoND generator.

    This suite is about the runner's ORDERING and its identity checks, not
    about AutoND's data, so it does not need the pinned checkout and must not
    depend on one.
    """
    bits = int(spec.get("representation", {}).get("feature_bits", 64))

    def generate(n: int):
        rng = np.random.default_rng(4242 + n)
        return (rng.integers(0, 2, size=(n, bits), dtype=np.uint8),
                rng.integers(0, 2, size=n, dtype=np.uint8))

    return generate, "test.autond.generate"


def _spec(**over) -> dict:
    spec = {
        "target": "autond",
        "dataset_id": "autond-speck3264",
        "cipher": "speck3264",
        "scenario": "single-key",
        "num_rounds": 8,
        "delta_plain_hex": "0x400000",
        "delta_key_hex": "0x0",
        "generator": "main.make_train_data",
        "randomness_source": "os.urandom",
        "feature_dtype": "uint8",
        "label_dtype": "uint8",
        "label_values": [0, 1],
        "train_samples": 400,
        "validation_samples": 100,
        "test_samples": 100,
        "reduced_scale": True,
        # The frozen case study's own flags. The fixture must model a VALID
        # instance, or every test below would pass for the wrong reason.
        "verified_configuration": True,
        "delta_plain_verified": True,
        "source_commit_observed": _FROZEN_PIN["upstream_commit"],
        "source_pin": {"source_digests": dict(_FROZEN_PIN["source_digests"]),
                       "upstream_commit": _FROZEN_PIN["upstream_commit"],
                       "enforced": True},
        "representation": {
            "feature_bits": 64, "plain_bits": 32, "key_bits": 64,
            "word_size": 16, "layout": "np.hstack([C0, C1])",
            "c0_columns": [0, 32], "c1_columns": [32, 64],
        },
        # The real frozen-pin digests: D1 now cross-checks the recorded source
        # identity against the pin in the case-study configuration, which lives
        # outside this manifest, so placeholders would be refused.
        "adapter_provenance": {
            "source_digests": dict(_FROZEN_PIN["source_digests"]),
        },
    }
    spec.update(over)
    return spec


class Spy:
    """Stands in for the generic D1 engine and records whether it ran."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        raise AssertionError(
            "the D1 engine was invoked; preflight should have stopped first")


@pytest.fixture
def spy(monkeypatch):
    s = Spy()
    monkeypatch.setattr(D1, "D1_ENGINE", s)
    return s


@pytest.fixture
def instance(tmp_path, monkeypatch):
    monkeypatch.setitem(C.GENERATOR_RESOLVERS, TEST_TARGET, _resolver)
    out = tmp_path / "d0"
    man = C.build(out, repo_root=tmp_path, spec=_spec(),
                  log=lambda *a, **k: None)
    return out, man


def _read(out: Path) -> dict:
    return json.loads((out / C.MANIFEST_NAME).read_text())


def _write(out: Path, man: dict) -> None:
    (out / C.MANIFEST_NAME).write_text(json.dumps(man, indent=2,
                                                  sort_keys=True))


def _reseal(man: dict) -> dict:
    """Recompute the identifier and specification digest after an edit.

    Without this the instance would fail on the identity check and the test
    would not be exercising the check it names.
    """
    man["canonical_instance_id"] = C.compute_instance_id(man["partitions"])
    man["specification_sha256"] = C.compute_specification_digest(
        man["specification"])
    return man


def _run(out: Path, tmp_path: Path):
    return D1.main(["--instance-dir", str(out),
                    "--out-dir", str(tmp_path / "d1out")])


# ---------------------------------------------------------------------
# 1. The control: a valid instance DOES reach the engine
# ---------------------------------------------------------------------

def test_a_valid_instance_reaches_the_engine(instance, tmp_path, monkeypatch):
    """Otherwise every test below would pass for the wrong reason."""
    reached = {"n": 0}

    def fake_engine(*a, **k):
        reached["n"] += 1
        return D1.audit_duplicates(*a, **k)

    monkeypatch.setattr(D1, "D1_ENGINE", fake_engine)
    out, _ = instance
    assert _run(out, tmp_path) == 0
    assert reached["n"] == 1


def test_preflight_passes_on_a_valid_instance(instance):
    out, man = instance
    pre = D1.preflight(out, log=lambda *a, **k: None)
    assert pre["verification"]["verified"] is True
    assert pre["manifest"]["canonical_instance_id"] == man[
        "canonical_instance_id"]


# ---------------------------------------------------------------------
# 2. Tampered data and metadata stop the engine
# ---------------------------------------------------------------------

def test_tampered_feature_data_stops_the_engine(instance, tmp_path, spy):
    out, man = instance
    path = out / man["partitions"]["train"]["file"]
    with np.load(path) as z:
        X, Y = z["X"].copy(), z["Y"].copy()
    X[0, 0] ^= 1
    np.savez(path, X=X, Y=Y)
    with pytest.raises(SystemExit, match="did not verify"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_tampered_instance_id_stops_the_engine(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    man["canonical_instance_id"] = "f" * 64
    _write(out, man)
    with pytest.raises(SystemExit, match="did not verify"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_wrong_recorded_sample_count_stops_the_engine(instance, tmp_path, spy):
    """§4's named case: an incorrect recorded sample count."""
    out, _ = instance
    man = _reseal({**_read(out)})
    man["partitions"]["train"]["samples"] = 999999
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="did not verify"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_wrong_recorded_label_counts_stop_the_engine(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    man["partitions"]["validation"]["label_counts"] = {"positive": 1,
                                                       "negative": 99}
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="did not verify"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_wrong_recorded_dtype_stops_the_engine(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    man["partitions"]["train"]["features"]["dtype"] = "float64"
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="did not verify"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_missing_partition_file_stops_the_engine(instance, tmp_path, spy):
    out, man = instance
    (out / man["partitions"]["test"]["file"]).unlink()
    with pytest.raises(SystemExit):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_malformed_manifest_stops_the_engine(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    del man["partitions"]["test"]
    _write(out, man)
    with pytest.raises(SystemExit, match="REFUSED"):
        _run(out, tmp_path)
    assert spy.calls == 0


# ---------------------------------------------------------------------
# 3. Wrong target identity stops the engine
# ---------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("target", "gohr-speck"),
    ("dataset_id", "autond-simon3264"),
    ("cipher", "simon3264"),
    ("scenario", "related-key"),
])
def test_wrong_identity_stops_the_engine(instance, tmp_path, spy, field, value):
    out, _ = instance
    man = _read(out)
    man["specification"][field] = value
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert "REFUSED" in str(exc.value)
    assert spy.calls == 0


def test_gohr_evidence_cannot_be_audited_as_autond(instance, tmp_path, spy):
    """A Gohr instance must not be accepted by the AutoND experiment."""
    out, _ = instance
    man = _read(out)
    man["specification"]["target"] = "gohr-speck"
    man["specification"]["dataset_id"] = "gohr-speck"
    man["specification"]["cipher"] = "speck3264"
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="REFUSED"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_mismatched_dataset_id_and_cipher_stops_the_engine(instance, tmp_path,
                                                           spy):
    out, _ = instance
    man = _read(out)
    man["specification"]["cipher"] = "simon3264"
    man["specification"]["dataset_id"] = "autond-speck3264"
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="REFUSED"):
        _run(out, tmp_path)
    assert spy.calls == 0


# ---------------------------------------------------------------------
# 4. Wrong representation stops the engine
# ---------------------------------------------------------------------

def test_wrong_declared_feature_bits_stops_the_engine(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    man["specification"]["representation"]["feature_bits"] = 32
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="REFUSED"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_missing_representation_stops_the_engine(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    del man["specification"]["representation"]
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="representation"):
        _run(out, tmp_path)
    assert spy.calls == 0


# ---------------------------------------------------------------------
# 5. A missing source record stops the engine
# ---------------------------------------------------------------------

def test_absent_source_digests_stop_the_engine(instance, tmp_path, spy):
    """An instance with no recorded source cannot be audited as the target."""
    out, _ = instance
    man = _read(out)
    man["specification"]["adapter_provenance"] = {}
    man["specification"].pop("source_digests", None)
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="source_digests"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_recorded_digests_disagreeing_with_the_pin_stop_the_engine(
        instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    man["specification"]["source_pin"] = {
        "source_digests": {"main.py": "c" * 64}}
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="do not match the pin"):
        _run(out, tmp_path)
    assert spy.calls == 0


# ---------------------------------------------------------------------
# 5b. The FROZEN case-study configuration
# ---------------------------------------------------------------------
# This experiment audits exactly one configuration. A different round count or
# input difference is a different experiment with different scope statements,
# so auditing one under this script's scope text would misdescribe it. None of
# these is reachable by editing a manifest.

@pytest.mark.parametrize("rounds", [5, 7, 9, 22])
def test_wrong_round_count_stops_the_engine(instance, tmp_path, spy, rounds):
    out, _ = instance
    man = _read(out)
    man["specification"]["num_rounds"] = rounds
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert "num_rounds" in str(exc.value)
    assert "frozen case-study" in str(exc.value)
    assert spy.calls == 0


@pytest.mark.parametrize("delta", ["0x200000", "0x11", "0x0", "0x800000"])
def test_wrong_plaintext_difference_stops_the_engine(instance, tmp_path, spy,
                                                     delta):
    out, _ = instance
    man = _read(out)
    man["specification"]["delta_plain_hex"] = delta
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert "plaintext difference" in str(exc.value)
    assert spy.calls == 0


def test_equivalent_hex_spelling_of_the_frozen_difference_is_accepted(
        instance, tmp_path, monkeypatch):
    """Compared numerically, so 0x0400000 is the same difference.

    Without this the frozen check would be a string match and would reject a
    legitimate instance over its formatting.
    """
    monkeypatch.setattr(D1, "D1_ENGINE", D1.audit_duplicates)
    out, _ = instance
    man = _read(out)
    man["specification"]["delta_plain_hex"] = "0x0400000"
    _write(out, _reseal(man))
    assert _run(out, tmp_path) == 0


def test_nonzero_key_difference_stops_the_engine(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    man["specification"]["delta_key_hex"] = "0x1"
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert "key difference" in str(exc.value)
    assert spy.calls == 0


def test_malformed_difference_stops_the_engine(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    man["specification"]["delta_plain_hex"] = "not-a-number"
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="hex integer"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_unverified_configuration_flag_stops_the_engine(instance, tmp_path,
                                                        spy):
    out, _ = instance
    man = _read(out)
    man["specification"]["verified_configuration"] = False
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="verified_configuration"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_missing_verified_configuration_flag_stops_the_engine(instance,
                                                              tmp_path, spy):
    out, _ = instance
    man = _read(out)
    del man["specification"]["verified_configuration"]
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="verified_configuration"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_unverified_delta_flag_stops_the_engine(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    man["specification"]["delta_plain_verified"] = False
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="delta_plain_verified"):
        _run(out, tmp_path)
    assert spy.calls == 0


@pytest.mark.parametrize("field,value", [
    ("layout", "np.vstack([C0, C1])"),
    ("c0_columns", [0, 64]),
    ("c1_columns", [0, 32]),
])
def test_wrong_feature_layout_stops_the_engine(instance, tmp_path, spy,
                                              field, value):
    """The source-defined concatenation is part of what the engine is told."""
    out, _ = instance
    man = _read(out)
    man["specification"]["representation"][field] = value
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert field in str(exc.value)
    assert spy.calls == 0


def test_wrong_declared_label_values_stop_the_engine(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    man["specification"]["label_values"] = [0, 1, 2]
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="label_values"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_declared_and_recorded_partition_sizes_must_agree(instance, tmp_path,
                                                          spy):
    """A manifest internally consistent in each half but not across them."""
    out, _ = instance
    man = _read(out)
    # Change the declared count only. D0's spec check catches it too; this
    # asserts the D1 preflight has its own independent size comparison.
    man["specification"]["train_samples"] = 123
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert spy.calls == 0
    assert "REFUSED" in str(exc.value)


def test_the_frozen_values_are_the_declared_ones():
    assert D1.EXPECTED_NUM_ROUNDS == 8
    assert D1.EXPECTED_DELTA_PLAIN == 0x400000
    assert D1.EXPECTED_DELTA_KEY == 0x0
    assert D1.EXPECTED_CIPHER == "speck3264"
    assert D1.EXPECTED_SCENARIO == "single-key"
    assert D1.EXPECTED_LAYOUT == "np.hstack([C0, C1])"
    assert D1.EXPECTED_FEATURE_BITS == 64
    assert "its own declared configuration" in D1.SCOPE_NOTE


# ---------------------------------------------------------------------
# 5c. Source identity is checked against the FROZEN pin, not the manifest
# ---------------------------------------------------------------------

def test_a_self_consistent_but_wrong_source_identity_is_refused(
        instance, tmp_path, spy):
    """The case the earlier check could not catch.

    An instance generated from an unapproved checkout could record THAT
    checkout's digests in both `adapter_provenance.source_digests` and
    `source_pin.source_digests`. The two agree, so a manifest-internal
    comparison passes. The comparison is therefore made against the frozen
    pin in the case-study configuration, which lives outside this manifest.
    """
    out, _ = instance
    man = _read(out)
    rogue = {"main.py": "9" * 64, "ciphers/speck3264.py": "8" * 64}
    man["specification"]["adapter_provenance"]["source_digests"] = dict(rogue)
    man["specification"]["source_pin"] = {"source_digests": dict(rogue)}
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    msg = str(exc.value)
    assert "FROZEN case-study pin" in msg
    assert "not against another field of the same manifest" in msg
    assert spy.calls == 0


def test_recorded_source_matching_the_frozen_pin_is_accepted(instance,
                                                             tmp_path,
                                                             monkeypatch):
    """The control: the frozen-pin check must accept the approved source."""
    from audit.dataset.experiments.d0_autond_canonical import (
        AUTOND_SOURCE_PIN)
    monkeypatch.setattr(D1, "D1_ENGINE", D1.audit_duplicates)
    out, _ = instance
    man = _read(out)
    man["specification"]["adapter_provenance"]["source_digests"] = dict(
        AUTOND_SOURCE_PIN["source_digests"])
    man["specification"]["source_pin"] = {
        "source_digests": dict(AUTOND_SOURCE_PIN["source_digests"]),
        "upstream_commit": AUTOND_SOURCE_PIN["upstream_commit"],
        "enforced": True}
    _write(out, _reseal(man))
    assert _run(out, tmp_path) == 0


def test_a_wrong_recorded_revision_is_refused(instance, tmp_path, spy):
    from audit.dataset.experiments.d0_autond_canonical import (
        AUTOND_SOURCE_PIN)
    out, _ = instance
    man = _read(out)
    man["specification"]["adapter_provenance"]["source_digests"] = dict(
        AUTOND_SOURCE_PIN["source_digests"])
    man["specification"]["source_commit_observed"] = "b" * 40
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="revision"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_an_unverified_revision_record_is_REFUSED(instance, tmp_path, spy):
    """Reverses the r3 assertion, deliberately.

    r3 accepted `source_commit_observed == "UNVERIFIED"` on the grounds that
    the digests still had to match the frozen pin. That is not good enough for
    the frozen production case study: "UNVERIFIED" is exactly the state
    produced when the checkout could not report a revision, so the instance
    cannot be shown to have come from the approved one. Matching digests
    identify a FILE SET, not a revision.

    This is a tightening, not a weakened test: the case that used to pass now
    has to be refused, and the refusal is asserted to happen before the
    engine.
    """
    from audit.dataset.experiments.d0_autond_canonical import (
        AUTOND_SOURCE_PIN)
    out, _ = instance
    man = _read(out)
    man["specification"]["adapter_provenance"]["source_digests"] = dict(
        AUTOND_SOURCE_PIN["source_digests"])
    man["specification"]["source_commit_observed"] = "UNVERIFIED"
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    msg = str(exc.value)
    assert "UNVERIFIED" in msg
    assert "do not establish a revision" in msg
    assert spy.calls == 0


def test_a_missing_observed_commit_is_refused(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    del man["specification"]["source_commit_observed"]
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert "no `source_commit_observed`" in str(exc.value)
    assert spy.calls == 0


def test_a_missing_source_pin_record_is_refused(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    del man["specification"]["source_pin"]
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="no `source_pin`"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_an_incomplete_source_pin_record_is_refused(instance, tmp_path, spy):
    """A pin weaker than the approved one is refused.

    An instance generated under a two-file pin ran four modules unchecked, so
    its own record of what was enforced must be rejected as insufficient.
    """
    out, _ = instance
    man = _read(out)
    full = man["specification"]["source_pin"]["source_digests"]
    man["specification"]["source_pin"]["source_digests"] = {
        "main.py": full["main.py"],
        "ciphers/speck3264.py": full["ciphers/speck3264.py"]}
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    msg = str(exc.value)
    assert "does not cover" in msg
    assert "weaker than the approved" in msg
    assert spy.calls == 0


def test_a_source_pin_naming_the_wrong_revision_is_refused(instance, tmp_path,
                                                           spy):
    out, _ = instance
    man = _read(out)
    man["specification"]["source_pin"]["upstream_commit"] = "c" * 40
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="not the approved"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_a_consistent_but_unapproved_checkout_identity_is_refused(
        instance, tmp_path, spy):
    """Revision AND digests consistently describing an unapproved checkout.

    Both fields agree with each other, and the pin record agrees too. Only the
    comparison against the frozen configuration catches it.
    """
    out, _ = instance
    man = _read(out)
    rogue = {k: ("7" * 64) for k in
             man["specification"]["source_pin"]["source_digests"]}
    man["specification"]["adapter_provenance"]["source_digests"] = dict(rogue)
    man["specification"]["source_digests"] = dict(rogue)
    man["specification"]["source_pin"] = {
        "source_digests": dict(rogue), "upstream_commit": "a" * 40,
        "enforced": True}
    man["specification"]["source_commit_observed"] = "a" * 40
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert spy.calls == 0
    assert "REFUSED" in str(exc.value)


def test_adapter_and_specification_source_identities_must_agree(
        instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    man["specification"]["source_digests"] = {
        k: ("6" * 64) for k in
        man["specification"]["adapter_provenance"]["source_digests"]}
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert spy.calls == 0
    assert "REFUSED" in str(exc.value)


def test_the_approved_reduced_scale_instance_remains_accepted(
        instance, tmp_path, monkeypatch):
    """The control for all of Gap 2: a valid r3-style instance still runs."""
    monkeypatch.setattr(D1, "D1_ENGINE", D1.audit_duplicates)
    out, _ = instance
    assert _run(out, tmp_path) == 0
    cert = json.loads(
        (tmp_path / "d1out" / "d1_autond_certificate.json").read_text())
    assert cert["production_status"] == "REDUCED_SCALE_SMOKE"


# ---------------------------------------------------------------------
# 5d. Gap 1 — every required configuration field is MANDATORY
# ---------------------------------------------------------------------
# Three fields were fail-open in r3: `delta_plain_verified` was rejected only
# when explicitly False, `label_values` was checked only when present, and a
# missing sample-count declaration passed. An absent assurance is not an
# assurance.

def test_missing_delta_plain_verified_is_refused(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    del man["specification"]["delta_plain_verified"]
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert "delta_plain_verified is ABSENT" in str(exc.value)
    assert spy.calls == 0


def test_missing_label_values_is_refused(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    del man["specification"]["label_values"]
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert "label_values is ABSENT" in str(exc.value)
    assert spy.calls == 0


@pytest.mark.parametrize("part", ["train", "validation", "test"])
def test_missing_declared_sample_count_is_refused(instance, tmp_path, spy,
                                                  part):
    out, _ = instance
    man = _read(out)
    del man["specification"][f"{part}_samples"]
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert f"{part}_samples is ABSENT" in str(exc.value)
    assert spy.calls == 0


@pytest.mark.parametrize("bad", [0, -5, "400", 400.0, True])
def test_invalid_declared_sample_count_is_refused(instance, tmp_path, spy,
                                                  bad):
    out, _ = instance
    man = _read(out)
    man["specification"]["train_samples"] = bad
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert spy.calls == 0
    assert "REFUSED" in str(exc.value)


def test_non_positive_recorded_partition_size_is_refused(instance, tmp_path,
                                                         spy):
    out, _ = instance
    man = _read(out)
    man["partitions"]["train"]["samples"] = 0
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert spy.calls == 0
    assert "REFUSED" in str(exc.value)


def test_malformed_label_values_is_refused(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    man["specification"]["label_values"] = "binary"
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert spy.calls == 0
    assert "label_values" in str(exc.value)


def test_missing_feature_dtype_is_refused(instance, tmp_path, spy):
    out, _ = instance
    man = _read(out)
    del man["specification"]["feature_dtype"]
    _write(out, _reseal(man))
    with pytest.raises(SystemExit, match="feature_dtype"):
        _run(out, tmp_path)
    assert spy.calls == 0


def test_a_reduced_instance_cannot_be_labelled_production(instance, tmp_path,
                                                          spy):
    """Declared production counts over reduced-scale arrays.

    The flag route was already closed; this closes the route of declaring
    production sample counts in the specification. D0's spec-to-array check
    catches the contradiction, and D1 refuses before the engine.
    """
    out, _ = instance
    man = _read(out)
    man["specification"]["reduced_scale"] = False
    for part, n in D1.PRODUCTION_SCALE.items():
        man["specification"][f"{part}_samples"] = n
    _write(out, _reseal(man))
    with pytest.raises(SystemExit) as exc:
        _run(out, tmp_path)
    assert spy.calls == 0
    assert "REFUSED" in str(exc.value)
    assert not (tmp_path / "d1out" / "d1_autond_certificate.json").exists()


def test_legitimate_reduced_scale_counts_are_accepted(instance, tmp_path,
                                                      monkeypatch):
    """Reduced-scale testing must keep working.

    Declared reduced counts that agree with the reduced arrays are accepted,
    and classified REDUCED_SCALE_SMOKE rather than being required to meet
    production sizes.
    """
    monkeypatch.setattr(D1, "D1_ENGINE", D1.audit_duplicates)
    out, man = instance
    assert man["specification"]["train_samples"] == 400
    assert _run(out, tmp_path) == 0
    cert = json.loads(
        (tmp_path / "d1out" / "d1_autond_certificate.json").read_text())
    assert cert["production_status"] == "REDUCED_SCALE_SMOKE"
    assert cert["scale_classification"]["actual_samples"]["train"] == 400


# ---------------------------------------------------------------------
# 6. No certificate is written when preflight fails
# ---------------------------------------------------------------------

def test_no_certificate_is_written_on_a_failed_preflight(instance, tmp_path,
                                                         spy):
    out, _ = instance
    man = _read(out)
    man["canonical_instance_id"] = "0" * 64
    _write(out, man)
    outdir = tmp_path / "d1out"
    with pytest.raises(SystemExit):
        D1.main(["--instance-dir", str(out), "--out-dir", str(outdir)])
    assert spy.calls == 0
    assert not (outdir / "d1_autond_certificate.json").exists(), (
        "a certificate was written for an instance that did not verify")


# ---------------------------------------------------------------------
# 7. Production status is measured, not declared
# ---------------------------------------------------------------------

def test_reduced_instance_is_classified_reduced(instance):
    out, man = instance
    scale = D1.classify_scale(man)
    assert scale["production_status"] == "REDUCED_SCALE_SMOKE"
    assert scale["actual_samples"]["train"] == 400
    assert "not from any specification flag" in scale["basis"]


def test_clearing_the_reduced_flag_does_not_promote_to_production(instance):
    """The core property: a flag cannot manufacture production evidence."""
    out, _ = instance
    man = _read(out)
    man["specification"]["reduced_scale"] = False
    man["specification"].pop("reduced_scale_note", None)
    scale = D1.classify_scale(man)
    assert scale["production_status"] == "REDUCED_SCALE_SMOKE", (
        "clearing the reduced_scale flag promoted a 400-row partition to "
        "production evidence")
    assert "flag_disagreement" in scale


def test_a_hand_edited_flag_is_reported_as_inconsistent(instance):
    out, _ = instance
    man = _read(out)
    man["specification"]["reduced_scale"] = False
    scale = D1.classify_scale(man)
    assert "measured counts decide" in scale["flag_disagreement"]


def test_full_scale_requires_every_partition_to_meet_the_frozen_config(
        instance):
    out, _ = instance
    man = _read(out)
    for p in C.PARTITIONS:
        man["partitions"][p]["samples"] = D1.PRODUCTION_SCALE[p]
    assert D1.classify_scale(man)["production_status"] == "FULL_SCALE"

    # One partition short of the frozen configuration is not production.
    man["partitions"]["validation"]["samples"] = (
        D1.PRODUCTION_SCALE["validation"] - 1)
    scale = D1.classify_scale(man)
    assert scale["production_status"] == "REDUCED_SCALE_SMOKE"
    assert scale["partitions_meeting_production_scale"]["validation"] is False


def test_frozen_production_configuration_is_the_declared_one(instance):
    assert D1.PRODUCTION_SCALE == {"train": 10_000_000,
                                   "validation": 1_000_000,
                                   "test": 1_000_000}


def test_certificate_records_the_measured_classification(instance, tmp_path,
                                                         monkeypatch):
    monkeypatch.setattr(D1, "D1_ENGINE", D1.audit_duplicates)
    out, _ = instance
    outdir = tmp_path / "d1out"
    assert D1.main(["--instance-dir", str(out), "--out-dir", str(outdir)]) == 0
    cert = json.loads(
        (outdir / "d1_autond_certificate.json").read_text())
    assert cert["production_status"] == "REDUCED_SCALE_SMOKE"
    assert cert["scale_classification"]["actual_samples"]["train"] == 400
    assert cert["d0_verification"]["verified"] is True
    assert "BEFORE the D1 engine was invoked" in cert["d0_verification"][
        "ordering"]
    assert cert["canonical_instance"]["identity_recomputed"] is True
    assert "not independent authentication" in " ".join(
        cert["scope_limitations"])
