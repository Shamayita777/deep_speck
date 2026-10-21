"""
Regression tests for the generic target resolver and binding status in
audit.dataset.canonical.

Scientific status
-----------------
NON-EVIDENTIARY.

Why these exist
---------------
Two generic changes were made to `canonical.py` so a second audited target
could exist at all. Both are small, and both touch code the Gohr evidence
depends on, so each is pinned here:

1. `_resolve_generator(num_rounds)` became `_resolve_generator(spec)` with a
   registry keyed on `spec["target"]`, defaulting to `gohr-speck`. The risk is
   that the Gohr path stops being the default, or that an unrecognised target
   silently generates the wrong dataset.

2. `instance_binding_status` became `_binding_status(spec)`. Schema 1
   hardcoded the claim that D1-D4 "audited its own unpersisted instance",
   which is FALSE for a fresh target that has no prior certificates at all.
   The risk is that a fresh target either keeps making that false claim, or
   that Gohr loses the true one.

Nothing in the repository parses `instance_binding_status` programmatically:
it is a manifest field for a reader. `NOT_RUN` is also used as an unrelated
field's status in `d2_gohr_sample_dependence.py`; that field is untouched and
is not what these tests are about.
"""
from __future__ import annotations

import pytest

from audit.dataset import canonical as C

DIMENSIONS = ("D1", "D2", "D3", "D4", "D5")


# ---------------------------------------------------------------------
# 1. Target resolution
# ---------------------------------------------------------------------

def test_default_target_is_gohr():
    assert C.DEFAULT_TARGET == "gohr-speck"


def test_spec_without_a_target_takes_the_gohr_path(monkeypatch):
    """The backward-compatibility guarantee, asserted directly.

    The resolver is exercised through the registry rather than by generating
    data: this test is about WHICH resolver is chosen.
    """
    chosen = {}

    def spy(spec):
        chosen["target"] = "gohr-speck"
        chosen["num_rounds"] = spec["num_rounds"]
        return (lambda n: (None, None)), "spy-gohr"

    monkeypatch.setitem(C.GENERATOR_RESOLVERS, "gohr-speck", spy)
    _, path = C._resolve_generator({"num_rounds": 5})
    assert path == "spy-gohr"
    assert chosen["target"] == "gohr-speck"
    assert chosen["num_rounds"] == 5


def test_explicit_gohr_target_resolves_through_the_same_path(monkeypatch):
    calls = []
    monkeypatch.setitem(
        C.GENERATOR_RESOLVERS, "gohr-speck",
        lambda spec: (calls.append(spec), ((lambda n: None), "spy-gohr"))[1])
    _, path = C._resolve_generator({"target": "gohr-speck", "num_rounds": 5})
    assert path == "spy-gohr"
    assert len(calls) == 1


def test_explicit_autond_target_resolves_through_its_own_path(monkeypatch):
    monkeypatch.setitem(
        C.GENERATOR_RESOLVERS, "autond",
        lambda spec: ((lambda n: None), "spy-autond"))
    _, path = C._resolve_generator({"target": "autond"})
    assert path == "spy-autond"


def test_both_targets_are_registered():
    assert set(C.GENERATOR_RESOLVERS) >= {"gohr-speck", "autond"}


def test_unknown_target_fails_safely():
    with pytest.raises(SystemExit) as exc:
        C._resolve_generator({"target": "no-such-target", "num_rounds": 5})
    msg = str(exc.value)
    assert "REFUSED" in msg
    assert "no-such-target" in msg
    # The refusal must list what IS registered, so the failure is actionable.
    assert "gohr-speck" in msg


def test_unknown_target_does_not_fall_back_to_a_default():
    """Silently generating the wrong dataset is the failure to avoid."""
    with pytest.raises(SystemExit):
        C._resolve_generator({"target": "", "num_rounds": 5})


def test_autond_resolver_refuses_an_incomplete_spec():
    """An AutoND instance is identified by five fields; none may be implicit."""
    with pytest.raises(SystemExit) as exc:
        C._resolve_generator({"target": "autond"})
    msg = str(exc.value)
    assert "REFUSED" in msg
    for field in ("autond_root", "cipher", "delta_plain_hex", "num_rounds"):
        assert field in msg


# ---------------------------------------------------------------------
# 2. Binding status
# ---------------------------------------------------------------------

def test_gohr_keeps_the_true_not_bound_claim():
    """Gohr D1-D4 certificates DO exist over unpersisted data."""
    status = C._binding_status({"target": "gohr-speck"})
    for dim in ("D1", "D2", "D3", "D4"):
        assert status[dim].startswith("NOT_BOUND"), (
            f"{dim} lost the NOT_BOUND claim, which is true for Gohr")
        assert "unpersisted" in status[dim]
    assert status["D5"] == "NOT_RUN"


def test_spec_without_a_target_keeps_gohr_binding_status():
    """Default behaviour must be unchanged for an untargeted spec."""
    assert C._binding_status({}) == C._binding_status({"target": "gohr-speck"})


def test_fresh_target_does_not_claim_prior_audits_ran():
    """The factual correction this change exists for.

    AutoND has no prior D-stage certificates, so there is nothing to be
    unbound FROM. Claiming "D1 audited its own unpersisted instance" would
    assert an audit that never happened.
    """
    status = C._binding_status({"target": "autond"})
    for dim in DIMENSIONS:
        assert status[dim] == "NOT_RUN", (
            f"{dim} is {status[dim]!r} for a fresh target; a target with no "
            f"certificates must not claim prior audits")
        assert "NOT_BOUND" not in status[dim]
        assert "unpersisted" not in status[dim]


def test_unregistered_target_is_treated_as_fresh():
    status = C._binding_status({"target": "some-future-target"})
    assert all(status[d] == "NOT_RUN" for d in DIMENSIONS)


def test_every_dimension_is_reported_for_every_target():
    for target in ("gohr-speck", "autond", "unknown"):
        status = C._binding_status({"target": target})
        for dim in DIMENSIONS:
            assert dim in status, f"{dim} missing for target {target}"
        assert "note" in status


def test_note_explains_both_statuses():
    note = C._binding_status({"target": "gohr-speck"})["note"]
    assert "NOT_BOUND" in note and "NOT_RUN" in note
    assert "retroactive" in note.lower()


def test_prior_unbound_registry_covers_only_gohr():
    """A new target must not inherit Gohr's history by accident."""
    assert set(C.PRIOR_UNBOUND_DIMENSIONS) == {"gohr-speck"}
    assert "autond" not in C.PRIOR_UNBOUND_DIMENSIONS


def test_existing_certificates_are_not_presented_as_bound_evidence():
    """NOT_BOUND must never read as bound.

    The status strings are what a reader of a manifest sees. Neither value may
    be mistakable for an affirmative binding claim.
    """
    for target in ("gohr-speck", "autond"):
        for dim in DIMENSIONS:
            value = C._binding_status({"target": target})[dim]
            assert value.startswith(("NOT_RUN", "NOT_BOUND"))
            assert "BOUND -" not in value.replace("NOT_BOUND -", "")


# ---------------------------------------------------------------------
# 3. The limitations text must describe the actual generator
# ---------------------------------------------------------------------

def test_limitations_text_is_spec_driven_not_hardcoded(tmp_path, monkeypatch):
    """Schema 1 hardcoded "speck.make_train_data" into the limitations.

    That string was factually wrong in an AutoND manifest, whose generator is
    `main.make_train_data`.
    """
    import numpy as np

    def resolver(spec):
        def gen(n):
            rng = np.random.default_rng(7)
            return (rng.integers(0, 2, size=(n, 4), dtype=np.uint8),
                    rng.integers(0, 2, size=n, dtype=np.uint8))
        return gen, "spy.generator"

    monkeypatch.setitem(C.GENERATOR_RESOLVERS, "spy-target", resolver)
    spec = {
        "target": "spy-target", "dataset_id": "spy",
        "generator": "spy.make_train_data",
        "randomness_source": "spy.urandom",
        "train_samples": 8, "validation_samples": 4, "test_samples": 4,
    }
    man = C.build(tmp_path / "d0", repo_root=tmp_path, spec=spec,
                  log=lambda *a, **k: None)
    text = man["limitations"][0]
    assert "spy.make_train_data" in text
    assert "spy.urandom" in text
    assert "speck.make_train_data" not in text
