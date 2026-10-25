"""
Shared D4 engine against a second, non-Gohr implementation.

``ThresholdAdapter`` is a deterministic learner on generic binary features
(no Keras, no cipher, no Gohr data generation).  These tests show that the
engine's contract, provenance checks, perturbation verification, decision
rules and certificate recomputation work for it.  They do not show that any
other real target works; that needs its own adapter and tests.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from audit.dataset import d4_controlled_perturbation as d4
from audit.dataset import integrity as I
from audit.dataset.integrity import IntegrityError, labelled_partition_digest
from audit.dataset.perturbations.label_shuffle import LabelShufflePerturbation

CONFIG = "c" * 64


class ThresholdAdapter:
    """Learns 'label = feature 0' if that rule fits the training labels, else predicts 0."""

    def __init__(self, seed, tamper=None):
        self.seed, self.tamper, self.rule, self.selection = seed, tamper, False, None

    def train(self, x, y, *, checkpoint_path, state_path, replicate, condition, run_config_hash):
        x, y = np.asarray(x), np.asarray(y).reshape(-1)
        self.rule = bool(np.mean(x[:, 0] == y) > 0.9)
        self.selection = {"condition": condition, "replicate": replicate, "seed": self.seed,
                          "config_hash": run_config_hash, "train_digest": labelled_partition_digest(x, y),
                          "rule": "single_fit", "selected_epoch": 1, "checkpoint_sha256": "n/a"}
        if self.tamper:
            self.tamper(self.selection, condition)
        return self

    def predict(self, x):
        x = np.asarray(x)
        return x[:, 0].astype(float) if self.rule else np.zeros(len(x))


def data(n=600, seed=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(np.uint8)
    x = np.stack([y, rng.integers(0, 2, size=n).astype(np.uint8)], axis=1)
    return x, y


def run(tmp_path, factory=ThresholdAdapter, rule="directional_iut_v2", replicates=3, **kw):
    x, y = data()
    tx, ty = data(seed=4)
    return d4.run_d4(perturbation=kw.pop("perturbation", LabelShufflePerturbation()),
                     train_features=x, train_labels=y, test_features=tx, test_labels=ty,
                     adapter_factory=factory, checkpoint_root=tmp_path / "fits", replicates=replicates,
                     audit_seed=0, effect_threshold=0.01, bootstrap_replicates=1000, alpha=0.05,
                     run_config_hash=CONFIG, decision_rule=rule, **kw)


def certificate_of(result):
    return d4.generate_certificate(result, dataset_id="synthetic", dataset_version="1",
                                   generation_procedure="numpy", generation_parameters={},
                                   audit_seed=0, confidence_level=0.95, alpha=0.05)


# ---------------------------------------------------------------- genericity


def test_engine_imports_no_target_code():
    code = ("import sys, audit.dataset.d4_controlled_perturbation, audit.dataset.integrity, "
            "audit.dataset.perturbations.label_shuffle; "
            "bad = [m for m in sys.modules if m.split('.')[0] in "
            "('keras','tensorflow','speck','train_nets') or 'gohr' in m]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_second_implementation_detects_effect_and_certificate_recomputes(tmp_path):
    result = run(tmp_path)
    assert result.effect_detected
    certificate = certificate_of(result)
    assert d4.recompute_certificate_decision(certificate)["outcome"] == "EFFECT_DETECTED"
    rows = certificate["findings"]["replicates"]
    for r in rows:
        assert r["baseline_selection"]["condition"] == "baseline"
        assert r["perturbed_selection"]["condition"] == "perturbed"
        assert r["baseline_selection"]["train_digest"] == certificate["settings"]["source_train_digest"]
        assert r["perturbed_selection"]["train_digest"] == r["perturbation_record"]["perturbed_train_digest"]
        assert r["perturbed_selection"]["train_digest"] != r["baseline_selection"]["train_digest"]
    assert len({r["seed"] for r in rows}) == len(rows)


@pytest.mark.parametrize("rule", d4.DECISION_RULES)
def test_identity_perturbation_yields_no_effect(tmp_path, rule):
    class Identity(d4.Perturbation):
        label_only = True

        def __init__(self):
            super().__init__(name="identity", description="no change")

        def apply(self, x, y, *, rng):
            return x, np.asarray(y).copy()

        def verify(self, x, y, px, py):
            assert np.array_equal(y, py)

    extra = ({"chance_margin": 0.05, "paired_endpoint_role": "secondary", "baseline_adequacy_threshold": 0.9}
             if rule == d4.NEAR_CHANCE_RULE else {})
    result = run(tmp_path, rule=rule, perturbation=Identity(), **extra)
    assert not result.effect_detected
    if rule == d4.NEAR_CHANCE_RULE:
        # The signal survives an identity "perturbation", so accuracy is far above the band.
        assert result.decision["outcome"] == "NOT_SUPPORTED"


# ---------------------------------------------------------------- contract violations


def test_adapter_without_selection_is_refused(tmp_path):
    class NoSelection(ThresholdAdapter):
        def train(self, *a, **k):
            super().train(*a, **k)
            self.selection = None
            return self

    with pytest.raises(IntegrityError, match="adapter contract"):
        run(tmp_path, factory=NoSelection)


@pytest.mark.parametrize("field, value", [
    ("condition", "perturbed"), ("replicate", 99), ("seed", 1), ("config_hash", "x" * 64),
    ("train_digest", "0" * 64),
])
def test_misattributed_selection_is_refused(tmp_path, field, value):
    def tamper(selection, condition):
        if condition == "baseline":
            selection[field] = value

    with pytest.raises(IntegrityError, match=field):
        run(tmp_path, factory=lambda seed: ThresholdAdapter(seed, tamper))


def test_selection_missing_required_field_is_refused(tmp_path):
    with pytest.raises(IntegrityError, match="lacks required fields"):
        run(tmp_path, factory=lambda seed: ThresholdAdapter(seed, lambda s, c: s.pop("checkpoint_sha256")))


def test_perturbation_without_verify_is_refused(tmp_path):
    class Unverifiable(d4.Perturbation):
        def __init__(self):
            super().__init__(name="u", description="u")

        def apply(self, x, y, *, rng):
            return x, y

    with pytest.raises(TypeError, match="verify"):
        run(tmp_path, perturbation=Unverifiable())


def test_missing_run_config_hash_is_refused(tmp_path):
    x, y = data()
    with pytest.raises(ValueError, match="run_config_hash"):
        d4.run_d4(perturbation=LabelShufflePerturbation(), train_features=x, train_labels=y,
                  test_features=x, test_labels=y, adapter_factory=ThresholdAdapter,
                  checkpoint_root=tmp_path, replicates=2, audit_seed=0, effect_threshold=0.01)


# ---------------------------------------------------------------- perturbation provenance


def sidecar(tmp_path, r=1):
    return tmp_path / "fits" / "perturbations" / f"replicate_{r:02d}.perturbation.json"


def test_resume_reuses_verified_perturbation(tmp_path):
    first = run(tmp_path)
    second = run(tmp_path)
    assert [r.perturbation_record for r in first.replicates] == [r.perturbation_record for r in second.replicates]


def indices_path(tmp_path, r=2):
    return tmp_path / "fits" / "perturbations" / f"replicate_{r:02d}_indices.npy"


def rewrite_indices(tmp_path, indices, r=2, fix_digest=True):
    """Replace the persisted indices; optionally re-record their SHA-256 so only
    the structural checks (not the digest) can catch the change."""
    np.save(indices_path(tmp_path, r), indices)
    if fix_digest:
        path = sidecar(tmp_path, r)
        record = json.loads(path.read_text())
        record["files"]["indices"]["sha256"] = I.sha256_file(indices_path(tmp_path, r))
        record["files"]["indices"]["dtype"] = np.asarray(indices).dtype.str
        record["files"]["indices"]["shape"] = list(np.asarray(indices).shape)
        path.write_text(json.dumps(record))


def test_persisted_indices_are_a_verified_int64_bijection(tmp_path):
    run(tmp_path)
    indices = np.load(indices_path(tmp_path))
    assert indices.dtype == np.dtype("<i8") and len(indices) == 600
    assert np.array_equal(np.sort(indices), np.arange(600))
    record = json.loads(sidecar(tmp_path, 2).read_text())
    assert record["kind"] == "permutation_indices" and record["condition"] == "perturbed"
    assert record["seed"] == run(tmp_path).replicates[1].seed     # resume reuses them


def test_tampered_indices_digest_is_refused(tmp_path):
    run(tmp_path)
    indices = np.load(indices_path(tmp_path))
    indices[[0, 1]] = indices[[1, 0]]
    rewrite_indices(tmp_path, indices, fix_digest=False)
    with pytest.raises(IntegrityError, match="digest"):
        run(tmp_path)


@pytest.mark.parametrize("make, match", [
    (lambda i: np.r_[i[:-1], i[0]], "bijection"),                    # duplicate index
    (lambda i: i.astype(np.int32), "dtype"),                           # wrong dtype
    (lambda i: i[:-1], "shape"),                                       # wrong length
    (lambda i: np.r_[i[:-1], 600], "outside"),                         # out of range
    (lambda i: i.astype(">i8"), "dtype"),                              # wrong byte order
    (lambda i: np.roll(i, 1), "regenerates"),                          # valid bijection, not the planned one
])
def test_structurally_invalid_or_unplanned_indices_are_refused(tmp_path, make, match):
    run(tmp_path)
    rewrite_indices(tmp_path, make(np.load(indices_path(tmp_path))))
    with pytest.raises(IntegrityError, match=match):
        run(tmp_path)


def test_bijection_check_directly():
    I.verify_permutation_indices(np.array([2, 0, 1], dtype="<i8"), 3)
    I.verify_permutation_indices(np.zeros(0, dtype="<i8"), 0)
    with pytest.raises(IntegrityError, match="bijection"):
        I.verify_permutation_indices(np.array([0, 0, 2], dtype="<i8"), 3)


def test_indices_bound_to_training_labels(tmp_path):
    run(tmp_path)
    x, y = data()
    other = y.copy()
    other[[0, 1, 2]] = 1 - other[[0, 1, 2]]
    tx, ty = data(seed=4)
    with pytest.raises(IntegrityError, match="source_train_digest|source_label_digest"):
        d4.run_d4(perturbation=LabelShufflePerturbation(), train_features=x, train_labels=other,
                  test_features=tx, test_labels=ty, adapter_factory=ThresholdAdapter,
                  checkpoint_root=tmp_path / "fits", replicates=3, audit_seed=0, effect_threshold=0.01,
                  bootstrap_replicates=1000, alpha=0.05, run_config_hash=CONFIG,
                  decision_rule="directional_iut_v2")


class FlipFirstLabel(d4.Perturbation):
    """Generic (non-index) label-only perturbation, to exercise the label-file path."""

    label_only = True

    def __init__(self):
        super().__init__(name="flip_first", description="flip label 0")

    def apply(self, x, y, *, rng):
        y = np.asarray(y).copy()
        y[0] = 1 - y[0]
        return x, y

    def verify(self, x, y, px, py):
        if int(np.sum(np.asarray(y) != np.asarray(py))) != 1:
            raise ValueError("flip_first must change exactly one label")


def test_generic_label_file_path_persists_and_detects_tampering(tmp_path):
    run(tmp_path, perturbation=FlipFirstLabel())
    record = json.loads(sidecar(tmp_path).read_text())
    assert record["kind"] == "labels"
    run(tmp_path, perturbation=FlipFirstLabel())                  # verified reuse
    path = tmp_path / "fits" / "perturbations" / "replicate_01_labels.npy"
    labels = np.load(path)
    labels[5] = 1 - labels[5]
    np.save(path, labels)
    with pytest.raises(IntegrityError, match="digest"):
        run(tmp_path, perturbation=FlipFirstLabel())


def test_label_shuffle_definition_check():
    x, y = data()
    shuffle = LabelShufflePerturbation()
    _, shuffled = shuffle.apply(x, y, rng=np.random.default_rng(0))
    shuffle.verify(x, y, x, shuffled)
    with pytest.raises(ValueError, match="permutation"):
        shuffle.verify(x, y, x, np.ones_like(y))
    with pytest.raises(ValueError, match="feature matrix"):
        shuffle.verify(x, y, 1 - x, shuffled)
    with pytest.raises(ValueError, match="dtype"):
        shuffle.verify(x, y, x, shuffled.astype(np.int64))


# ---------------------------------------------------------------- decision and certificate


def rows(*pairs, base=0.93, pert=0.50):
    return [{"baseline_score": base, "perturbed_score": pert, "mcnemar_b": b, "mcnemar_c": c} for b, c in pairs]


def decide(r, rule):
    return d4.evaluate_decision(r, rule=rule, alpha=0.05, effect_threshold=0.01, audit_seed=0,
                                bootstrap_replicates=1000, confidence_level=0.95)


def test_directional_rule_rejects_an_improvement_that_two_sided_rule_accepts():
    improving = rows((400, 10), (410, 12), (390, 9), base=0.50, pert=0.93)
    assert decide(improving, "conservative_two_sided_v1")["effect_detected"] is True
    assert decide(improving, "directional_iut_v2")["effect_detected"] is False


def test_directional_rule_needs_every_replicate():
    one_flat = rows((10, 400), (12, 410), (200, 200))
    assert decide(one_flat, "directional_iut_v2")["effect_detected"] is False
    assert decide(rows((10, 400), (12, 410), (9, 390)), "directional_iut_v2")["effect_detected"] is True


def test_one_sided_pvalue_direction():
    assert d4.mcnemar_one_sided_pvalue(0, 20) < 1e-5
    assert d4.mcnemar_one_sided_pvalue(20, 0) == pytest.approx(1.0)
    assert d4.mcnemar_one_sided_pvalue(0, 0) == 1.0


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["findings"]["replicates"][0].update(mcnemar_pvalue=0.5), "McNemar"),
    (lambda c: c["findings"]["replicates"][0].update(perturbed_accuracy=0.9), "difference"),
    (lambda c: c["findings"]["replicates"].pop(), "replicate rows"),
    (lambda c: c["decision"].update(outcome="NO_REPRODUCIBLE_EFFECT_DETECTED"), "recomputed"),
    (lambda c: c["settings"].update(decision_rule="conservative_two_sided_v1"), "recomputed"),
])
def test_certificate_tampering_is_detected(tmp_path, mutate, message):
    certificate = certificate_of(run(tmp_path))
    mutate(certificate)
    with pytest.raises(IntegrityError, match=message):
        d4.recompute_certificate_decision(certificate)


# ---------------------------------------------------------------- preregistration review


def test_shipped_gohr_draft_is_well_formed_and_entirely_pending():
    from audit.dataset.integrity import REPOSITORY_ROOT
    record = json.loads((REPOSITORY_ROOT / "audit/targets/gohr/preregistrations/d4_gohr_label_shuffle.json").read_text())
    review = d4.review_preregistration(record)
    assert review["problems"] == []
    assert review["values"] == {}
    assert {"approved", "approved_by", "approved_utc", "approval_statement",
            "data_examined_before_approval"} <= set(review["pending"])
    assert {f"decisions.{d}" for d in d4.REQUIRED_DECISIONS} <= set(review["pending"])
    with pytest.raises(IntegrityError, match="not approved"):
        d4.require_approved_preregistration(record)


def test_approved_values_outside_options_are_malformed():
    record = {"schema_version": d4.PREREGISTRATION_SCHEMA,
              "decisions": {"decision_rule": {"status": "APPROVED", "value": "made_up",
                                              "options": ["directional_iut_v2"], "rationale": "r"}}}
    assert any("made_up" in p for p in d4.review_preregistration(record)["problems"])
