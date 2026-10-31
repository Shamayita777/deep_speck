"""core: hypothesis direction, numerics, theory-consistency measurement, approval records, registry."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from audit.cryptography.core import approval as AP
from audit.cryptography.core import numerics as N
from audit.cryptography.core.capabilities import CapabilityError, TheoryConsistencyCapability
from audit.cryptography.core.hypothesis import AssociationStatus, HypothesisDirection, from_legacy
from audit.cryptography.core.theory_consistency import TheoryConsistencyMeasurement, UndefinedAssociation


# --------------------------------------------------------------- hypothesis


def test_two_sided_is_not_representable_by_the_legacy_enum():
    from audit.cryptography.test.base import EvidenceDirection
    assert {from_legacy(d) for d in EvidenceDirection} == {HypothesisDirection.POSITIVE, HypothesisDirection.NEGATIVE}
    assert HypothesisDirection.TWO_SIDED.supports(-0.1) and HypothesisDirection.TWO_SIDED.supports(0.1)
    assert not HypothesisDirection.TWO_SIDED.supports(0.0)
    assert HypothesisDirection.POSITIVE.supports(0.1) and not HypothesisDirection.POSITIVE.supports(-0.1)
    with pytest.raises(ValueError):
        HypothesisDirection.NO_DIRECTIONAL_CLAIM.supports(0.1)


def test_legacy_theory_consistency_test_still_encodes_a_positive_hypothesis():
    """Documents the legacy behaviour that core.theory_consistency replaces (the file is manifest-bound)."""
    from audit.cryptography.test.ce2.theory_consistency import TheoryConsistencyTest
    t = TheoryConsistencyTest()
    assert t.evidence_direction.value == "HIGHER_IS_BETTER" and "positively" in t.hypothesis


# ------------------------------------------------------------------ numerics


def test_normal_tail_continuity_across_the_underflow_boundary():
    for z in (5.0, 20.0, 37.0, 37.5):
        exact = math.log(0.5 * math.erfc(z / math.sqrt(2)))
        assert N.log_normal_upper_tail(z) == pytest.approx(exact, rel=1e-12)
    a, b = N.log_normal_upper_tail(38.4), N.log_normal_upper_tail(38.6)
    assert b < a and N.log_normal_upper_tail(1000.0) == pytest.approx(-500000 - math.log(1000) - 0.5 * math.log(2 * math.pi), rel=1e-9)


def test_pvalue_states():
    assert N.normal_two_sided(0.0, method="m").value == 1.0
    u = N.normal_two_sided(60.0, method="m")
    assert u.status == "UNDERFLOW" and u.value is None and u.log10 < -700 and u.rejects(0.05)
    d = N.normal_two_sided(float("inf"), method="m")
    assert d.status == "DEGENERATE" and not d.rejects(0.05)
    assert N.correlation_t_approximation(1.0, 50).status == "DEGENERATE"
    assert N.correlation_t_approximation(float("nan"), 50).status == "UNDEFINED"
    assert N.correlation_t_approximation(0.3, 2).status == "UNDEFINED"


def test_exact_pvalue_is_never_zero():
    p = N.exact_pvalue(N.exact_binomial_half_cdf(0, 2000), "sign")
    # 2^-2000 is exact but not representable: reported as UNDERFLOW with its exact log10, never 0.0
    assert p.status == "UNDERFLOW" and p.value is None and p.log10 == pytest.approx(-2000 * math.log10(2))
    assert p.rejects(0.05) and "exact rational" in p.note
    q = N.exact_pvalue(N.exact_binomial_half_cdf(0, 20), "sign")
    assert q.status == "EXACT" and q.value == 2 ** -20 and "1/1048576" in q.note


# ---------------------------------------------------- theory-consistency measurement


class _Adapter:
    def __init__(self, ref, pred):
        self.ref, self.pred = np.asarray(ref, float), np.asarray(pred, float)

    def load(self):
        return "model"

    def generate_theory_dataset(self):
        return "dataset"

    def compute_theoretical_reference(self, d):
        return self.ref

    def compute_model_predictions(self, m, d):
        return self.pred


def _coef(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise UndefinedAssociation(AssociationStatus.UNDEFINED, "constant input")
    return float(np.corrcoef(x, y)[0, 1])


def test_measurement_records_direction_and_never_fabricates_zero():
    m = TheoryConsistencyMeasurement(HypothesisDirection.TWO_SIDED, _coef)
    r = m.run(_Adapter([1, 2, 3, 4], [4, 3, 2, 1]))
    assert r.defined and r.statistic == pytest.approx(-1.0) and r.to_dict()["hypothesis"] == "TWO_SIDED"
    assert r.within_sample_p.status == "DEGENERATE"
    u = m.run(_Adapter([1, 2, 3, 4], [5, 5, 5, 5]))
    assert not u.defined and u.statistic is None and u.status is AssociationStatus.UNDEFINED
    assert u.within_sample_p.status == "UNDEFINED"


def test_capability_is_required():
    m = TheoryConsistencyMeasurement(HypothesisDirection.TWO_SIDED, _coef)
    with pytest.raises(CapabilityError):
        m.run(object())
    assert isinstance(_Adapter([1], [1]), TheoryConsistencyCapability)
    with pytest.raises(TypeError):
        TheoryConsistencyMeasurement("TWO_SIDED", _coef)


# ------------------------------------------------------------------- approval

TARGET = {"cipher": "X", "rounds": 5}
DECISIONS = {"n_runs": 20, "ladder": [2, 4, 8]}
SOURCES = {"a.py": "1" * 64}


def _approved(**over):
    r = AP.template(experiment_id="EXP", target=TARGET, design_version="v", design_sha256="a" * 64,
                    required_decisions=DECISIONS, source_sha256=SOURCES)
    r.update(status="APPROVED_FOR_PRODUCTION", approved_by="Accountable Person", approved_role="PI",
             approved_utc=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())
    r.update(over)
    return r


def _check(rec, **kw):
    args = dict(experiment_id="EXP", target=TARGET, design_sha256="a" * 64, required_decisions=DECISIONS,
                source_sha256=SOURCES)
    args.update(kw)
    return AP.check(rec, **args)


def test_valid_unsigned_approval_is_authorised_with_honest_assurance():
    c = _check(_approved())
    assert c.authorised and c.assurance.startswith("UNSIGNED_LOCAL_RECORD") and "not a cryptographic" in c.assurance


@pytest.mark.parametrize("case,rec,kw,needle", [
    ("missing", None, {}, "no approval record"),
    ("pending", "template", {}, "APPROVAL_PENDING"),
    ("incomplete", {"approved_by": ""}, {}, "approved_by is empty"),
    ("other_hash", {}, {"design_sha256": "b" * 64}, "does not match the current design"),
    ("other_experiment", {"experiment_id": "OTHER"}, {}, "experiment_id"),
    ("other_target", {}, {"target": {"cipher": "Y"}}, "target"),
    ("invalidated", {"invalidated": {"utc": "now", "by": "x", "reason": "design changed"}}, {}, "invalidated"),
    ("unresolved", {"decisions_resolved": {"n_runs": 20}}, {}, "unresolved"),
    ("wrong_value", {"decisions_resolved": {"n_runs": 10, "ladder": [2, 4, 8]}}, {}, "frozen design uses"),
    ("source_changed", {}, {"source_sha256": {"a.py": "2" * 64}}, "differs from the hash"),
    ("future", {"approved_utc": "2999-01-01T00:00:00+00:00"}, {}, "future"),
    ("naive_time", {"approved_utc": "2026-10-01T00:00:00"}, {}, "timezone"),
    ("no_signature_block", {"signature": None}, {}, "signature block"),
])
def test_production_cannot_start_without_a_valid_approval(case, rec, kw, needle):
    if rec == "template":
        record = AP.template(experiment_id="EXP", target=TARGET, design_version="v", design_sha256="a" * 64,
                             required_decisions=DECISIONS, source_sha256=SOURCES)
    elif rec is None:
        record = None
    else:
        record = _approved(**rec)
    c = _check(record, **kw)
    assert not c.authorised, case
    assert any(needle in p for p in c.problems), (case, c.problems)


def test_signed_records_are_not_claimed_verified():
    c = _check(_approved(signature={"type": "openpgp-detached", "value": "..."}))
    assert c.authorised and c.assurance.startswith("SIGNATURE_PRESENT_NOT_VERIFIED")


def test_template_never_contains_an_identity():
    t = AP.template(experiment_id="EXP", target=TARGET, design_version="v", design_sha256="a" * 64,
                    required_decisions=DECISIONS, source_sha256=SOURCES)
    assert t["status"] == "APPROVAL_PENDING" and not t["approved_by"] and not t["approved_utc"]
