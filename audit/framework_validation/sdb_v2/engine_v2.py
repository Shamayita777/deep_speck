"""
Claim-decision engine V2 (SDB-V2). The V1 engine (../claim_engine.py) is kept unchanged.

Implements only the manuscript's written rules (IEEE_SP_Claim_Centered_Audit_v2.tex,
SHA-256 190209a7...):

  dimension outcomes  PASS, FAIL, INCONCLUSIVE, N/A (l.927-933); N/A means excluded at
                      decomposition, so a dimension in R_c cannot be N/A (registration error).
                      An absent record is INCONCLUSIVE with reason `missing_prerequisite`
                      (l.856, l.932); absence is never an outcome of its own.
  admissibility       a FAIL counts only if admissible (l.935-944, l.1047-1049).
  L0..L4              Table evidence-levels (l.1003-1020): L1 reproducible predictive evidence
                      above a declared baseline; L2 PASS on every dimension in R_c; L3 a CE test
                      that is preregistered, adequately powered, SUPPORTED and whose criterion
                      no executed adversarial control satisfies; L4 two supporting lines with
                      the same interpretation, different method AND source, and no shared
                      unverified component capable of explaining the result.
  precedence          Table claim-decisions (l.1054-1076), first satisfied row.
  composite           l.1078-1087.
  reproduction-only   R_c empty, L* = L1: SUPPORTED when COMPLETE or SUBSTANTIAL (l.1095-1097).

Where the text does not decide, the result is UNSPECIFIED, never a guess:
  X1  SUBSTANTIAL reproduction for a component that is not reproduction-only;
  X2  PARTIAL reproduction;
  X3  a required dimension whose FAIL is inadmissible: is it an INCONCLUSIVE prerequisite
      (row 2), or merely not PASS (then no row applies when L* <= L1)?
  X4  a required CE test that is INCONCLUSIVE: is it a prerequisite under row 2?
The engine evaluates every reading of the items that are actually open for the component;
an outcome is returned only when all readings agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, product
from typing import Optional

DIMENSIONS = ("II", "DI", "EV")
REPRODUCTION_STATES = ("COMPLETE", "SUBSTANTIAL", "PARTIAL", "NON_REPRODUCIBLE")


class Dim(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    NA = "N/A"


class CE(str, Enum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNSPECIFIED = "UNSPECIFIED"          # the manuscript does not decide this input


class InvalidRegistration(ValueError):
    pass


@dataclass(frozen=True)
class DimRecord:
    outcome: Dim
    admissible: bool = True              # meaningful for FAIL only
    source: Optional[str] = None


@dataclass(frozen=True)
class CETest:
    test_id: str
    outcome: CE
    required: bool = False
    preregistered: bool = False
    adequately_powered: bool = False
    satisfied_by_executed_control: bool = False
    refs: tuple = ()

    @property
    def qualifies_l3(self) -> bool:
        return (self.outcome is CE.SUPPORTED and self.preregistered and self.adequately_powered
                and not self.satisfied_by_executed_control)


@dataclass(frozen=True)
class Line:
    line_id: str
    supports: bool
    interpretation: str
    method: str
    source: str
    unverified_components: frozenset = frozenset()


@dataclass(frozen=True)
class Component:
    claim_id: str
    required: frozenset
    min_level: int
    reproduction: str
    predictive_above_baseline: bool
    dims: dict = field(default_factory=dict)          # dimension -> DimRecord; absent = missing
    ce_tests: tuple = ()
    lines: tuple = ()

    def __post_init__(self):
        if not set(self.required) <= set(DIMENSIONS):
            raise InvalidRegistration(f"{self.claim_id}: unknown dimension in {sorted(self.required)}")
        if not 0 <= self.min_level <= 4:
            raise InvalidRegistration(f"{self.claim_id}: minimum level {self.min_level}")
        if self.reproduction not in REPRODUCTION_STATES:
            raise InvalidRegistration(f"{self.claim_id}: reproduction status {self.reproduction!r}")
        for d in self.required:
            rec = self.dims.get(d)
            if rec is not None and rec.outcome is Dim.NA:
                raise InvalidRegistration(f"{self.claim_id}: required dimension {d} cannot be N/A")


@dataclass
class Decision:
    claim_id: str
    verdict: Verdict
    level: Optional[int]
    rule: str
    blocking: list = field(default_factory=list)
    open_items: list = field(default_factory=list)


def _l4(lines) -> bool:
    sup = [l for l in lines if l.supports]
    return any(a.interpretation == b.interpretation and a.method != b.method and a.source != b.source
               and not (a.unverified_components & b.unverified_components)
               for a, b in combinations(sup, 2))


def _reproduced(c: Component) -> Optional[bool]:
    """True / False, or None when the text does not decide (X1, X2)."""
    if c.reproduction == "COMPLETE":
        return True
    if c.reproduction == "NON_REPRODUCIBLE":
        return False
    if c.reproduction == "SUBSTANTIAL" and not c.required and c.min_level == 1:
        return True                                                   # l.1095-1097
    return None


def _finish(c: Component, *, reproduced: bool, ce_inconclusive_is_prerequisite: bool,
            inadmissible_is_inconclusive: bool):
    req = sorted(c.required)
    inadm = [d for d in req if c.dims.get(d) is not None and c.dims[d].outcome is Dim.FAIL
             and not c.dims[d].admissible]
    req_ce_open = [t.test_id for t in c.ce_tests if t.required and t.outcome is CE.INCONCLUSIVE]
    if not reproduced:
        return Verdict.INCONCLUSIVE, 0, "2: reproduction not established", ["reproduction"]
    if ce_inconclusive_is_prerequisite and req_ce_open:
        return Verdict.INCONCLUSIVE, None, "2: required CE test inconclusive", req_ce_open
    if inadmissible_is_inconclusive and inadm:
        return Verdict.INCONCLUSIVE, None, "2: inadmissible failure left unresolved", inadm
    level = 1 if c.predictive_above_baseline else 0
    if level == 1 and all(c.dims[d].outcome is Dim.PASS for d in req):
        level = 2
        if any(t.qualifies_l3 for t in c.ce_tests):
            level = 3
            if _l4(c.lines):
                level = 4
    if level < c.min_level:
        return Verdict.INCONCLUSIVE, level, "3: level below required", [f"L{level} < L{c.min_level}"]
    if all(c.dims[d].outcome is Dim.PASS for d in req):
        return Verdict.SUPPORTED, level, "4: all requirements met", []
    return None, level, "no row applies", inadm


def decide(c: Component) -> Decision:
    req = sorted(c.required)
    fails = [d for d in req if c.dims.get(d) is not None and c.dims[d].outcome is Dim.FAIL
             and c.dims[d].admissible]
    ce_fail = [t.test_id for t in c.ce_tests if t.required and t.outcome is CE.NOT_SUPPORTED]
    if fails or ce_fail:
        return Decision(c.claim_id, Verdict.NOT_SUPPORTED, None, "1: admissible failure", fails + ce_fail)
    if c.reproduction == "NON_REPRODUCIBLE":
        return Decision(c.claim_id, Verdict.INCONCLUSIVE, 0, "2: reproduction not established", ["reproduction"])
    missing = [d for d in req if d not in c.dims]
    inconclusive = [d for d in req if d in c.dims and c.dims[d].outcome is Dim.INCONCLUSIVE]
    if missing or inconclusive:
        return Decision(c.claim_id, Verdict.INCONCLUSIVE, None, "2: prerequisite open",
                        inconclusive + [f"{d}: missing_prerequisite" for d in missing])

    rep = _reproduced(c)
    open_items = []
    rep_options = [rep] if rep is not None else [True, False]
    if rep is None:
        open_items.append("X1" if c.reproduction == "SUBSTANTIAL" else "X2")
    has_req_ce_open = any(t.required and t.outcome is CE.INCONCLUSIVE for t in c.ce_tests)
    ce_options = [True, False] if has_req_ce_open else [False]
    if has_req_ce_open:
        open_items.append("X4")
    has_inadm = any(c.dims[d].outcome is Dim.FAIL and not c.dims[d].admissible for d in req)
    inadm_options = [True, False] if has_inadm else [False]
    if has_inadm:
        open_items.append("X3")

    results = {(r, x4, x3): _finish(c, reproduced=r, ce_inconclusive_is_prerequisite=x4,
                                    inadmissible_is_inconclusive=x3)
               for r, x4, x3 in product(rep_options, ce_options, inadm_options)}
    verdicts = {v for v, *_ in results.values()}
    if len(verdicts) == 1 and None not in verdicts:
        v, level, rule, blocking = next(iter(results.values()))
        levels = {lv for _, lv, *_ in results.values()}
        return Decision(c.claim_id, v, level if len(levels) == 1 else None, rule, blocking, open_items)
    return Decision(c.claim_id, Verdict.UNSPECIFIED, None, "unspecified: the readings disagree",
                    sorted({str(v.value if v else "no row") for v in verdicts}), open_items)


def composite(verdicts) -> Verdict:
    v = list(verdicts)
    if any(x is Verdict.UNSPECIFIED for x in v):
        raise ValueError("composite of an UNSPECIFIED component is not defined by the manuscript")
    if v and all(x is Verdict.SUPPORTED for x in v):
        return Verdict.SUPPORTED
    if v and all(x is Verdict.NOT_SUPPORTED for x in v):
        return Verdict.NOT_SUPPORTED
    if v and all(x in (Verdict.SUPPORTED, Verdict.NOT_SUPPORTED) for x in v):
        return Verdict.PARTIALLY_SUPPORTED
    return Verdict.INCONCLUSIVE
