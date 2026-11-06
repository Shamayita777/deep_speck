"""
Engine V2 against the independently written oracle, over the state space blocks registered in
SDB_V2_DESIGN_CLEAN_DRAFT.md section 6.4, plus hand-written boundary literals and the paper-verdict
consistency check. No expected value is produced by an engine function.

  E1 exhaustive   per dimension (in R_c or not) x {PASS, FAIL admissible, FAIL inadmissible,
                  INCONCLUSIVE, N/A}; x 4 reproduction states x predictive flag x L* 0..4;
                  x CE configuration (none, or one test over outcome x required x preregistered
                  x powered x control-satisfied)                            1,960,000 cases
  E2 exhaustive   L4 line configurations on an otherwise L3-qualifying component      134
  E3 exhaustive   composite rule over every sequence of 0..4 component verdicts         121
  P  pairwise     strength-2 covering array over the full factor space (2-3 CE tests,
                  0-3 lines, absent records), deterministic greedy construction, seed 20261106
  R  random       100,000 uniform draws over the same factor space, seed 20261107
  B  boundary     literal expectations at every level transition and every open item
"""
from __future__ import annotations

import ast
import random
from itertools import product
from pathlib import Path

import pytest

from audit.framework_validation.sdb_v2 import engine_v2 as E
from audit.framework_validation.sdb_v2.oracle_v2 import oracle, oracle_composite

D = E.DIMENSIONS
OUTS = ("PASS", "FAIL_A", "FAIL_I", "INCONCLUSIVE", "N/A")
REPRO = E.REPRODUCTION_STATES


def dim_record(code):
    if code == "FAIL_A":
        return E.DimRecord(E.Dim.FAIL, admissible=True)
    if code == "FAIL_I":
        return E.DimRecord(E.Dim.FAIL, admissible=False)
    return E.DimRecord(E.Dim(code))


def to_oracle(req, dims, repro, pred, L, ce, lines):
    od = {}
    for d, code in dims.items():
        if code == "ABSENT":
            continue
        od[d] = {"outcome": "FAIL" if code.startswith("FAIL") else code, "admissible": code != "FAIL_I"}
    return {"required": sorted(req), "L": L, "reproduction": repro, "predictive": pred, "dims": od,
            "ce": [{"outcome": o, "required": r, "prereg": p, "powered": w, "control": c} for o, r, p, w, c in ce],
            "lines": [{"supports": s, "interp": i, "method": m, "source": so, "unverified": sorted(u)}
                      for s, i, m, so, u in lines]}


def to_engine(req, dims, repro, pred, L, ce, lines):
    return E.Component("c", frozenset(req), L, repro, pred,
                       {d: dim_record(code) for d, code in dims.items() if code != "ABSENT"},
                       tuple(E.CETest(f"t{k}", E.CE(o), required=r, preregistered=p, adequately_powered=w,
                                      satisfied_by_executed_control=c) for k, (o, r, p, w, c) in enumerate(ce)),
                       tuple(E.Line(f"l{k}", s, i, m, so, frozenset(u)) for k, (s, i, m, so, u) in enumerate(lines)))


def agree(req, dims, repro, pred, L, ce=(), lines=()):
    exp = oracle(to_oracle(req, dims, repro, pred, L, ce, lines))
    if exp[0] == "INVALID":
        with pytest.raises(E.InvalidRegistration):
            to_engine(req, dims, repro, pred, L, ce, lines)
        return exp[0]
    got = E.decide(to_engine(req, dims, repro, pred, L, ce, lines))
    assert got.verdict.value == exp[0], (req, dims, repro, pred, L, ce, lines, got, exp)
    if exp[0] != "UNSPECIFIED" and exp[2] != "mixed":
        assert got.rule.split(":")[0] == exp[2], (req, dims, repro, pred, L, ce, lines, got, exp)
        if exp[2] in ("3", "4"):
            assert got.level == exp[1], (req, dims, repro, pred, L, ce, lines, got, exp)
    return exp[0]


CE_ONE = [()] + [((o, r, p, w, c),) for o, r, p, w, c in
                 product(("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"), *[(False, True)] * 4)]


def test_E1_exhaustive():
    assert len(CE_ONE) == 49
    n, seen = 0, set()
    for slots in product(product((False, True), OUTS), repeat=3):
        req = [d for d, (inr, _) in zip(D, slots) if inr]
        dims = {d: code for d, (_, code) in zip(D, slots)}
        for repro, pred, L in product(REPRO, (False, True), range(5)):
            for ce in CE_ONE:
                seen.add(agree(req, dims, repro, pred, L, ce))
                n += 1
    assert n == 1_960_000
    assert seen == {"SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "UNSPECIFIED", "INVALID"}


def test_E2_l4_lines_exhaustive():
    dims = {d: "PASS" for d in D}
    q = (("SUPPORTED", False, True, True, False),)
    configs = [()] + [((s, "i", "m1", "s1", frozenset()),) for s in (False, True)]
    for s1, s2, same_i, diff_m, diff_s, disjoint in product(*[(False, True)] * 6):
        configs.append(((s1, "i", "m1", "s1", frozenset({"u"})),
                        (s2, "i" if same_i else "j", "m2" if diff_m else "m1", "s2" if diff_s else "s1",
                         frozenset({"v"}) if disjoint else frozenset({"u"}))))
    assert len(configs) == 67
    out = [agree(D, dims, "COMPLETE", True, L, q, lines) for L in (3, 4) for lines in configs]
    assert len(out) == 134 and out.count("SUPPORTED") == 67 + 1       # every L3 case, one L4 case


def test_E3_composite_exhaustive():
    V = (E.Verdict.SUPPORTED, E.Verdict.NOT_SUPPORTED, E.Verdict.INCONCLUSIVE)
    n = 0
    for k in range(5):
        for seq in product(V, repeat=k):
            assert E.composite(seq).value == oracle_composite([v.value for v in seq])
            n += 1
    assert n == 121
    with pytest.raises(ValueError):
        E.composite([E.Verdict.UNSPECIFIED])


# --- P and R: the full factor space ---------------------------------------------------------------

FACTORS = []
for d in D:
    FACTORS += [(f"req_{d}", (False, True)), (f"out_{d}", OUTS + ("ABSENT",))]
FACTORS += [("repro", REPRO), ("pred", (False, True)), ("L", tuple(range(5)))]
for k in range(3):
    FACTORS += [(f"ce{k}", (False, True)), (f"ce{k}_o", ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE")),
                (f"ce{k}_r", (False, True)), (f"ce{k}_p", (False, True)), (f"ce{k}_w", (False, True)),
                (f"ce{k}_c", (False, True))]
for k in range(3):
    FACTORS += [(f"ln{k}", (False, True)), (f"ln{k}_s", (False, True)), (f"ln{k}_i", ("a", "b")),
                (f"ln{k}_m", ("m1", "m2", "m3")), (f"ln{k}_so", ("s1", "s2", "s3")),
                (f"ln{k}_u", ((), ("u1",), ("u2",), ("u1", "u2")))]
NAMES = [n for n, _ in FACTORS]
LEVELS = dict(FACTORS)


def from_row(row):
    req = [d for d in D if row[f"req_{d}"]]
    dims = {d: row[f"out_{d}"] for d in D}
    ce = tuple((row[f"ce{k}_o"], row[f"ce{k}_r"], row[f"ce{k}_p"], row[f"ce{k}_w"], row[f"ce{k}_c"])
               for k in range(3) if row[f"ce{k}"])
    lines = tuple((row[f"ln{k}_s"], row[f"ln{k}_i"], row[f"ln{k}_m"], row[f"ln{k}_so"], frozenset(row[f"ln{k}_u"]))
                  for k in range(3) if row[f"ln{k}"])
    return req, dims, row["repro"], row["pred"], row["L"], ce, lines


def pairwise_rows(seed=20261106):
    """Greedy strength-2 covering array: each row fixes one uncovered pair, then fills greedily."""
    rng = random.Random(seed)
    uncovered = {(i, a, j, b) for i in range(len(NAMES)) for j in range(i + 1, len(NAMES))
                 for a in range(len(FACTORS[i][1])) for b in range(len(FACTORS[j][1]))}
    rows = []
    while uncovered:
        i, a, j, b = min(uncovered)
        pick = {i: a, j: b}
        for f in rng.sample(range(len(NAMES)), len(NAMES)):
            if f in pick:
                continue
            best, gain = 0, -1
            for v in range(len(FACTORS[f][1])):
                g = sum(((min(f, h), v if f < h else pick[h], max(f, h), pick[h] if f < h else v) in uncovered)
                        for h in pick)
                if g > gain:
                    best, gain = v, g
            pick[f] = best
        idx = [pick[f] for f in range(len(NAMES))]
        uncovered -= {(p, idx[p], q, idx[q]) for p in range(len(NAMES)) for q in range(p + 1, len(NAMES))}
        rows.append({NAMES[f]: FACTORS[f][1][idx[f]] for f in range(len(NAMES))})
    return rows


def test_P_pairwise_covering_array():
    rows = pairwise_rows()
    pairs = set()
    for r in rows:
        idx = [FACTORS[f][1].index(r[NAMES[f]]) for f in range(len(NAMES))]
        pairs |= {(p, idx[p], q, idx[q]) for p in range(len(NAMES)) for q in range(p + 1, len(NAMES))}
    total = sum(len(FACTORS[i][1]) * len(FACTORS[j][1]) for i in range(len(NAMES)) for j in range(i + 1, len(NAMES)))
    assert len(pairs) == total                                 # every pair of factor levels is covered
    for r in rows:
        agree(*from_row(r))


def test_R_random_draws():
    rng = random.Random(20261107)
    for _ in range(100_000):
        agree(*from_row({n: rng.choice(v) for n, v in FACTORS}))


# --- B: boundary literals, written by hand from the manuscript text ---------------------------------

Q = E.CETest("CE2", E.CE.SUPPORTED, preregistered=True, adequately_powered=True)
L1 = E.Line("a", True, "same", "m1", "s1", frozenset({"u1"}))
L2 = E.Line("b", True, "same", "m2", "s2", frozenset({"u2"}))
ALL_PASS = {d: E.DimRecord(E.Dim.PASS) for d in D}


def comp(*, dims=None, repro="COMPLETE", pred=True, L=4, ce=(Q,), lines=(L1, L2), req=D):
    return E.Component("b", frozenset(req), L, repro, pred, dict(ALL_PASS if dims is None else dims), ce, lines)


@pytest.mark.parametrize("name,c,verdict,level", [
    ("L4 reached", comp(), "SUPPORTED", 4),
    ("same method", comp(lines=(L1, E.Line("b", True, "same", "m1", "s2"))), "INCONCLUSIVE", 3),
    ("same source", comp(lines=(L1, E.Line("b", True, "same", "m2", "s1"))), "INCONCLUSIVE", 3),
    ("shared unverified", comp(lines=(L1, E.Line("b", True, "same", "m2", "s2", frozenset({"u1"})))), "INCONCLUSIVE", 3),
    ("other interpretation", comp(lines=(L1, E.Line("b", True, "other", "m2", "s2"))), "INCONCLUSIVE", 3),
    ("line not supporting", comp(lines=(L1, E.Line("b", False, "same", "m2", "s2"))), "INCONCLUSIVE", 3),
    ("CE not preregistered", comp(ce=(E.CETest("CE2", E.CE.SUPPORTED, adequately_powered=True),)), "INCONCLUSIVE", 2),
    ("CE underpowered", comp(ce=(E.CETest("CE2", E.CE.SUPPORTED, preregistered=True),)), "INCONCLUSIVE", 2),
    ("CE met by a control", comp(ce=(E.CETest("CE3", E.CE.SUPPORTED, preregistered=True, adequately_powered=True,
                                              satisfied_by_executed_control=True),)), "INCONCLUSIVE", 2),
    ("L3 suffices for L*=3", comp(L=3, lines=()), "SUPPORTED", 3),
    ("no predictive advantage", comp(pred=False), "INCONCLUSIVE", 0),
    ("non-reproducible", comp(repro="NON_REPRODUCIBLE"), "INCONCLUSIVE", None),
    ("dimension inconclusive", comp(dims={**ALL_PASS, "DI": E.DimRecord(E.Dim.INCONCLUSIVE)}), "INCONCLUSIVE", None),
    ("dimension absent", comp(dims={k: v for k, v in ALL_PASS.items() if k != "EV"}), "INCONCLUSIVE", None),
    ("admissible FAIL", comp(dims={**ALL_PASS, "II": E.DimRecord(E.Dim.FAIL)}), "NOT_SUPPORTED", None),
    ("required CE not supported", comp(ce=(Q, E.CETest("CE4", E.CE.NOT_SUPPORTED, required=True))), "NOT_SUPPORTED", None),
    ("inadmissible FAIL, L*=4", comp(dims={**ALL_PASS, "II": E.DimRecord(E.Dim.FAIL, admissible=False)}), "INCONCLUSIVE", None),
    ("X3: inadmissible FAIL, L*=1", comp(L=1, dims={**ALL_PASS, "II": E.DimRecord(E.Dim.FAIL, admissible=False)}),
     "UNSPECIFIED", None),
    ("X1: SUBSTANTIAL", comp(repro="SUBSTANTIAL"), "UNSPECIFIED", None),
    ("X2: PARTIAL", comp(repro="PARTIAL"), "UNSPECIFIED", None),
    # R_c empty: "PASS on every dimension in R_c" holds vacuously, so the level is L2 (>= L* = L1)
    ("X1 settled: SUBSTANTIAL reproduction-only", comp(repro="SUBSTANTIAL", L=1, req=(), dims={}, ce=(), lines=()),
     "SUPPORTED", 2),
    ("X4: required CE inconclusive, L*=2", comp(L=2, ce=(E.CETest("CE2", E.CE.INCONCLUSIVE, required=True),)),
     "UNSPECIFIED", None),
    ("X4 irrelevant once a row-1 FAIL exists",
     comp(L=2, dims={**ALL_PASS, "EV": E.DimRecord(E.Dim.FAIL)}, ce=(E.CETest("CE2", E.CE.INCONCLUSIVE, required=True),)),
     "NOT_SUPPORTED", None),
])
def test_B_boundary_literals(name, c, verdict, level):
    d = E.decide(c)
    assert d.verdict.value == verdict, (name, d)
    if level is not None:
        assert d.level == level, (name, d)


def test_registration_errors():
    with pytest.raises(E.InvalidRegistration):
        comp(dims={**ALL_PASS, "DI": E.DimRecord(E.Dim.NA)})
    with pytest.raises(E.InvalidRegistration):
        comp(req=("XX",))
    with pytest.raises(E.InvalidRegistration):
        comp(repro="MOSTLY")


# --- paper verdicts: a consistency check, not independent proof -------------------------------------
# The manuscript's verdicts were produced by applying these same rules, so agreement shows the engine
# encodes them as recorded; it does not show the rules are correct.

P_, I_, F_, NA_ = (E.DimRecord(E.Dim.PASS), E.DimRecord(E.Dim.INCONCLUSIVE), E.DimRecord(E.Dim.FAIL),
                   E.DimRecord(E.Dim.NA))
CE2 = E.CETest("CE2", E.CE.SUPPORTED, preregistered=True, adequately_powered=True)
CE3 = E.CETest("CE3", E.CE.SUPPORTED, satisfied_by_executed_control=True)
CE4 = E.CETest("CE4", E.CE.SUPPORTED, satisfied_by_executed_control=True)
PAPER = [  # (claim, protocol, R_c, L*, dims, CE, manuscript verdict, encoding note)
    ("G0", "v1", (), 1, {"II": NA_, "DI": NA_, "EV": NA_}, (), "SUPPORTED", "reproduction-only"),
    ("G1", "v1", D, 3, {"II": P_, "DI": I_, "EV": I_}, (CE2, CE3, CE4), "INCONCLUSIVE",
     "II PASS read from Table protocols l.1534 (v1 lists only DI, EV open)"),
    ("G1", "v1-alt", D, 3, {"II": I_, "DI": I_, "EV": I_}, (CE2, CE3, CE4), "INCONCLUSIVE",
     "II INCONCLUSIVE: l.1323 'the same under either reading'"),
    ("G1", "v2", D, 3, {"II": I_, "DI": I_, "EV": I_}, (CE2, CE3, CE4), "INCONCLUSIVE", ""),
    ("A1", "v1", D, 2, {"II": F_, "DI": I_, "EV": F_}, (), "NOT_SUPPORTED", "l.1535"),
    ("A1", "v2", D, 2, {"II": I_, "DI": I_, "EV": F_}, (), "NOT_SUPPORTED", "l.1535"),
    ("A2", "v1", ("DI",), 3, {"II": NA_, "DI": I_, "EV": NA_}, (CE2, CE3, CE4), "INCONCLUSIVE", "l.1536"),
    ("A2", "v2", ("DI",), 3, {"II": NA_, "DI": P_, "EV": NA_}, (CE2, CE3, CE4), "SUPPORTED", "l.1536"),
]


@pytest.mark.parametrize("claim,protocol,req,L,dims,ce,verdict,note", PAPER, ids=[f"{p[0]}-{p[1]}" for p in PAPER])
def test_paper_verdict_consistency(claim, protocol, req, L, dims, ce, verdict, note):
    d = E.decide(E.Component(claim, frozenset(req), L, "COMPLETE", True, dims, ce))
    assert d.verdict.value == verdict, (claim, protocol, d)


def test_a1_blocking_reasons_are_the_admissible_failures_not_di():
    v1 = E.decide(E.Component("A1", frozenset(D), 2, "COMPLETE", True, {"II": F_, "DI": I_, "EV": F_}))
    v2 = E.decide(E.Component("A1", frozenset(D), 2, "COMPLETE", True, {"II": I_, "DI": I_, "EV": F_}))
    assert v1.blocking == ["EV", "II"] and v2.blocking == ["EV"]


def test_the_oracle_shares_no_code_with_the_engine():
    tree = ast.parse((Path(__file__).parents[1] / "oracle_v2.py").read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    mods = {a.name for n in imports for a in n.names} | {n.module or "" for n in imports if isinstance(n, ast.ImportFrom)}
    assert mods <= {"__future__", "annotations", "itertools", "product"}, mods


def test_no_conditional_category_anywhere_in_v2_modules():
    word = "CONDITIONAL"                                     # this test file necessarily contains it
    for f in Path(__file__).parents[1].glob("*.py"):
        assert word not in f.read_text().upper(), f
