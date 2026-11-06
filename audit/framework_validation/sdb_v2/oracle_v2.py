"""
Oracle for the SDB-V2 decision engine: the manuscript's decision text as an explicit
predicate table over plain dictionaries. Written from the text, not from the engine; it
imports no project module and shares no code with engine_v2.py.

Input (all plain JSON-able values):
  {"required": ["II", ...], "L": 0..4, "reproduction": "COMPLETE"|"SUBSTANTIAL"|"PARTIAL"|"NON_REPRODUCIBLE",
   "predictive": bool,
   "dims": {"II": {"outcome": "PASS"|"FAIL"|"INCONCLUSIVE"|"N/A", "admissible": bool}, ...},   # absent = missing
   "ce": [{"outcome": "SUPPORTED"|"NOT_SUPPORTED"|"INCONCLUSIVE", "required": bool, "prereg": bool,
           "powered": bool, "control": bool}, ...],
   "lines": [{"supports": bool, "interp": str, "method": str, "source": str, "unverified": [str, ...]}, ...]}

Output: (verdict, level-or-None, row) with verdict one of SUPPORTED, NOT_SUPPORTED,
INCONCLUSIVE, UNSPECIFIED; "INVALID" for a required dimension registered N/A.
"""
from __future__ import annotations

from itertools import product

# Open questions the text leaves undecided. Each: (name, applies(x), answers).
QUESTIONS = (
    ("reproduced",
     lambda x: x["reproduction"] == "PARTIAL"
     or (x["reproduction"] == "SUBSTANTIAL" and not (not x["required"] and x["L"] == 1)),
     (True, False)),
    ("inconclusive_required_ce_is_prerequisite",
     lambda x: any(t["required"] and t["outcome"] == "INCONCLUSIVE" for t in x["ce"]),
     (True, False)),
    ("inadmissible_fail_is_prerequisite_open",
     lambda x: any((x["dims"].get(d) or {}).get("outcome") == "FAIL" and not x["dims"][d].get("admissible", True)
                   for d in x["required"]),
     (True, False)),
)


def _rec(x, d):
    return x["dims"].get(d)


def _settled_reproduced(x):
    r = x["reproduction"]
    if r == "COMPLETE":
        return True
    if r == "NON_REPRODUCIBLE":
        return False
    return True          # SUBSTANTIAL, reproduction-only (l.1095-1097); other cases go through QUESTIONS


# Evidence levels: each requires the previous one (Table evidence-levels).
def _level(x, reading):
    reproduced = reading.get("reproduced", _settled_reproduced(x))
    l1 = reproduced and x["predictive"]
    l2 = l1 and all((_rec(x, d) or {}).get("outcome") == "PASS" for d in x["required"])
    l3 = l2 and any(t["outcome"] == "SUPPORTED" and t["prereg"] and t["powered"] and not t["control"]
                    for t in x["ce"])
    sup = [ln for ln in x["lines"] if ln["supports"]]
    l4 = l3 and any(sup[i]["interp"] == sup[j]["interp"] and sup[i]["method"] != sup[j]["method"]
                    and sup[i]["source"] != sup[j]["source"]
                    and not set(sup[i]["unverified"]) & set(sup[j]["unverified"])
                    for i in range(len(sup)) for j in range(i + 1, len(sup)))
    return [l1, l2, l3, l4].count(True) if l1 else 0


# Table claim-decisions: first satisfied row.
ROWS = (
    ("1", "NOT_SUPPORTED",
     lambda x, r, lv: any((_rec(x, d) or {}).get("outcome") == "FAIL" and _rec(x, d).get("admissible", True)
                          for d in x["required"])
     or any(t["required"] and t["outcome"] == "NOT_SUPPORTED" for t in x["ce"])),
    ("2", "INCONCLUSIVE",
     lambda x, r, lv: not r.get("reproduced", _settled_reproduced(x))
     or any(_rec(x, d) is None or _rec(x, d)["outcome"] == "INCONCLUSIVE" for d in x["required"])
     or (r.get("inconclusive_required_ce_is_prerequisite")
         and any(t["required"] and t["outcome"] == "INCONCLUSIVE" for t in x["ce"]))
     or (r.get("inadmissible_fail_is_prerequisite_open")
         and any((_rec(x, d) or {}).get("outcome") == "FAIL" and not _rec(x, d).get("admissible", True)
                 for d in x["required"]))),
    ("3", "INCONCLUSIVE", lambda x, r, lv: lv < x["L"]),
    ("4", "SUPPORTED",
     lambda x, r, lv: all((_rec(x, d) or {}).get("outcome") == "PASS" for d in x["required"]) and lv >= x["L"]),
)


def oracle(x: dict):
    if any((_rec(x, d) or {}).get("outcome") == "N/A" for d in x["required"]):
        return ("INVALID", None, "registration")
    open_q = [(name, answers) for name, applies, answers in QUESTIONS if applies(x)]
    outcomes = set()
    for combo in product(*[a for _, a in open_q]) if open_q else [()]:
        reading = {name: v for (name, _), v in zip(open_q, combo)}
        lv = _level(x, reading)
        hit = next(((row, verdict) for row, verdict, pred in ROWS if pred(x, reading, lv)), ("none", "UNDEFINED"))
        outcomes.add((hit[1], hit[0], lv))
    verdicts = {o[0] for o in outcomes}
    if len(verdicts) == 1 and "UNDEFINED" not in verdicts:
        rows = {o[1] for o in outcomes}
        levels = {o[2] for o in outcomes}
        return (verdicts.pop(), levels.pop() if len(levels) == 1 else None, rows.pop() if len(rows) == 1 else "mixed")
    return ("UNSPECIFIED", None, "unspecified")


def oracle_composite(verdicts: list) -> str:
    n = len(verdicts)
    s, ns = verdicts.count("SUPPORTED"), verdicts.count("NOT_SUPPORTED")
    if n and s == n:
        return "SUPPORTED"
    if n and ns == n:
        return "NOT_SUPPORTED"
    if n and s + ns == n and s and ns:
        return "PARTIALLY_SUPPORTED"
    return "INCONCLUSIVE"
