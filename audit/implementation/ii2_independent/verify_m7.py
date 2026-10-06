"""
M7 VERIFICATION - independent differential theory.

Mirrors the structure of verify_m1.py: the independent implementation never
imports the audited one; the audited implementation is imported only inside
this comparator.

Checks, in increasing order of what they can establish:

  T1  word_size = 4, EXHAUSTIVE. All 4096 (alpha, beta, gamma) triples, each
      compared against exact enumeration over all 256 (x, y) pairs. Exact
      equality required. Ground truth is the definition of xdp+, not another
      model of it.

  T2  word_size = 8, sampled. Random triples against exact enumeration over
      all 65536 (x, y) pairs. Exact equality required.

  T3  word_size = 16, sampled. Random triples against the AUDITED
      Lipmaa-Moriai closed form in audit/cryptography/gohr/speck.py. This is
      the check that retires the single-point-of-failure: two independent
      algorithms, same numbers.

  T4  Normalisation. For random (alpha, beta) the probabilities over all
      gamma must sum to exactly 1 - a property the closed form and the
      automaton can only share if both are correct.

  T5  Degenerate and boundary cases, including the forced delta_0 = 0
      condition and the all-zero differential.

  T6  END TO END. The CE2 target is regenerated from identical cryptographic
      material by two fully independent paths - independent Speck plus the
      automaton, against audited Speck plus the closed form - and the 64-bit
      encodings and the per-sample targets are compared.

Usage, from the repository root:

    python -m audit.implementation.ii2_independent.verify_m7
    python -m audit.implementation.ii2_independent.verify_m7 --quick
    python -m audit.implementation.ii2_independent.verify_m7 \
        --out audit/implementation/evidence/m7/m7_theory_verification.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit.implementation.ii2_independent import theory_independent as TI

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sha256_array(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        if a.size:
            h.update(memoryview(a.reshape(-1)).cast("B"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# T1 / T2 - against exact enumeration (the definition)
# ---------------------------------------------------------------------------

def t1_exhaustive_word4() -> dict:
    w = 4
    vals = np.arange(1 << w, dtype=np.int64)
    A, B, G = np.meshgrid(vals, vals, vals, indexing="ij")
    dp = TI.xdp_plus_dp(A.reshape(-1), B.reshape(-1), G.reshape(-1), word_size=w)
    truth = np.empty_like(dp)
    for idx, (a, b, g) in enumerate(zip(A.reshape(-1), B.reshape(-1), G.reshape(-1))):
        truth[idx] = TI.xdp_plus_bruteforce(int(a), int(b), int(g), w)
    exact = bool(np.array_equal(dp, truth))
    return {"check": "T1 exhaustive word_size=4 vs exact enumeration",
            "n_triples": int(dp.size), "exact_match": exact,
            "max_abs_diff": float(np.abs(dp - truth).max()),
            "n_nonzero": int((truth > 0).sum()), "ok": exact}


def t2_sampled_word8(n_triples: int, seed: int) -> dict:
    w = 8
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 1 << w, n_triples)
    b = rng.integers(0, 1 << w, n_triples)
    # Half the triples are drawn from FEASIBLE gammas, so the test is not
    # dominated by trivially-zero cases.
    g = rng.integers(0, 1 << w, n_triples)
    g[: n_triples // 2] = (a[: n_triples // 2] ^ b[: n_triples // 2]
                           ^ rng.integers(0, 1 << w, n_triples // 2) * 2)
    g &= (1 << w) - 1
    dp = TI.xdp_plus_dp(a, b, g, word_size=w)
    truth = np.array([TI.xdp_plus_bruteforce(int(x), int(y), int(z), w)
                      for x, y, z in zip(a, b, g)])
    exact = bool(np.array_equal(dp, truth))
    return {"check": "T2 sampled word_size=8 vs exact enumeration",
            "n_triples": int(n_triples), "exact_match": exact,
            "max_abs_diff": float(np.abs(dp - truth).max()),
            "n_nonzero": int((truth > 0).sum()), "ok": exact}


# ---------------------------------------------------------------------------
# T3 - against the audited closed form
# ---------------------------------------------------------------------------

def t3_vs_audited_word16(n_triples: int, seed: int) -> dict:
    from audit.cryptography.gohr import speck as AUDITED      # comparator only

    w = 16
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 1 << w, n_triples)
    b = rng.integers(0, 1 << w, n_triples)
    g = rng.integers(0, 1 << w, n_triples)
    # include feasible triples: delta_0 == 0 and a realisable carry trajectory
    half = n_triples // 2
    x = rng.integers(0, 1 << w, half); y = rng.integers(0, 1 << w, half)
    m = (1 << w) - 1
    g[:half] = (((x + y) & m) ^ (((x ^ a[:half]) + (y ^ b[:half])) & m))

    dp = TI.xdp_plus_dp(a, b, g, word_size=w)
    ref = np.asarray(AUDITED.xdp_plus(a.astype(np.uint16), b.astype(np.uint16),
                                      g.astype(np.uint16)), dtype=np.float64)
    diff = np.abs(dp - ref)
    # both routes emit powers of two, so disagreement should be exactly zero
    ok = bool(diff.max() == 0.0)
    disagree = int((diff > 0).sum())
    return {"check": "T3 sampled word_size=16 vs audited Lipmaa-Moriai closed form",
            "n_triples": int(n_triples), "exact_match": ok,
            "max_abs_diff": float(diff.max()), "n_disagreeing": disagree,
            "n_feasible_reference": int((ref > 0).sum()),
            "n_feasible_independent": int((dp > 0).sum()),
            "zero_pattern_identical": bool(np.array_equal(ref > 0, dp > 0)),
            "ok": ok}


# ---------------------------------------------------------------------------
# T4 / T5 - internal consistency and boundaries
# ---------------------------------------------------------------------------

def t4_normalisation(n_pairs: int, seed: int) -> dict:
    w = 8
    rng = np.random.default_rng(seed)
    gammas = np.arange(1 << w, dtype=np.int64)
    worst = 0.0
    for _ in range(n_pairs):
        a = int(rng.integers(0, 1 << w)); b = int(rng.integers(0, 1 << w))
        total = TI.xdp_plus_dp(np.full(1 << w, a), np.full(1 << w, b), gammas,
                               word_size=w).sum()
        worst = max(worst, abs(total - 1.0))
    ok = worst < 1e-12
    return {"check": "T4 probabilities over all gamma sum to 1 (word_size=8)",
            "n_pairs": int(n_pairs), "max_abs_deviation_from_1": float(worst),
            "ok": ok}


def t5_boundaries() -> dict:
    w = 16
    cases, ok = [], True

    p = float(TI.xdp_plus_dp(np.array([0]), np.array([0]), np.array([0]),
                             word_size=w)[0])
    cases.append({"case": "alpha=beta=gamma=0", "expected": 1.0, "got": p})
    ok &= p == 1.0

    # delta_0 != 0 is impossible: neither addition carries into bit 0
    p = float(TI.xdp_plus_dp(np.array([0]), np.array([0]), np.array([1]),
                             word_size=w)[0])
    cases.append({"case": "delta_0 != 0 (gamma=1, alpha=beta=0)",
                  "expected": 0.0, "got": p})
    ok &= p == 0.0

    # top-bit difference is always free: the carry out is discarded
    top = 1 << (w - 1)
    p = float(TI.xdp_plus_dp(np.array([top]), np.array([0]), np.array([top]),
                             word_size=w)[0])
    cases.append({"case": "difference in the most significant bit only",
                  "expected": 1.0, "got": p})
    ok &= p == 1.0

    return {"check": "T5 boundary cases", "cases": cases, "ok": bool(ok)}


# ---------------------------------------------------------------------------
# T6 - end-to-end CE2 target regeneration
# ---------------------------------------------------------------------------

def t6_end_to_end(n_samples: int, rounds: int, differential, seed: int) -> dict:
    from audit.cryptography.gohr import speck as AUDITED      # comparator only

    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 1 << 16, size=(4, n_samples)).astype(np.uint16)
    p0l = rng.integers(0, 1 << 16, size=n_samples).astype(np.uint16)
    p0r = rng.integers(0, 1 << 16, size=n_samples).astype(np.uint16)

    X_ind, t_ind, F_ind = TI.independent_trail_targets(
        keys, p0l, p0r, rounds=rounds, differential=differential)

    # audited path, driven with the SAME material
    mask = AUDITED.MASK_VAL
    p1l = p0l ^ np.uint16(differential[0])
    p1r = p0r ^ np.uint16(differential[1])
    ks = AUDITED.expand_key(keys, rounds)
    x0, y0, x1, y1 = p0l.copy(), p0r.copy(), p1l.copy(), p1r.copy()
    facs = []
    for k in ks:
        a0, b0 = AUDITED.ror(x0, AUDITED.ALPHA()), y0
        a1, b1 = AUDITED.ror(x1, AUDITED.ALPHA()), y1
        gamma = ((a0.astype(np.int64) + b0) & mask) ^ ((a1.astype(np.int64) + b1) & mask)
        facs.append(AUDITED.xdp_plus(a0 ^ a1, b0 ^ b1, gamma.astype(np.uint16)))
        x0, y0 = AUDITED.enc_one_round((x0, y0), k)
        x1, y1 = AUDITED.enc_one_round((x1, y1), k)
    F_ref = np.asarray(facs)
    feasible = (F_ref > 0).all(axis=0)
    t_ref = np.where(feasible, np.exp2(np.log2(np.where(F_ref > 0, F_ref, 1.0)).sum(0)), 0.0)
    X_ref = AUDITED.convert_to_binary([x0, y0, x1, y1])

    enc_ok = bool(np.array_equal(X_ind, X_ref))
    fac_ok = bool(np.array_equal(F_ind, F_ref))
    tgt_diff = float(np.abs(t_ind - t_ref).max())
    ok = enc_ok and fac_ok and tgt_diff == 0.0
    return {"check": "T6 end-to-end CE2 target by two independent paths",
            "n_samples": int(n_samples), "rounds": int(rounds),
            "differential": [int(d) for d in differential],
            "ciphertext_encoding_identical": enc_ok,
            "per_round_factors_identical": fac_ok,
            "target_max_abs_diff": tgt_diff,
            "independent_target_sha256": _sha256_array(t_ind),
            "audited_target_sha256": _sha256_array(t_ref),
            "encoding_sha256": _sha256_array(X_ind),
            "ok": ok}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(quick: bool = False) -> dict:
    scale = 0.1 if quick else 1.0
    checks = [
        t1_exhaustive_word4(),
        t2_sampled_word8(int(400 * scale) or 40, seed=20261006),
        t3_vs_audited_word16(int(200_000 * scale) or 20_000, seed=20261007),
        t4_normalisation(int(60 * scale) or 6, seed=20261008),
        t5_boundaries(),
        t6_end_to_end(int(20_000 * scale) or 2_000, rounds=5,
                      differential=(0x0040, 0x0000), seed=20261009),
    ]
    failed = [c["check"] for c in checks if not c["ok"]]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True,
                                timeout=20).stdout.strip() or "UNAVAILABLE"
    except Exception:                                          # noqa: BLE001
        commit = "UNAVAILABLE"
    from audit.cryptography.gohr import speck as AUDITED
    return {
        "milestone": "M7",
        "title": "Independent implementation of the differential theory",
        "independent_implementation_id": TI.THEORY_IMPLEMENTATION_ID,
        "independent_algorithm": "carry-difference automaton (dynamic programming "
                                 "over bit positions)",
        "audited_algorithm": "Lipmaa-Moriai closed form (FSE 2001)",
        "ground_truth_for_small_words": "exhaustive enumeration over all (x, y)",
        "checks": checks,
        "verdict": "VERIFIED" if not failed else "MISMATCH",
        "failed_checks": failed,
        "quick_mode": bool(quick),
        "source_hashes": {
            "audit/implementation/ii2_independent/theory_independent.py":
                _sha256_file(HERE / "theory_independent.py"),
            "audit/implementation/ii2_independent/verify_m7.py":
                _sha256_file(HERE / "verify_m7.py"),
            "audit/cryptography/gohr/speck.py": _sha256_file(
                Path(AUDITED.__file__)),
        },
        "environment": {"python": sys.version.split()[0],
                        "numpy": np.__version__,
                        "platform": platform.platform()},
        "git_commit": commit,
        "utc": _utc(),
        "limitations": [
            "establishes agreement between two implementations of the same "
            "quantity; it does not establish that the quantity is the right "
            "target for CE2, which is a separate, documented scope question",
            "word_size 16 is verified by sampling, not exhaustively: the full "
            "triple space is 2^48",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M7 independent theory verification")
    ap.add_argument("--quick", action="store_true", help="reduced sample sizes")
    ap.add_argument("--out", type=Path, default=None, help="write the certificate here")
    args = ap.parse_args(argv)

    report = run(quick=args.quick)
    for c in report["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['check']}")
    print(f"\nM7 verdict: {report['verdict']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
        print(f"certificate: {args.out}")
    return 0 if report["verdict"] == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
