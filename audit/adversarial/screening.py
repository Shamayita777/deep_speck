"""
DEVELOPMENT ONLY - NOT SCIENTIFIC EVIDENCE.

Characterises candidate adversary features BEFORE they are registered, so that
an adversary is chosen on properties that can be defended in advance rather
than on a confirmatory outcome. Output is stamped and must never be cited as
evidence, nor used to revise a registration after a confirmatory run.

For each candidate it reports association with the CE2 analytical target on
real pairs, and real/random separation (AUC with a bootstrap interval) on
mixed pairs.

    python -m audit.adversarial.screening --n 300000
"""
from __future__ import annotations

import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from audit.cryptography.ce234 import frozen_plan as P
from audit.cryptography.ce234 import production as PR

BANNER = "DEVELOPMENT ONLY - NOT SCIENTIFIC EVIDENCE"
_W = 16
_PAIRS = np.array([(j, 2 * _W + j) for j in range(_W)]
                  + [(_W + j, 3 * _W + j) for j in range(_W)])


def pair_xor(Z):
    return (Z[:, _PAIRS[:, 0]] ^ Z[:, _PAIRS[:, 1]]).astype(np.uint8)


def _keyed_map(Z, key=b"screening"):
    packed = np.packbits(pair_xor(Z), axis=1)
    return np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), key=key,
                                                    digest_size=8).digest(), "big")
                     / float(1 << 64) for r in packed])


CANDIDATES = {
    "popcount(pair XOR), 32 bits": lambda Z: pair_xor(Z).sum(1).astype(float),
    "popcount(pair XOR), left word": lambda Z: pair_xor(Z)[:, :16].sum(1).astype(float),
    "popcount(C0) only (not a difference)": lambda Z: Z[:, :32].sum(1).astype(float),
    "parity(pair XOR)": lambda Z: (pair_xor(Z).sum(1) & 1).astype(float),
    "keyed BLAKE2b map(pair XOR)": _keyed_map,
}


def auc(f, y):
    if np.std(f) == 0:
        return 0.5
    r = stats.rankdata(f); n1 = float(y.sum()); n0 = float(y.size - n1)
    a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(a, 1 - a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=BANNER)
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from audit.cryptography.gohr import speck as sp
    print(BANNER)
    X, Y = sp.make_train_data(a.n, P.ROUNDS, diff=P.DIFFERENTIAL)
    Xr, tgt = PR.generate_theory_data(max(20_000, a.n // 3))
    rng = np.random.default_rng(0)
    rows = []
    for name, fn in CANDIDATES.items():
        f = fn(X)
        boots = [auc(f[i], Y[i]) for i in
                 (rng.integers(0, Y.size, Y.size) for _ in range(a.boot))]
        rows.append({"candidate": name, "auc": auc(f, Y),
                     "auc_ci95": [float(np.percentile(boots, 2.5)),
                                  float(np.percentile(boots, 97.5))],
                     "spearman_vs_ce2_target": float(
                         stats.spearmanr(fn(Xr), tgt).statistic),
                     "mean_real": float(f[Y == 1].mean()),
                     "mean_random": float(f[Y == 0].mean())})
        r = rows[-1]
        print(f"{name:38s} AUC {r['auc']:.4f} CI {r['auc_ci95']} "
              f"rho {r['spearman_vs_ce2_target']:+.4f}")
    rep = {"banner": BANNER, "n_mixed": int(a.n), "n_real": int(len(tgt)),
           "rounds": P.ROUNDS, "differential": list(P.DIFFERENTIAL),
           "candidates": rows, "utc": datetime.now(timezone.utc).isoformat(),
           "non_evidentiary": True}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=2))
        print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
