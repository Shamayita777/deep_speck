"""
ERRATUM TOOL - recompute the corrected secondary descriptive metrics for an
existing adversarial certificate, from its PERSISTED sealed set.

Read-only with respect to evidence: it never rewrites a certificate and never
regenerates a dataset. It exists so a delivered certificate can be corrected
by erratum rather than reissued, which keeps the original artifact intact.

    python erratum_recompute.py audit/adversarial/evidence/a4_20261006
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_W = 16
_PAIRS = np.array([(j, 2 * _W + j) for j in range(_W)]
                  + [(_W + j, 3 * _W + j) for j in range(_W)])


def sha256_array(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        if a.size:
            h.update(memoryview(a.reshape(-1)).cast("B"))
    return h.hexdigest()


def a4_scores(X: np.ndarray, key: bytes) -> np.ndarray:
    packed = np.packbits((X[:, _PAIRS[:, 0]] ^ X[:, _PAIRS[:, 1]]).astype(np.uint8),
                         axis=1)
    return np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), key=key,
                                                    digest_size=8).digest(), "big")
                     / float(1 << 64) for r in packed], dtype=np.float32)


def main(run_dir: str) -> int:
    run = Path(run_dir)
    cert_paths = sorted(run.glob("a?_certificate.json"))
    if not cert_paths:
        print(f"no a?_certificate.json in {run}"); return 1
    cert = json.loads(cert_paths[0].read_text())
    adv = cert["identity"]["adversary_id"]
    d = cert["distinguisher_assessment"]

    sealed_manifest = json.loads((run / "datasets" / "sealed_manifest.json").read_text())
    with np.load(run / "datasets" / "sealed.npz") as z:
        X, Y = z["X"], z["Y"]
    if sha256_array(X, Y) != sealed_manifest["content_sha256"]:
        print("REFUSED: the persisted sealed set does not match its manifest hash")
        return 2
    print(f"sealed set verified: n={Y.size} hash={sealed_manifest['content_sha256'][:16]}")

    if adv == "A4":
        spec = json.loads((run / "a4_xor_response.json").read_text())
        s = a4_scores(X, spec["key"].encode())
    else:
        import keras
        mp = next(run.glob("*.keras"))
        s = np.asarray(keras.models.load_model(mp).predict(
            X, batch_size=5000, verbose=0)).reshape(-1)
        print(f"loaded {mp.name}")

    thr = d["calibrated_threshold"]
    pred = (s > thr).astype(np.int64)
    if d.get("orientation_flipped"):
        pred = 1 - pred
    tp = int(((pred == 1) & (Y == 1)).sum()); fn = int(((pred == 0) & (Y == 1)).sum())
    tn = int(((pred == 0) & (Y == 0)).sum()); fp = int(((pred == 1) & (Y == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    reproduced = {
        "sealed_accuracy_at_0.5": float(((s > 0.5).astype(np.int64) == Y).mean()),
        "sealed_accuracy_at_calibrated_threshold": float((pred == Y).mean()),
    }
    print("\nreproduction of the certificate's own figures:")
    ok = True
    for k, v in reproduced.items():
        same = abs(v - float(d[k])) < 1e-9
        ok &= same
        print(f"  {k:46s} recomputed {v:.6f}  certificate {float(d[k]):.6f}  "
              f"{'match' if same else 'MISMATCH'}")

    print("\ncorrected secondary descriptive metric (ERRATUM 1):")
    print(f"  confusion                tp={tp} fn={fn} tn={tn} fp={fp}")
    print(f"  TPR {tpr:.6f}   TNR {tnr:.6f}   FPR {1 - tnr:.6f}")
    print(f"  balanced_accuracy CORRECT (TPR+TNR)/2 = {(tpr + tnr) / 2:.6f}")
    print(f"  balanced_accuracy as reported         = "
          f"{float(d['balanced_accuracy_at_calibrated_threshold']):.6f}")
    print("\nconstruction validity is decided by AUC and its bootstrap CI, which this "
          "erratum does not touch:")
    print(f"  auc {d['auc']:.6f}  ci95 {d['auc_ci95_bootstrap']}  "
          f"equivalence_satisfied={d['equivalence_satisfied']}  "
          f"construction={d['construction_status']}")
    if not ok:
        print("\nWARNING: a reproduced figure did not match. Investigate before "
              "publishing the erratum.")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
