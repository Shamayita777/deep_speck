from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ....core import evidence as EV
from ....targets.autond import upstream as UP
from . import adapter as A
from . import design as D
from . import runner as R

SCHEMA = "autond-ce3-representation-diagnostics-1"
CHUNK = 50_000
RELATIVE_THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-6, 1e-9)
ACTIVE_FRACTIONS = (1e-6, 1e-4, 1e-3, 1e-2)


class DiagnosticsError(RuntimeError):
    pass


def p1_inputs(up: UP.Upstream) -> tuple[np.ndarray, dict]:
    path = R.CRYPTO_ROOT / D.P1_RECORD_RELPATH
    if EV.sha256_file(path) != D.P1_RECORD_SHA256:
        raise DiagnosticsError("P1 record hash differs from the design pin")
    rec = D.parse_json_strict(path.read_text())
    data = path.parent / rec["data_file"]
    if EV.sha256_file(data) != rec["data_sha256"]:
        raise DiagnosticsError("P1 data hash differs from the record")
    with np.load(data, allow_pickle=False) as z:
        kw, p0, p1 = z["key_words"], z["pt0"], z["pt1"]
    c = up.cipher
    kb = np.concatenate([((kw[:, j:j + 1].astype(np.int64) >> np.arange(15, -1, -1)) & 1) for j in range(4)],
                        1).astype(np.uint8)
    b0 = np.concatenate([((p0[:, j:j + 1].astype(np.int64) >> np.arange(15, -1, -1)) & 1) for j in range(2)],
                        1).astype(np.uint8)
    b1 = np.concatenate([((p1[:, j:j + 1].astype(np.int64) >> np.arange(15, -1, -1)) & 1) for j in range(2)],
                        1).astype(np.uint8)
    c0 = c.convert_from_binary(c.encrypt(b0, kb, 8))
    c1 = c.convert_from_binary(c.encrypt(b1, kb, 8))
    return A.model_input(up, c0, c1), {"record": D.P1_RECORD_RELPATH, "record_sha256": D.P1_RECORD_SHA256,
                                       "data_sha256": rec["data_sha256"], "n": int(kw.shape[0]),
                                       "arrays_read": ["key_words", "pt0", "pt1"], "target_read": False}


def _tsqr_singular_values(H: np.ndarray, mean: np.ndarray, scale: np.ndarray | None) -> np.ndarray:
    R_ = None
    for s in range(0, H.shape[0], CHUNK):
        Z = H[s:s + CHUNK].astype(np.float64) - mean
        if scale is not None:
            Z = Z / scale
        R_ = np.linalg.qr(Z if R_ is None else np.vstack([R_, Z]), mode="r")
    return np.linalg.svd(R_, compute_uv=False)


def _spectrum(sv: np.ndarray, n: int) -> dict:
    sv = np.asarray(sv, dtype=np.float64)
    d = sv.size
    lam = sv ** 2 / n
    p = sv / sv.sum() if sv.sum() > 0 else sv
    pz = p[p > 0]
    tol64 = float(sv.max() * max(n, d) * np.finfo(np.float64).eps)
    tol32 = float(sv.max() * max(n, d) * np.finfo(np.float32).eps)
    return {"singular_values": sv.tolist(), "eigenvalues_per_sample": lam.tolist(),
            "numerical_rank": {"numpy_rule_float64_eps": int(np.count_nonzero(sv > tol64)),
                               "numpy_rule_float32_eps": int(np.count_nonzero(sv > tol32)),
                               "relative_to_largest": {f"{t:g}": int(np.count_nonzero(sv > t * sv.max()))
                                                       for t in RELATIVE_THRESHOLDS}},
            "effective_rank_entropy": float(np.exp(-np.sum(pz * np.log(pz)))) if pz.size else 0.0,
            "participation_ratio": float(lam.sum() ** 2 / np.sum(lam ** 2)) if lam.sum() > 0 else 0.0,
            "variance_fraction_top": {str(k): float(np.sum(lam[:k]) / lam.sum()) for k in (1, 2, 5, 10, 14, 20)
                                      if k <= d and lam.sum() > 0}}


def describe(H: np.ndarray) -> dict:
    n, d = H.shape
    total, zeros, sq = np.zeros(d), np.zeros(d, np.int64), np.zeros(d)
    lo, hi = np.full(d, np.inf), np.full(d, -np.inf)
    for s in range(0, n, CHUNK):
        Z = H[s:s + CHUNK].astype(np.float64)
        total += Z.sum(0)
        zeros += np.count_nonzero(Z == 0.0, axis=0)
        lo, hi = np.minimum(lo, Z.min(0)), np.maximum(hi, Z.max(0))
    mean = total / n
    for s in range(0, n, CHUNK):
        sq += ((H[s:s + CHUNK].astype(np.float64) - mean) ** 2).sum(0)
    var = sq / n
    active = n - zeros
    nonconstant = var > 0
    out = {"n": int(n), "units": int(d), "per_unit": {"mean": mean.tolist(), "variance": var.tolist(),
                                                     "fraction_zero": (zeros / n).tolist(),
                                                     "n_nonzero": active.tolist(), "minimum": lo.tolist(),
                                                     "maximum": hi.tolist()},
           "units_constant": int(np.count_nonzero(~nonconstant)),
           "units_never_active": int(np.count_nonzero(active == 0)),
           "units_nonconstant": int(np.count_nonzero(nonconstant)),
           "units_active_on_more_than_fraction": {f"{f:g}": int(np.count_nonzero(active > f * n))
                                                  for f in ACTIVE_FRACTIONS},
           "overall_fraction_zero": float(zeros.sum() / (n * d)),
           "variance_sorted_descending": np.sort(var)[::-1].tolist()}
    out["centred"] = _spectrum(_tsqr_singular_values(H, mean, None), n)
    if np.any(nonconstant):
        Hn = H[:, nonconstant]
        out["standardised_nonconstant_units"] = _spectrum(
            _tsqr_singular_values(Hn, mean[nonconstant], np.sqrt(var[nonconstant])), n)
    return out


def run(path: Path, *, upstream=None) -> dict:
    path = Path(path)
    if path.exists():
        raise DiagnosticsError(f"refusing to overwrite {path}")
    t0 = time.time()
    up = UP.load(upstream)
    model, twin, binding = A.load_models(up, R.CRYPTO_ROOT / D.CE2_RUN_RELPATH / D.CE2_TWIN_RELPATH)
    X, source = p1_inputs(up)
    out = {}
    for role, m in (("published", model), ("twin", twin)):
        t = time.time()
        H = A.extract(A.extractor(m), X)
        out[role] = {**describe(H), "seconds": time.time() - t}
        del H
    rec = {"schema": SCHEMA, "experiment_id": D.EXPERIMENT_ID, "evidentiary": False, "confirmatory": False,
           "inputs": source, "layer": {k: D.REPRESENTATION[k] for k in ("layer_index", "layer_class", "activation",
                                                                         "output_shape")},
           "layer_changed": False, "target_used": False, "probe_fitted": False,
           "binding": {"checkpoint_sha256": binding["sha256"], "published_weights_sha256": binding["weights_sha256"],
                       "twin_file_sha256": D.CE2_TWIN_FILE_SHA256, "twin_weights_sha256": D.CE2_TWIN_WEIGHTS_SHA256},
           "definitions": {
               "fraction_zero": "fraction of the n inputs on which the ReLU unit output is exactly 0.0",
               "numerical_rank": ("number of singular values of the centred n x 64 matrix above a threshold; the "
                                  "numpy rule is max(sv) * max(n, d) * eps; relative thresholds are fractions of "
                                  "max(sv)"),
               "singular_values": "of the centred representation, by TSQR in float64 over chunks of 50000 rows",
               "effective_rank_entropy": "exp(-sum p_i log p_i), p_i = sv_i / sum(sv) (Roy and Vetterli 2007)",
               "participation_ratio": "(sum lambda)^2 / sum lambda^2, lambda = sv^2 / n (covariance eigenvalues)",
               "standardised": "the same on the non-constant units after division by their standard deviation, "
                               "the geometry the probe's scaler presents to the logistic regression"},
           "environment": R.environment(), "representations": out, "seconds_total": time.time() - t0,
           "utc": EV.utc()}
    EV.write_json(path, rec)
    return rec
