from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..ce2 import verify as V2

CRYPTO_ROOT = Path(__file__).resolve().parents[3]
CE3_ROOT = Path(__file__).resolve().parent
CHECKPOINT_RELPATH = "model_weights/speck3264_single-key_0x400000_round8_polished.h5"
GENESIS = "0" * 64
JOURNAL_SCHEMA = "core-journal-2"
TERMINAL = {"VALID", "EXCLUDED", "FAILED"}
PERMITTED = {"NOT_STARTED": {"DATA_GENERATED", "FAILED"}, "DATA_GENERATED": {"DATA_VERIFIED", "FAILED"},
             "DATA_VERIFIED": {"REPRESENTATIONS_EXTRACTED", "FAILED"},
             "REPRESENTATIONS_EXTRACTED": {"PROBES_FITTED", "FAILED"},
             "PROBES_FITTED": {"EVIDENCE_WRITTEN", "FAILED"}, "EVIDENCE_WRITTEN": {"VALID", "EXCLUDED", "FAILED"}}
ARTIFACT_FILE = {"data": "data.npz", "calibration": "calibration.npz", "representations": "representations.npz",
                 "probes": "probes.npz", "measurement": "measurement.json"}
NO_EFFECTIVE = "NO_VALID_OBSERVATIONS"
MARGIN = 1e-9
CLASS_SUM_TOLERANCE = 1e-8
ROUNDS = 8
DX, DY = 0x0040, 0x0000

sha256_file = V2.sha256_file
canonical_sha256 = V2.canonical_sha256
content_hash = V2.content_hash
Findings = V2.Findings


class VerificationError(RuntimeError):
    pass


def _strict(text: str):
    def hook(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise VerificationError(f"duplicate JSON key {k!r}")
            out[k] = v
        return out
    return json.loads(text, object_pairs_hook=hook)


def load_json(p: Path):
    return _strict(Path(p).read_text())


def npz(p: Path) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def kfold_ids(n: int, seed: int, k: int = 5) -> np.ndarray:
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    sizes = np.full(k, n // k, dtype=np.int64)
    sizes[: n % k] += 1
    out = np.empty(n, dtype=np.int8)
    start = 0
    for f, s in enumerate(sizes):
        out[idx[start:start + s]] = f
        start += s
    return out


def class_of(t, edges) -> np.ndarray:
    return (np.asarray(t, dtype=np.int64)[:, None] > np.asarray(edges, dtype=np.int64)[None, :]).sum(1).astype(np.int8)


def select_edges(t, k: int, min_fraction: float):
    v, c = np.unique(np.asarray(t, dtype=np.int64), return_counts=True)
    n = int(c.sum())
    need = math.ceil(min_fraction * n)
    cum = np.cumsum(c).tolist()
    best = None
    for idx in itertools.combinations(range(v.size - 1), k - 1):
        cs = [cum[j] for j in idx]
        counts = [cs[0]] + [cs[j] - cs[j - 1] for j in range(1, k - 1)] + [n - cs[-1]]
        if min(counts) < need:
            continue
        cand = (sum(abs(k * cs[j] - (j + 1) * n) for j in range(k - 1)), tuple(int(v[j]) for j in idx))
        if best is None or cand < best:
            best = cand
    return None if best is None else list(best[1])


def coverage_ok(y, folds, k: int, n_folds: int, min_fraction: float, min_per_fold: int) -> tuple[bool, dict]:
    y = np.asarray(y)
    totals = [int(np.count_nonzero(y == c)) for c in range(k)]
    ok = sum(totals) == y.size and min(totals) >= math.ceil(min_fraction * y.size)
    rec = {"class_counts": totals, "folds": []}
    for f in range(n_folds):
        te = folds == f
        tr = [int(np.count_nonzero((y == c) & ~te)) for c in range(k)]
        ts = [int(np.count_nonzero((y == c) & te)) for c in range(k)]
        ok = ok and min(tr) >= min_per_fold and min(ts) >= min_per_fold
        rec["folds"].append({"fold": f, "train": tr, "test": ts})
    return ok, rec


def scaler(X: np.ndarray):
    n = X.shape[0]
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    eps = np.finfo(np.float64).eps
    const = var <= n * eps * var + (n * mean * eps) ** 2
    scale = np.sqrt(var)
    scale[const] = 1.0
    return mean, scale


def gradient(Z, y, k, coef, b, C=1.0) -> float:
    n = Z.shape[0]
    if k == 2:
        z = Z @ coef[0] + b[0]
        r = 1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0))) - (y == 1)
        g = np.concatenate([Z.T @ r / n + coef[0] / (C * n), [r.mean()]])
    else:
        L = Z @ coef.T + b
        L = L - L.max(axis=1, keepdims=True)
        P = np.exp(L)
        P = P / P.sum(axis=1, keepdims=True)
        P[np.arange(n), y] -= 1.0
        g = np.concatenate([(P.T @ Z / n + coef / (C * n)).ravel(), P.mean(axis=0)])
    return float(np.max(np.abs(g)))


def predict(Z, k, coef, b):
    if k == 2:
        z = Z @ coef[0] + b[0]
        return (z > 0).astype(np.int64), np.abs(z)
    L = Z @ coef.T + b
    top = np.sort(L, axis=1)
    return np.argmax(L, axis=1).astype(np.int64), top[:, -1] - top[:, -2]


def bal_acc(y, p, k) -> float:
    return float(np.mean([np.count_nonzero(p[y == c] == c) / np.count_nonzero(y == c) for c in range(k)]))


def bal_frac(y, p, k) -> Fraction:
    return sum((Fraction(int(np.count_nonzero(p[y == c] == c)), int(np.count_nonzero(y == c))) for c in range(k)),
               Fraction(0)) / k


def group_permutations(n: int, seed: int, b_count: int) -> tuple[np.ndarray, bool]:
    m = b_count + 1
    order = np.random.Generator(np.random.PCG64(seed)).permutation(n)
    lengths, rest = [m] * (n // m), n % m
    for d in sorted((d for d in range(1, m + 1) if m % d == 0), reverse=True):
        lengths += [d] * (rest // d)
        rest %= d
    gen = np.arange(n)
    pos = 0
    for ln in lengths:
        c = order[pos:pos + ln]
        gen[c] = c[(np.arange(ln) + 1) % ln]
        pos += ln
    powers, cur = [], np.arange(n)
    for _ in range(b_count):
        cur = gen[cur]
        powers.append(cur.copy())
    closes = np.array_equal(gen[cur], np.arange(n))
    return np.stack(powers), closes


def observed_rank(obs: Fraction, perm: list, tb) -> int:
    t = [int(v) for v in tb]
    return 1 + sum(1 for j, s in enumerate(perm) if s < obs or (s == obs and t[j + 1] < t[0]))


def sign_test(values) -> dict:
    x = [float(v) for v in values]
    n = len(x)
    pos, neg = sum(v > 0 for v in x), sum(v < 0 for v in x)
    base = {"n_valid": n, "n_positive": pos, "n_nonpositive": n - pos, "n_negative": neg, "n_zero": n - pos - neg}
    if n == 0:
        return {**base, "status": NO_EFFECTIVE, "p": None}
    return {**base, "status": "TESTED", "p": Fraction(sum(math.comb(n, j) for j in range(pos, n + 1)), 2 ** n)}


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(0, k + 1)) if k >= 0 else 0.0


def cp(k: int, n: int, conf: float = 0.95) -> list[float]:
    a = (1 - conf) / 2

    def bisect(above):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if above(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else bisect(lambda p: 1.0 - binom_cdf(k - 1, n, p) < a)
    upper = 1.0 if k == n else bisect(lambda p: binom_cdf(k, n, p) > a)
    return [lower, upper]


def decide(tests: dict, alpha: float, n_valid: int, n_required: int, precheck_ok: bool) -> str:
    if not precheck_ok:
        return "PRECHECK_FAILED"
    if n_valid < n_required:
        return "INCONCLUSIVE"
    a = Fraction(str(alpha))
    for step, parts in (("C", ("C_task", "C_twin")), ("S1", ("S1",)), ("S2", ("S2",))):
        if any(tests[p]["status"] == NO_EFFECTIVE for p in parts):
            return "INCONCLUSIVE"
        if not all(tests[p]["p"] < a for p in parts):
            return {"C": "POSITIVE_CONTROL_FAILED", "S1": "INCONCLUSIVE", "S2": "TARGET_DECODABLE_TWIN_UNRESOLVED"}[step]
    return "SUPPORTED"


def bits(words) -> np.ndarray:
    return V2._words_to_bits(np.asarray(words))


def verify_journal(out: Path, man: dict, f: Findings) -> dict:
    data = (out / "journal.jsonl").read_bytes()
    f.check("journal_ends_with_complete_line", data.endswith(b"\n"))
    entries, prev, chain = [], GENESIS, []
    for i, ln in enumerate(data.splitlines()):
        e = json.loads(ln)
        if e.get("seq") != i or e.get("prev_sha256") != prev:
            chain.append(i)
        if e.get("experiment_id") != man["experiment_id"] or e.get("schema") != JOURNAL_SCHEMA:
            chain.append(i)
        prev = hashlib.sha256(ln).hexdigest()
        entries.append(e)
    f.check("journal_chain_sequence_and_identity", not chain, chain[:10])
    init = [e for e in entries if e["event"] == "initialised"]
    f.check("run_manifest_hash_matches_initialised_event",
            len(init) == 1 and init[0].get("run_manifest_sha256") == sha256_file(out / "run_manifest.json"))
    ids = set(man["run_ids"])
    intents, problems = {}, []
    replay = {r: {"state": "NOT_STARTED", "fields": {}, "history": [], "seq": None, "artifacts": set()} for r in ids}
    for e in entries:
        rid = e.get("run_id")
        if rid is not None and rid not in ids:
            problems.append(f"unknown run {rid}")
            continue
        if e["event"] == "artifact_intent":
            intents[(rid, e["token"])] = e["artifact"]
        elif e["event"] == "run_transition":
            r = replay[rid]
            frm, to = e.get("from"), e.get("to")
            if frm != r["state"] or r["state"] in TERMINAL or to not in PERMITTED.get(frm, ()):
                problems.append(f"seq {e['seq']}: {rid} {frm} -> {to} (replayed {r['state']})")
            r["fields"].update(e.get("fields", {}))
            r["state"], r["seq"] = to, e["seq"]
            r["history"].append({"state": to, "seq": e["seq"], "utc": e["utc"]})
            for key, name in ARTIFACT_FILE.items():
                tok = e["fields"].get(f"{key}_token")
                if tok is None:
                    continue
                if intents.get((rid, tok)) != name:
                    problems.append(f"{rid}: {name} committed without a matching intent")
                p = out / "runs" / rid / name
                if not p.is_file() or sha256_file(p) != e["fields"].get(f"{key}_sha256"):
                    problems.append(f"{rid}: committed {name} missing or hash differs")
                r["artifacts"].add(name)
    f.check("journal_replay_state_machine_and_commits", not problems, problems[:10])
    quarantined = {e["destination"]: e["sha256"] for e in entries if e["event"] == "artifact_quarantined"}
    stray, rec_bad = [], []
    for rid in ids:
        rdir = out / "runs" / rid
        r = replay[rid]
        if rdir.exists():
            for p in sorted(rdir.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(out).as_posix()
                if p.parent == rdir and (p.name in r["artifacts"] or p.name == "record.json"):
                    continue
                if p.parent.name == "quarantine" and quarantined.get(rel) == sha256_file(p):
                    continue
                stray.append(rel)
        cp_ = rdir / "record.json"
        if r["seq"] is None:
            if cp_.exists():
                rec_bad.append(rid)
            continue
        if not cp_.exists():
            rec_bad.append(rid)
            continue
        cached = json.loads(cp_.read_text())
        expect = {**r["fields"], "state": r["state"], "history": r["history"], "journal_seq": r["seq"], "run_id": rid}
        if any(cached.get(k) != expect.get(k) for k in set(expect) | (set(cached) - {"schema"})):
            rec_bad.append(rid)
    f.check("every_run_file_committed_cached_or_quarantined", not stray, stray[:10])
    f.check("record_caches_equal_journal_replay", not rec_bad, rec_bad[:10])
    return {rid: {**r["fields"], "state": r["state"]} for rid, r in replay.items()}


def check_layer(model, rep: dict) -> bool:
    L = model.layers
    if len(L) != rep["n_layers"]:
        return False
    def desc(i):
        c = L[i].get_config()
        return type(L[i]).__name__, c.get("activation"), c.get("units")
    return (desc(rep["layer_index"])[:2] == (rep["layer_class"], rep["activation"])
            and list(L[rep["layer_index"]].output_shape) == rep["output_shape"]
            and desc(rep["inbound"]["index"])[0] == rep["inbound"]["class"]
            and desc(rep["pre_activation_dense"]["index"]) == (rep["pre_activation_dense"]["class"],
                                                                rep["pre_activation_dense"]["activation"],
                                                                rep["pre_activation_dense"]["units"])
            and desc(rep["consumer"]["index"]) == (rep["consumer"]["class"], rep["consumer"]["activation"],
                                                    rep["consumer"]["units"])
            and L[-1] is L[rep["consumer"]["index"]]
            and [n.outbound_layer for n in L[rep["layer_index"]]._outbound_nodes] == [L[rep["consumer"]["index"]]])


def verify_p1(design: dict, f: Findings) -> None:
    pin = design["validation_evidence"]["p1"]
    p = CRYPTO_ROOT / pin["relpath"]
    ok = p.is_file() and sha256_file(p) == pin["sha256"]
    f.check("p1_record_hash_matches_design_pin", ok)
    if not ok:
        return
    rec = load_json(p)
    dp = p.parent / rec["data_file"]
    f.check("p1_data_hash_matches_record", sha256_file(dp) == rec["data_sha256"])
    a = npz(dp)
    xs0, ys0 = V2._trajectory(a["key_words"], a["pt0"], ROUNDS)
    xs1, ys1 = V2._trajectory(a["key_words"], a["pt1"], ROUNDS)
    dx, dy = xs0 ^ xs1, ys0 ^ ys1
    w = np.stack([V2.xdp_weight(V2._ror(dx[:, r], 7), dy[:, r], dx[:, r + 1]) for r in range(ROUNDS)], 1)
    t = -w.sum(1)
    f.check("p1_input_difference", bool(np.all(dx[:, 0] == DX) and np.all(dy[:, 0] == DY)))
    f.check("p1_target_recomputed_with_independent_cipher_and_transfer_matrix",
            bool(np.all(w >= 0)) and np.array_equal(t, a["target"]))
    tg = design["target"]
    edges = select_edges(t, tg["n_classes"], tg["min_class_fraction"])
    f.check("p1_edges_independently_reselected_equal_design",
            edges == tg["edges"] == rec["selected_edges"] and rec["selected_n_classes"] == tg["n_classes"],
            {"reselected": edges, "design": tg["edges"], "record": rec["selected_edges"]})
    f.check("p1_five_classes_feasible_so_no_fallback", select_edges(t, 5, tg["min_class_fraction"]) is not None
            and tg["n_classes"] == 5)
    f.check("p1_non_evidentiary_model_free", rec.get("evidentiary") is False and rec.get("model_loaded") is False)


def verify(out: Path, *, prereg_path: Path | None = None, upstream: Path | None = None, recompute_models: bool = True,
           sklearn_refit: bool = False) -> dict:
    out = Path(out)
    f = Findings()
    man = load_json(out / "run_manifest.json")
    design = load_json(out / "design.json")
    base = {k: v for k, v in design.items() if k != "smoke_overrides"}
    f.check("design_file_hash_matches_manifest", sha256_file(out / "design.json") == man["design_file_sha256"])
    f.check("run_design_hash", canonical_sha256(design) == man["run_design_sha256"])
    f.check("base_design_hash", canonical_sha256(base) == man["design_sha256"])
    mode = man["mode"]
    f.check("mode_and_overrides_consistent", (mode == "production") == ("smoke_overrides" not in design)
            and man["evidentiary"] == (mode == "production"))
    if mode == "production":
        pp = Path(prereg_path or (CRYPTO_ROOT / "experiments/autond/ce3/preregistration/preregistration.json"))
        pr = load_json(pp)
        a = pr.get("approval") or {}
        f.check("preregistration_design_equals_run_design", pr.get("design") == base
                and pr.get("design_sha256") == man["design_sha256"])
        f.check("preregistration_file_hash_matches_manifest", sha256_file(pp) == (man.get("preregistration") or {}).get("sha256"))
        f.check("preregistration_approved", a.get("status") == "APPROVED" and a.get("approved_design_sha256") ==
                man["design_sha256"] and bool(a.get("approved_by")) and bool(a.get("approved_utc"))
                and base.get("status") == "FROZEN")
    deps = design["dependencies"]["sha256"]
    bad = [r for r, h in deps.items() if not (CRYPTO_ROOT / r).is_file() or sha256_file(CRYPTO_ROOT / r) != h]
    f.check("pinned_dependencies_unchanged", not bad, bad)
    src_bad = [n for n, h in man["producer_sources"].items() if sha256_file(CE3_ROOT / n) != h]
    f.check("producer_sources_unchanged_since_initialisation", not src_bad, src_bad)
    ce2 = design["model"]["untrained_twin"]["source"]
    croot = CRYPTO_ROOT / ce2["relpath"]
    f.check("ce2_source_certificate_manifest_verification_hashes",
            sha256_file(croot / "certificate.json") == ce2["certificate_sha256"]
            and sha256_file(croot / "run_manifest.json") == ce2["run_manifest_sha256"]
            and sha256_file(croot / ce2["verification_report"]) == ce2["verification_report_sha256"])
    cv = load_json(croot / ce2["verification_report"])
    f.check("ce2_source_verification_passed_exact", cv.get("passed") is True and cv.get("failures") == []
            and cv.get("numerical_status") == "EXACT" and cv.get("certificate_sha256") == ce2["certificate_sha256"])
    tw = man.get("untrained_twin") or {}
    f.check("twin_copy_equals_ce2_twin", bool(tw) and sha256_file(out / tw["file"]) == ce2["twin_file_sha256"]
            == sha256_file(croot / ce2["twin_file"]) and tw.get("weights_sha256") == ce2["twin_weights_sha256"])
    verify_p1(design, f)
    pins = {}
    for k in ("power", "benchmark", "representation_diagnostics"):
        pin = design["validation_evidence"][k]
        p = CRYPTO_ROOT / pin["relpath"]
        pins[k] = p.is_file() and sha256_file(p) == pin["sha256"]
    if mode == "production":
        f.check("validation_evidence_pins", all(pins.values()), pins)
        if pins["benchmark"]:
            b = load_json(CRYPTO_ROOT / design["validation_evidence"]["benchmark"]["relpath"])
            f.check("benchmark_consistent_with_design", b.get("selected_max_iter") == design["probe"]["logistic"]["max_iter"]
                    and b.get("scores_computed") is False and b.get("all_converged") is True)
    f.check("preflight_hash_matches_manifest", sha256_file(out / "preflight.json") == man["preflight_sha256"])
    pre = load_json(out / "preflight.json")
    precheck_ok = bool(pre["ok"]) and man.get("status") == "INITIALISED"
    cert_p = out / "certificate.json"
    cert = load_json(cert_p) if cert_p.exists() else None
    f.check("certificate_present", cert is not None)
    if cert:
        listed = cert["evidence_manifest"]
        mism = [p for p, h in listed.items() if not (out / p).is_file() or sha256_file(out / p) != h]
        present = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "certificate.json"
                   and not p.name.startswith("verification_") and ".staging" not in p.parts}
        f.check("evidence_manifest_hashes", not mism, mism[:10])
        f.check("no_unlisted_evidence_files", present == set(listed), sorted(present ^ set(listed))[:10])
    records = verify_journal(out, man, f)

    tg, rep, pr_cfg = design["target"], design["model"]["representation"], design["probe"]
    sm = design.get("smoke_overrides", {})
    n_perm = man["sizes"]["permutations"]
    min_fold = man["sizes"]["min_class_count_per_fold"]
    seeds = design["seeds"]
    kappa = design["endpoints"]["kappa"]
    gtol = pr_cfg["gradient_tolerance"]
    f.check("manifest_target_and_probe_match_design", man["edges"] == tg["edges"] and man["n_classes"] == tg["n_classes"]
            and man["max_iter"] == pr_cfg["logistic"]["max_iter"] and n_perm % 2 == 1
            and (not sm or sm["permutations_per_replicate"] == n_perm))

    mods, models = None, None
    if recompute_models:
        root = Path(upstream or man["upstream_root"])
        mods = V2.load_upstream(root, f)
        if mods is not None:
            import tensorflow as tf
            ck = root / CHECKPOINT_RELPATH
            f.check("published_checkpoint_sha256", sha256_file(ck) == design["model"]["checkpoint_sha256"])
            pub = mods["dbitnet"].make_model(64)
            pub.load_weights(str(ck))
            twin = mods["dbitnet"].make_model(64)
            twin.load_weights(str(out / tw["file"]))
            f.check("published_model_strict_hdf5_binding", V2.strict_matches_hdf5(pub, ck))
            f.check("twin_weights_hash", V2.weights_hash(twin) == ce2["twin_weights_sha256"])
            f.check("representation_layer_structure_published_and_twin", check_layer(pub, rep) and check_layer(twin, rep))
            models = {"pub": pub, "twin": twin, "pub_hash": V2.weights_hash(pub),
                      "ext_pub": tf.keras.Model(pub.input, pub.layers[rep["layer_index"]].output),
                      "ext_twin": tf.keras.Model(twin.input, twin.layers[rep["layer_index"]].output)}

    def run_model(m, X):
        return np.asarray(m.predict(X, batch_size=5000, verbose=0), dtype=np.float32)

    endpoints, states, numerics = {}, {}, {}
    for i, rid in enumerate(man["run_ids"]):
        rec = records[rid]
        states[rid] = rec["state"]
        if rec["state"] not in ("VALID", "EXCLUDED"):
            continue
        rd = out / "runs" / rid
        Dd, Cd, Rp, Pz = npz(rd / "data.npz"), npz(rd / "calibration.npz"), npz(rd / "representations.npz"), npz(rd / "probes.npz")
        M = load_json(rd / "measurement.json")
        n, nc = Dd["target"].shape[0], man["sizes"]["calibration_per_class"]
        f.check(f"{rid}:sizes", n == man["sizes"]["main_pairs"] and Cd["labels"].size == 2 * nc)
        f.check(f"{rid}:content_hashes", content_hash(Dd) == rec.get("data_content_sha256")
                and content_hash(Cd) == rec.get("calibration_content_sha256")
                and content_hash(Rp) == rec.get("representations_content_sha256")
                and content_hash(Pz) == rec.get("probes_content_sha256"))
        ok = True
        for nm, arr in (("main", Dd), ("cal", Cd)):
            xs0, ys0 = V2._trajectory(arr["key_words"], arr["pt0"], ROUNDS)
            xs1, ys1 = V2._trajectory(arr["key_words"], arr["pt1"], ROUNDS)
            ok = ok and bool(np.all((arr["pt0"][:, 0] ^ arr["pt1"][:, 0]) == DX) and np.all((arr["pt0"][:, 1] ^ arr["pt1"][:, 1]) == DY))
            ok = ok and np.array_equal(np.stack([xs0[:, -1], ys0[:, -1]], 1), arr["c0"]) and \
                np.array_equal(np.stack([xs1[:, -1], ys1[:, -1]], 1), arr["c1"])
            if nm == "main":
                dx, dy = xs0 ^ xs1, ys0 ^ ys1
                w = np.stack([V2.xdp_weight(V2._ror(dx[:, r], 7), dy[:, r], dx[:, r + 1]) for r in range(ROUNDS)], 1)
                f.check(f"{rid}:trajectory_weights_target", np.array_equal(dx, Dd["dx"]) and np.array_equal(dy, Dd["dy"])
                        and bool(np.all(w >= 0)) and np.array_equal(w, Dd["weights"]) and np.array_equal(-w.sum(1), Dd["target"]))
        f.check(f"{rid}:ciphertexts_and_input_difference_own_cipher", ok)
        y = class_of(Dd["target"], tg["edges"])
        f.check(f"{rid}:classes_from_frozen_edges", np.array_equal(y, Dd["classes"]))
        order = np.random.Generator(np.random.PCG64(seeds["calibration_shuffle"] + i)).permutation(2 * nc)
        yc = np.concatenate([np.ones(nc, np.int8), np.zeros(nc, np.int8)])[order]
        f.check(f"{rid}:calibration_order_labels_balance", np.array_equal(order, Cd["order"]) and np.array_equal(yc, Cd["labels"]))
        X = np.hstack([bits(Dd["c0"]), bits(Dd["c1"])])
        Xc = np.concatenate([np.hstack([bits(Cd["c0"]), bits(Cd["c1"])]),
                             np.unpackbits(Cd["random_inputs_packed"], axis=1)[:, :64]])[order]
        f.check(f"{rid}:model_input_hashes", content_hash({"X": X}) == rec.get("model_input_sha256")
                and content_hash({"X": Xc}) == rec.get("calibration_input_sha256"))
        fm = kfold_ids(n, seeds["main_folds"] + i)
        fc = kfold_ids(2 * nc, seeds["calibration_folds"] + i)
        perms, closes = group_permutations(n, seeds["permutations"] + i, n_perm)
        tb = Dd["s1_tiebreak"]
        f.check(f"{rid}:permutation_group_closes", closes)
        f.check(f"{rid}:s1_tiebreak_is_a_permutation_of_assignments", tb.shape == (n_perm + 1,)
                and sorted(int(v) for v in tb) == list(range(n_perm + 1)))
        f.check(f"{rid}:folds_reconstructed_label_independent", np.array_equal(fm, Pz["fold_main"]) and np.array_equal(fc, Pz["fold_cal"]))
        f.check(f"{rid}:permutations_reconstructed", np.array_equal(perms, Pz["perm_index"].astype(np.int64)))
        cov_all = {"A": coverage_ok(y, fm, tg["n_classes"], 5, tg["min_class_fraction"], min_fold)}
        for b in range(n_perm):
            cov_all[f"P{b:02d}"] = coverage_ok(y[perms[b]], fm, tg["n_classes"], 5, tg["min_class_fraction"], min_fold)
        cov_all["CAL"] = coverage_ok(yc, fc, 2, 5, tg["min_class_fraction"], min_fold)
        stored_cov = rec.get("coverage", {})
        f.check(f"{rid}:class_coverage_every_arm_and_permutation", all(v[0] for v in cov_all.values())
                and all(stored_cov.get(k, {}).get("class_counts") == v[1]["class_counts"]
                        and stored_cov.get(k, {}).get("folds") == v[1]["folds"] for k, v in cov_all.items()))
        nz = {k: int(np.count_nonzero(np.var(Rp[k].astype(np.float64), axis=0) > 0)) for k in ("twin_main", "twin_cal")}
        f.check(f"{rid}:twin_admissible", min(nz.values()) >= 2 and rec.get("twin_nonconstant") ==
                {"main": nz["twin_main"], "cal": nz["twin_cal"]})
        reps = {k: Rp[k] for k in ("trained_main", "twin_main", "trained_cal", "twin_cal")}
        num = "EXACT"
        if models is not None:
            W, bb = models["pub"].layers[rep["consumer"]["index"]].get_weights()
            for nm, h, sc in (("main", Rp["trained_main"], Rp["score_main"]), ("cal", Rp["trained_cal"], Rp["score_cal"])):
                z = h.astype(np.float64) @ W[:, 0].astype(np.float64) + float(bb[0])
                f.check(f"{rid}:head_consistency_{nm}", float(np.max(np.abs(1 / (1 + np.exp(-z)) - sc))) <= rep["head_tolerance"])
            new = {"trained_main": run_model(models["ext_pub"], X), "twin_main": run_model(models["ext_twin"], X),
                   "trained_cal": run_model(models["ext_pub"], Xc), "twin_cal": run_model(models["ext_twin"], Xc)}
            sm_new = {"score_main": run_model(models["pub"], X).ravel(), "score_cal": run_model(models["pub"], Xc).ravel()}
            diffs = {k: float(np.max(np.abs(new[k] - Rp[k]))) for k in new}
            diffs.update({k: float(np.max(np.abs(sm_new[k] - Rp[k]))) for k in sm_new})
            if not all(np.array_equal(new[k], Rp[k]) for k in new) or not all(np.array_equal(sm_new[k], Rp[k]) for k in sm_new):
                num = "DIFFERENT"
            f.check(f"{rid}:representations_reextracted", max(diffs.values()) <= 1e-5, diffs)
            f.check(f"{rid}:published_weights_hash_matches_record", models["pub_hash"] == rec.get("published_weights_sha256"))
            reps_new = new
        else:
            reps_new = None
        arms = [("A", "trained_main", y, fm, tg["n_classes"])]
        arms += [(f"P{b:02d}", "trained_main", y[perms[b]], fm, tg["n_classes"]) for b in range(n_perm)]
        arms += [("T", "twin_main", y, fm, tg["n_classes"]), ("R", None, y, fm, tg["n_classes"]),
                 ("CAL_trained", "trained_cal", yc.astype(np.int64), fc, 2), ("CAL_twin", "twin_cal", yc.astype(np.int64), fc, 2)]
        ba, ba_new, fold_ba, grad_max, margin_mismatch, iters_ok, scaler_bad = {}, {}, {}, 0.0, 0, True, []
        exact, exact_new = {}, {}
        class_sum_max = 0.0
        for name, key, labels, folds, k in arms:
            Xf = X.astype(np.float64) if key is None else reps[key].astype(np.float64)
            Xn = None if (key is None or reps_new is None) else reps_new[key].astype(np.float64)
            labels = np.asarray(labels, dtype=np.int64)
            stored_pred = Pz[f"{name}__pred"].astype(np.int64)
            own_pred = np.empty_like(stored_pred)
            new_pred = np.empty_like(stored_pred)
            fb = []
            for fo in range(5):
                te, tr = folds == fo, folds != fo
                mean, scale = scaler(Xf[tr])
                sm_, ss_ = Pz[f"{name}__mean"][fo], Pz[f"{name}__scale"][fo]
                if not (np.allclose(sm_, mean, rtol=1e-9, atol=1e-12) and np.allclose(ss_, scale, rtol=1e-9, atol=1e-12)):
                    scaler_bad.append(f"{name}:{fo}")
                coef, inter = Pz[f"{name}__coef"][fo].astype(np.float64), Pz[f"{name}__intercept"][fo].astype(np.float64)
                g = gradient((Xf[tr] - sm_) / ss_, labels[tr], k, coef, inter)
                if k > 2:
                    class_sum_max = max(class_sum_max, float(np.max(np.abs(coef.sum(axis=0)))))
                grad_max = max(grad_max, g)
                iters_ok = iters_ok and int(Pz[f"{name}__n_iter"][fo]) < pr_cfg["logistic"]["max_iter"]
                p_own, marg = predict((Xf[te] - sm_) / ss_, k, coef, inter)
                own_pred[te] = p_own
                margin_mismatch += int(np.count_nonzero((p_own != stored_pred[te]) & (marg >= MARGIN)))
                if Xn is not None:
                    new_pred[te] = predict((Xn[te] - sm_) / ss_, k, coef, inter)[0]
                fb.append(bal_acc(labels[te], stored_pred[te], k))
            ba[name] = bal_acc(labels, stored_pred, k)
            fold_ba[name] = fb
            ba_new[name] = bal_acc(labels, new_pred, k) if Xn is not None else ba[name]
            if name == "A" or name.startswith("P"):
                exact[name] = bal_frac(labels, stored_pred, k)
                exact_new[name] = bal_frac(labels, new_pred, k) if Xn is not None else exact[name]
            if Xn is not None and not np.array_equal(new_pred, stored_pred):
                num = "DECISION_CHECK_REQUIRED"
            if sklearn_refit:
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from threadpoolctl import threadpool_limits
                cfg = {kk: vv for kk, vv in pr_cfg["logistic"].items() if kk != "class"}
                same = True
                for fo in range(5):
                    te, tr = folds == fo, folds != fo
                    with threadpool_limits(limits=pr_cfg["blas_threads"]):
                        est = Pipeline([("s", StandardScaler()), ("l", LogisticRegression(**cfg))]).fit(Xf[tr], labels[tr])
                        same = same and np.array_equal(est.predict(Xf[te]).astype(np.int64), stored_pred[te])
                f.check(f"{rid}:{name}:sklearn_refit_predictions_identical", same)
        f.check(f"{rid}:scalers_fitted_on_training_folds_only", not scaler_bad, scaler_bad[:10])
        f.check(f"{rid}:stored_probe_parameters_satisfy_objective_gradient", grad_max <= gtol, {"max_gradient": grad_max})
        f.check(f"{rid}:iterations_below_max_iter", iters_ok)
        f.check(f"{rid}:multinomial_coefficients_sum_to_zero_across_classes", class_sum_max <= CLASS_SUM_TOLERANCE,
                {"max_abs_class_sum": class_sum_max})
        f.check(f"{rid}:predictions_follow_from_stored_parameters", margin_mismatch == 0, {"mismatches": margin_mismatch})
        f.check(f"{rid}:balanced_accuracies_recomputed", all(abs(ba[a] - M["balanced_accuracy"][a]) <= 1e-12 for a in ba)
                and all(np.allclose(fold_ba[a], M["fold_balanced_accuracy"][a], rtol=0, atol=1e-12) for a in fold_ba))
        score_cal = Rp["score_cal"] if models is None else sm_new["score_cal"]
        ba_m = bal_acc(yc.astype(np.int64), (score_cal > 0.5).astype(np.int64), 2)

        def eps(b, bm, ex):
            pm = float(np.median([b[f"P{j:02d}"] for j in range(n_perm)]))
            r = observed_rank(ex["A"], [ex[f"P{j:02d}"] for j in range(n_perm)], tb)
            return {"C_task": b["CAL_trained"] - 0.5 - kappa * max(bm - 0.5, 0.0), "C_twin": b["CAL_trained"] - b["CAL_twin"],
                    "S1": r - (n_perm + 2) / 2, "S2": b["A"] - b["T"], "raw_gap": b["A"] - b["R"], "S1_margin": b["A"] - pm}
        e = eps(ba, bal_acc(yc.astype(np.int64), (Rp["score_cal"] > 0.5).astype(np.int64), 2), exact)
        sr = M.get("s1_rank", {})
        f.check(f"{rid}:s1_rank_recomputed_exactly", sr.get("rank") == int(e["S1"] + (n_perm + 2) / 2)
                and sr.get("tiebreak") == [int(v) for v in tb]
                and sr.get("ties_with_observed") == sum(1 for j in range(n_perm) if exact[f"P{j:02d}"] == exact["A"])
                and M.get("balanced_accuracy_exact") == {kk: [v.numerator, v.denominator] for kk, v in exact.items()})
        f.check(f"{rid}:endpoints_recomputed", all(abs(e[k] - M["endpoints"][k]) <= 1e-12 for k in e)
                and (rec["state"] != "VALID" or rec.get("endpoints") == M["endpoints"]))
        e_new = eps(ba_new, ba_m, exact_new)
        numerics[rid] = {"status": num, "signs_identical": all(np.sign(e_new[k]) == np.sign(e[k])
                                                               for k in ("C_task", "C_twin", "S1", "S2"))}
        if num != "EXACT":
            f.check(f"{rid}:recomputed_endpoint_signs_identical", numerics[rid]["signs_identical"])
        finite = all(math.isfinite(v) for v in e.values())
        f.check(f"{rid}:state_matches_finiteness", (rec["state"] == "VALID") == finite)
        if rec["state"] == "VALID":
            endpoints[rid] = {"stored": e, "recomputed": e_new}

    n_req = man["sizes"]["n_required_valid"]
    alpha = design["statistics"]["alpha"]
    valid = sorted(r for r, s in states.items() if s == "VALID")
    for which in ("stored", "recomputed"):
        tests = {k: sign_test([endpoints[r][which][k] for r in valid]) for k in ("C_task", "C_twin", "S1", "S2")}
        dec = decide(tests, alpha, len(valid), n_req, precheck_ok)
        if which == "stored":
            stored_dec, stored_tests = dec, tests
        else:
            f.check("decision_invariant_under_recomputed_representations", dec == stored_dec)
    if cert:
        an = cert["analysis"]
        ok = cert["decision"] == stored_dec and an["n_valid"] == len(valid)
        for k, t in stored_tests.items():
            c = an["endpoints"][k]["test"]
            ok = ok and all(c[x] == t[x] for x in ("n_valid", "n_positive", "n_nonpositive", "n_negative", "n_zero",
                                                    "status"))
            if t["status"] == "TESTED":
                ok = ok and Fraction(c["p_numerator"], c["p_denominator"]) == t["p"]
                ok = ok and all(abs(a - b) <= 1e-9 for a, b in zip(c["ci95"], cp(t["n_positive"], t["n_valid"])))
            else:
                ok = ok and c["p_value"] is None
        f.check("certificate_statistics_and_decision_recomputed", ok, {"decision": stored_dec})
        f.check("certificate_run_states_match_journal", cert["run_states"] == states)
    status = "EXACT" if all(v["status"] == "EXACT" for v in numerics.values()) else (
        "DECISION_INVARIANT_NUMERICAL_DIFFERENCE" if all(v["signs_identical"] for v in numerics.values()) else
        "DECISION_RELEVANT_DIFFERENCE")
    return {"verifier": "audit/cryptography/experiments/autond/ce3/verify.py", "output_dir": str(out),
            "utc": datetime.now(timezone.utc).isoformat(), "mode": mode, "models_recomputed": models is not None,
            "sklearn_refit": sklearn_refit, "numerical_status": status, "decision_recomputed": stored_dec,
            "certificate_sha256": sha256_file(cert_p) if cert else None, "n_checks": len(f.checks),
            "failures": f.failures, "passed": not f.failures, "checks": f.checks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Independent verification of an AutoND CE3 output directory")
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--preregistration", type=Path, default=None)
    ap.add_argument("--upstream", type=Path, default=None)
    ap.add_argument("--skip-model-recompute", action="store_true")
    ap.add_argument("--sklearn-refit", action="store_true")
    a = ap.parse_args(argv)
    rep = verify(a.output_dir, prereg_path=a.preregistration, upstream=a.upstream,
                 recompute_models=not a.skip_model_recompute, sklearn_refit=a.sklearn_refit)
    path = a.output_dir / f"verification_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    path.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"{'VERIFIED' if rep['passed'] else 'VERIFICATION FAILED'}: {rep['n_checks']} checks, "
          f"{len(rep['failures'])} failures, {rep['numerical_status']} -> {path}")
    for name in rep["failures"]:
        print(f"  FAILED {name}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
