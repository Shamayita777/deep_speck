"""
CE2-CE4 production pipeline (isolated; CE1 is never read, written or imported
for execution).

    frozen plan  ->  model binding (hash + architecture + behaviour)
                 ->  fresh data generation at rounds=5, diff=(0x0040,0x0000)
                 ->  raw per-run observations persisted BEFORE any inference
                 ->  statistical analysis exactly as preregistered
                 ->  certificate bound to plan hash + model hash + raw hashes

Every raw observation is written to disk, so every reported statistic can be
recomputed by verify.py without importing anything from this module.

FAIL-CLOSED: no default round count, no default depth, no silent substitution
of a checkpoint, no generated stand-in for missing input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit.cryptography.ce234 import frozen_plan as P

CE234_ROOT = Path(__file__).resolve().parent
CRYPTO_ROOT = CE234_ROOT.parent
REPO_ROOT = CRYPTO_ROOT.parents[1]
REFERENCE_CHECKPOINT = CRYPTO_ROOT / "Archive" / "best5depth10.h5"
EVIDENCE_ROOT = CRYPTO_ROOT / "evidence_current"


class PreflightError(RuntimeError):
    """Fail-closed: a required precondition is absent or wrong."""


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        if a.size:
            h.update(memoryview(a.reshape(-1)).cast("B"))
    return h.hexdigest()


def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# =====================================================================
# Provenance
# =====================================================================

def environment() -> dict:
    import keras
    import scipy
    import sklearn
    import tensorflow as tf

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:                                   # noqa: BLE001
        commit, dirty = "", "UNAVAILABLE"
    return {
        "python": sys.version.split()[0], "tensorflow": tf.__version__,
        "keras": keras.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "gpus_visible": len(tf.config.list_physical_devices("GPU")),
        "git_commit": commit or "UNAVAILABLE",
        "git_working_tree": "dirty" if dirty and dirty != "UNAVAILABLE" else
                            ("clean" if commit else "UNAVAILABLE"),
        "command_line": " ".join(sys.argv),
        "cwd_is_repo_root": Path.cwd().resolve() == REPO_ROOT,
    }


def source_hashes() -> dict:
    """Hash every file whose contents define the experiment."""
    files = [CE234_ROOT / "frozen_plan.py", CE234_ROOT / "production.py",
             CE234_ROOT / "verify.py", CRYPTO_ROOT / "gohr" / "speck.py",
             CRYPTO_ROOT / "gohr" / "dataset.py", CRYPTO_ROOT / "gohr" / "model.py",
             CRYPTO_ROOT / "gohr" / "evaluate.py",
             CRYPTO_ROOT / "adapters" / "gohr.py",
             CRYPTO_ROOT / "probe" / "evaluation.py",
             CRYPTO_ROOT / "experiments" / "ce4" / "design.py",
             # the validation path and the matrix builder are part of the
             # evidence chain and must be hash-bound too
             CE234_ROOT / "build_matrices.py",
             CE234_ROOT / "tests" / "test_ce234.py"]
    return {rel(f): sha256_file(f) for f in files if f.exists()}


# =====================================================================
# Model binding - fail closed on ANY mismatch
# =====================================================================

def verify_reference_model(*, behavioural: bool = True, checkpoint=None) -> dict:
    """
    Bind CE2-CE4 to the frozen depth-10 reference model. Filenames are never
    trusted: the architecture is read out of the loaded graph, and the round
    count (which Keras does not store) is verified BEHAVIOURALLY on fresh data.
    """
    import keras

    from audit.cryptography.gohr import speck as sp

    ck = Path(checkpoint or REFERENCE_CHECKPOINT)
    if not ck.exists():
        raise PreflightError(f"reference checkpoint absent: {rel(ck)}")
    digest = sha256_file(ck)
    if digest != P.REFERENCE_CHECKPOINT_SHA256:
        raise PreflightError(
            f"checkpoint sha256 {digest} != frozen {P.REFERENCE_CHECKPOINT_SHA256}. "
            "Refusing to substitute a different model.")
    model = keras.models.load_model(ck, compile=False)
    props = {
        "conv1d_layers": sum(1 for l in model.layers if type(l).__name__ == "Conv1D"),
        "residual_merges": sum(1 for l in model.layers if type(l).__name__ == "Add"),
        "l2_values": sorted({round(float(l.kernel_regularizer.l2), 12)
                             for l in model.layers
                             if getattr(l, "kernel_regularizer", None) is not None}),
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
    }
    bad = {k: (props[k], v) for k, v in P.REQUIRED_MODEL_PROPERTIES.items()
           if props[k] != v}
    if bad:
        raise PreflightError(f"model architecture does not match the frozen design: {bad}")

    report = {"path": rel(ck), "sha256": digest, "properties": props,
              "realized_depth": props["residual_merges"]}
    if behavioural:
        b = P.BEHAVIOURAL_BINDING
        accs = {}
        for nr in [P.ROUNDS] + list(b["other_rounds"]):
            X, Y = sp.make_train_data(b["n_samples"], nr, diff=P.DIFFERENTIAL)
            pr = model.predict(X, batch_size=P.PREDICT_BATCH, verbose=0).ravel()
            accs[nr] = float(np.mean((pr > 0.5) == Y))
        if accs[P.ROUNDS] < b["min_accuracy_at_declared_rounds"]:
            raise PreflightError(
                f"behavioural binding failed: accuracy {accs[P.ROUNDS]:.4f} at "
                f"{P.ROUNDS} rounds is below {b['min_accuracy_at_declared_rounds']}; "
                "this model was not trained for the declared round count.")
        for nr in b["other_rounds"]:
            if accs[nr] > b["max_accuracy_at_other_rounds"]:
                raise PreflightError(
                    f"behavioural binding failed: accuracy {accs[nr]:.4f} at {nr} "
                    "rounds is too high; the round binding is ambiguous.")
        report["behavioural_binding"] = {
            "accuracy_by_rounds": {str(k): v for k, v in accs.items()},
            "declared_rounds": P.ROUNDS, "differential": list(P.DIFFERENTIAL),
            "gohr_reported_accuracy": b["reference_accuracy_reported_by_gohr"],
            "verdict": "PASS"}
    return report


def load_reference_model():
    import keras
    return keras.models.load_model(REFERENCE_CHECKPOINT, compile=False)


def untrained_twin(seed: int):
    """Identical architecture, never trained. Control for CE2 and CE3."""
    from audit.cryptography.gohr.model import GohrModel
    from audit.cryptography.gohr.trainer import GohrTrainer

    GohrTrainer.set_seed(seed)
    return GohrModel(depth=P.DEPTH, regularization=P.L2_REG).build()


# =====================================================================
# Data generation - explicit rounds/differential everywhere
# =====================================================================

GENERATOR_VERSION = "ce234-theory-generator-1"


def generate_theory_data(n: int, *, rounds: int = P.ROUNDS, differential=P.DIFFERENTIAL,
                         with_factors: bool = False, rng=None):
    """
    Real difference-bearing pairs plus their per-round Lipmaa-Moriai factors.

    Mirrors gohr.speck.estimate_trail_probabilities exactly (same cipher calls,
    same encoding) and additionally returns the per-round factors, which the
    preregistered CE2 secondary endpoints require. Verified against
    estimate_trail_probabilities in the test-suite.
    """
    from os import urandom

    from audit.cryptography.gohr import speck as S

    if rng is None:
        # PRODUCTION PATH. Gohr's generator draws from os.urandom, so exact
        # replay is unavailable; raw observations are persisted instead.
        keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
        p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    else:
        # TEST-ONLY deterministic path. Identical construction, replayable
        # draws. Never used by run_ce2/run_ce3/run_ce4, which never pass rng;
        # preflight asserts that (see production_generator_is_nondeterministic).
        keys = rng.integers(0, 1 << 16, size=(4, n), dtype=np.uint16)
        p0l = rng.integers(0, 1 << 16, size=n, dtype=np.uint16)
        p0r = rng.integers(0, 1 << 16, size=n, dtype=np.uint16)
    p1l, p1r = p0l ^ differential[0], p0r ^ differential[1]
    ks = S.expand_key(keys, rounds)
    x0, y0, x1, y1 = p0l.copy(), p0r.copy(), p1l.copy(), p1r.copy()
    factors = []
    for k in ks:
        a0, b0, a1, b1 = S.ror(x0, S.ALPHA()), y0, S.ror(x1, S.ALPHA()), y1
        gamma = ((a0 + b0) & S.MASK_VAL) ^ ((a1 + b1) & S.MASK_VAL)
        factors.append(S.xdp_plus(a0 ^ a1, b0 ^ b1, gamma))
        x0, y0 = S.enc_one_round((x0, y0), k)
        x1, y1 = S.enc_one_round((x1, y1), k)
    F = np.asarray(factors)                              # (rounds, n)
    feasible = (F > 0).all(axis=0)
    log2p = np.log2(np.where(F > 0, F, 1.0)).sum(axis=0)
    target = np.where(feasible, np.exp2(log2p), 0.0)
    X = S.convert_to_binary([x0, y0, x1, y1])
    return (X, target, F) if with_factors else (X, target)


def production_generator_is_nondeterministic() -> bool:
    """The three production entry points must never pass `rng=`."""
    import inspect
    bodies = "".join(inspect.getsource(f) for f in (run_ce2, run_ce3, run_ce4))
    return "generate_theory_data(" in bodies and "rng=" not in bodies.split(
        "generate_theory_data(")[1].split(")")[0]


def predict(model, X) -> np.ndarray:
    return model.predict(X, batch_size=P.PREDICT_BATCH, verbose=0).ravel()


# =====================================================================
# CE2
# =====================================================================

def run_ce2(out_dir: Path, *, n_runs: int, n_samples: int, seed: int,
            model=None, log=print) -> dict:
    from scipy.stats import spearmanr

    model = model or load_reference_model()
    control_model = untrained_twin(seed + 777)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for i in range(n_runs):
        t0 = time.time()
        rng = np.random.default_rng(seed + i)
        X, target, F = generate_theory_data(n_samples, with_factors=True)
        out = predict(model, X)
        rec = {"run_id": f"ce2_run{i:02d}", "evaluation_seed": seed + i,
               "n": int(n_samples), "rounds": P.ROUNDS,
               "differential": list(P.DIFFERENTIAL),
               "dataset_sha256": sha256_array(X), "target_sha256": sha256_array(target),
               "output_sha256": sha256_array(out)}
        if target.std() == 0 or out.std() == 0:
            rec.update(status="EXCLUDED", reason="constant target or output")
            runs.append(rec); continue
        rec["rho_primary"] = float(spearmanr(target, out).statistic)
        # preregistered controls
        perm = rng.permutation(target.size)
        rec["rho_C2_PERMUTED_TARGET"] = float(spearmanr(target[perm], out).statistic)
        cout = predict(control_model, X)
        rec["rho_C2_UNTRAINED_MODEL"] = (
            float(spearmanr(target, cout).statistic) if cout.std() > 0 else None)
        # preregistered secondary endpoints
        last = F[-1]
        prefix = np.exp2(np.log2(np.where(F[:-1] > 0, F[:-1], 1e-300)).sum(axis=0))
        rec["rho_S2_LAST_ROUND_ONLY"] = float(spearmanr(last, out).statistic)
        rec["rho_S2_PREFIX_ONLY"] = float(spearmanr(prefix, out).statistic)
        rec.update(status="OK", seconds=round(time.time() - t0, 2))
        np.savez_compressed(raw_dir / f"{rec['run_id']}.npz",
                            target=target.astype(np.float64),
                            output=out.astype(np.float32),
                            last_round_factor=last.astype(np.float64),
                            prefix_product=prefix.astype(np.float64),
                            permutation=perm.astype(np.int64),
                            control_output=cout.astype(np.float32))
        rec["raw_file"] = f"raw/{rec['run_id']}.npz"
        rec["raw_sha256"] = sha256_file(raw_dir / f"{rec['run_id']}.npz")
        runs.append(rec)
        log(f"  CE2 run {i+1}/{n_runs}: rho={rec['rho_primary']:+.4f} "
            f"({rec['seconds']}s)")
    return {"runs": runs, **analyse_ce2(runs)}


def analyse_ce2(runs) -> dict:
    valid = [r for r in runs if r.get("status") == "OK"]
    n = len(valid)
    if n < P.CE2.minimum_valid_runs:
        return {"decision": "INCONCLUSIVE", "n_valid_runs": n,
                "reason": f"{n} valid runs < minimum {P.CE2.minimum_valid_runs}"}
    primary = [r["rho_primary"] for r in valid]
    fam = {"primary": P.exact_sign_test(primary)["p_value"]}
    results = {"primary": {"rhos": primary, "median": float(np.median(primary)),
                           "mean": float(np.mean(primary)),
                           "ci95_bootstrap": P.bootstrap_ci(primary),
                           **P.exact_sign_test(primary)}}
    for key in ("rho_C2_PERMUTED_TARGET", "rho_C2_UNTRAINED_MODEL"):
        vals = [r[key] for r in valid if r.get(key) is not None]
        name = key.replace("rho_", "")
        results[name] = {"rhos": vals, "median": float(np.median(vals)) if vals else None,
                         **(P.exact_sign_test(vals) if vals else {})}
        fam[name] = results[name].get("p_value")
    for key in ("rho_S2_LAST_ROUND_ONLY", "rho_S2_PREFIX_ONLY"):
        vals = [r[key] for r in valid]
        results[key.replace("rho_", "")] = {
            "rhos": vals, "median": float(np.median(vals)),
            "role": "preregistered secondary endpoint, never primary"}
    adj = P.holm(fam)
    med = abs(results["primary"]["median"])
    separated = all(
        results[c]["median"] is None or abs(results[c]["median"]) < med / 2
        for c in ("C2_PERMUTED_TARGET", "C2_UNTRAINED_MODEL"))
    p_adj = adj.get("primary")
    if p_adj is not None and p_adj < P.ALPHA and separated:
        decision, reason = "SUPPORTED", (
            "Holm-adjusted primary sign test significant and both preregistered "
            "controls separated as specified")
    elif p_adj is not None and p_adj < P.ALPHA and not separated:
        decision, reason = "NOT_SUPPORTED", (
            "a preregistered control reproduces the association, so it is not "
            "attributable to the model/target pairing")
    else:
        decision, reason = "INCONCLUSIVE", (
            "primary not significant after Holm; this is NOT evidence that rho = 0")
    return {"decision": decision, "decision_reason": reason, "n_valid_runs": n,
            "endpoints": results, "holm_adjusted_p": adj,
            "multiplicity": P.WITHIN_CE_MULTIPLICITY, "alpha": P.ALPHA}


# =====================================================================
# CE3
# =====================================================================

def _representation(model, X) -> np.ndarray:
    import keras
    return np.asarray(keras.Model(model.input, model.layers[-2].output)
                      .predict(X, batch_size=P.PREDICT_BATCH, verbose=0))


def _degenerate(R: np.ndarray) -> bool:
    return int((R.std(axis=0) > 0).sum()) < 2


def run_ce3(out_dir: Path, *, n_replicates: int, n_samples: int, n_splits: int,
            seed: int, model=None, log=print) -> dict:
    from audit.cryptography.gohr import speck as sp
    from audit.cryptography.probe.evaluation import evaluate_selectivity
    from audit.cryptography.test.ce3.types import TargetSpecification, TargetType

    model = model or load_reference_model()
    control_model = untrained_twin(seed + 4242)
    raw_dir = out_dir / "raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    reps, cal = [], []
    for i in range(n_replicates):
        t0 = time.time()
        X, target = generate_theory_data(n_samples)
        q = np.quantile(target, [0.2, 0.4, 0.6, 0.8])
        labels = np.digitize(target, bins=q, right=False).astype(np.int64)
        spec = TargetSpecification(
            name="Analytical Trail Probability (quintile-discretised)",
            description="CE2 construct coarsened to 5 classes at its own quintiles.",
            target_type=TargetType.MULTICLASS, labels=labels,
            theoretical_interpretation=P.CE3.target_quantity)
        Rr, Rc = _representation(model, X), _representation(control_model, X)
        if _degenerate(Rc):
            raise PreflightError(
                "CE3 control representation is DEGENERATE (fewer than 2 features with "
                "non-zero variance). Refusing: a degenerate control measures chance, "
                "not a control.")
        ev = evaluate_selectivity(Rr, Rc, spec, n_splits=n_splits, n_repeats=1,
                                  seed=seed + i)
        # C3-RAW-INPUT: identical probe/folds on the raw ciphertext encoding
        raw_ev = evaluate_selectivity(X.astype(np.float32), Rc, spec,
                                      n_splits=n_splits, n_repeats=1, seed=seed + i)
        n_classes = int(len(np.unique(labels)))
        rec = {"replicate_id": f"ce3_rep{i:02d}", "seed": seed + i, "n": int(n_samples),
               "rounds": P.ROUNDS, "differential": list(P.DIFFERENTIAL),
               "n_classes": n_classes, "chance_level": 1.0 / n_classes,
               "twin_decodability_above_chance":
                   float(ev.control_score_mean - 1.0 / n_classes),
               "shared_examples_across_arms": True,
               "real_score": float(ev.real_score_mean),
               "control_score": float(ev.control_score_mean),
               "selectivity": float(ev.selectivity_mean),
               "raw_input_score": float(raw_ev.real_score_mean),
               "class_counts": np.bincount(labels, minlength=5).tolist(),
               "dataset_sha256": sha256_array(X), "status": "OK",
               "seconds": round(time.time() - t0, 2)}
        reps.append(rec)
        # calibration gate replicate (differential-class target, same pipeline)
        Xc, Yc = sp.make_train_data(n_samples, P.ROUNDS, diff=P.DIFFERENTIAL)
        cspec = TargetSpecification(
            name="Differential Class", description="Calibration positive control.",
            target_type=TargetType.BINARY, labels=Yc.astype(np.int64),
            theoretical_interpretation="methodological gate; not cryptographic evidence")
        cev = evaluate_selectivity(_representation(model, Xc),
                                   _representation(control_model, Xc), cspec,
                                   n_splits=n_splits, n_repeats=1, seed=seed + 10_000 + i)
        cal.append(float(cev.selectivity_mean))
        log(f"  CE3 replicate {i+1}/{n_replicates}: selectivity={rec['selectivity']:+.4f} "
            f"raw-input={rec['raw_input_score']:.4f} ({rec['seconds']}s)")
    np.savez_compressed(raw_dir / "ce3_replicates.npz",
                        selectivity=np.array([r["selectivity"] for r in reps]),
                        real_score=np.array([r["real_score"] for r in reps]),
                        control_score=np.array([r["control_score"] for r in reps]),
                        raw_input_score=np.array([r["raw_input_score"] for r in reps]),
                        calibration_selectivity=np.array(cal))
    return {"replicates": reps, "calibration_selectivity": cal,
            "raw_file": "raw/ce3_replicates.npz",
            "raw_sha256": sha256_file(raw_dir / "ce3_replicates.npz"),
            **analyse_ce3(reps, cal)}


def analyse_ce3(reps, calibration) -> dict:
    from audit.cryptography.statistics import replicate_level_summary

    valid = [r for r in reps if r.get("status") == "OK"]
    if len(valid) < P.CE3.minimum_valid_replicates:
        return {"decision": "INCONCLUSIVE",
                "reason": f"{len(valid)} valid replicates < "
                          f"{P.CE3.minimum_valid_replicates}"}
    cal_sum = replicate_level_summary(calibration, alpha=P.ALPHA)
    cal_p = (cal_sum["wilcoxon_signed_rank"] or {}).get("p_value")
    gate = cal_sum["mean"] > 0 and cal_p is not None and cal_p < P.ALPHA
    sel = [r["selectivity"] for r in valid]
    sel_sum = replicate_level_summary(sel, alpha=P.ALPHA)
    p = (sel_sum["wilcoxon_signed_rank"] or {}).get("p_value")
    raw_gap = float(np.mean([r["real_score"] - r["raw_input_score"] for r in valid]))
    if not gate:
        decision, reason = "INCONCLUSIVE", (
            "fixed-sequence step 1 failed: the calibration gate did not validate the "
            "probing pipeline, so the primary hypothesis was not tested")
    elif sel_sum["mean"] > 0 and p is not None and p < P.ALPHA:
        decision, reason = "SUPPORTED", (
            "fixed-sequence step 1 passed; mean replicate-level selectivity positive "
            f"and p < alpha={P.ALPHA}")
    elif sel_sum["mean"] <= 0:
        decision, reason = "NOT_SUPPORTED", "mean replicate-level selectivity not positive"
    else:
        decision, reason = "INCONCLUSIVE", f"p does not meet alpha={P.ALPHA}"
    return {"decision": decision, "decision_reason": reason,
            "primary": sel_sum, "calibration": {"gate_passed": bool(gate), **cal_sum},
            "twin_decodability_audit": {
                "mean_twin_score_above_chance": float(np.mean(
                    [r["twin_decodability_above_chance"] for r in valid])),
                "interpretation": (
                    "if the untrained twin decodes the target above chance, the "
                    "selectivity is a difference between two NON-TRIVIAL "
                    "decodabilities; reported either way, never hidden"),
                "role": "audit, never a decision input"},
            "C3_RAW_INPUT": {
                "role": "SECONDARY INFORMATIONAL ONLY - never a pass/fail criterion",
                "mean_representation_minus_raw_input": raw_gap,
                "interpretation": (
                    "positive means the penultimate representation is more decodable "
                    "than the raw ciphertext encoding; <= 0 means the representation "
                    "adds nothing and no statement about the representation is "
                    "licensed")},
            "statistical_unit": P.CE3.statistical_unit, "alpha": P.ALPHA,
            "multiplicity": P.CE3.multiplicity}


# =====================================================================
# CE4
# =====================================================================

def _mirrored_pairs() -> np.ndarray:
    """
    Column-index pairs (c0_bit, c1_bit) of the SAME bit position in the SAME
    16-bit Speck word, under speck.convert_to_binary([c0l, c0r, c1l, c1r]):
    columns 0-15 c0l, 16-31 c0r, 32-47 c1l, 48-63 c1r. Index p < 16 is a
    left-word position, p >= 16 a right-word position.
    """
    w = 16
    return np.array([(j, 2 * w + j) for j in range(w)] +
                    [(w + j, 3 * w + j) for j in range(w)])


def build_interventions(X: np.ndarray, k2: int, rng) -> dict:
    """
    Side-balanced, same-word-paired construction at magnitude 2k = k2.

    Per sample draw k disjoint SAME-WORD pairs of difference-bearing positions,
    giving S[:k] and S[k:] with identical per-word composition. Then
        control    : flip both sides at S[:k]
        structural : flip c0 at S[:k], c1 at S[k:]
    Both arms flip exactly k bits of c0 and k bits of c1; the c0-side flips are
    identical between arms. A sample that cannot supply k such pairs is
    ineligible and is counted, never silently dropped.
    """
    pairs = _mirrored_pairs()
    c0, c1 = pairs[:, 0], pairs[:, 1]
    k = k2 // 2
    D = X[:, c0] != X[:, c1]                      # (n, 32) difference-bearing positions
    left = np.flatnonzero(np.arange(32) < 16)
    right = np.flatnonzero(np.arange(32) >= 16)
    n = X.shape[0]
    eligible = np.zeros(n, dtype=bool)
    first = np.full((n, k), -1, dtype=np.int64)
    second = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        li = left[D[i, left]]
        ri = right[D[i, right]]
        if len(li) // 2 + len(ri) // 2 < k:
            continue
        rng.shuffle(li); rng.shuffle(ri)
        f, s2, taken = [], [], 0
        for src in (li, ri):
            while taken < k and len(src) - len(f) * 0 >= 2 and len(src) >= 2:
                f.append(src[0]); s2.append(src[1]); src = src[2:]
                taken += 1
            if taken >= k:
                break
        eligible[i] = True
        first[i], second[i] = f[:k], s2[:k]
    idx = np.flatnonzero(eligible)
    Xe = X[idx]
    f_pos, s_pos = first[idx], second[idx]
    rows = np.arange(Xe.shape[0])[:, None]
    Xc = Xe.copy()
    Xc[rows, c0[f_pos]] ^= 1
    Xc[rows, c1[f_pos]] ^= 1                       # XOR preserved
    Xs = Xe.copy()
    Xs[rows, c0[f_pos]] ^= 1                       # identical c0-side flips
    Xs[rows, c1[s_pos]] ^= 1                       # XOR toggled at 2k positions
    return {"eligible_mask": eligible, "X": Xe, "structural": Xs, "control": Xc,
            "first": f_pos, "second": s_pos, "n_total": int(n),
            "n_eligible": int(idx.size), "k2": int(k2)}


def intervention_checks(b: dict) -> dict:
    """Full control-validity audit. Every item is proved per sample, not asserted."""
    pairs = _mirrored_pairs()
    c0, c1 = pairs[:, 0], pairs[:, 1]
    Xe, Xs, Xc, k2 = b["X"], b["structural"], b["control"], b["k2"]
    ds, dc = (Xe ^ Xs), (Xe ^ Xc)
    xor_e, xor_s, xor_c = (Xe[:, c0] ^ Xe[:, c1]), (Xs[:, c0] ^ Xs[:, c1]), (Xc[:, c0] ^ Xc[:, c1])
    left_cols = np.r_[np.arange(0, 16), np.arange(32, 48)]
    right_cols = np.r_[np.arange(16, 32), np.arange(48, 64)]
    return {
        "magnitude_bits_2k": int(k2),
        "hamming_structural": sorted(set(ds.sum(1).tolist()))[:3],
        "hamming_control": sorted(set(dc.sum(1).tolist()))[:3],
        "equal_total_hamming": bool(np.array_equal(ds.sum(1), dc.sum(1))),
        "equal_per_ciphertext": bool(
            np.array_equal(ds[:, :32].sum(1), dc[:, :32].sum(1))
            and np.array_equal(ds[:, 32:].sum(1), dc[:, 32:].sum(1))),
        "equal_per_word": bool(
            np.array_equal(ds[:, left_cols].sum(1), dc[:, left_cols].sum(1))
            and np.array_equal(ds[:, right_cols].sum(1), dc[:, right_cols].sum(1))),
        "identical_c0_side_flips": bool(np.array_equal(ds[:, :32], dc[:, :32])),
        "control_preserves_pair_xor": bool(np.array_equal(xor_c, xor_e)),
        "structural_xor_positions_changed": sorted(set((xor_s ^ xor_e).sum(1).tolist()))[:3],
        "no_ineligible_position_touched": bool(
            (((ds[:, c0] | ds[:, c1]) | (dc[:, c0] | dc[:, c1]))
             & ~(Xe[:, c0] != Xe[:, c1])).sum() == 0),
        "marginal_bit_freq_max_abs_shift_structural": float(
            np.abs(Xs.mean(0) - Xe.mean(0)).max()),
        "marginal_bit_freq_max_abs_shift_control": float(
            np.abs(Xc.mean(0) - Xe.mean(0)).max()),
        # The c0 half is perturbed IDENTICALLY by both arms (identical c0-side
        # flips), so its marginal shift is identical by construction - asserted,
        # not assumed. The c1 half is perturbed at different positions by the two
        # arms, so its per-column marginal shift is NOT required to match: the
        # frozen matching criteria are counts, not per-column frequencies. The
        # c1 discrepancy is REPORTED so a reader can judge it.
        "marginal_shift_identical_on_c0": bool(np.allclose(
            Xs[:, :32].mean(0) - Xe[:, :32].mean(0),
            Xc[:, :32].mean(0) - Xe[:, :32].mean(0), atol=0, rtol=0)),
        "marginal_shift_max_abs_diff_on_c1": float(np.abs(
            (Xs[:, 32:].mean(0) - Xe[:, 32:].mean(0))
            - (Xc[:, 32:].mean(0) - Xe[:, 32:].mean(0))).max()),
        "pair_xor_weight_mean_original": float(xor_e.sum(1).mean()),
        "pair_xor_weight_mean_structural": float(xor_s.sum(1).mean()),
        "pair_xor_weight_mean_control": float(xor_c.sum(1).mean()),
    }


def _checks_pass(c: dict) -> bool:
    return bool(c["equal_total_hamming"] and c["equal_per_ciphertext"]
                and c["equal_per_word"] and c["identical_c0_side_flips"]
                and c["control_preserves_pair_xor"]
                and c["no_ineligible_position_touched"]
                and c["structural_xor_positions_changed"] == [c["magnitude_bits_2k"]]
                and c["hamming_structural"] == c["hamming_control"]
                == [c["magnitude_bits_2k"]])


def run_ce4(out_dir: Path, *, n_runs: int, n_samples: int, seed: int,
            magnitudes=None, model=None, log=print) -> dict:
    model = model or load_reference_model()
    magnitudes = list(magnitudes or P.CE4.magnitude_ladder_bits)
    raw_dir = out_dir / "raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for i in range(n_runs):
        t0 = time.time()
        rng = np.random.default_rng(seed + i)
        X, _ = generate_theory_data(n_samples)
        f_null = predict(model, X)
        rec = {"run_id": f"ce4_rep{i:02d}", "seed": seed + i, "n_total": int(n_samples),
               "rounds": P.ROUNDS, "differential": list(P.DIFFERENTIAL),
               "dataset_sha256": sha256_array(X),
               "C4_NULL_mean_abs_change": float(
                   np.abs(f_null - predict(model, X.copy())).mean()),
               "by_magnitude": {}, "status": "OK"}
        payload = {}
        for k2 in magnitudes:
            b = build_interventions(X, k2, rng)
            chk = intervention_checks(b)
            f0 = predict(model, b["X"]); fs = predict(model, b["structural"])
            fc = predict(model, b["control"])
            gap = np.abs(f0 - fs) - np.abs(f0 - fc)
            rec["by_magnitude"][str(k2)] = {
                "n_total": b["n_total"], "n_eligible": b["n_eligible"],
                "coverage": b["n_eligible"] / b["n_total"],
                "manipulation_checks": chk, "checks_pass": _checks_pass(chk),
                "structural_mean": float(np.abs(f0 - fs).mean()),
                "control_mean": float(np.abs(f0 - fc).mean()),
                "mean_gap": float(gap.mean()),
                "sd_gap": float(gap.std(ddof=1)) if gap.size > 1 else 0.0,
                "effect_size_dz": float(gap.mean() / gap.std(ddof=1))
                                  if gap.size > 1 and gap.std(ddof=1) > 0 else None,
            }
            payload[f"gap_{k2}"] = gap.astype(np.float32)
            payload[f"f0_{k2}"] = f0.astype(np.float32)
            payload[f"fs_{k2}"] = fs.astype(np.float32)
            payload[f"fc_{k2}"] = fc.astype(np.float32)
        np.savez_compressed(raw_dir / f"{rec['run_id']}.npz", **payload)
        rec["raw_file"] = f"raw/{rec['run_id']}.npz"
        rec["raw_sha256"] = sha256_file(raw_dir / f"{rec['run_id']}.npz")
        rec["seconds"] = round(time.time() - t0, 2)
        runs.append(rec)
        log(f"  CE4 replicate {i+1}/{n_runs}: " +
            " ".join(f"2k={k}:gap={rec['by_magnitude'][str(k)]['mean_gap']:+.4f}"
                     for k in magnitudes) + f" ({rec['seconds']}s)")
    return {"runs": runs, **analyse_ce4(runs, out_dir, magnitudes)}


def analyse_ce4(runs, out_dir: Path, magnitudes) -> dict:
    from scipy import stats

    valid = [r for r in runs if r.get("status") == "OK"]
    primary_k2 = str(P.CE4.primary_magnitude_bits)
    if len(valid) < P.CE4.minimum_valid_runs:
        return {"decision": "INCONCLUSIVE",
                "reason": f"{len(valid)} valid replicates < {P.CE4.minimum_valid_runs}"}
    bad = [r["run_id"] for r in valid
           for k in magnitudes if not r["by_magnitude"][str(k)]["checks_pass"]]
    null_max = max(r["C4_NULL_mean_abs_change"] for r in valid)

    per_mag = {}
    for k in magnitudes:
        means = [r["by_magnitude"][str(k)]["mean_gap"] for r in valid]
        # ANALYSIS 1: within-replicate paired effect, reported per replicate.
        within = []
        for r in valid:
            with np.load(out_dir / r["raw_file"]) as z:
                g = z[f"gap_{k}"].astype(np.float64)
            try:
                w = float(stats.wilcoxon(g, alternative="greater").pvalue)
            except ValueError:
                w = None
            within.append({"run_id": r["run_id"], "n_eligible": g.size,
                           "mean_gap": float(g.mean()),
                           "wilcoxon_p_within_replicate": w})
        # ANALYSIS 2: across-replicate reproducibility (PRIMARY inference).
        per_mag[str(k)] = {
            "analysis_1_within_replicate": {
                "inferential_target": P.CE4.analysis_1, "per_replicate": within},
            "analysis_2_across_replicates": {
                "inferential_target": P.CE4.analysis_2,
                "replicate_mean_gaps": means, "mean_of_means": float(np.mean(means)),
                "ci95_bootstrap": P.bootstrap_ci(means), **P.exact_sign_test(means)},
            "mean_coverage": float(np.mean(
                [r["by_magnitude"][str(k)]["coverage"] for r in valid])),
        }
    secondary = {k: per_mag[k]["analysis_2_across_replicates"]["p_value"]
                 for k in per_mag if k != primary_k2}
    prim = per_mag[primary_k2]["analysis_2_across_replicates"]
    gaps_by_mag = [per_mag[str(k)]["analysis_2_across_replicates"]["mean_of_means"]
                   for k in magnitudes]
    monotone = all(x <= y for x, y in zip(gaps_by_mag, gaps_by_mag[1:]))

    if bad:
        decision, reason = "INCONCLUSIVE", f"manipulation checks failed: {sorted(set(bad))}"
    elif null_max != 0.0:
        decision, reason = "INCONCLUSIVE", f"C4-NULL control is not exactly 0 ({null_max:.2e})"
    elif prim["mean_of_means"] <= 0:
        decision, reason = "NOT_SUPPORTED", "mean of replicate mean gaps is not positive"
    elif prim["p_value"] is not None and prim["p_value"] < P.ALPHA:
        decision, reason = "SUPPORTED", (
            "all manipulation checks passed, C4-NULL exactly 0, and the "
            "across-replicate sign test at the primary magnitude is significant with a "
            "positive mean of replicate means")
    else:
        decision, reason = "INCONCLUSIVE", "across-replicate test not significant"
    return {"decision": decision, "decision_reason": reason,
            "primary_magnitude_bits": P.CE4.primary_magnitude_bits,
            "primary": prim, "by_magnitude": per_mag,
            "secondary_dose_response": {
                "magnitudes": magnitudes, "mean_of_means_by_magnitude": gaps_by_mag,
                "monotone_nondecreasing": bool(monotone),
                "holm_adjusted_p": P.holm(secondary),
                "role": "reported; cannot change the primary decision"},
            "C4_NULL_max_mean_abs_change": null_max,
            "threshold_policy": P.CE4.threshold_policy, "alpha": P.ALPHA,
            "statistical_unit": P.CE4.statistical_unit}


# =====================================================================
# Orchestration
# =====================================================================

def preflight(*, behavioural: bool = True) -> dict:
    rep = {"plan_version": P.PLAN_VERSION, "plan_hash": P.plan_hash(),
           "environment": environment(), "source_hashes": source_hashes(),
           "unspecified_in_source": P.UNSPECIFIED_IN_SOURCE, "utc": utc()}
    rep["model"] = verify_reference_model(behavioural=behavioural)
    # CE1 isolation. The needle is assembled at runtime so that this guard
    # cannot match its own source text (a literal here would self-trigger).
    needles = ["ce" + "1/production_", "evidence" + "_current/ce1",
               "block0_" + "baseline", "sealed_" + "evaluation_set"]
    offenders = []
    for f in (CE234_ROOT / "production.py", CE234_ROOT / "frozen_plan.py",
              CE234_ROOT / "verify.py"):
        body = f.read_text()
        start = body.find("needles = [")
        guard = body[start:body.find("]", start) + 1] if start >= 0 else ""
        scanned = body.replace(guard, "") if guard else body
        offenders += [f"{rel(f)}:{n}" for n in needles if n in scanned]
    if offenders:
        raise PreflightError(f"CE1 isolation violated: {offenders}")
    if not production_generator_is_nondeterministic():
        raise PreflightError("a production run would use the TEST-ONLY deterministic "
                             "generator path; refusing")
    rep["generator"] = {"version": GENERATOR_VERSION,
                        "production_source": "os.urandom (Gohr convention)",
                        "exact_replay": "UNAVAILABLE - raw observations persisted instead",
                        "deterministic_path": "test-only, asserted unused in production"}
    rep["ce1_isolation"] = ("verified: CE2-CE4 source references no CE1 run directory, "
                            "no CE1 checkpoint and no CE1 sealed set")
    rep["status"] = "PREFLIGHT_OK"
    return rep


def certificate(ce: str, results: dict, pre: dict, *, production: bool,
                run_id: str, started: str) -> dict:
    plan = {"CE2": P.CE2, "CE3": P.CE3, "CE4": P.CE4}[ce]
    cert = {
        "certificate_schema_version": "ce234-certificate-1",
        "experiment_id": plan.experiment_id, "run_id": run_id,
        "plan_version": P.PLAN_VERSION, "plan_hash": P.plan_hash(),
        "frozen_plan": json.loads(json.dumps(plan, default=lambda o: o.__dict__)),
        "reference_model": pre["model"], "environment": pre["environment"],
        "source_hashes": pre["source_hashes"],
        "started_utc": started, "finished_utc": utc(),
        "results": results,
        "limitations": list(plan.limitations),
        "claim_scope": plan.estimand if ce == "CE4" else plan.question,
    }
    if not production:
        cert["non_evidentiary"] = True
        cert["claim_scope"] = "SMOKE ONLY - NOT SCIENTIFIC EVIDENCE"
    return cert


def execute(ce: str, out_dir: Path, *, production: bool, n1: int, n2: int,
            seed: int, n_splits: int = 5, behavioural: bool = True) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise PreflightError(f"{rel(out_dir)} already exists and is not empty; "
                             "production evidence is never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = utc()
    pre = preflight(behavioural=behavioural)
    model = load_reference_model()
    P.write_plan(out_dir / "frozen_plan.json")
    if ce == "CE2":
        res = run_ce2(out_dir, n_runs=n1, n_samples=n2, seed=seed, model=model)
    elif ce == "CE3":
        res = run_ce3(out_dir, n_replicates=n1, n_samples=n2, n_splits=n_splits,
                      seed=seed, model=model)
    else:
        res = run_ce4(out_dir, n_runs=n1, n_samples=n2, seed=seed, model=model,
                      magnitudes=P.CE4.magnitude_ladder_bits)
    cert = certificate(ce, res, pre, production=production,
                       run_id=out_dir.name, started=started)
    path = out_dir / "certificate.json"
    path.write_text(json.dumps(cert, indent=2, sort_keys=True, default=str))
    (out_dir / "preflight.json").write_text(json.dumps(pre, indent=2, default=str))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CE2-CE4 production pipeline")
    ap.add_argument("ce", choices=("CE2", "CE3", "CE4"))
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced data; output is marked NOT SCIENTIFIC EVIDENCE")
    ap.add_argument("--production", action="store_true",
                    help="REQUIRED to produce evidence; uses the frozen plan values")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--no-behavioural-binding", action="store_true",
                    help="skip the behavioural round check (smoke only)")
    a = ap.parse_args(argv)
    try:
        if a.preflight:
            print(json.dumps(preflight(behavioural=not a.no_behavioural_binding),
                             indent=2, default=str))
            return 0
        if a.smoke:
            out = a.out_dir or (EVIDENCE_ROOT / a.ce.lower() /
                                f"smoke_{datetime.now():%Y%m%d_%H%M%S}")
            n1, n2 = {"CE2": (3, 20_000), "CE3": (2, 4_000), "CE4": (2, 5_000)}[a.ce]
            p = execute(a.ce, out, production=False, n1=n1, n2=n2, seed=a.seed,
                        behavioural=not a.no_behavioural_binding)
            print(f"SMOKE ONLY - NOT SCIENTIFIC EVIDENCE: {rel(p)}")
            return 0
        if a.production:
            if a.out_dir is None:
                a.out_dir = (EVIDENCE_ROOT / a.ce.lower() /
                             f"production_{datetime.now():%Y%m%d}")
            n1, n2 = {"CE2": (P.CE2.n_runs, P.CE2.samples_per_run),
                      "CE3": (P.CE3.n_replicates, P.CE3.samples_per_replicate),
                      "CE4": (P.CE4.n_runs, P.CE4.samples_per_run)}[a.ce]
            p = execute(a.ce, a.out_dir, production=True, n1=n1, n2=n2, seed=a.seed)
            print(f"production certificate: {rel(p)}")
            return 0
    except PreflightError as exc:
        print(f"FAIL CLOSED: {exc}")
        return 1
    print("Nothing executed. Use --preflight, --smoke or --production.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
