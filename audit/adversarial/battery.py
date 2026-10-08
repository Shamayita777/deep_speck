"""
ADVERSARIAL BATTERY v2 - construction, evaluation, certificate.

Replaces a1_theory_mimic.py's run logic. A1 itself is NOT removed: it stays as
a historical, failed adversary, and this module refuses to re-run it.

The four defects that made v1's output uninterpretable are fixed at the level
of the data model, not the console text:

  1. three separate axes (scientific decision / adversarial outcome /
     construction status), with INCONCLUSIVE mapping to UNDETERMINED and never
     to CAUGHT; the only permitted mapping lives in preregistration_v2;
  2. sealed and calibration sets are persisted and hash-bound before use, and
     are never regenerated during scoring;
  3. thresholds are chosen on the calibration split only; the primary
     distinguisher metric is threshold-free AUC with an equivalence rule;
  4. the adversarial protocol runs at its own registered replication, which is
     matched to production, so a dimension never returns INCONCLUSIVE merely
     because the adversarial run was small.

    python -m audit.adversarial.battery --list
    python -m audit.adversarial.battery --adversary A3 --smoke
    python -m audit.adversarial.battery --adversary A3 --build
    python -m audit.adversarial.battery --adversary A3 --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit.adversarial import preregistration_v2 as PRE2
from audit.adversarial import sealed_sets as SS
from audit.cryptography.ce234 import frozen_plan as P
from audit.cryptography.ce234 import production as PR

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]        # <repo>/audit/adversarial -> <repo>
EVIDENCE_ROOT = HERE / "evidence"
A4_MAP_KEY = b"ciphermind-A4-xor-response-v1"      # registered constant


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# A3 - representation adversary, constant output
# ---------------------------------------------------------------------------

def build_a3(n_train: int, epochs: int, seed: int, log=print):
    """
    Train the penultimate layer to encode CE3's target through a temporary
    head, then zero the output layer so the shipped model emits a constant.

    Zeroing the final Dense kernel and bias makes the output sigmoid(0) = 0.5
    for every input, EXACTLY. The architecture is untouched, so layers[-2] is
    still the 64-unit penultimate layer CE3 probes.
    """
    import keras

    from audit.cryptography.gohr.model import GohrModel
    from audit.cryptography.gohr.trainer import GohrTrainer

    t0 = time.time()
    GohrTrainer.set_seed(seed)
    model = GohrModel(depth=P.DEPTH, regularization=P.L2_REG).build()

    X, target = PR.generate_theory_data(n_train)
    q = np.quantile(target, [0.2, 0.4, 0.6, 0.8])
    labels = np.digitize(target, bins=q, right=False).astype(np.int64)

    penult = model.layers[-2].output
    head = keras.layers.Dense(5, activation="softmax", name="a3_temp_head")(penult)
    trainer = keras.Model(model.input, head)
    trainer.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
    hist = trainer.fit(X, labels, batch_size=P.PREDICT_BATCH, epochs=epochs,
                       validation_split=0.1, verbose=2)

    out_layer = model.layers[-1]
    w, b = out_layer.get_weights()
    out_layer.set_weights([np.zeros_like(w), np.zeros_like(b)])

    probe = PR.predict(model, X[:2048])
    if float(probe.std()) != 0.0:
        raise RuntimeError("A3 output is not constant after zeroing the output layer")
    record = {
        "adversary": "A3", "n_train": int(n_train), "epochs": int(epochs),
        "seed": int(seed), "objective": "5-class CE3 target via a temporary head, "
                                        "then output layer zeroed",
        "temporary_head_discarded": True,
        "final_head_val_accuracy": float(hist.history["val_accuracy"][-1]),
        "output_is_constant": True, "constant_value": float(probe[0]),
        "rounds": P.ROUNDS, "differential": list(P.DIFFERENTIAL),
        "train_seconds": round(time.time() - t0, 1), "trained_utc": utc(),
        "entropy_source": "os.urandom via the Gohr generator; exact replay unavailable",
    }
    return model, record


# ---------------------------------------------------------------------------
# A4 - closed-form XOR response function
# ---------------------------------------------------------------------------

class A4XorResponse:
    """
    f(x) = h(C0 XOR C1), h a keyed BLAKE2b-derived deterministic map into
    [0, 1]. No formal pseudorandom-function claim is made: the construction
    needs only determinism and dependence on the pair XOR alone.

    Duck-typed to the predict() interface the dimensions use. Depends on the
    input ONLY through the ciphertext-pair XOR, so the CE4 control arm - which
    preserves that XOR exactly - leaves the output exactly unchanged.
    """

    adversary_id = "A4"

    def __init__(self, key: bytes = A4_MAP_KEY):
        self.key = key
        w = 16
        self._pairs = np.array([(j, 2 * w + j) for j in range(w)]
                               + [(w + j, 3 * w + j) for j in range(w)])

    def pair_xor(self, X: np.ndarray) -> np.ndarray:
        return (X[:, self._pairs[:, 0]] ^ X[:, self._pairs[:, 1]]).astype(np.uint8)

    def predict(self, X, batch_size=None, verbose=0):
        packed = np.packbits(self.pair_xor(np.asarray(X)), axis=1)
        out = np.empty(packed.shape[0], dtype=np.float32)
        for i, row in enumerate(packed):
            d = hashlib.blake2b(row.tobytes(), key=self.key, digest_size=8).digest()
            out[i] = int.from_bytes(d, "big") / float(1 << 64)
        return out.reshape(-1, 1)

    def save(self, path):
        Path(path).write_text(json.dumps(
            {"adversary": "A4", "kind": "closed-form", "map": "keyed BLAKE2b, digest/2**64",
             "key": self.key.decode(), "domain": "pair XOR over 32 mirrored positions",
             "note": "no learned parameters; the file is the specification"},
            indent=2))


def verify_a4_xor_dependence(model, n: int = 20_000, seed: int = 11) -> dict:
    """
    Fidelity for A4 is EXACT, not estimated: build pairs with identical pair
    XOR but different absolute ciphertexts and assert the outputs are equal.
    """
    rng = np.random.default_rng(seed)
    X, _ = PR.generate_theory_data(n)
    X2 = X.copy()
    pairs = model._pairs
    flip = rng.integers(0, 2, size=(n, 32)).astype(np.uint8)
    X2[:, pairs[:, 0]] ^= flip
    X2[:, pairs[:, 1]] ^= flip                      # mirrored flip preserves the XOR
    same_xor = bool(np.array_equal(model.pair_xor(X), model.pair_xor(X2)))
    a = PR.predict(model, X); b = PR.predict(model, X2)
    identical = bool(np.array_equal(a, b))
    return {"statistic": "exact invariance under XOR-preserving perturbation",
            "n": int(n), "pair_xor_preserved": same_xor,
            "outputs_identical": identical, "delivered": bool(same_xor and identical)}


# ---------------------------------------------------------------------------
# distinguisher assessment - sealed set, threshold-free primary metric
# ---------------------------------------------------------------------------

def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats as st
    if np.std(scores) == 0:
        return 0.5                                  # constant score: no ranking
    r = st.rankdata(scores)
    n1 = float(y.sum()); n0 = float(y.size - n1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def assess_distinguisher(model, run_dir: Path, *, n_sealed: int, n_calib: int,
                         n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Primary: ROC AUC on the sealed set, with a bootstrap CI, judged against the
    registered equivalence region. Secondary descriptive: accuracy at the fixed
    0.5 cut, and accuracy at a threshold chosen ONLY on the calibration split.
    """
    crit = PRE2.DISTINGUISHER_CRITERION
    d = run_dir / "datasets"
    calib = SS.build_or_load(d, "calibration", n_calib, rounds=P.ROUNDS,
                             differential=P.DIFFERENTIAL)
    sealed = SS.build_or_load(d, "sealed", n_sealed, rounds=P.ROUNDS,
                              differential=P.DIFFERENTIAL)
    if calib["content_sha256"] == sealed["content_sha256"]:
        raise RuntimeError("calibration and sealed sets are identical; refusing")

    s_cal = PR.predict(model, calib["X"])
    s_seal = PR.predict(model, sealed["X"])

    # threshold chosen on CALIBRATION ONLY
    cand = np.unique(np.quantile(s_cal, np.linspace(0, 1, 512)))
    accs = [(float(np.mean((s_cal > t) == calib["Y"])), float(t)) for t in cand]
    best_cal_acc, thr = max(accs)
    orient_flip = best_cal_acc < 0.5
    if orient_flip:
        best_cal_acc = 1.0 - best_cal_acc

    def acc_at(s, y, t):
        p = (s > t).astype(np.int64)
        if orient_flip:
            p = 1 - p
        return float(np.mean(p == y))

    y = sealed["Y"]
    auc = _auc(s_seal, y)
    auc_sym = max(auc, 1.0 - auc)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        boots[i] = _auc(s_seal[idx], y[idx])
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    region = crit["equivalence_region"]
    inside = bool(lo >= region[0] and hi <= region[1])

    acc_thr = acc_at(s_seal, y, thr)
    pred = (s_seal > thr).astype(np.int64)
    if orient_flip:
        pred = 1 - pred
    # true/false rates computed from the confusion matrix explicitly: the
    # previous form took the POSITIVE-prediction rate among y == 0, which is
    # the false positive rate, and reported (TPR + FPR) / 2 as balanced
    # accuracy. See ERRATUM 1.
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "primary_metric": "ROC AUC on the sealed set",
        "auc": auc, "auc_symmetrised": auc_sym,
        "auc_ci95_bootstrap": [lo, hi], "n_bootstrap": int(n_boot),
        "bootstrap_unit": "sample (a property of a fixed model on a fixed dataset; "
                          "NOT an independent scientific replicate)",
        "equivalence_region": region,
        "equivalence_satisfied": inside,
        "equivalence_rule": crit["equivalence_rule"],
        "construction_status": (PRE2.CONSTRUCTION_VALID if inside
                                else PRE2.CONSTRUCTION_FAILED),
        "threshold_provenance": "selected on the calibration split only; the sealed "
                                "set was never used to select anything",
        "calibrated_threshold": float(thr), "orientation_flipped": bool(orient_flip),
        "calibration_accuracy_at_threshold": best_cal_acc,
        "sealed_accuracy_at_calibrated_threshold": acc_thr,
        "sealed_accuracy_at_0.5": acc_at(s_seal, y, 0.5),
        "balanced_accuracy_at_calibrated_threshold": (tpr + tnr) / 2.0,
        "confusion_at_calibrated_threshold": {"tp": tp, "fn": fn, "tn": tn,
                                              "fp": fp, "tpr": tpr, "tnr": tnr},
        "sealed_dataset": {k: v for k, v in sealed.items() if k not in ("X", "Y")},
        "calibration_dataset": {k: v for k, v in calib.items() if k not in ("X", "Y")},
        "scores_sha256": SS.sha256_array(s_seal),
    }


# ---------------------------------------------------------------------------
# dimension evaluation - raw decisions preserved, mapping kept separate
# ---------------------------------------------------------------------------

def evaluate_dimensions(model, out_dir: Path, adversary, counts: dict,
                        samples: dict, seed: int, construction_status: str,
                        log=print) -> dict:
    """
    Run only the dimensions this adversary targets. For every other dimension
    the certificate records NOT_APPLICABLE with the registered reason, rather
    than running it and reporting a number that means nothing.
    """
    results = {}
    for dim in ("CE2", "CE3", "CE4"):
        key = {"CE2": "CE2_theory_consistency",
               "CE3": "CE3_representation_decodability",
               "CE4": "CE4_intervention_sensitivity"}[dim]
        # SCOPE RULE: `targets` is authoritative. A prediction entry alone must
        # never cause a dimension outside the registered scope to execute.
        in_scope = dim in adversary.targets
        if not in_scope or adversary.predictions.get(key) == PRE2.NOT_APPLICABLE:
            results[dim] = {
                "executed": False,
                "adversarial_outcome": PRE2.NOT_APPLICABLE,
                "scientific_decision": None,
                "in_registered_targets": in_scope,
                "reason": adversary.prediction_rationale.get(
                    key, "outside the adversary's registered target scope; not "
                         "executed")}
            continue
        log(f"  {dim} ...")
        if dim == "CE2":
            r = PR.run_ce2(out_dir / "ce2", n_runs=counts["CE2"],
                           n_samples=samples["CE2"], seed=seed, model=model, log=log)
            primary = {"replicate_rhos": [x.get("rho_primary")
                                          for x in r.get("runs", [])],
                       "median_rho": (r.get("endpoints", {}).get("primary", {})
                                      .get("median")),
                       "ci95": (r.get("endpoints", {}).get("primary", {})
                                .get("ci95_bootstrap")),
                       "p_value": (r.get("endpoints", {}).get("primary", {})
                                   .get("p_value")),
                       "n_valid": r.get("n_valid_runs")}
        elif dim == "CE3":
            r = PR.run_ce3(out_dir / "ce3", n_replicates=counts["CE3"],
                           n_samples=samples["CE3"], n_splits=5, seed=seed + 100,
                           model=model, log=log)
            primary = {"replicate_selectivity": [x.get("selectivity")
                                                 for x in r.get("replicates", [])],
                       "mean": (r.get("primary") or {}).get("mean"),
                       "ci95": (r.get("primary") or {}).get("ci95"),
                       "p_value": ((r.get("primary") or {})
                                   .get("wilcoxon_signed_rank", {}) or {})
                       .get("p_value"),
                       "n_valid": len(r.get("replicates", []))}
        else:
            r = PR.run_ce4(out_dir / "ce4", n_runs=counts["CE4"],
                           n_samples=samples["CE4"], seed=seed + 200, model=model,
                           magnitudes=P.CE4.magnitude_ladder_bits, log=log)
            primary = {"replicate_mean_gaps": (r.get("primary") or {})
                       .get("replicate_mean_gaps"),
                       "mean_of_means": (r.get("primary") or {}).get("mean_of_means"),
                       "ci95": (r.get("primary") or {}).get("ci95_bootstrap"),
                       "p_value": (r.get("primary") or {}).get("p_value"),
                       "n_valid": len(r.get("runs", []))}
        decision = r.get("decision")
        results[dim] = {
            "executed": True,
            # AXIS 1 - verbatim, never transformed
            "scientific_decision": decision,
            "scientific_decision_reason": r.get("decision_reason") or r.get("reason"),
            # AXIS 2 - the only permitted mapping
            "adversarial_outcome": PRE2.map_decision(decision, construction_status),
            "replication": {"registered": counts[dim], "observed": primary["n_valid"]},
            "primary_endpoint": primary,
            "scale_label": "adversarial protocol replication (registered); NOT a "
                           "production CE certificate",
        }
    return results


# ---------------------------------------------------------------------------
# certificate
# ---------------------------------------------------------------------------

A3_FREEZE_NAME = "a3_freeze_manifest.json"


def current_head() -> str:
    """Full HEAD SHA, or 'UNAVAILABLE'. Separated so it can be stubbed in tests."""
    return _git().get("commit", "UNAVAILABLE")


def verify_freeze_commit_recorded(manifest: dict) -> str:
    """
    The freeze manifest must identify the exact commit the A3 artifact was
    built from. This is PROVENANCE - it records where the artifact came from.
    It is NOT the source-integrity mechanism: see verify_source_integrity().

    Exact equality with the current HEAD is deliberately NOT required. The
    intended workflow commits the generated artifact and its manifest AFTER
    the build, which necessarily advances HEAD; demanding equality would make
    the correct workflow impossible while adding nothing, because the frozen
    source hashes already establish that the executing code is unchanged.
    """
    frozen = manifest.get("freeze_git_commit")
    if not frozen or frozen == "UNAVAILABLE":
        raise SystemExit(
            "REFUSED: the freeze manifest records no git commit, so the commit the "
            "A3 artifact was built from cannot be identified. Rebuild and re-freeze "
            "inside the repository.")
    if len(frozen) != 40 or any(c not in "0123456789abcdef" for c in frozen.lower()):
        raise SystemExit(
            f"REFUSED: the recorded freeze commit {frozen!r} is not a full 40-"
            "character git SHA; provenance cannot be established from it.")
    return frozen


def verify_source_integrity(manifest: dict, current: dict | None = None) -> dict:
    """
    AUTHORITATIVE SOURCE BINDING. Every source artifact hashed at freeze time
    must hash identically now. This is what establishes that the code about to
    interpret the adversary is the code that produced it - a descendant commit
    carrying only the generated artifact and manifest passes, any edit to a
    hashed source file does not.
    """
    frozen = manifest.get("source_hashes")
    if not frozen:
        raise SystemExit(
            "REFUSED: the freeze manifest records no source hashes, so the source "
            "state at freeze time cannot be verified. Rebuild and re-freeze.")
    current = source_hashes() if current is None else current
    if "_unhashed_modules" in frozen or "_unhashed_modules" in current:
        raise SystemExit(
            "REFUSED: one or more source modules could not be hashed, so source "
            "integrity cannot be established: "
            f"frozen={frozen.get('_unhashed_modules')} "
            f"current={current.get('_unhashed_modules')}")
    changed = {k: {"frozen": v, "current": current.get(k, "MISSING")}
               for k, v in frozen.items() if current.get(k) != v}
    added = sorted(set(current) - set(frozen))
    if changed or added:
        raise SystemExit(
            "REFUSED: the execution source state differs from the A3 freeze "
            f"manifest. Source-hash mismatch in {sorted(changed)}"
            + (f"; source files present now but not at freeze time: {added}" if added
               else "")
            + ". The code that would interpret the adversary is not the code that "
              "produced it; confirmatory execution is refused. Rebuild and re-freeze, "
              "or restore the frozen source state.")
    return {"verified": True, "n_source_files": len(frozen)}


def write_freeze_manifest(out: Path, model_path: Path, train_record: dict) -> dict:
    """Written by --build. The artifact a confirmatory run is allowed to use."""
    man = {
        "schema": "adversary-freeze-manifest-1",
        "adversary_id": "A3",
        "model_artifact": rel(model_path),
        "model_sha256": SS.sha256_file(model_path),
        "training_record": train_record,
        "source_hashes": source_hashes(),
        "prereg_version": PRE2.PREREG_VERSION,
        "prereg_hash": PRE2.plan_hash(),
        "environment": PR.environment(),
        "git": _git(),
        # Exact HEAD at the moment of freezing, full 40-hex SHA. PROVENANCE for
        # the built artifact: it records which commit produced it. Source
        # INTEGRITY is enforced separately by source_hashes above, so a later
        # commit that merely adds this artifact and manifest is acceptable.
        # Deliberately not part of the preregistration document, so it is not
        # hashed by plan_hash().
        "freeze_git_commit": current_head(),
        "frozen_utc": utc(),
        "rule": "a confirmatory run must load exactly this artifact and must hash "
                "every source file to the values recorded here; no rebuild, no "
                "selection among candidates, no manifest refresh",
    }
    g = man["git"]
    if g.get("working_tree") != "clean" or man["freeze_git_commit"] == "UNAVAILABLE":
        man["provenance_warning"] = (
            f"frozen with git working_tree={g.get('working_tree')!r} and "
            f"freeze_git_commit={man['freeze_git_commit']!r}. A confirmatory run "
            "REFUSES a manifest without a recorded commit, and refuses a dirty tree "
            "at run time. Rebuild from a clean, committed state.")
        print("WARNING: " + man["provenance_warning"])
    (out / A3_FREEZE_NAME).write_text(json.dumps(man, indent=2, sort_keys=True,
                                                 default=str))
    return man


def load_frozen_a3(out: Path, model_path: Path, *,
                   current_sources: dict | None = None) -> dict:
    """
    Verify the frozen A3 artifact for a confirmatory run. Hard-fails rather
    than building anything.

    Checks, in order: the freeze manifest exists; the artifact exists; a full
    freeze commit is recorded (provenance); every frozen source hash still
    matches (source integrity); the artifact SHA256 equals the frozen one.
    Each check is independent - satisfying one never excuses another.
    """
    man_path = out / A3_FREEZE_NAME
    if not man_path.exists():
        raise SystemExit(
            f"REFUSED: no freeze manifest at {rel(man_path)}. A confirmatory run "
            "never builds an adversary. Run --build first, commit the artifact and "
            "its manifest, then re-run.")
    man = json.loads(man_path.read_text())
    if not model_path.exists():
        raise SystemExit(
            f"REFUSED: the frozen A3 artifact {rel(model_path)} is missing. The "
            "adversary must be built, frozen and committed before confirmatory "
            "execution; --run will not reconstruct it.")
    verify_freeze_commit_recorded(man)
    verify_source_integrity(man, current_sources)
    actual = SS.sha256_file(model_path)
    if actual != man["model_sha256"]:
        raise SystemExit(
            f"REFUSED: A3 artifact hash {actual} does not match the frozen manifest "
            f"{man['model_sha256']}. The adversary changed after freezing.")
    return man


def _git() -> dict:
    """
    Git state of THIS repository. A failed invocation must never be read as a
    clean tree: `git status --porcelain` prints nothing both when the tree is
    clean and when the command fails, so the return code is checked explicitly.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=20)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=20)
        if head.returncode != 0 or status.returncode != 0:
            return {"commit": "UNAVAILABLE", "working_tree": "UNAVAILABLE",
                    "error": (head.stderr or status.stderr or "").strip()[:200],
                    "repo_root": str(REPO_ROOT)}
        commit = head.stdout.strip()
        return {"commit": commit or "UNAVAILABLE",
                "working_tree": "dirty" if status.stdout.strip() else "clean",
                "repo_root": str(REPO_ROOT)}
    except Exception as exc:                                    # noqa: BLE001
        return {"commit": "UNAVAILABLE", "working_tree": "UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}"[:200],
                "repo_root": str(REPO_ROOT)}


def source_hashes() -> dict:
    """
    Every source artifact that materially affects execution: the adversarial
    package, the CE234 plan/production/verifier that supplies the dimension
    estimators, and the Gohr cipher/dataset/model/trainer/evaluator, probe
    evaluator and CE4 design module they call.
    """
    import importlib

    files = [HERE / "preregistration_v2.py", HERE / "battery.py",
             HERE / "sealed_sets.py", HERE / "screening.py",
             HERE / "preregistration.py", HERE / "a1_theory_mimic.py"]
    modules = [
        "audit.cryptography.ce234.frozen_plan",
        "audit.cryptography.ce234.production",
        "audit.cryptography.ce234.verify",
        "audit.cryptography.gohr.speck",
        "audit.cryptography.gohr.dataset",
        "audit.cryptography.gohr.model",
        "audit.cryptography.gohr.trainer",
        "audit.cryptography.gohr.evaluate",
        "audit.cryptography.probe.evaluation",
        "audit.cryptography.experiments.ce4.design",
        "audit.cryptography.statistics",
    ]
    out = {rel(f): SS.sha256_file(f) for f in files if f.exists()}
    missing = []
    for name in modules:
        try:
            f = Path(importlib.import_module(name).__file__)
        except Exception:                                       # noqa: BLE001
            missing.append(name); continue
        out[rel(f)] = SS.sha256_file(f)
    if missing:
        out["_unhashed_modules"] = missing
    return out


def confirmatory_preflight(git_info: dict, allow_no_git: bool,
                           plan_hash: str | None = None,
                           expected_hash: str | None = None) -> dict:
    """
    Gate for --run only. A confirmatory certificate whose source hashes cannot
    be tied to a committed state is not independently reconstructable, so a
    dirty tree is refused outright.
    """
    plan_hash = PRE2.plan_hash() if plan_hash is None else plan_hash
    expected_hash = (PRE2.EXPECTED_PREREG_HASH if expected_hash is None
                     else expected_hash)
    if plan_hash != expected_hash:
        return {"ok": False, "reason": (
            f"REFUSED: the registration hash is {plan_hash}, but the frozen "
            f"confirmatory registration is {expected_hash}. The preregistration has "
            "changed since it was frozen; a clean commit does not make a changed "
            "registration the same registration.")}
    tree = git_info.get("working_tree")
    if tree == "dirty":
        return {"ok": False, "reason": (
            "REFUSED: the git working tree is dirty. A confirmatory run must be "
            "reconstructable from a committed state. Commit or stash, then re-run. "
            "(--smoke and --build are not gated.)")}
    if tree != "clean":
        if not allow_no_git:
            return {"ok": False, "reason": (
                "REFUSED: git metadata unavailable, so provenance cannot be "
                "established. Re-run inside the repository, or pass --allow-no-git "
                "to proceed with the limitation recorded in the certificate.")}
        return {"ok": True, "provenance_limitation":
                "git metadata unavailable; operator passed --allow-no-git"}
    return {"ok": True}


def compare_predictions(adversary, observed: dict) -> list:
    rows = []
    for key, predicted in adversary.predictions.items():
        if key == "distinguisher_auc":
            d = observed["distinguisher"]
            rows.append({"item": key, "predicted": predicted,
                         "observed": f"AUC {d['auc']:.4f} CI {d['auc_ci95_bootstrap']}",
                         "agrees": d["equivalence_satisfied"]})
        elif key == "model_binding_gate":
            rows.append({"item": key, "predicted": predicted,
                         "observed": observed["binding"]["outcome"],
                         "agrees": observed["binding"]["outcome"] == predicted})
        else:
            dim = key.split("_")[0]
            got = observed["dimensions"][dim]["adversarial_outcome"]
            rows.append({"item": key, "predicted": predicted, "observed": got,
                         "scientific_decision":
                             observed["dimensions"][dim]["scientific_decision"],
                         "agrees": got == predicted})
    return rows


def binding_check(adversary, model_path: Path) -> dict:
    """
    The production binding presupposes a checkpoint artifact. For an adversary
    without one it is NOT_APPLICABLE with a recorded reason - never CAUGHT,
    and never probed with a fabricated path.
    """
    if not adversary.binding_applicable:
        return {"outcome": PRE2.NOT_APPLICABLE, "refused": None,
                "executed": False,
                "reason": adversary.binding_not_applicable_reason}
    try:
        PR.verify_reference_model(checkpoint=model_path, behavioural=True)
        return {"outcome": PRE2.NOT_CAUGHT, "refused": False, "executed": True,
                "note": "THE BINDING DID NOT REFUSE THIS ADVERSARY - framework defect"}
    except Exception as exc:                                    # noqa: BLE001
        return {"outcome": PRE2.CAUGHT, "refused": True, "executed": True,
                "reason": str(exc)[:300]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Adversarial battery v2")
    ap.add_argument("--adversary", choices=("A3", "A4"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--allow-no-git", action="store_true",
                    help="proceed with a confirmatory run when git metadata is "
                         "unavailable; recorded as a provenance limitation")
    a = ap.parse_args(argv)

    if a.list:
        print(json.dumps({"prereg_version": PRE2.PREREG_VERSION,
                          "prereg_hash": PRE2.plan_hash(),
                          "registered": list(PRE2.ADVERSARIES),
                          "historical_failed": list(PRE2.HISTORICAL),
                          "rejected": list(PRE2.REJECTED_CANDIDATES)}, indent=2))
        return 0
    if not a.adversary or not (a.smoke or a.build or a.run):
        print("choose --list, or --adversary {A3,A4} with --smoke/--build/--run")
        return 1

    adv = PRE2.ADVERSARIES[a.adversary]
    smoke = a.smoke
    git_info = _git()
    precond = {"gated": False}
    if a.run:
        precond = {"gated": True, "registration_hash_verified": True,
                   **confirmatory_preflight(git_info, a.allow_no_git)}
        if not precond["ok"]:
            print(precond["reason"]); return 2
    out = a.out_dir or (EVIDENCE_ROOT /
                        f"{a.adversary.lower()}_{'smoke' if smoke else datetime.now():%Y%m%d}"
                        if not smoke else EVIDENCE_ROOT / f"{a.adversary.lower()}_smoke")
    out.mkdir(parents=True, exist_ok=True)
    started = utc()

    counts = ({"CE2": 2, "CE3": 2, "CE4": 2} if smoke
              else PRE2.ADVERSARIAL_PROTOCOL["replication"])
    samples = ({"CE2": 4_000, "CE3": 2_000, "CE4": 3_000} if smoke
               else PRE2.ADVERSARIAL_PROTOCOL["sample_counts"])
    n_sealed, n_calib = (8_000, 4_000) if smoke else (1_000_000, 200_000)

    import keras

    if a.adversary == "A3":
        model_path = out / "a3_representation.keras"
        if a.run:
            # CONFIRMATORY: never build. Verify the frozen artifact and load it.
            freeze = load_frozen_a3(out, model_path)
            print(f"A3 frozen at commit {freeze['freeze_git_commit']}; "
                  f"source hashes verified")
            model = keras.models.load_model(model_path)
            train_record = {**freeze["training_record"],
                            "frozen_manifest_verified": True,
                            "frozen_model_sha256": freeze["model_sha256"],
                            "freeze_git_commit": freeze["freeze_git_commit"],
                            "source_integrity_verified": True}
        elif a.build or not model_path.exists():
            if a.run:
                raise SystemExit("unreachable")
            model, train_record = build_a3(20_000 if smoke else 1_000_000,
                                           2 if smoke else 10, seed=20261006)
            model.save(model_path)
            (out / "a3_training.json").write_text(json.dumps(train_record, indent=2))
            man = write_freeze_manifest(out, model_path, train_record)
            print(f"A3 saved: {rel(model_path)}")
            print(f"A3 frozen: sha256 {man['model_sha256']}")
            if a.build:
                return 0
        else:
            model = keras.models.load_model(model_path)
            train_record = json.loads((out / "a3_training.json").read_text())
        fidelity = {"statistic": "output constant", "delivered": True,
                    "note": "verified during construction"}
    else:
        model = A4XorResponse()
        model_path = out / "a4_xor_response.json"
        model.save(model_path)
        train_record = {"adversary": "A4", "kind": "closed-form, no training",
                        "map": "keyed BLAKE2b, digest/2**64", "key": A4_MAP_KEY.decode()}
        fidelity = verify_a4_xor_dependence(model, n=2_000 if smoke else 20_000)
        print(f"A4 fidelity (exact XOR invariance): {fidelity['delivered']}")

    print("distinguisher assessment on the sealed set ...")
    dist = assess_distinguisher(model, out, n_sealed=n_sealed, n_calib=n_calib,
                                n_boot=200 if smoke else 2000)
    construction = dist["construction_status"]
    if not fidelity.get("delivered", True):
        construction = PRE2.CONSTRUCTION_FAILED
    print(f"  AUC {dist['auc']:.4f} CI {dist['auc_ci95_bootstrap']} -> "
          f"construction {construction}")

    print("model-binding gate ...")
    binding = binding_check(adv, model_path)
    print(f"  {binding['outcome']}")

    print("dimensions ...")
    dims = evaluate_dimensions(model, out / "dimensions", adv, counts, samples,
                               seed=20261006, construction_status=construction,
                               log=lambda *x: print("   ", *x))

    observed = {"fidelity": fidelity, "distinguisher": dist, "binding": binding,
                "dimensions": dims}
    cert = {
        "certificate_schema_version": "adversarial-certificate-2",
        "status": ("SMOKE ONLY - NOT SCIENTIFIC EVIDENCE" if smoke else
                   "ADVERSARIAL - EVIDENCE ABOUT THE AUDIT DIMENSIONS ONLY, "
                   "NOT ABOUT THE AUDITED MODEL"),
        "global_conclusion_unchanged": PRE2.GLOBAL_CONCLUSION_UNCHANGED,
        "identity": {
            "adversary_id": adv.adversary_id, "adversary_version": adv.version,
            "prereg_version": PRE2.PREREG_VERSION, "prereg_hash": PRE2.plan_hash(),
            "supersedes": PRE2.SUPERSEDES,
            "source_hashes": source_hashes(),
            "model_artifact": rel(model_path),
            "model_sha256": SS.sha256_file(model_path),
            "git": git_info, "confirmatory_preconditions": precond,
            "environment": PR.environment(),
            "command_line": " ".join(sys.argv),
            "started_utc": started, "finished_utc": utc()},
        "construction": {**train_record, "fidelity": fidelity,
                         "construction_status": construction,
                         "non_distinguisher_argument": adv.non_distinguisher_argument,
                         "analytic_guarantee": adv.analytic_guarantee},
        "distinguisher_assessment": dist,
        "model_binding": binding,
        "dimensions": dims,
        "prediction_comparison": compare_predictions(adv, observed),
        "protocol": PRE2.ADVERSARIAL_PROTOCOL,
        "interpretation_rules": PRE2.INTERPRETATION_RULES,
        "hypotheses": PRE2.HYPOTHESES,
        "limitations": [
            "evidence about the audit dimensions only; nothing here is evidence "
            "about the audited Gohr model or about Speck",
            "one adversary instance per registered adversary",
            "the sealed set is not seed-replayable (os.urandom); reproducibility "
            "rests on the persisted, hashed arrays",
        ],
    }
    if construction == PRE2.CONSTRUCTION_FAILED:
        cert["interpretation_suppressed"] = (
            "construction FAILED: every adversarial outcome is NOT_APPLICABLE and no "
            "hypothesis is tested. Raw scientific decisions are retained above.")
    (out / f"{adv.adversary_id.lower()}_certificate.json").write_text(
        json.dumps(cert, indent=2, sort_keys=True, default=str))

    print(f"\n--- {adv.adversary_id} vs frozen predictions ---")
    for r in cert["prediction_comparison"]:
        print(f"  {r['item']:<42} predicted={str(r['predicted'])[:22]:<24} "
              f"observed={str(r['observed'])[:28]:<30} "
              f"{'agrees' if r['agrees'] else 'DIFFERS'}")
    print(f"\ncertificate: {rel(out / (adv.adversary_id.lower() + '_certificate.json'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
