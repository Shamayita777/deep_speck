"""
A1 - THEORY-MIMIC ADVERSARY.

Construction, training, and evaluation against the frozen predictions in
preregistration.py.

A1 has the reference architecture but is trained by regression onto a
monotone rescaling of the analytical single-trail probability of real
difference-bearing ciphertext pairs. It never sees a random-class sample and
is never trained on the real/random label, so by construction it is not a
distinguisher.

Why a MONOTONE rescaling: CE2's estimand is Spearman rank correlation, which
is invariant under any strictly increasing transform of either variable. The
network's output passes through a sigmoid, so the regression target must live
in [0, 1]; min-max rescaling of log2(p) is the simplest monotone map that
does that. It changes nothing CE2 measures.

Isolation: this package imports audit.cryptography and audit.cryptography.ce234
read-only and modifies neither. It never touches a CE1 run directory and never
writes into evidence_current/ce{1,2,3,4}; adversarial output has its own root.

    python -m audit.adversarial.a1_theory_mimic --smoke
    python -m audit.adversarial.a1_theory_mimic --build
    python -m audit.adversarial.a1_theory_mimic --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit.adversarial import preregistration as PRE
from audit.cryptography.ce234 import frozen_plan as P
from audit.cryptography.ce234 import production as PR

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ADV_ROOT = REPO_ROOT / "audit" / "adversarial" / "evidence"

#: Frozen A1 training configuration. Declared here, before training.
A1_CONFIG = {
    "architecture": "reference GohrModel(depth=10, regularization=1e-5)",
    "objective": "regression (MSE) onto a monotone rescaling of log2(trail probability)",
    "target_transform": "min-max rescaling of log2(p) over the training sample; "
                        "strictly increasing, so Spearman is unchanged",
    "trained_on": "REAL difference-bearing pairs only; no random-class samples",
    "label_never_seen": "real/random class label",
    "rounds": P.ROUNDS,
    "differential": list(P.DIFFERENTIAL),
    "n_train": 1_000_000,
    "epochs": 10,
    "batch_size": P.PREDICT_BATCH,
    "optimizer": "adam",
    "seed": 20261006,
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def build_a1(n_train: int, epochs: int, seed: int, log=print):
    """Train A1. Returns (model, training_record)."""
    from audit.cryptography.gohr.model import GohrModel
    from audit.cryptography.gohr.trainer import GohrTrainer

    t0 = time.time()
    GohrTrainer.set_seed(seed)
    model = GohrModel(depth=P.DEPTH, regularization=P.L2_REG).build()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    X, target = PR.generate_theory_data(n_train)
    feasible = target > 0
    if feasible.sum() < 0.5 * target.size:
        raise RuntimeError("unexpectedly many infeasible trails; aborting")
    y = np.zeros_like(target, dtype=np.float32)
    lo = np.log2(target[feasible]).min()
    hi = np.log2(target[feasible]).max()
    y[feasible] = ((np.log2(target[feasible]) - lo) / (hi - lo)).astype(np.float32)

    Xv, tv = PR.generate_theory_data(max(10_000, n_train // 10))
    fv = tv > 0
    yv = np.zeros_like(tv, dtype=np.float32)
    yv[fv] = np.clip((np.log2(tv[fv]) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    hist = model.fit(X, y, batch_size=A1_CONFIG["batch_size"], epochs=epochs,
                     validation_data=(Xv, yv), verbose=2)
    record = {
        **A1_CONFIG, "n_train": int(n_train), "epochs": int(epochs), "seed": int(seed),
        "log2p_min": float(lo), "log2p_max": float(hi),
        "final_train_mse": float(hist.history["loss"][-1]),
        "final_val_mse": float(hist.history["val_loss"][-1]),
        "train_seconds": round(time.time() - t0, 1),
        "trained_utc": utc(),
    }
    return model, record


# ---------------------------------------------------------------------------
# step 1 - is A1 actually not a distinguisher?
# ---------------------------------------------------------------------------

def fidelity_check(model, n: int = 200_000) -> dict:
    """
    MANIPULATION CHECK: did we actually build a theory-mimic? Measured on
    held-out real pairs, independent of any dimension verdict.
    """
    from scipy.stats import spearmanr

    X, target = PR.generate_theory_data(n)
    pred = PR.predict(model, X)
    rho = float(spearmanr(target, pred).statistic)
    thr = PRE.A1.manipulation_check["threshold"]
    return {"n": int(n), "rho_output_vs_target": rho, "threshold": thr,
            "delivered": bool(rho >= thr),
            "interpretation": "A1 must track the analytical quantity to BE the "
                              "registered adversary; below threshold means the "
                              "treatment was not delivered"}


def distinguisher_accuracy(model, n: int = 200_000, seed: int = 7) -> dict:
    """
    Accuracy on the real/random task A1 was never trained for. If this is not
    close to chance, the construction has failed and the pre-registration says
    no conclusion about the dimensions may be drawn.
    """
    from audit.cryptography.gohr import speck as sp

    X, Y = sp.make_train_data(n, P.ROUNDS, diff=P.DIFFERENTIAL)
    pred = PR.predict(model, X)
    # A1's output is a trail-probability regressor, so a 0.5 cut is not
    # necessarily its best threshold. Report the BEST achievable accuracy over
    # all thresholds: that is the most generous possible reading, and if even
    # that is near chance the construction is sound.
    order = np.argsort(pred)
    y_sorted = Y[order]
    cum = np.cumsum(y_sorted)
    total_pos = int(Y.sum()); n_tot = Y.size
    # predict "random" below the cut, "real" above it
    correct = (np.arange(1, n_tot + 1) - cum) + (total_pos - cum)
    best = float(max(correct.max() / n_tot, 1.0 - correct.min() / n_tot))
    return {"n": int(n), "accuracy_at_0.5": float(np.mean((pred > 0.5) == Y)),
            "best_accuracy_over_all_thresholds": best,
            "output_mean": float(pred.mean()), "output_std": float(pred.std()),
            "interpretation": "near 0.5 means A1 is not a distinguisher, as designed"}


# ---------------------------------------------------------------------------
# step 2 - the ordinary model-binding gate
# ---------------------------------------------------------------------------

def gate_refuses(model_path: Path) -> dict:
    """Submit A1 to the ordinary CE2-CE4 preflight binding and record the result."""
    try:
        PR.verify_reference_model(checkpoint=model_path, behavioural=True)
        return {"outcome": PRE.NOT_CAUGHT, "refused": False,
                "note": "THE GATE DID NOT REFUSE A1 - this is a framework defect "
                        "and must be reported, not patched silently"}
    except PR.PreflightError as exc:
        return {"outcome": PRE.CAUGHT, "refused": True, "reason": str(exc)[:300]}


# ---------------------------------------------------------------------------
# step 3 - the counterfactual: dimensions with the gate deliberately bypassed
# ---------------------------------------------------------------------------

def run_dimensions(model, out_dir: Path, scale: dict, seed: int, log=print) -> dict:
    """
    Run CE2, CE3 and CE4 on A1 with the binding gate BYPASSED, by calling the
    dimension estimators directly. This is the counterfactual the hypothesis
    needs: what would these dimensions have concluded if the gate had let A1
    through?
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}

    log("  CE2 ...")
    ce2 = PR.run_ce2(out_dir / "ce2", n_runs=scale["ce2"]["n_runs"],
                     n_samples=scale["ce2"]["n_samples"], seed=seed, model=model,
                     log=log)
    res["CE2"] = {"decision": ce2.get("decision"),
                  "median_rho": (ce2.get("endpoints", {}).get("primary", {})
                                 .get("median")),
                  "n_valid_runs": ce2.get("n_valid_runs"),
                  "reason": ce2.get("decision_reason") or ce2.get("reason")}

    log("  CE3 ...")
    ce3 = PR.run_ce3(out_dir / "ce3", n_replicates=scale["ce3"]["n_replicates"],
                     n_samples=scale["ce3"]["n_samples"],
                     n_splits=scale["ce3"]["n_splits"], seed=seed + 100, model=model,
                     log=log)
    res["CE3"] = {"decision": ce3.get("decision"),
                  "mean_selectivity": (ce3.get("primary") or {}).get("mean"),
                  "n_replicates": len(ce3.get("replicates", [])),
                  "reason": ce3.get("decision_reason") or ce3.get("reason")}

    log("  CE4 ...")
    ce4 = PR.run_ce4(out_dir / "ce4", n_runs=scale["ce4"]["n_runs"],
                     n_samples=scale["ce4"]["n_samples"], seed=seed + 200,
                     model=model, magnitudes=P.CE4.magnitude_ladder_bits, log=log)
    res["CE4"] = {"decision": ce4.get("decision"),
                  "mean_of_means": (ce4.get("primary") or {}).get("mean_of_means"),
                  "reason": ce4.get("decision_reason") or ce4.get("reason")}
    return res


# ---------------------------------------------------------------------------
# scoring against the frozen predictions
# ---------------------------------------------------------------------------

def score(observed: dict) -> dict:
    """Compare every observation with the frozen prediction. No re-interpretation."""
    a1 = PRE.A1
    rows = []

    def positive(d):
        return PRE.NOT_CAUGHT if d == "SUPPORTED" else PRE.CAUGHT

    rows.append({
        "component": "model_binding_gate",
        "predicted": a1.predictions["model_binding_gate"],
        "observed": observed["gate"]["outcome"],
        "agrees": observed["gate"]["outcome"] == a1.predictions["model_binding_gate"]})

    for dim, key in (("CE2_theory_consistency", "CE2"),
                     ("CE3_representation_decodability", "CE3"),
                     ("CE4_intervention_sensitivity", "CE4")):
        obs = positive(observed["dimensions"][key]["decision"])
        rows.append({"component": dim, "predicted": a1.predictions[dim],
                     "observed": obs,
                     "decision": observed["dimensions"][key]["decision"],
                     "agrees": obs == a1.predictions[dim]})

    acc = observed["distinguisher"]["best_accuracy_over_all_thresholds"]
    fid = observed["fidelity"]
    rows.append({"component": "manipulation_check (treatment fidelity)",
                 "predicted": f"rho >= {fid['threshold']}",
                 "observed": f"{fid['rho_output_vs_target']:.4f}",
                 "agrees": fid["delivered"]})
    construction_ok = abs(acc - 0.5) < 0.05 and fid["delivered"]
    rows.append({"component": "distinguisher_accuracy_on_sealed_set",
                 "predicted": a1.predictions["distinguisher_accuracy_on_sealed_set"],
                 "observed": f"{acc:.4f}", "agrees": construction_ok})

    not_caught = [r["component"] for r in rows
                  if r.get("observed") == PRE.NOT_CAUGHT]
    caught = [r["component"] for r in rows if r.get("observed") == PRE.CAUGHT]
    if not construction_ok:
        finding = ("CONSTRUCTION FAILED: A1 separates real from random better than "
                   "chance, so no conclusion about the dimensions may be drawn from "
                   "it. The adversary must be rebuilt and re-registered.")
    elif caught == ["model_binding_gate"] and len(not_caught) == 3:
        finding = ("H-ADV-1 SUPPORTED: a model that is not a distinguisher was "
                   "SUPPORTED by CE2, CE3 and CE4 and was stopped only by the "
                   "model-binding gate. Those dimensions' verdicts must therefore "
                   "always be reported as conditional on the binding.")
    elif not caught:
        finding = ("FRAMEWORK DEFECT: nothing caught A1, including the binding gate. "
                   "Report and fix.")
    else:
        finding = (f"PARTIAL: caught by {caught}; not caught by {not_caught}. Record "
                   "which dimensions discriminate on their own.")

    return {"rows": rows, "all_predictions_correct": all(r["agrees"] for r in rows),
            "caught_by": caught, "not_caught_by": not_caught, "finding": finding}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import keras

    ap = argparse.ArgumentParser(description="A1 theory-mimic adversary")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny scale for pipeline validation; NOT a result")
    ap.add_argument("--build", action="store_true", help="train A1 and stop")
    ap.add_argument("--run", action="store_true", help="full battery for A1")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--model-path", type=Path, default=None)
    a = ap.parse_args(argv)
    if not (a.smoke or a.build or a.run):
        print("choose --smoke, --build or --run"); return 1

    smoke = a.smoke
    out = a.out_dir or (ADV_ROOT / ("a1_smoke" if smoke else
                                    f"a1_{datetime.now():%Y%m%d}"))
    out.mkdir(parents=True, exist_ok=True)
    model_path = a.model_path or (out / "a1_theory_mimic.keras")

    n_train = 20_000 if smoke else A1_CONFIG["n_train"]
    epochs = 2 if smoke else A1_CONFIG["epochs"]
    scale = (PRE.ADVERSARIAL_SCALE if not smoke else
             {"ce2": {"n_runs": 2, "n_samples": 5_000},
              "ce3": {"n_replicates": 2, "n_samples": 2_000, "n_splits": 3},
              "ce4": {"n_runs": 2, "n_samples": 2_000}})

    if model_path.exists() and not a.build:
        print(f"loading A1 from {model_path}")
        model = keras.models.load_model(model_path)
        train_record = json.loads((out / "a1_training.json").read_text())
    else:
        print("training A1 ...")
        model, train_record = build_a1(n_train, epochs, A1_CONFIG["seed"])
        model.save(model_path)
        (out / "a1_training.json").write_text(json.dumps(train_record, indent=2))
        print(f"A1 saved: {model_path}  (val mse {train_record['final_val_mse']:.5f}, "
              f"{train_record['train_seconds']}s)")
        if a.build:
            return 0

    print("step 0: manipulation check - was the adversary actually built?")
    fid = fidelity_check(model, n=20_000 if smoke else 200_000)
    print(f"  rho(A1 output, analytical target) = {fid['rho_output_vs_target']:.4f} "
          f"(threshold {fid['threshold']}) -> "
          f"{'delivered' if fid['delivered'] else 'NOT DELIVERED'}")

    print("step 1: is A1 a distinguisher?")
    dist = distinguisher_accuracy(model, n=20_000 if smoke else 200_000)
    print(f"  best accuracy over all thresholds: "
          f"{dist['best_accuracy_over_all_thresholds']:.4f}")

    print("step 2: ordinary model-binding gate")
    gate = gate_refuses(model_path)
    print(f"  {gate['outcome']}: {gate.get('reason', gate.get('note',''))[:120]}")

    print("step 3: dimensions with the gate deliberately bypassed")
    dims = run_dimensions(model, out / "dimensions", scale,
                          seed=20261006, log=lambda *x: print("   ", *x))

    observed = {"fidelity": fid, "distinguisher": dist, "gate": gate,
                "dimensions": dims}
    verdict = score(observed)
    cert = {
        "certificate_schema_version": "adversarial-certificate-1",
        "adversary_id": "A1", "adversary_name": PRE.A1.name,
        "status": "ADVERSARIAL - NOT EVIDENCE ABOUT THE AUDITED MODEL",
        "smoke": bool(smoke),
        "prereg_version": PRE.PREREG_VERSION, "prereg_hash": PRE.plan_hash(),
        "hypothesis": PRE.HYPOTHESIS,
        "scale": scale, "frozen_predictions": PRE.A1.predictions,
        "training": train_record, "model_sha256": _sha256_file(model_path),
        "observed": observed, "scoring": verdict,
        "environment": PR.environment(), "utc": utc(),
    }
    if smoke:
        cert["status"] = "SMOKE ONLY - NOT SCIENTIFIC EVIDENCE"
    (out / "a1_certificate.json").write_text(
        json.dumps(cert, indent=2, sort_keys=True, default=str))

    print("\n--- A1 vs frozen predictions ---")
    for r in verdict["rows"]:
        mark = "agrees" if r["agrees"] else "DIFFERS FROM PREDICTION"
        print(f"  {r['component']:<42} predicted={r['predicted']:<14} "
              f"observed={r['observed']:<14} {mark}")
    print(f"\n{verdict['finding']}")
    print(f"certificate: {out / 'a1_certificate.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
