from __future__ import annotations

import math

import numpy as np

from . import adapter as A
from . import design as D
from . import probes as PR
from .target_bins import coverage


class AnalysisError(RuntimeError):
    pass


FIELDS = ("mean", "scale", "coef", "intercept", "n_iter", "gradient", "pred")


def arm_names(n_permutations: int) -> list[str]:
    return ["A"] + [f"P{b:02d}" for b in range(n_permutations)] + ["T", "R", "CAL_trained", "CAL_twin"]


def check_inputs(*, classes, fold_main, perm_index, twin_main, twin_cal, cal_labels, fold_cal,
                 min_per_fold: int) -> dict:
    out = {"twin_nonconstant_main": A.n_nonconstant(twin_main), "twin_nonconstant_cal": A.n_nonconstant(twin_cal)}
    if min(out["twin_nonconstant_main"], out["twin_nonconstant_cal"]) < 2:
        raise AnalysisError(f"untrained-twin control is degenerate: {out}")
    cov = {"A": coverage(classes, fold_main, D.N_FOLDS, D.N_CLASSES, D.MIN_CLASS_FRACTION, min_per_fold)}
    for b in range(perm_index.shape[0]):
        cov[f"P{b:02d}"] = coverage(classes[perm_index[b]], fold_main, D.N_FOLDS, D.N_CLASSES, D.MIN_CLASS_FRACTION,
                                    min_per_fold)
    cov["CAL"] = coverage(cal_labels, fold_cal, D.N_FOLDS, 2, D.MIN_CLASS_FRACTION, min_per_fold)
    bad = {k: v["failures"][:3] for k, v in cov.items() if not v["passed"]}
    if bad:
        raise AnalysisError(f"class coverage failed (no rebinning): {bad}")
    out["coverage"] = cov
    return out


def fit_all(*, classes, fold_main, perm_index, trained_main, twin_main, raw_main, trained_cal, twin_cal, cal_labels,
            fold_cal, max_iter: int) -> dict:
    y = classes.astype(np.int64)
    yc = cal_labels.astype(np.int64)
    jobs = [("A", trained_main, y, fold_main, D.N_CLASSES)]
    jobs += [(f"P{b:02d}", trained_main, y[perm_index[b]], fold_main, D.N_CLASSES) for b in range(perm_index.shape[0])]
    jobs += [("T", twin_main, y, fold_main, D.N_CLASSES), ("R", raw_main, y, fold_main, D.N_CLASSES),
             ("CAL_trained", trained_cal, yc, fold_cal, 2), ("CAL_twin", twin_cal, yc, fold_cal, 2)]
    arrays = {"fold_main": fold_main.astype(np.int8), "fold_cal": fold_cal.astype(np.int8),
              "perm_index": perm_index.astype(np.int32)}
    seconds = {}
    for name, X, labels, folds, k in jobs:
        try:
            r = PR.fit_arm(X, labels, folds, k, max_iter=max_iter)
        except PR.ProbeError as exc:
            raise AnalysisError(f"arm {name}: {exc}") from exc
        for f in FIELDS:
            arrays[f"{name}__{f}"] = np.asarray(r[f])
        seconds[name] = float(np.sum(r["fit_seconds"]))
    return {"arrays": arrays, "fit_seconds": seconds}


def measure(probes: dict, *, classes, cal_labels, score_cal, n_permutations: int, tiebreak) -> dict:
    y = classes.astype(np.int64)
    yc = cal_labels.astype(np.int64)
    if not PR.tiebreak_ok(tiebreak, n_permutations):
        raise AnalysisError("the recorded S1 tie-break is not a permutation of 0 .. B")
    ba, fold_ba, exact = {}, {}, {}
    for name in arm_names(n_permutations):
        if name.startswith("P"):
            labels = y[probes["perm_index"][int(name[1:])]]
        else:
            labels = yc if name.startswith("CAL") else y
        folds = probes["fold_cal"] if name.startswith("CAL") else probes["fold_main"]
        k = 2 if name.startswith("CAL") else D.N_CLASSES
        pred = probes[f"{name}__pred"].astype(np.int64)
        ba[name] = PR.balanced_accuracy(labels, pred, k)
        fold_ba[name] = [PR.balanced_accuracy(labels[folds == f], pred[folds == f], k) for f in range(D.N_FOLDS)]
        if name == "A" or name.startswith("P"):
            exact[name] = PR.balanced_accuracy_exact(labels, pred, k)
    model_pred = (np.asarray(score_cal) > 0.5).astype(np.int64)
    ba_model = PR.balanced_accuracy(yc, model_pred, 2)
    perm_ba = [ba[f"P{b:02d}"] for b in range(n_permutations)]
    rank = PR.s1_rank(exact["A"], [exact[f"P{b:02d}"] for b in range(n_permutations)], tiebreak)
    endpoints = {
        "C_task": ba["CAL_trained"] - 0.5 - D.KAPPA * max(ba_model - 0.5, 0.0),
        "C_twin": ba["CAL_trained"] - ba["CAL_twin"],
        "S1": float(rank["centred_rank"]),
        "S2": ba["A"] - ba["T"],
        "raw_gap": ba["A"] - ba["R"],
        "S1_margin": ba["A"] - float(np.median(perm_ba)),
    }
    finite = all(math.isfinite(v) for v in endpoints.values())
    convergence = {name: {"max_n_iter": int(np.max(probes[f"{name}__n_iter"])),
                          "max_gradient": float(np.max(probes[f"{name}__gradient"]))}
                   for name in arm_names(n_permutations)}
    return {"status": "VALID" if finite else "EXCLUDED",
            "exclusion_reason": None if finite else "non-finite endpoint",
            "balanced_accuracy": ba, "fold_balanced_accuracy": fold_ba, "ba_cal_model": ba_model,
            "balanced_accuracy_exact": {k: [v.numerator, v.denominator] for k, v in exact.items()},
            "s1_rank": {"rank": rank["rank"], "n_assignments": rank["n_assignments"],
                        "ties_with_observed": rank["ties_with_observed"], "upper_half": rank["upper_half"],
                        "tiebreak": [int(v) for v in tiebreak]},
            "permuted_ba_median": float(np.median(perm_ba)), "endpoints": endpoints, "convergence": convergence,
            "zero_endpoints": [k for k, v in endpoints.items() if v == 0.0]}
