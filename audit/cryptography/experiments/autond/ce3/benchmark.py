from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ....core import evidence as EV
from ....targets.autond import upstream as UP
from . import adapter as A
from . import design as D
from . import probes as PR
from . import runner as R

SCHEMA = "autond-ce3-blinded-benchmark-2"
TRUE_TARGET_ARMS = ("A", "T", "R")
LABEL_USE = {
    "A": "true target classes on a fresh non-evidentiary dataset (fit only)",
    "P_b": "group-permuted target classes (fit only)",
    "T": "true target classes on a fresh non-evidentiary dataset (fit only)",
    "R": "true target classes on a fresh non-evidentiary dataset (fit only)",
    "CAL_trained": "calibration labels genuine versus random, not the target (fit only)",
    "CAL_twin": "calibration labels genuine versus random, not the target (fit only)",
}


def _arms(classes, perm_index, reps, X, cal_labels):
    y = classes.astype(np.int64)
    yield "A", reps["trained_main"], y, "main", D.N_CLASSES
    for b in range(perm_index.shape[0]):
        yield f"P{b:02d}", reps["trained_main"], y[perm_index[b]], "main", D.N_CLASSES
    yield "T", reps["twin_main"], y, "main", D.N_CLASSES
    yield "R", X.astype(np.float64), y, "main", D.N_CLASSES
    yield "CAL_trained", reps["trained_cal"], cal_labels.astype(np.int64), "cal", 2
    yield "CAL_twin", reps["twin_cal"], cal_labels.astype(np.int64), "cal", 2


def run(path: Path, *, upstream=None) -> dict:
    path = Path(path)
    if path.exists():
        raise R.RunnerError(f"refusing to overwrite {path}")
    t0 = time.time()
    up = UP.load(upstream)
    env = R.environment()
    model, twin, binding = A.load_models(up, R.CRYPTO_ROOT / D.CE2_RUN_RELPATH / D.CE2_TWIN_RELPATH)
    ext = {"published": A.extractor(model), "twin": A.extractor(twin)}
    i = D.BENCHMARK_REPLICATE_INDEX
    main = A.main_dataset(up, D.N_MAIN_PAIRS)
    cal = A.calibration_dataset(up, D.N_CALIBRATION_PER_CLASS, i)
    X = A.model_input(up, main["arrays"]["c0"], main["arrays"]["c1"])
    Xc = A.calibration_input(up, cal["arrays"])
    reps = {"trained_main": A.extract(ext["published"], X), "twin_main": A.extract(ext["twin"], X),
            "trained_cal": A.extract(ext["published"], Xc), "twin_cal": A.extract(ext["twin"], Xc)}
    classes = A.classes_for(main["arrays"]["target"])
    folds = {"main": PR.fold_ids(classes.size, PR.main_fold_seed(i)),
             "cal": PR.fold_ids(cal["arrays"]["labels"].size, PR.calibration_fold_seed(i))}
    perm_index = PR.permutations(classes.size, i, D.N_PERMUTATIONS)
    runs = {}
    for max_iter in (D.MAX_ITER, D.FALLBACK_MAX_ITER):
        rows = {}
        t = time.time()
        for name, feats, labels, fk, k in _arms(classes, perm_index, reps, X, cal["arrays"]["labels"]):
            rows[name] = PR.fit_only(feats, labels, folds[fk], k, max_iter)
        conv = all(r["converged"] for v in rows.values() for r in v)
        runs[str(max_iter)] = {"all_converged": conv, "seconds": time.time() - t,
                               "converged_per_arm": {a: all(r["converged"] for r in v) for a, v in rows.items()},
                               "convergence_warnings": sum(r["convergence_warning"] for v in rows.values() for r in v),
                               "n_fits": sum(len(v) for v in rows.values()),
                               "max_n_iter": max(r["n_iter"] for v in rows.values() for r in v),
                               "max_gradient": max(r["gradient_inf_norm"] for v in rows.values() for r in v)}
        del rows
        if conv:
            break
    selected = next((int(k) for k, v in runs.items() if v["all_converged"]), None)
    rec = {"schema": SCHEMA, "experiment_id": D.EXPERIMENT_ID, "evidentiary": False, "confirmatory": False,
           "scores_computed": False, "predictions_computed": False,
           "blinding": ("probes are fitted on a fresh non-evidentiary dataset but never asked to predict: no held-out "
                        "prediction, balanced accuracy, accuracy, loss value or score of any arm is computed or "
                        "stored; coefficients are discarded; the record keeps only per-arm convergence flags and the "
                        "maximum iteration count, maximum gradient norm, warning count and time over all fits"),
           "true_target_labels_used_in_fitting": list(TRUE_TARGET_ARMS),
           "label_use": LABEL_USE,
           "training_fold_quantities": ("fitting necessarily evaluates the training-fold class probabilities inside "
                                        "lbfgs and in the independent gradient check; they are reduced at once to a "
                                        "single gradient norm per fit and are neither stored nor turned into an "
                                        "accuracy"),
           "side_channel_disclosed": ("the maximum iteration count over all arms could weakly reflect signal "
                                      "strength; per-arm iteration counts are not recorded"),
           "permutation_construction": D.DESIGN["sampling"]["permutation_construction"],
           "replicate_index": i, "main_pairs": D.N_MAIN_PAIRS, "calibration_per_class": D.N_CALIBRATION_PER_CLASS,
           "permutations": D.N_PERMUTATIONS, "edges": D.TARGET_EDGES, "probe": D.PROBE,
           "generation_gates_passed": bool(main["gates"]["E"]["passed"] and main["gates"]["F"]["passed"]
                                           and cal["gates"]["E"]["passed"] and cal["gates"]["F"]["passed"]),
           "binding": {"checkpoint_sha256": binding["sha256"], "published_weights_sha256": binding["weights_sha256"],
                       "twin_file_sha256": D.CE2_TWIN_FILE_SHA256, "twin_weights_sha256": D.CE2_TWIN_WEIGHTS_SHA256,
                       "representation": {"published": A.bind_representation(model)["ok"],
                                          "twin": A.bind_representation(twin)["ok"]}},
           "data_retained": False, "data_content_sha256": A.content_sha256(main["arrays"]),
           "environment": env, "runtime_problems": R.runtime_problems(env), "runs": runs,
           "all_converged": selected is not None, "selected_max_iter": selected,
           "seconds_total": time.time() - t0, "utc": EV.utc()}
    EV.write_json(path, rec)
    return rec
