from __future__ import annotations

import itertools
import math

import numpy as np


class TargetBinningError(ValueError):
    pass


def integer_target(values) -> np.ndarray:
    x = np.asarray(values)
    if x.ndim != 1 or x.size == 0:
        raise TargetBinningError("target must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(x)):
        raise TargetBinningError("target contains non-finite values")
    if not np.all(x == np.floor(x)):
        raise TargetBinningError("target values must be exact integers")
    return x.astype(np.int64, copy=False)


def select_partition(values, n_classes: int, min_fraction: float) -> dict:
    x = integer_target(values)
    if n_classes < 2:
        raise TargetBinningError("at least two classes are required")
    if not (0.0 < min_fraction < 1.0 / n_classes):
        raise TargetBinningError("minimum class fraction must be positive and below 1/K")
    support, counts = np.unique(x, return_counts=True)
    n = int(x.size)
    minimum = int(math.ceil(min_fraction * n))
    out = {"n_classes": n_classes, "n": n, "support_size": int(support.size), "minimum_count": minimum}
    if support.size < n_classes:
        return {**out, "feasible": False}
    prefix = np.cumsum(counts, dtype=np.int64).tolist()
    best = None
    for idx in itertools.combinations(range(support.size - 1), n_classes - 1):
        cum = [prefix[j] for j in idx]
        cc = [cum[0]] + [cum[j] - cum[j - 1] for j in range(1, len(cum))] + [n - cum[-1]]
        if min(cc) < minimum:
            continue
        objective = sum(abs(n_classes * cum[j] - (j + 1) * n) for j in range(n_classes - 1))
        key = (objective, tuple(int(support[j]) for j in idx))
        if best is None or key < best[0]:
            best = (key, cc)
    if best is None:
        return {**out, "feasible": False}
    (objective, edges), cc = best
    return {**out, "feasible": True, "edges": list(edges), "class_counts": [int(c) for c in cc],
            "class_fractions": [c / n for c in cc], "objective_integer": int(objective)}


def choose_target_binning(values, min_fraction: float) -> dict:
    five = select_partition(values, 5, min_fraction)
    three = select_partition(values, 3, min_fraction)
    if five["feasible"]:
        chosen = five
    elif three["feasible"]:
        chosen = three
    else:
        raise TargetBinningError("neither five nor three whole-value classes satisfy the minimum class fraction")
    return {"selected": chosen, "candidates": {"5": five, "3": three}, "three_class_branch_taken": not five["feasible"]}


def apply_edges(values, edges, n_classes: int) -> np.ndarray:
    x = integer_target(values)
    e = np.asarray(edges)
    if e.ndim != 1 or e.size != n_classes - 1:
        raise TargetBinningError(f"exactly {n_classes - 1} edges are required for {n_classes} classes")
    if not np.all(np.isfinite(e)) or not np.all(e == np.floor(e)):
        raise TargetBinningError("edges must be finite integers")
    e = e.astype(np.int64)
    if not np.all(e[:-1] < e[1:]):
        raise TargetBinningError("edges must be strictly increasing")
    return np.searchsorted(e, x, side="left").astype(np.int8)


def coverage(labels, fold_ids, n_folds: int, n_classes: int, min_fraction: float, min_per_fold: int) -> dict:
    y = np.asarray(labels)
    f = np.asarray(fold_ids)
    if y.ndim != 1 or y.size == 0 or f.shape != y.shape:
        raise TargetBinningError("labels and fold ids must be equal-length one-dimensional arrays")
    if not np.all((f >= 0) & (f < n_folds)):
        raise TargetBinningError("fold id outside 0 .. n_folds-1")
    classes = list(range(n_classes))
    totals = [int(np.count_nonzero(y == c)) for c in classes]
    failures = []
    if int(np.count_nonzero(np.isin(y, classes))) != y.size:
        failures.append({"scope": "labels", "detail": "label outside 0 .. K-1"})
    need = int(math.ceil(min_fraction * y.size))
    for c, t in zip(classes, totals):
        if t < need:
            failures.append({"scope": "full", "class": c, "count": t, "minimum": need})
    folds = []
    for k in range(n_folds):
        test = f == k
        rec = {"fold": k, "train": [int(np.count_nonzero((y == c) & ~test)) for c in classes],
               "test": [int(np.count_nonzero((y == c) & test)) for c in classes]}
        for part in ("train", "test"):
            for c, cnt in zip(classes, rec[part]):
                if cnt < min_per_fold:
                    failures.append({"scope": f"fold_{k}_{part}", "class": c, "count": cnt, "minimum": min_per_fold})
        folds.append(rec)
    return {"passed": not failures, "class_counts": totals, "minimum_full": need, "minimum_per_fold": min_per_fold,
            "folds": folds, "failures": failures}
