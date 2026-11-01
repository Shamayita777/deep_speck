from __future__ import annotations

import time
import warnings
from fractions import Fraction
from random import SystemRandom

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import design as D


class ProbeError(RuntimeError):
    pass


def fold_ids(n: int, seed: int, n_folds: int = D.N_FOLDS) -> np.ndarray:
    if n < n_folds:
        raise ProbeError("fewer samples than folds")
    out = np.full(n, -1, dtype=np.int8)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(np.zeros((n, 1), dtype=np.uint8))):
        out[test] = k
    if np.any(out < 0):
        raise ProbeError("KFold left a sample without a test fold")
    return out


def main_fold_seed(replicate_index: int) -> int:
    return D.FOLD_SEED_BASE + int(replicate_index)


def calibration_fold_seed(replicate_index: int) -> int:
    return D.CALIBRATION_FOLD_SEED_BASE + int(replicate_index)


def permutation_seed(replicate_index: int) -> int:
    if replicate_index < 0:
        raise ProbeError("replicate index outside the seed scheme")
    return D.PERMUTATION_SEED_BASE + int(replicate_index)


def _cycle_lengths(n: int, order: int) -> list[int]:
    divisors = [d for d in range(order, 0, -1) if order % d == 0]
    q, r = divmod(n, order)
    out = [order] * q
    for d in divisors:
        while r >= d:
            out.append(d)
            r -= d
    return out


def cyclic_generator(n: int, order: int, seed: int) -> np.ndarray:
    if n < order:
        raise ProbeError("fewer samples than the permutation group order")
    shuffled = np.random.Generator(np.random.PCG64(seed)).permutation(n)
    sigma = np.empty(n, dtype=np.int64)
    start = 0
    for length in _cycle_lengths(n, order):
        cyc = shuffled[start:start + length]
        sigma[cyc] = np.roll(cyc, -1)
        start += length
    return sigma


def permutations(n: int, replicate_index: int, n_permutations: int) -> np.ndarray:
    if n_permutations < 1 or n_permutations % 2 == 0:
        raise ProbeError("the number of permutations must be a positive odd number")
    sigma = cyclic_generator(n, n_permutations + 1, permutation_seed(replicate_index))
    out = np.empty((n_permutations, n), dtype=np.int64)
    cur = sigma
    for b in range(n_permutations):
        out[b] = cur
        cur = sigma[cur]
    if not np.array_equal(cur, np.arange(n)):
        raise ProbeError("the permutation group does not close")
    return out


def draw_tiebreak(n_permutations: int) -> np.ndarray:
    return np.asarray(SystemRandom().sample(range(n_permutations + 1), n_permutations + 1), dtype=np.int16)


def tiebreak_ok(tiebreak, n_permutations: int) -> bool:
    t = np.asarray(tiebreak)
    return t.shape == (n_permutations + 1,) and np.array_equal(np.sort(t.astype(np.int64)),
                                                               np.arange(n_permutations + 1))


def balanced_accuracy_exact(y: np.ndarray, pred: np.ndarray, n_classes: int) -> Fraction:
    total = Fraction(0)
    for c in range(n_classes):
        m = y == c
        k = int(np.count_nonzero(m))
        if k == 0:
            raise ProbeError("balanced accuracy is undefined for an absent class")
        total += Fraction(int(np.count_nonzero(pred[m] == c)), k)
    return total / n_classes


def s1_rank(observed: Fraction, permuted: list, tiebreak) -> dict:
    t = [int(v) for v in tiebreak]
    if len(t) != len(permuted) + 1:
        raise ProbeError("one tie-break value is required for the observed and for each permuted assignment")
    below = sum(1 for b, s in enumerate(permuted) if (s, t[b + 1]) < (observed, t[0]))
    ties = sum(1 for s in permuted if s == observed)
    rank = below + 1
    m = len(permuted) + 1
    return {"rank": rank, "n_assignments": m, "ties_with_observed": ties, "upper_half": rank > m // 2,
            "centred_rank": rank - Fraction(m + 1, 2)}


def pipeline(max_iter: int) -> Pipeline:
    if max_iter not in (D.MAX_ITER, D.FALLBACK_MAX_ITER):
        raise ProbeError("max_iter must be the primary value or the pre-registered fallback")
    cfg = {k: v for k, v in D.PROBE["logistic"].items() if k != "class"}
    cfg["max_iter"] = max_iter
    return Pipeline([("scale", StandardScaler(copy=True, with_mean=True, with_std=True)),
                     ("logistic", LogisticRegression(**cfg))])


def objective_gradient(Z: np.ndarray, y: np.ndarray, classes: np.ndarray, coef: np.ndarray, intercept: np.ndarray,
                       C: float) -> float:
    n = Z.shape[0]
    if classes.size == 2:
        z = Z @ coef[0] + intercept[0]
        r = 1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0))) - (y == classes[1])
        g = np.concatenate([Z.T @ r / n + coef[0] / (C * n), [r.mean()]])
    else:
        L = Z @ coef.T + intercept
        L -= L.max(axis=1, keepdims=True)
        P = np.exp(L)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(n), np.searchsorted(classes, y)] -= 1.0
        g = np.concatenate([(P.T @ Z / n + coef / (C * n)).ravel(), P.mean(axis=0)])
    return float(np.max(np.abs(g)))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        m = y == c
        k = int(np.count_nonzero(m))
        if k == 0:
            raise ProbeError("balanced accuracy is undefined for an absent class")
        recalls.append(np.count_nonzero(pred[m] == c) / k)
    return float(np.mean(recalls))


def _fit_fold(X, y, train, max_iter):
    est = pipeline(max_iter)
    with threadpool_limits(limits=D.PROBE["blas_threads"]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            est.fit(X[train], y[train])
        sc, lr = est.named_steps["scale"], est.named_steps["logistic"]
        n_iter = int(np.max(np.asarray(lr.n_iter_)))
        Z = (X[train] - sc.mean_) / sc.scale_
        grad = objective_gradient(Z, y[train], lr.classes_, lr.coef_.astype(np.float64),
                                  np.asarray(lr.intercept_, dtype=np.float64), float(D.PROBE["logistic"]["C"]))
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    converged = (not warned) and n_iter < max_iter and grad <= D.GRADIENT_TOLERANCE
    return est, {"n_iter": n_iter, "gradient_inf_norm": grad, "convergence_warning": warned, "converged": converged}


def fit_arm(features, labels, folds: np.ndarray, n_classes: int, *, max_iter: int, predict: bool = True) -> dict:
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    if X.ndim != 2 or X.shape[0] != y.size or folds.shape != y.shape:
        raise ProbeError("features must be N x d and labels and folds of length N")
    if not np.all(np.isfinite(X)):
        raise ProbeError("features contain non-finite values")
    n_folds = int(folds.max()) + 1
    d, k_out = X.shape[1], (1 if n_classes == 2 else n_classes)
    out = {"mean": np.zeros((n_folds, d)), "scale": np.zeros((n_folds, d)), "coef": np.zeros((n_folds, k_out, d)),
           "intercept": np.zeros((n_folds, k_out)), "n_iter": np.zeros(n_folds, np.int64),
           "gradient": np.zeros(n_folds), "fit_seconds": np.zeros(n_folds)}
    pred = np.full(y.size, -1, dtype=np.int8)
    fold_ba = []
    for k in range(n_folds):
        test = folds == k
        train = ~test
        if np.unique(y[train]).size != n_classes or np.unique(y[test]).size != n_classes:
            raise ProbeError(f"fold {k} lacks a class")
        t = time.time()
        est, info = _fit_fold(X, y, train, max_iter)
        out["fit_seconds"][k] = time.time() - t
        if not info["converged"]:
            raise ProbeError(f"fold {k} did not converge: {info}; no retry is permitted")
        sc, lr = est.named_steps["scale"], est.named_steps["logistic"]
        if not np.array_equal(lr.classes_, np.arange(n_classes)):
            raise ProbeError("fitted class order differs from 0 .. K-1")
        out["mean"][k], out["scale"][k] = sc.mean_, sc.scale_
        out["coef"][k], out["intercept"][k] = lr.coef_, np.asarray(lr.intercept_).ravel()
        out["n_iter"][k], out["gradient"][k] = info["n_iter"], info["gradient_inf_norm"]
        if predict:
            with threadpool_limits(limits=D.PROBE["blas_threads"]):
                p = est.predict(X[test]).astype(np.int8)
            pred[test] = p
            fold_ba.append(balanced_accuracy(y[test], p, n_classes))
    if predict:
        if np.any(pred < 0):
            raise ProbeError("a sample received no held-out prediction")
        out["pred"] = pred
        out["fold_ba"] = fold_ba
        out["ba"] = balanced_accuracy(y, pred, n_classes)
    return out


def fit_only(features, labels, folds: np.ndarray, n_classes: int, max_iter: int) -> list[dict]:
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    rows = []
    for k in range(int(folds.max()) + 1):
        train = folds != k
        t = time.time()
        _, info = _fit_fold(X, y, train, max_iter)
        rows.append({"fold": k, **info, "seconds": time.time() - t})
    return rows
