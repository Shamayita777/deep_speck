import numpy as np
import pytest

from audit.cryptography.experiments.autond.ce3 import analysis as AN
from audit.cryptography.experiments.autond.ce3 import design as D
from audit.cryptography.experiments.autond.ce3 import probes as PR
from audit.cryptography.experiments.autond.ce3 import target_bins as TB
from audit.cryptography.experiments.autond.ce3 import verify as V


def discrete_target(rng, n=20_000):
    return -rng.binomial(56, 0.5, n).astype(np.int64) - 35


def test_edges_never_split_equal_values_and_match_verifier():
    rng = np.random.default_rng(1)
    for _ in range(5):
        t = discrete_target(rng)
        sel = TB.select_partition(t, 5, 0.15)
        y = TB.apply_edges(t, sel["edges"], 5)
        for v in np.unique(t):
            assert np.unique(y[t == v]).size == 1
        assert sel["edges"] == V.select_edges(t, 5, 0.15)
        assert np.array_equal(y, V.class_of(t, sel["edges"]))


def test_three_class_branch_only_when_five_infeasible():
    t = np.repeat([1, 2, 3], [400, 300, 300])
    c = TB.choose_target_binning(t, 0.15)
    assert c["three_class_branch_taken"] and c["selected"]["n_classes"] == 3
    rng = np.random.default_rng(2)
    assert not TB.choose_target_binning(discrete_target(rng), 0.15)["three_class_branch_taken"]


def test_frozen_edges_and_class_count_fixed_in_design():
    assert D.N_CLASSES == 5 and D.TARGET_EDGES == [-68, -63, -59, -55]
    with pytest.raises(TB.TargetBinningError):
        TB.apply_edges(np.arange(10), [-63, -59], 5)


def test_coverage_fails_closed_without_rebinning():
    y = np.array([0] * 990 + [1] * 10)
    f = PR.fold_ids(1000, 3)
    r = TB.coverage(y, f, 5, 2, 0.15, 1)
    assert not r["passed"] and any(x["scope"] == "full" for x in r["failures"])


def test_fold_ids_equal_independent_kfold_reconstruction_in_pinned_sklearn():
    for n, seed in ((1000, 40_000_000), (6001, 60_000_007), (200_000, 40_000_019)):
        assert np.array_equal(PR.fold_ids(n, seed), V.kfold_ids(n, seed))


def test_folds_depend_only_on_size_and_seed():
    a = PR.fold_ids(5000, 40_000_003)
    assert np.array_equal(a, PR.fold_ids(5000, 40_000_003))
    assert np.bincount(a).tolist() == [1000] * 5


def test_permutations_form_a_cyclic_group_seeded_and_count_preserving():
    y = np.random.default_rng(3).integers(0, 5, 3000)
    p = PR.permutations(y.size, 4, 5)
    assert p.shape == (5, 3000) and PR.permutation_seed(4) == 50_000_004
    sigma = p[0]
    for b in range(5):
        assert np.array_equal(np.bincount(y[p[b]]), np.bincount(y))
        assert np.count_nonzero(p[b] == np.arange(3000)) == 0
    assert np.array_equal(sigma[p[4]], np.arange(3000))
    for j in range(4):
        assert np.array_equal(sigma[p[j]], p[j + 1])
    assert len({x.tobytes() for x in p}) == 5
    with pytest.raises(PR.ProbeError):
        PR.permutations(10, 0, 4)


@pytest.mark.parametrize("b", [1, 3, 5, 11, 21])
def test_group_construction_for_every_remainder_matches_the_verifier(b):
    m = b + 1
    for n in range(5 * m, 6 * m):
        p = PR.permutations(n, 7, b)
        q, closes = V.group_permutations(n, 50_000_007, b)
        assert closes and np.array_equal(p, q)
        sigma = p[0]
        seen, lengths = np.zeros(n, bool), []
        for i in range(n):
            if not seen[i]:
                j, c = i, 0
                while not seen[j]:
                    seen[j], j, c = True, sigma[j], c + 1
                lengths.append(c)
        assert all(m % c == 0 for c in lengths) and lengths.count(m) == n // m


def test_cyclic_shift_identity_behind_the_exchangeability_argument():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 5, 997)
    p = PR.permutations(997, 2, 5)
    group = [np.arange(997)] + list(p)
    shifted = [y[p[0]][g] for g in group]
    assert all(np.array_equal(shifted[j], y[group[(j + 1) % 6]]) for j in range(6))


def synthetic(rng, n=4000, k=5, signal=0.5):
    y = rng.integers(0, k, n)
    X = np.maximum(rng.standard_normal((n, 16)), 0)
    X[:, 0] += signal * y
    return X, y


def test_probe_scaler_is_training_fold_only_and_parameters_verify_independently():
    rng = np.random.default_rng(4)
    X, y = synthetic(rng)
    f = PR.fold_ids(y.size, 9)
    r = PR.fit_arm(X, y, f, 5, max_iter=D.MAX_ITER)
    for k in range(5):
        tr = f != k
        m, s = V.scaler(X[tr])
        assert np.allclose(r["mean"][k], m, rtol=1e-9, atol=1e-12) and np.allclose(r["scale"][k], s, rtol=1e-9, atol=1e-12)
        Z = (X[tr] - r["mean"][k]) / r["scale"][k]
        assert V.gradient(Z, y[tr], 5, r["coef"][k], r["intercept"][k]) <= D.GRADIENT_TOLERANCE
        p, _ = V.predict((X[f == k] - r["mean"][k]) / r["scale"][k], 5, r["coef"][k], r["intercept"][k])
        assert np.array_equal(p, r["pred"][f == k])
    assert r["ba"] == V.bal_acc(y, r["pred"].astype(np.int64), 5)


def test_gradient_check_rejects_perturbed_parameters():
    rng = np.random.default_rng(5)
    X, y = synthetic(rng)
    f = PR.fold_ids(y.size, 1)
    r = PR.fit_arm(X, y, f, 5, max_iter=D.MAX_ITER)
    tr = f != 0
    Z = (X[tr] - r["mean"][0]) / r["scale"][0]
    assert V.gradient(Z, y[tr], 5, r["coef"][0] + np.eye(5, 16) * 0.05, r["intercept"][0]) > D.GRADIENT_TOLERANCE


def test_probe_is_deterministic():
    rng = np.random.default_rng(6)
    X, y = synthetic(rng)
    f = PR.fold_ids(y.size, 2)
    a, b = PR.fit_arm(X, y, f, 5, max_iter=D.MAX_ITER), PR.fit_arm(X, y, f, 5, max_iter=D.MAX_ITER)
    assert np.array_equal(a["coef"], b["coef"]) and np.array_equal(a["pred"], b["pred"])


def test_non_convergence_fails_closed_without_retry(monkeypatch):
    rng = np.random.default_rng(7)
    X, y = synthetic(rng)
    monkeypatch.setattr(D, "GRADIENT_TOLERANCE", 0.0)
    with pytest.raises(PR.ProbeError, match="no retry"):
        PR.fit_arm(X, y, PR.fold_ids(y.size, 3), 5, max_iter=D.MAX_ITER)


def test_max_iter_only_primary_or_fallback():
    with pytest.raises(PR.ProbeError):
        PR.pipeline(1000)


def test_degenerate_twin_and_permuted_coverage_fail_the_replicate():
    rng = np.random.default_rng(8)
    n = 2000
    y = rng.integers(0, 5, n).astype(np.int8)
    f = PR.fold_ids(n, 1)
    yc = np.repeat(np.array([0, 1], np.int8), n // 2)
    fc = PR.fold_ids(n, 2)
    good = rng.standard_normal((n, 8)).astype(np.float32)
    with pytest.raises(AN.AnalysisError, match="degenerate"):
        AN.check_inputs(classes=y, fold_main=f, perm_index=PR.permutations(n, 0, 5), twin_main=np.zeros((n, 8), np.float32),
                        twin_cal=good, cal_labels=yc, fold_cal=fc, min_per_fold=10)
    with pytest.raises(AN.AnalysisError, match="coverage"):
        AN.check_inputs(classes=y, fold_main=f, perm_index=PR.permutations(n, 0, 5), twin_main=good, twin_cal=good,
                        cal_labels=yc, fold_cal=fc, min_per_fold=90)


def test_s1_rank_through_measure_uses_exact_balanced_accuracy_and_the_tiebreak():
    rng = np.random.default_rng(9)
    n = 3000
    y = rng.integers(0, 5, n).astype(np.int8)
    f = PR.fold_ids(n, 1)
    perm = PR.permutations(n, 0, 5)
    yc = rng.permutation(np.repeat(np.array([0, 1], np.int8), n // 2))
    fc = PR.fold_ids(n, 2)
    const = np.zeros(n, np.int8)
    probes = {"fold_main": f, "fold_cal": fc, "perm_index": perm}
    for name in AN.arm_names(5):
        probes[f"{name}__pred"] = const if not name.startswith("CAL") else yc
        probes[f"{name}__n_iter"] = np.array([1])
        probes[f"{name}__gradient"] = np.array([0.0])
    for tb, rank in (([0, 1, 2, 3, 4, 5], 1), ([5, 0, 1, 2, 3, 4], 6), ([3, 0, 1, 2, 4, 5], 4)):
        m = AN.measure(probes, classes=y, cal_labels=yc, score_cal=np.full(n, 0.9, np.float32), n_permutations=5,
                       tiebreak=np.array(tb, np.int16))
        assert m["s1_rank"]["rank"] == rank and m["s1_rank"]["ties_with_observed"] == 5
        assert m["endpoints"]["S1"] == rank - 3.5 and m["endpoints"]["S1_margin"] == 0.0
    with pytest.raises(AN.AnalysisError):
        AN.measure(probes, classes=y, cal_labels=yc, score_cal=np.full(n, 0.9, np.float32), n_permutations=5,
                   tiebreak=np.array([0, 0, 1, 2, 3, 4], np.int16))


def test_blinded_benchmark_fits_never_predict_score_or_compute_accuracy(monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from audit.cryptography.experiments.autond.ce3 import benchmark as BM

    def forbidden(*a, **k):
        raise AssertionError("prediction or accuracy computed in the blinded benchmark")
    for cls in (LogisticRegression, Pipeline):
        for name in ("predict", "predict_proba", "predict_log_proba", "decision_function", "score"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, forbidden)
    monkeypatch.setattr(PR, "balanced_accuracy", forbidden)
    monkeypatch.setattr(PR, "balanced_accuracy_exact", forbidden)
    monkeypatch.setattr(PR, "fit_arm", forbidden)
    rng = np.random.default_rng(11)
    n = 1500
    y = rng.integers(0, 5, n).astype(np.int8)
    reps = {k: np.maximum(rng.standard_normal((n, 8)), 0).astype(np.float32)
            for k in ("trained_main", "twin_main", "trained_cal", "twin_cal")}
    X = rng.integers(0, 2, (n, 64)).astype(np.uint8)
    cal = rng.permutation(np.repeat(np.array([0, 1], np.int8), n // 2))
    folds = {"main": PR.fold_ids(n, 1), "cal": PR.fold_ids(n, 2)}
    arms = list(BM._arms(y, PR.permutations(n, 0, 5), reps, X, cal))
    assert [a[0] for a in arms] == ["A", "P00", "P01", "P02", "P03", "P04", "T", "R", "CAL_trained", "CAL_twin"]
    assert [a[0] for a in arms if a[2] is not None and a[0] in BM.TRUE_TARGET_ARMS] == ["A", "T", "R"]
    for name, feats, labels, fk, k in arms:
        rows = PR.fit_only(feats, labels, folds[fk], k, D.MAX_ITER)
        assert len(rows) == 5 and set(rows[0]) == {"fold", "n_iter", "gradient_inf_norm", "convergence_warning",
                                                   "converged", "seconds"}


def test_benchmark_source_never_calls_prediction_or_accuracy():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "benchmark.py").read_text()
    names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    assert not names & {"predict", "predict_proba", "decision_function", "score", "balanced_accuracy",
                        "balanced_accuracy_exact", "fit_arm", "measure"}
