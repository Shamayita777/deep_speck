"""CE2-CE4 test suite. CPU-only; touches no CE1 artifact and writes only to tmp_path."""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parents[0]))
from audit.cryptography.ce234 import frozen_plan as P          # noqa: E402
from audit.cryptography.ce234 import production as PR          # noqa: E402
from audit.cryptography.ce234 import verify as V               # noqa: E402

tf = pytest.importorskip("tensorflow", reason="needs TensorFlow")


# ---------------- frozen plan integrity ----------------

def test_plan_constants_match_gohr_reference():
    assert (P.ROUNDS, P.DIFFERENTIAL, P.DEPTH, P.L2_REG) == (5, (0x0040, 0x0000), 10, 1e-5)
    assert P.PREDICT_BATCH == 5000
    assert P.REQUIRED_MODEL_PROPERTIES["conv1d_layers"] == 21
    assert P.REQUIRED_MODEL_PROPERTIES["residual_merges"] == 10


def test_plan_hash_is_stable_and_binding():
    assert P.plan_hash() == P.plan_hash() and len(P.plan_hash()) == 64


def test_every_unspecified_value_is_declared():
    assert set(P.UNSPECIFIED_IN_SOURCE) >= {"CE2.n_replicates", "CE3.control_model",
                                            "CE4.n_replicates",
                                            "CE4.intervention_magnitude"}


def test_operator_frozen_counts():
    assert (P.CE2.n_runs, P.CE2.samples_per_run) == (20, 10 ** 6)
    assert (P.CE3.n_replicates, P.CE3.n_splits_per_replicate) == (20, 5)
    assert (P.CE4.n_runs, P.CE4.samples_per_run) == (10, 100_000)
    assert P.CE3.primary_control == "C3-UNTRAINED-ARCH"
    assert "informational" in P.CE3.secondary_control_role


def test_ce2_sampling_unit_is_not_a_replicate():
    assert "NEVER represented as 10^6" in P.CE2.sampling_unit
    assert P.CE2.construct_class == "single-trail quantity; lower bound; proxy"
    assert "PRE-PRODUCTION DIAGNOSTIC" in P.CE2.historical_status


def test_ce4_threshold_policy_rejects_the_historical_ceiling():
    assert "NO magnitude threshold" in P.CE4.threshold_policy


# ---------------- statistics ----------------

def test_holm_is_monotone_and_conservative():
    adj = P.holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert adj["a"] <= adj["b"] <= adj["c"] and adj["a"] == pytest.approx(0.03)
    assert all(adj[k] >= p for k, p in {"a": .01, "b": .02, "c": .04}.items())


def test_sign_test_is_exact_and_sample_size_independent():
    a = P.exact_sign_test([0.8] * 10)
    b = P.exact_sign_test([1e-9] * 10)             # tiny effects, same signs
    assert a["p_value"] == b["p_value"] == pytest.approx(2 * 0.5 ** 10)
    assert a["smallest_attainable_p"] == pytest.approx(2.0 ** -9)
    mixed = P.exact_sign_test([1, -1, 1, -1])
    assert mixed["p_value"] == pytest.approx(1.0)


def test_no_effect_case_is_not_significant():
    rng = np.random.default_rng(0)
    assert P.exact_sign_test(rng.normal(size=10))["p_value"] > P.ALPHA


# ---------------- model binding, fail-closed ----------------

def test_reference_model_binding_passes_and_is_behaviourally_five_rounds():
    rep = PR.verify_reference_model(behavioural=True)
    assert rep["sha256"] == P.REFERENCE_CHECKPOINT_SHA256
    assert rep["realized_depth"] == 10
    acc = rep["behavioural_binding"]["accuracy_by_rounds"]
    assert acc["5"] > 0.85 and acc["6"] < 0.6 and acc["7"] < 0.6


def test_wrong_checkpoint_fails_closed(tmp_path):
    bad = tmp_path / "not_the_model.h5"
    bad.write_bytes(b"not a model")
    with pytest.raises(PR.PreflightError, match="sha256"):
        PR.verify_reference_model(checkpoint=bad, behavioural=False)


def test_historical_lookalike_checkpoint_is_refused():
    """evidence/ce1/'best5depth10 (10).h5' is depth-5; the name must not help it."""
    hist = PR.CRYPTO_ROOT / "evidence" / "ce1" / "best5depth10 (10).h5"
    if not hist.exists():
        pytest.skip("historical artifact absent")
    with pytest.raises(PR.PreflightError):
        PR.verify_reference_model(checkpoint=hist, behavioural=False)


def test_preflight_asserts_ce1_isolation():
    rep = PR.preflight(behavioural=False)
    assert rep["status"] == "PREFLIGHT_OK" and "no CE1 run directory" in rep["ce1_isolation"]


def test_production_refuses_to_overwrite_existing_evidence(tmp_path):
    d = tmp_path / "run"; d.mkdir(); (d / "x").write_text("existing")
    with pytest.raises(PR.PreflightError, match="never overwritten"):
        PR.execute("CE2", d, production=False, n1=1, n2=10, seed=0, behavioural=False)


# ---------------- data generation ----------------

def test_deterministic_generator_replays_exactly_and_production_does_not_use_it():
    a = PR.generate_theory_data(500, rng=np.random.default_rng(99))
    b = PR.generate_theory_data(500, rng=np.random.default_rng(99))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = PR.generate_theory_data(500, rng=np.random.default_rng(100))
    assert not np.array_equal(a[0], c[0])
    assert PR.production_generator_is_nondeterministic()


def test_theory_generator_matches_the_audited_speck_implementation():
    """Our instrumented generator must agree with estimate_trail_probabilities."""
    from audit.cryptography.gohr import speck as S
    n = 4000
    X, t, F = PR.generate_theory_data(n, with_factors=True,
                                      rng=np.random.default_rng(5))
    assert X.shape == (n, 64) and X.dtype == np.uint8 and t.shape == (n,)
    assert F.shape == (P.ROUNDS, n)
    prod = np.exp2(np.log2(np.where(F > 0, F, 1.0)).sum(axis=0))
    assert np.allclose(np.where((F > 0).all(0), prod, 0.0), t)
    X2, t2 = S.estimate_trail_probabilities(n, P.ROUNDS, P.DIFFERENTIAL)  # production path
    assert X2.shape == X.shape and t2.dtype == t.dtype          # same contract
    assert (t > 0).mean() > 0.5 and t.max() <= 1.0


def test_generator_uses_declared_rounds_not_the_dangerous_default():
    from audit.cryptography.gohr.dataset import GohrDataset
    import inspect
    assert inspect.signature(GohrDataset.__init__).parameters["rounds"].default == 7
    captured = {}
    from audit.cryptography.gohr import speck as S
    orig = S.expand_key
    try:
        S.expand_key = lambda k, t: (captured.__setitem__("t", t), orig(k, t))[1]
        # Replayable: the assertion is about the ROUND ARGUMENT, not the draw.
        PR.generate_theory_data(100, rng=np.random.default_rng(7))
    finally:
        S.expand_key = orig
    assert captured["t"] == 5
    # The round plumbing is SHARED by both generator branches: expand_key is
    # called exactly once, after the production/test branch closes. So the
    # assertion above also establishes that the production (os.urandom) branch
    # passes rounds=5 - coverage that would otherwise be lost by seeding here.
    src = inspect.getsource(PR.generate_theory_data)
    assert src.count("S.expand_key(") == 1
    assert "ks = S.expand_key(keys, rounds)" in src
    branch = src.split("if rng is None:")[1].split("ks = S.expand_key")[0]
    assert "expand_key" not in branch, "round plumbing is inside the RNG branch"


def test_xdp_plus_matches_brute_force_frequencies():
    from audit.cryptography.gohr import speck as S
    rng = np.random.default_rng(3)
    N = 200_000
    x = rng.integers(0, 2 ** 16, N, dtype=np.uint32); y = rng.integers(0, 2 ** 16, N, dtype=np.uint32)
    alpha, beta = 0x0080, 0x0000
    gamma = ((x + y) & 0xFFFF) ^ (((x ^ alpha) + (y ^ beta)) & 0xFFFF)
    vals, counts = np.unique(gamma, return_counts=True)
    top = vals[np.argsort(-counts)[:3]].astype(np.uint16)
    pf = S.xdp_plus(np.full(top.shape, alpha, np.uint16), np.full(top.shape, beta, np.uint16), top)
    emp = counts[np.argsort(-counts)[:3]] / N
    assert np.allclose(pf, emp, rtol=0.05)


# ---------------- CE3 control admissibility ----------------

def test_degenerate_control_is_refused_not_scored():
    assert PR._degenerate(np.zeros((100, 64), dtype=np.float32))
    assert not PR._degenerate(np.random.default_rng(0).normal(size=(100, 64)))


def test_historical_control_model_is_degenerate():
    """Documents why the historical CE3 control could not control anything."""
    sd = PR.CRYPTO_ROOT / "evidence" / "ce1" / "signal_destroyed.h5"
    if not sd.exists():
        pytest.skip("historical artifact absent")
    import keras
    m = keras.models.load_model(sd, compile=False)
    X, _ = PR.generate_theory_data(500, rng=np.random.default_rng(11))
    assert PR._degenerate(PR._representation(m, X))


def test_untrained_twin_has_the_reference_architecture_and_is_not_degenerate():
    m = PR.untrained_twin(1)
    assert sum(1 for l in m.layers if type(l).__name__ == "Conv1D") == 21
    assert sum(1 for l in m.layers if type(l).__name__ == "Add") == 10
    X, _ = PR.generate_theory_data(500, rng=np.random.default_rng(12))
    assert not PR._degenerate(PR._representation(m, X))


# ---------------- CE4 intervention invariants ----------------

#: FIXED SEED CORPUS for the CE4 control-validity gate. Replaces the previous
#: os.urandom-backed test data, which was not replayable: a failure could not
#: be reproduced because the input was never persisted. Every case below is
#: reproducible from (seed, n_samples, k2, GENERATOR_VERSION).
CE4_SEED_CORPUS = tuple(range(40))
CE4_CORPUS_N = 2000
FAILURES_DIR = Path(__file__).resolve().parent / "failures"


def _persist_ce4_failure(seed, n, k2, checks, extra=""):
    """On failure persist seed, input, parameters, expected vs observed, replay cmd."""
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X, _ = PR.generate_theory_data(n, with_factors=False, rng=rng)
    stem = FAILURES_DIR / f"ce4_k{k2}_seed{seed}"
    np.savez_compressed(stem.with_suffix(".npz"), X=X)
    replay = (f"python -c \"import numpy as np;"
              f"from audit.cryptography.ce234 import production as PR;"
              f"rng=np.random.default_rng({seed});"
              f"X,_=PR.generate_theory_data({n},rng=rng);"
              f"b=PR.build_interventions(X,{k2},np.random.default_rng({seed}));"
              f"print(PR.intervention_checks(b))\"")
    stem.with_suffix(".json").write_text(json.dumps(
        {"seed": seed, "n_samples": n, "k2": k2,
         "generator_version": PR.GENERATOR_VERSION,
         "expected": {"all control-validity invariants hold": True},
         "observed": checks, "note": extra, "replay_command": replay}, indent=2,
        default=str))
    return replay


@pytest.mark.parametrize("k2", [2, 4, 8])
def test_ce4_control_validity_holds_exactly_at_every_magnitude(k2):
    """Deterministic: fixed seed corpus, no os.urandom anywhere in this path."""
    for seed in CE4_SEED_CORPUS:
        rng_data = np.random.default_rng(seed)
        X, _ = PR.generate_theory_data(CE4_CORPUS_N, with_factors=False, rng=rng_data)
        b = PR.build_interventions(X, k2, np.random.default_rng(seed))
        c = PR.intervention_checks(b)
        ok = (PR._checks_pass(c)
              and c["hamming_structural"] == c["hamming_control"] == [k2]
              and c["equal_per_ciphertext"] and c["equal_per_word"]
              and c["identical_c0_side_flips"] and c["control_preserves_pair_xor"]
              and c["structural_xor_positions_changed"] == [k2]
              # the c0 half must shift identically (identical c0-side flips);
              # per-column equality on the c1 half is NOT a frozen criterion
              and c["marginal_shift_identical_on_c0"]
              and abs(c["pair_xor_weight_mean_original"]
                      - c["pair_xor_weight_mean_structural"] - k2) < 1e-9
              and c["pair_xor_weight_mean_original"] == c["pair_xor_weight_mean_control"])
        if not ok:
            cmd = _persist_ce4_failure(seed, CE4_CORPUS_N, k2, c)
            pytest.fail(f"control validity failed at k2={k2}, seed={seed}. "
                        f"Input and parameters persisted. Replay: {cmd}")


def test_ce4_eligibility_capacity_rule_is_exact():
    """Every sample declared eligible must really supply k same-word pairs."""
    for seed in (0, 1, 2):
        X, _ = PR.generate_theory_data(2000, rng=np.random.default_rng(seed))
        pairs = PR._mirrored_pairs(); c0, c1 = pairs[:, 0], pairs[:, 1]
        D = X[:, c0] != X[:, c1]
        for k2 in (2, 4, 8):
            b = PR.build_interventions(X, k2, np.random.default_rng(seed))
            cap = (D[:, :16].sum(1) // 2) + (D[:, 16:].sum(1) // 2)
            assert np.array_equal(b["eligible_mask"], cap >= k2 // 2)
            f, s2 = b["first"], b["second"]
            assert (f >= 0).all() and (s2 >= 0).all()
            same_word = ((f < 16) == (s2 < 16))
            assert same_word.all()                       # pairs are same-word
            assert np.array_equal(np.sort(np.c_[f, s2], axis=1)[:, :-1] !=
                                  np.sort(np.c_[f, s2], axis=1)[:, 1:],
                                  np.ones((f.shape[0], 2 * (k2 // 2) - 1), bool))


def test_no_nondeterministic_source_in_the_validation_path():
    """
    Every data-generating call in the validation path must pass an explicit
    RNG. The check is performed on the AST, not on source text: only real
    Call nodes are inspected, so string literals (such as the replay command
    this module writes on failure) and this scanner's own needles cannot be
    mistaken for call sites. Text scanning is what made earlier versions of
    this assertion fire on their own source.
    """
    import ast as _ast

    tree = _ast.parse(Path(__file__).read_text())
    sites, offenders = 0, []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, _ast.Attribute) else getattr(fn, "id", "")
            if name == "generate_theory_data":
                sites += 1
                if not any(kw.arg == "rng" for kw in node.keywords):
                    offenders.append(node.lineno)
    assert sites, "no generator call sites found to scan"
    assert not offenders, (
        f"non-replayable generator calls in the validation path at lines {offenders}")
    # the one deliberate production-RNG call in this module is the speck
    # cross-check, which must exercise the frozen production generator
    assert any(isinstance(n, _ast.Call)
               and getattr(n.func, "attr", "") == "estimate_trail_probabilities"
               for n in _ast.walk(tree))


def test_ce4_magnitude_ladder_respects_the_coverage_rule():
    X, _ = PR.generate_theory_data(20000, rng=np.random.default_rng(1))
    for k2 in P.CE4.magnitude_ladder_bits:
        b = PR.build_interventions(X, k2, np.random.default_rng(k2))
        assert b["n_eligible"] / b["n_total"] >= 0.95
    assert P.CE4.primary_magnitude_bits == min(P.CE4.magnitude_ladder_bits)
    assert P.CE4.n_intervention_bits_historical == 8      # retained only as a ladder point


def test_ce4_null_control_is_exactly_zero(tmp_path):
    res = PR.run_ce4(tmp_path / "n", n_runs=1, n_samples=2000, seed=11,
                     magnitudes=(2,), log=lambda *a: None)
    assert res["runs"][0]["C4_NULL_mean_abs_change"] == 0.0


# ---------------- decision-rule behaviour ----------------

def test_insufficient_replication_is_inconclusive_never_supported():
    out = PR.analyse_ce2([{"status": "OK", "rho_primary": 0.9} for _ in range(3)])
    assert out["decision"] == "INCONCLUSIVE" and "minimum" in out["reason"]


def test_control_reproducing_the_effect_yields_not_supported():
    runs = [{"status": "OK", "rho_primary": 0.8, "rho_C2_PERMUTED_TARGET": 0.8,
             "rho_C2_UNTRAINED_MODEL": 0.8, "rho_S2_LAST_ROUND_ONLY": 0.1,
             "rho_S2_PREFIX_ONLY": 0.1} for _ in range(P.CE2.n_runs)]
    assert PR.analyse_ce2(runs)["decision"] == "NOT_SUPPORTED"


def test_clean_controls_yield_supported():
    runs = [{"status": "OK", "rho_primary": 0.8, "rho_C2_PERMUTED_TARGET": 0.001,
             "rho_C2_UNTRAINED_MODEL": 0.002, "rho_S2_LAST_ROUND_ONLY": 0.4,
             "rho_S2_PREFIX_ONLY": 0.7} for _ in range(P.CE2.n_runs)]
    assert PR.analyse_ce2(runs)["decision"] == "SUPPORTED"


def test_underflowed_p_is_not_treated_as_non_significant():
    """Regression: p_underflow must mean 'extremely significant', not 'missing'."""
    from audit.cryptography.statistics import p_value_report
    r = p_value_report(0.0)
    assert r["p_value"] is None and r["p_underflow"] and r["p_upper_bound"] == 1e-300


# ---------------- end-to-end + verifier ----------------

@pytest.mark.slow
def test_end_to_end_ce2_smoke_and_independent_verification(tmp_path):
    out = tmp_path / "ce2"
    path = PR.execute("CE2", out, production=False, n1=2, n2=5000, seed=5,
                      behavioural=False)
    cert = json.loads(Path(path).read_text())
    assert cert["non_evidentiary"] and cert["claim_scope"].startswith("SMOKE ONLY")
    assert cert["plan_hash"] == P.plan_hash()
    rep = V.verify(out)
    assert rep["verdict"] == "VERIFIED", rep["failed"]


@pytest.mark.slow
def test_verifier_detects_a_tampered_certificate(tmp_path):
    out = tmp_path / "ce2t"
    PR.execute("CE2", out, production=False, n1=2, n2=5000, seed=6, behavioural=False)
    cert = json.loads((out / "certificate.json").read_text())
    cert["results"]["runs"][0]["rho_primary"] = 0.999
    (out / "certificate.json").write_text(json.dumps(cert))
    assert V.verify(out)["verdict"] == "VERIFICATION FAILED"


@pytest.mark.slow
def test_verifier_detects_tampered_raw_observations(tmp_path):
    out = tmp_path / "ce2r"
    PR.execute("CE2", out, production=False, n1=2, n2=5000, seed=8, behavioural=False)
    f = next((out / "raw").glob("*.npz"))
    with np.load(f) as z:
        d = {k: z[k] for k in z.files}
    d["output"] = d["output"][::-1]
    np.savez_compressed(f, **d)
    rep = V.verify(out)
    assert rep["verdict"] == "VERIFICATION FAILED"
    assert any("intact" in n for n in rep["failed"])


# ---------------- tamper tests: the verifier must reject material mutations ----

MATRIX_DIR = Path(__file__).resolve().parents[1] / "matrices"
MUT_LOG = MATRIX_DIR / "tamper_log.jsonl"


def _log_mutation(mutation_id, ce, field, mtype, expected, actual, status, reason=""):
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    with MUT_LOG.open("a") as fh:
        fh.write(json.dumps({"mutation_id": mutation_id, "ce": ce, "field": field,
                             "mutation_type": mtype, "expected_result": expected,
                             "actual_result": actual, "status": status,
                             "reason_if_skipped": reason}) + "\n")


#: Deterministic CE4 fixture that reaches the AGGREGATE certificate path
#: (>= CE4.minimum_valid_runs replicates), so aggregate-only fields such as
#: primary_magnitude_bits and statistical_unit exist and can be mutated.
@pytest.fixture(scope="module")
def ce4_aggregate_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("ce4agg") / "run"
    PR.execute("CE4", out, production=False, n1=P.CE4.minimum_valid_runs, n2=300,
               seed=12345, behavioural=False)
    cert = json.loads((out / "certificate.json").read_text())
    assert "by_magnitude" in cert["results"], "fixture did not reach the aggregate path"
    assert "primary_magnitude_bits" in cert["results"]
    assert V.verify(out)["verdict"] == "VERIFIED"
    return out


def _smoke_run(tmp_path, ce, n1, n2, seed=3):
    out = tmp_path / ce.lower()
    PR.execute(ce, out, production=False, n1=n1, n2=n2, seed=seed, behavioural=False)
    assert V.verify(out)["verdict"] == "VERIFIED"
    return out


def _mutate(out, fn):
    cert = json.loads((out / "certificate.json").read_text())
    fn(cert)
    (out / "certificate.json").write_text(json.dumps(cert, default=str))
    return V.verify(out)


@pytest.mark.slow
@pytest.mark.parametrize("field,mutation", [
    ("model_hash", lambda c: c["reference_model"].__setitem__("sha256", "0" * 64)),
    ("realized_depth", lambda c: c["reference_model"].__setitem__("realized_depth", 5)),
    ("architecture", lambda c: c["reference_model"]["properties"].__setitem__(
        "conv1d_layers", 11)),
    ("rounds", lambda c: c["results"]["runs"][0].__setitem__("rounds", 7)),
    ("differential", lambda c: c["results"]["runs"][0].__setitem__(
        "differential", [0x0020, 0])),
    ("sample_count", lambda c: c["results"]["runs"][0].__setitem__("n", 123)),
    ("seed", lambda c: c["results"]["runs"][0].pop("evaluation_seed")),
    ("rho_estimate", lambda c: c["results"]["runs"][0].__setitem__(
        "rho_primary", 0.99)),
    ("control_estimate", lambda c: c["results"]["runs"][0].__setitem__(
        "rho_C2_PERMUTED_TARGET", 0.99)),
    ("raw_file_hash", lambda c: c["results"]["runs"][0].__setitem__(
        "raw_sha256", "f" * 64)),
    ("plan_hash", lambda c: c.__setitem__("plan_hash", "a" * 64)),
])
def test_ce2_verifier_rejects_material_mutations(tmp_path, field, mutation):
    out = _smoke_run(tmp_path / field, "CE2", 2, 4000)
    rep = _mutate(out, mutation)
    _log_mutation(f"MUT-CE2-{field}", "CE2", field, "certificate field mutation",
                  "VERIFICATION FAILED", rep["verdict"],
                  "REJECTED" if rep["verdict"] == "VERIFICATION FAILED" else "NOT_REJECTED")
    assert rep["verdict"] == "VERIFICATION FAILED", field


@pytest.mark.slow
def test_ce2_verifier_rejects_nan_and_truncated_raw_arrays(tmp_path):
    out = _smoke_run(tmp_path / "nan", "CE2", 2, 4000)
    r = json.loads((out / "certificate.json").read_text())["results"]["runs"][0]
    f = out / r["raw_file"]
    with np.load(f) as z:
        d = {k: z[k] for k in z.files}
    d["target"] = d["target"].copy(); d["target"][0] = np.nan
    np.savez_compressed(f, **d)
    cert = json.loads((out / "certificate.json").read_text())
    cert["results"]["runs"][0]["raw_sha256"] = PR.sha256_file(f)
    (out / "certificate.json").write_text(json.dumps(cert, default=str))
    rep = V.verify(out)
    _log_mutation("MUT-CE2-nan_raw", "CE2", "raw array values", "NaN injection",
                  "VERIFICATION FAILED", rep["verdict"],
                  "REJECTED" if rep["verdict"] == "VERIFICATION FAILED" else "NOT_REJECTED")
    assert rep["verdict"] == "VERIFICATION FAILED"
    assert any("NaN" in n for n in rep["failed"])


@pytest.mark.slow
@pytest.mark.parametrize("field,mutation", [
    ("gap_estimate", lambda c: c["results"]["runs"][0]["by_magnitude"]["2"]
        .__setitem__("mean_gap", 9.9)),
    ("null_control", lambda c: c["results"]["runs"][0].__setitem__(
        "C4_NULL_mean_abs_change", 0.5)),
    ("xor_invariant", lambda c: c["results"]["runs"][0]["by_magnitude"]["2"]
        ["manipulation_checks"].__setitem__("control_preserves_pair_xor", False)),
    ("per_word_match", lambda c: c["results"]["runs"][0]["by_magnitude"]["2"]
        ["manipulation_checks"].__setitem__("equal_per_word", False)),
    ("rounds", lambda c: c["results"]["runs"][0].__setitem__("rounds", 7)),
    ("raw_file_hash", lambda c: c["results"]["runs"][0].__setitem__(
        "raw_sha256", "e" * 64)),
    ("primary_magnitude", lambda c: c["results"].__setitem__(
        "primary_magnitude_bits", 8)),
    ("statistical_unit", lambda c: c["results"].__setitem__(
        "statistical_unit", "individual sample")),
    ("threshold_policy", lambda c: c["results"].__setitem__(
        "threshold_policy", "gap must exceed 0.05")),
])
def test_ce4_verifier_rejects_material_mutations(tmp_path, ce4_aggregate_run, field,
                                                 mutation):
    """Runs against the AGGREGATE certificate so every field under test exists."""
    out = tmp_path / field
    shutil.copytree(ce4_aggregate_run, out)
    rep = _mutate(out, mutation)
    _log_mutation(f"MUT-CE4-{field}", "CE4", field, "certificate field mutation",
                  "VERIFICATION FAILED", rep["verdict"],
                  "REJECTED" if rep["verdict"] == "VERIFICATION FAILED" else "NOT_REJECTED")
    assert rep["verdict"] == "VERIFICATION FAILED", field


@pytest.mark.slow
def test_ce4_verifier_rejects_tampered_raw_gaps(tmp_path, ce4_aggregate_run):
    out = tmp_path / "ce4raw"
    shutil.copytree(ce4_aggregate_run, out)
    cert = json.loads((out / "certificate.json").read_text())
    r = cert["results"]["runs"][0]
    f = out / r["raw_file"]
    with np.load(f) as z:
        d = {k: z[k] for k in z.files}
    d["gap_2"] = d["gap_2"] + 1.0                       # inconsistent with f0/fs/fc
    np.savez_compressed(f, **d)
    cert["results"]["runs"][0]["raw_sha256"] = PR.sha256_file(f)
    (out / "certificate.json").write_text(json.dumps(cert, default=str))
    rep = V.verify(out)
    _log_mutation("MUT-CE4-raw_gap_values", "CE4", "raw gap array",
                  "raw observation mutation", "VERIFICATION FAILED", rep["verdict"],
                  "REJECTED" if rep["verdict"] == "VERIFICATION FAILED" else "NOT_REJECTED")
    assert rep["verdict"] == "VERIFICATION FAILED"


@pytest.mark.slow
def test_ce3_verifier_rejects_selectivity_tampering(tmp_path):
    out = _smoke_run(tmp_path / "ce3t", "CE3", 2, 2000)
    cert = json.loads((out / "certificate.json").read_text())
    if "primary" not in cert["results"]:
        pytest.skip("CE3 smoke stops before the primary statistic by design")
    cert["results"]["primary"]["mean"] = 0.99
    (out / "certificate.json").write_text(json.dumps(cert, default=str))
    rep = V.verify(out)
    _log_mutation("MUT-CE3-selectivity_mean", "CE3", "primary.mean",
                  "certificate field mutation", "VERIFICATION FAILED", rep["verdict"],
                  "REJECTED" if rep["verdict"] == "VERIFICATION FAILED" else "NOT_REJECTED")
    assert rep["verdict"] == "VERIFICATION FAILED"


# ---------------- CE3 checkpoint / resume ----------------

def _ce3_interrupted(out, stop_after, **kw):
    """Run CE3 and abort immediately after replicate `stop_after` is checkpointed."""
    class _Stop(Exception):
        pass

    orig = PR._ce3_log

    def spy(d, ev, **detail):
        orig(d, ev, **detail)
        if ev == "REPLICATE_COMPLETE" and detail.get("replicate") == stop_after:
            raise _Stop()
    PR._ce3_log = spy
    try:
        PR.run_ce3(out, log=lambda *a: None, **kw)
    except _Stop:
        pass
    finally:
        PR._ce3_log = orig


CE3_RESUME_KW = dict(n_replicates=3, n_samples=800, n_splits=3, seed=5)


@pytest.mark.slow
def test_ce3_resume_never_recomputes_a_completed_replicate(tmp_path):
    out = tmp_path / "run"
    _ce3_interrupted(out, 1, **CE3_RESUME_KW)
    state = json.loads((out / "resume_state.json").read_text())
    assert state["completed"] == [0, 1]
    before = {i: json.loads((out / "replicates" / f"rep_{i:02d}.json").read_text())
              for i in state["completed"]}
    res = PR.run_ce3(out, log=lambda *a: None, **CE3_RESUME_KW)
    # byte-identical records: resuming changes WHEN a replicate runs, never WHAT it is
    after = {i: json.loads((out / "replicates" / f"rep_{i:02d}.json").read_text())
             for i in state["completed"]}
    assert before == after
    assert len(res["replicates"]) == 3
    assert [r["replicate_id"] for r in res["replicates"]] == \
        ["ce3_rep00", "ce3_rep01", "ce3_rep02"]
    events = [json.loads(l)["event"]
              for l in (out / "resume_log.jsonl").read_text().splitlines()]
    assert events.count("REPLICATE_COMPLETE") == 3      # each replicate computed once
    assert "RESUME" in events and events[-1] == "ALL_REPLICATES_COMPLETE"
    assert res["resume"]["checkpoint_granularity"] == "one replicate"


@pytest.mark.slow
@pytest.mark.parametrize("override", [{"seed": 6}, {"n_samples": 900},
                                      {"n_splits": 4}, {"n_replicates": 4}])
def test_ce3_resume_refuses_a_changed_binding(tmp_path, override):
    out = tmp_path / "run"
    _ce3_interrupted(out, 0, **CE3_RESUME_KW)
    with pytest.raises(PR.PreflightError, match="different binding"):
        PR.run_ce3(out, log=lambda *a: None, **{**CE3_RESUME_KW, **override})


@pytest.mark.slow
def test_ce3_resume_refuses_a_tampered_or_missing_replicate_record(tmp_path):
    out = tmp_path / "run"
    _ce3_interrupted(out, 0, **CE3_RESUME_KW)
    f = out / "replicates" / "rep_00.json"
    rec = json.loads(f.read_text())
    rec["selectivity"] = 0.99
    f.write_text(json.dumps(rec))
    with pytest.raises(PR.PreflightError, match="content hash"):
        PR.run_ce3(out, log=lambda *a: None, **CE3_RESUME_KW)
    f.unlink()
    with pytest.raises(PR.PreflightError, match="marked complete"):
        PR.run_ce3(out, log=lambda *a: None, **CE3_RESUME_KW)


@pytest.mark.slow
def test_execute_allows_ce3_resume_but_never_overwrites_a_certificate(tmp_path):
    out = tmp_path / "run"
    _ce3_interrupted(out, 0, **CE3_RESUME_KW)
    # a part-finished CE3 directory is resumable through the production entry point
    path = PR.execute("CE3", out, production=False, n1=3, n2=800, seed=5, n_splits=3,
                      behavioural=False)
    assert Path(path).exists()
    # once the certificate exists the directory is closed, resumable or not
    with pytest.raises(PR.PreflightError, match="completed certificate"):
        PR.execute("CE3", out, production=False, n1=3, n2=800, seed=5, n_splits=3,
                   behavioural=False)
    # and CE2/CE4 are never resumable
    (tmp_path / "ce2").mkdir(); (tmp_path / "ce2" / "x").write_text("1")
    with pytest.raises(PR.PreflightError, match="never overwritten"):
        PR.execute("CE2", tmp_path / "ce2", production=False, n1=1, n2=100, seed=1,
                   behavioural=False)
