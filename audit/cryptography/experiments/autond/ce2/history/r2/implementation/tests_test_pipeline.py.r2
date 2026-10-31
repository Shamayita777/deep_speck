"""AutoND CE2 with TensorFlow: upstream isolation, model binding, runner faults, verifier tampering."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from audit.cryptography.core import evidence as EV
from audit.cryptography.experiments.autond.ce2 import adapter as A
from audit.cryptography.experiments.autond.ce2 import analysis as S
from audit.cryptography.experiments.autond.ce2 import design as FD
from audit.cryptography.experiments.autond.ce2 import runner as R
from audit.cryptography.experiments.autond.ce2 import verify as V
from audit.cryptography.targets.autond import model_binding as MB
from audit.cryptography.targets.autond import upstream as UP

CE2 = Path(__file__).resolve().parents[1]
CRYPTO = CE2.parents[2]
REPO = CRYPTO.parents[1]
SOURCES = {p.stem: p for p in (CE2 / "adapter.py", CE2 / "analysis.py", CE2 / "design.py", CE2 / "power.py",
                               CE2 / "runner.py", CE2 / "theory.py", CE2 / "verify.py")}


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            out.add(mod)
            out |= {f"{mod}.{a.name}" for a in node.names}
    return out


# ============================================================ isolation


def test_sources_do_not_import_gohr_ce1_ce234_or_legacy_framework():
    banned = ("gohr", "ce234", "adapters", "frozen_design", "experiments.ce1", "experiments.ce2", "audit_config",
              "provenance", "certificate", "test.ce2", "engine", "results")
    for name, path in SOURCES.items():
        for imp in _imports(path):
            leaf = imp.lstrip(".")
            for b in banned:
                assert not (leaf == b or leaf.startswith(b + ".") or f".{b}." in f".{leaf}." or leaf.endswith("." + b)), \
                    f"{name} imports {imp}"


def test_verifier_imports_nothing_from_the_producer():
    imps = _imports(SOURCES["verify"])
    assert not any(x in i for i in imps for x in ("runner", "analysis", "theory", "adapter", "design", "power",
                                                    "core", "targets"))


def test_no_ce1_evidence_path_is_referenced():
    for path in SOURCES.values():
        assert "autond_ce1/evidence" not in path.read_text()


@pytest.fixture(scope="module")
def up():
    return UP.load()


@pytest.fixture(scope="module")
def published(up):
    return MB.load_published(up, FD.DESIGN["reference"]["observed_architecture"])


def test_upstream_import_is_isolated_and_identity_stable(up):
    assert str(up.root) not in sys.path
    for name in UP.TOP_LEVEL:
        assert name not in sys.modules or not UP._inside(sys.modules[name], up.root)
    assert UP.load() is up and UP.load().main is up.main
    assert all(up.root in Path(m.__file__).resolve().parents for m in (up.main, up.dbitnet, up.cipher))


def test_upstream_import_does_not_clobber_or_use_a_foreign_module(tmp_path):
    copy = tmp_path / "up"
    shutil.copytree(UP.DEFAULT_ROOT, copy, ignore=shutil.ignore_patterns("__pycache__"))
    foreign = types.ModuleType("main")
    foreign.__file__ = str(tmp_path / "elsewhere" / "main.py")
    sys.modules["main"] = foreign
    try:
        u = UP.load(copy)
        assert sys.modules["main"] is foreign and u.main is not foreign
        assert copy.resolve() in Path(u.main.__file__).resolve().parents
    finally:
        del sys.modules["main"]
        UP._CACHE.pop(copy.resolve(), None)


def test_upstream_verification_detects_tampering(tmp_path):
    copy = tmp_path / "up"
    shutil.copytree(UP.DEFAULT_ROOT, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert UP.verify(copy)["all_verified"]
    (copy / "dbitnet.py").write_text((copy / "dbitnet.py").read_text() + "\n# altered\n")
    assert not UP.verify(copy)["all_verified"]
    with pytest.raises(UP.UpstreamError):
        UP.load(copy)


def test_upstream_import_writes_no_bytecode(tmp_path):
    copy = tmp_path / "up"
    shutil.copytree(UP.DEFAULT_ROOT, copy, ignore=shutil.ignore_patterns("__pycache__"))
    code = ("import sys; sys.dont_write_bytecode = True\n"
            "from audit.cryptography.targets.autond import upstream as UP\n"
            "sys.dont_write_bytecode = False\n"
            f"u = UP.load({str(copy)!r}); u.delta_bits()\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO,
                   env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})
    assert not list(copy.rglob("__pycache__"))


def test_upstream_runtime_imports_are_only_tensorflow():
    for f in ("dbitnet.py", "main.py", "ciphers/speck3264.py"):
        tree = ast.parse((UP.DEFAULT_ROOT / f).read_text())
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                    assert mod.split(".")[0] == "tensorflow", (f, fn.name, mod)


# ========================================================= model / data


def test_published_model_binding(published):
    _, b = published
    assert b["sha256"] == FD.PUBLISHED_CHECKPOINT_SHA256
    assert b["strict_load"]["all_arrays_bit_identical"] and b["strict_load"]["weight_arrays_in_file"] == 68
    arch = b["architecture"]
    assert (arch["layers"], arch["weight_bearing_layers"]) == (34, 23)
    assert (arch["params_total"], arch["params_trainable"], arch["params_non_trainable"]) == (298113, 296065, 2048)
    assert arch["input_shape"] == [None, 64, 1] and arch["output_shape"] == [None, 1]


def test_weight_hash_ignores_variable_names_but_not_values(up, published):
    m = published[0]
    again = MB.load_published(up, FD.DESIGN["reference"]["observed_architecture"])[0]
    assert MB.weights_sha256(m) == MB.weights_sha256(again)
    w = again.weights[0]
    w.assign(w.numpy() + 1e-7)
    assert MB.weights_sha256(m) != MB.weights_sha256(again)
    assert V.weights_hash(m) == MB.weights_sha256(m) and V.strict_matches_hdf5(m, up.checkpoint)


def test_genuine_pairs_gates_and_upstream_encoding(up):
    g = A.generate_genuine_pairs(up, 512)
    assert g["gates"]["E"]["passed"] and g["gates"]["F"]["passed"] and all(g["gates"]["F"]["checks"].values())
    assert A.delta_words(up) == (0x0040, 0x0000)
    assert R.upstream_encoding_check(up, n=512)["genuine_rows_identical"]
    a = g["arrays"]
    assert len({tuple(k) for k in a["key_words"].tolist()}) > 500          # per-pair keys, not one global key
    X = A.model_input(up, a["c0"], a["c1"])
    assert A.content_sha256({"X": X}) == V.content_hash({"X": np.hstack([V._words_to_bits(a["c0"]),
                                                                       V._words_to_bits(a["c1"])])})


def test_trajectory_mismatch_is_detected(up):
    a = A.generate_genuine_pairs(up, 64)["arrays"]
    a["c0"] = a["c0"].copy()
    a["c0"][0, 0] ^= 1
    _, gates = A.trajectory_and_weights(up, a, 8)
    assert not gates["F"]["passed"]


def _context(up, published, n=256, twin_scores=None):
    a = A.generate_genuine_pairs(up, n)["arrays"]
    ctx = A.RunContext(data=A.FrozenRunData.create(a), published_model=published[0],
                       twin_model=MB.build_untrained_twin(up, 1),
                       permutation=np.random.Generator(np.random.PCG64(0)).permutation(n),
                       model_input=A.model_input(up, a["c0"], a["c1"]))
    if twin_scores is not None:
        ctx.scores["twin"] = twin_scores
    return ctx


def test_measurement_uses_one_frozen_dataset_and_is_two_sided(up, published):
    from audit.cryptography.core.hypothesis import HypothesisDirection
    from audit.cryptography.core.theory_consistency import TheoryConsistencyMeasurement
    ctx = _context(up, published)
    m = TheoryConsistencyMeasurement(HypothesisDirection.TWO_SIDED, S.spearman)
    res = {c: m.run(A.AutoNDCE2Adapter(ctx, c)) for c in S.CONDITIONS}
    assert {x["dataset_sha256"] for x in ctx.calls} == {ctx.data.sha256} and len(ctx.calls) == 12
    assert res["primary"].statistic == S.spearman(ctx.scores["published"], ctx.data["target"].astype(float))
    assert all(r.hypothesis is HypothesisDirection.TWO_SIDED for r in res.values())
    with pytest.raises(ValueError):
        ctx.data["target"][0] = 0


def test_undefined_twin_is_reported_undefined_not_zero(up, published):
    from audit.cryptography.core.hypothesis import HypothesisDirection
    from audit.cryptography.core.theory_consistency import TheoryConsistencyMeasurement
    ctx = _context(up, published, twin_scores=np.full(256, 0.5, dtype=np.float32))
    r = TheoryConsistencyMeasurement(HypothesisDirection.TWO_SIDED, S.spearman).run(
        A.AutoNDCE2Adapter(ctx, "untrained_twin"))
    assert not r.defined and r.statistic is None and r.reason == "constant model output"
    ctx.score("published")
    m = R.measure(ctx, {k: ctx.data[k] for k in ctx.data.arrays})
    assert m["status"] == "EXCLUDED" and "untrained_twin: constant model output" in m["exclusion_reason"]


def test_adapter_exposes_only_its_capability(up, published):
    from audit.cryptography.core.capabilities import (InterventionCapability, RepresentationCapability,
                                                      SignalDestructionCapability, TheoryConsistencyCapability)
    ad = A.AutoNDCE2Adapter(_context(up, published, n=32), "primary")
    assert isinstance(ad, TheoryConsistencyCapability)
    for cap in (SignalDestructionCapability, RepresentationCapability, InterventionCapability):
        assert not isinstance(ad, cap)


# ============================================================= pipeline


@pytest.fixture()
def tiny(monkeypatch):
    for k, v in {"n_runs": 2, "min_valid_runs": 2, "pairs_per_run": 512, "behavioural_n": 2000,
                 "gate_c_word_sizes": [4], "gate_d_word_sizes": [4], "word16_transitions": 0,
                 "preflight_corpus_pairs": 256, "bootstrap_resamples": 100}.items():
        monkeypatch.setitem(FD.SMOKE, k, v)


@pytest.fixture()
def initialised(tmp_path, tiny):
    out = tmp_path / "autond_ce2_smoke_init"
    man = R.initialise(out, "smoke")
    assert man["status"] == "INITIALISED" and man["hypothesis_direction"] == "TWO_SIDED"
    return out


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for k, v in {"n_runs": 2, "min_valid_runs": 2, "pairs_per_run": 512, "behavioural_n": 2000,
                 "gate_c_word_sizes": [4], "gate_d_word_sizes": [4], "word16_transitions": 0,
                 "preflight_corpus_pairs": 256, "bootstrap_resamples": 100}.items():
        mp.setitem(FD.SMOKE, k, v)
    out = tmp_path_factory.mktemp("ce2") / "autond_ce2_smoke_done"
    R.initialise(out, "smoke")
    recs = R.run_all(out)
    assert [r["state"] for r in recs] == ["VALID", "VALID"]
    R.finalize(out)
    yield out
    mp.undo()


def test_smoke_pipeline_verifies_exactly_from_recomputed_scores(finished):
    cert = json.loads((finished / "certificate.json").read_text())
    assert cert["schema"] == FD.CERTIFICATE_SCHEMA and not cert["evidentiary"]
    assert cert["status"]["independent_verification"] == "NOT_RUN"
    rep = V.verify(finished)
    assert rep["passed"], rep["failures"]
    assert rep["inference_from"] == "recomputed scores" and rep["numerical_status"] == "EXACT"
    names = {c["check"] for c in rep["checks"]}
    for want in ("run_00:model_input_sha256", "run_00:published_weights_sha256", "run_00:twin_weights_sha256",
                 "journal_replay_state_machine_and_commits", "record_caches_equal_journal_replay",
                 "preflight_gate_CD_word4_independently_recomputed", "preflight_gate_CD_word6_independently_recomputed",
                 "decision_recomputed_from_recomputed_scores", "numerical_reproducibility_policy"):
        assert want in names, want
    with pytest.raises(R.RunnerError):
        R.finalize(finished)
    before = (finished / "runs" / "run_00" / "record.json").read_bytes()
    assert R.run_all(finished)[0]["state"] == "VALID"
    assert (finished / "runs" / "run_00" / "record.json").read_bytes() == before


@pytest.mark.parametrize("point", ["before_write", "during_write", "after_write", "after_hash", "after_promote",
                                   "after_commit", "during_quarantine"])
def test_runner_recovers_from_interruption_at_every_boundary(initialised, point):
    out = initialised
    if point == "during_quarantine":
        EV.arm_fault("after_promote")
        with pytest.raises(EV.InjectedFault):
            R.run_all(out, only=0)
    EV.arm_fault(point)
    try:
        with pytest.raises(EV.InjectedFault):
            R.run_all(out, only=0)
    finally:
        EV.disarm_faults()
    done = R.run_all(out)
    assert [r["state"] for r in done] == ["VALID", "VALID"]
    R.finalize(out)
    rep = V.verify(out, recompute_models=False)
    assert rep["passed"], rep["failures"]
    cert = json.loads((out / "certificate.json").read_text())
    if point in ("during_write", "after_write", "after_hash", "after_promote", "during_quarantine"):
        assert cert["quarantined_artifacts"], point
        for q in cert["quarantined_artifacts"]:
            assert (out / q["destination"]).exists() or point == "during_quarantine"


def test_interruption_during_scoring_resumes_from_committed_data(initialised, monkeypatch):
    out = initialised
    real = MB.predict

    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(MB, "predict", interrupted)
    with pytest.raises(KeyboardInterrupt):
        R.run_all(out, only=0)
    rec = R.store_for(out).record("run_00")
    assert rec["state"] == "DATA_VERIFIED"
    monkeypatch.setattr(MB, "predict", real)
    done = R.run_all(out, only=0)[0]
    assert done["state"] == "VALID" and done["data_sha256"] == rec["data_sha256"]


def test_committed_artifact_damage_fails_the_run_and_is_never_regenerated(initialised):
    out = initialised
    EV.arm_fault("after_commit")
    with pytest.raises(EV.InjectedFault):
        R.run_all(out, only=0)
    data = out / "runs" / "run_00" / "data.npz"
    data.write_bytes(data.read_bytes()[:-10])
    rec = R.run_all(out, only=0)[0]
    assert rec["state"] == "FAILED" and "InconsistentEvidence" in rec["failure"]


def test_interrupted_initialisation_is_abandoned(tmp_path, tiny, monkeypatch):
    out = tmp_path / "autond_ce2_smoke_abandoned"

    def boom(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(R, "behavioural_validation", boom)
    with pytest.raises(KeyboardInterrupt):
        R.initialise(out, "smoke")
    with pytest.raises(R.RunnerError, match="abandoned"):
        R.run_all(out)
    with pytest.raises(R.RunnerError, match="not empty"):
        R.initialise(out, "smoke")


def test_output_directory_rules(tmp_path):
    for path, mode in ((tmp_path / "not_smoke_named", "smoke"), (tmp_path / "autond_ce2_smoke_x", "production"),
                       (tmp_path / "autond_ce1" / "evidence" / "x", "production"), (tmp_path / "x", "other")):
        with pytest.raises(R.RunnerError):
            R._check_out_dir(path, mode)


def test_production_initialisation_refused_while_pending(tmp_path):
    with pytest.raises(R.RunnerError, match="production refused"):
        R.initialise(tmp_path / "prod", "production")
    assert not (tmp_path / "prod").exists()


# ============================================================ tampering


@pytest.fixture()
def copy_of(finished, tmp_path):
    dst = tmp_path / finished.name
    shutil.copytree(finished, dst)
    for v in dst.glob("verification_*"):
        v.unlink()
    return dst


def _rechain(out: Path, edit) -> None:
    """Rewrite journal entries with `edit(entry)` and recompute every prev hash (a forger's step)."""
    lines = (out / "journal.jsonl").read_bytes().splitlines()
    prev, new = EV.GENESIS, []
    for ln in lines:
        e = json.loads(ln)
        edit(e)
        e["prev_sha256"] = prev
        b = EV.canonical(e)
        new.append(b)
        prev = hashlib.sha256(b).hexdigest()
    (out / "journal.jsonl").write_bytes(b"".join(b + b"\n" for b in new))


def _forge_scores(out: Path, transform) -> None:
    """Replace run_00's stored scores and make every hash, record, measurement and the certificate
    consistent with the forgery. Only independent score recomputation can detect it."""
    rd = out / "runs" / "run_00"
    O = V.npz(rd / "outputs.npz")
    D = V.npz(rd / "data.npz")
    O["score_published"] = transform(O["score_published"]).astype(np.float32)
    (rd / "outputs.npz").write_bytes(R.npz_bytes(O))
    t = D["target"].astype(float)
    rho = {"primary": S.spearman(O["score_published"], t),
           "permuted_target": S.spearman(O["score_published"], t[O["permutation"]]),
           "untrained_twin": S.spearman(O["score_twin"], t)}
    m = json.loads((rd / "measurement.json").read_text())
    m["rho"] = rho
    (rd / "measurement.json").write_bytes((json.dumps(m, indent=1, sort_keys=True) + "\n").encode())
    sha_o, sha_m = EV.sha256_file(rd / "outputs.npz"), EV.sha256_file(rd / "measurement.json")

    def edit(e):
        if e.get("event") == "run_transition" and e.get("run_id") == "run_00":
            if e["to"] == "CONTROLS_EVALUATED":
                e["fields"]["outputs_sha256"] = sha_o
                e["fields"]["outputs_content_sha256"] = A.content_sha256(O)
            elif e["to"] == "EVIDENCE_WRITTEN":
                e["fields"]["measurement_sha256"] = sha_m
            elif e["to"] == "VALID":
                e["fields"]["rho"] = rho
    _rechain(out, edit)
    store = R.store_for(out)
    (rd / "record.json").write_text(json.dumps(store.record_from_journal("run_00"), indent=1, sort_keys=True) + "\n")
    (out / "certificate.json").unlink()
    lines = (out / "journal.jsonl").read_bytes().splitlines()
    (out / "journal.jsonl").write_bytes(b"".join(l + b"\n" for l in lines if b'"finalizing"' not in l))
    counter = iter(range(10 ** 6))
    _rechain(out, lambda e: e.update(seq=next(counter)))
    R.finalize(out)


def test_forged_scores_with_consistent_hashes_are_caught_by_recomputation(copy_of):
    _forge_scores(copy_of, lambda s: 1.0 - s)          # reverses every rank: sign of rho flips
    rep_stored = V.verify(copy_of, recompute_models=False)
    assert rep_stored["passed"], rep_stored["failures"]    # every hash and replay check is consistent
    rep = V.verify(copy_of)
    assert not rep["passed"]
    assert "run_00:rho_from_stored_scores_equals_record" not in rep["failures"]
    assert "numerical_reproducibility_policy" in rep["failures"]
    detail = next(c for c in rep["checks"] if c["check"] == "numerical_reproducibility_policy")["detail"]
    assert detail["status"] == "DECISION_RELEVANT_DIFFERENCE"
    assert not detail["per_run"]["run_00"]["published"]["ranks_equal"]


def test_small_rank_perturbation_is_reported_but_decision_invariant(copy_of):
    def swap_two(s):
        s = s.copy()
        o = np.argsort(s)
        i, j = o[100], o[101]
        s[i], s[j] = s[j], s[i]
        return s
    _forge_scores(copy_of, swap_two)
    rep = V.verify(copy_of)
    detail = next(c for c in rep["checks"] if c["check"] == "numerical_reproducibility_policy")["detail"]
    assert detail["status"] == "DECISION_INVARIANT_NUMERICAL_DIFFERENCE"
    assert detail["per_run"]["run_00"]["published"]["ranks_equal"] is False
    assert detail["per_run"]["run_00"]["published"]["max_abs_diff"] > 0


EXPECTED_FAILURE = {
    "data_bytes": "journal_replay_state_machine_and_commits",
    "record_field": "record_caches_equal_journal_replay",
    "journal_line": "journal_chain_sequence_and_identity",
    "journal_rechained_state": "record_caches_equal_journal_replay",
    "outputs_scores": "journal_replay_state_machine_and_commits",
    "run_manifest": "run_manifest_hash_matches_initialised_event",
    "twin_identity": "untrained_twin_weights_hash",
    "certificate_decision": "decision_recomputed_from_recomputed_scores",
    "preregistration": "preregistration_design_hash",
    "preflight": "preflight_hash_matches_manifest",
    "unlisted_file": "every_run_file_committed_cached_or_quarantined",
    "measurement": "journal_replay_state_machine_and_commits",
}


@pytest.mark.parametrize("what", sorted(EXPECTED_FAILURE))
def test_verifier_rejects_tampering(copy_of, tmp_path, what):
    out = copy_of
    kw = {"recompute_models": False}
    rd = out / "runs" / "run_01"
    if what == "data_bytes":
        raw = bytearray((rd / "data.npz").read_bytes()); raw[-100] ^= 0xFF; (rd / "data.npz").write_bytes(bytes(raw))
    elif what == "record_field":
        r = json.loads((rd / "record.json").read_text()); r["rho"]["primary"] = 0.9
        (rd / "record.json").write_text(json.dumps(r))
    elif what == "journal_line":
        lines = (out / "journal.jsonl").read_bytes().splitlines()
        e = json.loads(lines[5]); e["utc"] = "2000-01-01T00:00:00+00:00"; lines[5] = EV.canonical(e)
        (out / "journal.jsonl").write_bytes(b"".join(l + b"\n" for l in lines))
    elif what == "journal_rechained_state":
        _rechain(out, lambda e: e.update(to="EXCLUDED") if e.get("to") == "VALID" and e.get("run_id") == "run_01" else None)
    elif what == "outputs_scores":
        O = V.npz(rd / "outputs.npz"); O["score_published"] = O["score_published"][::-1].copy()
        (rd / "outputs.npz").write_bytes(R.npz_bytes(O))
    elif what == "run_manifest":
        m = json.loads((out / "run_manifest.json").read_text()); m["sizes"]["pairs_per_run"] = 511
        (out / "run_manifest.json").write_text(json.dumps(m))
    elif what == "twin_identity":
        m = json.loads((out / "run_manifest.json").read_text()); m["untrained_twin"]["weights_sha256"] = "0" * 64
        (out / "run_manifest.json").write_text(json.dumps(m)); kw = {}
    elif what == "certificate_decision":
        c = json.loads((out / "certificate.json").read_text()); c["decision"] = "SUPPORTED"
        (out / "certificate.json").write_text(json.dumps(c))
    elif what == "preregistration":
        p = json.loads((CRYPTO / FD.PREREGISTRATION_RELPATH).read_text()); p["design"]["statistics"]["alpha"] = 0.1
        f = tmp_path / "prereg.json"; f.write_text(json.dumps(p)); kw["prereg_path"] = f
    elif what == "preflight":
        p = json.loads((out / "preflight.json").read_text()); p["ok"] = True; p["mathematical_gates"]["C"][0]["possible_transitions"] += 1
        (out / "preflight.json").write_text(json.dumps(p))
    elif what == "unlisted_file":
        (rd / "extra.npz").write_bytes(b"x")
    elif what == "measurement":
        m = json.loads((rd / "measurement.json").read_text()); m["status"] = "EXCLUDED"
        (rd / "measurement.json").write_text(json.dumps(m))
    rep = V.verify(out, **kw)
    assert not rep["passed"], what
    assert EXPECTED_FAILURE[what] in rep["failures"], (what, rep["failures"])
