"""
AutoND CE2 runner: preflight, initialisation, the per-run state machine, finalisation.

Output directory (one per execution, never reused across modes)::

    journal.jsonl              AUTHORITATIVE hash-chained journal (core.evidence, core-journal-2)
    preflight.json             upstream, runtime, model binding, mathematical gates A-F
    behavioural.json           behavioural binding (+ behavioural/validation.npz)
    controls/untrained_twin.weights.h5
    run_manifest.json          written LAST by initialise: its presence marks a complete initialisation
    runs/run_XX/data.npz       committed artifact: genuine pairs, trajectory differences, factors, target
    runs/run_XX/outputs.npz    committed artifact: published and twin scores, target permutation
    runs/run_XX/measurement.json  committed artifact
    runs/run_XX/record.json    cache materialised from the journal (not authoritative)
    runs/run_XX/quarantine/    uncommitted artifacts found on resume (retained, journalled)
    certificate.json           written once by finalize

Run states::

    NOT_STARTED -> DATA_GENERATED -> DATA_VERIFIED -> CONTROLS_EVALUATED -> EVIDENCE_WRITTEN -> VALID | EXCLUDED
    any non-terminal state -> FAILED

Each forward transition that produces an artifact commits it with the protocol of
``core.evidence`` (intent, staged write, hash, validate, atomic promote, transition). On resume
``recover_run`` quarantines anything not committed and re-executes that step; uncommitted
artifacts are never analysed, so re-execution cannot select on an outcome. A committed artifact
that is missing or altered makes the run FAILED. A record cache that cannot be reconciled with the
journal stops the runner (InconsistentEvidence); it is never repaired.

An initialisation that stops before ``run_manifest.json`` is abandoned: the directory is kept and
``--resume`` refuses it. Nothing here reads or writes any AutoND CE1 path.
"""

from __future__ import annotations

import io
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np

from ....core import evidence as EV
from ....core.hypothesis import HypothesisDirection
from ....core.theory_consistency import TheoryConsistencyMeasurement
from ....targets.autond import model_binding as MB
from ....targets.autond import upstream as UP
from . import adapter as A
from . import analysis as S
from . import design as FD
from . import theory as T

MANIFEST_SCHEMA = "autond-ce2-run-manifest-2"
RECORD_SCHEMA = "autond-ce2-run-record-2"
SMOKE_PREFIX = "autond_ce2_smoke_"
WILSON_Z_ONE_SIDED_95 = 1.6448536269514722
CRYPTO_ROOT = UP.CRYPTO_ROOT
ARTIFACTS = ("data.npz", "outputs.npz", "measurement.json")
TERMINAL = ("VALID", "EXCLUDED", "FAILED")

MACHINE = EV.StateMachine(
    initial="NOT_STARTED",
    transitions={"NOT_STARTED": ("DATA_GENERATED", "FAILED"),
                 "DATA_GENERATED": ("DATA_VERIFIED", "FAILED"),
                 "DATA_VERIFIED": ("CONTROLS_EVALUATED", "FAILED"),
                 "CONTROLS_EVALUATED": ("EVIDENCE_WRITTEN", "FAILED"),
                 "EVIDENCE_WRITTEN": ("VALID", "EXCLUDED", "FAILED")},
    terminal=TERMINAL)


class RunnerError(RuntimeError):
    pass


utc = EV.utc
write_json = EV.write_json


def store_for(out: Path) -> EV.EvidenceStore:
    return EV.EvidenceStore(Path(out), FD.EXPERIMENT_ID, MACHINE, record_schema=RECORD_SCHEMA)


# ------------------------------------------------------------------- files


def npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def load_npz(path: Path, expected_sha256: str) -> dict:
    got = EV.sha256_file(path)
    if got != expected_sha256:
        raise EV.InconsistentEvidence(f"{path}: SHA-256 {got} != committed {expected_sha256}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _npz_validator(expected_content_sha256: str):
    def validate(path: Path) -> None:
        with np.load(path, allow_pickle=False) as z:
            got = A.content_sha256({k: z[k] for k in z.files})
        if got != expected_content_sha256:
            raise RunnerError(f"staged {path.name} content hash {got} != in-memory {expected_content_sha256}")
    return validate


def _commit_npz(store: EV.EvidenceStore, rid: str, name: str, arrays: dict) -> dict:
    data = npz_bytes(arrays)
    info = store.commit_artifact(rid, name, store.writer_with_fault(lambda p: p.write_bytes(data)),
                                 validate=_npz_validator(A.content_sha256(arrays)))
    info[f"{name.split('.')[0]}_content_sha256"] = A.content_sha256(arrays)
    return info


def _commit_json(store: EV.EvidenceStore, rid: str, name: str, obj) -> dict:
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False, default=str) + "\n").encode()

    def validate(path: Path) -> None:
        json.loads(path.read_text())
    return store.commit_artifact(rid, name, store.writer_with_fault(lambda p: p.write_bytes(data)),
                                 validate=validate)


# ------------------------------------------------------------ environment


def environment() -> dict:
    import h5py
    import keras
    import tensorflow as tf
    env = {"python": platform.python_version(), "implementation": platform.python_implementation(),
           "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
           "numpy": np.__version__, "tensorflow": tf.__version__, "keras": keras.__version__,
           "h5py": h5py.__version__, "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "tf_deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS")}
    for mod in ("scipy", "pandas"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            env[mod] = None
    try:
        env["gpu_details"] = [tf.config.experimental.get_device_details(d)
                              for d in tf.config.list_physical_devices("GPU")]
    except Exception as exc:  # informational only; the failure itself is recorded
        env["gpu_details"] = f"unavailable: {type(exc).__name__}"
    return env


def runtime_problems(env: dict) -> list[str]:
    req = FD.DESIGN["runtime"]
    out = []
    if not env["python"].startswith(req["python"] + "."):
        out.append(f"python {env['python']} (required {req['python']})")
    for k in ("tensorflow", "keras", "numpy", "pandas", "scipy"):
        if env.get(k) != req[k]:
            out.append(f"{k} {env.get(k)} (required {req[k]})")
    return out


# --------------------------------------------------------------- preflight


def math_gates(mode: str, word16_transitions) -> dict:
    g = FD.DESIGN["mathematical_gates"]
    c_sizes = FD.SMOKE["gate_c_word_sizes"] if mode == "smoke" else g["C"]["word_sizes"]
    d_sizes = FD.SMOKE["gate_d_word_sizes"] if mode == "smoke" else g["D"]["word_sizes_exhaustive_literal"]
    out = {"A": g["A"], "B": g["B"],
           "C": [T.gate_c_exhaustive(n) for n in c_sizes],
           "D": [T.gate_d_literal(n, exhaustive=True) for n in d_sizes] + [T.gate_d_word16(word16_transitions)]}
    out["passed"] = all(r["passed"] for r in out["C"] + out["D"])
    return out


class _ByteStream:
    """Deterministic byte source (PCG64) used only to compare encodings with upstream."""

    def __init__(self, seed: int):
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def __call__(self, n: int) -> bytes:
        return self.rng.bytes(n)


def upstream_encoding_check(up: UP.Upstream, n: int = 2048, seed: int = 1) -> dict:
    """Compare CE2 genuine-pair inputs with upstream make_train_data (Y = 1 rows) from one byte stream."""
    main = up.main
    original = main.urandom
    try:
        main.urandom = _ByteStream(seed)
        X, Y = main.make_train_data(up.cipher.encrypt, up.cipher.plain_bits, up.cipher.key_bits, n,
                                    FD.DESIGN["cipher"]["rounds"], A.delta_bits(up), 0)
    finally:
        main.urandom = original
    ours = A.generate_genuine_pairs(up, n, randbytes=_ByteStream(seed))["arrays"]
    Xo = A.model_input(up, ours["c0"], ours["c1"])
    rows = Y == 1
    return {"n": n, "n_genuine_rows_compared": int(rows.sum()),
            "genuine_rows_identical": bool(np.array_equal(X[rows], Xo[rows])),
            "upstream_label_convention": "Y = 1 genuine pair, Y = 0 uniform random 64-bit input",
            "note": "deterministic byte stream substituted for os.urandom in this check only"}


def preflight(up: UP.Upstream, mode: str) -> dict:
    t0 = time.time()
    env = environment()
    problems = runtime_problems(env)
    model, binding = MB.load_published(up, FD.DESIGN["reference"]["observed_architecture"])
    n_corpus = FD.SMOKE["preflight_corpus_pairs"] if mode == "smoke" else \
        FD.DESIGN["mathematical_gates"]["preflight_corpus_pairs"]
    corpus = A.generate_genuine_pairs(up, n_corpus)
    a = corpus["arrays"]
    alpha = up.cipher.ror(a["dx"][0, :-1], up.cipher.ALPHA())
    first = sorted({(int(x), int(y), int(z)) for x, y, z in zip(alpha, a["dy"][0, :-1], a["dx"][0, 1:])})
    if mode == "smoke":
        first = first[:FD.SMOKE["word16_transitions"]]
    gates = math_gates(mode, first)
    gates["E"] = corpus["gates"]["E"]
    gates["F"] = corpus["gates"]["F"]
    gates["corpus"] = {"pairs": n_corpus, "distinct_transitions": int(len({
        (int(x), int(y), int(z)) for x, y, z in zip(up.cipher.ror(a["dx"][:, :-1], up.cipher.ALPHA()).ravel(),
                                                    a["dy"][:, :-1].ravel(), a["dx"][:, 1:].ravel())})),
        "distinct_transitions_checked": "all, closed form against automaton (gate E)"}
    encoding = upstream_encoding_check(up)
    X = A.model_input(up, a["c0"], a["c1"])
    inference = MB.inference_mode_check(model, X)
    ok = (gates["passed"] and gates["E"]["passed"] and gates["F"]["passed"] and
          encoding["genuine_rows_identical"] and inference["finite"] and
          inference["predict_vs_call_training_false_max_abs_diff"] < 1e-5)
    if mode == "production":
        ok = ok and not problems
    return {"mode": mode, "utc": utc(), "ok": bool(ok), "environment": env, "runtime_problems": problems,
            "runtime_gate": "enforced" if mode == "production" else "recorded only (smoke)",
            "upstream": up.verification, "model_binding": binding, "inference_mode": inference,
            "mathematical_gates": gates, "upstream_encoding": encoding, "seconds": time.time() - t0}


def wilson_lower(k: int, n: int, z: float = WILSON_Z_ONE_SIDED_95) -> float:
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def behavioural_validation(up: UP.Upstream, model, twin, n: int, out_dir: Path) -> dict:
    b = FD.DESIGN["behavioural_binding"]
    half = n // 2
    gen = A.generate_genuine_pairs(up, half)
    g = gen["arrays"]
    Xg = A.model_input(up, g["c0"], g["c1"])
    Xr = A.random_pairs_input(n - half)
    X = np.concatenate([Xg, Xr])
    Y = np.concatenate([np.ones(half, np.uint8), np.zeros(n - half, np.uint8)])
    order = np.random.Generator(np.random.PCG64(b["shuffle_seed"])).permutation(n)
    X, Y = X[order], Y[order]
    score = MB.predict(model, X)
    twin_score = MB.predict(twin, X)
    correct = int(np.sum((score > 0.5).astype(np.uint8) == Y))
    twin_correct = int(np.sum((twin_score > 0.5).astype(np.uint8) == Y))
    lb = wilson_lower(correct, n)
    checks = {"label_balance_exact": int(Y.sum()) == half and n - int(Y.sum()) == n - half,
              "scores_finite": bool(np.all(np.isfinite(score)) and np.all(np.isfinite(twin_score))),
              "generation_gates_passed": bool(gen["gates"]["E"]["passed"] and gen["gates"]["F"]["passed"])}
    path = out_dir / "behavioural" / "validation.npz"
    sha = EV.atomic_write_bytes(path, npz_bytes({
        "key_words": g["key_words"], "pt0": g["pt0"], "pt1": g["pt1"], "c0": g["c0"], "c1": g["c1"],
        "random_inputs_packed": np.packbits(Xr, axis=1), "order": order.astype(np.int64),
        "labels": Y, "score": score, "twin_score": twin_score}))
    return {"n": n, "n_genuine": half, "n_random": n - half, "n_correct": correct, "accuracy": correct / n,
            "tpr": float(np.mean(score[Y == 1] > 0.5)), "tnr": float(np.mean(score[Y == 0] <= 0.5)),
            "wilson_one_sided_95_lower": lb, "requirement": b["requirement"],
            "checks": checks, "passed": bool(lb > 0.5 and all(checks.values())),
            "twin_diagnostic": {"n_correct": twin_correct, "accuracy": twin_correct / n,
                                "wilson_one_sided_95_lower": wilson_lower(twin_correct, n),
                                "role": "diagnostic only, not a gate"},
            "generation_gates": gen["gates"], "file": "behavioural/validation.npz", "sha256": sha,
            "note": b["note"], "scope": b["scope"]}


# ------------------------------------------------------------- initialise


def sizes(mode: str) -> dict:
    if mode == "smoke":
        return {k: FD.SMOKE[k] for k in ("n_runs", "pairs_per_run", "min_valid_runs", "behavioural_n",
                                         "bootstrap_resamples")}
    r = FD.DESIGN["runs"]
    return {"n_runs": r["n_runs"], "pairs_per_run": r["pairs_per_run"], "min_valid_runs": r["min_valid_runs"],
            "behavioural_n": FD.DESIGN["behavioural_binding"]["n"],
            "bootstrap_resamples": FD.DESIGN["statistics"]["sensitivity"]["n_resamples"]}


def run_ids(n: int) -> list[str]:
    return [f"run_{i:02d}" for i in range(n)]


def _check_out_dir(out: Path, mode: str) -> None:
    if mode not in ("smoke", "production"):
        raise RunnerError(f"unknown mode {mode!r}")
    if mode == "smoke" and not out.name.startswith(SMOKE_PREFIX):
        raise RunnerError(f"smoke output directory must be named {SMOKE_PREFIX}*")
    if mode == "production" and out.name.startswith(SMOKE_PREFIX):
        raise RunnerError("a production output directory may not use the smoke prefix")
    if "autond_ce1" in out.resolve().parts:
        raise RunnerError("CE2 output may not be written inside any AutoND CE1 directory")
    if out.exists() and any(out.iterdir()):
        raise RunnerError(f"{out} exists and is not empty; use --resume for an initialised directory")


def initialise(out: Path, mode: str, *, upstream=None, prereg_path: Path | None = None) -> dict:
    out = Path(out)
    _check_out_dir(out, mode)
    prereg_record = None
    if mode == "production":
        path = prereg_path or (CRYPTO_ROOT / FD.PREREGISTRATION_RELPATH)
        prereg = FD.load_preregistration(path)
        ok, why = FD.production_authorisation(prereg)
        if not ok:
            raise RunnerError(f"production refused: {why}")
        prereg_record = {"path": str(path), "sha256": EV.sha256_file(path), "approval": prereg["approval"]}
    out.mkdir(parents=True, exist_ok=True)
    store = store_for(out)
    store.journal.append("initialise_started", mode=mode, design_sha256=FD.design_sha256())
    up = UP.load(upstream)
    pre = preflight(up, mode)
    write_json(out / "preflight.json", pre)
    sz = sizes(mode)
    manifest = {"schema": MANIFEST_SCHEMA, "experiment_id": FD.EXPERIMENT_ID, "mode": mode,
                "evidentiary": mode == "production",
                "label": "CONFIRMATORY" if mode == "production" else "SMOKE_NOT_CONFIRMATORY",
                "design_version": FD.DESIGN_VERSION, "design_sha256": FD.design_sha256(),
                "hypothesis_direction": FD.DESIGN["hypothesis"]["direction"],
                "preregistration": prereg_record, "sizes": sz, "run_ids": run_ids(sz["n_runs"]),
                "permutation_seed_base": FD.DESIGN["controls"]["permuted_target"]["permutation_seed_base"],
                "upstream_root": str(up.root), "upstream_commit": FD.UPSTREAM_COMMIT,
                "journal_schema": EV.JOURNAL_SCHEMA, "record_schema": RECORD_SCHEMA,
                "state_machine": {"initial": MACHINE.initial, "transitions": MACHINE.transitions,
                                  "terminal": list(MACHINE.terminal)},
                "created_utc": utc(), "preflight_sha256": EV.sha256_file(out / "preflight.json"),
                "preflight_ok": pre["ok"]}
    if not pre["ok"]:
        manifest["status"] = "PRECHECK_FAILED"
        write_json(out / "run_manifest.json", manifest)
        store.journal.append("precheck_failed", run_manifest_sha256=EV.sha256_file(out / "run_manifest.json"))
        return manifest
    model, _ = MB.load_published(up, FD.DESIGN["reference"]["observed_architecture"])
    seed = FD.DESIGN["controls"]["untrained_twin"]["twin_seed"]
    twin = MB.build_untrained_twin(up, seed)
    twin_path = out / "controls" / "untrained_twin.weights.h5"
    twin_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = twin_path.with_name(".untrained_twin.tmp.weights.h5")
    twin.save_weights(str(tmp))
    os.replace(tmp, twin_path)
    twin_sha = MB.weights_sha256(twin)
    MB.load_untrained_twin(up, twin_path, twin_sha)
    manifest["untrained_twin"] = {"file": "controls/untrained_twin.weights.h5", "seed": seed,
                                  "file_sha256": EV.sha256_file(twin_path), "weights_sha256": twin_sha,
                                  "architecture": MB.architecture(twin), "trained": False}
    beh = behavioural_validation(up, model, twin, sz["behavioural_n"], out)
    write_json(out / "behavioural.json", beh)
    manifest["behavioural_sha256"] = EV.sha256_file(out / "behavioural.json")
    manifest["behavioural_passed"] = beh["passed"]
    manifest["status"] = "INITIALISED" if (beh["passed"] or mode == "smoke") else "PRECHECK_FAILED"
    write_json(out / "run_manifest.json", manifest)
    store.journal.append("initialised", status=manifest["status"], behavioural_passed=beh["passed"],
                         run_manifest_sha256=EV.sha256_file(out / "run_manifest.json"))
    return manifest


# ------------------------------------------------------------- run states


class _Session:
    """Per-process cache of the bound models for one output directory."""

    def __init__(self, out: Path, manifest: dict, upstream=None):
        self.out, self.manifest = out, manifest
        self.up = UP.load(upstream or manifest["upstream_root"])
        self.model, self.binding = MB.load_published(self.up, FD.DESIGN["reference"]["observed_architecture"])
        tw = manifest["untrained_twin"]
        twin_path = out / tw["file"]
        if EV.sha256_file(twin_path) != tw["file_sha256"]:
            raise RunnerError("untrained twin weights file changed since initialisation")
        self.twin = MB.load_untrained_twin(self.up, twin_path, tw["weights_sha256"])


def load_manifest(out: Path) -> dict:
    out = Path(out)
    p = out / "run_manifest.json"
    if not p.exists():
        if (out / "journal.jsonl").exists():
            raise RunnerError(f"{out}: initialisation never completed (no run_manifest.json); the directory "
                              "is abandoned and must not be resumed; start a new output directory")
        raise RunnerError(f"{out} is not an initialised CE2 output directory")
    m = json.loads(p.read_text())
    if m.get("schema") != MANIFEST_SCHEMA:
        raise RunnerError("unexpected run manifest schema")
    if m["design_sha256"] != FD.design_sha256():
        raise RunnerError("run manifest design hash differs from the design in this source tree")
    return m


def execute_run(store: EV.EvidenceStore, index: int, session: _Session) -> dict:
    m = session.manifest
    rid = m["run_ids"][index]
    rec = store.record(rid)                 # raises InconsistentEvidence if the cache contradicts the journal
    if rec["state"] in TERMINAL:
        return rec
    n = m["sizes"]["pairs_per_run"]
    timings = dict(rec.get("timings", {}))
    try:
        actions = store.recover_run(rid, ARTIFACTS)
        if actions:
            rec = store.record(rid)
        if rec["state"] == "NOT_STARTED":
            t = time.time()
            gen = A.generate_genuine_pairs(session.up, n)
            timings["generate_s"] = time.time() - t
            info = _commit_npz(store, rid, "data.npz", gen["arrays"])
            rec = store.transition(rid, "DATA_GENERATED", **info, generation_gates=gen["gates"], timings=timings)
        data = load_npz(store.run_dir(rid) / "data.npz", rec["data_sha256"])
        if rec["state"] == "DATA_GENERATED":
            t = time.time()
            traj, gates = A.trajectory_and_weights(session.up, data, FD.DESIGN["cipher"]["rounds"])
            same = all(np.array_equal(traj[k], data[k]) for k in ("dx", "dy", "weights", "target"))
            timings["verify_s"] = time.time() - t
            if not (gates["E"]["passed"] and gates["F"]["passed"] and same):
                return store.transition(rid, "FAILED", failure="gate E/F failed on committed data",
                                        verification_gates=gates, recomputation_identical=same, timings=timings)
            rec = store.transition(rid, "DATA_VERIFIED", verification_gates=gates, timings=timings)
        frozen = A.FrozenRunData.create(data)
        X = A.model_input(session.up, data["c0"], data["c1"])
        perm = np.random.Generator(np.random.PCG64(m["permutation_seed_base"] + index)).permutation(frozen.n)
        ctx = A.RunContext(data=frozen, published_model=session.model, twin_model=session.twin,
                           permutation=perm, model_input=X)
        if rec["state"] == "DATA_VERIFIED":
            t = time.time()
            ctx.score("published")
            timings["published_inference_s"] = time.time() - t
            t = time.time()
            ctx.score("twin")
            timings["twin_inference_s"] = time.time() - t
            info = _commit_npz(store, rid, "outputs.npz", {"score_published": ctx.scores["published"],
                                                           "score_twin": ctx.scores["twin"], "permutation": perm})
            rec = store.transition(rid, "CONTROLS_EVALUATED", **info, timings=timings,
                                   model_input_sha256=A.content_sha256({"X": X}),
                                   twin_weights_sha256=MB.weights_sha256(session.twin),
                                   published_weights_sha256=session.binding["weights_sha256"])
        outs = load_npz(store.run_dir(rid) / "outputs.npz", rec["outputs_sha256"])
        if not np.array_equal(outs["permutation"], perm):
            return store.transition(rid, "FAILED", failure="recorded permutation differs from its seed")
        ctx.scores = {"published": outs["score_published"], "twin": outs["score_twin"]}
        for s in ctx.scores.values():
            s.setflags(write=False)
        if rec["state"] == "CONTROLS_EVALUATED":
            measurement = measure(ctx, data)
            measurement["timings"] = timings
            info = _commit_json(store, rid, "measurement.json", measurement)
            rec = store.transition(rid, "EVIDENCE_WRITTEN", **info)
        mpath = store.run_dir(rid) / "measurement.json"
        if EV.sha256_file(mpath) != rec["measurement_sha256"]:
            raise EV.InconsistentEvidence("measurement.json changed after it was committed")
        measurement = json.loads(mpath.read_text())
        if measurement["status"] == "EXCLUDED":
            return store.transition(rid, "EXCLUDED", reason=measurement["exclusion_reason"])
        return store.transition(rid, "VALID", rho=measurement["rho"])
    except EV.InjectedFault:
        raise
    except Exception as exc:  # recorded and retained; the run is never replaced
        rec = store.record(rid)
        if rec["state"] in TERMINAL:
            raise
        return store.transition(rid, "FAILED", failure=f"{type(exc).__name__}: {exc}",
                                traceback=traceback.format_exc(), timings=timings)


def measure(ctx: A.RunContext, data: dict) -> dict:
    """Explicit definedness checks, then one two-sided core measurement per condition."""
    target = data["target"].astype(np.float64)
    inputs = {"primary": (ctx.scores["published"], target),
              "permuted_target": (ctx.scores["published"], target[ctx.permutation]),
              "untrained_twin": (ctx.scores["twin"], target)}
    profile = {"target": S.tie_profile(data["target"]), "score_published": S.tie_profile(ctx.scores["published"]),
               "score_twin": S.tie_profile(ctx.scores["twin"])}
    reasons = {c: S.definedness(*inputs[c]) for c in S.CONDITIONS}
    base = {"dataset_sha256": ctx.data.sha256, "n": ctx.data.n, "tie_profiles": profile, "definedness": reasons,
            "hypothesis_direction": HypothesisDirection.TWO_SIDED.value}
    if any(reasons.values()):
        return {**base, "status": "EXCLUDED",
                "exclusion_reason": "; ".join(f"{c}: {r}" for c, r in reasons.items() if r)}
    test = TheoryConsistencyMeasurement(HypothesisDirection.TWO_SIDED, S.spearman)
    results, rho = {}, {}
    for c in S.CONDITIONS:
        res = test.run(A.AutoNDCE2Adapter(ctx, c))
        direct = S.spearman(*inputs[c])
        if not res.defined or res.statistic != direct:
            raise RunnerError(f"core measurement for {c} disagrees with the direct calculation")
        results[c] = {k: v for k, v in res.to_dict().items() if k != "runtime_s"}
        rho[c] = res.statistic
    hashes = {x["dataset_sha256"] for x in ctx.calls}
    if hashes != {ctx.data.sha256}:
        raise RunnerError("measurements did not all see the run's frozen dataset")
    return {**base, "status": "VALID", "rho": rho, "measurements": results,
            "measurement_calls": {"n_measurements": len(S.CONDITIONS), "n_adapter_calls": len(ctx.calls),
                                  "dataset_hashes_seen": sorted(hashes)},
            "secondary": S.secondary_descriptives(data["weights"], ctx.scores["published"])}


# ----------------------------------------------------------------- drivers


def run_all(out: Path, *, upstream=None, only: int | None = None) -> list[dict]:
    out = Path(out)
    m = load_manifest(out)
    if m.get("status") != "INITIALISED":
        raise RunnerError(f"output directory status is {m.get('status')}; no run may execute")
    store = store_for(out)
    for note in store.open():
        print(note)
    session = _Session(out, m, upstream)
    indices = [only] if only is not None else range(len(m["run_ids"]))
    return [execute_run(store, i, session) for i in indices]


def evidence_manifest(out: Path) -> dict:
    files = {}
    for p in sorted(Path(out).rglob("*")):
        rel = p.relative_to(out).as_posix()
        if p.is_file() and rel != "certificate.json" and not rel.startswith("verification") \
                and EV.STAGING not in p.parts:
            files[rel] = EV.sha256_file(p)
    return files


def finalize(out: Path) -> dict:
    out = Path(out)
    if (out / "certificate.json").exists():
        raise RunnerError("certificate.json already exists; finalisation is performed once")
    m = load_manifest(out)
    store = store_for(out)
    store.open()
    pre = json.loads((out / "preflight.json").read_text())
    beh = json.loads((out / "behavioural.json").read_text()) if (out / "behavioural.json").exists() else None
    precheck_ok = bool(pre["ok"]) and (beh is not None and (beh["passed"] or m["mode"] == "smoke"))
    records = [store.record(rid) for rid in m["run_ids"]]
    if precheck_ok and any(r["state"] not in TERMINAL for r in records):
        raise RunnerError("not every planned run is terminal; finalisation refused")
    outcomes = [S.RunOutcome(r["run_id"], r["state"], r.get("rho", {}), r.get("reason") or r.get("failure"))
                for r in records if r["state"] in TERMINAL]
    sz = m["sizes"]
    analysis = S.analyse(outcomes, alpha=FD.DESIGN["statistics"]["alpha"], min_valid_runs=sz["min_valid_runs"],
                         n_planned=sz["n_runs"], bootstrap_resamples=sz["bootstrap_resamples"],
                         bootstrap_seed=FD.DESIGN["statistics"]["sensitivity"]["seed"],
                         precheck_ok=precheck_ok,
                         precheck_reason=None if precheck_ok else "preflight or behavioural binding failed")
    store.journal.append("finalizing", decision=analysis["decision"])
    secondary = {}
    for r in records:
        mp = store.run_dir(r["run_id"]) / "measurement.json"
        if r["state"] == "VALID" and mp.exists():
            secondary[r["run_id"]] = json.loads(mp.read_text()).get("secondary")
    quarantined = [e for e in store.journal.entries() if e["event"] == "artifact_quarantined"]
    cert = {
        "schema": FD.CERTIFICATE_SCHEMA, "experiment_id": FD.EXPERIMENT_ID, "mode": m["mode"],
        "evidentiary": m["mode"] == "production", "label": m["label"],
        "design_version": FD.DESIGN_VERSION, "design_sha256": m["design_sha256"],
        "hypothesis_direction": FD.DESIGN["hypothesis"]["direction"],
        "preregistration": m.get("preregistration"),
        "reference_model": {"checkpoint_sha256": FD.PUBLISHED_CHECKPOINT_SHA256,
                            "binding": pre.get("model_binding"), "upstream_commit": FD.UPSTREAM_COMMIT,
                            "upstream_hashes": pre.get("upstream")},
        "environment": pre.get("environment"), "runtime_problems": pre.get("runtime_problems"),
        "mathematical_gates_passed": pre.get("mathematical_gates", {}).get("passed"),
        "behavioural_binding": beh and {k: beh.get(k) for k in ("n", "accuracy", "wilson_one_sided_95_lower",
                                                                "checks", "passed", "twin_diagnostic",
                                                                "sha256", "note", "scope")},
        "untrained_twin": m.get("untrained_twin"), "pairs_per_run": sz["pairs_per_run"],
        "run_states": {r["run_id"]: r["state"] for r in records},
        "quarantined_artifacts": [{k: e[k] for k in ("run_id", "source", "destination", "sha256", "reason")}
                                  for e in quarantined],
        "analysis": analysis, "decision": analysis["decision"], "decision_reason": analysis["decision_reason"],
        "secondary_descriptive": secondary,
        "evidence_manifest": evidence_manifest(out),
        "status": {"production": "COMPLETE" if m["mode"] == "production" else "NOT_APPLICABLE (smoke)",
                   "independent_verification": "NOT_RUN",
                   "certificate": "PENDING_INDEPENDENT_VERIFICATION"},
        "independent_verification": ("NOT_PERFORMED_BY_FINALIZE: run experiments.autond.ce2.verify on this "
                                     "directory; the certificate alone does not establish verification"),
        "claim_scope": FD.DESIGN["research_question"] + " Observational; " + FD.DESIGN["target"]["interpretation"],
        "limitations": LIMITATIONS,
        "ce1_relation": "independent of AutoND CE1; uses no CE1 evidence and does not amend or explain CE1",
        "generated_utc": utc(),
    }
    write_json(out / "certificate.json", cert)
    return cert


LIMITATIONS = [
    "observational: association only, no causal interpretation",
    "the target is a Markov-model single-characteristic probability proxy, not the exact fixed-key "
    "characteristic probability and not the differential probability of the full cipher",
    "round keys are derived by the key schedule from one master key and are not independent random variables",
    "conditional on one published checkpoint; no claim about independently trained AutoND models",
    "the untrained-twin control is one fixed random initialisation (twin_seed); a control result is "
    "conditional on that twin and is not a statement about untrained DBitNets in general",
    "genuine pairs only, unlike the balanced classification population the distinguisher was trained on",
    "Spearman association does not establish learned differential cryptanalysis or an internal representation",
    "secondary per-round quantities are descriptive latent trajectory quantities",
    "the CE2 behavioural accuracy and the published AutoND accuracy come from different evaluation conditions",
    "does not amend or explain AutoND CE1",
]
