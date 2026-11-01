from __future__ import annotations

import io
import json
import os
import platform
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from ....core import evidence as EV
from ....targets.autond import model_binding as MB
from ....targets.autond import upstream as UP
from . import adapter as A
from . import analysis as AN
from . import design as D
from . import probes as PR
from . import provenance as PV
from . import statistics as ST
from .target_bins import select_partition

CRYPTO_ROOT = UP.CRYPTO_ROOT
CE3_ROOT = CRYPTO_ROOT / D.CE3_RELPATH
EVIDENCE_ROOT = CRYPTO_ROOT / D.EVIDENCE_RELPATH
MANIFEST_SCHEMA = "autond-ce3-run-manifest-1"
RECORD_SCHEMA = "autond-ce3-run-record-1"
SMOKE_PREFIX = "autond_ce3_smoke_"
TWIN_FILE = "controls/untrained_twin.weights.h5"
ARTIFACTS = ("data.npz", "calibration.npz", "representations.npz", "probes.npz", "measurement.json")
TERMINAL = ("VALID", "EXCLUDED", "FAILED")
PRODUCER_SOURCES = ("__init__.py", "adapter.py", "analysis.py", "design.py", "p1.py", "probes.py", "provenance.py",
                    "runner.py", "statistics.py", "target_bins.py")

MACHINE = EV.StateMachine(
    initial="NOT_STARTED",
    transitions={"NOT_STARTED": ("DATA_GENERATED", "FAILED"),
                 "DATA_GENERATED": ("DATA_VERIFIED", "FAILED"),
                 "DATA_VERIFIED": ("REPRESENTATIONS_EXTRACTED", "FAILED"),
                 "REPRESENTATIONS_EXTRACTED": ("PROBES_FITTED", "FAILED"),
                 "PROBES_FITTED": ("EVIDENCE_WRITTEN", "FAILED"),
                 "EVIDENCE_WRITTEN": ("VALID", "EXCLUDED", "FAILED")},
    terminal=TERMINAL)


class RunnerError(RuntimeError):
    pass


utc = EV.utc
write_json = EV.write_json


def store_for(out: Path) -> EV.EvidenceStore:
    return EV.EvidenceStore(Path(out), D.EXPERIMENT_ID, MACHINE, record_schema=RECORD_SCHEMA)


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


def _commit_npz(store, rid, name, arrays) -> dict:
    data = npz_bytes(arrays)
    want = A.content_sha256(arrays)

    def validate(path: Path) -> None:
        with np.load(path, allow_pickle=False) as z:
            if A.content_sha256({k: z[k] for k in z.files}) != want:
                raise RunnerError(f"staged {name} content differs from memory")
    info = store.commit_artifact(rid, name, store.writer_with_fault(lambda p: p.write_bytes(data)), validate=validate)
    info[f"{name.split('.')[0]}_content_sha256"] = want
    return info


def _commit_json(store, rid, name, obj) -> dict:
    data = (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    return store.commit_artifact(rid, name, store.writer_with_fault(lambda p: p.write_bytes(data)),
                                 validate=lambda p: json.loads(p.read_text()))


def environment() -> dict:
    import h5py
    import keras
    import scipy
    import sklearn
    import tensorflow as tf
    import threadpoolctl
    env = {"python": platform.python_version(), "implementation": platform.python_implementation(),
           "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
           "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
           "threadpoolctl": threadpoolctl.__version__, "tensorflow": tf.__version__, "keras": keras.__version__,
           "h5py": h5py.__version__, "gpus": [d.name for d in tf.config.list_physical_devices("GPU")]}
    try:
        import pandas
        env["pandas"] = pandas.__version__
    except ImportError:
        env["pandas"] = None
    return env


def runtime_problems(env: dict) -> list[str]:
    req = D.DESIGN["runtime"]
    out = [] if env["python"].startswith(req["python"] + ".") else [f"python {env['python']} (required {req['python']})"]
    for k in ("tensorflow", "keras", "numpy", "pandas", "scipy", "scikit_learn", "threadpoolctl"):
        if env.get(k) != req[k]:
            out.append(f"{k} {env.get(k)} (required {req[k]})")
    return out


def source_hashes() -> dict:
    return {name: EV.sha256_file(CE3_ROOT / name) for name in PRODUCER_SOURCES}


def check_p1(up: UP.Upstream) -> dict:
    path = CRYPTO_ROOT / D.P1_RECORD_RELPATH
    out = {"record": D.P1_RECORD_RELPATH, "pinned_sha256": D.P1_RECORD_SHA256}
    if not path.is_file() or EV.sha256_file(path) != D.P1_RECORD_SHA256:
        return {**out, "ok": False, "problem": "P1 record missing or hash differs from the design pin"}
    rec = D.parse_json_strict(path.read_text())
    data = path.parent / rec["data_file"]
    if EV.sha256_file(data) != rec["data_sha256"]:
        return {**out, "ok": False, "problem": "P1 data hash differs from the record"}
    with np.load(data, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    c = up.cipher
    rounds = 8
    full = {"key_words": a["key_words"], "pt0": a["pt0"], "pt1": a["pt1"]}
    kb = np.concatenate([((a["key_words"][:, j:j + 1].astype(np.int64) >> np.arange(15, -1, -1)) & 1)
                         for j in range(4)], 1).astype(np.uint8)
    pb = np.concatenate([((a["pt0"][:, j:j + 1].astype(np.int64) >> np.arange(15, -1, -1)) & 1)
                         for j in range(2)], 1).astype(np.uint8)
    pb1 = pb ^ A.CE2A.delta_bits(up)
    full["c0"] = c.convert_from_binary(c.encrypt(pb, kb, rounds))
    full["c1"] = c.convert_from_binary(c.encrypt(pb1, kb, rounds))
    traj, gates = A.trajectory_and_weights(up, full, rounds)
    same_target = bool(np.array_equal(traj["target"], a["target"]))
    sel = select_partition(a["target"], D.N_CLASSES, D.MIN_CLASS_FRACTION)
    ok = (same_target and gates["E"]["passed"] and gates["F"]["passed"] and sel["feasible"]
          and sel["edges"] == D.TARGET_EDGES and rec["selected_edges"] == D.TARGET_EDGES
          and rec["selected_n_classes"] == D.N_CLASSES and rec["model_loaded"] is False and rec["evidentiary"] is False
          and bool(np.array_equal(a["pt0"] ^ a["pt1"], np.tile(np.array(A.CE2A.delta_words(up), np.uint16), (a["pt0"].shape[0], 1)))))
    return {**out, "ok": bool(ok), "n": int(a["target"].size), "target_recomputed_identical": same_target,
            "gates_passed": bool(gates["E"]["passed"] and gates["F"]["passed"]), "edges_reselected": sel.get("edges"),
            "class_fractions": sel.get("class_fractions")}


def pinned_file(rel: str, sha: str) -> dict:
    p = CRYPTO_ROOT / rel
    if sha == D.UNSET:
        return {"relpath": rel, "ok": False, "status": "NOT_YET_RECORDED"}
    return {"relpath": rel, "ok": p.is_file() and EV.sha256_file(p) == sha, "sha256": sha}


def benchmark_consistent() -> dict:
    pin = pinned_file(D.BENCHMARK_RELPATH, D.BENCHMARK_SHA256)
    if not pin["ok"]:
        return pin
    rec = D.parse_json_strict((CRYPTO_ROOT / D.BENCHMARK_RELPATH).read_text())
    ok = (rec.get("selected_max_iter") == D.PROBE["logistic"]["max_iter"] and rec.get("all_converged") is True
          and rec.get("scores_computed") is False and rec.get("main_pairs") == D.N_MAIN_PAIRS
          and rec.get("permutations") == D.N_PERMUTATIONS and rec.get("edges") == D.TARGET_EDGES
          and rec.get("predictions_computed") is False
          and rec.get("permutation_construction") == D.DESIGN["sampling"]["permutation_construction"])
    return {**pin, "ok": bool(ok), "selected_max_iter": rec.get("selected_max_iter")}


def preflight(up: UP.Upstream, mode: str) -> dict:
    t0 = time.time()
    env = environment()
    problems = runtime_problems(env)
    deps = PV.dependency_problems(CRYPTO_ROOT)
    try:
        ce2 = {"ok": True, **PV.verify_ce2_source_run(CRYPTO_ROOT, hash_all_evidence=(mode == "production"))}
    except PV.ProvenanceError as exc:
        ce2 = {"ok": False, "problem": str(exc)}
    model, twin, binding = A.load_models(up, CRYPTO_ROOT / D.CE2_RUN_RELPATH / D.CE2_TWIN_RELPATH)
    rb = {"published": A.bind_representation(model), "twin": A.bind_representation(twin)}
    gen = A.main_dataset(up, 2048)
    X = A.model_input(up, gen["arrays"]["c0"], gen["arrays"]["c1"])
    heads = {}
    for role, m in (("published", model), ("twin", twin)):
        ext = A.extractor(m)
        h1, h2 = A.extract(ext, X), A.extract(ext, X)
        heads[role] = {"head_max_abs_diff": A.head_difference(m, h1, MB.predict(m, X)),
                       "extraction_repeatable": bool(np.array_equal(h1, h2)), "finite": bool(np.all(np.isfinite(h1))),
                       "nonconstant_features": A.n_nonconstant(h1)}
    p1 = check_p1(up)
    pins = {"power": pinned_file(D.POWER_EVIDENCE_RELPATH, D.POWER_EVIDENCE_SHA256), "benchmark": benchmark_consistent(),
            "representation_diagnostics": pinned_file(D.REPRESENTATION_DIAGNOSTICS_RELPATH,
                                                      D.REPRESENTATION_DIAGNOSTICS_SHA256)}
    ok = (not deps and ce2["ok"] and rb["published"]["ok"] and rb["twin"]["ok"] and p1["ok"]
          and gen["gates"]["E"]["passed"] and gen["gates"]["F"]["passed"]
          and all(h["head_max_abs_diff"] <= D.HEAD_TOLERANCE and h["extraction_repeatable"] and h["finite"]
                  for h in heads.values()))
    if mode == "production":
        ok = ok and not problems and all(p["ok"] for p in pins.values())
    return {"mode": mode, "utc": utc(), "ok": bool(ok), "environment": env, "runtime_problems": problems,
            "runtime_gate": "enforced" if mode == "production" else "recorded only (smoke)",
            "dependency_problems": deps, "ce2_source_run": ce2, "upstream": up.verification, "model_binding": binding,
            "representation_binding": rb, "head_and_extraction": heads, "generation_gates": gen["gates"], "p1": p1,
            "validation_pins": pins,
            "validation_pin_gate": "enforced" if mode == "production" else "recorded only (smoke)",
            "seconds": time.time() - t0}


def run_ids(n: int) -> list[str]:
    return [f"rep_{i:02d}" for i in range(n)]


def _check_out_dir(out: Path, mode: str) -> None:
    if mode not in ("smoke", "production"):
        raise RunnerError(f"unknown mode {mode!r}")
    r = out.resolve()
    inside_ev = EVIDENCE_ROOT.resolve() in r.parents
    if mode == "production" and (not inside_ev or out.name.startswith(SMOKE_PREFIX)):
        raise RunnerError(f"production output must be a non-smoke directory under {EVIDENCE_ROOT}")
    if mode == "smoke" and not out.name.startswith(SMOKE_PREFIX):
        raise RunnerError(f"smoke output directory must be named {SMOKE_PREFIX}*")
    repo = CRYPTO_ROOT.parents[1].resolve()
    if (repo in r.parents or r == repo) and not inside_ev:
        raise RunnerError(f"inside the repository CE3 may only write under {EVIDENCE_ROOT}")
    if out.exists() and any(out.iterdir()):
        raise RunnerError(f"{out} exists and is not empty; use --resume for an initialised directory")


def initialise(out: Path, mode: str, *, upstream=None, prereg_path: Path | None = None) -> dict:
    out = Path(out)
    _check_out_dir(out, mode)
    prereg_record = None
    if mode == "production":
        path = Path(prereg_path or (CRYPTO_ROOT / D.PREREGISTRATION_RELPATH))
        if not path.is_file():
            raise RunnerError(f"production refused: no preregistration at {path}")
        prereg = D.load_preregistration(path)
        ok, why = D.production_authorisation(prereg)
        if not ok:
            raise RunnerError(f"production refused: {why}")
        prereg_record = {"path": str(path), "sha256": EV.sha256_file(path), "approval": prereg["approval"]}
    out.mkdir(parents=True, exist_ok=True)
    store = store_for(out)
    rd = D.run_design(mode)
    write_json(out / "design.json", rd)
    store.journal.append("initialise_started", mode=mode, design_sha256=D.design_sha256(),
                         run_design_sha256=D.design_sha256(rd))
    up = UP.load(upstream)
    pre = preflight(up, mode)
    write_json(out / "preflight.json", pre)
    sz = D.sizes(mode)
    manifest = {"schema": MANIFEST_SCHEMA, "experiment_id": D.EXPERIMENT_ID, "mode": mode,
                "evidentiary": mode == "production",
                "label": "CONFIRMATORY" if mode == "production" else "SMOKE_NOT_CONFIRMATORY",
                "design_version": D.DESIGN_VERSION, "design_sha256": D.design_sha256(),
                "run_design_sha256": D.design_sha256(rd), "design_file_sha256": EV.sha256_file(out / "design.json"),
                "preregistration": prereg_record, "sizes": sz, "run_ids": run_ids(sz["n_replicates"]),
                "edges": D.TARGET_EDGES, "n_classes": D.N_CLASSES, "max_iter": D.PROBE["logistic"]["max_iter"],
                "upstream_root": str(up.root), "upstream_commit": D.UPSTREAM_COMMIT,
                "producer_sources": source_hashes(), "journal_schema": EV.JOURNAL_SCHEMA, "record_schema": RECORD_SCHEMA,
                "state_machine": {"initial": MACHINE.initial, "transitions": MACHINE.transitions,
                                  "terminal": list(MACHINE.terminal)},
                "created_utc": utc(), "preflight_sha256": EV.sha256_file(out / "preflight.json"),
                "preflight_ok": pre["ok"]}
    if not pre["ok"]:
        manifest["status"] = "PRECHECK_FAILED"
        write_json(out / "run_manifest.json", manifest)
        store.journal.append("precheck_failed", run_manifest_sha256=EV.sha256_file(out / "run_manifest.json"))
        return manifest
    src = CRYPTO_ROOT / D.CE2_RUN_RELPATH / D.CE2_TWIN_RELPATH
    dst = out / TWIN_FILE
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(".twin.tmp.weights.h5")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    if EV.sha256_file(dst) != D.CE2_TWIN_FILE_SHA256:
        raise RunnerError("copied twin file hash differs from the CE2 pin")
    MB.load_untrained_twin(up, dst, D.CE2_TWIN_WEIGHTS_SHA256)
    manifest["untrained_twin"] = {"file": TWIN_FILE, "file_sha256": D.CE2_TWIN_FILE_SHA256,
                                  "weights_sha256": D.CE2_TWIN_WEIGHTS_SHA256, "trained": False,
                                  "source": f"{D.CE2_RUN_RELPATH}/{D.CE2_TWIN_RELPATH}",
                                  "seed": D.CE2_SOURCE_RUN["twin_seed"]}
    manifest["status"] = "INITIALISED"
    write_json(out / "run_manifest.json", manifest)
    store.journal.append("initialised", status=manifest["status"],
                         run_manifest_sha256=EV.sha256_file(out / "run_manifest.json"))
    return manifest


def load_manifest(out: Path) -> dict:
    out = Path(out)
    p = out / "run_manifest.json"
    if not p.exists():
        if (out / "journal.jsonl").exists():
            raise RunnerError(f"{out}: initialisation never completed; the directory is abandoned")
        raise RunnerError(f"{out} is not an initialised CE3 output directory")
    m = json.loads(p.read_text())
    if m.get("schema") != MANIFEST_SCHEMA:
        raise RunnerError("unexpected run manifest schema")
    if m["design_sha256"] != D.design_sha256():
        raise RunnerError("run manifest design hash differs from the design in this source tree")
    if m["producer_sources"] != source_hashes():
        raise RunnerError("CE3 producer sources changed since initialisation")
    return m


class _Session:
    def __init__(self, out: Path, manifest: dict, upstream=None):
        self.out, self.manifest = out, manifest
        self.up = UP.load(upstream or manifest["upstream_root"])
        tw = manifest["untrained_twin"]
        if EV.sha256_file(out / tw["file"]) != tw["file_sha256"]:
            raise RunnerError("untrained twin file changed since initialisation")
        self.model, self.twin, self.binding = A.load_models(self.up, out / tw["file"])
        self.ext = {"published": A.extractor(self.model), "twin": A.extractor(self.twin)}


def _data_ok(up, data: dict) -> tuple[bool, dict]:
    traj, gates = A.trajectory_and_weights(up, data, 8)
    same = all(np.array_equal(traj[k], data[k]) for k in ("dx", "dy", "weights", "target"))
    return bool(gates["E"]["passed"] and gates["F"]["passed"] and same), gates


def execute_replicate(store: EV.EvidenceStore, index: int, s: _Session) -> dict:
    m = s.manifest
    rid = m["run_ids"][index]
    rec = store.record(rid)
    if rec["state"] in TERMINAL:
        return rec
    sz = m["sizes"]
    timings = dict(rec.get("timings", {}))
    try:
        if store.recover_run(rid, ARTIFACTS):
            rec = store.record(rid)
        if rec["state"] == "NOT_STARTED":
            t = time.time()
            main = A.main_dataset(s.up, sz["main_pairs"])
            arrays = dict(main["arrays"])
            arrays["classes"] = A.classes_for(arrays["target"], m["edges"])
            arrays["s1_tiebreak"] = PR.draw_tiebreak(sz["permutations"])
            cal = A.calibration_dataset(s.up, sz["calibration_per_class"], index)
            timings["generate_s"] = time.time() - t
            info = _commit_npz(store, rid, "data.npz", arrays)
            info.update(_commit_npz(store, rid, "calibration.npz", cal["arrays"]))
            rec = store.transition(rid, "DATA_GENERATED", **info, generation_gates={"main": main["gates"],
                                                                                    "calibration": cal["gates"]},
                                   timings=timings)
        data = load_npz(store.run_dir(rid) / "data.npz", rec["data_sha256"])
        cal = load_npz(store.run_dir(rid) / "calibration.npz", rec["calibration_sha256"])
        if rec["state"] == "DATA_GENERATED":
            ok_main, g_main = _data_ok(s.up, data)
            _, g_cal = A.trajectory_and_weights(s.up, {k: cal[k] for k in ("key_words", "pt0", "pt1", "c0", "c1")}, 8)
            ok_cal = bool(g_cal["E"]["passed"] and g_cal["F"]["passed"])
            classes_ok = bool(np.array_equal(A.classes_for(data["target"], m["edges"]), data["classes"]))
            n_cal = sz["calibration_per_class"]
            order_ok = bool(np.array_equal(cal["order"], np.random.Generator(np.random.PCG64(
                D.CALIBRATION_SHUFFLE_SEED_BASE + index)).permutation(2 * n_cal)))
            labels_ok = bool(np.array_equal(cal["labels"], np.concatenate(
                [np.ones(n_cal, np.int8), np.zeros(n_cal, np.int8)])[cal["order"]]))
            tiebreak_ok = PR.tiebreak_ok(data["s1_tiebreak"], sz["permutations"])
            if not (ok_main and ok_cal and classes_ok and order_ok and labels_ok and tiebreak_ok):
                return store.transition(rid, "FAILED", failure="committed data failed verification",
                                        checks={"main": ok_main, "calibration": ok_cal, "classes": classes_ok,
                                                "order": order_ok, "labels": labels_ok, "tiebreak": tiebreak_ok},
                                        timings=timings)
            rec = store.transition(rid, "DATA_VERIFIED", verification_gates={"main": g_main, "calibration": g_cal},
                                   timings=timings)
        X = A.model_input(s.up, data["c0"], data["c1"])
        Xc = A.calibration_input(s.up, cal)
        if rec["state"] == "DATA_VERIFIED":
            t = time.time()
            reps = {"trained_main": A.extract(s.ext["published"], X), "twin_main": A.extract(s.ext["twin"], X),
                    "trained_cal": A.extract(s.ext["published"], Xc), "twin_cal": A.extract(s.ext["twin"], Xc),
                    "score_main": MB.predict(s.model, X), "score_cal": MB.predict(s.model, Xc)}
            timings["extract_s"] = time.time() - t
            head = {"main": A.head_difference(s.model, reps["trained_main"], reps["score_main"]),
                    "cal": A.head_difference(s.model, reps["trained_cal"], reps["score_cal"])}
            finite = all(bool(np.all(np.isfinite(v))) for v in reps.values())
            info = _commit_npz(store, rid, "representations.npz", reps)
            if max(head.values()) > D.HEAD_TOLERANCE or not finite:
                return store.transition(rid, "FAILED", failure="head consistency or finiteness failed",
                                        head_max_abs_diff=head, finite=finite, **info, timings=timings)
            rec = store.transition(rid, "REPRESENTATIONS_EXTRACTED", **info, head_max_abs_diff=head,
                                   model_input_sha256=A.content_sha256({"X": X}),
                                   calibration_input_sha256=A.content_sha256({"X": Xc}),
                                   published_weights_sha256=s.binding["weights_sha256"],
                                   twin_weights_sha256=MB.weights_sha256(s.twin), timings=timings)
        reps = load_npz(store.run_dir(rid) / "representations.npz", rec["representations_sha256"])
        fold_main = PR.fold_ids(data["classes"].size, PR.main_fold_seed(index))
        fold_cal = PR.fold_ids(cal["labels"].size, PR.calibration_fold_seed(index))
        perm_index = PR.permutations(data["classes"].size, index, sz["permutations"])
        if rec["state"] == "REPRESENTATIONS_EXTRACTED":
            try:
                chk = AN.check_inputs(classes=data["classes"], fold_main=fold_main, perm_index=perm_index,
                                      twin_main=reps["twin_main"], twin_cal=reps["twin_cal"], cal_labels=cal["labels"],
                                      fold_cal=fold_cal, min_per_fold=sz["min_class_count_per_fold"])
                t = time.time()
                fitted = AN.fit_all(classes=data["classes"], fold_main=fold_main, perm_index=perm_index,
                                    trained_main=reps["trained_main"], twin_main=reps["twin_main"],
                                    raw_main=X.astype(np.float64), trained_cal=reps["trained_cal"],
                                    twin_cal=reps["twin_cal"], cal_labels=cal["labels"], fold_cal=fold_cal,
                                    max_iter=m["max_iter"])
                timings["fit_s"] = time.time() - t
            except AN.AnalysisError as exc:
                return store.transition(rid, "FAILED", failure=f"AnalysisError: {exc}", timings=timings)
            info = _commit_npz(store, rid, "probes.npz", fitted["arrays"])
            rec = store.transition(rid, "PROBES_FITTED", **info, coverage=chk["coverage"],
                                   twin_nonconstant={"main": chk["twin_nonconstant_main"],
                                                     "cal": chk["twin_nonconstant_cal"]},
                                   fit_seconds=fitted["fit_seconds"], timings=timings)
        pz = load_npz(store.run_dir(rid) / "probes.npz", rec["probes_sha256"])
        if not (np.array_equal(pz["fold_main"], fold_main) and np.array_equal(pz["fold_cal"], fold_cal)
                and np.array_equal(pz["perm_index"], perm_index)):
            return store.transition(rid, "FAILED", failure="recorded folds or permutations differ from their seeds")
        if rec["state"] == "PROBES_FITTED":
            meas = AN.measure(pz, classes=data["classes"], cal_labels=cal["labels"], score_cal=reps["score_cal"],
                              n_permutations=sz["permutations"], tiebreak=data["s1_tiebreak"])
            meas["timings"] = timings
            info = _commit_json(store, rid, "measurement.json", meas)
            rec = store.transition(rid, "EVIDENCE_WRITTEN", **info)
        mp = store.run_dir(rid) / "measurement.json"
        if EV.sha256_file(mp) != rec["measurement_sha256"]:
            raise EV.InconsistentEvidence("measurement.json changed after it was committed")
        meas = json.loads(mp.read_text())
        if meas["status"] == "EXCLUDED":
            return store.transition(rid, "EXCLUDED", reason=meas["exclusion_reason"])
        return store.transition(rid, "VALID", endpoints=meas["endpoints"])
    except EV.InjectedFault:
        raise
    except Exception as exc:
        rec = store.record(rid)
        if rec["state"] in TERMINAL:
            raise
        return store.transition(rid, "FAILED", failure=f"{type(exc).__name__}: {exc}",
                                traceback=traceback.format_exc(), timings=timings)


def run_all(out: Path, *, upstream=None, only: int | None = None) -> list[dict]:
    out = Path(out)
    m = load_manifest(out)
    if m.get("status") != "INITIALISED":
        raise RunnerError(f"output directory status is {m.get('status')}; no replicate may execute")
    store = store_for(out)
    for note in store.open():
        print(note)
    s = _Session(out, m, upstream)
    idx = [only] if only is not None else range(len(m["run_ids"]))
    return [execute_replicate(store, i, s) for i in idx]


def evidence_manifest(out: Path) -> dict:
    files = {}
    for p in sorted(Path(out).rglob("*")):
        rel = p.relative_to(out).as_posix()
        if p.is_file() and rel != "certificate.json" and not rel.startswith("verification_") \
                and EV.STAGING not in p.parts:
            files[rel] = EV.sha256_file(p)
    return files


LIMITATIONS = [
    "decodability, not use: A3 shows a representation can carry a decodable target the network does not use",
    "a fixed linear probe: a non-rejection does not establish that the target is absent or not decodable nonlinearly",
    "probe accessibility, not information: no representation of the input carries more information about T than the "
    "input; the raw-input contrast is descriptive",
    "the target is the CE2 Markov single-characteristic proxy coarsened into five whole-value classes; CE3 is not "
    "independent of CE2",
    "one published checkpoint, one layer, one fixed untrained twin; nothing is claimed for other models or twins",
    "the permuted-label control removes the sample-specific target while keeping its marginal; it is not a "
    "Hewitt-Liang control task",
    "genuine pairs only in the main set; calibration uses the balanced genuine-versus-random task",
    "producer and verifier share TensorFlow/Keras inference, the pinned upstream DBitNet builder, the pinned CE2 r3 "
    "verifier primitives (cipher, trajectory, xdp+, HDF5 binding, weights and content hashing, findings) and the "
    "core-journal-2 evidence format written by core.evidence",
]


def finalize(out: Path) -> dict:
    out = Path(out)
    if (out / "certificate.json").exists():
        raise RunnerError("certificate.json already exists; finalisation is performed once")
    m = load_manifest(out)
    store = store_for(out)
    store.open()
    pre = json.loads((out / "preflight.json").read_text())
    precheck_ok = bool(pre["ok"]) and m.get("status") == "INITIALISED"
    records = [store.record(rid) for rid in m["run_ids"]]
    if precheck_ok and any(r["state"] not in TERMINAL for r in records):
        raise RunnerError("not every planned replicate is terminal; finalisation refused")
    states = {r["run_id"]: r["state"] for r in records}
    endpoints = {r["run_id"]: r.get("endpoints") for r in records if r["state"] == "VALID"}
    analysis = ST.analyse(endpoints, states, alpha=D.ALPHA, n_required=m["sizes"]["n_required_valid"],
                          precheck_ok=precheck_ok,
                          precheck_reason=None if precheck_ok else "preflight or initialisation failed")
    store.journal.append("finalizing", decision=analysis["decision"])
    quarantined = [e for e in store.journal.entries() if e["event"] == "artifact_quarantined"]
    cert = {
        "schema": D.CERTIFICATE_SCHEMA, "experiment_id": D.EXPERIMENT_ID, "mode": m["mode"],
        "evidentiary": m["mode"] == "production", "label": m["label"], "design_version": D.DESIGN_VERSION,
        "design_sha256": m["design_sha256"], "run_design_sha256": m["run_design_sha256"],
        "preregistration": m.get("preregistration"),
        "reference_model": {"checkpoint_sha256": D.PUBLISHED_CHECKPOINT_SHA256, "upstream_commit": D.UPSTREAM_COMMIT,
                            "binding": pre.get("model_binding"),
                            "representation": pre.get("representation_binding", {}).get("published")},
        "untrained_twin": m.get("untrained_twin"), "ce2_source_run": pre.get("ce2_source_run"),
        "environment": pre.get("environment"), "runtime_problems": pre.get("runtime_problems"),
        "target": {"edges": m["edges"], "n_classes": m["n_classes"], "p1": pre.get("p1")},
        "sizes": m["sizes"], "max_iter": m["max_iter"], "run_states": states,
        "replicate_endpoints": endpoints,
        "quarantined_artifacts": [{k: e[k] for k in ("run_id", "source", "destination", "sha256", "reason")}
                                  for e in quarantined],
        "analysis": analysis, "decision": analysis["decision"], "decision_reason": analysis["reason"],
        "evidence_manifest": evidence_manifest(out),
        "status": {"production": "COMPLETE" if m["mode"] == "production" else "NOT_APPLICABLE (smoke)",
                   "independent_verification": "NOT_RUN", "certificate": "PENDING_INDEPENDENT_VERIFICATION"},
        "claim_scope": D.DESIGN["claim_scope"], "limitations": LIMITATIONS, "generated_utc": utc(),
    }
    write_json(out / "certificate.json", cert)
    return cert
