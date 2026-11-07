"""
Deterministic SDB-V2 fixtures, built once BEFORE registration and hashed into the manifest.

Every synthetic input is generated here from a fixed seed; existing artifacts (checkpoints,
certificates, the H2 bundle) are referenced by repository path and hash, never copied or
modified. Each case directory holds a `bundle.json`: what the "authors" publish (files, reported
accuracy, described architecture, published checksums, environment) and the B2 applicability of
each checklist item with its reason. No model is trained to build any fixture.

The K7 and U4 fixtures are PAR1 runs of the frozen FakeAdapter (the PAR1 test factory), built
through the frozen PAR1 scheduler into this directory; U4's certified accuracy is then edited.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIXTURES = HERE / "fixtures"


def _rel(p) -> str:
    """Repository-relative when inside the repository (registered fixtures); absolute for a smoke root."""
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)

H1_CKPT = "audit/cryptography/evidence/ce1/best5depth10 (10).h5"
K3_CKPT = "audit/cryptography/Archive/best5depth10.h5"
S2_CKPT = "audit/cryptography/evidence/ce1/signal_destroyed.h5"
REFERENCE_SPECK = "audit/implementation/reference/speck.py"
RELEASED_SPECK = "speck.py"
A3_CERT = "audit/adversarial/evidence/a3_20261006/a3_certificate.json"
E6_CERT = "audit/adversarial/evidence/e6_ce2_20261105/e6_certificate.json"
V5_CERT = "audit/implementation/evidence/implementation-certificate-V5.json"
AD4_CERT = "audit/dataset/evidence/d4_autond_speck3264_3ad26073acf0/result/d4_certificate.json"
H2_BUNDLE = "audit/framework_validation/sdb_v2/evidence/h2_ad4"

EXPECTED_HASHES = {
    H1_CKPT: "de630afcf95e7d0d", K3_CKPT: "256eb4a5ba93414f", S2_CKPT: "a110fad1e5b3adc3",
    A3_CERT: "d90ec393f69e785d", E6_CERT: "f071aae2cf01d4fc", AD4_CERT: "b4527041ff788e53",
}

REPORTED_ACCURACY = 0.9291             # V5 certificate observation.findings[0].declared (Gohr, depth 10)
PUBLISHED_ARCHITECTURE = {"conv1d": 21, "add": 10, "params": 102497,
                          "source": "train_nets.make_resnet(depth=10, reg_param=1e-5), released code"}
ENV = {"python": "3.13.0", "numpy": "2.1.3", "tensorflow": "2.20.0", "keras": "3.13.2",
       "note": "constructed for the fixture; uninformative about any real publication"}

SEEDS = {"partitions": 20261106, "d2n_observed": 20261108, "d2n_null_base": 20261200, "d2n_query_base": 20261300,
         "s4_plant": 20261109, "ev": 20261110}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _b2(**items):
    """B2 applicability per item: 'E' executed, ('NS', reason), ('NA', reason)."""
    out = {}
    for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):
        v = items.get(k, ("NA", "subject absent by construction"))
        out[k] = {"status": "E"} if v == "E" else {"status": v[0], "reason": v[1]}
    return out


C4_NS = ("NS", "no approved B1 tolerance; B1 is descriptive")


def _write(d: Path, name: str, obj) -> str:
    p = d / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    return _rel(p)


def _bundle(d: Path, case: str, **fields) -> None:
    files = fields.pop("files", [])
    checks = fields.pop("published_checksums", {})
    body = {"case": case, "files": files, "environment": ENV, "published_checksums": checks, **fields}
    (d / "bundle.json").write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")


def _rows(rng, n):
    return rng.integers(0, 2, size=(n, 64), dtype=np.uint8)


def _np(d: Path, name: str, **arrays) -> str:
    p = d / name
    np.savez(p, **arrays)
    return _rel(p)


def _fit(condition, replicate, round_max, epochs=10):
    """Per-epoch records of one fit; each round's max val_acc is `round_max[i]`, rounds from 5."""
    recs = []
    for i, m in enumerate(round_max):
        r = 5 + i
        for e in range(1, epochs + 1):
            acc = m - 0.0004 * (epochs - e) if e != 7 else m          # epoch 7 attains the round maximum
            acc = min(acc, m)
            loss = 0.25 - 0.1 * (m - 0.5) - 0.0001 * e
            ck = f"stage_r{r:02d}/checkpoint_epoch_{e:03d}.keras"
            recs.append({"condition": condition, "replicate": replicate, "stage_round": r, "stage_epoch": e,
                         "val_acc": round(acc, 6), "val_loss": round(loss, 6), "checkpoint": ck,
                         "checkpoint_sha256": hashlib.sha256(f"{condition}/{replicate}/{ck}".encode()).hexdigest()})
    return recs


def build(root: Path = FIXTURES, seed_offset: int = 0) -> dict:
    """seed_offset != 0 only for the pre-registration smoke root, never for the registered fixtures."""
    global FIXTURES
    FIXTURES = Path(root)
    for k in SEEDS:
        SEEDS[k] += seed_offset
    if FIXTURES.exists():
        raise SystemExit(f"REFUSED: {FIXTURES} exists; fixtures are built once")
    for p, h in EXPECTED_HASHES.items():
        if not sha(REPO / p).startswith(h):
            raise SystemExit(f"REFUSED: {p} hash differs from the design")
    FIXTURES.mkdir()
    rng = np.random.default_rng(SEEDS["partitions"])
    mk = lambda c: (FIXTURES / c).mkdir() or (FIXTURES / c)

    # --- historical and checkpoint cases -------------------------------------------------------------
    for case, ck, checks, c7 in (
            ("H1", H1_CKPT, {}, ("NS", "no published checksum exists for the historical file")),
            ("S2", S2_CKPT, {S2_CKPT: sha(REPO / K3_CKPT)}, "E"),
            ("K3", K3_CKPT, {K3_CKPT: sha(REPO / K3_CKPT)}, "E")):
        _bundle(mk(case), case, files=[ck, RELEASED_SPECK], model=ck, cipher_module=RELEASED_SPECK,
                generator=f"{RELEASED_SPECK}:make_train_data", reported_accuracy=REPORTED_ACCURACY,
                reported_accuracy_source=f"{V5_CERT} observation.findings[0].declared",
                published_architecture=PUBLISHED_ARCHITECTURE, published_checksums=checks,
                b2=_b2(C1="E", C2="E", C3="E", C4=C4_NS, C5="E", C7=c7, C8="E"))

    # --- S1 / K4: cipher modules --------------------------------------------------------------------
    src = (REPO / REFERENCE_SPECK).read_text()
    if src.count("def ALPHA():\n    return(7);") != 1:
        raise SystemExit("REFUSED: reference ALPHA definition not found exactly once")
    d = mk("S1")
    (d / "speck_mutated.py").write_text(src.replace("def ALPHA():\n    return(7);", "def ALPHA():\n    return(8);"))
    mut = _rel(d / "speck_mutated.py")
    _bundle(d, "S1", files=[mut], cipher_module=mut, mutation="ALPHA 7 -> 8 in a copy of " + REFERENCE_SPECK,
            reported_accuracy=REPORTED_ACCURACY, published_checksums={mut: sha(REPO / mut)},
            b2=_b2(C1="E", C2="E", C3="E", C4=C4_NS, C5="E", C7="E", C8=("NS", "no model exists")))
    _bundle(mk("K4"), "K4", files=[REFERENCE_SPECK], cipher_module=REFERENCE_SPECK,
            reported_accuracy=REPORTED_ACCURACY, published_checksums={REFERENCE_SPECK: sha(REPO / REFERENCE_SPECK)},
            b2=_b2(C1="E", C2="E", C3="E", C4=C4_NS, C5="E", C7="E", C8=("NS", "no model exists")))

    # --- S3 / K5: D1 partitions ---------------------------------------------------------------------
    for case, copy in (("S3", 200), ("K5", 0)):
        d = mk(case)
        tr, va, te = _rows(rng, 5000), _rows(rng, 1000), _rows(rng, 1000)
        ytr, yva, yte = (rng.integers(0, 2, size=n, dtype=np.uint8) for n in (5000, 1000, 1000))
        if copy:
            te[:copy], yte[:copy] = tr[:copy], ytr[:copy]
        allrows = np.concatenate([tr, va, te])
        if not copy and len({r.tobytes() for r in allrows}) != len(allrows):
            raise SystemExit("REFUSED: the clean partitions contain a chance duplicate; change the seed")
        p = _np(d, "partitions.npz", train=tr, validation=va, test=te, train_y=ytr, validation_y=yva, test_y=yte)
        _bundle(d, case, files=[p], partitions=p, planted_cross_partition_copies=copy,
                reported_accuracy=REPORTED_ACCURACY, published_checksums={p: sha(REPO / p)},
                b2=_b2(C1="E", C2="E", C3="E", C4=C4_NS, C6="E", C7="E", C8=("NS", "no model exists")))

    # --- S4 / K6: D2-N fixtures ---------------------------------------------------------------------
    for case, planted in (("S4", 400), ("K6", 0)):
        d = mk(case)
        r = np.random.default_rng(SEEDS["d2n_observed"] + (0 if planted else 1))
        train, test = _rows(r, 20000), _rows(r, 20000)
        if planted:
            pr = np.random.default_rng(SEEDS["s4_plant"])
            idx = pr.permutation(20000)
            src_i, dst_i = idx[:planted], idx[planted:2 * planted]
            test[dst_i] = test[src_i]
            test[dst_i, pr.integers(0, 64, size=planted)] ^= 1                 # one bit flipped per copy
        p = _np(d, "partitions.npz", train=train, test=test)
        _bundle(d, case, files=[p], partitions=p, planted_one_bit_near_duplicates=planted,
                d2n={"radius": 8, "queries": 2048, "null_draws": 59, "alpha": 0.05, "partitions_in_family": 3,
                     "alpha_per": 0.05 / 3, "ratio_gate": 1.25, "null_seed_base": SEEDS["d2n_null_base"],
                     "query_seed_base": SEEDS["d2n_query_base"], "null_model": "uniform i.i.d. bits, fresh draw"},
                reported_accuracy=REPORTED_ACCURACY, published_checksums={p: sha(REPO / p)},
                b2=_b2(C1="E", C2="E", C3="E", C4=C4_NS, C6="E", C7="E", C8=("NS", "no model exists")))

    # --- S5 / K8: progression records ---------------------------------------------------------------
    base = [0.9185, 0.7738, 0.6046, 0.5086, 0.5013]
    for case, pert in (("S5", [0.507, 0.503]), ("K8", [0.503])):
        d = mk(case)
        fits = [_fit("perturbed", k, pert) for k in range(1, 6)] + [_fit("baseline", k, base) for k in range(1, 6)]
        p = _write(d, "fits.json", {"rounds": [5, 6, 7, 8, 9], "epochs_per_round": 10, "threshold": 0.505,
                                    "starting_round": 5, "fits": fits})
        _bundle(d, case, files=[p], records=p, reported_accuracy=REPORTED_ACCURACY,
                published_checksums={p: sha(REPO / p)},
                b2=_b2(C1="E", C2="E", C3="E", C4=C4_NS, C7="E", C8=("NS", "no model exists")))

    # --- S6 / K2 / U6: EV per-unit records ----------------------------------------------------------
    a = [0.9201, 0.9188, 0.9195, 0.9210, 0.9179, 0.9204]
    jit = [0.002, -0.002, 0.001, -0.001, 0.0005, -0.0005]
    for case, delta, valid, b2 in (
            ("S6", -0.05, 6, _b2(C1="E", C2="E", C3="E", C4=C4_NS, C7="E", C8=("NS", "no model exists"))),
            ("K2", 0.0, 6, _b2(C1="E", C2="E", C3="E", C4=C4_NS, C7="E", C8=("NS", "no model exists"))),
            ("U6", 0.0, 2, _b2(C1="E", C2="E", C3="E", C7="E"))):
        d = mk(case)
        j = jit if delta else [x / 6.6667 for x in jit]
        units = [{"unit": f"u{i + 1}", "a": a[i], "b": round(a[i] + delta + j[i], 6),
                  "status": "valid" if i < valid else "failed", "failure_reason": None if i < valid else "divergence"}
                 for i in range(6)]
        p = _write(d, "units.json", {"units": units, "epsilon": 0.01, "epsilon_predeclared": True,
                                     "alpha": 0.05, "minimum_valid": 5, "statistics_seed": SEEDS["ev"],
                                     "design": "single declared factor; a = original, b = changed"})
        _bundle(d, case, files=[p], records=p, reported_accuracy=None if case == "U6" else REPORTED_ACCURACY,
                published_checksums={p: sha(REPO / p)}, b2=b2)

    # --- U1: D4 near-chance counts ------------------------------------------------------------------
    d = mk("U1")
    rows = [{"replicate": k, "test_examples": 1_000_000, "perturbed_correct": 505_300, "baseline_correct": 918_000 + k,
             "mcnemar_b": 46_000, "mcnemar_c": 459_000} for k in range(1, 6)]
    p = _write(d, "rows.json", {"rows": rows, "planned_replicates": 5, "chance": 0.5, "margin": 0.005, "alpha": 0.05,
                                "effect_threshold": 0.01, "baseline_threshold": 0.9, "paired_endpoint_role": "secondary"})
    _bundle(d, "U1", files=[p], records=p, published_checksums={p: sha(REPO / p)}, b2=_b2(C1="E", C2="E", C3="E", C7="E"))

    # --- K7 / U4: PAR1 fake-adapter runs ------------------------------------------------------------
    sys.path.insert(0, str(REPO))
    from audit.implementation.ii4_parallel.scheduler import ParallelComparisonRunner, execute_parallel
    from audit.implementation.ii4_parallel.tests import ii4_parallel_fixtures as F
    factory = "audit.implementation.ii4_parallel.tests.ii4_parallel_fixtures:make"
    for case in ("K7", "U4"):
        d = mk(case)
        root = d / "run"
        plan, ad = F.make(root)
        execute_parallel(ParallelComparisonRunner(plan=plan, adapter=ad), gpus=("0", "1"), factory=factory,
                         factory_kwargs={})
        edit = None
        if case == "U4":
            an = json.loads((root / "analysis.json").read_text())
            key = ("blocks", 0, "values", "reference")             # the certified block-0 accuracy
            edit = {"json_path": list(key), "before": _get(an, key)}
            _set(an, key, round(_get(an, key) + 0.01, 6))
            edit["after"] = _get(an, key)
            (root / "analysis.json").write_text(json.dumps(an, indent=2, sort_keys=True))
        files = sorted(_rel(p) for p in root.rglob("*") if p.is_file())
        par = json.loads((root / "parallel_execution.json").read_text())
        _bundle(d, case, files=files, run_dir=_rel(root), edit_after_run=edit,
                published_checksums={_rel(root / "analysis.json"): par["analysis_sha256"]},
                reported_accuracy=REPORTED_ACCURACY if case == "K7" else None,
                b2=_b2(C1="E", C2="E", C3="E", C4=C4_NS if case == "K7" else ("NA", "no reported accuracy"),
                       C7="E", C8=("NS", "fake adapter; no model file") if case == "K7" else ("NA", "no model")))

    # --- engine-only and certificate cases ----------------------------------------------------------
    mk("U2")
    _bundle(FIXTURES / "U2", "U2", files=[A3_CERT], certificate=A3_CERT, published_checksums={A3_CERT: sha(REPO / A3_CERT)},
            b2=_b2(C1="E", C2="E", C7="E", C3=("NA", "a certificate, not a publication")))
    allpass = {"II": "PASS", "DI": "PASS", "EV": "PASS"}
    ce2 = {"id": "CE2", "outcome": "SUPPORTED", "required": False, "prereg": True, "powered": True, "control": False}
    declared = {
        "P1": [{"id": "P1", "required": ["DI", "EV", "II"], "L": 3, "reproduction": "COMPLETE", "predictive": True,
                "dims": allpass, "ce": [dict(ce2, control_from={"certificate": E6_CERT,
                                                                "field": ["interpretation", "ce2_admission"],
                                                                "control_satisfied_unless": "ADMITTED"})]}],
        "U2": [{"id": "U2", "required": ["DI", "EV", "II"], "L": 3, "reproduction": "COMPLETE", "predictive": True,
                "dims": allpass, "ce": [{"id": "CE3", "required": False, "prereg": True, "powered": True,
                                         "outcome_from": {"certificate": A3_CERT,
                                                          "field": ["dimensions", "CE3", "scientific_decision"]},
                                         "control_from": {"certificate": A3_CERT,
                                                          "field": ["dimensions", "CE3", "adversarial_outcome"],
                                                          "control_satisfied_if": "NOT_CAUGHT"}}]}],
        "U3": [{"id": "U3", "required": ["DI"], "L": 3, "reproduction": "COMPLETE", "predictive": True,
                "dims": {"DI": "INCONCLUSIVE", "II": "N/A", "EV": "N/A"}, "ce": [ce2]}],
        "U5": [{"id": "U5", "required": ["EV"], "L": 2, "reproduction": "COMPLETE", "predictive": True,
                "dims": {"II": "PASS", "DI": "PASS"}, "ce": []}],
        "U7": [{"id": "a", "required": ["DI", "EV", "II"], "L": 3, "reproduction": "COMPLETE", "predictive": True,
                "dims": allpass, "ce": [dict(ce2, prereg=False)]},
               {"id": "b", "required": ["DI", "EV", "II"], "L": 3, "reproduction": "COMPLETE", "predictive": True,
                "dims": allpass, "ce": [dict(ce2, powered=False)]}],
    }
    for case in ("P1", "U3", "U5", "U7"):
        d = mk(case)
        p = _write(d, "declared.json", {"asserted_inputs": declared[case]})
        _bundle(d, case, files=[p], declarations_only=True, b2=_b2(C1="E"))
    p = _write(FIXTURES / "U2", "declared.json", {"asserted_inputs": declared["U2"]})
    h2_files = [f"{H2_BUNDLE}/BUNDLE_MANIFEST.json", AD4_CERT, f"{H2_BUNDLE}/dataset/test_y.npy"] + sorted(
        f"{H2_BUNDLE}/fits/predictions/{p.name}" for p in (REPO / H2_BUNDLE / "fits/predictions").glob("*.npy"))
    _bundle(mk("H2"), "H2", files=h2_files, certificate=AD4_CERT,
            evidence_bundle=H2_BUNDLE, published_checksums={AD4_CERT: sha(REPO / AD4_CERT)},
            b2=_b2(C1="E", C2="E", C3="E", C4=("NS", "stored-prediction rescoring is B1-S, not B1"),
                   C5=("NS", "cipher code not allowlisted"), C6=("NS", "partitions not allowlisted"), C7="E",
                   C8=("NS", ".keras checkpoints are not loaded by policy")))
    return {_rel(p): sha(p) for p in sorted(FIXTURES.rglob("*")) if p.is_file()}


def _get(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def _set(obj, path, value):
    for k in path[:-1]:
        obj = obj[k]
    obj[path[-1]] = value


if __name__ == "__main__":
    m = build()
    print(len(m), "fixture files")
