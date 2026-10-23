"""Post-run integrity verification of a Gohr D3 confirmatory output directory (read-only).

Re-derives, from the files on disk and the current code, everything the D3
certificate claims, as far as the engine's interfaces permit:

  * the output directory holds exactly the certificate, sealed audit and run
    manifest; the manifest's file hashes and the seal agree everywhere;
  * the preregistration is approved, committed and matches the anchor, and the
    certificate names that commit;
  * the calibration verifies for the Gohr tool and bound suite, and the
    certificate names that calibration run and evidence;
  * the D0 instance re-verifies (every file and array hash), its build
    attestation verifies against the frozen specification, and the certificate
    names that instance, manifest and attestation;
  * the instance-bound D1 evidence still satisfies the frozen D1 gate (complete,
    bound to this instance, outcome PASS) and is the evidence the certificate
    and run manifest name;
  * the code the certificate records is the code on disk now;
  * every family's categories are re-derived from its records, every gating
    statistic is recomputed from its sufficient statistics (engine
    _reclassify / _recompute_mismatches), and the family, provisional and
    reported outcomes follow from them.

Every check is guarded: an exception or a refusal (SystemExit) is a FAIL,
never a pass. Exit status 0 only if every check passes. Writes nothing.

USAGE
    python -B -m audit.dataset.tools.d3_gohr_postrun_verify \\
        OUT_DIR INSTANCE_DIR D1_DIR PREREGISTRATION PREREGISTRATION_SHA256 CALIBRATION_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from audit.dataset import canonical as C
from audit.dataset import d3_distribution_statistics as d3
from audit.dataset.experiments import d3_gohr_confirmatory as R
from audit.dataset.tools import d0_gohr_attested_build as ATT


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(out: Path, inst: Path, d1_dir: Path, prereg: str, anchor: str, cal_dir: str, *, log=print) -> bool:
    results: list = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except (Exception, SystemExit) as exc:                     # noqa: BLE001  (runner gates raise SystemExit)
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(bool(ok))
        log(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + str(detail)) if detail else ''}")
        return ok

    state: dict = {}

    def files():
        names = sorted(p.name for p in Path(out).iterdir())
        state["cert"] = json.loads((out / R.CERTIFICATE_NAME).read_text())
        state["rman"] = json.loads((out / R.RUN_MANIFEST_NAME).read_text())
        state["sealed"] = (out / R.AUDIT_FILE_NAME).read_bytes()
        state["seal"] = hashlib.sha256(state["sealed"]).hexdigest()
        return names == sorted([R.CERTIFICATE_NAME, R.AUDIT_FILE_NAME, R.RUN_MANIFEST_NAME]), names
    if not check("output files are exactly certificate, sealed audit, run manifest", files):
        log("SOME CHECKS FAILED — do not use this output")
        return False
    cert, rman, seal = state["cert"], state["rman"], state["seal"]
    dec = cert["decision"]

    check("run manifest COMPLETE and its file hashes match the files",
          lambda: (rman.get("run_status") == "COMPLETE" and rman["files"] == {
              R.CERTIFICATE_NAME: _sha(out / R.CERTIFICATE_NAME), R.AUDIT_FILE_NAME: seal}, rman.get("run_status")))
    check("seal agrees: sealed file = manifest = certificate chain = decision",
          lambda: (seal == rman["audit_seal_sha256"] == cert["chain"]["audit_seal_sha256"] == dec["audit_seal_sha256"],
                   seal))
    check("run id agrees (manifest = certificate)",
          lambda: (rman["run_id"] == cert["run_identity"]["run_id"], rman["run_id"]))

    def prereg_ok():
        b = d3.verify_preregistration(prereg, expected_sha256=anchor)     # approval, commit, anchor, timing
        content = d3.validate_preregistration_content(b.record())
        R.check_protocol(content, d3.verify_calibration_evidence(
            cal_dir, tool_path=R.CALIBRATION_TOOL_PATH, test_suite_path=R.CALIBRATION_SUITE_PATH).design())
        ok = (b.record()["approved"] is True and cert["chain"]["preregistration"]["sha256"] == anchor
              == rman["preregistration_sha256"] and cert["chain"]["preregistration"]["commit"] == b.commit
              and dec["preregistration"]["sha256"] == anchor)
        return ok, f"commit {b.commit} approved_utc {content['approved_utc']}"
    check("preregistration approved, committed, anchored; protocol exact; named by the certificate", prereg_ok)

    def cal_ok():
        cal = d3.verify_calibration_evidence(cal_dir, tool_path=R.CALIBRATION_TOOL_PATH,
                                             test_suite_path=R.CALIBRATION_SUITE_PATH)
        c = cert["chain"]["calibration"]
        return (c["run_id"] == cal.run_id == rman["calibration_run_id"] and c["evidence_sha256"] == cal.evidence_sha256
                and c["engine_sha256"] == cal.engine_sha256 and c["tool_sha256"] == cal.tool_sha256
                and c["test_suite_sha256"] == cal.test_suite_sha256), cal.run_id
    check("calibration verifies for the Gohr tool and bound suite; named by the certificate", cal_ok)

    def d0_ok():
        man = C.load_manifest(inst)
        expected = dict(R.EXPECTED_SPEC, **{f"{n}_samples": R.PRODUCTION_SIZES[n] for n in C.PARTITIONS})
        att = ATT.verify_build_attestation(inst, expected_spec=expected)
        ver = C.verify(inst, log=lambda *a, **k: None)
        d = cert["chain"]["d0_instance"]
        state["man"] = man
        return (ver.get("verified") is True and d["canonical_instance_id"] == man["canonical_instance_id"]
                == rman["canonical_instance_id"] and d["manifest_sha256"] == _sha(Path(inst) / C.MANIFEST_NAME)
                and cert["chain"]["build_attestation"]["sha256"] == att["sha256"] == rman["build_attestation_sha256"]
                ), man["canonical_instance_id"]
    check("D0 instance re-verified (all hashes), attestation verified, both named by the certificate", d0_ok)

    def d1_ok():
        d1 = R.verify_d1_evidence(d1_dir, inst)
        return (cert["chain"]["d1_census"] == d1 and rman["d1_certificate_sha256"] == d1["certificate_sha256"]), \
            f"D1 {d1['outcome']} run {d1['run_id']} certificate {d1['certificate_sha256']}"
    check("D1 evidence complete, bound to this instance, outcome PASS; named by certificate and manifest", d1_ok)

    def prov_ok():
        p = json.loads(state["sealed"])
        state["payload"] = p
        man = state["man"]
        expected = {n: {"features_sha256": man["partitions"][n]["features_sha256"],
                        "labels_sha256": man["partitions"][n]["labels_sha256"]} for n in C.PARTITIONS}
        return p["input_provenance"] == cert["loaded_partition_hashes"] == expected, ""
    check("sealed input provenance = loaded hashes = D0 manifest", prov_ok)

    def code_ok():
        now = R._code_identity()
        rec = {k: v for k, v in cert["code_identity"].items() if k != "git"}
        return (rec == now and state["payload"]["engine_sha256"] == d3.engine_sha256()
                == now["engine"]["sha256"]), d3.engine_sha256()
    check("recorded code identity = code on disk (engine, runner, tools, suite, generator)", code_ok)

    fam_out: dict = {}
    for f in state.get("payload", {}).get("families", []):
        def fam_ok(f=f):
            cats = d3._reclassify(f)
            mism = d3._recompute_mismatches(f)
            derived = (d3._combine_outcomes([d3.CATEGORY_OUTCOME[c] for c in cats])
                       if f["role"] == "confirmatory" else "NOT_GATED")
            fam_out[f["family_id"]] = derived
            return (cats == d3._recorded_categories(f) and not mism
                    and derived == f["outcome"] == dec["family_outcomes"][f["family_id"]]), \
                f"outcome {derived}, recompute mismatches {len(mism)}"
        check(f"family {f['family_id']}: categories re-derived, statistics recomputed, outcome", fam_ok)

    def decision_ok():
        prov = d3._combine_outcomes([fam_out[f] for f in R.CONFIRMATORY_FAMILIES])
        return (prov == dec["provisional_outcome"]
                and dec["outcome"] == (prov if not dec["outcome_overridden_because"] else "INCONCLUSIVE")
                and set(fam_out) == set(dec["family_outcomes"])), \
            f"outcome {dec['outcome']} overrides {dec['outcome_overridden_because']}"
    check("provisional and reported outcome re-derived", decision_ok)

    log(f"\nRUN {rman.get('run_id')}  OUTCOME {dec.get('outcome')}  FAMILIES {dec.get('family_outcomes')}")
    ok = bool(results) and all(results)
    log("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED — do not use this output")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Post-run integrity verification of a Gohr D3 output directory.")
    for a in ("out_dir", "instance_dir", "d1_dir", "preregistration", "preregistration_sha256", "calibration_dir"):
        p.add_argument(a)
    a = p.parse_args(argv)
    return 0 if verify(Path(a.out_dir), Path(a.instance_dir), Path(a.d1_dir), a.preregistration,
                       a.preregistration_sha256, a.calibration_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
