from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

MODES = ("check_model", "p1", "verify_p1", "power", "benchmark", "representation_diagnostics", "preflight",
         "render_preregistration", "smoke", "run_all", "resume", "run", "finalize", "verify")


def _parse(argv):
    ap = argparse.ArgumentParser(description="AutoND CE3 representation decodability")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check-model", action="store_true")
    g.add_argument("--p1", action="store_true")
    g.add_argument("--verify-p1", action="store_true")
    g.add_argument("--power", action="store_true")
    g.add_argument("--benchmark", action="store_true")
    g.add_argument("--representation-diagnostics", action="store_true")
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--render-preregistration", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--run-all", action="store_true")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--run", type=int, metavar="K")
    g.add_argument("--finalize", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--upstream", type=Path, default=None)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-model-recompute", action="store_true")
    ap.add_argument("--sklearn-refit", action="store_true")
    a = ap.parse_args(argv)
    needs_out = a.smoke or a.run_all or a.resume or a.run is not None or a.finalize or a.verify or a.p1
    if needs_out and a.out is None:
        ap.error("this mode requires --out")
    if not needs_out and a.out is not None:
        ap.error("--out is only valid for p1/smoke/run-all/resume/run/finalize/verify")
    if a.report is not None and not (a.check_model or a.verify_p1 or a.power or a.benchmark or a.preflight
                                     or a.render_preregistration or a.representation_diagnostics):
        ap.error("--report is only valid for check-model/verify-p1/power/benchmark/representation-diagnostics/"
                 "preflight/render-preregistration")
    if (a.power or a.benchmark or a.render_preregistration or a.representation_diagnostics) and a.report is None:
        ap.error("this mode requires --report")
    if (a.skip_model_recompute or a.sklearn_refit) and not a.verify:
        ap.error("--skip-model-recompute and --sklearn-refit are only valid with --verify")
    return a


def _emit(obj, report: Path | None):
    text = json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n"
    if report is None:
        print(text)
        return
    if report.exists():
        raise SystemExit(f"[FATAL] refusing to overwrite {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text)
    print(f"wrote {report}")


def main(argv=None) -> int:
    a = _parse(argv)
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    root = (a.repo_root or Path.cwd()).resolve()
    if not (root / "audit" / "cryptography").is_dir():
        print(f"[FATAL] no audit/cryptography package under {root}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(root))

    if a.p1:
        from audit.cryptography.experiments.autond.ce3 import p1 as P1
        rec = P1.generate(a.out, upstream=a.upstream)
        print(f"P1: K={rec['selected_n_classes']} edges={rec['selected_edges']} -> {a.out}")
        return 0
    if a.power:
        from audit.cryptography.experiments.autond.ce3 import power as PW
        from audit.cryptography.experiments.autond.ce3 import design as D
        rec = json.loads((root / "audit" / "cryptography" / D.P1_RECORD_RELPATH).read_text())
        rep = PW.run(rec["selection"]["selected"]["class_fractions"],
                     reps_per_n={50_000: 48, 100_000: 48, 200_000: 48}, workers=a.workers)
        rep["utc"] = datetime.now(timezone.utc).isoformat()
        _emit(rep, a.report)
        return 0

    from audit.cryptography.experiments.autond.ce3 import adapter as A
    from audit.cryptography.experiments.autond.ce3 import design as D
    from audit.cryptography.experiments.autond.ce3 import runner as R
    from audit.cryptography.experiments.autond.ce3 import verify as V
    from audit.cryptography.targets.autond import model_binding as MB
    from audit.cryptography.targets.autond import upstream as UP

    if a.check_model:
        up = UP.load(a.upstream)
        model, twin, binding = A.load_models(up, R.CRYPTO_ROOT / D.CE2_RUN_RELPATH / D.CE2_TWIN_RELPATH)
        g = A.main_dataset(up, 256)["arrays"]
        X = A.model_input(up, g["c0"], g["c1"])
        _emit({"report": "AutoND CE3 read-only model and representation binding check", "utc": R.utc(),
               "environment": R.environment(), "runtime_problems": R.runtime_problems(R.environment()),
               "model_binding": binding, "representation": {"published": A.bind_representation(model),
                                                            "twin": A.bind_representation(twin)},
               "inference_mode": MB.inference_mode_check(model, X),
               "head_max_abs_diff": A.head_difference(model, A.extract(A.extractor(model), X), MB.predict(model, X)),
               "not_performed": ["training", "saving weights", "writing evidence"]}, a.report)
        return 0
    if a.verify_p1:
        f = V.Findings()
        V.verify_p1(D.design(), f)
        _emit({"report": "AutoND CE3 independent P1 verification", "utc": R.utc(), "checks": f.checks,
               "failures": f.failures, "passed": not f.failures}, a.report)
        return 0 if not f.failures else 1
    if a.benchmark:
        from audit.cryptography.experiments.autond.ce3 import benchmark as BM
        rec = BM.run(a.report, upstream=a.upstream)
        print(f"benchmark: all_converged={rec['all_converged']} selected_max_iter={rec['selected_max_iter']} "
              f"({rec['seconds_total']:.0f} s) -> {a.report}")
        return 0 if rec["all_converged"] else 1
    if a.representation_diagnostics:
        from audit.cryptography.experiments.autond.ce3 import diagnostics as DG
        rec = DG.run(a.report, upstream=a.upstream)
        for role, r in rec["representations"].items():
            print(f"{role}: nonconstant={r['units_nonconstant']} "
                  f"effective_rank={r['centred']['effective_rank_entropy']:.2f} "
                  f"participation_ratio={r['centred']['participation_ratio']:.2f}")
        print(f"-> {a.report}")
        return 0
    if a.preflight:
        rep = R.preflight(UP.load(a.upstream), "production")
        _emit(rep, a.report)
        return 0 if rep["ok"] else 1
    if a.render_preregistration:
        _emit(D.render_preregistration(), a.report)
        print(f"pending items: {D.pending_items()}")
        return 0
    if a.smoke:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = a.out if a.out.name.startswith(R.SMOKE_PREFIX) else a.out / f"{R.SMOKE_PREFIX}{stamp}"
        man = R.initialise(out, "smoke", upstream=a.upstream)
        if man["status"] == "INITIALISED":
            R.run_all(out, upstream=a.upstream)
        cert = R.finalize(out)
        print(f"SMOKE_NOT_CONFIRMATORY decision={cert['decision']} -> {out}")
        return V.main([str(out)] + (["--upstream", str(a.upstream)] if a.upstream else []))
    if a.run_all:
        if (a.out / "run_manifest.json").exists():
            print("[FATAL] output directory already initialised; use --resume", file=sys.stderr)
            return 2
        man = R.initialise(a.out, "production", upstream=a.upstream)
        if man["status"] != "INITIALISED":
            print(f"[STOP] initialisation status {man['status']}; finalize to record PRECHECK_FAILED")
            return 1
        recs = R.run_all(a.out, upstream=a.upstream)
    elif a.resume:
        recs = R.run_all(a.out, upstream=a.upstream)
    elif a.run is not None:
        recs = R.run_all(a.out, upstream=a.upstream, only=a.run)
    elif a.finalize:
        cert = R.finalize(a.out)
        print(f"decision={cert['decision']}: {cert['decision_reason']}")
        return 0
    else:
        return V.main([str(a.out)] + (["--upstream", str(a.upstream)] if a.upstream else []) +
                      (["--skip-model-recompute"] if a.skip_model_recompute else []) +
                      (["--sklearn-refit"] if a.sklearn_refit else []))
    for r in recs:
        print(f"{r['run_id']}: {r['state']}" + (f" ({r.get('failure') or r.get('reason')})"
                                               if r["state"] in ("FAILED", "EXCLUDED") else ""))
    return 0 if all(r["state"] in ("VALID", "EXCLUDED") for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main())
