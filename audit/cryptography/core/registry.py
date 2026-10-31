"""
Registry of production routes, framework components and frozen files (machine-readable; tested).

Paths are relative to ``audit/cryptography``. ``tests/test_registry.py`` checks every path, every
frozen-file binding and the usage claims against the source tree, so this file cannot silently drift.

Route status values:
    AUTHORITATIVE        the only route that may produce new evidence for that experiment
    COMPLETED_HISTORICAL the route that produced existing evidence; kept for reproduction only
    LEGACY               superseded; must not produce new evidence
    NOT_RUN              authoritative but not yet executed in production
"""

from __future__ import annotations

ROUTES = {
    "GOHR-CE1": {
        "status": "AUTHORITATIVE",
        "entry": "experiments/ce1/controller.py",
        "evidence": ["evidence_current/ce1/production_20260930"],
        "notes": ("resumable block controller; experiments/ce1/gohr_signal_destruction.py run(production=True) is "
                  "retired (its docstring says so); production complete (certificatece1.json)"),
    },
    "GOHR-CE2": {
        "status": "COMPLETED_HISTORICAL",
        "entry": "ce234/production.py",
        "evidence": ["evidence_current/ce2/production_20261003"],
        "new_production": "ce234/authorised_production.py",
        "notes": "produced the 2026-10-03 evidence; has no approval gate; new production only via the authorised route",
    },
    "GOHR-CE3": {
        "status": "COMPLETED_HISTORICAL",
        "entry": "ce234/production.py",
        "evidence": ["evidence_current/ce3"],
        "new_production": "ce234/authorised_production.py",
        "notes": "resumable CE3; same governance as CE2",
    },
    "GOHR-CE4": {
        "status": "COMPLETED_HISTORICAL",
        "entry": "ce234/production.py",
        "evidence": ["evidence_current/ce4/production_20261004"],
        "new_production": "ce234/authorised_production.py",
        "notes": "produced the 2026-10-04 evidence; same governance as CE2",
    },
    "GOHR-CE2-LEGACY": {
        "status": "LEGACY",
        "entry": "experiments/ce2/gohr_theory_consistency.py",
        "evidence": ["evidence/ce2"],
        "notes": ("its docstring still calls it 'the authoritative production path' and the root README says the "
                  "same; both are stale and manifest-bound (PACKAGE_MANIFEST.sha256). Its CLI cannot produce "
                  "evidence (production mode exits 2 without run results); run(production=True) is programmatic only. "
                  "Superseded by ce234 for CE2-CE4."),
    },
    "GOHR-CE3-LEGACY": {"status": "LEGACY", "entry": "experiments/ce3/gohr_representation_interpretation.py",
                        "evidence": ["evidence/ce3"], "notes": "as GOHR-CE2-LEGACY"},
    "GOHR-CE4-LEGACY": {"status": "LEGACY", "entry": "experiments/ce4/gohr_casual_intervention.py",
                        "evidence": ["evidence/ce4"], "notes": "as GOHR-CE2-LEGACY"},
    "AUTOND-CE1": {
        "status": "COMPLETED_HISTORICAL",
        "entry": "kaggle/autond_ce1_kaggle.py",
        "evidence": ["autond_ce1/evidence/confirmatory_r3"],
        "notes": "completed and immutable; never re-run (no --run-all/--resume/--finalize)",
    },
    "AUTOND-CE2": {
        "status": "NOT_RUN",
        "entry": "kaggle/autond_ce2_kaggle.py",
        "implementation": "experiments/autond/ce2/runner.py",
        "verifier": "experiments/autond/ce2/verify.py",
        "evidence": [],
        "notes": "preregistration r2 PENDING_REVIEW; production refused until approved",
    },
}

#: Framework modules and whether any production route imports them (verified by test).
COMPONENTS = {
    "engine.py": {"used_by_production": False, "decision": "RETAINED_AS_EXTENSION_INTERFACE",
                  "notes": "CryptographicEngine has no importer; no production route uses it"},
    "reporting.py": {"used_by_production": False, "decision": "RETAINED_UNUSED"},
    "utils.py": {"used_by_production": False, "decision": "RETAINED_UNUSED"},
    "evaluation.py": {"used_by_production": False, "decision": "RETAINED_TESTS_ONLY"},
    "adapters/gohr.py": {"used_by_production": False, "decision": "RETAINED_HASH_BOUND",
                         "notes": "no importer, but hash-bound by the CE2-CE4 certificates' source_hashes"},
    "test/ce1/signal_destruction.py": {"used_by_production": False, "decision": "RETAINED_LEGACY_DEFINITION"},
    "test/ce2/theory_consistency.py": {"used_by_production": False, "decision": "SUPERSEDED_BY_CORE",
                                       "notes": ("encodes a positive hypothesis and maps undefined to 0.0; "
                                                 "superseded by core/theory_consistency.py")},
    "adapters/base.py": {"used_by_production": True, "decision": "RETAINED_HASH_BOUND",
                         "notes": "CryptographicAdapter: used by the AutoND CE1 adapter; capability protocols in core/"},
}

#: Files that must not be edited in place, and what binds them.
FROZEN = {
    "ce234/frozen_plan.py": ["CE234_MANIFEST v1", "CE2-CE4 certificates", "A3 adversarial source binding"],
    "ce234/production.py": ["CE234_MANIFEST v1", "CE2-CE4 certificates", "A3 adversarial source binding"],
    "ce234/verify.py": ["CE234_MANIFEST v1", "CE2-CE4 certificates", "A3 adversarial source binding"],
    "ce234/build_matrices.py": ["CE234_MANIFEST v1", "CE2-CE4 certificates"],
    "ce234/tests/test_ce234.py": ["CE234_MANIFEST v1", "CE2-CE4 certificates"],
    "gohr/speck.py": ["PACKAGE_MANIFEST", "A3 adversarial source binding"],
    "gohr/dataset.py": ["PACKAGE_MANIFEST", "CE2-CE4 certificates", "A3 adversarial source binding"],
    "gohr/model.py": ["PACKAGE_MANIFEST", "CE2-CE4 certificates", "A3 adversarial source binding"],
    "gohr/trainer.py": ["PACKAGE_MANIFEST", "A3 adversarial source binding"],
    "gohr/evaluate.py": ["PACKAGE_MANIFEST", "CE2-CE4 certificates", "A3 adversarial source binding"],
    "probe/evaluation.py": ["PACKAGE_MANIFEST", "CE2-CE4 certificates", "A3 adversarial source binding"],
    "experiments/ce4/design.py": ["PACKAGE_MANIFEST", "CE2-CE4 certificates", "A3 adversarial source binding"],
    "statistics.py": ["PACKAGE_MANIFEST", "A3 adversarial source binding"],
    "adapters/base.py": ["PACKAGE_MANIFEST", "AutoND CE1 package SHA256SUMS"],
    "frozen_design.py": ["PACKAGE_MANIFEST", "AutoND CE1 package SHA256SUMS"],
    "experiments/ce1/design.py": ["PACKAGE_MANIFEST", "AutoND CE1 package SHA256SUMS"],
    "experiments/ce1/controller.py": ["PACKAGE_MANIFEST", "Gohr CE1 production (do not alter)"],
    "README.md": ["PACKAGE_MANIFEST"],
    "test/ce2/theory_consistency.py": ["PACKAGE_MANIFEST"],
    "results.py": ["PACKAGE_MANIFEST"],
    "test/base.py": ["PACKAGE_MANIFEST"],
    "upstream_sha256.json": ["AutoND CE1 package SHA256SUMS"],
    "adapters/autond.py": ["AutoND CE1 package SHA256SUMS"],
    "Archive/best5depth10.h5": ["PACKAGE_MANIFEST", "CE2-CE4 plan REFERENCE_CHECKPOINT_SHA256"],
}

#: The A3 adversarial freeze re-hashes exactly these modules at run time (audit/adversarial/battery.py).
A3_BOUND_MODULES = (
    "audit.cryptography.ce234.frozen_plan", "audit.cryptography.ce234.production", "audit.cryptography.ce234.verify",
    "audit.cryptography.gohr.speck", "audit.cryptography.gohr.dataset", "audit.cryptography.gohr.model",
    "audit.cryptography.gohr.trainer", "audit.cryptography.gohr.evaluate", "audit.cryptography.probe.evaluation",
    "audit.cryptography.experiments.ce4.design", "audit.cryptography.statistics",
)
