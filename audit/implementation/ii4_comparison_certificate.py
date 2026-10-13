#!/usr/bin/env python3
"""
Comparison run -> Stage II-4 / II-5 components -> Implementation
Integrity certificate. GENERIC.

Runs no experiment and recomputes no statistic. Every number it reports
is READ from the run artifact produced at execution time, and every
verdict is read from the interpretation the run recorded under its
preregistered rule - never recomputed from a code constant, so that a
post-hoc edit to the interpretation rule cannot re-interpret a finished
run.

WHAT THIS MODULE DECIDES, AND WHAT IT REFUSES TO DECIDE
-------------------------------------------------------
It decides the STATUS of the II-4 and II-5 stage records, from two
inputs and nothing else:

    the axis kind of the run   (can this run satisfy Stage II-4 at all?)
    the run's own validity     (did enough blocks survive?)

It does NOT decide the pillar verdict. That is `certificate.py`'s job,
driven by `stages.validate_stage_record` and the CORE-stage rule. If a
conformance-axis run is handed to this module it produces a
CONFORMANCE_IMPACT_EXPERIMENT component and leaves II-4 PARTIAL,
exactly as the existing conformance certificate does - the axis decides,
not the caller's intention.

HISTORICAL ARTIFACTS ARE NEVER MODIFIED. The version registry is
append-only; a new certificate receives the next version number;
earlier certificates and the completed conformance experiment's bound
artifacts are read-only inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from audit.common.provenance import sha256_file
from audit.implementation.comparison import AxisKind
from audit.implementation.stages import (
    STAGE_TIERS,
    ComponentKind,
    StageComponent,
    StageId,
    StageOutcome,
    StageStatus,
)

CONVERTER_VERSION = "ii4-comparison-certificate-v1"


class ComparisonCertificateError(RuntimeError):
    """The run artifact cannot support a stage record."""


def _require(obj: dict, key: str, where: str) -> Any:
    if key not in obj:
        raise ComparisonCertificateError(
            f"{where}: required field {key!r} is absent from the run artifact. Refusing to "
            "assemble a stage record from an incomplete run.")
    return obj[key]


def load_run(run_dir: Path) -> dict[str, Any]:
    """Load and hash-record the run artifact."""
    path = Path(run_dir) / "analysis.json"
    if not path.is_file():
        raise ComparisonCertificateError(
            f"no analysis.json in {run_dir}: the run did not complete, and a stage record may "
            "not be assembled from a run that produced no artifact.")
    report = json.loads(path.read_text())
    report["_artifact_path"] = str(path)
    report["_artifact_sha256"] = sha256_file(path)
    return report


def _evidence_map(run_dir: Path, report: dict) -> dict[str, str]:
    """Path -> sha256 for every artifact the component binds."""
    evidence = {report["_artifact_path"]: report["_artifact_sha256"]}
    for name in ("preregistration.json", "state.json", "firewall.json"):
        p = Path(run_dir) / name
        if p.is_file():
            evidence[str(p)] = sha256_file(p)
    for label, rec in (report.get("source_manifest") or {}).items():
        evidence[f"source:{rec['path']}"] = rec["sha256"]
    return evidence


def read_analysis(analysis: dict) -> dict[str, Any]:
    """
    Flat view of the frozen analysis artifact.

    `ii4_analysis.II4AnalysisResult.to_dict` nests its output under
    `confirmatory`, `replication` and `secondary_robustness`. Reading
    those keys off the top level silently yields None, which would turn
    a real verdict into a blank field in a certificate. The accessor is
    written once, here, and pinned by a test against a real artifact so
    that a change to the analysis schema fails loudly instead of
    emptying a certificate.
    """
    conf = analysis.get("confirmatory", {})
    rep = analysis.get("replication", {})
    sec = analysis.get("secondary_robustness", {})
    missing = [k for k in ("confirmatory", "replication") if k not in analysis]
    if missing:
        raise ComparisonCertificateError(
            f"analysis artifact is missing section(s) {missing}; its schema is not the frozen "
            "ii4_analysis schema and a stage record must not be assembled from it.")
    ci95 = conf.get("ci95") or {}
    return {
        "n_blocks_valid": rep.get("n_blocks_valid"),
        "n_blocks_requested": rep.get("n_blocks_requested"),
        "min_valid_required": rep.get("min_valid_required"),
        "sufficient": rep.get("sufficient"),
        "mean_difference": conf.get("mean_difference"),
        "sd_difference": conf.get("sd_difference"),
        "ci95_low": ci95.get("low"), "ci95_high": ci95.get("high"),
        "materiality": (conf.get("materiality") or {}).get("verdict"),
        "equivalence": (conf.get("equivalence") or {}).get("verdict"),
        "tost_p": (conf.get("equivalence") or {}).get("p_tost"),
        "paired_t_p": (conf.get("paired_t") or {}).get("p_value"),
        "sign_flip_p": sec.get("sign_flip_p"),
        "sign_flip_feasible": sec.get("sign_flip_feasible"),
        "warnings": list(analysis.get("warnings", [])),
    }


def _format_summary(report: dict) -> str:
    a = read_analysis(report["analysis"])
    valid = report.get("n_blocks_valid", a["n_blocks_valid"])
    mean = a["mean_difference"]
    lo, hi = a["ci95_low"], a["ci95_high"]
    bits = [f"{valid} valid block(s)"]
    if mean is not None:
        bits.append(f"mean signed difference {mean:.6g}")
    if lo is not None and hi is not None:
        bits.append(f"95% CI [{lo:.6g}, {hi:.6g}]")
    bits.append(f"materiality = {a['materiality']}")
    bits.append(f"equivalence = {a['equivalence']}")
    if report.get("interpretation"):
        bits.append(f"preregistered conclusion: {report['interpretation']['conclusion']}")
    return "; ".join(bits) + "."


def _refuse_non_evidentiary(report: dict) -> None:
    """
    A non-evidentiary run may never license a stage record.

    A SMOKE or PILOT run exercises every code path and writes an artifact
    of exactly the same shape as a production run. That is what makes it
    useful, and it is also what makes it dangerous: nothing in the
    artifact's STRUCTURE distinguishes it, so the distinction must be
    enforced on its recorded execution mode. Without this guard a
    two-block, two-epoch rehearsal would assemble a CORE stage record
    reading COMPLETED - the precise failure mode ("no smoke setting may
    leak into evidence") that the production plan validation exists to
    prevent on the way in.
    """
    mode = report.get("execution_mode")
    if report.get("non_evidentiary") or mode != "PRODUCTION":
        raise ComparisonCertificateError(
            f"run {report.get('experiment_id')!r} has execution_mode={mode!r} "
            f"(non_evidentiary={report.get('non_evidentiary')}). A non-production run may not "
            "license a stage record, however many blocks it completed. Use "
            "`allow_non_evidentiary=True` only to inspect what a rehearsal WOULD produce; the "
            "components it returns are then marked NOT_EVIDENCE.")


def build_components(run_dir: Path, report: Optional[dict] = None,
                     allow_non_evidentiary: bool = False,
                     ) -> tuple[StageComponent, StageComponent]:
    """
    Build the II-4 and II-5 components this run supports.

    A component is COMPLETED only if the run reached min_valid_blocks.
    A run that fell short is recorded PARTIAL with its shortfall stated,
    never omitted - an experiment that ran and did not resolve is
    evidence about the design, and deleting it would bias the record.
    """
    run_dir = Path(run_dir)
    report = report or load_run(run_dir)
    if not allow_non_evidentiary:
        _refuse_non_evidentiary(report)
    axis = _require(report, "comparison_axis", "run artifact")
    analysis = _require(report, "analysis", "run artifact")
    prereg = (run_dir / "preregistration.json")
    if not prereg.is_file():
        raise ComparisonCertificateError(
            f"no preregistration.json in {run_dir}: a confirmatory stage record requires the "
            "pre-execution plan that governed the run.")
    design = json.loads(prereg.read_text())["design"]

    kind_name = axis["component_kind"]
    kind = ComponentKind(kind_name)
    valid = report.get("n_blocks_valid", 0)
    sufficient = valid >= design["min_valid_blocks"]
    status = StageOutcome.COMPLETED if sufficient else StageOutcome.PARTIAL

    limitations = list(report.get("recorded_deviations", []))
    if report.get("non_evidentiary") or report.get("execution_mode") != "PRODUCTION":
        limitations.insert(0, (
            f"NOT EVIDENCE: this run has execution_mode="
            f"{report.get('execution_mode')!r} and is non-evidentiary. It is reproduced here "
            "for inspection only and must not appear in a certificate."))
    limitations.append(
        "Inference is conditional on the single sealed confirmatory test set; generalisation "
        "beyond it is not claimed.")
    for name, basis in (report.get("shared_components") or {}).items():
        limitations.append(
            f"{name} is SHARED by both arms and is therefore not a compared component: {basis}")
    if report.get("replacements"):
        limitations.append(
            f"{len(report['replacements'])} block(s) were replaced under the preregistered "
            "technical-failure rule; the failed blocks are retained in the artifact.")
    if not sufficient:
        limitations.append(
            f"Only {valid} of the required {design['min_valid_blocks']} valid blocks were "
            "obtained; the preregistered rule returns INCONCLUSIVE and this component cannot "
            "complete the stage.")
    flat = read_analysis(analysis)
    if flat["sign_flip_feasible"] is False:
        limitations.append(
            "The exact sign-flip secondary could not reach alpha at the realized block count "
            "and is reported as infeasible, not as a non-significant result.")
    if axis["kind"] != AxisKind.IMPLEMENTATION.value:
        limitations.append(
            "This run varies a parameter within ONE implementation. It cannot satisfy Stage "
            "II-4 or II-5 however many blocks completed.")
    limitations.extend(flat["warnings"])

    evidence = _evidence_map(run_dir, report)
    summary = _format_summary(report)
    experiment_id = _require(report, "experiment_id", "run artifact")

    ii4 = StageComponent(
        component_id=experiment_id, kind=kind, status=status,
        question=axis["scientific_question"], summary=summary,
        evidence=evidence, limitations=list(limitations))
    ii5 = StageComponent(
        component_id=f"{experiment_id}-statistics", kind=kind, status=status,
        question=f"Statistical assessment of {experiment_id}.",
        summary=summary, evidence=evidence, limitations=list(limitations))
    return ii4, ii5


def build_stage_records(run_dir: Path, report: Optional[dict] = None,
                        extra_components: Optional[dict[str, list]] = None,
                        allow_non_evidentiary: bool = False,
                        ) -> tuple[StageStatus, StageStatus]:
    """
    Assemble the II-4 and II-5 StageStatus records.

    The stage status is the strongest status its components license, and
    for II-4/II-5 only a COMPLETED FORMAL_IMPLEMENTATION_COMPARISON
    licenses COMPLETED - which `stages.validate_stage_record` enforces
    again downstream, so an error here fails the certificate build
    rather than producing an overclaiming artifact.
    """
    report = report or load_run(Path(run_dir))
    ii4_comp, ii5_comp = build_components(run_dir, report,
                                          allow_non_evidentiary=allow_non_evidentiary)
    extra = extra_components or {}
    ii4_components = tuple(extra.get("II-4", [])) + (ii4_comp,)
    ii5_components = tuple(extra.get("II-5", [])) + (ii5_comp,)

    def _stage_status(components) -> StageOutcome:
        formal_done = any(c.kind is ComponentKind.FORMAL_IMPLEMENTATION_COMPARISON
                          and c.status is StageOutcome.COMPLETED for c in components)
        return StageOutcome.COMPLETED if formal_done else StageOutcome.PARTIAL

    axis = report["comparison_axis"]
    conclusion = (report.get("interpretation") or {}).get("conclusion")
    completed = _stage_status(ii4_components) is StageOutcome.COMPLETED

    ii4_summary = (
        f"{'COMPLETED' if completed else 'PARTIAL'}. "
        f"{axis['name']} comparison ({axis['kind']}): {_format_summary(report)} "
        + ("" if completed else
           "The formal comparison of the reference and independent implementations under "
           "identical experimental protocols is not completed by this run."))
    ii5_summary = (
        f"{'COMPLETED' if completed else 'PARTIAL'}. "
        f"{_format_summary(report)}"
        + (f" Preregistered conclusion: {conclusion}." if conclusion else ""))

    ii4 = StageStatus(
        stage_id=StageId.II_4_COMPARATIVE_EVALUATION,
        tier=STAGE_TIERS[StageId.II_4_COMPARATIVE_EVALUATION],
        status=_stage_status(ii4_components), summary=ii4_summary,
        components=ii4_components)
    ii5 = StageStatus(
        stage_id=StageId.II_5_STATISTICAL_ASSESSMENT,
        tier=STAGE_TIERS[StageId.II_5_STATISTICAL_ASSESSMENT],
        status=_stage_status(ii5_components), summary=ii5_summary,
        components=ii5_components)
    return ii4, ii5


def conclusion_of(report: dict) -> Optional[str]:
    """The run's own preregistered conclusion. Never recomputed here."""
    return (report.get("interpretation") or {}).get("conclusion")
