#!/usr/bin/env python3
"""
AutoND II-1 and II-3 stage records, built through the generic framework.

THIS ADDS NO AUDIT MACHINERY. Every type, status value, component kind
and validation rule comes from `audit.implementation.stages`. What lives
here is AutoND-specific and nothing else: which artifacts constitute the
evidence, what each one is expected to hash to, and how a set of verified
artifacts maps onto the framework's existing vocabulary. If a rule about
what makes a stage satisfied appears in this file, it is in the wrong
file.

VALIDATION BEFORE EXECUTION. The default mode is `validate`, which opens
and hashes evidence already on disk and runs nothing. Re-running the
search or the training is neither required nor offered here: both have
completed, their outputs are preserved, and a stage record is built from
those outputs. `run_search.py` in ./search/ remains the only way to
execute a search, and it is never invoked from this module.

FAIL CLOSED, WITH ONE DISTINCTION KEPT. A missing file, a hash mismatch,
a wrong upstream commit or malformed required metadata is an INTEGRITY
failure and the component FAILS. A result that disagrees with the
publication is NOT an integrity failure: it is a scientific observation,
recorded with its discrepancy intact and left for adjudication against
the stage criteria. The two are never merged, because a reproduction that
honestly fails to reproduce is still a sound reproduction.

RECORDED VERSUS VERIFIED. Hashes written into `EXPECTED` below were
reported from the operator's machine and have NOT been confirmed by the
author of this module, who has no access to that filesystem. They are
`RECORDED`. Running this module on the operator's machine is what turns
them into `VERIFIED`; until then every stage record says so in its own
limitations, and `--strict` refuses to emit a record containing any
unverified required artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from audit.implementation.stages import (
    SATISFYING_OUTCOMES, STAGE_TIERS, ComponentKind, StageComponent, StageId,
    StageOutcome, StageStatus, validate_stage_record,
)

#: The framework's own notion of a satisfied outcome. Imported rather than
#: restated: a local tuple would be duplicate audit logic and would drift
#: the moment the framework changed its mind.
SATISFYING = SATISFYING_OUTCOMES
from audit.targets.autond import paper_source_architecture as psa
from audit.targets.autond import evaluation_results as evres

ADAPTER_VERSION = "autond-implementation-stages-v2"

try:                        # the pinned commit belongs to the Phase 1 package
    from audit.targets.autond import PINNED_UPSTREAM_COMMIT as PINNED_COMMIT
except Exception:                                                 # noqa: BLE001
    PINNED_COMMIT = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"

SPECK_CHECKPOINT_SHA = "a7be788dbbb760a7be9053748e6e0f37dedc5768db3a74a5c3ebd294dca4cc26"
SIMON_CHECKPOINT_SHA = "769becfe854acd238f45ed4ade2810bb1e454166db3f5c6d557aa817275d25af"
TRAINING_NPZ_SHA = "303d3eb3ff48d67bcaf44a5ffa3de7b98f94d64ff00a9712f697258311b73561"

#: The 12 October staged-training evidence bundle, whose own SHA256SUMS.txt
#: this module's author verified file-by-file (12/12 OK). These turn the
#: staged-training component from an accuracy-only claim into a provenance
#: -bound one: the checkpoint evaluated, the per-round training histories,
#: the source snapshot the run used, and the executing environment.
TRAINING_BUNDLE_SHA = {
    "bundle_manifest":
        None,   # SHA256SUMS.txt lists the others; its own digest is open
    "checkpoint_h5":
        "815062f11d9a0631477405fd206ef5b351e469775a9283c0e66fa76f3a0081d0",
    "source_snapshot":
        "bd72609c19a1d5741033a4c67cff45041f68fd1af4a41318632cacb3604672d2",
    "pip_freeze":
        "5d32680bad8ae7268848280abb393b1291d52cab0ba4a6f9149ee9c95443601f",
    "results_summary":
        "3e0c2e563c61bcde79136fb3b95abd71f0b8811a79d4f040601ca94241475499",
    "epoch_diagnostic":
        "76719077033e3a7d650a49f1a15673b358761fc3fa732969975318b729621a39",
    "history_round5":
        "986b6de1f811092e0bfebd7999181adb0028de706bdfff41b8052b3e553d2d01",
    "history_round6":
        "4a023bc96525a133fef54817017db4f45f1e130db4a9bc2e7d99c82ac09f07c5",
    "history_round7":
        "5a33dd3175e74813e5686716ea3a0a9fed993c2547ac498a43b932dcc0953e5a",
    "history_round8":
        "b4c8f456fb6b1acb5f2b93c166c26db40a62fc5b6a2eb063b20e44615970bd79",
    "history_round9":
        "8588470c5828c5e9814a67561fc1d8fdf1044ccda97bff054725c6d9d028b958",
}

#: Directory, relative to the repo root, holding the bundle above.
TRAINING_BUNDLE_DIR = "audit/targets/autond/evidence/training"


class Provenance(str, Enum):
    """How an expected hash came to be in this file."""
    #: Reported by the operator; not confirmed by this module's author.
    RECORDED = "RECORDED"
    #: Confirmed by this module's author against the actual artifact.
    AUTHOR_VERIFIED = "AUTHOR_VERIFIED"
    #: No expected value; whatever is on disk is hashed and recorded.
    OPEN = "OPEN"


@dataclass(frozen=True)
class ExpectedArtifact:
    """One evidence file the stage record binds to."""
    label: str
    relative_to: str              # "repo" | "home" | "upstream"
    path: str
    expected_sha256: Optional[str]
    provenance: Provenance
    required: bool
    describes: str

    def resolve(self, repo: Path, home: Path, upstream: Path) -> Path:
        base = {"repo": repo, "home": home, "upstream": upstream}[self.relative_to]
        return (base / self.path).expanduser()


#: The AutoND evidence inventory. Paths are the operator's real layout.
EXPECTED: tuple[ExpectedArtifact, ...] = (
    # ---- II-1: repository reproduction -------------------------------
    ExpectedArtifact(
        "r1_search_manifest", "home",
        "Documents/autond_search_runs/r1_20261015T035553Z/MANIFEST.json",
        None, Provenance.OPEN, True,
        "manifest of the completed R1 search run; self-hashing, so its own "
        "digest is open and its contents are checked instead"),
    ExpectedArtifact(
        "r1_search_result", "home",
        "Documents/autond_search_runs/r1_20261015T035553Z/run_000/result.json",
        None, Provenance.OPEN, True,
        "R1 result record: pinned commit, patches applied, returned differences"),
    ExpectedArtifact(
        "r1_generation_log", "home",
        "Documents/autond_search_runs/r1_20261015T035553Z/run_000/generations.jsonl",
        None, Provenance.OPEN, True,
        "per-scoring-call log of the R1 search"),
    ExpectedArtifact(
        "training_evidence_npz", "repo",
        "audit/targets/autond/evidence/training/"
        "speck3264_single-key_0x400000_dbitnet_round8_eval_10epoch_reproduction.npz",
        TRAINING_NPZ_SHA, Provenance.AUTHOR_VERIFIED, False,
        "five-set evaluation of the 12 October staged-training round-8 checkpoint; "
        "hash confirmed by this module's author against the evidence bundle"),
    ExpectedArtifact(
        "speck_polished_checkpoint", "upstream",
        "model_weights/speck3264_single-key_0x400000_round8_polished.h5",
        SPECK_CHECKPOINT_SHA, Provenance.AUTHOR_VERIFIED, True,
        "shipped polished SPECK32/64 round-8 checkpoint"),
    ExpectedArtifact(
        "simon_polished_checkpoint", "upstream",
        "model_weights/simon3264_single-key_0x400_round11_polished.h5",
        SIMON_CHECKPOINT_SHA, Provenance.AUTHOR_VERIFIED, True,
        "shipped polished SIMON32/64 round-11 checkpoint"),
    # ---- II-3: controlled verification --------------------------------
    ExpectedArtifact(
        "architecture_census", "repo",
        "audit/targets/autond/evidence/census/census_v2_20261015T043855Z.json",
        "8e1e085f97d86f0d78f39a1ac7237342e9169d6c6bf55c1d400d9ace3ebb7ca2",
        Provenance.RECORDED, True,
        "h5py tensor-inventory census of both shipped checkpoints"),
    ExpectedArtifact(
        "architecture_reconciliation", "repo",
        "audit/targets/autond/evidence/census/arch_reconciliation_20261015T043855Z.json",
        "49881b1c2f5540ebeccbe5f3968bba78eabd484244d89a871f392200255cae4a",
        Provenance.RECORDED, True,
        "reconciliation of the census against the source-derived architecture"),
    ExpectedArtifact(
        "checkpoint_load_verification", "repo",
        "audit/targets/autond/evidence/census/load_verify_r2_20261015T051547Z.json",
        "51b58e11fe346dd02664663e7dd236527349f829184ac8699a3e36ea4f34b4f0",
        Provenance.RECORDED, True,
        "TensorFlow-side load verification, element-wise against the files"),
    ExpectedArtifact(
        "paper_source_architecture_record", "repo",
        "audit/targets/autond/evidence/architecture/"
        "paper_source_architecture_v2.json",
        None, Provenance.OPEN, False,
        "serialised paper-to-source comparison; written by "
        "paper_source_architecture.write_record so the stage binds the sha256 "
        "of a real file rather than a content digest with no file behind it"),
    # ---- II-1: staged-training provenance (12 October bundle) ---------
    ExpectedArtifact(
        "training_bundle_manifest", "repo",
        f"{TRAINING_BUNDLE_DIR}/SHA256SUMS.txt",
        None, Provenance.OPEN, False,
        "the bundle's own manifest; its contents are checked rather than its "
        "digest, which is open because the manifest lists the other files"),
    ExpectedArtifact(
        "training_checkpoint_h5", "repo",
        f"{TRAINING_BUNDLE_DIR}/speck3264_single-key_0x400000_dbitnet_round8.h5",
        TRAINING_BUNDLE_SHA["checkpoint_h5"], Provenance.AUTHOR_VERIFIED, False,
        "the UNPOLISHED round-8 checkpoint actually evaluated by the staged "
        "training run; distinct from the shipped polished checkpoint"),
    ExpectedArtifact(
        "training_source_snapshot", "repo",
        f"{TRAINING_BUNDLE_DIR}/AutoND_source_snapshot.tar.gz",
        TRAINING_BUNDLE_SHA["source_snapshot"], Provenance.AUTHOR_VERIFIED, False,
        "snapshot of the target source the staged training ran against"),
    ExpectedArtifact(
        "training_environment", "repo",
        f"{TRAINING_BUNDLE_DIR}/pip_freeze.txt",
        TRAINING_BUNDLE_SHA["pip_freeze"], Provenance.AUTHOR_VERIFIED, False,
        "the executing environment of the staged training run"),
    ExpectedArtifact(
        "training_results_summary", "repo",
        f"{TRAINING_BUNDLE_DIR}/results_summary.json",
        TRAINING_BUNDLE_SHA["results_summary"], Provenance.AUTHOR_VERIFIED, False,
        "per-set accuracies, TPR/TNR and the Table 6 comparison target for the "
        "staged training run; the component's numbers are READ FROM HERE"),
    ExpectedArtifact(
        "training_epoch_diagnostic", "repo",
        f"{TRAINING_BUNDLE_DIR}/checkpoint_epoch_diagnostic.csv",
        TRAINING_BUNDLE_SHA["epoch_diagnostic"], Provenance.AUTHOR_VERIFIED, False,
        "why the round's best reported validation accuracy is not necessarily "
        "the saved checkpoint's own"),
) + tuple(
    ExpectedArtifact(
        f"training_history_round{r}", "repo",
        f"{TRAINING_BUNDLE_DIR}/"
        f"speck3264_single-key_0x400000_dbitnet_training_history_round{r}.pkl",
        TRAINING_BUNDLE_SHA[f"history_round{r}"], Provenance.AUTHOR_VERIFIED, False,
        f"per-epoch training history for round {r} of the staged schedule")
    for r in (5, 6, 7, 8, 9)
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class ArtifactCheck:
    """Integrity outcome for one expected artifact. Never a science verdict."""
    label: str
    path: str
    present: bool
    observed_sha256: Optional[str]
    expected_sha256: Optional[str]
    provenance: str
    required: bool
    verdict: str                  # VERIFIED | HASH_MISMATCH | MISSING | RECORDED_ONLY
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict in ("VERIFIED", "RECORDED_ONLY")

    def to_dict(self) -> dict:
        return {
            "label": self.label, "path": self.path, "present": self.present,
            "observed_sha256": self.observed_sha256,
            "expected_sha256": self.expected_sha256,
            "provenance": self.provenance, "required": self.required,
            "verdict": self.verdict, "detail": self.detail,
        }


def check_artifacts(repo: Path, home: Path, upstream: Path,
                    subset: Optional[Iterable[str]] = None) -> list[ArtifactCheck]:
    wanted = set(subset) if subset is not None else None
    out: list[ArtifactCheck] = []
    for art in EXPECTED:
        if wanted is not None and art.label not in wanted:
            continue
        p = art.resolve(repo, home, upstream)
        if not p.is_file():
            out.append(ArtifactCheck(
                art.label, str(p), False, None, art.expected_sha256,
                art.provenance.value, art.required, "MISSING",
                "the stage cannot bind evidence it cannot open"))
            continue
        got = sha256_file(p)
        if art.expected_sha256 is None:
            out.append(ArtifactCheck(
                art.label, str(p), True, got, None, art.provenance.value,
                art.required, "RECORDED_ONLY",
                "no expected digest declared; the observed value is recorded so a "
                "later run can detect drift"))
        elif got == art.expected_sha256:
            out.append(ArtifactCheck(
                art.label, str(p), True, got, art.expected_sha256,
                art.provenance.value, art.required, "VERIFIED"))
        else:
            out.append(ArtifactCheck(
                art.label, str(p), True, got, art.expected_sha256,
                art.provenance.value, art.required, "HASH_MISMATCH",
                "the artifact on disk is not the one this stage record expects; "
                "resolve the provenance before the record is used"))
    return out


def evidence_map(checks: Iterable[ArtifactCheck]) -> dict[str, str]:
    """path -> sha256, the shape StageComponent.evidence requires."""
    return {c.path: c.observed_sha256 for c in checks
            if c.observed_sha256 is not None}


def verify_upstream_commit(upstream: Path) -> tuple[bool, str, str]:
    try:
        got = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
    except Exception as exc:                                        # noqa: BLE001
        return False, "", f"git could not be run: {type(exc).__name__}: {exc}"
    if got.returncode != 0:
        return False, "", f"git rev-parse failed: {got.stderr.strip()}"
    head = got.stdout.strip()
    if head != PINNED_COMMIT:
        return False, head, (f"checkout is at {head}, not the pinned "
                             f"{PINNED_COMMIT}")
    return True, head, ""


# ----------------------------------------------------------------------
# II-1
# ----------------------------------------------------------------------

def _stage_summary(label: str, components: list[StageComponent]) -> str:
    """
    Describe a stage by what its components ACTUALLY concluded.

    The previous revision wrote a fixed sentence - "tensor structure and
    checkpoint loading verified against the pinned source" - into the
    stage summary unconditionally, so a stage whose evidence was entirely
    absent still read as though the checks had passed. A stage summary is
    now assembled from the component statuses and cannot say more than
    they do.
    """
    if not components:
        return f"{label}: no components were built."
    by_status: dict[str, list[str]] = {}
    for c in components:
        by_status.setdefault(c.status.value, []).append(c.component_id)
    parts = [f"{status} ({', '.join(sorted(ids))})"
             for status, ids in sorted(by_status.items())]
    n_sat = sum(1 for c in components if c.status in SATISFYING)
    return (f"{label}: {n_sat}/{len(components)} component(s) satisfied. "
            + "; ".join(parts) + ".")


def _evidence_state(checks: list[ArtifactCheck]) -> tuple[str, list[str]]:
    """
    Map artifact verdicts onto one of three states.

    The distinction matters and is not cosmetic:

      MISMATCH  a file is present and is the WRONG file. The check ran
                and failed. That is an integrity FAILURE.
      MISSING   a required file is absent. The check never ran, so there
                is nothing to have failed. That is NOT_PERFORMED.
      OK        every required artifact verified (or was recorded-only).

    Recording a missing file as FAILED would claim a check was performed
    and came out badly, which is a different and stronger statement than
    the truth.
    """
    mismatched = [c for c in checks if c.verdict == "HASH_MISMATCH"]
    if mismatched:
        return "MISMATCH", [
            f"{c.label}: observed {c.observed_sha256} != expected {c.expected_sha256}"
            for c in mismatched]
    missing_required = [c for c in checks if c.verdict == "MISSING" and c.required]
    if missing_required:
        return "MISSING", [f"{c.label}: absent at {c.path}" for c in missing_required]
    missing_optional = [c for c in checks if c.verdict == "MISSING"]
    if missing_optional and not [c for c in checks if c.ok]:
        return "MISSING", [f"{c.label}: absent at {c.path}" for c in missing_optional]
    return "OK", []


def _read_json(check: Optional[ArtifactCheck]) -> Optional[dict]:
    """Parse an artifact's JSON, or None if it is absent or unreadable."""
    if check is None or not check.present:
        return None
    try:
        return json.loads(Path(check.path).read_text())
    except Exception:                                               # noqa: BLE001
        return None


def _load_search_verifier():
    """
    The search harness's own validator, imported lazily.

    R1 evidence is validated by audit.targets.autond.search.verify_outputs,
    which already recomputes every manifest hash, checks that the manifest
    covers the directory, confirms the result's source commit against the
    pinned commit, checks generations.jsonl for contiguous sequence numbers
    against the result's own logged-call count, and surfaces
    behaviour-changing patches. Reimplementing those checks here would be
    duplicate logic with a second place to drift.

    If the harness is not installed the answer is "cannot validate", which
    blocks the component. It is never a silent pass.
    """
    try:
        from audit.targets.autond.search.verify_outputs import verify
        return verify, None
    except Exception as exc:                                        # noqa: BLE001
        return None, (f"search harness not importable "
                      f"({type(exc).__name__}: {exc}); R1 evidence cannot be "
                      "validated without it. Install autond_search_harness.zip.")


def validate_r1(repo: Path, home: Path, upstream: Path,
                checks_by_label: dict[str, ArtifactCheck]) -> dict:
    """
    Validate the R1 search evidence through the search harness.

    Returns a record with `integrity_ok`, `problems`, and - only when
    integrity holds - the scientific reading. A malformed or inconsistent
    bundle yields integrity_ok False and NO scientific reading at all, so
    a broken bundle can never be interpreted as a valid search result.
    """
    manifest_check = checks_by_label.get("r1_search_manifest")
    out: dict = {"integrity_ok": False, "problems": [], "science": None,
                 "harness": None, "run_dir": None, "manifest_verified": []}

    if manifest_check is None or not manifest_check.present:
        out["problems"].append(
            "r1_search_manifest is absent; without MANIFEST.json there is no "
            "bundle to validate and no result to read")
        return out

    run_dir = Path(manifest_check.path).parent
    out["run_dir"] = str(run_dir)

    verify, why = _load_search_verifier()
    if verify is None:
        out["problems"].append(why)
        return out

    try:
        ok, problems, summary = verify(run_dir)
    except Exception as exc:                                        # noqa: BLE001
        out["problems"].append(f"the search harness raised "
                               f"{type(exc).__name__}: {exc} on {run_dir}")
        return out

    out["harness"] = {"ok": bool(ok), "problems": list(problems),
                      "summary": summary,
                      "validator": "audit.targets.autond.search.verify_outputs.verify",
                      "mode": "validation-only; executes no search"}
    out["problems"].extend(problems)

    # The harness checks the bundle. These are the plan-level requirements
    # the harness does not know about.
    runs = summary.get("runs") or []
    if not runs:
        out["problems"].append(f"{run_dir} contains no run_* directory with a "
                               "result.json")
    for row in runs:
        # Compared case-insensitively, and only after confirming it IS a
        # string. This value is written by the search harness into
        # result.json, so its casing is the target's convention rather than
        # ours; a strict `!= "completed"` rejected "COMPLETED" and crashed
        # the comparison on a null or numeric status. Fix contributed by the
        # operator, with the regression test in
        # tests/test_corrections.TestR1Validation.
        status = row.get("status")
        if not isinstance(status, str) or status.casefold() != "completed":
            out["problems"].append(
                f"{row.get('run')}: status {status!r} is not "
                "'completed'; an unfinished run is not a search result")
        if row.get("behaviour_changing_patches"):
            out["problems"].append(
                f"{row.get('run')}: behaviour-changing patches were applied "
                f"{row['behaviour_changing_patches']}; this is not an unmodified "
                "repository reproduction")

    if out["problems"]:
        return out

    # Integrity holds. Only now is the scientific result read.
    out["integrity_ok"] = True
    # The harness recomputed every digest in MANIFEST.json, confirmed the
    # manifest covers the directory, and confirmed the result's own source
    # commit. The R1 files are therefore integrity-established even though
    # this adapter declares no separate expected digest for each of them -
    # the manifest is the binding, and hard-coding a second copy of each
    # digest here would be a second place to drift. Revision 2's --strict
    # could not express this and so could never pass on valid evidence.
    out["manifest_verified"] = [
        "r1_search_manifest", "r1_search_result", "r1_generation_log"]

    # Operating-system / interpreter debris the harness excused. Carried
    # up so the component DISCLOSES it rather than passing silently: a
    # reader must be able to see that the directory held files the
    # manifest does not name, and why that did not invalidate it.
    out["unmanifested_benign"] = list(summary.get("unmanifested_benign") or [])
    if out["unmanifested_benign"]:
        out["unmanifested_benign_note"] = summary.get(
            "unmanifested_benign_note", "")
    row = runs[0]
    raw = row.get("differences") or []
    values: list[int] = []
    for v in raw:
        t = str(v)
        try:
            values.append(int(t, 16) if t.lower().startswith("0x") else int(t))
        except ValueError:
            continue
    hexes = [hex(v) for v in values]
    out["science"] = {
        "n_runs": len(runs),
        "status": row.get("status"),
        "n_differences": len(values),
        "differences_hex": hexes,
        "recovered_expected_difference": 0x400000 in values,
        "extra_differences": [h for h in hexes if h != "0x400000"],
        "highest_non_random_round": row.get("highest_non_random_round"),
        "log_rows": row.get("log_rows"),
        "mode": row.get("mode"),
        "bit_reproducible": row.get("bit_reproducible"),
        "note": ("the result record's returned_differences_hex field holds "
                 "decimal strings, inheriting upstream's mislabelling of an "
                 "integer return as hex; parsed defensively"),
    }
    return out


# ----------------------------------------------------------------------
# II-1
# ----------------------------------------------------------------------

def build_ii1(repo: Path, home: Path, upstream: Path) -> tuple[StageStatus, dict]:
    """
    Repository reproduction. Three published claims, three components.

    Integrity and science are kept apart throughout. A reproduction that
    recovers the expected difference while also returning an extra one has
    not failed; it has produced a result that differs from the
    documentation, which is what it was run to find out. A bundle whose
    manifest does not validate produces no scientific reading at all.
    """
    commit_ok, head, commit_detail = verify_upstream_commit(upstream)
    checks = check_artifacts(repo, home, upstream)
    by_label = {c.label: c for c in checks}
    components: list[StageComponent] = []
    findings: list[str] = []

    # --- component 1: the R1 search reproduction ----------------------
    r1_labels = ("r1_search_manifest", "r1_search_result", "r1_generation_log")
    r1_checks = [by_label[k] for k in r1_labels if k in by_label]
    r1_state, r1_state_problems = _evidence_state(r1_checks)
    r1 = validate_r1(repo, home, upstream, by_label)

    r1_question = ("Does the released search, run as documented, return the "
                   "input difference the published SPECK32/64 result uses?")
    if not commit_ok:
        components.append(StageComponent(
            component_id="II1-SEARCH-R1", kind=ComponentKind.REFERENCE_VERIFICATION,
            status=StageOutcome.FAILED,
            question=r1_question,
            summary=f"upstream checkout is not the pinned revision: {commit_detail}",
            evidence=evidence_map(r1_checks),
            limitations=["an integrity failure is not a statement about the "
                         "target; it means this record cannot bind its evidence"]))
        findings.append(f"II1-SEARCH-R1: {commit_detail}")
    elif r1_state == "MISMATCH":
        components.append(StageComponent(
            component_id="II1-SEARCH-R1", kind=ComponentKind.REFERENCE_VERIFICATION,
            status=StageOutcome.FAILED,
            question=r1_question,
            summary=("R1 evidence is present but is not the expected content: "
                     + "; ".join(r1_state_problems)),
            evidence=evidence_map(r1_checks),
            limitations=["a hash mismatch is an integrity failure: the check ran "
                         "and the artifact was the wrong one"]))
        findings.append("II1-SEARCH-R1 integrity failure: "
                        + "; ".join(r1_state_problems))
    elif r1_state == "MISSING":
        components.append(StageComponent(
            component_id="II1-SEARCH-R1", kind=ComponentKind.REFERENCE_VERIFICATION,
            status=StageOutcome.NOT_PERFORMED,
            question=r1_question,
            summary=("R1 search evidence is not present on this machine, so no "
                     "validation was attempted and no result was read: "
                     + "; ".join(r1_state_problems)),
            evidence=evidence_map(r1_checks),
            limitations=["absent evidence means this check did not run; it does "
                         "NOT mean the check ran and failed, and it is not a "
                         "statement about the target"]))
        findings.append("II1-SEARCH-R1: evidence absent on this machine; "
                        "the component was not performed here.")
    elif not r1["integrity_ok"]:
        components.append(StageComponent(
            component_id="II1-SEARCH-R1", kind=ComponentKind.REFERENCE_VERIFICATION,
            status=StageOutcome.FAILED,
            question=r1_question,
            summary=("R1 evidence failed validation, so no scientific result was "
                     "read from it: " + "; ".join(r1["problems"][:4])),
            evidence=evidence_map(r1_checks),
            limitations=[
                "validated by the search harness in validation-only mode; no "
                "search was executed",
                "a bundle that does not validate is not interpreted as a search "
                "result, however complete its result.json appears",
            ]))
        findings.append("II1-SEARCH-R1 evidence failed validation: "
                        + "; ".join(r1["problems"][:4]))
    else:
        sci = r1["science"]
        matched = sci["recovered_expected_difference"]
        extra = sci["extra_differences"]
        status = (StageOutcome.PARTIAL if extra or not matched
                  else StageOutcome.CONFORMANCE_VERIFIED)
        components.append(StageComponent(
            component_id="II1-SEARCH-R1", kind=ComponentKind.REFERENCE_VERIFICATION,
            status=status,
            question=r1_question,
            summary=(
                f"bundle validated by the search harness ({sci['n_runs']} run(s), "
                f"status {sci['status']}, {sci['log_rows']} logged scoring calls). "
                f"0x400000 recovered: {matched}. Returned "
                f"{sci['n_differences']} epsilon-close difference(s) "
                f"{sci['differences_hex']}; the README documents exactly one, "
                f"['0x400000']. highest_non_random_round="
                f"{sci['highest_non_random_round']}, against 7 in the README "
                "prose and 6 implied by its own log sample and training output. "
                "Run with no behaviour-changing patches, at the released "
                "T_b=0.05."),
            evidence=evidence_map(r1_checks),
            limitations=[
                "one execution of an unseeded search; establishes nothing about "
                "run-to-run reliability",
                "the released code hardcodes T_b=0.05 where the paper states 0.01; "
                "R1 used the released value because this is repository "
                "reproduction, and the difference is a finding, not a defect here",
                sci["note"],
            ] + ([r1["unmanifested_benign_note"]]
                 if r1.get("unmanifested_benign") else [])))
        if extra or not matched:
            findings.append(
                "II-FINDING-AUTOND-SEARCH-R1: the released search returned "
                f"{sci['n_differences']} epsilon-close differences where the "
                "README documents one. PARTIAL match, preserved as observed.")

    # --- component 2: the staged-training reproduction ----------------
    # Provenance-bound: the checkpoint evaluated, the per-round training
    # histories, the source snapshot, the environment and the bundle
    # manifest, not the NPZ alone. Numbers are READ from results_summary.
    bundle_labels = ("training_evidence_npz", "training_bundle_manifest",
                     "training_checkpoint_h5", "training_source_snapshot",
                     "training_environment", "training_results_summary",
                     "training_epoch_diagnostic") + tuple(
                         f"training_history_round{r}" for r in (5, 6, 7, 8, 9))
    bundle_checks = [by_label[k] for k in bundle_labels if k in by_label]
    bundle_present = [c for c in bundle_checks if c.present]
    bundle_state, bundle_problems = _evidence_state(bundle_checks)
    summary_rec = _read_json(by_label.get("training_results_summary"))

    tr_question = ("Does the released staged training, from round 5 with the "
                   "fixed difference, reach the published round-8 accuracy?")
    if bundle_state == "MISMATCH":
        components.append(StageComponent(
            component_id="II1-STAGED-TRAINING",
            kind=ComponentKind.REFERENCE_VERIFICATION,
            status=StageOutcome.FAILED,
            question=tr_question,
            summary=("staged-training bundle is present but its contents are not "
                     "the expected ones: " + "; ".join(bundle_problems)),
            evidence=evidence_map(bundle_present),
            limitations=["a hash mismatch in the bundle is an integrity failure"]))
        findings.append("II1-STAGED-TRAINING integrity failure: "
                        + "; ".join(bundle_problems))
    elif summary_rec is None:
        components.append(StageComponent(
            component_id="II1-STAGED-TRAINING",
            kind=ComponentKind.REFERENCE_VERIFICATION,
            status=StageOutcome.NOT_PERFORMED,
            question=tr_question,
            summary=("the 12 October evidence bundle was not found at "
                     f"{TRAINING_BUNDLE_DIR}, so no accuracy was read. The run "
                     "itself is complete and must NOT be repeated to satisfy "
                     "this record; install the existing bundle instead."),
            evidence=evidence_map(bundle_present),
            limitations=["evidence not located; status reflects the record, not "
                         "the experiment"]))
        findings.append("II1-STAGED-TRAINING: evidence bundle not installed at "
                        f"{TRAINING_BUNDLE_DIR}; do not rerun the training.")
    else:
        accs = [float(a) for a in summary_rec.get("evaluation_scores", [])]
        stats = evres.mean_and_t_interval(accs) if accs else {}
        target = summary_rec.get("table6_comparison_target")
        mean = stats.get("mean")
        n_bound = len(bundle_present)
        n_total = len(bundle_checks)
        # Provenance completeness decides the status, not the accuracy.
        full_provenance = all(
            by_label[k].present for k in
            ("training_checkpoint_h5", "training_source_snapshot",
             "training_environment", "training_results_summary")
            if k in by_label)
        status = (StageOutcome.CONFORMANCE_VERIFIED if full_provenance
                  else StageOutcome.PARTIAL)
        diff_txt = (f"{abs(mean - target):.2e}"
                    if (mean is not None and target is not None) else "n/a")
        components.append(StageComponent(
            component_id="II1-STAGED-TRAINING",
            kind=ComponentKind.REFERENCE_VERIFICATION,
            status=status,
            question=tr_question,
            summary=(
                f"UNPOLISHED round-8 checkpoint "
                f"({(by_label['training_checkpoint_h5'].observed_sha256 or '')[:12]}...) "
                f"evaluated on {len(accs)} fresh 10^6 test sets: mean "
                f"{mean:.7f}, sample SD {stats.get('sample_sd'):.7f}, against "
                f"the bundle's recorded Table 6 target {target} "
                f"(difference {diff_txt}). "
                f"{n_bound}/{n_total} provenance artifacts bound: checkpoint, "
                "five per-round training histories, source snapshot, "
                "environment, epoch diagnostic and bundle manifest."
                if mean is not None else
                f"{n_bound}/{n_total} provenance artifacts bound, but "
                "results_summary.json carried no evaluation_scores"),
            evidence=evidence_map(bundle_present),
            limitations=[
                "ONE staged-training execution; no evidence of training-to-"
                "training robustness",
                "the evolutionary search was deliberately bypassed and the "
                "difference supplied directly",
                "checkpoint selection used minimum validation loss at epoch 5 "
                "while the round's best validation accuracy occurred at epoch 2; "
                "see checkpoint_epoch_diagnostic.csv",
                "this is the UNPOLISHED round-8 checkpoint against the 0.5096 "
                "figure. It is a DIFFERENT artifact from the shipped polished "
                "checkpoint (a7be788d...) whose published value is 0.5144, and "
                "the two must not be compared with each other",
                "the source snapshot establishes which source the run used; it "
                "does not establish that the shipped checkpoints came from it",
            ]))

    # --- component 3: confirmatory checkpoint evaluation --------------
    # Ingests a validated result bundle when one exists. This component is
    # NOT hard-wired to NOT_PERFORMED: the gate decides whether a run is
    # permitted, and the schema validator decides whether a produced
    # bundle may be read.
    prereg_ok, prereg_blockers = _preregistration_gate(upstream, repo)
    eval_ingest = _ingest_evaluation(repo)
    ck_checks = [by_label[k] for k in
                 ("speck_polished_checkpoint", "simon_polished_checkpoint")
                 if k in by_label]
    eval_question = ("Do the shipped polished checkpoints reproduce the published "
                     "accuracies on freshly generated test sets?")

    if eval_ingest["state"] == "VALID":
        res = eval_ingest["result"]
        ev = dict(evidence_map(ck_checks))
        ev.update(eval_ingest["evidence"])
        lines = []
        for cid, st in sorted((res.get("derived") or {}).items()):
            if st.get("mean") is not None:
                lines.append(f"{cid} mean {st['mean']:.6f} (n={st['n']}, "
                             f"SD {st['sample_sd']:.7f})")
        components.append(StageComponent(
            component_id="II1-SHIPPED-CHECKPOINT-EVAL",
            kind=ComponentKind.REFERENCE_VERIFICATION,
            status=StageOutcome.COMPLETED,
            question=eval_question,
            summary=("a result bundle validated against "
                     f"{evres.SCHEMA_VERSION} and bound to the frozen plan was "
                     "ingested: " + "; ".join(lines)),
            evidence=ev,
            limitations=[
                "the decision rule in the preregistration, not this component, "
                "decides whether the result supports a hypothesis",
                "inference only on shipped weights; says nothing about the "
                "training pipeline or the search",
            ]))
    else:
        components.append(StageComponent(
            component_id="II1-SHIPPED-CHECKPOINT-EVAL",
            kind=ComponentKind.REFERENCE_VERIFICATION,
            status=StageOutcome.NOT_PERFORMED,
            question=eval_question,
            summary=(
                "A historical evaluation produced SPECK 0.5143236 and SIMON "
                "0.5175946, but its raw NPZ outputs were NOT retained, so no "
                "hash-bound record exists. A fresh inference-only run is "
                f"required. Result bundle: {eval_ingest['detail']}. "
                f"Confirmatory run permitted by the successor preregistration: "
                f"{prereg_ok}."),
            evidence=evidence_map(ck_checks),
            limitations=[
                "the historical accuracies are prose in this project's records; "
                "the artifacts behind them do not exist and must never be "
                "described as retained evidence",
                "published values disagree between sources and are preserved "
                "separately: paper Table 6 SPECK 0.5144 / SIMON 0.5179; README "
                "SPECK 0.5144 +- 0.0003 / SIMON 0.5182 +- 0.0004",
                "this component is gated on the successor preregistration being "
                "frozen with a verified sidecar before the run; it is not "
                "hard-wired to remain unperformed, and ingests a validated "
                "bundle as soon as one exists",
            ]))
        if not prereg_ok:
            findings.append("II1-SHIPPED-CHECKPOINT-EVAL blocked by the successor "
                            f"preregistration gate: {'; '.join(prereg_blockers)}")

    satisfied = [c for c in components if c.status in SATISFYING]
    stage_status = (StageOutcome.CONFORMANCE_VERIFIED
                    if len(satisfied) == len(components)
                    else StageOutcome.FAILED
                    if any(c.status is StageOutcome.FAILED for c in components)
                    else StageOutcome.PARTIAL)
    stage = StageStatus(
        stage_id=StageId.II_1_BASELINE_RECONSTRUCTION,
        tier=STAGE_TIERS[StageId.II_1_BASELINE_RECONSTRUCTION],
        status=stage_status,
        summary=_stage_summary("II-1 repository reproduction", components),
        findings=findings,
        components=tuple(components))
    return stage, {"checks": [c.to_dict() for c in checks],
                   "commit_ok": commit_ok, "head": head,
                   "commit_detail": commit_detail,
                   "r1": r1,
                   "r1_science": r1.get("science"),
                   "evaluation_ingest": eval_ingest}


def _preregistration_gate(upstream: Optional[Path] = None,
                          repo_root: Optional[Path] = None
                          ) -> tuple[bool, list[str]]:
    """
    Consult the successor preregistration's own gate.

    Imported lazily and defensively: if the plan module is absent the
    answer is "blocked", which is the correct conservative result and
    not an error in this adapter. The checkout is passed through so the
    plan can RECOMPUTE its recorded digests rather than trust a flag.
    """
    try:
        from audit.targets.autond.preregistrations import autond_ii_v2 as plan
    except Exception as exc:                                        # noqa: BLE001
        return False, [f"successor preregistration not importable: "
                       f"{type(exc).__name__}: {exc}"]
    try:
        return plan.confirmatory_run_permitted(upstream, repo_root)
    except Exception as exc:                                        # noqa: BLE001
        return False, [f"preregistration gate raised {type(exc).__name__}: {exc}"]


#: Where a produced evaluation bundle is looked for, relative to the repo.
EVALUATION_BUNDLE_DIR = "audit/targets/autond/evidence/evaluation"


def _ingest_evaluation(repo: Path) -> dict:
    """
    Look for a produced evaluation result bundle and validate it.

    Returns {"state": "VALID"|"ABSENT"|"INVALID", "detail": str,
             "result": {...}|None, "evidence": {path: sha256}}.

    A bundle is ingested only if it validates against the schema AND is
    bound to the frozen plan's hash. An unfrozen plan therefore cannot
    have a valid bundle, which is the intended coupling.
    """
    d = Path(repo) / EVALUATION_BUNDLE_DIR
    if not d.is_dir():
        return {"state": "ABSENT", "result": None, "evidence": {},
                "detail": f"none produced yet (looked in {EVALUATION_BUNDLE_DIR})"}
    try:
        from audit.targets.autond.preregistrations import autond_ii_v2 as plan
        expected_hash = plan.plan_hash()
        frozen, _ = plan.is_frozen()
        if not frozen:
            expected_hash = None
    except Exception:                                               # noqa: BLE001
        expected_hash = None
        frozen = False

    # require_plan_binding tracks `frozen`: an unfrozen plan has no hash to
    # bind to, and the clearer message for that case is the one below, not
    # the validator's generic complaint about a missing expected hash.
    res = evres.validate_bundle(d, expected_plan_hash=expected_hash,
                                require_plan_binding=bool(frozen))
    if not res["ok"]:
        return {"state": "INVALID", "result": res, "evidence": {},
                "detail": (f"a bundle exists at {EVALUATION_BUNDLE_DIR} but failed "
                           f"validation ({len(res['problems'])} problem(s)): "
                           + "; ".join(res["problems"][:3]))}
    if not frozen:
        return {"state": "INVALID", "result": res, "evidence": {},
                "detail": ("a bundle exists and is internally valid, but the plan "
                           "is not frozen, so it cannot be confirmatory evidence")}
    evidence = {}
    for f in sorted(d.rglob("*")):
        if f.is_file():
            evidence[str(f)] = sha256_file(f)
    return {"state": "VALID", "result": res, "evidence": evidence,
            "detail": f"validated against {evres.SCHEMA_VERSION}"}


# ----------------------------------------------------------------------
# II-3
# ----------------------------------------------------------------------

def build_ii3(repo: Path, home: Path, upstream: Path) -> tuple[StageStatus, dict]:
    """
    Controlled verification: declared architecture against realised artifact.

    Three components. Every number in every summary is READ from the
    evidence file it is bound to; nothing is transcribed. A component
    whose evidence is absent says so and claims nothing.
    """
    checks = check_artifacts(repo, home, upstream)
    by_label = {c.label: c for c in checks}
    components: list[StageComponent] = []
    findings: list[str] = []

    # --- component 1: the architecture census -------------------------
    census_labels = ("architecture_census", "architecture_reconciliation")
    census_checks = [by_label[k] for k in census_labels if k in by_label]
    census_state, census_problems = _evidence_state(census_checks)
    census = _read_json(by_label.get("architecture_census"))
    recon = _read_json(by_label.get("architecture_reconciliation"))

    census_question = ("Does the realised tensor structure of the shipped "
                       "checkpoints match the architecture the pinned source "
                       "constructs?")
    if census_state == "MISMATCH":
        components.append(StageComponent(
            component_id="II3-ARCHITECTURE-CENSUS",
            kind=ComponentKind.ARCHITECTURE_CENSUS,
            status=StageOutcome.FAILED,
            question=census_question,
            summary=("census evidence is present but is not the expected "
                     "content: " + "; ".join(census_problems)),
            evidence=evidence_map([c for c in census_checks if c.present]),
            limitations=["a hash mismatch is an integrity failure: the artifact "
                         "on disk is not the one this record expects"]))
        findings.append("II3-ARCHITECTURE-CENSUS integrity failure: "
                        + "; ".join(census_problems))
    elif census is None or recon is None:
        components.append(StageComponent(
            component_id="II3-ARCHITECTURE-CENSUS",
            kind=ComponentKind.ARCHITECTURE_CENSUS,
            status=StageOutcome.NOT_PERFORMED,
            question=census_question,
            summary=("census evidence is not present on this machine, so no "
                     "tensor inventory was read and no conformance is claimed: "
                     + ("; ".join(census_problems) or "files absent")),
            evidence=evidence_map([c for c in census_checks if c.present]),
            limitations=["absent evidence means this check did not run here; it "
                         "is not a claim about the checkpoints"]))
        findings.append("II3-ARCHITECTURE-CENSUS: evidence absent on this "
                        "machine; the component was not performed here.")
    else:
        models = census.get("models") or []
        derived = [m.get("derived", {}) for m in models]
        recon_block = recon.get("reconciliation", {})
        all_reconcile = bool(recon_block.get("all_models_reconcile"))
        refusals = census.get("refusals") or []
        layer_counts = sorted({d.get("layer_count") for d in derived})
        tensor_counts = sorted({d.get("tensor_count") for d in derived})
        param_counts = sorted({d.get("total_parameters") for d in derived})
        n_distinct = census.get("structural_digests_distinct")
        ok = all_reconcile and not refusals
        components.append(StageComponent(
            component_id="II3-ARCHITECTURE-CENSUS",
            kind=ComponentKind.ARCHITECTURE_CENSUS,
            status=(StageOutcome.CONFORMANCE_VERIFIED if ok
                    else StageOutcome.PARTIAL),
            question=census_question,
            summary=(
                f"{len(models)} checkpoint(s) censused: layer_count "
                f"{layer_counts}, tensor_count {tensor_counts}, "
                f"total_parameters {param_counts}, "
                f"{n_distinct} distinct structural digest(s). "
                f"all_models_reconcile={all_reconcile}, "
                f"{len(refusals)} refusal(s). "
                "Every figure read from the bound census and reconciliation "
                "files, not transcribed."),
            evidence=evidence_map([c for c in census_checks if c.present]),
            limitations=[
                "a weights-only HDF5 cannot establish padding, dilation "
                "semantics, activations, residual wiring or input "
                "normalisation; the census records these under not_in_file",
                "the two shipped models share one structural digest because "
                "DBitNet is sized by block width and both ciphers are 32-bit. "
                "This is the expected result and says nothing about their "
                "trained weights",
            ]))
        if not ok:
            findings.append(
                f"II3-ARCHITECTURE-CENSUS: all_models_reconcile={all_reconcile}, "
                f"refusals={len(refusals)}.")

    # --- component 2: checkpoint load verification --------------------
    load_check = by_label.get("checkpoint_load_verification")
    load_state, load_problems = _evidence_state([load_check] if load_check else [])
    load = _read_json(load_check)
    load_question = ("Does each shipped checkpoint load completely into the model "
                     "the pinned source constructs, consuming every tensor?")
    if load_state == "MISMATCH":
        components.append(StageComponent(
            component_id="II3-CHECKPOINT-LOAD",
            kind=ComponentKind.DETERMINISTIC_EQUIVALENCE,
            status=StageOutcome.FAILED,
            question=load_question,
            summary=("load-verification evidence is present but is not the "
                     "expected content: " + "; ".join(load_problems)),
            evidence=evidence_map([load_check] if load_check and load_check.present
                                  else []),
            limitations=["a hash mismatch is an integrity failure"]))
        findings.append("II3-CHECKPOINT-LOAD integrity failure: "
                        + "; ".join(load_problems))
    elif load is None:
        components.append(StageComponent(
            component_id="II3-CHECKPOINT-LOAD",
            kind=ComponentKind.DETERMINISTIC_EQUIVALENCE,
            status=StageOutcome.NOT_PERFORMED,
            question=load_question,
            summary=("load-verification evidence is not present on this machine, "
                     "so no checkpoint was loaded and no equivalence is claimed"),
            evidence={},
            limitations=["absent evidence means this check did not run here"]))
        findings.append("II3-CHECKPOINT-LOAD: evidence absent on this machine; "
                        "the component was not performed here.")
    else:
        models = load.get("models") or []
        all_verified = bool(load.get("all_verified"))
        raised = [m.get("checkpoint") for m in models
                  if m.get("load_weights_raised")]
        unconsumed = sum(len(m.get("unconsumed_file_tensors") or [])
                         for m in models)
        unchanged = sum(len(m.get("unchanged_on_load") or []) for m in models)
        params = sorted({m.get("total_parameters_loaded") for m in models})
        tcf = sorted({m.get("tensor_count_file") for m in models})
        tcm = sorted({m.get("tensor_count_model") for m in models})
        names_agree = sorted({bool(m.get("layer_names_agree")) for m in models})
        ok = all_verified and not raised and unconsumed == 0
        components.append(StageComponent(
            component_id="II3-CHECKPOINT-LOAD",
            kind=ComponentKind.DETERMINISTIC_EQUIVALENCE,
            status=(StageOutcome.CONFORMANCE_VERIFIED if ok
                    else StageOutcome.PARTIAL),
            question=load_question,
            summary=(
                f"{len(models)} checkpoint(s) loaded: all_verified="
                f"{all_verified}, tensors file/model {tcf}/{tcm}, "
                f"{params} parameters loaded, {unconsumed} unconsumed, "
                f"{unchanged} unchanged-on-load, {len(raised)} load error(s). "
                f"layer_names_agree={names_agree} (recorded as observation "
                "only; matching is positional). Every figure read from the "
                "bound load-verification file."),
            evidence=evidence_map([load_check]),
            limitations=[
                "loadability is not provenance: it does not show the checkpoint "
                "was produced by this source or by the published training "
                "procedure",
                "matching is positional because Keras's layer-name counter is "
                "session-global; an earlier name-keyed verifier produced a "
                "spurious mismatch on the second model and was corrected",
            ]))
        if not ok:
            findings.append(
                f"II3-CHECKPOINT-LOAD: all_verified={all_verified}, "
                f"unconsumed={unconsumed}, load errors={len(raised)}.")

    # --- component 3: the paper-to-source architecture comparison -----
    # The evidence map binds REAL FILE HASHES. A content digest with no
    # file behind it is reported separately, never as a file hash.
    # The comparison is a claim about the PINNED source, so the cited
    # source files are verified before any AGREES verdict is allowed to
    # stand. Without that, an "8 AGREES" line would be an assertion about
    # files this process never opened.
    source_verification = psa.verify_sources(upstream)
    declared_summary = psa.summarise()
    arch_summary = psa.effective_summary(source_verification)
    differs = arch_summary["differs"]
    unresolved = (arch_summary["underspecified"]
                  + arch_summary["not_checkable_here"])
    arch_status = (StageOutcome.FAILED if differs
                   else StageOutcome.PARTIAL if unresolved
                   else StageOutcome.CONFORMANCE_VERIFIED)

    arch_evidence: dict[str, str] = {}
    arch_notes: list[str] = []
    if not source_verification["verified"]:
        arch_notes.append(
            "the cited source files were NOT verified against the pinned "
            "checkout, so every AGREES verdict is reported as "
            "NOT_CHECKABLE_HERE: "
            + "; ".join(source_verification["problems"]))
    else:
        for rel, rec in sorted(source_verification["files"].items()):
            if rec.get("observed_sha256"):
                arch_evidence[str(Path(upstream) / rel)] = rec["observed_sha256"]
    module_file = Path(psa.__file__)
    if module_file.is_file():
        arch_evidence[str(module_file)] = sha256_file(module_file)
    record_check = by_label.get("paper_source_architecture_record")
    if record_check is not None and record_check.present:
        arch_evidence[record_check.path] = record_check.observed_sha256
        stored = _read_json(record_check)
        live = psa.record_digest()
        if stored is not None and stored.get("record_digest") != live:
            arch_notes.append(
                f"the serialised record at {record_check.path} carries "
                f"record_digest {str(stored.get('record_digest'))[:16]}... but the "
                f"live comparison computes {live[:16]}.... The file is stale; "
                "regenerate it with --write-architecture-record.")
    else:
        arch_notes.append(
            "the full machine-readable comparison has not been serialised on this "
            f"machine. Write it with --write-architecture-record; it lands at "
            "audit/targets/autond/evidence/architecture/"
            "paper_source_architecture_v2.json and is then bound by its own file "
            "hash.")

    components.append(StageComponent(
        component_id="II3-PAPER-SOURCE-ARCHITECTURE",
        kind=ComponentKind.REFERENCE_VERIFICATION,
        status=arch_status,
        question=("Does the architecture the publication describes match the "
                  "architecture the pinned source builds?"),
        summary=(f"{arch_summary['total_claims']} properties compared against "
                 f"ePrint {psa.PAPER['eprint_id']} "
                 f"({psa.PAPER['revision_compared']}); source verified: "
                 f"{source_verification['verified']}. Effective verdicts: "
                 f"{arch_summary['by_verdict']}. "
                 f"Agreeing: {', '.join(arch_summary['agrees']) or 'none'}. "
                 f"{len(differs)} differ. "
                 f"{len(unresolved)} not comparable here. Declared verdicts "
                 f"(against the pinned source) were "
                 f"{declared_summary['by_verdict']}. Comparison content digest "
                 f"{psa.record_digest(source_verification)[:16]}... (not a "
                 "file hash)."),
        evidence=arch_evidence,
        limitations=psa.does_not_establish() + arch_notes + [
            "the paper PDF's own sha256 is NOT bound: the environment that built "
            f"this comparison could not fetch it. {psa.PAPER['pdf_sha256_status']}",
            "UNDERSPECIFIED verdicts mean the claim was not located in the "
            "authoritative revision's text. Figure 4c, which may state several "
            "of them, carries no machine-extractable text",
        ]))
    if unresolved:
        findings.append(
            f"II3-PAPER-SOURCE-ARCHITECTURE: {len(unresolved)} architectural "
            f"properties unresolved ({', '.join(unresolved)}); the architecture "
            "conformance claim cannot be adjudicated on the census alone.")
    if differs:
        findings.append(
            f"II3-PAPER-SOURCE-ARCHITECTURE: {len(differs)} property(ies) DIFFER "
            f"between the publication and the pinned source: {differs}.")

    satisfied = [c for c in components if c.status in SATISFYING]
    stage_status = (StageOutcome.CONFORMANCE_VERIFIED
                    if len(satisfied) == len(components)
                    else StageOutcome.FAILED
                    if any(c.status is StageOutcome.FAILED for c in components)
                    else StageOutcome.PARTIAL)
    stage = StageStatus(
        stage_id=StageId.II_3_CONTROLLED_VERIFICATION,
        tier=STAGE_TIERS[StageId.II_3_CONTROLLED_VERIFICATION],
        status=stage_status,
        summary=_stage_summary("II-3 controlled verification", components),
        findings=findings,
        components=tuple(components))
    return stage, {"checks": [c.to_dict() for c in checks],
                   "architecture_comparison": arch_summary,
                   "architecture_declared_summary": declared_summary,
                   "source_verification": source_verification,
                   "architecture_record_digest":
                       psa.record_digest(source_verification)}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_record(repo: Path, home: Path, upstream: Path, strict: bool) -> dict:
    """
    Build both stage records and decide whether the result may be written.

    FAIL-CLOSED, WITH THREE DISTINCT GATES.

      stage_record_problems   the generic validator rejected a record.
      blocking_problems       something is wrong regardless of --strict: a
                              hash mismatch, a wrong upstream commit, or a
                              stage that came out FAILED. Revision 2 let
                              all three pass through `ok` whenever
                              --strict was off, so a record built against
                              the wrong revision of the target reported
                              ok=True and exit 0.
      strict_problems         --strict only: a required artifact whose
                              integrity is not established.

    An artifact's integrity counts as established when it is VERIFIED
    against its own recorded digest, OR when it is bound by a manifest the
    search harness validated. The R1 files have no separately hard-coded
    digest by design - the manifest is their binding - so revision 2's
    --strict could never succeed on complete, valid R1 evidence.
    """
    ii1, ii1_detail = build_ii1(repo, home, upstream)
    ii3, ii3_detail = build_ii3(repo, home, upstream)
    problems: list[str] = []
    for st in (ii1, ii3):
        problems.extend(validate_stage_record(st))

    all_checks = ii1_detail["checks"]
    manifest_verified = set(ii1_detail.get("r1", {}).get("manifest_verified") or [])

    def established(c: dict) -> bool:
        if c["verdict"] == "VERIFIED":
            return True
        return (c["verdict"] == "RECORDED_ONLY"
                and c["label"] in manifest_verified)

    unestablished_required = [c for c in all_checks
                              if c["required"] and not established(c)]
    strict_problems: list[str] = []
    if strict and unestablished_required:
        strict_problems.append(
            f"--strict: {len(unestablished_required)} required artifact(s) "
            "have no established integrity on this machine: "
            f"{[c['label'] for c in unestablished_required]}")

    blocking_problems: list[str] = []
    mismatched = [c for c in all_checks if c["verdict"] == "HASH_MISMATCH"]
    if mismatched:
        blocking_problems.append(
            f"{len(mismatched)} artifact(s) HASH_MISMATCH: "
            f"{[c['label'] for c in mismatched]}. This is an integrity failure "
            "whether or not --strict was passed.")
    if not ii1_detail["commit_ok"]:
        blocking_problems.append(
            f"upstream checkout is not the pinned revision: "
            f"{ii1_detail['commit_detail'] or 'commit could not be read'}. "
            "Nothing read from it describes the pinned target, so no stage "
            "record built here may be treated as valid.")
    for st in (ii1, ii3):
        if st.status is StageOutcome.FAILED:
            blocking_problems.append(
                f"{st.stage_id.value} came out FAILED: {st.summary}")

    ok = not (problems or blocking_problems or strict_problems)
    return {
        "artifact": ADAPTER_VERSION,
        "built_at_utc": utc_now(),
        "target": {"name": "AutoND", "pinned_commit": PINNED_COMMIT,
                   "upstream_root": str(upstream)},
        "repo_root": str(repo),
        "stages": [ii1.to_dict(), ii3.to_dict()],
        "stage_record_problems": problems,
        "blocking_problems": blocking_problems,
        "strict_problems": strict_problems,
        "strict": bool(strict),
        "ok": ok,
        "integrity_established": sorted(
            c["label"] for c in all_checks if established(c)),
        "manifest_verified": sorted(manifest_verified),
        "artifact_checks": all_checks,
        "upstream_commit": {"ok": ii1_detail["commit_ok"],
                            "head": ii1_detail["head"],
                            "detail": ii1_detail["commit_detail"]},
        "r1_validation": ii1_detail["r1"],
        "r1_scientific_result": ii1_detail["r1_science"],
        "evaluation_ingest": {
            k: v for k, v in ii1_detail["evaluation_ingest"].items()
            if k != "result"},
        "architecture_comparison": ii3_detail["architecture_comparison"],
        "architecture_declared_summary": ii3_detail["architecture_declared_summary"],
        "source_verification": ii3_detail["source_verification"],
        "architecture_record_digest": ii3_detail["architecture_record_digest"],
        "paper": psa.PAPER,
        "mode": "validate (no experiment was executed by this module)",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build AutoND II-1 and II-3 stage records from existing "
                    "evidence. Runs no experiment.")
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--home", type=Path, default=Path.home())
    ap.add_argument("--autond-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="fail closed: exit non-zero AND write no output unless "
                         "every required artifact is VERIFIED")
    ap.add_argument("--write-architecture-record", type=Path, default=None,
                    metavar="PATH",
                    help="serialise the paper-to-source comparison so the stage "
                         "can bind the sha256 of a real file (default path: "
                         "audit/targets/autond/evidence/architecture/"
                         "paper_source_architecture_v2.json)")
    args = ap.parse_args(argv)

    if args.write_architecture_record is not None:
        info = psa.write_record(args.write_architecture_record)
        print(f"architecture record written: {info['path']}")
        print(f"  file sha256:   {info['sha256']}")
        print(f"  record digest: {info['record_digest']}\n")

    rec = build_record(args.repo_root.resolve(), args.home.expanduser(),
                       args.autond_root.expanduser().resolve(), args.strict)

    print(f"{ADAPTER_VERSION}   mode: validate, no experiment executed\n")
    print(f"  upstream commit      {rec['upstream_commit']['head'] or '(unreadable)'}"
          f"   ok={rec['upstream_commit']['ok']}")
    print("\n  ARTIFACT INTEGRITY")
    for c in rec["artifact_checks"]:
        flag = "  " if c["verdict"] in ("VERIFIED", "RECORDED_ONLY") else "!!"
        req = "required" if c["required"] else "optional"
        print(f"  {flag} {c['verdict']:14} {c['label']:32} [{req}, {c['provenance']}]")
    print("\n  STAGES")
    for st in rec["stages"]:
        print(f"    {st['stage_id']}  {st['status']}  (tier {st['tier']})")
        for comp in st.get("components", []):
            print(f"       {comp['status']:22} {comp['component_id']}  "
                  f"[{comp['kind']}]  evidence: {len(comp['evidence'])}")
        for f in st["findings"]:
            print(f"       FINDING  {f}")
    if rec["stage_record_problems"]:
        print("\n  STAGE RECORD PROBLEMS")
        for p in rec["stage_record_problems"]:
            print(f"    - {p}")
    else:
        print("\n  stage records pass validate_stage_record()")

    if rec["blocking_problems"]:
        print("\n  BLOCKING PROBLEMS (independent of --strict)")
        for p in rec["blocking_problems"]:
            print(f"    - {p}")
    if rec["strict_problems"]:
        print("\n  STRICT PROBLEMS")
        for p in rec["strict_problems"]:
            print(f"    - {p}")
    if rec["manifest_verified"]:
        print(f"\n  integrity established via a validated manifest: "
              f"{rec['manifest_verified']}")

    # Fail closed: a record that did not pass is not written at all, so a
    # downstream consumer cannot pick up a file that failed validation.
    blocked = not rec["ok"]
    if args.output:
        if blocked:
            print(f"\n  REFUSING to write {args.output}: validation did not pass. "
                  "A record that failed validation must not be left on disk for "
                  "something else to read.")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(rec, indent=2, sort_keys=True, default=str))
            print(f"\n  wrote {args.output}")

    # Derived, not asserted: say what the stages actually came out as.
    print()
    for st in rec["stages"]:
        print(f"  {st['stage_id']}: {st['status']}")
    print("  Statuses above are computed from the evidence present on THIS "
          "machine.")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
