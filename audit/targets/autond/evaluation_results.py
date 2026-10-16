#!/usr/bin/env python3
"""
Schema and validator for the shipped-checkpoint evaluation result bundle.

WHY THIS EXISTS
    The II-1 component `II1-SHIPPED-CHECKPOINT-EVAL` must not be wired to
    report NOT_PERFORMED forever. When the preregistered evaluation is
    eventually run, it will write a result bundle; this module defines
    exactly what that bundle must contain and refuses anything that does
    not conform. `implementation_stages` then ingests a bundle that
    validates and builds a real component from it.

    Nothing here runs an evaluation. It validates files that already
    exist, or reports that they do not.

WHAT A CONFORMING BUNDLE LOOKS LIKE

    <bundle>/
      RESULT.json          the record described by RESULT_SCHEMA below
      <per-checkpoint npz> raw accs/tprs/tnrs, one per checkpoint
      SHA256SUMS.txt       digest of every file in the bundle but itself
      env/pip_freeze.txt   the executing environment
      logs/<name>.log      full stdout/stderr of each evaluation

    RESULT.json binds the plan that authorised the run, the checkpoint
    digests actually evaluated, and the per-set accuracies. A bundle whose
    recorded plan hash is not the frozen plan's hash is REJECTED: a result
    cannot be confirmatory evidence for a plan it was not run under.

VALIDATION IS NOT ADJUDICATION
    `validate_bundle` establishes that the bundle is intact, complete and
    bound to the frozen plan. It does not decide whether the result
    supports a hypothesis. That decision belongs to the plan's own
    decision rule (see autond_ii_v2.decide), applied to a validated
    bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "autond-evaluation-result-v2"

#: Required structure of RESULT.json. Each entry is
#: (dotted key, python type, description). A missing or wrongly typed
#: key is a validation failure, never a warning.
RESULT_SCHEMA: tuple[tuple[str, type, str], ...] = (
    ("schema_version", str, "must equal SCHEMA_VERSION"),
    ("preregistration.id", str, "the plan that authorised this run"),
    ("preregistration.plan_hash", str,
     "the plan hash at freeze time; must match the frozen sidecar"),
    ("preregistration.frozen_at_utc", str, "freeze timestamp copied from the sidecar"),
    ("execution.started_utc", str, "ISO-8601 UTC"),
    ("execution.finished_utc", str, "ISO-8601 UTC"),
    ("execution.command", str, "the exact command line executed"),
    ("execution.upstream_commit", str, "HEAD of the checkout evaluated against"),
    ("execution.upstream_write_check", str,
     "evidence that the upstream checkout was not written to"),
    ("environment.python", str, "python version string"),
    ("environment.tensorflow", str, "tensorflow version string"),
    ("environment.numpy", str, "numpy version string"),
    ("checkpoints", dict,
     "per-checkpoint records keyed by cipher id, each with the fields in "
     "CHECKPOINT_SCHEMA"),
)

CHECKPOINT_SCHEMA: tuple[tuple[str, type, str], ...] = (
    ("path", str, "path of the COPY that was evaluated, outside the checkout"),
    ("sha256", str, "digest of the copy actually evaluated"),
    ("upstream_sha256", str, "digest recorded for this checkpoint in the plan"),
    ("round_number", int, "round the checkpoint distinguishes"),
    ("input_difference", str, "the fixed input difference"),
    ("accuracies", list,
     "one accuracy per fresh test set, individually retained, pooled across "
     "the registered invocations in order"),
    ("tprs", list, "one TPR per test set"),
    ("tnrs", list, "one TNR per test set"),
    ("npz", (str, list),
     "bundle-relative path(s) of the raw npz, one per invocation in the "
     "registered suffix order; a bare string is a pre-r3d bundle"),
    ("npz_sha256", (str, list),
     "digest(s) of the raw npz, positionally matching npz"),
    ("n_samples_per_set", int, "samples in each fresh test set"),
)

#: Fixed by upstream eval_nets.py module constants, not by this schema.
#: This is the count PER INVOCATION (eval_nets.nEval), which is why it
#: did not change when the registered sample size rose to ten.
EXPECTED_SETS_PER_INVOCATION = 5

#: Retained under its original name: callers and tests written against
#: revision 3 use it to mean "accuracies in one NPZ", which is still
#: exactly what it is.
EXPECTED_N_SETS = EXPECTED_SETS_PER_INVOCATION

#: The registered design: invocations per checkpoint, and the pooled
#: sample size the analysis uses.
#:
#: These are stated here rather than imported from the preregistration
#: because the preregistration imports THIS module (for
#: mean_and_t_interval), so importing it back would be circular. They are
#: therefore a DUPLICATE of the plan's SETS_PER_INVOCATION,
#: INVOCATIONS_PER_CHECKPOINT and N_TEST_SETS, and a duplicate that
#: silently disagreed with the plan would be worse than no constant at
#: all: the validator would enforce a sample size the plan never
#: registered. Agreement is asserted by assert_matches_preregistration()
#: below, which the assembler calls before it reads any NPZ, and by a
#: test that fails if either side is edited alone.
EXPECTED_INVOCATIONS_PER_CHECKPOINT = 2
EXPECTED_N_POOLED = EXPECTED_SETS_PER_INVOCATION * \
    EXPECTED_INVOCATIONS_PER_CHECKPOINT


def assert_matches_preregistration() -> dict:
    """
    Verify this module's design constants against the plan's.

    Imported lazily, inside the function, precisely because the plan
    imports this module at its own import time; by the time anyone calls
    this, both modules exist and the cycle is not a problem.

    Raises ValueError on disagreement rather than returning a flag: a
    mismatch means the validator and the registered design have diverged,
    and nothing downstream should proceed on the assumption that the
    numbers it is checking are the ones that were preregistered.
    """
    from audit.targets.autond.preregistrations import autond_ii_v2 as _plan

    pairs = (
        ("sets per invocation", EXPECTED_SETS_PER_INVOCATION,
         _plan.SETS_PER_INVOCATION),
        ("invocations per checkpoint", EXPECTED_INVOCATIONS_PER_CHECKPOINT,
         _plan.INVOCATIONS_PER_CHECKPOINT),
        ("pooled sample size", EXPECTED_N_POOLED, _plan.N_TEST_SETS),
        ("pooled sample size (DESIGN)", EXPECTED_N_POOLED,
         _plan.DESIGN["n_test_sets"]),
    )
    bad = [f"{what}: this module says {mine}, the preregistration says {theirs}"
           for what, mine, theirs in pairs if mine != theirs]
    if bad:
        raise ValueError(
            "the result validator's design constants disagree with the "
            "registered preregistration, so it would enforce a design that "
            "was never registered: " + "; ".join(bad))
    return {"sets_per_invocation": EXPECTED_SETS_PER_INVOCATION,
            "invocations_per_checkpoint": EXPECTED_INVOCATIONS_PER_CHECKPOINT,
            "n_pooled": EXPECTED_N_POOLED}

EXPECTED_SAMPLES_PER_SET = 10 ** 6


def _type_name(typ) -> str:
    """Readable name for a schema type that may be a tuple of types."""
    if isinstance(typ, tuple):
        return " or ".join(t.__name__ for t in typ)
    return typ.__name__


def _dig(obj: Any, dotted: str) -> tuple[bool, Any]:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


# --------------------------------------------------------------------------
# Student-t: re-exported from the GENERIC module
#
# These lived here in revision 3. They are not AutoND-specific, and the
# preregistration needs the same functions, so keeping a second copy here
# would be exactly the duplicated generic logic the methodology forbids.
# They now live in audit.common.equivalence and are re-exported under
# their original names so existing callers and tests keep working.
# --------------------------------------------------------------------------

from audit.common.equivalence import (          # noqa: E402
    regularised_incomplete_beta,
    student_t_cdf,
    student_t_ppf,
    student_t_sf,
)

__all__ = [
    "SCHEMA_VERSION", "RESULT_SCHEMA", "CHECKPOINT_SCHEMA",
    "EXPECTED_N_SETS", "EXPECTED_SETS_PER_INVOCATION",
    "EXPECTED_INVOCATIONS_PER_CHECKPOINT", "EXPECTED_N_POOLED",
    "assert_matches_preregistration",
    "EXPECTED_SAMPLES_PER_SET", "ATTESTATION_RE",
    "regularised_incomplete_beta", "student_t_cdf", "student_t_ppf",
    "student_t_sf", "mean_and_t_interval", "validate_bundle", "sha256_file",
]


def mean_and_t_interval(values: list[float], confidence: float = 0.95
                        ) -> dict[str, Optional[float]]:
    """
    Mean with a Student-t interval, computed without scipy.

    Returns the mean, the sample SD, the standard error, the half-width
    and the interval. With fewer than two values the interval is None
    rather than a fabricated zero-width one.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sample_sd": None, "se": None,
                "half_width": None, "low": None, "high": None}
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "sample_sd": None, "se": None,
                "half_width": None, "low": None, "high": None,
                "note": "a single observation supports no interval"}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    df = n - 1
    t_crit = student_t_ppf(1.0 - (1.0 - confidence) / 2.0, df)
    half = t_crit * se
    return {"n": n, "mean": mean, "sample_sd": sd, "se": se,
            "half_width": half, "low": mean - half, "high": mean + half,
            "t_critical": t_crit, "df": df, "confidence": confidence}


#: The exact grammar the launcher writes for the upstream cleanliness
#: attestation. Anything else is rejected: revision 2 accepted any string
#: containing "clean" or "unmodified", so "not clean" passed the check.
ATTESTATION_RE = re.compile(
    r"^tracked-clean;untracked-benign=(0|[1-9]\d*)(?::[^;]+)?$")


def _parse_utc(value: object) -> tuple[bool, object]:
    """Parse an ISO-8601 UTC instant, returning (ok, datetime|None)."""
    if not isinstance(value, str) or not _ISO_UTC_RE.match(value):
        return False, None
    try:
        from datetime import datetime
        return True, datetime.strptime(
            value.replace(".000", ""), "%Y-%m-%dT%H:%M:%SZ")
    except Exception:                                               # noqa: BLE001
        try:
            from datetime import datetime
            return True, datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:                                           # noqa: BLE001
            return False, None


_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def _safe_relpath(bundle: Path, rel: str) -> tuple[bool, Optional[Path]]:
    """
    Resolve a bundle-relative path, refusing anything that escapes.

    A record naming "../../etc/passwd" or an absolute path must not be
    followed: the digests in a bundle describe the bundle's own contents.
    """
    if not isinstance(rel, str) or not rel:
        return False, None
    candidate = Path(rel)
    if candidate.is_absolute() or ".." in candidate.parts:
        return False, None
    p = (bundle / candidate).resolve()
    try:
        p.relative_to(Path(bundle).resolve())
    except ValueError:
        return False, None
    return True, p


def _npz_metrics(path: Path) -> tuple[Optional[dict], Optional[str]]:
    try:
        import numpy as np
    except Exception as exc:                                        # noqa: BLE001
        return None, (f"numpy is not importable ({type(exc).__name__}), so the "
                      "NPZ could not be checked against the record. This is a "
                      "verification failure, not a pass.")
    try:
        with np.load(path) as z:
            return ({k: [float(v) for v in np.asarray(z[k]).ravel()]
                     for k in ("accs", "tprs", "tnrs") if k in z.files},
                    None)
    except Exception as exc:                                        # noqa: BLE001
        return None, f"{path.name} could not be loaded as an NPZ ({exc})"


def validate_bundle(bundle_dir: Path, *, expected_plan_hash: Optional[str] = None,
                    expected_checkpoint_hashes: Optional[dict[str, str]] = None,
                    expected_upstream_commit: Optional[str] = None,
                    expected_checkpoint_ids: Optional[tuple] = None,
                    require_plan_binding: bool = True,
                    ) -> dict:
    """
    Validate an evaluation result bundle.

    Returns {"ok", "problems", "record", "derived", "plan_bound"}. `ok` is
    True only if every check passes. A bundle that does not exist is
    reported with ok False; it is never an empty success.

    `plan_bound` is True only when an expected plan hash was supplied AND
    matched. With `require_plan_binding` (the default) a bundle validated
    without an expected hash is a FAILURE, because "valid" and "valid for
    this plan" are different statements and only the second one makes a
    result confirmatory.

    Defaults are taken from the preregistration when the caller does not
    override them, so the ordinary call checks the checkpoint digests,
    the checkpoint set and the pinned commit rather than skipping all
    three.
    """
    d = Path(bundle_dir).expanduser()
    problems: list[str] = []
    out: dict = {"ok": False, "problems": problems, "record": None,
                 "derived": {}, "bundle_dir": str(d), "plan_bound": False}

    # --- defaults from the plan ---------------------------------------
    plan_mod = None
    try:
        from audit.targets.autond.preregistrations import autond_ii_v2 as plan_mod
    except Exception:                                               # noqa: BLE001
        plan_mod = None
    if plan_mod is not None:
        if expected_checkpoint_hashes is None:
            expected_checkpoint_hashes = {
                cid: rec["sha256"]
                for cid, rec in plan_mod.TARGET["checkpoints"].items()}
        if expected_checkpoint_ids is None:
            expected_checkpoint_ids = tuple(sorted(plan_mod.TARGET["checkpoints"]))
        if expected_upstream_commit is None:
            expected_upstream_commit = plan_mod.TARGET["pinned_commit"]

    if not d.is_dir():
        problems.append(f"no bundle directory at {d}")
        return out

    rpath = d / "RESULT.json"
    if not rpath.is_file():
        problems.append(f"no RESULT.json in {d}; the evaluation did not finish "
                        "writing its bundle")
        return out
    try:
        record = json.loads(rpath.read_text())
    except Exception as exc:                                        # noqa: BLE001
        problems.append(f"RESULT.json is not readable JSON: "
                        f"{type(exc).__name__}: {exc}")
        return out
    out["record"] = record

    # --- 1. top-level schema ------------------------------------------
    for dotted, typ, _desc in RESULT_SCHEMA:
        present, value = _dig(record, dotted)
        if not present:
            problems.append(f"RESULT.json missing required key {dotted}")
        elif not isinstance(value, typ):
            problems.append(f"RESULT.json {dotted}: expected "
                            f"{_type_name(typ)}, "
                            f"got {type(value).__name__}")

    if record.get("schema_version") not in (None, SCHEMA_VERSION):
        problems.append(f"schema_version {record.get('schema_version')!r} != "
                        f"{SCHEMA_VERSION!r}")

    # --- 2. binding to the frozen plan --------------------------------
    got_hash = record.get("preregistration", {}).get("plan_hash")
    if expected_plan_hash is None:
        if require_plan_binding:
            problems.append(
                "no expected plan hash was supplied, so this bundle was NOT "
                "checked against a frozen plan. Validation of structure alone "
                "does not make a result confirmatory; pass the frozen plan's "
                "hash, or set require_plan_binding=False to assert structure "
                "only and accept that the result is not plan-bound.")
    else:
        if not _is_sha256(got_hash):
            problems.append(f"preregistration.plan_hash is not a sha256: "
                            f"{got_hash!r}")
        elif got_hash != expected_plan_hash:
            problems.append(
                f"this bundle records plan_hash {got_hash}, but the frozen plan "
                f"hashes to {expected_plan_hash}. A result is not confirmatory "
                "evidence for a plan it was not run under.")
        else:
            out["plan_bound"] = True

    # --- 3. timestamps: real values, correctly ordered ----------------
    ex = record.get("execution", {})
    ok_s, t_start = _parse_utc(ex.get("started_utc"))
    ok_f, t_finish = _parse_utc(ex.get("finished_utc"))
    if not ok_s:
        problems.append(f"execution.started_utc {ex.get('started_utc')!r} is "
                        "not an ISO-8601 UTC instant of the form "
                        "YYYY-MM-DDTHH:MM:SSZ")
    if not ok_f:
        problems.append(f"execution.finished_utc {ex.get('finished_utc')!r} is "
                        "not an ISO-8601 UTC instant")
    if ok_s and ok_f and t_finish < t_start:
        problems.append(
            f"execution.finished_utc {ex['finished_utc']} precedes "
            f"started_utc {ex['started_utc']}; a run cannot end before it began")
    frozen_at = record.get("preregistration", {}).get("frozen_at_utc")
    ok_fz, t_frozen = _parse_utc(frozen_at)
    if not ok_fz:
        problems.append(f"preregistration.frozen_at_utc {frozen_at!r} is not an "
                        "ISO-8601 UTC instant")
    elif ok_s and t_start < t_frozen:
        problems.append(
            f"the run started {ex['started_utc']} but the plan was frozen "
            f"{frozen_at}. A confirmatory run cannot precede its own "
            "preregistration.")

    # --- 4. the pinned commit -----------------------------------------
    commit = ex.get("upstream_commit")
    if expected_upstream_commit is not None:
        if commit != expected_upstream_commit:
            problems.append(
                f"execution.upstream_commit {commit!r} is not the pinned "
                f"{expected_upstream_commit!r}")
    elif not (isinstance(commit, str) and len(commit) == 40):
        problems.append(f"execution.upstream_commit {commit!r} is not a "
                        "40-character commit id")

    # --- 5. the cleanliness attestation, exact grammar ----------------
    wc = ex.get("upstream_write_check")
    if not isinstance(wc, str) or not ATTESTATION_RE.match(wc):
        problems.append(
            f"execution.upstream_write_check {wc!r} does not match the accepted "
            f"grammar {ATTESTATION_RE.pattern!r}. The launcher writes "
            "'tracked-clean;untracked-benign=<n>[:<names>]'; a free-text "
            "attestation is not checkable, and a substring test for the word "
            "'clean' would accept 'not clean'.")

    # --- 6. SHA256SUMS covers the bundle and matches ------------------
    sums = d / "SHA256SUMS.txt"
    if not sums.is_file():
        problems.append("no SHA256SUMS.txt in the bundle")
    else:
        declared: dict[str, str] = {}
        for line in sums.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                problems.append(f"SHA256SUMS.txt: unparseable line {line!r}")
                continue
            digest, name = parts[0], parts[1].strip()
            if not _is_sha256(digest):
                problems.append(f"SHA256SUMS.txt: {name}: {digest!r} is not a "
                                "sha256")
                continue
            ok_rel, _p = _safe_relpath(d, name)
            if not ok_rel:
                problems.append(f"SHA256SUMS.txt names {name!r}, which escapes "
                                "the bundle directory")
                continue
            declared[name] = digest
        for name, digest in sorted(declared.items()):
            p = d / name
            if not p.is_file():
                problems.append(f"SHA256SUMS.txt names {name}, which is missing")
                continue
            got = sha256_file(p)
            if got != digest:
                problems.append(f"{name}: sha256 {got} != declared {digest}")
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS.txt":
                rel = str(p.relative_to(d))
                if rel not in declared:
                    problems.append(f"{rel} is present but not covered by "
                                    "SHA256SUMS.txt")

    # --- 7. per-checkpoint records ------------------------------------
    checkpoints = record.get("checkpoints")
    derived: dict[str, Any] = {}
    if not isinstance(checkpoints, dict) or not checkpoints:
        problems.append("RESULT.json checkpoints is missing or empty")
    else:
        if expected_checkpoint_ids:
            missing_ids = sorted(set(expected_checkpoint_ids) - set(checkpoints))
            unexpected = sorted(set(checkpoints) - set(expected_checkpoint_ids))
            if missing_ids:
                problems.append(
                    f"the plan covers {list(expected_checkpoint_ids)} but the "
                    f"bundle has no result for {missing_ids}. The hypothesis "
                    "family spans both checkpoints; a partial bundle is not a "
                    "valid result for it.")
            if unexpected:
                problems.append(f"the bundle reports checkpoints the plan does "
                                f"not list: {unexpected}")

        for cid, rec in sorted(checkpoints.items()):
            if not isinstance(rec, dict):
                problems.append(f"checkpoints.{cid} is not an object")
                continue
            for dotted, typ, _desc in CHECKPOINT_SCHEMA:
                present, value = _dig(rec, dotted)
                if not present:
                    problems.append(f"checkpoints.{cid} missing {dotted}")
                elif not isinstance(value, typ):
                    problems.append(f"checkpoints.{cid}.{dotted}: expected "
                                    f"{_type_name(typ)}, got "
                                    f"{type(value).__name__}")

            for field in ("sha256", "upstream_sha256"):
                if field in rec and not _is_sha256(rec[field]):
                    problems.append(f"checkpoints.{cid}.{field} is not a sha256: "
                                    f"{rec[field]!r}")

            # npz_sha256 is per-invocation: a list of digests under the
            # registered design, a bare string in a pre-r3d bundle. Every
            # entry must still be a sha256, so the shape change does not
            # become a hole in the digest check.
            if "npz_sha256" in rec:
                raw = rec["npz_sha256"]
                entries = raw if isinstance(raw, list) else [raw]
                if not entries:
                    problems.append(f"checkpoints.{cid}.npz_sha256 is empty; "
                                    "the raw artifacts are unidentified")
                for i, dig in enumerate(entries):
                    if not _is_sha256(dig):
                        problems.append(
                            f"checkpoints.{cid}.npz_sha256[{i}] is not a "
                            f"sha256: {dig!r}")
                if len(set(map(str, entries))) != len(entries):
                    problems.append(
                        f"checkpoints.{cid}.npz_sha256 repeats a digest "
                        f"{entries}; two invocations drawing from os.urandom "
                        "cannot produce byte-identical artifacts, so one file "
                        "was read twice")

            want = (expected_checkpoint_hashes or {}).get(cid)
            if want is not None:
                if rec.get("upstream_sha256") != want:
                    problems.append(
                        f"checkpoints.{cid}.upstream_sha256 "
                        f"{rec.get('upstream_sha256')} != the plan's recorded "
                        f"{want}")
                if rec.get("sha256") != want:
                    problems.append(
                        f"checkpoints.{cid}: the copy evaluated hashes to "
                        f"{rec.get('sha256')}, not the plan's recorded {want}. "
                        "The copy is not the checkpoint the plan authorised.")
            elif expected_checkpoint_hashes is not None:
                problems.append(f"checkpoints.{cid}: no expected digest is "
                                "known for this id, so it was not checked")

            # the checkpoint copy must be in the bundle and hash correctly
            ok_rel, cpath = _safe_relpath(d, rec.get("path", ""))
            if not ok_rel:
                problems.append(f"checkpoints.{cid}.path "
                                f"{rec.get('path')!r} escapes the bundle "
                                "directory or is not set")
            elif not cpath.is_file():
                problems.append(f"checkpoints.{cid}: the copy evaluated "
                                f"({rec.get('path')}) is not in the bundle")
            elif _is_sha256(rec.get("sha256")) and sha256_file(cpath) != rec["sha256"]:
                problems.append(f"checkpoints.{cid}: the copy on disk does not "
                                "hash to the recorded sha256")

            accs = rec.get("accuracies")
            if isinstance(accs, list):
                if len(accs) != EXPECTED_N_POOLED:
                    problems.append(
                        f"checkpoints.{cid}: {len(accs)} accuracies, expected "
                        f"{EXPECTED_N_POOLED} - "
                        f"{EXPECTED_INVOCATIONS_PER_CHECKPOINT} invocations of "
                        f"{EXPECTED_SETS_PER_INVOCATION} sets each, the latter "
                        f"fixed by upstream eval_nets.nEval. A count of "
                        f"{EXPECTED_SETS_PER_INVOCATION} means only one "
                        f"invocation's NPZ was pooled, which the plan's "
                        f"missing-data rule reports as INCONCLUSIVE rather than "
                        f"analysing as a smaller registered design.")
                bad = [a for a in accs
                       if not isinstance(a, (int, float)) or not 0.0 <= a <= 1.0]
                if bad:
                    problems.append(f"checkpoints.{cid}: accuracies outside "
                                    f"[0,1] or non-numeric: {bad}")
                else:
                    derived[cid] = mean_and_t_interval([float(a) for a in accs])
            for field in ("tprs", "tnrs"):
                seq = rec.get(field)
                if isinstance(seq, list) and isinstance(accs, list) \
                        and len(seq) != len(accs):
                    problems.append(f"checkpoints.{cid}.{field} has {len(seq)} "
                                    f"entries but accuracies has {len(accs)}")
            n_samp = rec.get("n_samples_per_set")
            if isinstance(n_samp, int) and n_samp != EXPECTED_SAMPLES_PER_SET:
                problems.append(
                    f"checkpoints.{cid}.n_samples_per_set {n_samp} != "
                    f"{EXPECTED_SAMPLES_PER_SET} (upstream "
                    "eval_nets.num_val_samples)")

            # --- the record's metrics must BE the NPZs' metrics -------
            #
            # The registered design pools two invocations, so `npz` and
            # `npz_sha256` are LISTS, one entry per invocation in the
            # registered suffix order. A single string is still accepted:
            # a bundle produced before the two-invocation design is read,
            # and then rejected by the pooled-count check above rather
            # than crashing here.
            raw_npz = rec.get("npz", "")
            raw_sha = rec.get("npz_sha256", "")
            npz_list = raw_npz if isinstance(raw_npz, list) else [raw_npz]
            sha_list = raw_sha if isinstance(raw_sha, list) else [raw_sha]

            if len(sha_list) != len(npz_list):
                problems.append(
                    f"checkpoints.{cid}: {len(npz_list)} npz path(s) but "
                    f"{len(sha_list)} digest(s); each raw artifact must carry "
                    "its own digest")

            if len(set(map(str, npz_list))) != len(npz_list):
                problems.append(
                    f"checkpoints.{cid}.npz lists the same path twice "
                    f"({npz_list}); two invocations that read one file are not "
                    "two independent draws")

            # Verified per invocation, then the CONCATENATION is compared
            # with the pooled record, so pooling itself is checked rather
            # than assumed.
            pooled: dict[str, list] = {"accs": [], "tprs": [], "tnrs": []}
            readable = True
            for i, entry in enumerate(npz_list):
                ok_rel, npath = _safe_relpath(d, entry or "")
                if not ok_rel:
                    problems.append(
                        f"checkpoints.{cid}.npz[{i}] {entry!r} escapes the "
                        "bundle directory or is not set")
                    readable = False
                    continue
                if not npath.is_file():
                    problems.append(f"checkpoints.{cid}: npz {entry} is missing")
                    readable = False
                    continue
                want = sha_list[i] if i < len(sha_list) else None
                if _is_sha256(want) and sha256_file(npath) != want:
                    problems.append(
                        f"checkpoints.{cid}: npz digest mismatch for {entry}")
                metrics, merr = _npz_metrics(npath)
                if merr:
                    problems.append(f"checkpoints.{cid}: {merr}")
                    readable = False
                    continue
                if metrics is None:
                    readable = False
                    continue
                for key in ("accs", "tprs", "tnrs"):
                    actual = metrics.get(key)
                    if actual is None:
                        problems.append(f"checkpoints.{cid}: the NPZ {entry} "
                                        f"has no '{key}' array")
                        readable = False
                        continue
                    if len(actual) != EXPECTED_SETS_PER_INVOCATION:
                        problems.append(
                            f"checkpoints.{cid}: {entry} holds {len(actual)} "
                            f"'{key}' values, not the "
                            f"{EXPECTED_SETS_PER_INVOCATION} that upstream "
                            "eval_nets.nEval fixes for one invocation")
                    pooled[key].extend(actual)

            if readable:
                for field, key in (("accuracies", "accs"), ("tprs", "tprs"),
                                   ("tnrs", "tnrs")):
                    claimed = rec.get(field)
                    actual = pooled[key]
                    if not isinstance(claimed, list):
                        continue
                    if len(claimed) != len(actual):
                        problems.append(
                            f"checkpoints.{cid}.{field}: the record lists "
                            f"{len(claimed)} values, the raw NPZ(s) hold "
                            f"{len(actual)} in total")
                        continue
                    off = [(i, c, a) for i, (c, a)
                           in enumerate(zip(claimed, actual))
                           if abs(float(c) - a) > 1e-12]
                    if off:
                        problems.append(
                            f"checkpoints.{cid}.{field}: the record "
                            f"disagrees with the raw NPZ at {off[:3]}. The "
                            "reported metrics are not the measured ones, or "
                            "they were pooled in an order other than the "
                            "registered invocation order.")

    out["derived"] = derived
    out["ok"] = not problems
    return out


def main(argv=None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Validate an AutoND evaluation result bundle. Runs no "
                    "evaluation.")
    ap.add_argument("bundle_dir", type=Path)
    ap.add_argument("--plan-hash", default=None,
                    help="the frozen plan's hash. Without it the bundle is "
                         "NOT plan-bound and validation fails unless "
                         "--structure-only is given.")
    ap.add_argument("--structure-only", action="store_true",
                    help="assert schema and digests only, accepting that the "
                         "result is not bound to any plan")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    res = validate_bundle(args.bundle_dir, expected_plan_hash=args.plan_hash,
                          require_plan_binding=not args.structure_only)
    print(f"bundle: {res['bundle_dir']}")
    print(f"schema: {SCHEMA_VERSION}")
    # Stated before the metrics, so a reader cannot mistake a structurally
    # valid bundle for a plan-bound one.
    if res["plan_bound"]:
        print(f"plan-bound: YES, to {args.plan_hash}")
    elif args.structure_only:
        print("plan-bound: NO - structure checked only, at the caller's "
              "request. This bundle is NOT confirmatory evidence for any "
              "preregistration.")
    else:
        print("plan-bound: NO")
    for cid, stats in sorted(res["derived"].items()):
        if stats.get("mean") is None:
            continue
        lo, hi = stats.get("low"), stats.get("high")
        span = f"[{lo:.6f}, {hi:.6f}]" if lo is not None else "(no interval)"
        print(f"  {cid}: n={stats['n']} mean={stats['mean']:.6f} "
              f"sd={stats['sample_sd']:.7f} 95% t-interval {span}")
    if res["problems"]:
        print(f"\n  {len(res['problems'])} PROBLEM(S):")
        for p in res["problems"]:
            print(f"    - {p}")
    else:
        print("\n  bundle valid: schema complete, digests match, metrics agree "
              "with the raw NPZ data"
              + (", bound to the frozen plan." if res["plan_bound"]
                 else ", NOT bound to a plan."))
    print("\n  Validation establishes INTEGRITY only. Whether the result "
          "supports a\n  hypothesis is decided by the plan's decision rule, "
          "not here.")
    if args.json:
        args.json.write_text(json.dumps(res, indent=2, sort_keys=True, default=str))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
