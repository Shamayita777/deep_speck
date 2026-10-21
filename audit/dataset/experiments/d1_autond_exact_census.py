"""D1 — duplicate and overlap census for the canonical AutoND instance.

Orchestration only. Every check is performed by the UNCHANGED generic engine
`audit.dataset.d1_duplicate_detection`; this script verifies the canonical D0
instance, then hands it over. No D1 algorithm is reimplemented here.

EXECUTION ORDER IS A SAFETY PROPERTY
------------------------------------
The earlier version loaded the arrays and ran duplicate detection before the
canonical manifest had been fully verified. Expensive audit work could
therefore run against an instance whose metadata did not match its data, and
produce a certificate for it. The order is now:

    1. load the candidate manifest
    2. verify the COMPLETE D0 manifest and every persisted partition
    3. validate the dataset specification, target identity and representation
    4. confirm the adapter configuration and the manifest name the same
       pinned AutoND source and configuration
    5. only then invoke the generic D1 engine
    6. bind the resulting evidence to the verified canonical instance

If any of 2-4 fails, this module raises before the engine is called. The
engine is reached through the module-level `D1_ENGINE` indirection so a test
can substitute a spy and prove the call never happens.

PRODUCTION SCALE IS MEASURED, NOT DECLARED
------------------------------------------
`production_status` is derived from the ACTUAL row counts recorded for each
partition and re-verified against the persisted arrays in step 2 -- never from
a caller-supplied `reduced_scale` flag. Editing a flag, or editing a manifest
by hand, cannot promote a reduced-size smoke test to production evidence.

SCOPE OF A D1 PASS
------------------
A PASS means the declared duplicate, overlap and label-consistency checks
passed within their documented scope, for THIS instance, at THIS round count
and difference. It is not evidence that the AutoND generator is correct, that
the dataset is cryptographically sound, or that any other round count or
difference behaves the same way.

USAGE
    python -m audit.dataset.experiments.d1_autond_exact_census \\
        --instance-dir evidence/d0_autond_speck3264_r8 \\
        --out-dir evidence/d1_autond_speck3264_r8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from audit.dataset import canonical as C
from audit.dataset.d1_duplicate_detection import (
    audit_duplicates,
    build_d1_certificate,
    evaluate_d1,
    print_report,
)

#: AutoND's representation, as verified against the pinned source. Passed to
#: the engine as the EXPECTED schema so a representation drift is a failure
#: rather than something the engine silently accommodates.
FEATURE_ENCODING = "binary-bits-uint8"
REPRESENTATION_CONVENTION = "hstack(C0,C1) little-endian-free bit columns"

#: The engine entry point, held behind one name so a test can replace it with
#: a spy and prove that a failed preflight prevents the audit from starting.
D1_ENGINE = audit_duplicates

#: The frozen production configuration. `production_status` is FULL_SCALE only
#: when the actual, verified row counts meet every one of these.
PRODUCTION_SCALE = {
    "train": 10_000_000,
    "validation": 1_000_000,
    "test": 1_000_000,
}

#: Identity this experiment audits. A manifest describing anything else is
#: refused: this script's scope statements, representation expectations and
#: production configuration are all specific to it.
EXPECTED_TARGET = "autond"
EXPECTED_DATASET_ID = "autond-speck3264"
EXPECTED_CIPHER = "speck3264"
EXPECTED_SCENARIO = "single-key"
EXPECTED_FEATURE_BITS = 64
EXPECTED_PLAIN_BITS = 32
EXPECTED_FEATURE_DTYPE = "uint8"
EXPECTED_LABEL_DTYPE = "uint8"

#: The rest of the FROZEN case study. These are not defaults and not a range:
#: a different round count or a different input difference is a different
#: experiment, with its own scope statements and its own evidence. Auditing one
#: under this script's scope text would misdescribe it, so anything outside
#: these values is refused before the engine runs.
EXPECTED_NUM_ROUNDS = 8
EXPECTED_DELTA_PLAIN = 0x400000
EXPECTED_DELTA_KEY = 0x0
EXPECTED_LAYOUT = "np.hstack([C0, C1])"
EXPECTED_LABEL_VALUES = (0, 1)

#: Widening the case study is a deliberate act, not a side effect of editing a
#: manifest. Stated here so the refusal messages can say so.
SCOPE_NOTE = (
    "This experiment audits exactly ONE frozen configuration. Another round "
    "count, input difference, scenario or cipher requires its own declared "
    "configuration and its own scope statements; it is not reachable by "
    "editing a manifest.")


def expected_schemas(manifest: dict) -> tuple[dict, dict]:
    rep = manifest["specification"]["representation"]
    feature_bits = int(rep["feature_bits"])
    feature_schema: dict[str, Any] = {
        "ndim": 2,
        "columns": feature_bits,
        "dtype": "uint8",
        "value_set": [0, 1],
    }
    label_schema: dict[str, Any] = {
        "ndim": 1,
        "dtype": "uint8",
        "value_set": [0, 1],
    }
    return feature_schema, label_schema


def classify_scale(manifest: dict) -> dict:
    """Derive the scale classification from the verified row counts.

    Deliberately ignores `specification.reduced_scale` for the DECISION; the
    flag is read only so a disagreement can be reported. A flag is
    caller-controlled and a manifest is editable, so neither may decide
    whether something counts as production evidence.
    """
    actual = {p: int(manifest["partitions"][p]["samples"])
              for p in C.PARTITIONS}
    meets = {p: actual[p] >= PRODUCTION_SCALE[p] for p in C.PARTITIONS}
    full = all(meets.values())
    declared_reduced = bool(manifest["specification"].get("reduced_scale"))
    out = {
        "production_status": "FULL_SCALE" if full else "REDUCED_SCALE_SMOKE",
        "actual_samples": actual,
        "frozen_production_configuration": dict(PRODUCTION_SCALE),
        "partitions_meeting_production_scale": meets,
        "basis": ("derived from the verified per-partition row counts, not "
                  "from any specification flag"),
        "declared_reduced_scale_flag": declared_reduced,
    }
    if full and declared_reduced:
        out["flag_disagreement"] = (
            "the manifest's reduced_scale flag is set but the actual row "
            "counts meet the frozen production configuration; the measured "
            "counts decide, and the flag is reported as inconsistent.")
    if (not full) and (not declared_reduced):
        out["flag_disagreement"] = (
            "the manifest does not declare reduced_scale but the actual row "
            "counts are below the frozen production configuration; the "
            "measured counts decide.")
    if not full:
        out["production_status_note"] = (
            "Partition sizes are below the frozen production configuration. "
            "This establishes that the integration path works; it is NOT "
            "production evidence and must not be reported as a completed D1 "
            "audit.")
    return out


def preflight(instance_dir: Path, *, log=print) -> dict:
    """Verify everything that must hold before the D1 engine may run.

    Raises SystemExit on any failure, using the repository's REFUSED
    convention. Returns the material the audit needs, so a caller cannot
    accidentally proceed without having run this.
    """
    instance_dir = Path(instance_dir)

    # (1) Load the candidate manifest.
    manifest = C.load_manifest(instance_dir)

    # (2) FULL verification of the manifest and every persisted partition.
    # This re-hashes the files and the decoded arrays, recomputes the canonical
    # instance id, checks the specification digest, and compares every
    # partition's recorded metadata against the arrays on disk.
    log("D1 preflight — verifying the canonical D0 instance")
    verification = C.verify(instance_dir, log=log)
    if not verification.get("verified"):
        failed = [p for p, v in verification["partitions"].items()
                  if not v.get("verified")]
        detail = []
        for p in failed:
            detail.extend(verification["partitions"][p].get(
                "metadata_problems", [verification["partitions"][p].get(
                    "reason", "unspecified")]))
        idt = verification["identity"]
        if not idt["canonical_instance_id_consistent"]:
            detail.append(
                f"canonical_instance_id recorded "
                f"{idt['canonical_instance_id_recorded']} but the partition "
                f"digests give {idt['canonical_instance_id_recomputed']}")
        if idt["specification_consistent"] is False:
            detail.append("specification_sha256 does not match the recorded "
                          "specification")
        raise SystemExit(
            "REFUSED: the canonical D0 instance did not verify, so D1 was not "
            "run. No certificate is produced for an unverified instance.\n  - "
            + "\n  - ".join(str(d) for d in detail))

    spec = manifest["specification"]

    # (3) The instance is the one this experiment is scoped to.
    identity_problems = []
    if str(spec.get("target")) != EXPECTED_TARGET:
        identity_problems.append(
            f"target is {spec.get('target')!r}, expected "
            f"{EXPECTED_TARGET!r}")
    if str(spec.get("dataset_id")) != EXPECTED_DATASET_ID:
        identity_problems.append(
            f"dataset_id is {spec.get('dataset_id')!r}, expected "
            f"{EXPECTED_DATASET_ID!r}")
    if str(spec.get("cipher")) != EXPECTED_CIPHER:
        identity_problems.append(
            f"cipher is {spec.get('cipher')!r}, expected {EXPECTED_CIPHER!r}")
    if str(spec.get("scenario")) != EXPECTED_SCENARIO:
        identity_problems.append(
            f"scenario is {spec.get('scenario')!r}, expected "
            f"{EXPECTED_SCENARIO!r}")
    if spec.get("dataset_id") != f"autond-{spec.get('cipher')}":
        identity_problems.append(
            f"dataset_id {spec.get('dataset_id')!r} does not describe cipher "
            f"{spec.get('cipher')!r}")
    if identity_problems:
        raise SystemExit(
            "REFUSED: the canonical instance is not the configuration this "
            "experiment audits, so D1 was not run:\n  - "
            + "\n  - ".join(identity_problems))

    # (3b) The FROZEN experimental configuration: rounds, differences and the
    # verified-configuration flag. Compared numerically for the differences, so
    # an equivalent hex spelling is accepted and a different value is not.
    frozen_problems = []
    try:
        actual_rounds = int(spec.get("num_rounds"))
    except (TypeError, ValueError):
        actual_rounds = None
        frozen_problems.append(
            f"num_rounds is {spec.get('num_rounds')!r}, not an integer")
    if actual_rounds is not None and actual_rounds != EXPECTED_NUM_ROUNDS:
        frozen_problems.append(
            f"num_rounds is {actual_rounds}, expected {EXPECTED_NUM_ROUNDS}")
    for field, expected, label in (
            ("delta_plain_hex", EXPECTED_DELTA_PLAIN, "plaintext difference"),
            ("delta_key_hex", EXPECTED_DELTA_KEY, "key difference")):
        raw = spec.get(field)
        try:
            value = int(str(raw), 16)
        except (TypeError, ValueError):
            frozen_problems.append(f"{field} is {raw!r}, not a hex integer")
            continue
        if value != expected:
            frozen_problems.append(
                f"the {label} is {hex(value)}, expected {hex(expected)}")
    # These three were FAIL-OPEN: `delta_plain_verified` was rejected only
    # when explicitly False, `label_values` was checked only when present, and
    # a missing sample-count declaration passed. A missing field is now a
    # refusal: an absent assurance is not an assurance, and a manifest can omit
    # a field as easily as it can falsify one.
    if spec.get("verified_configuration") is not True:
        frozen_problems.append(
            f"verified_configuration is "
            f"{spec.get('verified_configuration')!r}, expected True; this "
            f"instance was not generated from the verified cipher and "
            f"difference")
    if "delta_plain_verified" not in spec:
        frozen_problems.append(
            "delta_plain_verified is ABSENT. The frozen case study requires an "
            "explicit assurance that the input difference is the one verified "
            "against the pinned source; a missing field is not that assurance")
    elif spec.get("delta_plain_verified") is not True:
        frozen_problems.append(
            f"delta_plain_verified is "
            f"{spec.get('delta_plain_verified')!r}, expected True: the input "
            f"difference is not the one verified against the pinned source")

    if "label_values" not in spec:
        frozen_problems.append(
            f"label_values is ABSENT. The approved binary label set "
            f"{list(EXPECTED_LABEL_VALUES)} must be declared explicitly, or "
            f"nothing constrains what the labels may contain")
    else:
        declared_labels = spec.get("label_values")
        try:
            as_tuple = tuple(int(v) for v in declared_labels)
        except (TypeError, ValueError):
            frozen_problems.append(
                f"label_values is {declared_labels!r}, which is not a list of "
                f"integers")
        else:
            if as_tuple != EXPECTED_LABEL_VALUES:
                frozen_problems.append(
                    f"label_values is {declared_labels!r}, expected "
                    f"{list(EXPECTED_LABEL_VALUES)}")

    # Sample counts: present, integral and strictly positive in BOTH the
    # specification and the partition records. Agreement between them, and
    # with the arrays, is checked below and by D0's verifier.
    for part in C.PARTITIONS:
        key = f"{part}_samples"
        if key not in spec:
            frozen_problems.append(
                f"{key} is ABSENT from the specification. Every partition "
                f"size must be declared, or the instance's scale is whatever "
                f"happens to be on disk")
            continue
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            frozen_problems.append(
                f"{key} is {value!r}, not an integer")
        elif value < 1:
            frozen_problems.append(
                f"{key} is {value}, which is not a positive count")
    for part in C.PARTITIONS:
        rec = manifest["partitions"].get(part, {})
        recorded = rec.get("samples")
        if isinstance(recorded, bool) or not isinstance(recorded, int):
            frozen_problems.append(
                f"the {part} partition record's `samples` is {recorded!r}, "
                f"not an integer")
        elif recorded < 1:
            frozen_problems.append(
                f"the {part} partition record declares {recorded} samples, "
                f"which is not a positive count")

    # The declared feature dtype must be present and be the frozen one.
    if spec.get("feature_dtype") != EXPECTED_FEATURE_DTYPE:
        frozen_problems.append(
            f"feature_dtype is {spec.get('feature_dtype')!r}, expected "
            f"{EXPECTED_FEATURE_DTYPE!r}")
    if frozen_problems:
        raise SystemExit(
            "REFUSED: the canonical instance is outside the frozen case-study "
            "configuration, so D1 was not run:\n  - "
            + "\n  - ".join(frozen_problems)
            + f"\n{SCOPE_NOTE}")

    # (4) The representation is the one the engine is handed as EXPECTED.
    rep = spec.get("representation")
    if not isinstance(rep, dict):
        raise SystemExit(
            "REFUSED: the canonical specification records no `representation`; "
            "the expected feature schema cannot be established.")
    rep_problems = []
    if int(rep.get("feature_bits", -1)) != EXPECTED_FEATURE_BITS:
        rep_problems.append(
            f"feature_bits is {rep.get('feature_bits')!r}, expected "
            f"{EXPECTED_FEATURE_BITS}")
    if int(rep.get("plain_bits", -1)) != EXPECTED_PLAIN_BITS:
        rep_problems.append(
            f"plain_bits is {rep.get('plain_bits')!r}, expected "
            f"{EXPECTED_PLAIN_BITS}")
    # The source-defined concatenation. The engine is told C0 occupies the
    # first half and C1 the second; a different layout would make that
    # statement false while every digest stayed valid.
    if str(rep.get("layout")) != EXPECTED_LAYOUT:
        rep_problems.append(
            f"layout is {rep.get('layout')!r}, expected {EXPECTED_LAYOUT!r}")
    half = EXPECTED_FEATURE_BITS // 2
    if list(rep.get("c0_columns", [])) != [0, half]:
        rep_problems.append(
            f"c0_columns is {rep.get('c0_columns')!r}, expected {[0, half]}")
    if list(rep.get("c1_columns", [])) != [half, EXPECTED_FEATURE_BITS]:
        rep_problems.append(
            f"c1_columns is {rep.get('c1_columns')!r}, expected "
            f"{[half, EXPECTED_FEATURE_BITS]}")
    for p in C.PARTITIONS:
        prec = manifest["partitions"][p]
        if str(prec["features"]["dtype"]) != EXPECTED_FEATURE_DTYPE:
            rep_problems.append(
                f"{p} features are {prec['features']['dtype']}, expected "
                f"{EXPECTED_FEATURE_DTYPE}")
        if str(prec["labels"]["dtype"]) != EXPECTED_LABEL_DTYPE:
            rep_problems.append(
                f"{p} labels are {prec['labels']['dtype']}, expected "
                f"{EXPECTED_LABEL_DTYPE}")
        if int(prec["features"]["shape"][1]) != EXPECTED_FEATURE_BITS:
            rep_problems.append(
                f"{p} features are {prec['features']['shape'][1]} columns "
                f"wide, expected {EXPECTED_FEATURE_BITS}")
    if rep_problems:
        raise SystemExit(
            "REFUSED: the canonical instance's representation is not the one "
            "this experiment audits, so D1 was not run:\n  - "
            + "\n  - ".join(rep_problems))

    # (5) The source the instance was generated from must be recorded, and the
    # recorded digests must be the ones the pin approved.
    prov = spec.get("adapter_provenance") or {}
    digests = prov.get("source_digests") or spec.get("source_digests")
    if not digests:
        raise SystemExit(
            "REFUSED: the canonical specification records no `source_digests`, "
            "so the AutoND source this instance was generated from cannot be "
            "established. D1 was not run. Rebuild the instance with a D0 "
            "runner that enforces the source pin.")
    pin = spec.get("source_pin") or {}
    if pin:
        bad = {k: {"pinned": v, "recorded": digests.get(k)}
               for k, v in pin.get("source_digests", {}).items()
               if digests.get(k) != v}
        if bad:
            raise SystemExit(
                f"REFUSED: the instance's recorded source digests do not "
                f"match the pin recorded alongside them: {bad}. D1 was not "
                f"run.")

    # The manifest's own `source_pin` lives in the manifest, so agreeing with
    # it proves only internal consistency. An instance generated from an
    # unapproved checkout could record that checkout's digests in BOTH fields
    # and pass. So the recorded digests are compared against the FROZEN pin
    # imported from the case-study configuration -- a value outside this
    # manifest entirely.
    try:
        from audit.dataset.experiments.d0_autond_canonical import (
            AUTOND_SOURCE_PIN as FROZEN_PIN)
    except Exception as exc:                                   # noqa: BLE001
        raise SystemExit(
            f"REFUSED: the frozen AutoND source pin could not be loaded "
            f"({type(exc).__name__}: {exc}), so the instance's source "
            f"identity cannot be checked against a trusted value. D1 was not "
            f"run.")
    frozen_digests = FROZEN_PIN.get("source_digests") or {}
    if not frozen_digests:
        raise SystemExit(
            "REFUSED: the frozen AutoND source pin declares no digests, so "
            "there is nothing trusted to compare against. D1 was not run.")
    against_frozen = {k: {"frozen_pin": v, "instance_recorded": digests.get(k)}
                      for k, v in frozen_digests.items()
                      if digests.get(k) != v}
    if against_frozen:
        raise SystemExit(
            f"REFUSED: the instance's recorded AutoND source does not match "
            f"the FROZEN case-study pin: {against_frozen}. This is checked "
            f"against the pin in the case-study configuration, not against "
            f"another field of the same manifest, so a self-consistent "
            f"manifest naming an unapproved checkout is still refused. D1 was "
            f"not run.")
    # The revision must be PRESENT, VERIFIED and equal to the frozen one.
    #
    # An earlier version accepted `source_commit_observed == "UNVERIFIED"`,
    # which is exactly the state produced when the checkout could not report a
    # revision. For the frozen production case study that is not acceptable:
    # it means the source this instance came from was never pinned to a
    # revision at all, and the digests alone cannot say which revision a file
    # set belongs to.
    expected_commit = FROZEN_PIN.get("upstream_commit")
    if not expected_commit:
        raise SystemExit(
            "REFUSED: the frozen AutoND pin declares no upstream_commit, so "
            "there is no approved revision to compare against. D1 was not "
            "run.")
    recorded_commit = spec.get("source_commit_observed")
    if recorded_commit is None:
        raise SystemExit(
            f"REFUSED: the instance records no `source_commit_observed`, so "
            f"the AutoND revision it was generated from is unknown. The "
            f"frozen case study requires revision {expected_commit}. D1 was "
            f"not run.")
    if recorded_commit == "UNVERIFIED":
        raise SystemExit(
            f"REFUSED: the instance records source_commit_observed="
            f"'UNVERIFIED'. The checkout it was generated from could not "
            f"report a revision, so it cannot be shown to be the approved "
            f"revision {expected_commit}. Matching digests do not establish a "
            f"revision. Rebuild the instance from a checkout that reports "
            f"its revision. D1 was not run.")
    if recorded_commit != expected_commit:
        raise SystemExit(
            f"REFUSED: the instance records AutoND revision "
            f"{recorded_commit}, but the frozen pin is {expected_commit}. D1 "
            f"was not run.")

    # The source-pin record must exist and be complete: it is what the
    # instance claims was enforced at generation time.
    if not pin:
        raise SystemExit(
            f"REFUSED: the instance records no `source_pin`, so there is no "
            f"record that a pin was enforced when it was generated. D1 was "
            f"not run.")
    pin_digests = pin.get("source_digests") or {}
    missing_in_pin = sorted(set(frozen_digests) - set(pin_digests))
    if missing_in_pin:
        raise SystemExit(
            f"REFUSED: the instance's recorded source_pin does not cover "
            f"{missing_in_pin}, which the frozen pin requires. The pin that "
            f"was enforced was weaker than the approved one. D1 was not run.")
    if pin.get("upstream_commit") != expected_commit:
        raise SystemExit(
            f"REFUSED: the instance's recorded source_pin names revision "
            f"{pin.get('upstream_commit')!r}, not the approved "
            f"{expected_commit}. D1 was not run.")

    # The adapter's verified provenance and the digests bound alongside the
    # canonical instance must be the same source identity.
    adapter_digests = (spec.get("adapter_provenance") or {}).get(
        "source_digests") or {}
    spec_digests = spec.get("source_digests") or {}
    if adapter_digests and spec_digests and adapter_digests != spec_digests:
        raise SystemExit(
            f"REFUSED: the adapter's recorded source digests and the "
            f"specification's differ, so the instance names two different "
            f"sources. D1 was not run.")

    # (6) Partition sizes must agree with the frozen experiment's declared
    # counts. D0's verify() already compared the arrays with the declared
    # counts; this catches a manifest whose partition records and specification
    # agree with each other but not with each other's sizes.
    size_problems = []
    for name in C.PARTITIONS:
        declared = spec.get(f"{name}_samples")
        recorded = manifest["partitions"][name].get("samples")
        if declared is not None and int(declared) != int(recorded):
            size_problems.append(
                f"{name}: specification declares {int(declared)} samples but "
                f"the partition record says {int(recorded)}")
    if size_problems:
        raise SystemExit(
            "REFUSED: declared and recorded partition sizes disagree, so D1 "
            "was not run:\n  - " + "\n  - ".join(size_problems))

    scale = classify_scale(manifest)
    log(f"  preflight PASSED — {scale['production_status']}")
    return {
        "manifest": manifest,
        "verification": verification,
        "scale": scale,
        "source_digests": digests,
        "source_pin": pin,
    }


def load_verified_partitions(instance_dir: Path, spec: dict):
    """Load the persisted arrays. Only called after preflight has passed."""
    X: dict[str, np.ndarray] = {}
    Y: dict[str, np.ndarray] = {}
    for name in C.PARTITIONS:
        x, y = C.load_partition(instance_dir, name, spec=spec)
        X[name], Y[name] = x, y
    return X, Y


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--instance-dir", type=Path, required=True,
                   help="the D0 canonical instance directory")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    # Steps 1-5. Raises before anything expensive if the instance is not
    # verified, is the wrong configuration, or has no recorded source.
    pre = preflight(args.instance_dir)
    manifest = pre["manifest"]
    spec = manifest["specification"]
    scale = pre["scale"]

    X, Y = load_verified_partitions(args.instance_dir, spec)

    print()
    print("D1 — duplicate and overlap census, AutoND")
    print(f"  canonical instance  {manifest['canonical_instance_id'][:16]}…")
    print(f"  dataset_id          {spec['dataset_id']}")
    print(f"  cipher / scenario   {spec['cipher']} / {spec['scenario']}")
    print(f"  num_rounds          {spec['num_rounds']}")
    print(f"  delta_plain         {spec['delta_plain_hex']}")
    for name in C.PARTITIONS:
        print(f"  {name:<18}  X{X[name].shape} {X[name].dtype}  "
              f"Y{Y[name].shape} {Y[name].dtype}")
    print(f"  production_status   {scale['production_status']}")
    if scale["production_status"] != "FULL_SCALE":
        print("  *** REDUCED SCALE — integration smoke, NOT production "
              "evidence ***")
    print()

    feature_schema, label_schema = expected_schemas(manifest)

    # Step 5: the generic engine, unchanged, reached through the indirection.
    results = D1_ENGINE(
        X["train"], X["validation"], X["test"],
        Y["train"], Y["validation"], Y["test"],
        expected_feature_schema=feature_schema,
        expected_label_schema=label_schema,
        representation_convention=REPRESENTATION_CONVENTION,
        feature_encoding=FEATURE_ENCODING,
    )

    verdict = evaluate_d1(results)

    prov = spec.get("adapter_provenance", {})
    certificate = build_d1_certificate(
        results,
        dataset_id=spec["dataset_id"],
        dataset_version=spec.get("dataset_version"),
        generation_procedure=spec.get("generation_procedure"),
        generation_parameters=prov.get("generation_parameters"),
        random_seed=None,          # os.urandom: no seed exists on this path
    )

    # Step 6: bind to the exact verified instance.
    certificate["canonical_instance"] = C.binding_record(args.instance_dir)
    certificate["d0_verification"] = {
        "verified": pre["verification"]["verified"],
        "checked_utc": pre["verification"]["checked_utc"],
        "canonical_instance_id": pre["verification"]["canonical_instance_id"],
        "schema_version": pre["verification"]["schema_version"],
        "verification_scope": pre["verification"]["verification_scope"],
        "ordering": ("the complete D0 manifest and all persisted partitions "
                     "were verified BEFORE the D1 engine was invoked"),
    }
    certificate["autond_source"] = {
        "source_digests": pre["source_digests"],
        "source_pin": pre["source_pin"] or None,
        "note": ("digests of the AutoND source files actually imported when "
                 "this instance was generated, as recorded by the adapter"),
    }
    certificate["scale_classification"] = scale
    certificate["scope_limitations"] = [
        "Applies to ONE dataset instance at the recorded cipher, scenario, "
        "input difference and round count. Another round count or difference "
        "is a different instance.",
        "A PASS covers the declared duplicate, overlap and label-consistency "
        "checks only. It is not evidence of generator correctness or "
        "cryptographic soundness.",
        "exact_replay_available is False: the generator draws from os.urandom, "
        "so this instance cannot be regenerated, only re-read.",
        "D0 verification compares the manifest with the bytes it names. It is "
        "not independent authentication: an attacker able to rewrite the data "
        "AND the whole manifest can produce a self-consistent instance.",
    ]
    certificate["production_status"] = scale["production_status"]
    if "production_status_note" in scale:
        certificate["production_status_note"] = scale["production_status_note"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cert_path = args.out_dir / "d1_autond_certificate.json"
    cert_path.write_text(json.dumps(certificate, indent=2, sort_keys=True,
                                    default=str))

    try:
        print_report(results)
    except Exception as exc:                                   # noqa: BLE001
        print(f"(print_report unavailable: {type(exc).__name__}: {exc})")

    print()
    print(f"  D1 verdict: {verdict}")
    print(f"  certificate: {cert_path}")
    print(f"  production_status: {scale['production_status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
