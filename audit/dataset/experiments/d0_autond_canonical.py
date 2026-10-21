"""D0 — build the canonical AutoND dataset instance.

Establishes ONE immutable dataset instance from the pinned AutoND generator and
persists it with a provenance manifest, so that D1, D2 and D3 analyse the same
bytes rather than independently regenerated datasets that merely share a
nominal configuration. That distinction matters here more than for a seeded
generator: AutoND draws from `os.urandom`, so a regenerated dataset is a
DIFFERENT instance and no seed can recover the first one.

This script is orchestration only. It contains no audit logic: the canonical
instance is built by the generic `audit.dataset.canonical.build`, which is
unchanged apart from the spec-driven generator resolution it already needed to
support a second target.

SCOPE, stated narrowly
----------------------
One cipher, one scenario, one difference, one FIXED round count. An AutoND
dataset at a different round count or difference is a different instance and
must get its own canonical build and its own evidence. In particular this is
NOT an audit of AutoND's staged training pipeline, which trains across a
sequence of rounds; that is a separately scoped experiment.

USAGE
    python -m audit.dataset.experiments.d0_autond_canonical build \\
        --out-dir evidence/d0_autond_speck3264_r8 \\
        --autond-root ~/Documents/AutoND-upstream \\
        --num-rounds 8
    python -m audit.dataset.experiments.d0_autond_canonical verify \\
        --out-dir evidence/d0_autond_speck3264_r8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audit.dataset import canonical as C

#: Proposed initial case study. Every value is verified against the pinned
#: source by the adapter at construction time; none is a silent default.
#: The ONE configuration whose representation has been verified against the
#: pinned source. The experiment restricts itself to this cipher: the reviewed
#: version accepted `--cipher` while continuing to emit the fixed identifier
#: `autond-speck3264`, so SIMON32/64 data would have been recorded as SPECK
#: data, with SPECK's verified input difference silently reused for a cipher it
#: was never chosen for.
#:
#: Generalising to more ciphers needs, per cipher: a verified feature
#: representation, an input difference chosen FOR that cipher, and its own
#: identifier. None of that exists yet, so an unsupported value is refused
#: rather than quietly mislabelled.
#: REVIEWED SOURCE PIN for the AutoND case study.
#:
#: PROVENANCE OF THESE VALUES -- they were established, not copied from a
#: report:
#:   1. Read directly from the checked-out target at commit
#:      8d66e0a43a9e6fbbb3624d68296c1b77aa2da779.
#:   2. Corroborated against a SECOND, independent checkout at the same
#:      commit; both agree on both digests.
#:   3. The cipher digest independently matches `cipher_module_sha256` as
#:      recorded in an Arm B runtime-calibration record written before this
#:      work began, so it is attested by evidence produced for another
#:      purpose.
#:
#: The production experiment ALWAYS enforces this pin. The adapter's
#: `expected_source_digests` stays optional so the adapter remains reusable
#: for controlled development, but this experiment does not expose a way to
#: generate the case-study dataset from an unapproved checkout.
AUTOND_SOURCE_PIN: dict = {
    "upstream_commit": "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779",
    # EVERY module whose top-level code executes when the generator is
    # imported. Earlier revisions pinned only main.py and the cipher, so
    # optimizer.py, train_nets.py, dbitnet.py and gohrnet.py ran unchecked.
    # The set was determined by importing the pinned `main` and listing the
    # sys.modules entries resolving inside the checkout, not assumed from
    # upstream's current layout.
    "source_digests": {
        "main.py":
            "52554c2966edefdc718f298d12476e7cd5f1faf8e99eafad2fc278a523690665",
        "optimizer.py":
            "2d58e4fd94973040e1ead21166b188527b6935d499a7a818a3c613ac4888d505",
        "train_nets.py":
            "f38be25f879df0ece662cea78d8dd6a78da3b9b076a4b83bc22f2cb25e53ec8e",
        "dbitnet.py":
            "b19adceb6e6a9ad39c648aa41e58d23b6a0b338c9bd2be9c4b0c7799ea631736",
        "gohrnet.py":
            "324c3768e19a6230dbedbe0446f54c6139d72f5dc642f5cfb60a8fa9a9ea1d56",
        "ciphers/speck3264.py":
            "fd4a04be7fb632cea422338e6ae8446e421049649944f9dab87cc05e0e5ca9fe",
    },
    "digest_provenance": [
        "read from the checked-out pinned target",
        "all six corroborated by a second independent checkout at the same "
        "commit",
        "cipher digest matches cipher_module_sha256 in a pre-existing Arm B "
        "runtime-calibration record",
        "optimizer.py digest matches the value the II-4A search harness "
        "recorded independently before this work",
        "checkout reports 0 modified tracked files and 0 untracked .py files "
        "at the import path",
    ],
}

SUPPORTED_CIPHERS = ("speck3264",)

#: Input differences verified against the pinned source, per cipher. SPECK's
#: 0x400000 must not be reused for another cipher by default.
VERIFIED_DELTA_PLAIN_HEX = {"speck3264": "0x400000"}

AUTOND_CANONICAL_SPEC: dict = {
    "target": "autond",
    "dataset_id": "autond-speck3264",
    "cipher": "speck3264",
    "scenario": "single-key",
    "delta_plain_hex": "0x400000",
    "delta_key_hex": "0x0",
    "num_rounds": 8,
    "train_samples": 10_000_000,
    "validation_samples": 1_000_000,
    "test_samples": 1_000_000,
    "randomness_source": "os.urandom",
    "exact_replay_available": False,
    # Consumed by canonical.validate_generated_partition BEFORE persistence,
    # so a generator that returns the wrong dtype or a stray label value is
    # refused rather than recorded as whatever it produced.
    "feature_dtype": "uint8",
    "label_values": [0, 1],
    # dataset_version / generator / generation_procedure are filled from the
    # adapter's own provenance so the manifest cannot disagree with the code
    # that produced the arrays.
}


def _observed_commit(root: Path) -> str | None:
    """The checkout's git HEAD, or None if it is not a git working tree."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    rev = out.stdout.strip()
    return rev or None


def build_spec(args, *, source_pin: dict | None = None) -> dict:
    spec = dict(AUTOND_CANONICAL_SPEC)
    spec["autond_root"] = str(Path(args.autond_root).expanduser().resolve())
    if args.num_rounds is not None:
        spec["num_rounds"] = int(args.num_rounds)
    if args.cipher is not None:
        spec["cipher"] = str(args.cipher)
    if spec["cipher"] not in SUPPORTED_CIPHERS:
        raise SystemExit(
            f"REFUSED: cipher {spec['cipher']!r} is not a supported "
            f"configuration for this experiment. Supported: "
            f"{list(SUPPORTED_CIPHERS)}.\n"
            f"Only speck3264 has a verified feature representation and a "
            f"verified input difference. Generating another cipher here would "
            f"record it under an identifier and a difference chosen for "
            f"SPECK32/64, so the manifest would describe data it did not "
            f"generate. Add the cipher to SUPPORTED_CIPHERS and "
            f"VERIFIED_DELTA_PLAIN_HEX, with its own verification, first.")
    if args.delta_plain_hex is not None:
        spec["delta_plain_hex"] = str(args.delta_plain_hex)
    verified_delta = VERIFIED_DELTA_PLAIN_HEX[spec["cipher"]]
    if int(spec["delta_plain_hex"], 16) != int(verified_delta, 16):
        # Allowed, but it is a DIFFERENT instance and must not be presented as
        # the verified case study.
        spec["delta_plain_verified"] = False
        spec["delta_plain_verified_note"] = (
            f"delta_plain_hex {spec['delta_plain_hex']} is not the verified "
            f"difference {verified_delta} for {spec['cipher']}. This is a "
            f"different dataset instance and its representation has not been "
            f"checked against the pinned source.")
    else:
        spec["delta_plain_verified"] = True
    if args.train_samples is not None:
        spec["train_samples"] = int(args.train_samples)
    if args.validation_samples is not None:
        spec["validation_samples"] = int(args.validation_samples)
    if args.test_samples is not None:
        spec["test_samples"] = int(args.test_samples)

    reduced = (spec["train_samples"] < AUTOND_CANONICAL_SPEC["train_samples"]
               or spec["validation_samples"]
               < AUTOND_CANONICAL_SPEC["validation_samples"]
               or spec["test_samples"]
               < AUTOND_CANONICAL_SPEC["test_samples"])
    spec["reduced_scale"] = bool(reduced)
    if reduced:
        # A reduced instance must announce itself in its own manifest, or a
        # later reader cannot tell a smoke test from a production audit.
        spec["reduced_scale_note"] = (
            "PARTITION SIZES BELOW THE DECLARED CASE STUDY. This is an "
            "integration smoke instance, NOT production evidence. Audits over "
            "it establish that the path works, not that the production "
            "dataset passed.")

    # -- SOURCE PIN ENFORCEMENT ---------------------------------------
    # The pin is validated and passed into the adapter, which refuses in its
    # CONSTRUCTOR on a mismatch -- i.e. before any data is generated.
    pin = AUTOND_SOURCE_PIN if source_pin is None else source_pin
    if not isinstance(pin, dict) or not pin.get("source_digests"):
        raise SystemExit(
            "REFUSED: no source pin is available for the AutoND case study. "
            "The production experiment will not generate a canonical dataset "
            "from an unverified checkout, because the resulting instance would "
            "be described as the approved target without that having been "
            "established. Define AUTOND_SOURCE_PIN.")
    required_files = ("main.py", f"ciphers/{spec['cipher']}.py")
    absent = [f for f in required_files if f not in pin["source_digests"]]
    if absent:
        raise SystemExit(
            f"REFUSED: the source pin does not cover {absent}, which this "
            f"configuration generates from. An uncovered file could differ "
            f"from the approved revision without the pin noticing.")

    # Revision check, where the checkout can report one. A checkout that is
    # not a git working tree cannot, and that is recorded rather than assumed.
    observed_commit = _observed_commit(Path(spec["autond_root"]))
    expected_commit = pin.get("upstream_commit")
    if observed_commit and expected_commit and observed_commit != expected_commit:
        raise SystemExit(
            f"REFUSED: the checkout at {spec['autond_root']} is at commit "
            f"{observed_commit}, but the reviewed pin is {expected_commit}. "
            f"Generation would produce a dataset from a revision that was "
            f"never approved.")

    # Fill the descriptive fields from the adapter itself.
    from audit.dataset.adapters.autond import AutoNDDatasetAdapter
    a = AutoNDDatasetAdapter(
        autond_root=spec["autond_root"], cipher_name=spec["cipher"],
        delta_plain_int=int(spec["delta_plain_hex"], 16),
        delta_key_int=int(spec["delta_key_hex"], 16),
        num_rounds=int(spec["num_rounds"]), scenario=spec["scenario"],
        expected_source_digests=pin["source_digests"],
        expected_upstream_commit=pin.get("upstream_commit"))
    prov = a.dataset_provenance(
        train_samples=spec["train_samples"],
        validation_samples=spec["validation_samples"],
        test_samples=spec["test_samples"])
    spec["dataset_version"] = prov["dataset_version"]
    spec["generator"] = prov["generation_parameters"]["generator"]
    spec["generation_procedure"] = prov["generation_procedure"]
    spec["input_difference"] = prov["generation_parameters"]["input_difference"]
    spec["representation"] = prov["generation_parameters"]["representation"]
    spec["adapter_provenance"] = prov

    # The identifier must describe the cipher actually configured. The adapter
    # derives it; this asserts the spec agrees, so a future edit cannot
    # reintroduce a fixed identifier that contradicts the data.
    spec["dataset_id"] = a.dataset_id
    if spec["dataset_id"] != f"autond-{spec['cipher']}":
        raise SystemExit(
            f"REFUSED: dataset_id {spec['dataset_id']!r} does not describe "
            f"cipher {spec['cipher']!r}.")
    spec["verified_configuration"] = bool(
        a.is_verified_configuration and spec.get("delta_plain_verified", False))

    # Record the VERIFIED source identity: the pin that was enforced, the
    # digests of the files actually imported, and where they were imported
    # from. The adapter computes the digests from each loaded module's own
    # __file__, so these name executed code.
    spec["source_pin"] = {
        "upstream_commit": pin.get("upstream_commit"),
        "source_digests": dict(pin["source_digests"]),
        "digest_provenance": list(pin.get("digest_provenance", [])),
        "enforced": True,
    }
    spec["source_digests"] = dict(a.source_digests)
    spec["source_paths"] = dict(a.source_paths)
    spec["source_commit_observed"] = observed_commit or "UNVERIFIED"
    if not observed_commit:
        spec["source_commit_note"] = (
            "the checkout did not report a git revision, so the commit could "
            "not be confirmed independently of the file digests")
    # Defence in depth: the adapter already refused a mismatch, so this can
    # only fire if that path is ever weakened.
    for rel, want in pin["source_digests"].items():
        if a.source_digests.get(rel) != want:
            raise SystemExit(
                f"REFUSED: source digest for {rel} is "
                f"{a.source_digests.get(rel)!r} but the pin requires "
                f"{want!r}.")
    return spec


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="generate and persist the instance once")
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--autond-root", type=Path, required=True)
    b.add_argument("--repo-root", type=Path, default=Path("."))
    b.add_argument("--num-rounds", type=int, default=None)
    b.add_argument("--cipher", default=None)
    b.add_argument("--delta-plain-hex", default=None)
    b.add_argument("--train-samples", type=int, default=None)
    b.add_argument("--validation-samples", type=int, default=None)
    b.add_argument("--test-samples", type=int, default=None)

    v = sub.add_parser("verify", help="re-hash the persisted instance")
    v.add_argument("--out-dir", type=Path, required=True)

    s = sub.add_parser("show", help="print the manifest")
    s.add_argument("--out-dir", type=Path, required=True)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    if args.command == "build":
        spec = build_spec(args)
        print("D0 — canonical AutoND dataset instance")
        for k in ("target", "dataset_id", "cipher", "scenario", "num_rounds",
                  "delta_plain_hex", "train_samples", "validation_samples",
                  "test_samples", "reduced_scale"):
            print(f"  {k:<20} {spec[k]}")
        if spec["reduced_scale"]:
            print("  *** REDUCED SCALE — integration smoke, not production ***")
        manifest = C.build(args.out_dir, repo_root=args.repo_root, spec=spec)
        print(f"\nmanifest: {args.out_dir / C.MANIFEST_NAME}")
        return 0

    if args.command == "verify":
        report = C.verify(args.out_dir)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if report.get("ok") else 1

    if args.command == "show":
        print(json.dumps(C.load_manifest(args.out_dir), indent=2,
                         sort_keys=True, default=str))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
