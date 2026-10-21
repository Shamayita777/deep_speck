"""Runtime preflight for the AutoND dataset-integrity case study.

Establishes that the environment can run the PINNED AutoND source correctly,
before any canonical dataset is generated. Every check is executed and its
outcome printed; nothing is assumed from a previous report.

WHY A PREFLIGHT IS NEEDED AT ALL
--------------------------------
The pinned source is incompatible with NumPy 2.x. Under NEP 50,

    ciphers/speck3264.py:103
        X[:, i] += 2**(WORD_SIZE()-1-j)*arr[:, pos]

raises `OverflowError: Python integer 32768 out of bounds for uint8`, because
`arr` is uint8 and the Python int 2**15 must now fit the array's dtype.
NumPy 1.26 promoted instead, under value-based casting. The failure occurs
inside `check_testvectors()`, which the cipher module runs at import, so under
NumPy 2.x the pinned cipher cannot even self-test.

The pinned source is NOT modified to work around this. The runtime is the fix,
and this script is what proves the runtime is the right one.

MANDATORY DEPENDENCIES
----------------------
`main.py` imports `optimizer` and `train_nets` as top-level modules, and
`train_nets` imports `tensorflow` at module scope. Importing the GENERATOR
therefore pulls in TensorFlow, Keras and pandas even though D0/D1 never train
anything. They are mandatory, not optional. This script reports which of them
actually loaded, so the dependency claim is observed rather than asserted.

USAGE
    python -m audit.dataset.experiments.autond_runtime_preflight \\
        --autond-root ~/Documents/AutoND-upstream
Exit status is 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

CHECKS: list = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append({"check": name, "passed": bool(ok), "detail": detail})
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  —  {detail}" if detail else ""))
    return ok


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--autond-root", type=Path, required=True,
                   help="the PINNED AutoND checkout (contains main.py and "
                        "ciphers/). NOT audit/targets/autond.")
    p.add_argument("--cipher", default="speck3264")
    p.add_argument("--delta-plain-hex", default="0x400000")
    p.add_argument("--num-rounds", type=int, default=8)
    p.add_argument("--samples", type=int, default=20000,
                   help="size of the small real generation. Deliberately "
                        "small: this is a preflight, not an audit.")
    p.add_argument("--out", type=Path, default=None,
                   help="optional JSON report path")
    args = p.parse_args(argv)

    root = args.autond_root.expanduser().resolve()
    print("AutoND runtime preflight")
    print(f"  checkout: {root}")
    print()

    # -- 1. Runtime versions ------------------------------------------
    print("1. Runtime")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    for mod in ("pandas", "tensorflow", "keras"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:                               # noqa: BLE001
            versions[mod] = f"NOT AVAILABLE ({type(exc).__name__})"
    for k, v in versions.items():
        print(f"     {k:<12} {v}")
    major = int(np.__version__.split(".")[0])
    record("numpy major version is 1 (NEP 50 in numpy 2.x breaks the pinned "
           "cipher)", major == 1, f"numpy {np.__version__}")
    print()

    # -- 2. Pinned files present and digested -------------------------
    print("2. Pinned source")
    main_py = root / "main.py"
    cipher_py = root / "ciphers" / f"{args.cipher}.py"
    if not record("main.py exists", main_py.is_file(), str(main_py)):
        return _finish(args, versions, {})
    if not record(f"ciphers/{args.cipher}.py exists", cipher_py.is_file(),
                  str(cipher_py)):
        return _finish(args, versions, {})
    digests = {"main.py": sha256_file(main_py),
               f"ciphers/{args.cipher}.py": sha256_file(cipher_py)}
    for rel, d in digests.items():
        print(f"     {rel:<24} {d}")

    # Compare against the reviewed pin, when this cipher is covered by it.
    try:
        from audit.dataset.experiments.d0_autond_canonical import (
            AUTOND_SOURCE_PIN)
        pinned = AUTOND_SOURCE_PIN["source_digests"]
        covered = {k: v for k, v in pinned.items() if k in digests}
        if covered:
            bad = {k: {"pinned": v, "observed": digests[k]}
                   for k, v in covered.items() if digests[k] != v}
            record("source digests match the reviewed pin", not bad,
                   json.dumps(bad) if bad else
                   f"{len(covered)} file(s) match")
        else:
            record("reviewed pin covers this cipher", False,
                   f"{args.cipher} is not in the pin; only "
                   f"{sorted(pinned)} are")
    except Exception as exc:                                   # noqa: BLE001
        record("reviewed pin could be loaded", False,
               f"{type(exc).__name__}: {exc}")
    print()

    # -- 3. Cipher known-answer test, run explicitly ------------------
    print("3. Cipher known-answer test")
    inserted = False
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        inserted = True
    try:
        try:
            import importlib
            cipher = importlib.import_module(f"ciphers.{args.cipher}")
            record("cipher module imports (its import-time "
                   "check_testvectors() ran)", True)
        except Exception as exc:                               # noqa: BLE001
            record("cipher module imports", False,
                   f"{type(exc).__name__}: {exc}")
            return _finish(args, versions, digests)

        if hasattr(cipher, "check_testvectors"):
            try:
                cipher.check_testvectors()
                record("check_testvectors() called explicitly", True,
                       "Speck32/64 vector 0x6574694c / key 0x1918…0100 "
                       "-> 0xa868 0x42f2 after 22 rounds")
            except Exception as exc:                           # noqa: BLE001
                record("check_testvectors() called explicitly", False,
                       f"{type(exc).__name__}: {exc}")
        else:
            record("cipher exposes check_testvectors", False,
                   "no known-answer test in the source; one would have to be "
                   "established before trusting this implementation")

        # Independent re-derivation of the published vector, so the check does
        # not rest solely on the source's own assertion.
        try:
            pt = np.uint16([0x6574, 0x694c]).reshape(-1, 1)
            key = np.uint16([0x1918, 0x1110, 0x0908, 0x0100]).reshape(-1, 1)
            ct = cipher.convert_from_binary(
                cipher.encrypt(cipher.convert_to_binary(pt),
                               cipher.convert_to_binary(key), 22))
            got = [int(v) for v in ct[0]]
            record("published Speck32/64 test vector re-derived "
                   "independently", got == [0xa868, 0x42f2],
                   f"got {[hex(v) for v in got]}, expected ['0xa868', "
                   f"'0x42f2']")
        except Exception as exc:                               # noqa: BLE001
            record("published test vector re-derived", False,
                   f"{type(exc).__name__}: {exc}")
        print()

        # -- 4. Small REAL generation ---------------------------------
        print("4. Small real generation through the pinned generator")
        import importlib
        M = importlib.import_module("main")
        # This check previously passed `True` unconditionally, so it reported
        # success whatever happened. The result is now derived from an
        # observed condition: the modules the import graph requires must
        # actually be present in sys.modules afterwards.
        required_after_import = ("optimizer", "train_nets", "pandas",
                                 "tensorflow")
        loaded = [m for m in ("pandas", "tensorflow", "keras", "optimizer",
                              "train_nets") if m in sys.modules]
        absent = [m for m in required_after_import if m not in sys.modules]
        record("importing the generator pulled in its mandatory "
               "dependencies", not absent,
               f"loaded: {loaded}" if not absent
               else f"expected but not loaded: {absent}; loaded: {loaded}")
        record("the generator exposes make_train_data",
               hasattr(M, "make_train_data"))
        record("the generator exposes integer_to_binary_array",
               hasattr(M, "integer_to_binary_array"))

        plain_bits = int(cipher.plain_bits)
        key_bits = int(cipher.key_bits)
        delta_int = int(args.delta_plain_hex, 16)
        delta = M.integer_to_binary_array(delta_int, plain_bits)

        # Difference encoding: MSB-first, shape (1, plain_bits).
        set_bits = np.flatnonzero(np.asarray(delta).ravel()).tolist()
        expected_index = plain_bits - 1 - (delta_int.bit_length() - 1)
        record("difference encoding is the source's own, MSB-first",
               list(np.asarray(delta).shape) == [1, plain_bits]
               and int(np.asarray(delta).sum()) == bin(delta_int).count("1"),
               f"shape {tuple(np.asarray(delta).shape)}, popcount "
               f"{int(np.asarray(delta).sum())}, set bit(s) {set_bits}")
        if bin(delta_int).count("1") == 1:
            record("single-bit difference sits at the MSB-first index",
                   set_bits == [expected_index],
                   f"{args.delta_plain_hex} -> index {set_bits}, expected "
                   f"[{expected_index}]")

        X, Y = M.make_train_data(cipher.encrypt, plain_bits, key_bits,
                                 int(args.samples), int(args.num_rounds),
                                 delta, 0)
        record("generation returned arrays", X is not None and Y is not None)
        record("feature shape is (n, 2*plain_bits)",
               X.shape == (int(args.samples), 2 * plain_bits),
               f"{X.shape}, expected {(int(args.samples), 2 * plain_bits)}")
        record("feature dtype is uint8", X.dtype == np.uint8, str(X.dtype))
        record("label shape is (n,)", Y.shape == (int(args.samples),),
               str(Y.shape))
        record("label dtype is uint8", Y.dtype == np.uint8, str(Y.dtype))
        record("features are binary",
               set(np.unique(X).tolist()) <= {0, 1},
               f"values {sorted(np.unique(X).tolist())}")
        record("labels are binary",
               set(np.unique(Y).tolist()) <= {0, 1},
               f"values {sorted(np.unique(Y).tolist())}")
        record("both classes are present",
               set(np.unique(Y).tolist()) == {0, 1})

        # Label construction: make_train_data draws Y uniformly, so the rate
        # should be near 0.5. Bound stated rather than eyeballed: a 5-sigma
        # binomial band at this n.
        n = int(args.samples)
        rate = float(Y.mean())
        band = 5.0 * (0.25 / n) ** 0.5
        record("class balance is within a 5-sigma binomial band of 0.5",
               abs(rate - 0.5) <= band,
               f"rate {rate:.4f}, band +/-{band:.4f}")

        # Row-wise independence / unseeded source: two identical calls differ.
        X2, _ = M.make_train_data(cipher.encrypt, plain_bits, key_bits,
                                  256, int(args.num_rounds), delta, 0)
        X3, _ = M.make_train_data(cipher.encrypt, plain_bits, key_bits,
                                  256, int(args.num_rounds), delta, 0)
        record("identical calls produce different data (os.urandom, no seed)",
               not np.array_equal(X2, X3),
               "exact_replay_available is therefore False")
        print()

        # -- 5. Pinned files unchanged after all of this ---------------
        print("5. Pinned source unchanged by the preflight")
        after = {"main.py": sha256_file(main_py),
                 f"ciphers/{args.cipher}.py": sha256_file(cipher_py)}
        record("pinned source files are byte-identical to before",
               after == digests,
               "no file under the audited checkout was modified")
    finally:
        if inserted:
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass

    return _finish(args, versions, digests)


def _finish(args, versions: dict, digests: dict) -> int:
    passed = sum(1 for c in CHECKS if c["passed"])
    total = len(CHECKS)
    print()
    print(f"  preflight: {passed}/{total} checks passed")
    failed = [c["check"] for c in CHECKS if not c["passed"]]
    if failed:
        print("  FAILED:")
        for f in failed:
            print(f"    - {f}")
        print("  The runtime is NOT established. Do not generate a canonical "
              "dataset in this environment.")
    report = {
        "preflight": "autond-runtime-preflight-1",
        "checkout": str(args.autond_root.expanduser().resolve()),
        "cipher": args.cipher,
        "num_rounds": args.num_rounds,
        "samples": args.samples,
        "runtime": versions,
        "source_digests": digests,
        "checks": CHECKS,
        "passed": passed,
        "total": total,
        "all_passed": not failed,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"  report: {args.out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
