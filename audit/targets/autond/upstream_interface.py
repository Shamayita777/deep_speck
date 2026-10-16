#!/usr/bin/env python3
"""
The pinned checkout's own command-line interface, read from its source.

WHY THIS EXISTS
    Revision 2's gated launcher built `eval_nets.py --model <path>`. The
    pinned checkout has no `--model` option: it takes `--model_path`, and
    it also needs `--model_type`, `--cipher`, `--scenario`,
    `--input_difference` and `--round_number`. A launcher that would have
    been refused by the target's own argument parser is not a gate on a
    real run, it is a gate on a run that could never start. The commands
    are now derived from the parser definitions in the checkout rather
    than written from memory.

HOW THE INTERFACE IS READ
    By parsing `eval_nets.py` with `ast` and collecting every
    `parser.add_argument(...)` call inside `parseTheCommandLine`. Nothing
    is imported: importing `eval_nets` pulls in TensorFlow and executes
    module-level code in the target, neither of which belongs in an audit
    that is supposed to read the target rather than run it.

    `EVAL_NETS_ARGSPEC` below is the extraction taken from the pinned
    revision, recorded with the source digest it came from. When the
    checkout is present, `extract_argspec()` recomputes the extraction
    and `verify_argspec()` compares the two, so the recorded copy cannot
    drift from the source without a test failing. When the checkout is
    absent, the recorded copy still lets the command builder and its
    tests run, clearly marked as recorded rather than verified.

TWO BEHAVIOURS OF THE TARGET THAT CONSTRAIN THE LAUNCHER
    1. `filename_results = args.model_path.replace('.h5', f'_eval{args.add_str}.npz')`
       The NPZ is written NEXT TO the model path it is handed. This is
       why the checkpoints are copied out of the checkout and the copies
       are evaluated: handing it a path inside the checkout would make
       the target write into the checkout.
    2. `ciphers_list = glob.glob('ciphers/*.py')` runs under
       `if __name__ == '__main__'`, with a RELATIVE glob. The working
       directory must therefore be the checkout root. The launcher sets
       cwd to the checkout and keeps every written path outside it.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Optional

INTERFACE_VERSION = "autond-upstream-interface-v1"

EVAL_NETS_RELPATH = "eval_nets.py"

#: sha256 of eval_nets.py at the pinned commit, computed on the pinned
#: checkout. `verify_argspec()` recomputes it.
EVAL_NETS_SHA256 = "838587127c08834f6f9ecd7b27fc578da3942141fdcf5d9c513d46c646d63048"

#: The argument definitions extracted from that exact file. Each entry is
#: (dest, {required, default, choices}). Recorded so the command builder
#: and its tests work without the checkout; verified against the checkout
#: whenever one is supplied.
EVAL_NETS_ARGSPEC: dict[str, dict[str, Any]] = {
    "model_path": {"required": False, "default": None, "choices": None},
    "model_type": {"required": False, "default": "dbitnet",
                   "choices": ["gohr-depth1", "dbitnet"]},
    "cipher": {"required": True, "default": "speck3264", "choices": None},
    "scenario": {"required": False, "default": "single-key",
                 "choices": ["single-key", "related-key"]},
    "dataset_path_X": {"required": False, "default": None, "choices": None},
    "dataset_path_Y": {"required": False, "default": None, "choices": None},
    "input_difference": {"required": False, "default": None, "choices": None},
    "round_number": {"required": False, "default": None, "choices": None},
    "add_str": {"required": False, "default": "", "choices": None},
}

#: Arguments the target reads but which this audit must never pass,
#: because they would replace the freshly generated test data the plan
#: specifies with a pre-existing dataset.
FORBIDDEN_ARGS = ("dataset_path_X", "dataset_path_Y")

#: How the target names the NPZ it writes.
def npz_name_for(model_path: Path, add_str: str = "") -> Path:
    """Reproduce eval_nets.py's own output-name arithmetic exactly."""
    p = str(model_path)
    if not p.endswith(".h5"):
        raise ValueError(
            f"the target derives its NPZ name by replacing '.h5'; {p!r} does "
            "not end in .h5, so the name it would write is not predictable")
    return Path(p.replace(".h5", f"_eval{add_str}.npz"))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:                                               # noqa: BLE001
        return None


def extract_argspec(upstream: Path) -> dict[str, dict[str, Any]]:
    """
    Parse the checkout's eval_nets.py and return its argument definitions.

    Reads the file; imports nothing. Raises FileNotFoundError if the
    checkout does not hold the file, because guessing is what this module
    exists to prevent.
    """
    src = Path(upstream) / EVAL_NETS_RELPATH
    if not src.is_file():
        raise FileNotFoundError(f"no {EVAL_NETS_RELPATH} at {src}")
    tree = ast.parse(src.read_text())
    spec: dict[str, dict[str, Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
            continue
        if not node.args:
            continue
        flag = _literal(node.args[0])
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        entry: dict[str, Any] = {"required": False, "default": None,
                                 "choices": None}
        for kw in node.keywords:
            if kw.arg in ("required", "default", "choices"):
                entry[kw.arg] = _literal(kw.value)
        spec[flag[2:]] = entry
    return spec


def verify_argspec(upstream: Optional[Path]) -> dict:
    """
    Confirm the recorded interface still matches the checkout.

    Returns {"ok", "problems", "source_sha256", "observed", "verified"}.
    With no checkout, ok is False and `verified` is False: the recorded
    spec is usable but unverified, and says so.
    """
    out: dict = {"ok": False, "verified": False, "problems": [],
                 "source_sha256": None, "observed": None,
                 "recorded_sha256": EVAL_NETS_SHA256}
    if upstream is None:
        out["problems"].append(
            "no upstream checkout supplied; the recorded interface was not "
            "verified against the target's source")
        return out
    src = Path(upstream) / EVAL_NETS_RELPATH
    if not src.is_file():
        out["problems"].append(f"no {EVAL_NETS_RELPATH} at {src}")
        return out

    digest = sha256_file(src)
    out["source_sha256"] = digest
    if digest != EVAL_NETS_SHA256:
        out["problems"].append(
            f"{EVAL_NETS_RELPATH} hashes to {digest}, not the recorded "
            f"{EVAL_NETS_SHA256}. The interface recorded here was read from a "
            "different revision of the file, so the commands built from it may "
            "not be the ones this checkout accepts.")

    observed = extract_argspec(upstream)
    out["observed"] = observed
    missing = sorted(set(EVAL_NETS_ARGSPEC) - set(observed))
    extra = sorted(set(observed) - set(EVAL_NETS_ARGSPEC))
    if missing:
        out["problems"].append(f"recorded arguments absent from the source: {missing}")
    if extra:
        out["problems"].append(f"source defines arguments not recorded here: {extra}")
    for name in sorted(set(EVAL_NETS_ARGSPEC) & set(observed)):
        rec, obs = EVAL_NETS_ARGSPEC[name], observed[name]
        for field in ("required", "default", "choices"):
            if rec[field] != obs[field]:
                out["problems"].append(
                    f"--{name}.{field}: recorded {rec[field]!r} != source "
                    f"{obs[field]!r}")

    out["verified"] = not out["problems"]
    out["ok"] = out["verified"]
    return out


def build_eval_command(*, python: str, model_path: Path, cipher: str,
                       model_type: str, scenario: str, input_difference: str,
                       round_number: int, add_str: str = "",
                       argspec: Optional[dict] = None) -> list[str]:
    """
    Build one `eval_nets.py` invocation the pinned parser accepts.

    Validated against the argument spec before it is returned, so a flag
    the target does not define, a choice it would reject, or a missing
    required argument raises here rather than failing at launch.
    """
    spec = argspec or EVAL_NETS_ARGSPEC
    args = {
        "model_path": str(model_path),
        "model_type": model_type,
        "cipher": cipher,
        "scenario": scenario,
        "input_difference": input_difference,
        "round_number": str(round_number),
    }
    if add_str:
        args["add_str"] = add_str

    for name in args:
        if name not in spec:
            raise ValueError(
                f"--{name} is not an argument of the pinned eval_nets.py; its "
                f"arguments are {sorted(spec)}")
        if name in FORBIDDEN_ARGS:
            raise ValueError(
                f"--{name} would substitute a pre-existing dataset for the "
                "freshly generated test data the plan specifies")
        choices = spec[name].get("choices")
        if choices and args[name] not in choices:
            raise ValueError(
                f"--{name}={args[name]!r} is not one of the target's choices "
                f"{choices}")
    for name, entry in spec.items():
        if entry.get("required") and name not in args:
            raise ValueError(
                f"--{name} is required by the pinned eval_nets.py and was not "
                "supplied")

    cmd = [python, EVAL_NETS_RELPATH]
    for name in ("model_path", "model_type", "cipher", "scenario",
                 "input_difference", "round_number", "add_str"):
        if name in args:
            cmd += [f"--{name}", args[name]]
    return cmd


def main(argv=None) -> int:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(
        description="Report the pinned eval_nets.py interface. Imports "
                    "nothing from the target and runs no evaluation.")
    ap.add_argument("--autond-root", type=Path, default=None)
    args = ap.parse_args(argv)

    print(f"{INTERFACE_VERSION}")
    res = verify_argspec(args.autond_root)
    print(f"  recorded eval_nets.py sha256 : {EVAL_NETS_SHA256}")
    print(f"  observed                     : {res['source_sha256'] or '(no checkout)'}")
    print(f"  interface verified           : {res['verified']}")
    for p in res["problems"]:
        print(f"    - {p}")
    print("\n  recorded arguments:")
    for name, e in sorted(EVAL_NETS_ARGSPEC.items()):
        req = "required" if e["required"] else "optional"
        ch = f" choices={e['choices']}" if e["choices"] else ""
        print(f"    --{name:18} {req:8} default={e['default']!r}{ch}")
    return 0 if res["verified"] else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
